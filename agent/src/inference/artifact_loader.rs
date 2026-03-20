use crate::errors::{AgentError, Result};
use crate::model::registry::ShardRegistry;
use crate::model::shard::ShardAssignment;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

use super::forward_pass::ModelWeights;

#[async_trait]
pub trait ShardLoader: Send + Sync {
    async fn load_shard(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        registry: &ShardRegistry,
    ) -> Result<ModelWeights>;

    async fn is_cached(&self, model_id: &str) -> bool;

    fn estimate_memory(&self, assignment: &ShardAssignment) -> u64;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShardArtifactManifest {
    model_id: String,
    worker_position: u32,
    total_workers: u32,
    expected_sha256: String,
}

pub struct ArtifactShardLoader {
    model_store_dir: PathBuf,
}

impl ArtifactShardLoader {
    pub fn new(model_store_dir: PathBuf) -> Self {
        Self { model_store_dir }
    }

    pub fn with_defaults() -> Self {
        let base = std::env::var_os("MESHNET_MODEL_STORE")
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|home| home.join(".meshnet").join("models")))
            .unwrap_or_else(|| PathBuf::from(".meshnet/models"));
        Self::new(base)
    }

    fn manifest_path(&self, assignment: &ShardAssignment) -> PathBuf {
        self.model_store_dir
            .join(&assignment.model_id)
            .join(format!(
                "shard-{}-of-{}.manifest.json",
                assignment.worker_position, assignment.total_workers
            ))
    }

    fn weights_path(&self, assignment: &ShardAssignment) -> PathBuf {
        self.model_store_dir
            .join(&assignment.model_id)
            .join(format!(
                "shard-{}-of-{}.cbor",
                assignment.worker_position, assignment.total_workers
            ))
    }

    fn load_manifest(&self, assignment: &ShardAssignment) -> Result<ShardArtifactManifest> {
        let manifest_path = self.manifest_path(assignment);
        let manifest = fs::read_to_string(&manifest_path).map_err(|e| {
            AgentError::Config(format!(
                "Failed to read shard manifest {}: {}",
                manifest_path.display(),
                e
            ))
        })?;

        serde_json::from_str(&manifest).map_err(|e| {
            AgentError::Config(format!(
                "Failed to parse shard manifest {}: {}",
                manifest_path.display(),
                e
            ))
        })
    }

    fn read_and_verify_weights(
        &self,
        assignment: &ShardAssignment,
        manifest: &ShardArtifactManifest,
    ) -> Result<Vec<u8>> {
        let weights_path = self.weights_path(assignment);
        let bytes = fs::read(&weights_path).map_err(|e| {
            AgentError::Config(format!(
                "Failed to read shard artifact {}: {}",
                weights_path.display(),
                e
            ))
        })?;

        let digest = hex::encode(Sha256::digest(&bytes));
        if digest != manifest.expected_sha256 {
            return Err(AgentError::Config(format!(
                "Artifact hash mismatch for {}: expected {}, got {}",
                weights_path.display(),
                manifest.expected_sha256,
                digest
            )));
        }

        Ok(bytes)
    }

    fn validate_manifest(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        manifest: &ShardArtifactManifest,
    ) -> Result<()> {
        if manifest.model_id != model_id
            || manifest.worker_position != assignment.worker_position
            || manifest.total_workers != assignment.total_workers
        {
            return Err(AgentError::Config(format!(
                "Shard manifest does not match requested assignment for model {}",
                model_id
            )));
        }

        Ok(())
    }
}

#[async_trait]
impl ShardLoader for ArtifactShardLoader {
    async fn load_shard(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        registry: &ShardRegistry,
    ) -> Result<ModelWeights> {
        registry
            .update_status(model_id, crate::model::registry::ShardStatus::Downloading, None)
            .await?;

        let manifest = self.load_manifest(assignment)?;
        self.validate_manifest(model_id, assignment, &manifest)?;

        let weights_path = self.weights_path(assignment);
        let bytes = self.read_and_verify_weights(assignment, &manifest)?;
        registry
            .mark_downloaded(model_id, weights_path.clone(), manifest.expected_sha256.clone())
            .await?;

        let weights: ModelWeights = ciborium::from_reader(bytes.as_slice()).map_err(|e| {
            AgentError::Config(format!(
                "Failed to deserialize shard artifact {}: {}",
                weights_path.display(),
                e
            ))
        })?;

        registry
            .mark_loaded(model_id, weights.memory_usage() as u64)
            .await?;

        info!(
            model_id = %model_id,
            worker_position = assignment.worker_position,
            total_workers = assignment.total_workers,
            artifact = %weights_path.display(),
            "Loaded verified shard artifact"
        );

        Ok(weights)
    }

    async fn is_cached(&self, model_id: &str) -> bool {
        self.model_store_dir.join(model_id).exists()
    }

    fn estimate_memory(&self, assignment: &ShardAssignment) -> u64 {
        let artifact_path = self.weights_path(assignment);
        fs::metadata(artifact_path).map(|meta| meta.len()).unwrap_or(0)
    }
}

pub fn artifact_exists(base_dir: &Path, assignment: &ShardAssignment) -> bool {
    let path = base_dir
        .join(&assignment.model_id)
        .join(format!(
            "shard-{}-of-{}.cbor",
            assignment.worker_position, assignment.total_workers
        ));
    debug!(artifact = %path.display(), exists = path.exists(), "Checked shard artifact");
    path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::forward_pass::ModelConfig;
    use crate::inference::mock_validation;
    use tempfile::TempDir;

    fn write_artifact(
        root: &Path,
        assignment: &ShardAssignment,
        weights: &ModelWeights,
    ) -> Result<()> {
        let model_dir = root.join(&assignment.model_id);
        fs::create_dir_all(&model_dir)?;

        let artifact_path = model_dir.join(format!(
            "shard-{}-of-{}.cbor",
            assignment.worker_position, assignment.total_workers
        ));
        let manifest_path = model_dir.join(format!(
            "shard-{}-of-{}.manifest.json",
            assignment.worker_position, assignment.total_workers
        ));

        let mut bytes = Vec::new();
        ciborium::into_writer(weights, &mut bytes)
            .map_err(|e| AgentError::Config(format!("Failed to encode test weights: {}", e)))?;
        fs::write(&artifact_path, &bytes)?;

        let manifest = ShardArtifactManifest {
            model_id: assignment.model_id.clone(),
            worker_position: assignment.worker_position,
            total_workers: assignment.total_workers,
            expected_sha256: hex::encode(Sha256::digest(&bytes)),
        };
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap())?;
        Ok(())
    }

    #[tokio::test]
    async fn loads_verified_artifact() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ShardRegistry::new(temp_dir.path().join("registry")).unwrap();
        let assignment = ShardAssignment::new("llama-70b".into(), 0, 2);
        registry.assign_shard(assignment.clone()).await.unwrap();

        let config = ModelConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 128,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = mock_validation::generate_mock_weights(&config, 32, 42);
        write_artifact(temp_dir.path(), &assignment, &weights).unwrap();

        let loader = ArtifactShardLoader::new(temp_dir.path().to_path_buf());
        let loaded = loader
            .load_shard("llama-70b", &assignment, &registry)
            .await
            .unwrap();

        assert_eq!(loaded.model_id, weights.model_id);
        assert_eq!(loaded.layers.len(), weights.layers.len());
    }
}
