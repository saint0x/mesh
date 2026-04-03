use crate::errors::{AgentError, Result};
use crate::inference::forward_pass::{LayerWeights, ModelConfig, ModelWeights};
use crate::inference::tensor_ops::{Tensor1D, Tensor2D};
use crate::model::registry::ShardRegistry;
use crate::model::shard::ShardAssignment;
use async_trait::async_trait;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

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
                "shard-{}-of-{}.safetensors",
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

    fn decode_weights(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        bytes: &[u8],
    ) -> Result<ModelWeights> {
        let (_, metadata) = SafeTensors::read_metadata(bytes)
            .map_err(|e| AgentError::Config(format!("Failed to read safetensors metadata: {}", e)))?;
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| AgentError::Config(format!("Failed to open safetensors shard: {}", e)))?;

        validate_metadata(model_id, assignment, metadata.metadata().as_ref())?;
        let config = load_model_config(metadata.metadata().as_ref())?;

        let embedding = load_tensor_2d(&tensors, "embedding")?;
        let final_norm = load_tensor_1d(&tensors, "final_norm")?;
        let lm_head = load_tensor_2d(&tensors, "lm_head")?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            layers.push(LayerWeights {
                layer_idx,
                w_q: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_q"))?,
                w_k: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_k"))?,
                w_v: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_v"))?,
                w_o: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_o"))?,
                w_up: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_up"))?,
                w_gate: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_gate"))?,
                w_down: load_tensor_2d(&tensors, &format!("layers.{layer_idx}.w_down"))?,
                attn_norm: load_tensor_1d(&tensors, &format!("layers.{layer_idx}.attn_norm"))?,
                mlp_norm: load_tensor_1d(&tensors, &format!("layers.{layer_idx}.mlp_norm"))?,
            });
        }

        validate_weight_shapes(&config, assignment, &embedding, &final_norm, &lm_head, &layers)?;

        Ok(ModelWeights {
            model_id: model_id.to_string(),
            embedding,
            layers,
            final_norm,
            lm_head,
            config,
        })
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
            .update_status(
                model_id,
                crate::model::registry::ShardStatus::Downloading,
                None,
            )
            .await?;

        let manifest = self.load_manifest(assignment)?;
        self.validate_manifest(model_id, assignment, &manifest)?;

        let weights_path = self.weights_path(assignment);
        let bytes = self.read_and_verify_weights(assignment, &manifest)?;
        registry
            .mark_downloaded(
                model_id,
                weights_path.clone(),
                manifest.expected_sha256.clone(),
            )
            .await?;

        let weights = self.decode_weights(model_id, assignment, &bytes)?;

        registry
            .mark_loaded(model_id, weights.memory_usage() as u64)
            .await?;

        info!(
            model_id = %model_id,
            worker_position = assignment.worker_position,
            total_workers = assignment.total_workers,
            artifact = %weights_path.display(),
            "Loaded verified safetensors shard"
        );

        Ok(weights)
    }

    async fn is_cached(&self, model_id: &str) -> bool {
        self.model_store_dir.join(model_id).exists()
    }

    fn estimate_memory(&self, assignment: &ShardAssignment) -> u64 {
        fs::metadata(self.weights_path(assignment))
            .map(|meta| meta.len())
            .unwrap_or(0)
    }
}

pub fn artifact_exists(base_dir: &Path, assignment: &ShardAssignment) -> bool {
    let path = base_dir.join(&assignment.model_id).join(format!(
        "shard-{}-of-{}.safetensors",
        assignment.worker_position, assignment.total_workers
    ));
    debug!(artifact = %path.display(), exists = path.exists(), "Checked shard artifact");
    path.exists()
}

fn metadata_required<'a>(metadata: &'a HashMap<String, String>, key: &str) -> Result<&'a str> {
    metadata
        .get(key)
        .map(|value| value.as_str())
        .ok_or_else(|| AgentError::Config(format!("Missing safetensors metadata key {}", key)))
}

fn metadata_usize(metadata: &HashMap<String, String>, key: &str) -> Result<usize> {
    metadata_required(metadata, key)?
        .parse()
        .map_err(|_| AgentError::Config(format!("Invalid usize metadata for {}", key)))
}

fn metadata_u32(metadata: &HashMap<String, String>, key: &str) -> Result<u32> {
    metadata_required(metadata, key)?
        .parse()
        .map_err(|_| AgentError::Config(format!("Invalid u32 metadata for {}", key)))
}

fn metadata_f32(metadata: &HashMap<String, String>, key: &str) -> Result<f32> {
    metadata_required(metadata, key)?
        .parse()
        .map_err(|_| AgentError::Config(format!("Invalid f32 metadata for {}", key)))
}

fn load_model_config(metadata: Option<&HashMap<String, String>>) -> Result<ModelConfig> {
    let metadata = metadata
        .ok_or_else(|| AgentError::Config("Safetensors shard missing metadata".to_string()))?;
    Ok(ModelConfig {
        hidden_dim: metadata_usize(metadata, "mesh.hidden_dim")?,
        num_heads: metadata_usize(metadata, "mesh.num_heads")?,
        num_kv_heads: metadata_usize(metadata, "mesh.num_kv_heads")?,
        num_layers: metadata_usize(metadata, "mesh.num_layers")?,
        vocab_size: metadata_usize(metadata, "mesh.vocab_size")?,
        intermediate_size: metadata_usize(metadata, "mesh.intermediate_size")?,
        rms_norm_eps: metadata_f32(metadata, "mesh.rms_norm_eps")?,
        rope_base: metadata_f32(metadata, "mesh.rope_base")?,
    })
}

fn validate_metadata(
    model_id: &str,
    assignment: &ShardAssignment,
    metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    let metadata = metadata
        .ok_or_else(|| AgentError::Config("Safetensors shard missing metadata".to_string()))?;
    let shard_model_id = metadata_required(metadata, "mesh.model_id")?;
    if shard_model_id != model_id {
        return Err(AgentError::Config(format!(
            "Safetensors metadata model_id mismatch: expected {}, got {}",
            model_id, shard_model_id
        )));
    }

    let worker_position = metadata_u32(metadata, "mesh.worker_position")?;
    let total_workers = metadata_u32(metadata, "mesh.total_workers")?;
    if worker_position != assignment.worker_position || total_workers != assignment.total_workers {
        return Err(AgentError::Config(format!(
            "Safetensors metadata shard assignment mismatch for model {}",
            model_id
        )));
    }

    Ok(())
}

fn partition_columns(total_columns: usize, worker_position: u32, total_workers: u32) -> usize {
    if total_workers == 0 {
        return total_columns;
    }

    let total_workers = total_workers as usize;
    let worker_position = worker_position as usize;
    let columns_per_worker = total_columns / total_workers;
    let remainder = total_columns % total_workers;

    if worker_position < remainder {
        columns_per_worker + 1
    } else {
        columns_per_worker
    }
}

fn load_tensor_2d(tensors: &SafeTensors<'_>, name: &str) -> Result<Tensor2D> {
    let tensor = tensors
        .tensor(name)
        .map_err(|e| AgentError::Config(format!("Missing tensor {}: {}", name, e)))?;
    if tensor.dtype() != Dtype::F32 {
        return Err(AgentError::Config(format!(
            "Tensor {} must be F32, got {:?}",
            name,
            tensor.dtype()
        )));
    }
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(AgentError::Config(format!(
            "Tensor {} must be rank-2, got shape {:?}",
            name, shape
        )));
    }

    Tensor2D::new(
        bytes_to_f32_vec(tensor.data(), name)?,
        shape[0],
        shape[1],
    )
}

fn load_tensor_1d(tensors: &SafeTensors<'_>, name: &str) -> Result<Tensor1D> {
    let tensor = tensors
        .tensor(name)
        .map_err(|e| AgentError::Config(format!("Missing tensor {}: {}", name, e)))?;
    if tensor.dtype() != Dtype::F32 {
        return Err(AgentError::Config(format!(
            "Tensor {} must be F32, got {:?}",
            name,
            tensor.dtype()
        )));
    }
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(AgentError::Config(format!(
            "Tensor {} must be rank-1, got shape {:?}",
            name, shape
        )));
    }

    Ok(Tensor1D::new(bytes_to_f32_vec(tensor.data(), name)?))
}

fn bytes_to_f32_vec(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(AgentError::Config(format!(
            "Tensor {} byte length {} is not divisible by 4",
            name,
            bytes.len()
        )));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn validate_weight_shapes(
    config: &ModelConfig,
    assignment: &ShardAssignment,
    embedding: &Tensor2D,
    final_norm: &Tensor1D,
    lm_head: &Tensor2D,
    layers: &[LayerWeights],
) -> Result<()> {
    if config.hidden_dim % config.num_heads != 0 {
        return Err(AgentError::Config(format!(
            "Unsupported attention geometry: hidden_dim {} num_heads {}",
            config.hidden_dim, config.num_heads
        )));
    }
    if config.num_kv_heads == 0 || config.num_heads % config.num_kv_heads != 0 {
        return Err(AgentError::Config(format!(
            "Unsupported grouped-query attention geometry: num_heads {} num_kv_heads {}",
            config.num_heads, config.num_kv_heads
        )));
    }

    let expected_cols = assignment.num_columns() as usize;
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_total_cols = config.num_kv_heads * head_dim;
    let expected_kv_cols =
        partition_columns(kv_total_cols, assignment.worker_position, assignment.total_workers);

    if embedding.rows != config.vocab_size || embedding.cols != config.hidden_dim {
        return Err(AgentError::Config(format!(
            "Embedding shape mismatch: expected {}x{}, got {}x{}",
            config.vocab_size, config.hidden_dim, embedding.rows, embedding.cols
        )));
    }
    if final_norm.len() != config.hidden_dim {
        return Err(AgentError::Config(format!(
            "Final norm shape mismatch: expected {}, got {}",
            config.hidden_dim,
            final_norm.len()
        )));
    }
    if lm_head.rows != config.hidden_dim || lm_head.cols != config.vocab_size {
        return Err(AgentError::Config(format!(
            "LM head shape mismatch: expected {}x{}, got {}x{}",
            config.hidden_dim, config.vocab_size, lm_head.rows, lm_head.cols
        )));
    }
    if layers.len() != config.num_layers {
        return Err(AgentError::Config(format!(
            "Layer count mismatch: expected {}, got {}",
            config.num_layers,
            layers.len()
        )));
    }

    for layer in layers {
        validate_layer_shape(&layer.w_q, config.hidden_dim, expected_cols, "w_q", layer.layer_idx)?;
        validate_layer_shape(
            &layer.w_k,
            config.hidden_dim,
            expected_kv_cols,
            "w_k",
            layer.layer_idx,
        )?;
        validate_layer_shape(
            &layer.w_v,
            config.hidden_dim,
            expected_kv_cols,
            "w_v",
            layer.layer_idx,
        )?;
        validate_layer_shape(&layer.w_o, expected_cols, config.hidden_dim, "w_o", layer.layer_idx)?;
        validate_layer_shape(&layer.w_up, config.hidden_dim, expected_cols, "w_up", layer.layer_idx)?;
        validate_layer_shape(&layer.w_gate, config.hidden_dim, expected_cols, "w_gate", layer.layer_idx)?;
        validate_layer_shape(&layer.w_down, expected_cols, config.hidden_dim, "w_down", layer.layer_idx)?;

        if layer.attn_norm.len() != config.hidden_dim || layer.mlp_norm.len() != config.hidden_dim
        {
            return Err(AgentError::Config(format!(
                "Layer {} norm shape mismatch",
                layer.layer_idx
            )));
        }
    }

    Ok(())
}

fn validate_layer_shape(
    tensor: &Tensor2D,
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
    layer_idx: usize,
) -> Result<()> {
    if tensor.rows != expected_rows || tensor.cols != expected_cols {
        return Err(AgentError::Config(format!(
            "Layer {} tensor {} shape mismatch: expected {}x{}, got {}x{}",
            layer_idx, name, expected_rows, expected_cols, tensor.rows, tensor.cols
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, Dtype, View};
    use std::borrow::Cow;
    use tempfile::TempDir;

    struct TestTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for &TestTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn tensor2(shape: [usize; 2], start: f32) -> TestTensor {
        let len = shape[0] * shape[1];
        let data = (0..len).map(|idx| start + idx as f32).collect::<Vec<_>>();
        TestTensor {
            dtype: Dtype::F32,
            shape: shape.to_vec(),
            data: f32_bytes(&data),
        }
    }

    fn tensor1(len: usize, start: f32) -> TestTensor {
        let data = (0..len).map(|idx| start + idx as f32).collect::<Vec<_>>();
        TestTensor {
            dtype: Dtype::F32,
            shape: vec![len],
            data: f32_bytes(&data),
        }
    }

    fn write_test_shard(root: &Path, assignment: &ShardAssignment) -> Result<()> {
        let hidden_dim = 8usize;
        let shard_cols = assignment.num_columns() as usize;
        let vocab_size = 16usize;
        let intermediate_size = 16usize;
        let num_layers = 2usize;

        let model_dir = root.join(&assignment.model_id);
        fs::create_dir_all(&model_dir)?;

        let artifact_path = model_dir.join(format!(
            "shard-{}-of-{}.safetensors",
            assignment.worker_position, assignment.total_workers
        ));
        let manifest_path = model_dir.join(format!(
            "shard-{}-of-{}.manifest.json",
            assignment.worker_position, assignment.total_workers
        ));

        let mut tensors = vec![
            ("embedding".to_string(), tensor2([vocab_size, hidden_dim], 0.0)),
            ("final_norm".to_string(), tensor1(hidden_dim, 1000.0)),
            ("lm_head".to_string(), tensor2([hidden_dim, vocab_size], 2000.0)),
        ];
        for layer_idx in 0..num_layers {
            let prefix = format!("layers.{layer_idx}");
            tensors.push((format!("{prefix}.w_q"), tensor2([hidden_dim, shard_cols], 10.0)));
            tensors.push((format!("{prefix}.w_k"), tensor2([hidden_dim, shard_cols], 20.0)));
            tensors.push((format!("{prefix}.w_v"), tensor2([hidden_dim, shard_cols], 30.0)));
            tensors.push((format!("{prefix}.w_o"), tensor2([shard_cols, hidden_dim], 40.0)));
            tensors.push((format!("{prefix}.w_up"), tensor2([hidden_dim, shard_cols], 50.0)));
            tensors.push((format!("{prefix}.w_gate"), tensor2([hidden_dim, shard_cols], 60.0)));
            tensors.push((format!("{prefix}.w_down"), tensor2([shard_cols, hidden_dim], 70.0)));
            tensors.push((format!("{prefix}.attn_norm"), tensor1(hidden_dim, 80.0)));
            tensors.push((format!("{prefix}.mlp_norm"), tensor1(hidden_dim, 90.0)));
        }

        let metadata = HashMap::from([
            ("mesh.model_id".to_string(), assignment.model_id.clone()),
            (
                "mesh.worker_position".to_string(),
                assignment.worker_position.to_string(),
            ),
            (
                "mesh.total_workers".to_string(),
                assignment.total_workers.to_string(),
            ),
            ("mesh.hidden_dim".to_string(), hidden_dim.to_string()),
            ("mesh.num_heads".to_string(), "2".to_string()),
            ("mesh.num_kv_heads".to_string(), "2".to_string()),
            ("mesh.num_layers".to_string(), num_layers.to_string()),
            ("mesh.vocab_size".to_string(), vocab_size.to_string()),
            (
                "mesh.intermediate_size".to_string(),
                intermediate_size.to_string(),
            ),
            ("mesh.rms_norm_eps".to_string(), "0.00001".to_string()),
            ("mesh.rope_base".to_string(), "10000".to_string()),
        ]);

        let bytes = serialize(tensors.iter().map(|(name, tensor)| (name.clone(), tensor)), Some(metadata))
            .map_err(|e| AgentError::Config(format!("Failed to serialize test shard: {}", e)))?;
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
    async fn loads_verified_safetensors_artifact() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ShardRegistry::new(temp_dir.path().join("registry")).unwrap();
        let assignment = ShardAssignment::new("tinyllama-1.1b".into(), 0, 2, 8);
        registry.assign_shard(assignment.clone()).await.unwrap();
        write_test_shard(temp_dir.path(), &assignment).unwrap();

        let loader = ArtifactShardLoader::new(temp_dir.path().to_path_buf());
        let loaded = loader
            .load_shard("tinyllama-1.1b", &assignment, &registry)
            .await
            .unwrap();

        assert_eq!(loaded.model_id, "tinyllama-1.1b");
        assert_eq!(loaded.config.num_layers, 2);
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.layers[0].w_q.rows, 8);
        assert_eq!(loaded.layers[0].w_q.cols, 4);
    }
}
