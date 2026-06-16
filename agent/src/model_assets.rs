use crate::errors::{AgentError, Result};
use crate::inference::forward_pass::SharedModelResidency;
use crate::inference::{ArtifactShardLoader, ShardLoader};
use crate::model::{ShardAssignment, ShardRegistry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub tensor_parallelism_dim: u32,
    pub total_model_bytes: u64,
    #[serde(default = "default_tokenizer_file")]
    pub tokenizer_file: String,
}

#[derive(Debug, Clone)]
pub struct ProductionArtifactProbeTarget {
    pub model_id: String,
    pub assignment: ShardAssignment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionArtifactProbeResult {
    pub model_id: String,
    pub worker_position: u32,
    pub total_workers: u32,
    pub column_start: u32,
    pub column_end: u32,
    pub resident_bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ShardArtifactManifest {
    model_id: String,
    worker_position: u32,
    total_workers: u32,
    column_start: u32,
    column_end: u32,
}

fn default_tokenizer_file() -> String {
    "tokenizer.json".to_string()
}

#[derive(Clone)]
struct CachedModelAssets {
    manifest: ModelManifest,
    tokenizer: Arc<Tokenizer>,
}

fn model_asset_cache() -> &'static RwLock<HashMap<String, CachedModelAssets>> {
    static CACHE: OnceLock<RwLock<HashMap<String, CachedModelAssets>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn model_store_dir() -> PathBuf {
    std::env::var_os("MESHNET_MODEL_STORE")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".meshnet").join("models")))
        .unwrap_or_else(|| PathBuf::from(".meshnet/models"))
}

pub fn model_dir(model_id: &str) -> PathBuf {
    model_store_dir().join(model_id)
}

fn first_shard_manifest_path(model_id: &str) -> Result<Option<PathBuf>> {
    let dir = model_dir(model_id);
    if !dir.is_dir() {
        return Ok(None);
    }

    let mut manifests = fs::read_dir(&dir)
        .map_err(|e| {
            AgentError::Config(format!(
                "Failed to read model artifact directory {}: {}",
                dir.display(),
                e
            ))
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_type()
                .map(|kind| kind.is_file())
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("shard-") && name.ends_with(".manifest.json"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    manifests.sort();
    Ok(manifests.into_iter().next())
}

fn load_shard_probe_target(
    model_id: &str,
    manifest_path: &Path,
) -> Result<ProductionArtifactProbeTarget> {
    let bytes = fs::read(manifest_path).map_err(|e| {
        AgentError::Config(format!(
            "Failed to read shard manifest {}: {}",
            manifest_path.display(),
            e
        ))
    })?;
    let manifest: ShardArtifactManifest = serde_json::from_slice(&bytes).map_err(|e| {
        AgentError::Config(format!(
            "Failed to parse shard manifest {}: {}",
            manifest_path.display(),
            e
        ))
    })?;
    if manifest.model_id != model_id {
        return Err(AgentError::Config(format!(
            "Shard manifest {} declares model_id {}, expected {}",
            manifest_path.display(),
            manifest.model_id,
            model_id
        )));
    }

    Ok(ProductionArtifactProbeTarget {
        model_id: model_id.to_string(),
        assignment: ShardAssignment::from_column_range(
            model_id.to_string(),
            manifest.worker_position,
            manifest.total_workers,
            manifest.column_start,
            manifest.column_end,
        ),
    })
}

pub fn discover_local_production_artifact_probe_target(
    preferred_model_id: Option<&str>,
) -> Result<Option<ProductionArtifactProbeTarget>> {
    if let Some(model_id) = preferred_model_id {
        return first_shard_manifest_path(model_id)?
            .map(|path| load_shard_probe_target(model_id, &path))
            .transpose();
    }

    let store = model_store_dir();
    if !store.is_dir() {
        return Ok(None);
    }

    let mut model_ids = fs::read_dir(&store)
        .map_err(|e| {
            AgentError::Config(format!(
                "Failed to read model store {}: {}",
                store.display(),
                e
            ))
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    model_ids.sort();

    for model_id in model_ids {
        if let Some(path) = first_shard_manifest_path(&model_id)? {
            return load_shard_probe_target(&model_id, &path).map(Some);
        }
    }

    Ok(None)
}

pub async fn probe_local_production_artifact_materialization(
    preferred_model_id: Option<&str>,
) -> Result<Option<ProductionArtifactProbeResult>> {
    let Some(target) = discover_local_production_artifact_probe_target(preferred_model_id)? else {
        return Ok(None);
    };

    let probe_root = std::env::temp_dir().join(format!(
        "meshnet-local-artifact-probe-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(&probe_root).map_err(|e| {
        AgentError::Config(format!(
            "Failed to create artifact probe temp directory {}: {}",
            probe_root.display(),
            e
        ))
    })?;
    let registry = ShardRegistry::new(probe_root.join("registry"))?;
    registry.assign_shard(target.assignment.clone()).await?;

    let loader = ArtifactShardLoader::new(model_store_dir());
    let weights = loader
        .load_shard(&target.model_id, &target.assignment, &registry)
        .await?;
    let residency = SharedModelResidency::from_host(weights)?;
    let resident_bytes = residency.resident_bytes();
    if resident_bytes == 0 {
        return Err(AgentError::Config(format!(
            "Real artifact probe for model {} materialized zero resident bytes",
            target.model_id
        )));
    }
    let _ = fs::remove_dir_all(&probe_root);

    Ok(Some(ProductionArtifactProbeResult {
        model_id: target.model_id,
        worker_position: target.assignment.worker_position,
        total_workers: target.assignment.total_workers,
        column_start: target.assignment.column_start,
        column_end: target.assignment.column_end,
        resident_bytes,
    }))
}

fn validate_manifest(
    requested_model_id: &str,
    path: &Path,
    manifest: ModelManifest,
) -> Result<ModelManifest> {
    if manifest.model_id != requested_model_id {
        return Err(AgentError::Config(format!(
            "Model manifest {} declares model_id {}, expected {}",
            path.display(),
            manifest.model_id,
            requested_model_id
        )));
    }
    if manifest.tensor_parallelism_dim == 0 {
        return Err(AgentError::Config(format!(
            "Model manifest {} must declare tensor_parallelism_dim > 0",
            path.display()
        )));
    }
    if manifest.total_model_bytes == 0 {
        return Err(AgentError::Config(format!(
            "Model manifest {} must declare total_model_bytes > 0",
            path.display()
        )));
    }
    Ok(manifest)
}

fn load_model_manifest_uncached(model_id: &str) -> Result<ModelManifest> {
    let path = model_dir(model_id).join("model.json");
    let bytes = fs::read(&path).map_err(|e| {
        AgentError::Config(format!(
            "Failed to read model manifest {}: {}",
            path.display(),
            e
        ))
    })?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes).map_err(|e| {
        AgentError::Config(format!(
            "Failed to parse model manifest {}: {}",
            path.display(),
            e
        ))
    })?;

    validate_manifest(model_id, &path, manifest)
}

fn load_model_assets_uncached(model_id: &str) -> Result<CachedModelAssets> {
    let manifest = load_model_manifest_uncached(model_id)?;
    let path = model_dir(model_id).join(&manifest.tokenizer_file);
    let tokenizer = Tokenizer::from_file(&path).map_err(|e| {
        AgentError::Config(format!(
            "Failed to load tokenizer {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(CachedModelAssets {
        manifest,
        tokenizer: Arc::new(tokenizer),
    })
}

fn get_model_assets(model_id: &str) -> Result<CachedModelAssets> {
    if let Ok(cache) = model_asset_cache().read() {
        if let Some(assets) = cache.get(model_id) {
            return Ok(assets.clone());
        }
    }

    let assets = load_model_assets_uncached(model_id)?;
    let mut cache = model_asset_cache().write().map_err(|_| {
        AgentError::Config("Failed to acquire model asset cache write lock".to_string())
    })?;
    let cached = cache
        .entry(model_id.to_string())
        .or_insert_with(|| assets.clone())
        .clone();
    Ok(cached)
}

pub fn load_model_manifest(model_id: &str) -> Result<ModelManifest> {
    Ok(get_model_assets(model_id)?.manifest)
}

pub fn decode_tokens(model_id: &str, token_ids: &[u32]) -> Result<String> {
    get_model_assets(model_id)?
        .tokenizer
        .decode(token_ids, true)
        .map_err(|e| {
            AgentError::Config(format!(
                "Failed to decode tokens for model {}: {}",
                model_id, e
            ))
        })
}
