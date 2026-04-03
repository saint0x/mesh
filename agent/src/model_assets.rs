use crate::errors::{AgentError, Result};
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
