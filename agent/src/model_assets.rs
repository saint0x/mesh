use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
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

pub fn model_store_dir() -> PathBuf {
    std::env::var_os("MESHNET_MODEL_STORE")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".meshnet").join("models")))
        .unwrap_or_else(|| PathBuf::from(".meshnet/models"))
}

pub fn model_dir(model_id: &str) -> PathBuf {
    model_store_dir().join(model_id)
}

pub fn load_model_manifest(model_id: &str) -> Result<ModelManifest> {
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
    if manifest.model_id != model_id {
        return Err(AgentError::Config(format!(
            "Model manifest {} declares model_id {}, expected {}",
            path.display(),
            manifest.model_id,
            model_id
        )));
    }
    if manifest.tensor_parallelism_dim == 0 {
        return Err(AgentError::Config(format!(
            "Model manifest {} must declare tensor_parallelism_dim > 0",
            path.display()
        )));
    }
    Ok(manifest)
}

pub fn decode_tokens(model_id: &str, token_ids: &[u32]) -> Result<String> {
    let manifest = load_model_manifest(model_id)?;
    let path = model_dir(model_id).join(&manifest.tokenizer_file);
    let tokenizer = Tokenizer::from_file(&path).map_err(|e| {
        AgentError::Config(format!("Failed to load tokenizer {}: {}", path.display(), e))
    })?;
    tokenizer
        .decode(token_ids, true)
        .map_err(|e| AgentError::Config(format!("Failed to decode tokens for model {}: {}", model_id, e)))
}
