use crate::api::error::ApiError;
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

pub fn model_manifest_path(model_id: &str) -> PathBuf {
    model_dir(model_id).join("model.json")
}

fn validate_manifest(
    requested_model_id: &str,
    path: &Path,
    manifest: ModelManifest,
) -> Result<ModelManifest, ApiError> {
    if manifest.model_id != requested_model_id {
        return Err(ApiError::BadRequest(format!(
            "Model manifest {} declares model_id {}, expected {}",
            path.display(),
            manifest.model_id,
            requested_model_id
        )));
    }
    if manifest.tensor_parallelism_dim == 0 {
        return Err(ApiError::BadRequest(format!(
            "Model manifest {} must declare tensor_parallelism_dim > 0",
            path.display()
        )));
    }
    if manifest.total_model_bytes == 0 {
        return Err(ApiError::BadRequest(format!(
            "Model manifest {} must declare total_model_bytes > 0",
            path.display()
        )));
    }

    Ok(manifest)
}

fn load_model_manifest_uncached(model_id: &str) -> Result<ModelManifest, ApiError> {
    let path = model_manifest_path(model_id);
    let bytes = fs::read(&path)
        .map_err(|e| ApiError::BadRequest(format!("Failed to read model manifest {}: {}", path.display(), e)))?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes)
        .map_err(|e| ApiError::BadRequest(format!("Failed to parse model manifest {}: {}", path.display(), e)))?;
    validate_manifest(model_id, &path, manifest)
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, ApiError> {
    Tokenizer::from_file(path)
        .map_err(|e| ApiError::BadRequest(format!("Failed to load tokenizer {}: {}", path.display(), e)))
}

fn load_model_assets_uncached(model_id: &str) -> Result<CachedModelAssets, ApiError> {
    let manifest = load_model_manifest_uncached(model_id)?;
    let path = tokenizer_path(model_id, &manifest);
    let tokenizer = load_tokenizer(&path)?;
    Ok(CachedModelAssets {
        manifest,
        tokenizer: Arc::new(tokenizer),
    })
}

fn get_model_assets(model_id: &str) -> Result<CachedModelAssets, ApiError> {
    if let Ok(cache) = model_asset_cache().read() {
        if let Some(assets) = cache.get(model_id) {
            return Ok(assets.clone());
        }
    }

    let assets = load_model_assets_uncached(model_id)?;
    let mut cache = model_asset_cache()
        .write()
        .map_err(|_| ApiError::Internal("Failed to acquire model asset cache write lock".to_string()))?;
    let cached = cache
        .entry(model_id.to_string())
        .or_insert_with(|| assets.clone())
        .clone();
    Ok(cached)
}

pub fn load_model_manifest(model_id: &str) -> Result<ModelManifest, ApiError> {
    Ok(get_model_assets(model_id)?.manifest)
}

#[cfg(test)]
pub fn clear_model_asset_cache() {
    if let Ok(mut cache) = model_asset_cache().write() {
        cache.clear();
    }
}

fn tokenizer_path(model_id: &str, manifest: &ModelManifest) -> PathBuf {
    model_dir(model_id).join(&manifest.tokenizer_file)
}

pub fn tokenize_prompt(model_id: &str, prompt: &str) -> Result<Vec<u32>, ApiError> {
    let assets = get_model_assets(model_id)?;
    let encoding = assets
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| ApiError::BadRequest(format!("Failed to tokenize prompt for model {}: {}", model_id, e)))?;
    let ids = encoding
        .get_ids()
        .iter()
        .copied()
        .collect::<Vec<u32>>();
    if ids.is_empty() {
        return Err(ApiError::BadRequest(format!(
            "Prompt tokenization produced zero tokens for model {}",
            model_id
        )));
    }
    Ok(ids)
}

#[cfg(test)]
pub mod testsupport {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    static TEST_MODEL_STORE: OnceLock<PathBuf> = OnceLock::new();
    static TEST_MODEL_WRITE_LOCK: Mutex<()> = Mutex::new(());

    fn atomic_write(path: &Path, bytes: &[u8]) {
        let temp_path = path.with_extension("tmp");
        fs::write(&temp_path, bytes).unwrap();
        fs::rename(&temp_path, path).unwrap();
    }

    pub fn ensure_test_model(model_id: &str, tensor_parallelism_dim: u32) -> PathBuf {
        let _guard = TEST_MODEL_WRITE_LOCK.lock().unwrap();
        let root = TEST_MODEL_STORE
            .get_or_init(|| {
                let root = std::env::temp_dir().join("meshnet-test-model-store");
                fs::create_dir_all(&root).unwrap();
                std::env::set_var("MESHNET_MODEL_STORE", &root);
                root
            })
            .clone();

        let model_dir = root.join(model_id);
        fs::create_dir_all(&model_dir).unwrap();

        let manifest = ModelManifest {
            model_id: model_id.to_string(),
            tensor_parallelism_dim,
            total_model_bytes: 1024 * 1024,
            tokenizer_file: "tokenizer.json".to_string(),
        };
        atomic_write(
            &model_dir.join("model.json"),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        );

        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("from".to_string(), 2);
        vocab.insert("vast".to_string(), 3);
        vocab.insert("live".to_string(), 4);
        vocab.insert("mesh".to_string(), 5);
        let wordlevel = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(wordlevel);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer_tmp_path = tokenizer_path.with_extension("tmp");
        tokenizer.save(&tokenizer_tmp_path, false).unwrap();
        fs::rename(tokenizer_tmp_path, tokenizer_path).unwrap();

        model_dir
    }
}
