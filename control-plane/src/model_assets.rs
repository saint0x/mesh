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
    #[serde(default = "default_tokenizer_config_file")]
    pub tokenizer_config_file: String,
}

fn default_tokenizer_file() -> String {
    "tokenizer.json".to_string()
}

fn default_tokenizer_config_file() -> String {
    "tokenizer_config.json".to_string()
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
}

#[derive(Clone)]
struct CachedModelAssets {
    manifest: ModelManifest,
    tokenizer: Arc<Tokenizer>,
    tokenizer_config: Arc<TokenizerConfig>,
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
    let bytes = fs::read(&path).map_err(|e| {
        ApiError::BadRequest(format!(
            "Failed to read model manifest {}: {}",
            path.display(),
            e
        ))
    })?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes).map_err(|e| {
        ApiError::BadRequest(format!(
            "Failed to parse model manifest {}: {}",
            path.display(),
            e
        ))
    })?;
    validate_manifest(model_id, &path, manifest)
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, ApiError> {
    Tokenizer::from_file(path).map_err(|e| {
        ApiError::BadRequest(format!(
            "Failed to load tokenizer {}: {}",
            path.display(),
            e
        ))
    })
}

fn load_tokenizer_config(path: &Path) -> Result<TokenizerConfig, ApiError> {
    let bytes = fs::read(path).map_err(|e| {
        ApiError::BadRequest(format!(
            "Failed to read tokenizer config {}: {}",
            path.display(),
            e
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        ApiError::BadRequest(format!(
            "Failed to parse tokenizer config {}: {}",
            path.display(),
            e
        ))
    })
}

fn load_model_assets_uncached(model_id: &str) -> Result<CachedModelAssets, ApiError> {
    let manifest = load_model_manifest_uncached(model_id)?;
    let path = tokenizer_path(model_id, &manifest);
    let tokenizer = load_tokenizer(&path)?;
    let tokenizer_config = load_tokenizer_config(&tokenizer_config_path(model_id, &manifest))?;
    Ok(CachedModelAssets {
        manifest,
        tokenizer: Arc::new(tokenizer),
        tokenizer_config: Arc::new(tokenizer_config),
    })
}

fn get_model_assets(model_id: &str) -> Result<CachedModelAssets, ApiError> {
    if let Ok(cache) = model_asset_cache().read() {
        if let Some(assets) = cache.get(model_id) {
            return Ok(assets.clone());
        }
    }

    let assets = load_model_assets_uncached(model_id)?;
    let mut cache = model_asset_cache().write().map_err(|_| {
        ApiError::Internal("Failed to acquire model asset cache write lock".to_string())
    })?;
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

fn tokenizer_config_path(model_id: &str, manifest: &ModelManifest) -> PathBuf {
    model_dir(model_id).join(&manifest.tokenizer_config_file)
}

fn render_generation_prompt(prompt: &str, tokenizer_config: &TokenizerConfig) -> String {
    match (
        &tokenizer_config.chat_template,
        tokenizer_config.eos_token.as_deref(),
    ) {
        (Some(template), Some(eos))
            if template.contains("<|user|>") && template.contains("<|assistant|>") =>
        {
            format!("<|user|>\n{prompt}{eos}\n<|assistant|>\n")
        }
        _ => prompt.to_string(),
    }
}

pub fn tokenize_prompt(model_id: &str, prompt: &str) -> Result<Vec<u32>, ApiError> {
    let assets = get_model_assets(model_id)?;
    let rendered_prompt = render_generation_prompt(prompt, &assets.tokenizer_config);
    let encoding = assets
        .tokenizer
        .encode(rendered_prompt, true)
        .map_err(|e| {
            ApiError::BadRequest(format!(
                "Failed to tokenize prompt for model {}: {}",
                model_id, e
            ))
        })?;
    let ids = encoding.get_ids().iter().copied().collect::<Vec<u32>>();
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
            tokenizer_config_file: "tokenizer_config.json".to_string(),
        };
        atomic_write(
            &model_dir.join("model.json"),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        );

        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("<|user|>".to_string(), 1);
        vocab.insert("<|assistant|>".to_string(), 2);
        vocab.insert("</s>".to_string(), 3);
        vocab.insert("hello".to_string(), 4);
        vocab.insert("from".to_string(), 5);
        vocab.insert("vast".to_string(), 6);
        vocab.insert("live".to_string(), 7);
        vocab.insert("mesh".to_string(), 8);
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

        atomic_write(
            &model_dir.join("tokenizer_config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}",
                "eos_token": "</s>"
            }))
            .unwrap()
            .as_bytes(),
        );

        model_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_prompt_applies_chat_template() {
        testsupport::ensure_test_model("chat-template-test", 64);
        clear_model_asset_cache();

        let ids = tokenize_prompt("chat-template-test", "hello from vast").unwrap();
        let assets = get_model_assets("chat-template-test").unwrap();
        let expected_prompt = render_generation_prompt("hello from vast", &assets.tokenizer_config);
        let expected_ids = assets
            .tokenizer
            .encode(expected_prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();
        let raw_ids = assets
            .tokenizer
            .encode("hello from vast", true)
            .unwrap()
            .get_ids()
            .to_vec();

        assert_eq!(ids, expected_ids);
        assert_ne!(ids, raw_ids);
    }
}
