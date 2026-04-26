use crate::api::error::ApiError;
use crate::api::types::{ExecutionGroupMember, ExecutionPhase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ServingLayoutKind {
    TensorParallel,
    FullReplica,
    Pipeline,
    TensorPipelineHybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ProviderConstraints {
    #[serde(default)]
    pub allowed_providers: Vec<String>,
    #[serde(default)]
    pub requires_homogeneous: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineStageLayerRange {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServingLayoutRule {
    #[serde(default)]
    pub variant_name: Option<String>,
    pub layout: ServingLayoutKind,
    #[serde(default)]
    pub min_members: Option<u32>,
    #[serde(default)]
    pub max_members: Option<u32>,
    #[serde(default)]
    pub tensor_parallel_degree: Option<u32>,
    #[serde(default)]
    pub pipeline_parallel_degree: Option<u32>,
    #[serde(default)]
    pub pipeline_stages: Vec<PipelineStageLayerRange>,
    #[serde(default)]
    pub provider_constraints: ProviderConstraints,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ModelExecutionLayout {
    #[serde(default)]
    pub prefill: Vec<ServingLayoutRule>,
    #[serde(default)]
    pub decode: Vec<ServingLayoutRule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedServingLayout {
    pub variant_name: Option<String>,
    pub layout: ServingLayoutKind,
    pub member_count: usize,
    pub tensor_parallel_degree: u32,
    pub pipeline_parallel_degree: u32,
    pub pipeline_stages: Vec<PipelineStageLayerRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub tensor_parallelism_dim: u32,
    pub total_model_bytes: u64,
    #[serde(default)]
    pub transformer_layer_count: Option<u32>,
    #[serde(default = "default_tokenizer_file")]
    pub tokenizer_file: String,
    #[serde(default = "default_tokenizer_config_file")]
    pub tokenizer_config_file: String,
    #[serde(default)]
    pub execution_layout: ModelExecutionLayout,
}

fn default_tokenizer_file() -> String {
    "tokenizer.json".to_string()
}

fn default_tokenizer_config_file() -> String {
    "tokenizer_config.json".to_string()
}

fn default_prefill_layout_rules() -> Vec<ServingLayoutRule> {
    vec![ServingLayoutRule {
        variant_name: Some("default_tensor_parallel".to_string()),
        layout: ServingLayoutKind::TensorParallel,
        min_members: None,
        max_members: None,
        tensor_parallel_degree: None,
        pipeline_parallel_degree: None,
        pipeline_stages: Vec::new(),
        provider_constraints: ProviderConstraints::default(),
    }]
}

fn default_decode_layout_rules() -> Vec<ServingLayoutRule> {
    vec![
        ServingLayoutRule {
            variant_name: Some("default_full_replica".to_string()),
            layout: ServingLayoutKind::FullReplica,
            min_members: None,
            max_members: None,
            tensor_parallel_degree: None,
            pipeline_parallel_degree: None,
            pipeline_stages: Vec::new(),
            provider_constraints: ProviderConstraints::default(),
        },
        ServingLayoutRule {
            variant_name: Some("default_tensor_parallel".to_string()),
            layout: ServingLayoutKind::TensorParallel,
            min_members: None,
            max_members: None,
            tensor_parallel_degree: None,
            pipeline_parallel_degree: None,
            pipeline_stages: Vec::new(),
            provider_constraints: ProviderConstraints::default(),
        },
    ]
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
    validate_layout_rules(
        path,
        manifest.transformer_layer_count,
        manifest.tensor_parallelism_dim,
        &manifest.execution_layout,
    )?;

    Ok(manifest)
}

fn validate_layout_rules(
    path: &Path,
    transformer_layer_count: Option<u32>,
    tensor_parallelism_dim: u32,
    layout: &ModelExecutionLayout,
) -> Result<(), ApiError> {
    for (phase_name, rules) in [("prefill", &layout.prefill), ("decode", &layout.decode)] {
        for (index, rule) in rules.iter().enumerate() {
            if let Some(min_members) = rule.min_members {
                if min_members == 0 {
                    return Err(ApiError::BadRequest(format!(
                        "Model manifest {} declares {} layout rule {} with min_members=0",
                        path.display(),
                        phase_name,
                        index
                    )));
                }
            }
            if let (Some(min_members), Some(max_members)) = (rule.min_members, rule.max_members) {
                if max_members < min_members {
                    return Err(ApiError::BadRequest(format!(
                        "Model manifest {} declares {} layout rule {} with max_members < min_members",
                        path.display(),
                        phase_name,
                        index
                    )));
                }
            }
            if let Some(degree) = rule.tensor_parallel_degree {
                if degree == 0 || degree > tensor_parallelism_dim {
                    return Err(ApiError::BadRequest(format!(
                        "Model manifest {} declares {} layout rule {} with invalid tensor_parallel_degree {}",
                        path.display(),
                        phase_name,
                        index,
                        degree
                    )));
                }
            }
            if matches!(
                rule.layout,
                ServingLayoutKind::Pipeline | ServingLayoutKind::TensorPipelineHybrid
            ) {
                let Some(pipeline_degree) = rule.pipeline_parallel_degree else {
                    return Err(ApiError::BadRequest(format!(
                        "Model manifest {} declares {} layout rule {} without pipeline_parallel_degree",
                        path.display(),
                        phase_name,
                        index
                    )));
                };
                if pipeline_degree == 0 {
                    return Err(ApiError::BadRequest(format!(
                        "Model manifest {} declares {} layout rule {} with pipeline_parallel_degree=0",
                        path.display(),
                        phase_name,
                        index
                    )));
                }
                validate_pipeline_stage_metadata(
                    path,
                    phase_name,
                    index,
                    transformer_layer_count,
                    pipeline_degree,
                    &rule.pipeline_stages,
                )?;
            }
            if matches!(rule.layout, ServingLayoutKind::TensorPipelineHybrid)
                && rule.tensor_parallel_degree.is_none()
            {
                return Err(ApiError::BadRequest(format!(
                    "Model manifest {} declares {} layout rule {} without tensor_parallel_degree",
                    path.display(),
                    phase_name,
                    index
                )));
            }
            if matches!(rule.layout, ServingLayoutKind::Pipeline)
                && rule.tensor_parallel_degree.is_some()
            {
                return Err(ApiError::BadRequest(format!(
                    "Model manifest {} declares {} layout rule {} with unexpected tensor_parallel_degree",
                    path.display(),
                    phase_name,
                    index
                )));
            }
        }
    }

    Ok(())
}

fn validate_pipeline_stage_metadata(
    path: &Path,
    phase_name: &str,
    index: usize,
    transformer_layer_count: Option<u32>,
    pipeline_degree: u32,
    stages: &[PipelineStageLayerRange],
) -> Result<(), ApiError> {
    let total_layers = transformer_layer_count.ok_or_else(|| {
        ApiError::BadRequest(format!(
            "Model manifest {} declares {} pipeline layout rule {} without transformer_layer_count",
            path.display(),
            phase_name,
            index
        ))
    })?;
    if total_layers == 0 {
        return Err(ApiError::BadRequest(format!(
            "Model manifest {} declares transformer_layer_count=0 for {} pipeline layout rule {}",
            path.display(),
            phase_name,
            index
        )));
    }
    if stages.len() != pipeline_degree as usize {
        return Err(ApiError::BadRequest(format!(
            "Model manifest {} declares {} pipeline layout rule {} with {} stage ranges, expected {}",
            path.display(),
            phase_name,
            index,
            stages.len(),
            pipeline_degree
        )));
    }
    let mut sorted = stages.to_vec();
    sorted.sort_by_key(|stage| stage.stage_index);
    let mut cursor = 0;
    for (expected_stage, stage) in sorted.iter().enumerate() {
        if stage.stage_index != expected_stage as u32 {
            return Err(ApiError::BadRequest(format!(
                "Model manifest {} declares {} pipeline layout rule {} with non-contiguous stage index {}",
                path.display(),
                phase_name,
                index,
                stage.stage_index
            )));
        }
        if stage.layer_start >= stage.layer_end {
            return Err(ApiError::BadRequest(format!(
                "Model manifest {} declares {} pipeline layout rule {} stage {} with empty layer range {}..{}",
                path.display(),
                phase_name,
                index,
                stage.stage_index,
                stage.layer_start,
                stage.layer_end
            )));
        }
        if stage.layer_start != cursor {
            return Err(ApiError::BadRequest(format!(
                "Model manifest {} declares {} pipeline layout rule {} stage {} starting at {}, expected {}",
                path.display(),
                phase_name,
                index,
                stage.stage_index,
                stage.layer_start,
                cursor
            )));
        }
        cursor = stage.layer_end;
    }
    if cursor != total_layers {
        return Err(ApiError::BadRequest(format!(
            "Model manifest {} declares {} pipeline layout rule {} covering layers 0..{}, expected 0..{}",
            path.display(),
            phase_name,
            index,
            cursor,
            total_layers
        )));
    }
    Ok(())
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

impl ModelManifest {
    pub fn layout_rules_for_phase(&self, phase: ExecutionPhase) -> Vec<ServingLayoutRule> {
        let configured = match phase {
            ExecutionPhase::Prefill => &self.execution_layout.prefill,
            ExecutionPhase::Decode => &self.execution_layout.decode,
        };
        if configured.is_empty() {
            match phase {
                ExecutionPhase::Prefill => default_prefill_layout_rules(),
                ExecutionPhase::Decode => default_decode_layout_rules(),
            }
        } else {
            configured.clone()
        }
    }
}

pub fn validate_execution_group_members(
    manifest: &ModelManifest,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
) -> Result<ResolvedServingLayout, String> {
    if members.is_empty() {
        return Err("execution group has no members".to_string());
    }

    let rules = manifest.layout_rules_for_phase(phase);
    let mut errors = Vec::with_capacity(rules.len());
    for rule in rules {
        match validate_against_rule(manifest, members, &rule) {
            Ok(layout) => return Ok(layout),
            Err(err) => errors.push(format!("{:?}: {}", rule.layout, err)),
        }
    }

    Err(if errors.is_empty() {
        "no execution layout rules are available".to_string()
    } else {
        format!("no legal {:?} layout matched: {}", phase, errors.join("; "))
    })
}

pub fn validate_execution_group(
    model_id: &str,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
) -> Result<ResolvedServingLayout, ApiError> {
    let manifest = load_model_manifest(model_id)?;
    validate_execution_group_members(&manifest, phase, members).map_err(ApiError::Conflict)
}

fn validate_against_rule(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    rule: &ServingLayoutRule,
) -> Result<ResolvedServingLayout, String> {
    validate_member_count(members, rule)?;
    validate_member_shards(manifest, members)?;
    validate_provider_constraints(members, &rule.provider_constraints)?;

    let layout = match rule.layout {
        ServingLayoutKind::TensorParallel => {
            validate_tensor_parallel_layout(manifest, members, rule.tensor_parallel_degree)?;
            ResolvedServingLayout {
                variant_name: rule.variant_name.clone(),
                layout: rule.layout,
                member_count: members.len(),
                tensor_parallel_degree: rule.tensor_parallel_degree.unwrap_or(members.len() as u32),
                pipeline_parallel_degree: 1,
                pipeline_stages: Vec::new(),
            }
        }
        ServingLayoutKind::FullReplica => {
            validate_full_replica_layout(manifest, members)?;
            ResolvedServingLayout {
                variant_name: rule.variant_name.clone(),
                layout: rule.layout,
                member_count: members.len(),
                tensor_parallel_degree: 1,
                pipeline_parallel_degree: 1,
                pipeline_stages: Vec::new(),
            }
        }
        ServingLayoutKind::Pipeline => validate_pipeline_layout(manifest, members, rule)?,
        ServingLayoutKind::TensorPipelineHybrid => {
            validate_tensor_pipeline_hybrid_layout(manifest, members, rule)?
        }
    };

    Ok(layout)
}

fn validate_member_count(
    members: &[ExecutionGroupMember],
    rule: &ServingLayoutRule,
) -> Result<(), String> {
    let member_count = members.len() as u32;
    if let Some(min_members) = rule.min_members {
        if member_count < min_members {
            return Err(format!(
                "requires at least {} members, found {}",
                min_members, member_count
            ));
        }
    }
    if let Some(max_members) = rule.max_members {
        if member_count > max_members {
            return Err(format!(
                "allows at most {} members, found {}",
                max_members, member_count
            ));
        }
    }
    Ok(())
}

fn validate_member_shards(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
) -> Result<(), String> {
    for member in members {
        if member.shard.column_start >= member.shard.column_end {
            return Err(format!(
                "member {} has an empty shard span {}..{}",
                member.device_id, member.shard.column_start, member.shard.column_end
            ));
        }
        if member.shard.column_end > manifest.tensor_parallelism_dim {
            return Err(format!(
                "member {} shard span {}..{} exceeds tensor parallel dimension {}",
                member.device_id,
                member.shard.column_start,
                member.shard.column_end,
                manifest.tensor_parallelism_dim
            ));
        }
        if member.shard.column_start >= manifest.tensor_parallelism_dim {
            return Err(format!(
                "member {} shard span {}..{} starts outside tensor parallel dimension {}",
                member.device_id,
                member.shard.column_start,
                member.shard.column_end,
                manifest.tensor_parallelism_dim
            ));
        }
    }
    Ok(())
}

fn validate_provider_constraints(
    members: &[ExecutionGroupMember],
    constraints: &ProviderConstraints,
) -> Result<(), String> {
    if !constraints.allowed_providers.is_empty() {
        for member in members {
            if !constraints
                .allowed_providers
                .iter()
                .any(|allowed| allowed == &member.execution_provider)
            {
                return Err(format!(
                    "member {} uses provider {}, allowed providers are {}",
                    member.device_id,
                    member.execution_provider,
                    constraints.allowed_providers.join(", ")
                ));
            }
        }
    }
    if constraints.requires_homogeneous {
        let first = members
            .first()
            .map(|member| member.execution_provider.as_str())
            .unwrap_or_default();
        if members
            .iter()
            .any(|member| member.execution_provider.as_str() != first)
        {
            return Err("providers must be homogeneous across the group".to_string());
        }
    }
    Ok(())
}

fn validate_tensor_parallel_layout(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    exact_degree: Option<u32>,
) -> Result<(), String> {
    if let Some(exact_degree) = exact_degree {
        if members.len() as u32 != exact_degree {
            return Err(format!(
                "requires tensor_parallel_degree={}, found {} members",
                exact_degree,
                members.len()
            ));
        }
    }

    let mut intervals = members
        .iter()
        .map(|member| {
            (
                member.device_id.as_str(),
                member.shard.column_start,
                member.shard.column_end,
            )
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|(_, start, end)| (*start, *end));

    let mut cursor = 0;
    for (device_id, start, end) in intervals {
        if start != cursor {
            if start < cursor {
                return Err(format!(
                    "member {} overlaps the existing tensor-parallel span at {}..{}",
                    device_id, start, end
                ));
            }
            return Err(format!(
                "tensor-parallel layout leaves a gap before {}..{}",
                start, end
            ));
        }
        cursor = end;
    }

    if cursor != manifest.tensor_parallelism_dim {
        return Err(format!(
            "tensor-parallel layout ends at {}, expected {}",
            cursor, manifest.tensor_parallelism_dim
        ));
    }

    Ok(())
}

fn validate_full_replica_layout(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
) -> Result<(), String> {
    for member in members {
        if member.shard.column_start != 0
            || member.shard.column_end != manifest.tensor_parallelism_dim
        {
            return Err(format!(
                "member {} is not a full replica because it serves {}..{} instead of 0..{}",
                member.device_id,
                member.shard.column_start,
                member.shard.column_end,
                manifest.tensor_parallelism_dim
            ));
        }
    }
    Ok(())
}

fn validate_pipeline_layout(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    rule: &ServingLayoutRule,
) -> Result<ResolvedServingLayout, String> {
    validate_full_replica_layout(manifest, members)?;
    let pipeline_degree = rule
        .pipeline_parallel_degree
        .ok_or_else(|| "pipeline layouts require pipeline_parallel_degree".to_string())?;
    if members.len() as u32 != pipeline_degree {
        return Err(format!(
            "requires pipeline_parallel_degree={}, found {} members",
            pipeline_degree,
            members.len()
        ));
    }
    Ok(ResolvedServingLayout {
        variant_name: rule.variant_name.clone(),
        layout: rule.layout,
        member_count: members.len(),
        tensor_parallel_degree: 1,
        pipeline_parallel_degree: pipeline_degree,
        pipeline_stages: rule.pipeline_stages.clone(),
    })
}

fn validate_tensor_pipeline_hybrid_layout(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    rule: &ServingLayoutRule,
) -> Result<ResolvedServingLayout, String> {
    let pipeline_degree = rule.pipeline_parallel_degree.ok_or_else(|| {
        "tensor/pipeline hybrid layouts require pipeline_parallel_degree".to_string()
    })?;
    let tensor_degree = rule.tensor_parallel_degree.ok_or_else(|| {
        "tensor/pipeline hybrid layouts require tensor_parallel_degree".to_string()
    })?;
    let expected_members = pipeline_degree
        .checked_mul(tensor_degree)
        .ok_or_else(|| "hybrid layout member count overflow".to_string())?;
    if members.len() as u32 != expected_members {
        return Err(format!(
            "requires tensor_parallel_degree={} and pipeline_parallel_degree={}, found {} members",
            tensor_degree,
            pipeline_degree,
            members.len()
        ));
    }

    let mut sorted = members.to_vec();
    sorted.sort_by_key(|member| member.ring_position);
    for stage_index in 0..pipeline_degree as usize {
        let start = stage_index * tensor_degree as usize;
        let end = start + tensor_degree as usize;
        validate_tensor_parallel_layout(manifest, &sorted[start..end], Some(tensor_degree))?;
    }

    Ok(ResolvedServingLayout {
        variant_name: rule.variant_name.clone(),
        layout: rule.layout,
        member_count: members.len(),
        tensor_parallel_degree: tensor_degree,
        pipeline_parallel_degree: pipeline_degree,
        pipeline_stages: rule.pipeline_stages.clone(),
    })
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
        ensure_test_model_with_manifest(ModelManifest {
            model_id: model_id.to_string(),
            tensor_parallelism_dim,
            total_model_bytes: 1024 * 1024,
            transformer_layer_count: None,
            tokenizer_file: "tokenizer.json".to_string(),
            tokenizer_config_file: "tokenizer_config.json".to_string(),
            execution_layout: ModelExecutionLayout::default(),
        })
    }

    pub fn ensure_test_model_with_manifest(manifest: ModelManifest) -> PathBuf {
        let _guard = TEST_MODEL_WRITE_LOCK.lock().unwrap();
        let root = TEST_MODEL_STORE
            .get_or_init(|| {
                let root = std::env::temp_dir().join("meshnet-test-model-store");
                fs::create_dir_all(&root).unwrap();
                std::env::set_var("MESHNET_MODEL_STORE", &root);
                root
            })
            .clone();

        let model_dir = root.join(&manifest.model_id);
        fs::create_dir_all(&model_dir).unwrap();
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
