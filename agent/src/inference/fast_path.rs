use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::errors::{AgentError, Result};
use crate::provider::ExecutionProviderKind;

use super::engine::{BackendOptimizationProfile, ExecutionPhase};

const KV_PAGE_TOKENS: usize = 16;
const DECODE_BUCKET_BATCHES: &[usize] = &[1, 2, 4, 8];
const DECODE_BUCKET_KV_TOKENS: &[usize] = &[2_048, 8_192, 16_384, 32_768, 65_536];
const PREFILL_BUCKET_TOKENS: &[usize] = &[128, 512, 2_048, 8_192, 16_384];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphCaptureStrategy {
    Unsupported,
    LayoutValidated,
    ReplayPreferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillBucketStrategy {
    SingleSequenceBuckets,
    MultiSequenceReserved,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FastPathBucketKey {
    pub phase: ExecutionPhase,
    pub provider: ExecutionProviderKind,
    pub optimization_profile: BackendOptimizationProfile,
    pub batch_size_ceiling: usize,
    pub token_ceiling: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataFieldLayout {
    pub name: String,
    pub element_size_bytes: usize,
    pub element_count: usize,
    pub stride_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketMetadataLayout {
    pub version: u32,
    pub total_bytes: usize,
    pub layout_hash: u64,
    pub fields: Vec<MetadataFieldLayout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRequirements {
    pub bytes: usize,
    pub alignment_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FastPathExecutionPlan {
    pub bucket: FastPathBucketKey,
    pub bucket_label: String,
    pub actual_batch_size: usize,
    pub actual_token_count: usize,
    pub max_sequence_len: usize,
    pub metadata: BucketMetadataLayout,
    pub workspace: WorkspaceRequirements,
    pub capture_strategy: GraphCaptureStrategy,
    pub prefill_strategy: Option<PrefillBucketStrategy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastPathBackendContext {
    pub provider: ExecutionProviderKind,
    pub optimization_profile: BackendOptimizationProfile,
    pub model_id: Option<String>,
    pub logical_kv_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceReservation {
    pub reused_existing_arena: bool,
    pub reserved_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastPathInvariantError {
    UnsupportedBucket {
        phase: ExecutionPhase,
        batch_size: usize,
        token_count: usize,
        profile: BackendOptimizationProfile,
    },
    MetadataDrift {
        bucket_label: String,
        expected_layout_hash: u64,
        observed_layout_hash: u64,
    },
    WorkspaceGrowthProhibited {
        bucket_label: String,
        reserved_bytes: usize,
        requested_bytes: usize,
    },
    CaptureUnsafe {
        bucket_label: String,
        detail: String,
    },
}

impl fmt::Display for FastPathInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBucket {
                phase,
                batch_size,
                token_count,
                profile,
            } => write!(
                f,
                "no stable {:?} bucket supports batch_size={} token_count={} for {:?}",
                phase, batch_size, token_count, profile
            ),
            Self::MetadataDrift {
                bucket_label,
                expected_layout_hash,
                observed_layout_hash,
            } => write!(
                f,
                "fast-path metadata drift for {}: expected layout hash {}, observed {}",
                bucket_label, expected_layout_hash, observed_layout_hash
            ),
            Self::WorkspaceGrowthProhibited {
                bucket_label,
                reserved_bytes,
                requested_bytes,
            } => write!(
                f,
                "workspace arena for {} cannot grow after registration: reserved {} bytes, requested {}",
                bucket_label, reserved_bytes, requested_bytes
            ),
            Self::CaptureUnsafe {
                bucket_label,
                detail,
            } => write!(f, "fast-path capture invariant failed for {}: {}", bucket_label, detail),
        }
    }
}

impl From<FastPathInvariantError> for AgentError {
    fn from(value: FastPathInvariantError) -> Self {
        AgentError::Execution(value.to_string())
    }
}

#[derive(Default)]
struct FastPathRuntimeState {
    layout_hashes: HashMap<FastPathBucketKey, u64>,
    reserved_workspaces: HashMap<FastPathBucketKey, usize>,
}

pub struct FastPathRuntime;

impl FastPathRuntime {
    fn state() -> &'static Mutex<FastPathRuntimeState> {
        static STATE: OnceLock<Mutex<FastPathRuntimeState>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(FastPathRuntimeState::default()))
    }

    pub fn prepare(plan: &FastPathExecutionPlan) -> Result<WorkspaceReservation> {
        let state = Self::state();
        let mut guard = state.lock().map_err(|_| {
            AgentError::Execution("fast-path runtime state mutex is poisoned".to_string())
        })?;

        if let Some(existing) = guard.layout_hashes.get(&plan.bucket).copied() {
            if existing != plan.metadata.layout_hash {
                return Err(FastPathInvariantError::MetadataDrift {
                    bucket_label: plan.bucket_label.clone(),
                    expected_layout_hash: existing,
                    observed_layout_hash: plan.metadata.layout_hash,
                }
                .into());
            }
        } else {
            guard
                .layout_hashes
                .insert(plan.bucket.clone(), plan.metadata.layout_hash);
        }

        match guard.reserved_workspaces.get(&plan.bucket).copied() {
            Some(reserved_bytes) if plan.workspace.bytes > reserved_bytes => {
                Err(FastPathInvariantError::WorkspaceGrowthProhibited {
                    bucket_label: plan.bucket_label.clone(),
                    reserved_bytes,
                    requested_bytes: plan.workspace.bytes,
                }
                .into())
            }
            Some(reserved_bytes) => Ok(WorkspaceReservation {
                reused_existing_arena: true,
                reserved_bytes,
            }),
            None => {
                guard
                    .reserved_workspaces
                    .insert(plan.bucket.clone(), plan.workspace.bytes);
                Ok(WorkspaceReservation {
                    reused_existing_arena: false,
                    reserved_bytes: plan.workspace.bytes,
                })
            }
        }
    }
}

pub struct FastPathPlanner;

impl FastPathPlanner {
    pub fn supported_decode_buckets(
        provider: ExecutionProviderKind,
        profile: BackendOptimizationProfile,
    ) -> Vec<FastPathBucketKey> {
        DECODE_BUCKET_BATCHES
            .iter()
            .flat_map(|batch_size_ceiling| {
                DECODE_BUCKET_KV_TOKENS
                    .iter()
                    .map(move |token_ceiling| FastPathBucketKey {
                        phase: ExecutionPhase::Decode,
                        provider,
                        optimization_profile: profile,
                        batch_size_ceiling: *batch_size_ceiling,
                        token_ceiling: *token_ceiling,
                    })
            })
            .collect()
    }

    pub fn supported_prefill_buckets(
        provider: ExecutionProviderKind,
        profile: BackendOptimizationProfile,
    ) -> Vec<FastPathBucketKey> {
        PREFILL_BUCKET_TOKENS
            .iter()
            .map(|token_ceiling| FastPathBucketKey {
                phase: ExecutionPhase::Prefill,
                provider,
                optimization_profile: profile,
                batch_size_ceiling: 1,
                token_ceiling: *token_ceiling,
            })
            .collect()
    }

    pub fn plan_decode(
        context: &FastPathBackendContext,
        batch_size: usize,
        total_kv_tokens: usize,
        max_sequence_len: usize,
    ) -> Result<FastPathExecutionPlan> {
        let bucket = Self::supported_decode_buckets(context.provider, context.optimization_profile)
            .into_iter()
            .find(|bucket| {
                batch_size <= bucket.batch_size_ceiling && total_kv_tokens <= bucket.token_ceiling
            })
            .ok_or_else(|| FastPathInvariantError::UnsupportedBucket {
                phase: ExecutionPhase::Decode,
                batch_size,
                token_count: total_kv_tokens,
                profile: context.optimization_profile,
            })?;

        Ok(Self::build_plan(
            bucket,
            batch_size,
            total_kv_tokens,
            max_sequence_len,
            None,
        ))
    }

    pub fn plan_prefill(
        context: &FastPathBackendContext,
        prompt_tokens: usize,
    ) -> Result<FastPathExecutionPlan> {
        let bucket =
            Self::supported_prefill_buckets(context.provider, context.optimization_profile)
                .into_iter()
                .find(|bucket| prompt_tokens <= bucket.token_ceiling)
                .ok_or_else(|| FastPathInvariantError::UnsupportedBucket {
                    phase: ExecutionPhase::Prefill,
                    batch_size: 1,
                    token_count: prompt_tokens,
                    profile: context.optimization_profile,
                })?;

        Ok(Self::build_plan(
            bucket,
            1,
            prompt_tokens,
            prompt_tokens,
            Some(PrefillBucketStrategy::SingleSequenceBuckets),
        ))
    }

    pub fn validate_decode_contexts(
        plan: &FastPathExecutionPlan,
        contexts: &[FastPathBackendContext],
    ) -> Result<()> {
        if contexts.len() != plan.actual_batch_size {
            return Err(FastPathInvariantError::CaptureUnsafe {
                bucket_label: plan.bucket_label.clone(),
                detail: format!(
                    "planned batch size {} but executor received {} requests",
                    plan.actual_batch_size,
                    contexts.len()
                ),
            }
            .into());
        }

        let Some(primary) = contexts.first() else {
            return Ok(());
        };

        if primary.provider != plan.bucket.provider
            || primary.optimization_profile != plan.bucket.optimization_profile
        {
            return Err(FastPathInvariantError::CaptureUnsafe {
                bucket_label: plan.bucket_label.clone(),
                detail: "primary backend context does not match planned provider/profile"
                    .to_string(),
            }
            .into());
        }

        let primary_model_id = primary.model_id.as_deref();
        for context in contexts {
            if context.provider != plan.bucket.provider
                || context.optimization_profile != plan.bucket.optimization_profile
            {
                return Err(FastPathInvariantError::CaptureUnsafe {
                    bucket_label: plan.bucket_label.clone(),
                    detail:
                        "mixed provider/profile contexts cannot reuse a single fast-path bucket"
                            .to_string(),
                }
                .into());
            }

            if context.logical_kv_tokens > plan.bucket.token_ceiling {
                return Err(FastPathInvariantError::CaptureUnsafe {
                    bucket_label: plan.bucket_label.clone(),
                    detail: format!(
                        "session KV footprint {} exceeds bucket token ceiling {}",
                        context.logical_kv_tokens, plan.bucket.token_ceiling
                    ),
                }
                .into());
            }

            if let (Some(expected), Some(observed)) =
                (primary_model_id, context.model_id.as_deref())
            {
                if expected != observed {
                    return Err(FastPathInvariantError::CaptureUnsafe {
                        bucket_label: plan.bucket_label.clone(),
                        detail: "mixed model residencies cannot share one replay-valid bucket"
                            .to_string(),
                    }
                    .into());
                }
            }
        }

        let observed_layout_hash = Self::layout_hash(&plan.bucket, &plan.metadata.fields);
        if observed_layout_hash != plan.metadata.layout_hash {
            return Err(FastPathInvariantError::MetadataDrift {
                bucket_label: plan.bucket_label.clone(),
                expected_layout_hash: plan.metadata.layout_hash,
                observed_layout_hash,
            }
            .into());
        }

        Ok(())
    }

    fn build_plan(
        bucket: FastPathBucketKey,
        actual_batch_size: usize,
        actual_token_count: usize,
        max_sequence_len: usize,
        prefill_strategy: Option<PrefillBucketStrategy>,
    ) -> FastPathExecutionPlan {
        let fields = Self::metadata_fields(&bucket);
        let layout_hash = Self::layout_hash(&bucket, &fields);
        let total_bytes = fields.iter().map(|field| field.stride_bytes).sum();
        let capture_strategy = Self::capture_strategy(bucket.optimization_profile);
        let workspace = Self::workspace_requirements(&bucket, total_bytes);
        let bucket_label = match bucket.phase {
            ExecutionPhase::Decode => format!(
                "decode-b{}-kv{}-{}",
                bucket.batch_size_ceiling,
                bucket.token_ceiling,
                bucket.provider.as_str()
            ),
            ExecutionPhase::Prefill => format!(
                "prefill-t{}-{}",
                bucket.token_ceiling,
                bucket.provider.as_str()
            ),
        };

        FastPathExecutionPlan {
            bucket,
            bucket_label,
            actual_batch_size,
            actual_token_count,
            max_sequence_len,
            metadata: BucketMetadataLayout {
                version: 1,
                total_bytes,
                layout_hash,
                fields,
            },
            workspace,
            capture_strategy,
            prefill_strategy,
        }
    }

    fn capture_strategy(profile: BackendOptimizationProfile) -> GraphCaptureStrategy {
        match profile {
            BackendOptimizationProfile::CpuSerial => GraphCaptureStrategy::Unsupported,
            BackendOptimizationProfile::MetalVectorized => GraphCaptureStrategy::LayoutValidated,
            BackendOptimizationProfile::CudaFused => GraphCaptureStrategy::ReplayPreferred,
        }
    }

    fn metadata_fields(bucket: &FastPathBucketKey) -> Vec<MetadataFieldLayout> {
        let page_slots = bucket.token_ceiling.div_ceil(KV_PAGE_TOKENS);
        vec![
            MetadataFieldLayout {
                name: "slot_sequence_lengths".to_string(),
                element_size_bytes: std::mem::size_of::<u32>(),
                element_count: bucket.batch_size_ceiling,
                stride_bytes: bucket.batch_size_ceiling * std::mem::size_of::<u32>(),
            },
            MetadataFieldLayout {
                name: "slot_positions".to_string(),
                element_size_bytes: std::mem::size_of::<u32>(),
                element_count: bucket.batch_size_ceiling,
                stride_bytes: bucket.batch_size_ceiling * std::mem::size_of::<u32>(),
            },
            MetadataFieldLayout {
                name: "slot_block_tables".to_string(),
                element_size_bytes: std::mem::size_of::<u32>(),
                element_count: bucket.batch_size_ceiling * page_slots,
                stride_bytes: bucket.batch_size_ceiling * page_slots * std::mem::size_of::<u32>(),
            },
            MetadataFieldLayout {
                name: "slot_mapping".to_string(),
                element_size_bytes: std::mem::size_of::<u32>(),
                element_count: bucket.batch_size_ceiling,
                stride_bytes: bucket.batch_size_ceiling * std::mem::size_of::<u32>(),
            },
        ]
    }

    fn workspace_requirements(
        bucket: &FastPathBucketKey,
        metadata_bytes: usize,
    ) -> WorkspaceRequirements {
        let per_token_bytes = match bucket.optimization_profile {
            BackendOptimizationProfile::CpuSerial => 16,
            BackendOptimizationProfile::MetalVectorized => 64,
            BackendOptimizationProfile::CudaFused => 96,
        };
        let per_batch_bytes = match bucket.optimization_profile {
            BackendOptimizationProfile::CpuSerial => 4 * 1_024,
            BackendOptimizationProfile::MetalVectorized => 16 * 1_024,
            BackendOptimizationProfile::CudaFused => 32 * 1_024,
        };

        WorkspaceRequirements {
            bytes: metadata_bytes
                .saturating_add(bucket.token_ceiling.saturating_mul(per_token_bytes))
                .saturating_add(bucket.batch_size_ceiling.saturating_mul(per_batch_bytes)),
            alignment_bytes: 256,
        }
    }

    fn layout_hash(bucket: &FastPathBucketKey, fields: &[MetadataFieldLayout]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bucket.hash(&mut hasher);
        for field in fields {
            field.name.hash(&mut hasher);
            field.element_size_bytes.hash(&mut hasher);
            field.element_count.hash(&mut hasher);
            field.stride_bytes.hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_context() -> FastPathBackendContext {
        FastPathBackendContext {
            provider: ExecutionProviderKind::Cuda,
            optimization_profile: BackendOptimizationProfile::CudaFused,
            model_id: Some("llama-70b".to_string()),
            logical_kv_tokens: 1_024,
        }
    }

    #[test]
    fn decode_plan_selects_smallest_supported_bucket() {
        let plan = FastPathPlanner::plan_decode(&cuda_context(), 3, 5_000, 2_048)
            .expect("decode plan should resolve");
        assert_eq!(plan.bucket.phase, ExecutionPhase::Decode);
        assert_eq!(plan.bucket.batch_size_ceiling, 4);
        assert_eq!(plan.bucket.token_ceiling, 8_192);
        assert_eq!(plan.capture_strategy, GraphCaptureStrategy::ReplayPreferred);
    }

    #[test]
    fn prefill_plan_uses_explicit_single_sequence_strategy() {
        let plan =
            FastPathPlanner::plan_prefill(&cuda_context(), 400).expect("prefill plan should work");
        assert_eq!(plan.bucket.phase, ExecutionPhase::Prefill);
        assert_eq!(plan.bucket.batch_size_ceiling, 1);
        assert_eq!(
            plan.prefill_strategy,
            Some(PrefillBucketStrategy::SingleSequenceBuckets)
        );
    }

    #[test]
    fn runtime_rejects_workspace_growth_after_bucket_registration() {
        let mut plan =
            FastPathPlanner::plan_decode(&cuda_context(), 1, 1_024, 1_024).expect("plan");
        FastPathRuntime::prepare(&plan).expect("first reservation");
        plan.workspace.bytes = plan.workspace.bytes.saturating_add(1);
        let err = FastPathRuntime::prepare(&plan).unwrap_err();
        assert!(err.to_string().contains("cannot grow after registration"));
    }

    #[test]
    fn validate_decode_contexts_rejects_provider_drift() {
        let plan = FastPathPlanner::plan_decode(&cuda_context(), 1, 1_024, 1_024).expect("plan");
        let contexts = vec![FastPathBackendContext {
            provider: ExecutionProviderKind::Cpu,
            optimization_profile: BackendOptimizationProfile::CpuSerial,
            model_id: Some("llama-70b".to_string()),
            logical_kv_tokens: 1_024,
        }];
        let err = FastPathPlanner::validate_decode_contexts(&plan, &contexts).unwrap_err();
        assert!(err
            .to_string()
            .contains("does not match planned provider/profile"));
    }
}
