use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::errors::{AgentError, Result};
use crate::provider::ExecutionProviderKind;

use super::engine::{BackendOptimizationProfile, ExecutionPhase};
use super::kv_cache::{LiveKVBlockTable, LiveKVWindow, DEFAULT_LIVE_KV_PAGE_TOKENS};

const DECODE_BUCKET_BATCHES: &[usize] = &[1, 2, 4, 8];
const DECODE_BUCKET_KV_TOKENS: &[usize] = &[2_048, 8_192, 16_384, 32_768, 65_536];
const PREFILL_BUCKET_TOKENS: &[usize] = &[128, 512, 2_048, 8_192, 16_384];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeSlotState {
    pub position: u32,
    pub block_table: LiveKVBlockTable,
}

impl DecodeSlotState {
    pub fn new(position: u32, block_table: LiveKVBlockTable) -> Result<Self> {
        let window = LiveKVWindow::new(
            position,
            position.saturating_sub(block_table.cached_tokens() as u32),
            block_table.cached_tokens() as u32,
        )?;
        block_table.validate_window(&window)?;
        Ok(Self {
            position,
            block_table,
        })
    }

    pub fn sequence_len(&self) -> u32 {
        self.block_table.cached_tokens() as u32
    }
}

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

#[derive(Debug)]
struct DecodeWorkspaceBuffers {
    scratch_bytes: Vec<u8>,
    token_ids: Vec<u32>,
    positions: Vec<u32>,
    slot_sequence_lengths: Vec<u32>,
    slot_positions: Vec<u32>,
    slot_block_tables: Vec<u32>,
    slot_mapping: Vec<u32>,
}

#[derive(Debug)]
struct PrefillWorkspaceBuffers {
    scratch_bytes: Vec<u8>,
    positions: Vec<u32>,
}

impl PrefillWorkspaceBuffers {
    fn for_bucket(bucket: &FastPathBucketKey) -> Self {
        let metadata_bytes = FastPathPlanner::metadata_fields(bucket)
            .iter()
            .map(|field| field.stride_bytes)
            .sum();
        let reserved_bytes = FastPathPlanner::workspace_requirements(bucket, metadata_bytes).bytes;
        Self {
            scratch_bytes: vec![0; reserved_bytes],
            positions: vec![0; bucket.token_ceiling],
        }
    }

    fn reset(&mut self) {
        self.scratch_bytes.fill(0);
        self.positions.fill(0);
    }
}

impl DecodeWorkspaceBuffers {
    fn for_bucket(bucket: &FastPathBucketKey) -> Self {
        let page_slots = bucket.token_ceiling.div_ceil(DEFAULT_LIVE_KV_PAGE_TOKENS);
        let metadata_bytes = FastPathPlanner::metadata_fields(bucket)
            .iter()
            .map(|field| field.stride_bytes)
            .sum();
        let reserved_bytes = FastPathPlanner::workspace_requirements(bucket, metadata_bytes).bytes;
        Self {
            scratch_bytes: vec![0; reserved_bytes],
            token_ids: vec![0; bucket.batch_size_ceiling],
            positions: vec![0; bucket.batch_size_ceiling],
            slot_sequence_lengths: vec![0; bucket.batch_size_ceiling],
            slot_positions: vec![0; bucket.batch_size_ceiling],
            slot_block_tables: vec![u32::MAX; bucket.batch_size_ceiling * page_slots],
            slot_mapping: vec![u32::MAX; bucket.batch_size_ceiling],
        }
    }

    fn reset(&mut self) {
        self.scratch_bytes.fill(0);
        self.token_ids.fill(0);
        self.positions.fill(0);
        self.slot_sequence_lengths.fill(0);
        self.slot_positions.fill(0);
        self.slot_block_tables.fill(u32::MAX);
        self.slot_mapping.fill(u32::MAX);
    }
}

pub struct DecodeWorkspaceLease {
    bucket: FastPathBucketKey,
    buffers: Option<DecodeWorkspaceBuffers>,
}

pub struct PrefillWorkspaceLease {
    bucket: FastPathBucketKey,
    buffers: Option<PrefillWorkspaceBuffers>,
}

impl DecodeWorkspaceLease {
    fn stage_with_slot_reader(
        &mut self,
        tokens: &[u32],
        mut slot_reader: impl FnMut(usize) -> Result<DecodeSlotState>,
    ) -> Result<()> {
        if tokens.len() > self.bucket.batch_size_ceiling {
            return Err(FastPathInvariantError::CaptureUnsafe {
                bucket_label: Self::bucket_label(&self.bucket),
                detail: format!(
                    "runtime batch size {} exceeds bucket ceiling {}",
                    tokens.len(),
                    self.bucket.batch_size_ceiling
                ),
            }
            .into());
        }

        let page_slots = self
            .bucket
            .token_ceiling
            .div_ceil(DEFAULT_LIVE_KV_PAGE_TOKENS);
        let buffers = self
            .buffers
            .as_mut()
            .ok_or_else(|| AgentError::Execution("decode workspace lease is empty".to_string()))?;
        buffers.reset();

        buffers.token_ids[..tokens.len()].copy_from_slice(tokens);

        for slot_idx in 0..tokens.len() {
            let slot_state = slot_reader(slot_idx)?;
            let sequence_len = slot_state.sequence_len();
            if slot_state.block_table.page_tokens != DEFAULT_LIVE_KV_PAGE_TOKENS {
                return Err(FastPathInvariantError::CaptureUnsafe {
                    bucket_label: Self::bucket_label(&self.bucket),
                    detail: format!(
                        "slot {} exported live page size {} but fast path requires {}",
                        slot_idx, slot_state.block_table.page_tokens, DEFAULT_LIVE_KV_PAGE_TOKENS
                    ),
                }
                .into());
            }
            if sequence_len as usize > self.bucket.token_ceiling {
                return Err(FastPathInvariantError::CaptureUnsafe {
                    bucket_label: Self::bucket_label(&self.bucket),
                    detail: format!(
                        "slot {} sequence length {} exceeds bucket token ceiling {}",
                        slot_idx, sequence_len, self.bucket.token_ceiling
                    ),
                }
                .into());
            }
            if slot_state.block_table.block_table_len() > page_slots {
                return Err(FastPathInvariantError::CaptureUnsafe {
                    bucket_label: Self::bucket_label(&self.bucket),
                    detail: format!(
                        "slot {} block table length {} exceeds workspace capacity {}",
                        slot_idx,
                        slot_state.block_table.block_table_len(),
                        page_slots
                    ),
                }
                .into());
            }
            buffers.positions[slot_idx] = slot_state.position;
            buffers.slot_sequence_lengths[slot_idx] = sequence_len;
            buffers.slot_positions[slot_idx] = slot_state.position;
            buffers.slot_mapping[slot_idx] = slot_idx as u32;
            let table_start = slot_idx * page_slots;
            for (page_idx, page_id) in slot_state.block_table.page_ids().enumerate() {
                buffers.slot_block_tables[table_start + page_idx] = page_id;
            }
        }

        Ok(())
    }

    pub fn stage(&mut self, tokens: &[u32], slot_states: &[DecodeSlotState]) -> Result<()> {
        if tokens.len() != slot_states.len() {
            return Err(AgentError::Execution(format!(
                "decode workspace staging mismatch: tokens={} slot_states={}",
                tokens.len(),
                slot_states.len(),
            )));
        }
        self.stage_with_slot_reader(tokens, |slot_idx| Ok(slot_states[slot_idx].clone()))
    }

    pub fn stage_from_slot_reader(
        &mut self,
        tokens: &[u32],
        slot_reader: impl FnMut(usize) -> Result<DecodeSlotState>,
    ) -> Result<()> {
        self.stage_with_slot_reader(tokens, slot_reader)
    }

    pub fn token_ids(&self, len: usize) -> &[u32] {
        &self
            .buffers
            .as_ref()
            .expect("decode workspace lease missing buffers")
            .token_ids[..len]
    }

    pub fn positions(&self, len: usize) -> &[u32] {
        &self
            .buffers
            .as_ref()
            .expect("decode workspace lease missing buffers")
            .positions[..len]
    }

    #[cfg(test)]
    pub fn sequence_lengths(&self, len: usize) -> &[u32] {
        &self
            .buffers
            .as_ref()
            .expect("decode workspace lease missing buffers")
            .slot_sequence_lengths[..len]
    }

    #[cfg(test)]
    pub fn block_table(&self, slot_idx: usize) -> &[u32] {
        let buffers = self
            .buffers
            .as_ref()
            .expect("decode workspace lease missing buffers");
        let page_slots = self
            .bucket
            .token_ceiling
            .div_ceil(DEFAULT_LIVE_KV_PAGE_TOKENS);
        let table_start = slot_idx * page_slots;
        &buffers.slot_block_tables[table_start..table_start + page_slots]
    }

    fn bucket_label(bucket: &FastPathBucketKey) -> String {
        format!(
            "decode-b{}-kv{}-{}",
            bucket.batch_size_ceiling,
            bucket.token_ceiling,
            bucket.provider.as_str()
        )
    }
}

impl PrefillWorkspaceLease {
    pub fn stage_positions(
        &mut self,
        absolute_position_start: usize,
        token_count: usize,
    ) -> Result<&[u32]> {
        if token_count > self.bucket.token_ceiling {
            return Err(FastPathInvariantError::CaptureUnsafe {
                bucket_label: format!(
                    "prefill-t{}-{}",
                    self.bucket.token_ceiling,
                    self.bucket.provider.as_str()
                ),
                detail: format!(
                    "prefill token count {} exceeds bucket ceiling {}",
                    token_count, self.bucket.token_ceiling
                ),
            }
            .into());
        }

        let buffers = self
            .buffers
            .as_mut()
            .ok_or_else(|| AgentError::Execution("prefill workspace lease is empty".to_string()))?;
        buffers.reset();
        for (offset, position) in buffers.positions[..token_count].iter_mut().enumerate() {
            *position = absolute_position_start
                .saturating_add(offset)
                .try_into()
                .map_err(|_| {
                    AgentError::Execution(
                        "prefill absolute position exceeded u32 addressable range".to_string(),
                    )
                })?;
        }
        Ok(&buffers.positions[..token_count])
    }
}

impl Drop for DecodeWorkspaceLease {
    fn drop(&mut self) {
        let Some(mut buffers) = self.buffers.take() else {
            return;
        };
        buffers.reset();
        if let Ok(mut guard) = FastPathRuntime::state().lock() {
            guard
                .decode_workspaces
                .entry(self.bucket.clone())
                .or_default()
                .push(buffers);
        }
    }
}

impl Drop for PrefillWorkspaceLease {
    fn drop(&mut self) {
        let Some(mut buffers) = self.buffers.take() else {
            return;
        };
        buffers.reset();
        if let Ok(mut guard) = FastPathRuntime::state().lock() {
            guard
                .prefill_workspaces
                .entry(self.bucket.clone())
                .or_default()
                .push(buffers);
        }
    }
}

#[derive(Default)]
struct FastPathRuntimeState {
    layout_hashes: HashMap<FastPathBucketKey, u64>,
    reserved_workspaces: HashMap<FastPathBucketKey, usize>,
    decode_workspaces: HashMap<FastPathBucketKey, Vec<DecodeWorkspaceBuffers>>,
    prefill_workspaces: HashMap<FastPathBucketKey, Vec<PrefillWorkspaceBuffers>>,
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

    pub fn checkout_decode_workspace(
        plan: &FastPathExecutionPlan,
    ) -> Result<(WorkspaceReservation, DecodeWorkspaceLease)> {
        if plan.bucket.phase != ExecutionPhase::Decode {
            return Err(AgentError::Execution(format!(
                "decode workspace checkout requires decode plan, got {:?}",
                plan.bucket.phase
            )));
        }

        let reservation = Self::prepare(plan)?;
        let state = Self::state();
        let mut guard = state.lock().map_err(|_| {
            AgentError::Execution("fast-path runtime state mutex is poisoned".to_string())
        })?;
        let buffers = guard
            .decode_workspaces
            .entry(plan.bucket.clone())
            .or_default()
            .pop()
            .unwrap_or_else(|| DecodeWorkspaceBuffers::for_bucket(&plan.bucket));

        Ok((
            reservation,
            DecodeWorkspaceLease {
                bucket: plan.bucket.clone(),
                buffers: Some(buffers),
            },
        ))
    }

    pub fn checkout_prefill_workspace(
        plan: &FastPathExecutionPlan,
    ) -> Result<(WorkspaceReservation, PrefillWorkspaceLease)> {
        if plan.bucket.phase != ExecutionPhase::Prefill {
            return Err(AgentError::Execution(format!(
                "prefill workspace checkout requires prefill plan, got {:?}",
                plan.bucket.phase
            )));
        }

        let reservation = Self::prepare(plan)?;
        let state = Self::state();
        let mut guard = state.lock().map_err(|_| {
            AgentError::Execution("fast-path runtime state mutex is poisoned".to_string())
        })?;
        let buffers = guard
            .prefill_workspaces
            .entry(plan.bucket.clone())
            .or_default()
            .pop()
            .unwrap_or_else(|| PrefillWorkspaceBuffers::for_bucket(&plan.bucket));

        Ok((
            reservation,
            PrefillWorkspaceLease {
                bucket: plan.bucket.clone(),
                buffers: Some(buffers),
            },
        ))
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

    pub fn decode_token_ceiling_for_context(context: &FastPathBackendContext) -> Option<usize> {
        Self::supported_decode_buckets(context.provider, context.optimization_profile)
            .into_iter()
            .filter(|bucket| bucket.batch_size_ceiling >= 1)
            .map(|bucket| bucket.token_ceiling)
            .filter(|token_ceiling| context.logical_kv_tokens <= *token_ceiling)
            .min()
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
            BackendOptimizationProfile::CpuSerial => GraphCaptureStrategy::LayoutValidated,
            BackendOptimizationProfile::MetalVectorized => GraphCaptureStrategy::LayoutValidated,
            BackendOptimizationProfile::CudaFused => GraphCaptureStrategy::ReplayPreferred,
        }
    }

    fn metadata_fields(bucket: &FastPathBucketKey) -> Vec<MetadataFieldLayout> {
        let page_slots = bucket.token_ceiling.div_ceil(DEFAULT_LIVE_KV_PAGE_TOKENS);
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
    use crate::inference::kv_cache::LiveKVBlockSpan;

    fn cuda_context() -> FastPathBackendContext {
        FastPathBackendContext {
            provider: ExecutionProviderKind::Cuda,
            optimization_profile: BackendOptimizationProfile::CudaFused,
            model_id: Some("llama-70b".to_string()),
            logical_kv_tokens: 1_024,
        }
    }

    fn slot_state(position: u32, cached_tokens: usize) -> DecodeSlotState {
        DecodeSlotState::new(
            position,
            LiveKVBlockTable::sequential(DEFAULT_LIVE_KV_PAGE_TOKENS, cached_tokens),
        )
        .expect("slot state")
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

    #[test]
    fn decode_workspace_lease_stages_batch_metadata_and_reuses_bucket_pool() {
        let plan = FastPathPlanner::plan_decode(&cuda_context(), 3, 5_000, 2_048)
            .expect("decode plan should resolve");
        let (first_reservation, mut first_lease) =
            FastPathRuntime::checkout_decode_workspace(&plan).expect("first checkout");
        assert!(!first_reservation.reused_existing_arena);
        first_lease
            .stage(
                &[11, 12, 13],
                &[slot_state(31, 7), slot_state(32, 8), slot_state(33, 9)],
            )
            .expect("workspace stage should succeed");
        assert_eq!(first_lease.token_ids(3), &[11, 12, 13]);
        assert_eq!(first_lease.positions(3), &[31, 32, 33]);
        assert_eq!(first_lease.sequence_lengths(3), &[7, 8, 9]);
        drop(first_lease);

        let (second_reservation, mut second_lease) =
            FastPathRuntime::checkout_decode_workspace(&plan).expect("second checkout");
        assert!(second_reservation.reused_existing_arena);
        second_lease
            .stage(&[21, 22], &[slot_state(41, 3), slot_state(42, 4)])
            .expect("workspace restage should succeed");
        assert_eq!(second_lease.token_ids(2), &[21, 22]);
        assert_eq!(second_lease.positions(2), &[41, 42]);
        assert_eq!(second_lease.sequence_lengths(2), &[3, 4]);
    }

    #[test]
    fn decode_workspace_lease_stages_batch_metadata_from_slot_reader() {
        let plan = FastPathPlanner::plan_decode(&cuda_context(), 3, 5_000, 2_048)
            .expect("decode plan should resolve");
        let (_reservation, mut lease) =
            FastPathRuntime::checkout_decode_workspace(&plan).expect("checkout");
        lease
            .stage_from_slot_reader(&[11, 12, 13], |slot_idx| {
                Ok(slot_state((31 + slot_idx) as u32, 7 + slot_idx))
            })
            .expect("workspace stage should succeed");

        assert_eq!(lease.token_ids(3), &[11, 12, 13]);
        assert_eq!(lease.positions(3), &[31, 32, 33]);
        assert_eq!(lease.sequence_lengths(3), &[7, 8, 9]);
    }

    #[test]
    fn decode_workspace_lease_preserves_real_block_table_page_ids() {
        let plan = FastPathPlanner::plan_decode(&cuda_context(), 1, 5_000, 2_048)
            .expect("decode plan should resolve");
        let (_reservation, mut lease) =
            FastPathRuntime::checkout_decode_workspace(&plan).expect("checkout");
        let slot = DecodeSlotState::new(
            48,
            LiveKVBlockTable {
                page_tokens: DEFAULT_LIVE_KV_PAGE_TOKENS,
                spans: vec![
                    LiveKVBlockSpan {
                        page_id: 9,
                        start_token: 2,
                        token_count: 14,
                    },
                    LiveKVBlockSpan {
                        page_id: 4,
                        start_token: 0,
                        token_count: 16,
                    },
                    LiveKVBlockSpan {
                        page_id: 1,
                        start_token: 0,
                        token_count: 16,
                    },
                ],
            },
        )
        .expect("slot state");
        lease
            .stage(&[11], &[slot])
            .expect("workspace stage should succeed");

        let block_table = lease.block_table(0);
        assert_eq!(&block_table[..3], &[9, 4, 1]);
    }

    #[test]
    fn prefill_workspace_lease_stages_positions_and_reuses_bucket_pool() {
        let plan =
            FastPathPlanner::plan_prefill(&cuda_context(), 400).expect("prefill plan should work");
        let (first_reservation, mut first_lease) =
            FastPathRuntime::checkout_prefill_workspace(&plan).expect("first checkout");
        assert!(!first_reservation.reused_existing_arena);
        let first_positions = first_lease
            .stage_positions(64, 4)
            .expect("prefill stage should succeed")
            .to_vec();
        assert_eq!(first_positions, vec![64, 65, 66, 67]);
        drop(first_lease);

        let (second_reservation, mut second_lease) =
            FastPathRuntime::checkout_prefill_workspace(&plan).expect("second checkout");
        assert!(second_reservation.reused_existing_arena);
        let second_positions = second_lease
            .stage_positions(128, 3)
            .expect("prefill restage should succeed")
            .to_vec();
        assert_eq!(second_positions, vec![128, 129, 130]);
    }
}
