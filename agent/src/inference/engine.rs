use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::provider::ExecutionProviderKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceRuntimeMode {
    FitFirst,
    ThroughputFirst,
    LatencyFirst,
    ResilientEdge,
}

impl Default for InferenceRuntimeMode {
    fn default() -> Self {
        Self::ThroughputFirst
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferPolicy {
    CoLocated,
    ExportOnHandoff,
    RemoteAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportCapabilityTier {
    RelayFallback,
    DirectTcp,
    DirectPreferred,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAssignment {
    pub session_id: Uuid,
    pub phase: ExecutionPhase,
    pub kv_owner_device_id: String,
    pub shard_owner_device_ids: Vec<String>,
    pub participant_device_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInstanceSpec {
    pub provider: ExecutionProviderKind,
    pub shard_column_range: (u32, u32),
    pub transport_tier: TransportCapabilityTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPauseReason {
    Regroup,
    Failover,
    LocalKvNotReady,
    RuntimeMemoryPressure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPauseState {
    pub reason: SessionPauseReason,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEvictionReason {
    RuntimeMemoryPressure,
    IdleSession,
    CompletedSession,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEvictionState {
    pub reason: SessionEvictionReason,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRuntimeStatus {
    Active,
    Paused,
    Evicted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendOptimizationProfile {
    CpuSerial,
    MetalVectorized,
    CudaFused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RuntimeMemoryBudget {
    pub max_active_sessions: usize,
    pub max_total_kv_cache_bytes: usize,
    pub max_total_runtime_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSessionState {
    pub assignment: SessionAssignment,
    pub backend: BackendInstanceSpec,
    pub runtime_mode: InferenceRuntimeMode,
    pub runtime_status: SessionRuntimeStatus,
    pub paused: Option<SessionPauseState>,
    pub evicted: Option<SessionEvictionState>,
    pub memory_budget: RuntimeMemoryBudget,
    pub max_tokens: u32,
    pub prompt_tokens: usize,
}

impl EngineSessionState {
    pub fn new(
        session_id: Uuid,
        phase: ExecutionPhase,
        kv_owner_device_id: String,
        shard_owner_device_ids: Vec<String>,
        participant_device_ids: Vec<String>,
        runtime_mode: InferenceRuntimeMode,
        provider: ExecutionProviderKind,
        shard_column_range: (u32, u32),
        transport_tier: TransportCapabilityTier,
        memory_budget: RuntimeMemoryBudget,
        max_tokens: u32,
        prompt_tokens: usize,
    ) -> Self {
        Self {
            assignment: SessionAssignment {
                session_id,
                phase,
                kv_owner_device_id,
                shard_owner_device_ids,
                participant_device_ids,
            },
            backend: BackendInstanceSpec {
                provider,
                shard_column_range,
                transport_tier,
            },
            runtime_mode,
            runtime_status: SessionRuntimeStatus::Active,
            paused: None,
            evicted: None,
            memory_budget,
            max_tokens,
            prompt_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeTask {
    pub session_id: Uuid,
    pub fairness_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatchPolicy {
    pub max_batch_size: usize,
    pub max_total_kv_tokens: usize,
}

impl Default for DecodeBatchPolicy {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            max_total_kv_tokens: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatchSlot {
    pub session_id: Uuid,
    pub fairness_epoch: u64,
    pub kv_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBatchPlan {
    pub slots: Vec<DecodeBatchSlot>,
    pub deferred: Vec<DecodeTask>,
    pub total_kv_tokens: usize,
    pub deferred_for_capacity: usize,
    pub deferred_for_kv_budget: usize,
}

impl DecodeBatchPlan {
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}
