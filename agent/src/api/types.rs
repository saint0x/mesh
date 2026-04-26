use crate::connectivity::{DeviceConnectivityState, DirectPeerCandidate, NetworkConnectivity};
use crate::device::DeviceCapabilities;
use serde::{Deserialize, Serialize};

/// Request to register a new device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDeviceRequest {
    pub device_id: String,
    pub network_id: String,
    pub name: String,
    pub public_key: Vec<u8>,
    pub peer_id: String,
    pub capabilities: DeviceCapabilities,
}

/// Response to device registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDeviceResponse {
    pub success: bool,
    pub certificate: Option<Vec<u8>>,
    pub connectivity: NetworkConnectivity,
    pub message: Option<String>,
}

/// Request to update device heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub connectivity_state: DeviceConnectivityState,
    pub listen_addrs: Vec<String>,
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

/// Response to heartbeat update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub last_seen: String,
    pub connectivity_state: DeviceConnectivityState,
    pub listen_addrs: Vec<String>,
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub model_id: String,
    pub column_start: u32,
    pub column_end: u32,
    pub estimated_memory: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub device_id: String,
    pub peer_id: String,
    pub position: u32,
    pub status: String,
    pub contributed_memory: u64,
    pub shard: ShardInfo,
    pub left_neighbor: String,
    pub right_neighbor: String,
    pub connectivity_state: Option<DeviceConnectivityState>,
    pub listen_addrs: Vec<String>,
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingTopologyResponse {
    pub workers: Vec<WorkerInfo>,
    pub ring_stable: bool,
    #[serde(default)]
    pub peer_punch_plans: Vec<PeerPunchPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PunchPathReason {
    RelayPath,
    DegradedConnectivity,
    PrivateReachabilityOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PunchPathStrategy {
    SimultaneousDial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerPunchPlan {
    pub source_device_id: String,
    pub target_device_id: String,
    pub target_peer_id: String,
    pub strategy: PunchPathStrategy,
    pub reason: PunchPathReason,
    pub relay_rendezvous_required: bool,
    pub attempt_window_ms: u64,
    pub issued_at_ms: u64,
    #[serde(default)]
    pub target_candidates: Vec<DirectPeerCandidate>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgressEventKind {
    PrefillComplete,
    DecodeProgress,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferPolicy {
    CoLocated,
    ExportOnHandoff,
    RemoteAccess,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransportCapabilityTier {
    RelayFallback,
    DirectTcp,
    DirectPreferred,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGroupMember {
    pub device_id: String,
    pub peer_id: String,
    pub ring_position: u32,
    pub status: String,
    pub contributed_memory: u64,
    pub shard: ShardInfo,
    pub left_neighbor: String,
    pub right_neighbor: String,
    pub connectivity_state: Option<DeviceConnectivityState>,
    pub listen_addrs: Vec<String>,
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
    pub assigned_capacity_units: u32,
    pub execution_provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGroup {
    pub group_id: String,
    pub model_id: String,
    pub phase: ExecutionPhase,
    pub transport_tier: TransportCapabilityTier,
    pub kv_transfer_policy: KvTransferPolicy,
    pub total_capacity_units: u32,
    pub members: Vec<ExecutionGroupMember>,
    #[serde(default)]
    pub peer_punch_plans: Vec<PeerPunchPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSegment {
    pub segment_id: String,
    pub session_id: String,
    pub execution_group_id: String,
    pub phase: ExecutionPhase,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub kv_owner_device_id: String,
    pub shard_owner_device_ids: Vec<String>,
    pub participant_device_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceExecutionPlan {
    pub plan_id: String,
    #[serde(default)]
    pub runtime_mode: InferenceRuntimeMode,
    pub execution_groups: Vec<ExecutionGroup>,
    pub segments: Vec<ExecutionSegment>,
    pub initial_segment_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimInferenceAssignmentRequest {
    pub device_id: String,
    pub network_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceExecutionLease {
    pub lease_id: String,
    pub job_id: String,
    pub network_id: String,
    pub device_id: String,
    pub model_id: String,
    pub reserved_credits: f64,
    pub available_completion_tokens: u32,
    pub model_size_factor: f64,
    pub lease_expires_at: String,
    pub execution_plan: InferenceExecutionPlan,
    pub active_segment: ExecutionSegment,
    pub session: InferenceSessionLease,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimInferenceAssignmentResponse {
    pub success: bool,
    pub assignment: Option<InferenceExecutionLease>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcknowledgeInferenceAssignmentRequest {
    pub device_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportInferenceAssignmentRequest {
    pub device_id: String,
    pub segment_id: String,
    pub success: bool,
    pub completion: Option<String>,
    pub completion_tokens: Option<u32>,
    pub execution_time_ms: u64,
    pub time_to_first_token_ms: Option<u64>,
    pub kv_cache_seq_len: Option<u32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportInferenceAssignmentProgressRequest {
    pub device_id: String,
    pub segment_id: String,
    pub phase: ExecutionPhase,
    pub event: ProgressEventKind,
    pub completion_tokens: u32,
    pub execution_time_ms: u64,
    pub time_to_first_token_ms: Option<u64>,
    pub kv_cache_seq_len: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSessionLease {
    pub session_id: String,
    pub status: String,
    pub active_segment_id: Option<String>,
    pub kv_owner_device_id: String,
    pub kv_transfer_policy: KvTransferPolicy,
    pub kv_sequence_position: Option<u32>,
    pub kv_checkpoint_device_id: Option<String>,
    pub kv_checkpoint_created_at: Option<String>,
    pub updated_at: String,
    pub checkpoint: Option<InferenceSessionCheckpointStatus>,
    pub local_replica: Option<InferenceSessionReplicaStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSessionCheckpointStatus {
    pub checkpoint_id: String,
    pub source_device_id: String,
    pub source_segment_id: String,
    pub phase: ExecutionPhase,
    pub kv_sequence_position: u32,
    pub size_bytes: u64,
    pub sha256: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSessionReplicaStatus {
    pub device_id: String,
    pub status: String,
    pub active_segment_id: Option<String>,
    pub kv_sequence_position: Option<u32>,
    pub checkpoint_created_at: Option<String>,
    pub updated_at: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadInferenceSessionCheckpointRequest {
    pub device_id: String,
    pub session_id: String,
    pub segment_id: String,
    pub phase: ExecutionPhase,
    pub kv_sequence_position: u32,
    pub checkpoint_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadInferenceSessionCheckpointResponse {
    pub success: bool,
    pub checkpoint: Option<InferenceSessionCheckpointPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSessionCheckpointPayload {
    pub metadata: InferenceSessionCheckpointStatus,
    pub checkpoint_hex: String,
}
