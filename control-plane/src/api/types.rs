use crate::connectivity::{
    DeviceConnectivityState, DirectPeerCandidate, InferenceSchedulingPolicy, NetworkConnectivity,
};
use crate::device::DeviceCapabilities;
use serde::{Deserialize, Serialize};

/// Request to register a new device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDeviceRequest {
    /// Unique device identifier (UUID as string)
    pub device_id: String,
    /// Network this device wants to join
    pub network_id: String,
    /// Human-readable device name
    pub name: String,
    /// Ed25519 public key (32 bytes)
    pub public_key: Vec<u8>,
    /// Libp2p peer ID derived from the device identity key
    pub peer_id: String,
    /// Device hardware capabilities
    pub capabilities: DeviceCapabilities,
    /// Optional: Memory to contribute to the pool (bytes).
    /// If provided, device will automatically join the ring.
    #[serde(default)]
    pub contributed_memory: Option<u64>,
}

/// Response to device registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDeviceResponse {
    /// Whether registration succeeded
    pub success: bool,
    /// Self-signed certificate blob (CBOR-encoded)
    pub certificate: Option<Vec<u8>>,
    /// Connectivity profile for this network
    pub connectivity: NetworkConnectivity,
    /// Error or success message
    pub message: Option<String>,
    /// Ring position info (if device joined ring)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ring_position: Option<RingPositionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNetworkRequest {
    pub network_id: String,
    pub name: String,
    pub owner_user_id: String,
    pub connectivity: NetworkConnectivity,
    #[serde(default)]
    pub scheduling_policy: InferenceSchedulingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateNetworkResponse {
    pub success: bool,
    pub network: NetworkInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListNetworksResponse {
    pub success: bool,
    pub networks: Vec<NetworkInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub network_id: String,
    pub name: String,
    pub owner_user_id: String,
    pub created_at: String,
    pub connectivity: NetworkConnectivity,
    #[serde(default)]
    pub scheduling_policy: InferenceSchedulingPolicy,
}

/// Ring position information in registration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingPositionInfo {
    /// Assigned ring position
    pub position: u32,
    /// Assigned shard information
    pub shard: ShardInfo,
    /// Left neighbor device ID
    pub left_neighbor: String,
    /// Right neighbor device ID
    pub right_neighbor: String,
}

/// Request to update device heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub connectivity_state: DeviceConnectivityState,
    #[serde(default)]
    pub listen_addrs: Vec<String>,
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

/// Response to heartbeat update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    /// Whether heartbeat succeeded
    pub success: bool,
    /// Updated last_seen timestamp (ISO 8601)
    pub last_seen: String,
    /// Recorded connectivity state
    pub connectivity_state: DeviceConnectivityState,
    /// Recorded listen addresses
    pub listen_addrs: Vec<String>,
    /// Recorded direct-connect candidates
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

/// Request to join the ring topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingJoinRequest {
    /// Device ID joining the ring
    pub device_id: String,
    /// Network ID
    pub network_id: String,
    /// Memory contributed to the pool (bytes)
    pub contributed_memory: u64,
}

/// Response to ring join request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingJoinResponse {
    /// Whether join succeeded
    pub success: bool,
    /// Assigned ring position
    pub position: u32,
    /// Assigned shard information
    pub shard: ShardInfo,
    /// Left neighbor device ID
    pub left_neighbor: String,
    /// Right neighbor device ID
    pub right_neighbor: String,
}

/// Shard information in API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Model ID this shard belongs to
    pub model_id: String,
    /// Column range start (inclusive)
    pub column_start: u32,
    /// Column range end (exclusive)
    pub column_end: u32,
    /// Estimated memory requirement in bytes
    pub estimated_memory: u64,
}

/// Response to ring topology query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingTopologyResponse {
    /// Workers in the ring, ordered by position
    pub workers: Vec<WorkerInfo>,
    /// Whether the ring is stable
    pub ring_stable: bool,
    /// Explicit punched-path coordination plans
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

/// Worker information in topology response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Device ID
    pub device_id: String,
    /// Libp2p peer ID
    pub peer_id: String,
    /// Ring position
    pub position: u32,
    /// Reported device status
    pub status: String,
    /// Reported contributed memory in bytes
    pub contributed_memory: u64,
    /// Assigned shard
    pub shard: ShardInfo,
    /// Left neighbor device ID
    pub left_neighbor: String,
    /// Right neighbor device ID
    pub right_neighbor: String,
    /// Latest reported connectivity state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connectivity_state: Option<DeviceConnectivityState>,
    /// Latest reported direct listen addresses
    #[serde(default)]
    pub listen_addrs: Vec<String>,
    /// Ranked direct-connect candidates for peer dialing
    #[serde(default)]
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

/// Response to ring leave request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingLeaveResponse {
    /// Whether leave succeeded
    pub success: bool,
}

// ==================== Handoff API Types ====================

/// Request to create a shard handoff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateHandoffRequest {
    /// Network ID
    pub network_id: String,
    /// Source device (giving up shard)
    pub source_device: String,
    /// Target device (receiving shard)
    pub target_device: String,
    /// Column range to transfer (start, end)
    pub column_start: u32,
    pub column_end: u32,
    /// Model ID
    pub model_id: String,
}

/// Response to handoff creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateHandoffResponse {
    /// Whether creation succeeded
    pub success: bool,
    /// Handoff ID
    pub handoff_id: String,
    /// Initial status
    pub status: String,
}

/// Request to update handoff status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateHandoffRequest {
    /// New status
    pub status: String,
    /// Optional bytes transferred
    pub bytes_transferred: Option<u64>,
    /// Optional total bytes
    pub total_bytes: Option<u64>,
    /// Optional error message
    pub error: Option<String>,
}

/// Response with handoff details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffInfo {
    /// Handoff ID
    pub handoff_id: String,
    /// Network ID
    pub network_id: String,
    /// Source device
    pub source_device: String,
    /// Target device
    pub target_device: String,
    /// Column range
    pub column_start: u32,
    pub column_end: u32,
    /// Model ID
    pub model_id: String,
    /// Current status
    pub status: String,
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Progress (0.0 - 1.0)
    pub progress: f32,
    /// Start timestamp
    pub started_at: u64,
    /// Completion timestamp
    pub completed_at: Option<u64>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Response listing handoffs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListHandoffsResponse {
    /// Active handoffs
    pub handoffs: Vec<HandoffInfo>,
}

/// Request to register worker callback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterCallbackRequest {
    /// Device ID
    pub device_id: String,
    /// Optional callback URL for webhooks
    pub callback_url: Option<String>,
}

/// Response to callback registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterCallbackResponse {
    /// Whether registration succeeded
    pub success: bool,
}

/// Request for topology version check (polling)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyVersionRequest {
    /// Network ID
    pub network_id: String,
    /// Last known version
    pub since_version: u64,
}

/// Response with topology version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyVersionResponse {
    /// Current version
    pub current_version: u64,
    /// Whether there are updates since the provided version
    pub has_updates: bool,
}

// ==================== Distributed Inference API Types ====================

/// Request to submit a distributed inference job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitInferenceRequest {
    /// Device ID submitting the job
    pub device_id: String,
    /// Network ID
    pub network_id: String,
    /// Model ID to use
    pub model_id: String,
    /// Input prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
}

/// Response to inference submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitInferenceResponse {
    /// Whether submission succeeded
    pub success: bool,
    /// Job ID for tracking
    pub job_id: String,
    /// Generated completion text (if immediate execution)
    pub completion: Option<String>,
    /// Number of tokens generated
    pub completion_tokens: u32,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
}

/// Worker assignment for a distributed inference job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceAssignment {
    pub assignment_id: String,
    pub job_id: String,
    pub network_id: String,
    pub device_id: String,
    pub ring_position: u32,
    pub model_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub lease_expires_at: String,
}

/// Request for a worker to claim its next assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimInferenceAssignmentRequest {
    pub device_id: String,
    pub network_id: String,
}

/// Response for claiming a worker assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimInferenceAssignmentResponse {
    pub success: bool,
    pub assignment: Option<InferenceAssignment>,
}

/// Request to acknowledge worker execution start
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcknowledgeInferenceAssignmentRequest {
    pub device_id: String,
}

/// Request to record a worker result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportInferenceAssignmentRequest {
    pub device_id: String,
    pub success: bool,
    pub completion: Option<String>,
    pub completion_tokens: Option<u32>,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

/// Response describing a submitted inference job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJobStatusResponse {
    pub success: bool,
    pub job_id: String,
    pub network_id: String,
    pub model_id: String,
    pub status: String,
    pub completion: Option<String>,
    pub completion_tokens: u32,
    pub execution_time_ms: u64,
    pub error: Option<String>,
    pub assignments: Vec<InferenceJobAssignmentStatus>,
}

/// Assignment status inside a job status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJobAssignmentStatus {
    pub device_id: String,
    pub ring_position: u32,
    pub status: String,
    pub failure_reason: Option<String>,
}
