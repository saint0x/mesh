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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimInferenceAssignmentRequest {
    pub device_id: String,
    pub network_id: String,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimInferenceAssignmentResponse {
    pub success: bool,
    pub assignment: Option<InferenceAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcknowledgeInferenceAssignmentRequest {
    pub device_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportInferenceAssignmentRequest {
    pub device_id: String,
    pub success: bool,
    pub completion: Option<String>,
    pub completion_tokens: Option<u32>,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}
