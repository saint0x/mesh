use crate::connectivity::NetworkConnectivity;
use crate::device::DeviceCapabilities;
use serde::{Deserialize, Serialize};

/// Request to register a new device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDeviceRequest {
    pub device_id: String,
    pub network_id: String,
    pub name: String,
    pub public_key: Vec<u8>,
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
pub struct HeartbeatRequest {}

/// Response to heartbeat update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub last_seen: String,
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
