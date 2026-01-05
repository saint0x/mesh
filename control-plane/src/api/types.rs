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
    /// List of relay server addresses
    pub relay_addresses: Vec<String>,
    /// Error or success message
    pub message: Option<String>,
    /// Ring position info (if device joined ring)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ring_position: Option<RingPositionInfo>,
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
    // Empty for MVP, can add metrics later
}

/// Response to heartbeat update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    /// Whether heartbeat succeeded
    pub success: bool,
    /// Updated last_seen timestamp (ISO 8601)
    pub last_seen: String,
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
}

/// Worker information in topology response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Device ID
    pub device_id: String,
    /// Ring position
    pub position: u32,
    /// Assigned shard
    pub shard: ShardInfo,
    /// Left neighbor device ID
    pub left_neighbor: String,
    /// Right neighbor device ID
    pub right_neighbor: String,
}

/// Response to ring leave request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingLeaveResponse {
    /// Whether leave succeeded
    pub success: bool,
}
