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
