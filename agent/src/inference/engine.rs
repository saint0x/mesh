use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::provider::ExecutionProviderKind;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSessionState {
    pub assignment: SessionAssignment,
    pub backend: BackendInstanceSpec,
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
        provider: ExecutionProviderKind,
        shard_column_range: (u32, u32),
        transport_tier: TransportCapabilityTier,
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
            max_tokens,
            prompt_tokens,
        }
    }
}
