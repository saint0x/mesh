// Re-export DeviceCapabilities and Tier from agent crate for use in control plane
// This ensures consistency between agent and control plane

use serde::{Deserialize, Serialize};

use crate::provider::{
    BackendContractDescriptor, ExecutionProviderInfo, ExecutionProviderKind, MemoryModel,
};

/// Device hardware capabilities and tier classification
/// (Mirrors agent::device::DeviceCapabilities)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceCapabilities {
    pub tier: Tier,
    pub cpu_cores: usize,
    pub ram_mb: usize,
    pub gpu_present: bool,
    pub gpu_vram_mb: Option<usize>,
    pub os: String,
    pub arch: String,
    #[serde(default)]
    pub execution_providers: Vec<ExecutionProviderInfo>,
    pub default_execution_provider: ExecutionProviderKind,
    #[serde(default)]
    pub provider_contracts: Vec<BackendContractDescriptor>,
    pub default_provider_contract_hash: String,
    pub memory_model: MemoryModel,
}

impl DeviceCapabilities {
    pub fn default_backend_contract(&self) -> BackendContractDescriptor {
        self.provider_contracts
            .iter()
            .find(|contract| contract.contract_hash == self.default_provider_contract_hash)
            .cloned()
            .unwrap_or_else(|| {
                BackendContractDescriptor::for_provider(self.default_execution_provider)
            })
    }

    pub fn validate_provider_contracts(&self) -> Result<(), String> {
        if self.provider_contracts.is_empty() {
            return Err("device did not report any provider contracts".to_string());
        }
        if !self
            .provider_contracts
            .iter()
            .any(|contract| contract.contract_hash == self.default_provider_contract_hash)
        {
            return Err(format!(
                "default provider contract hash {} is not present in the reported inventory",
                self.default_provider_contract_hash
            ));
        }
        for contract in &self.provider_contracts {
            contract.validate_runtime_consistency()?;
        }
        Ok(())
    }
}

/// Device tier based on hardware capabilities
/// (Mirrors agent::device::Tier)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    /// Minimal device (1-2 cores, <4GB RAM)
    Tier0,
    /// Low-end device (2-4 cores, 4-8GB RAM)
    Tier1,
    /// Mid-range device (4-8 cores, 8-16GB RAM)
    Tier2,
    /// High-end device (8-16 cores, 16-32GB RAM)
    Tier3,
    /// Server-class device (16+ cores, 32GB+ RAM)
    Tier4,
}
