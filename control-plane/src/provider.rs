use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    Cpu,
    Metal,
    Cuda,
}

impl ExecutionProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "cpu" => Some(Self::Cpu),
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCompatibilityClass {
    CpuPortable,
    MetalFastPath,
    CudaFastPath,
    HeterogeneousPortable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryModel {
    SystemRam,
    DiscreteVram,
    UnifiedMemory,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendContractDescriptor {
    pub provider: ExecutionProviderKind,
    pub compatibility_class: ProviderCompatibilityClass,
    pub optimization_profile: String,
    pub supports_decode_microbatch: bool,
    pub supports_paged_kv: bool,
    pub supports_checkpoint_handoff: bool,
    pub supports_device_sampling: bool,
    pub fast_path_eligible: bool,
    pub memory_model: MemoryModel,
    pub contract_hash: String,
}

impl BackendContractDescriptor {
    pub fn for_provider(provider: ExecutionProviderKind) -> Self {
        let compatibility_class = match provider {
            ExecutionProviderKind::Cpu => ProviderCompatibilityClass::CpuPortable,
            ExecutionProviderKind::Metal => ProviderCompatibilityClass::MetalFastPath,
            ExecutionProviderKind::Cuda => ProviderCompatibilityClass::CudaFastPath,
        };
        let optimization_profile = match provider {
            ExecutionProviderKind::Cpu => "cpu_serial",
            ExecutionProviderKind::Metal => "metal_vectorized",
            ExecutionProviderKind::Cuda => "cuda_fused",
        }
        .to_string();
        let supports_decode_microbatch = !matches!(provider, ExecutionProviderKind::Cpu);
        let supports_paged_kv = !matches!(provider, ExecutionProviderKind::Cpu);
        let supports_checkpoint_handoff = true;
        let supports_device_sampling = !matches!(provider, ExecutionProviderKind::Cpu);
        let fast_path_eligible = !matches!(provider, ExecutionProviderKind::Cpu);
        let memory_model = match provider {
            ExecutionProviderKind::Cpu => MemoryModel::SystemRam,
            ExecutionProviderKind::Metal => MemoryModel::UnifiedMemory,
            ExecutionProviderKind::Cuda => MemoryModel::DiscreteVram,
        };
        let mut descriptor = Self {
            provider,
            compatibility_class,
            optimization_profile,
            supports_decode_microbatch,
            supports_paged_kv,
            supports_checkpoint_handoff,
            supports_device_sampling,
            fast_path_eligible,
            memory_model,
            contract_hash: String::new(),
        };
        descriptor.contract_hash = descriptor.compute_contract_hash();
        descriptor
    }

    pub fn supports_production_serving(&self) -> bool {
        self.fast_path_eligible
            && self.supports_decode_microbatch
            && self.supports_paged_kv
            && self.supports_device_sampling
    }

    fn compute_contract_hash(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.provider.hash(&mut hasher);
        self.compatibility_class.hash(&mut hasher);
        self.optimization_profile.hash(&mut hasher);
        self.supports_decode_microbatch.hash(&mut hasher);
        self.supports_paged_kv.hash(&mut hasher);
        self.supports_checkpoint_handoff.hash(&mut hasher);
        self.supports_device_sampling.hash(&mut hasher);
        self.fast_path_eligible.hash(&mut hasher);
        self.memory_model.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProviderInfo {
    pub kind: ExecutionProviderKind,
    pub available: bool,
    pub reason: Option<String>,
    pub contract: BackendContractDescriptor,
}
