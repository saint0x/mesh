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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderImplementationMaturity {
    #[default]
    VerifiedFastPath,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct VerifiedRuntimeCapabilities {
    pub fast_path_serving: bool,
    pub decode_microbatch: bool,
    pub paged_kv: bool,
    pub checkpoint_handoff: bool,
    pub device_sampling: bool,
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
    #[serde(default)]
    pub implementation_maturity: ProviderImplementationMaturity,
    #[serde(default)]
    pub verified_runtime: VerifiedRuntimeCapabilities,
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
        let supports_decode_microbatch = true;
        let supports_paged_kv = true;
        let supports_checkpoint_handoff = true;
        let supports_device_sampling = !matches!(provider, ExecutionProviderKind::Cpu);
        let fast_path_eligible = true;
        let memory_model = match provider {
            ExecutionProviderKind::Cpu => MemoryModel::SystemRam,
            ExecutionProviderKind::Metal => MemoryModel::UnifiedMemory,
            ExecutionProviderKind::Cuda => MemoryModel::DiscreteVram,
        };
        let implementation_maturity = ProviderImplementationMaturity::VerifiedFastPath;
        let verified_runtime = match provider {
            ExecutionProviderKind::Metal => VerifiedRuntimeCapabilities {
                fast_path_serving: true,
                decode_microbatch: true,
                paged_kv: true,
                checkpoint_handoff: true,
                device_sampling: true,
            },
            ExecutionProviderKind::Cpu => VerifiedRuntimeCapabilities {
                fast_path_serving: true,
                decode_microbatch: true,
                paged_kv: true,
                checkpoint_handoff: true,
                device_sampling: false,
            },
            ExecutionProviderKind::Cuda => VerifiedRuntimeCapabilities {
                fast_path_serving: true,
                decode_microbatch: true,
                paged_kv: true,
                checkpoint_handoff: true,
                device_sampling: true,
            },
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
            implementation_maturity,
            verified_runtime,
            contract_hash: String::new(),
        };
        descriptor.contract_hash = descriptor.compute_contract_hash();
        descriptor
    }

    pub fn supports_production_serving(&self) -> bool {
        self.fast_path_eligible
            && self.supports_decode_microbatch
            && self.supports_paged_kv
            && self.verified_runtime.fast_path_serving
            && self.verified_runtime.decode_microbatch
            && self.verified_runtime.paged_kv
    }

    pub fn validate_runtime_consistency(&self) -> Result<(), String> {
        if self.fast_path_eligible && !self.verified_runtime.fast_path_serving {
            return Err(format!(
                "provider {} advertises fast-path eligibility without runtime verification",
                self.provider.as_str()
            ));
        }
        if self.supports_decode_microbatch && !self.verified_runtime.decode_microbatch {
            return Err(format!(
                "provider {} advertises decode microbatch without runtime verification",
                self.provider.as_str()
            ));
        }
        if self.supports_paged_kv && !self.verified_runtime.paged_kv {
            return Err(format!(
                "provider {} advertises paged KV without runtime verification",
                self.provider.as_str()
            ));
        }
        if self.supports_checkpoint_handoff && !self.verified_runtime.checkpoint_handoff {
            return Err(format!(
                "provider {} advertises checkpoint handoff without runtime verification",
                self.provider.as_str()
            ));
        }
        if self.supports_device_sampling && !self.verified_runtime.device_sampling {
            return Err(format!(
                "provider {} advertises device sampling without runtime verification",
                self.provider.as_str()
            ));
        }
        Ok(())
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
        self.implementation_maturity.hash(&mut hasher);
        self.verified_runtime.hash(&mut hasher);
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
