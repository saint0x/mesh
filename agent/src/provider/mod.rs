use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

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
        match value.trim().to_ascii_lowercase().as_str() {
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

    pub fn production_readiness_summary(&self) -> String {
        if self.supports_production_serving() {
            format!(
                "provider {} satisfies fast-path serving requirements",
                self.provider.as_str()
            )
        } else {
            format!(
                "provider {} is not production serving ready (fast_path_eligible={}, supports_decode_microbatch={}, supports_paged_kv={}, supports_device_sampling={})",
                self.provider.as_str(),
                self.fast_path_eligible,
                self.supports_decode_microbatch,
                self.supports_paged_kv,
                self.supports_device_sampling
            )
        }
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

pub fn detect_execution_providers() -> Vec<ExecutionProviderInfo> {
    let mut providers = vec![ExecutionProviderInfo {
        kind: ExecutionProviderKind::Cpu,
        available: true,
        reason: None,
        contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu),
    }];

    #[cfg(target_os = "macos")]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Metal,
            available: cfg!(target_arch = "aarch64"),
            reason: if cfg!(target_arch = "aarch64") {
                None
            } else {
                Some("metal provider requires Apple Silicon for production support".to_string())
            },
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Metal),
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Metal,
            available: false,
            reason: Some("metal provider is only available on macOS".to_string()),
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Metal),
        });
    }

    #[cfg(target_os = "linux")]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cuda,
            available: true,
            reason: None,
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cuda),
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cuda,
            available: false,
            reason: Some("cuda provider is only available on Linux builds".to_string()),
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cuda),
        });
    }

    providers
}

pub fn default_execution_provider(providers: &[ExecutionProviderInfo]) -> ExecutionProviderKind {
    providers
        .iter()
        .find(|provider| provider.available && provider.kind != ExecutionProviderKind::Cpu)
        .map(|provider| provider.kind)
        .unwrap_or(ExecutionProviderKind::Cpu)
}

pub fn default_execution_contract(
    providers: &[ExecutionProviderInfo],
) -> BackendContractDescriptor {
    let selected = default_execution_provider(providers);
    providers
        .iter()
        .find(|provider| provider.kind == selected)
        .map(|provider| provider.contract.clone())
        .unwrap_or_else(|| BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu))
}

pub fn resolve_requested_provider(
    requested: Option<ExecutionProviderKind>,
    providers: &[ExecutionProviderInfo],
) -> Result<ExecutionProviderKind> {
    let selected = requested.unwrap_or_else(|| default_execution_provider(providers));
    let descriptor = providers
        .iter()
        .find(|provider| provider.kind == selected)
        .ok_or_else(|| {
            AgentError::Config(format!(
                "Execution provider {} is not described on this node",
                selected.as_str()
            ))
        })?;

    if !descriptor.available {
        return Err(AgentError::Config(format!(
            "Execution provider {} is unavailable: {}",
            selected.as_str(),
            descriptor
                .reason
                .clone()
                .unwrap_or_else(|| "no reason provided".to_string())
        )));
    }

    Ok(selected)
}

static SELECTED_PROVIDER: OnceLock<ExecutionProviderKind> = OnceLock::new();

pub fn set_selected_execution_provider(provider: ExecutionProviderKind) -> Result<()> {
    match SELECTED_PROVIDER.set(provider) {
        Ok(()) => Ok(()),
        Err(existing) if existing == provider => Ok(()),
        Err(existing) => Err(AgentError::Config(format!(
            "Execution provider already initialized to {}, cannot change to {}",
            existing.as_str(),
            provider.as_str()
        ))),
    }
}

pub fn selected_execution_provider() -> Option<ExecutionProviderKind> {
    SELECTED_PROVIDER.get().copied()
}
