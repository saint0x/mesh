use crate::errors::{AgentError, Result};
use crate::inference::runtime;
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

    pub fn production_readiness_summary(&self) -> String {
        if self.supports_production_serving() {
            format!(
                "provider {} satisfies fast-path serving requirements",
                self.provider.as_str()
            )
        } else {
            format!(
                "provider {} is not production serving ready (maturity={:?}, fast_path_eligible={}, supports_decode_microbatch={}, supports_paged_kv={}, supports_device_sampling={}, verified_fast_path_serving={}, verified_decode_microbatch={}, verified_paged_kv={}, verified_checkpoint_handoff={}, verified_device_sampling={})",
                self.provider.as_str(),
                self.implementation_maturity,
                self.fast_path_eligible,
                self.supports_decode_microbatch,
                self.supports_paged_kv,
                self.supports_device_sampling,
                self.verified_runtime.fast_path_serving,
                self.verified_runtime.decode_microbatch,
                self.verified_runtime.paged_kv,
                self.verified_runtime.checkpoint_handoff,
                self.verified_runtime.device_sampling
            )
        }
    }

    pub fn validate_runtime_consistency(&self) -> Result<()> {
        if self.fast_path_eligible && !self.verified_runtime.fast_path_serving {
            return Err(AgentError::Config(format!(
                "provider {} advertises fast-path eligibility without runtime verification",
                self.provider.as_str()
            )));
        }
        if self.supports_decode_microbatch && !self.verified_runtime.decode_microbatch {
            return Err(AgentError::Config(format!(
                "provider {} advertises decode microbatch without runtime verification",
                self.provider.as_str()
            )));
        }
        if self.supports_paged_kv && !self.verified_runtime.paged_kv {
            return Err(AgentError::Config(format!(
                "provider {} advertises paged KV without runtime verification",
                self.provider.as_str()
            )));
        }
        if self.supports_checkpoint_handoff && !self.verified_runtime.checkpoint_handoff {
            return Err(AgentError::Config(format!(
                "provider {} advertises checkpoint handoff without runtime verification",
                self.provider.as_str()
            )));
        }
        if self.supports_device_sampling && !self.verified_runtime.device_sampling {
            return Err(AgentError::Config(format!(
                "provider {} advertises device sampling without runtime verification",
                self.provider.as_str()
            )));
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

pub fn detect_execution_providers() -> Vec<ExecutionProviderInfo> {
    static DETECTED_PROVIDERS: OnceLock<Vec<ExecutionProviderInfo>> = OnceLock::new();
    DETECTED_PROVIDERS
        .get_or_init(detect_execution_providers_uncached)
        .clone()
}

fn detect_execution_providers_uncached() -> Vec<ExecutionProviderInfo> {
    vec![
        ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cpu,
            available: true,
            reason: None,
            contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu),
        },
        build_provider_info(
            ExecutionProviderKind::Metal,
            runtime::probe_provider(ExecutionProviderKind::Metal),
        ),
        build_provider_info(
            ExecutionProviderKind::Cuda,
            runtime::probe_provider(ExecutionProviderKind::Cuda),
        ),
    ]
}

fn build_provider_info(
    kind: ExecutionProviderKind,
    probe: (bool, Option<String>),
) -> ExecutionProviderInfo {
    let (available, reason) = probe;
    ExecutionProviderInfo {
        kind,
        available,
        reason,
        contract: BackendContractDescriptor::for_provider(kind),
    }
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

pub fn resolve_live_requested_provider(
    requested: Option<ExecutionProviderKind>,
) -> Result<ExecutionProviderKind> {
    let providers = detect_execution_providers();
    resolve_requested_provider(requested, &providers)
}

static SELECTED_PROVIDER: OnceLock<ExecutionProviderKind> = OnceLock::new();

pub fn set_selected_execution_provider(provider: ExecutionProviderKind) -> Result<()> {
    let providers = detect_execution_providers();
    resolve_requested_provider(Some(provider), &providers)?;
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
