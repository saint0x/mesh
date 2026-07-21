use crate::errors::{AgentError, Result};
use crate::inference::runtime;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderKind {
    Cpu,
    Metal,
    Cuda,
    Rocm,
}

impl ExecutionProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            "rocm" => Some(Self::Rocm),
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
    RocmFastPath,
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
    RuntimeUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveKvCacheLayout {
    HostPaged,
    DevicePaged,
    DeviceContiguousWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct VerifiedRuntimeCapabilities {
    pub fast_path_serving: bool,
    pub decode_microbatch: bool,
    pub live_kv: bool,
    pub checkpoint_handoff: bool,
    pub device_sampling: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendContractDescriptor {
    pub provider: ExecutionProviderKind,
    pub compatibility_class: ProviderCompatibilityClass,
    pub optimization_profile: String,
    pub supports_decode_microbatch: bool,
    pub supports_live_kv: bool,
    pub kv_cache_layout: LiveKvCacheLayout,
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
        Self::for_provider_with_maturity(provider, ProviderImplementationMaturity::VerifiedFastPath)
    }

    pub fn runtime_unavailable(provider: ExecutionProviderKind) -> Self {
        Self::for_provider_with_maturity(
            provider,
            ProviderImplementationMaturity::RuntimeUnavailable,
        )
    }

    fn for_provider_with_maturity(
        provider: ExecutionProviderKind,
        implementation_maturity: ProviderImplementationMaturity,
    ) -> Self {
        let compatibility_class = match provider {
            ExecutionProviderKind::Cpu => ProviderCompatibilityClass::CpuPortable,
            ExecutionProviderKind::Metal => ProviderCompatibilityClass::MetalFastPath,
            ExecutionProviderKind::Cuda => ProviderCompatibilityClass::CudaFastPath,
            ExecutionProviderKind::Rocm => ProviderCompatibilityClass::RocmFastPath,
        };
        let optimization_profile = match implementation_maturity {
            ProviderImplementationMaturity::RuntimeUnavailable => "runtime_unavailable",
            ProviderImplementationMaturity::VerifiedFastPath => match provider {
                ExecutionProviderKind::Cpu => "cpu_serial",
                ExecutionProviderKind::Metal => "metal_vectorized",
                ExecutionProviderKind::Cuda => "cuda_fused",
                ExecutionProviderKind::Rocm => "rocm_fused",
            },
        }
        .to_string();
        let supports_decode_microbatch = !matches!(provider, ExecutionProviderKind::Rocm);
        let supports_live_kv = true;
        let kv_cache_layout = match provider {
            ExecutionProviderKind::Cpu => LiveKvCacheLayout::HostPaged,
            ExecutionProviderKind::Metal | ExecutionProviderKind::Cuda => {
                LiveKvCacheLayout::DevicePaged
            }
            ExecutionProviderKind::Rocm => LiveKvCacheLayout::DeviceContiguousWindow,
        };
        let supports_checkpoint_handoff = true;
        let supports_device_sampling = !matches!(provider, ExecutionProviderKind::Rocm);
        let fast_path_eligible = true;
        let memory_model = match provider {
            ExecutionProviderKind::Cpu => MemoryModel::SystemRam,
            ExecutionProviderKind::Metal => MemoryModel::UnifiedMemory,
            ExecutionProviderKind::Cuda => MemoryModel::DiscreteVram,
            ExecutionProviderKind::Rocm => MemoryModel::DiscreteVram,
        };
        let verified_runtime = match implementation_maturity {
            ProviderImplementationMaturity::RuntimeUnavailable => {
                VerifiedRuntimeCapabilities::default()
            }
            ProviderImplementationMaturity::VerifiedFastPath => match provider {
                ExecutionProviderKind::Metal => VerifiedRuntimeCapabilities {
                    fast_path_serving: true,
                    decode_microbatch: true,
                    live_kv: true,
                    checkpoint_handoff: true,
                    device_sampling: true,
                },
                ExecutionProviderKind::Cpu => VerifiedRuntimeCapabilities {
                    fast_path_serving: true,
                    decode_microbatch: true,
                    live_kv: true,
                    checkpoint_handoff: true,
                    device_sampling: true,
                },
                ExecutionProviderKind::Cuda => VerifiedRuntimeCapabilities {
                    fast_path_serving: true,
                    decode_microbatch: true,
                    live_kv: true,
                    checkpoint_handoff: true,
                    device_sampling: true,
                },
                ExecutionProviderKind::Rocm => VerifiedRuntimeCapabilities {
                    fast_path_serving: true,
                    decode_microbatch: false,
                    live_kv: true,
                    checkpoint_handoff: true,
                    device_sampling: false,
                },
            },
        };
        let supports_decode_microbatch = supports_decode_microbatch
            && matches!(
                implementation_maturity,
                ProviderImplementationMaturity::VerifiedFastPath
            );
        let supports_live_kv = supports_live_kv
            && matches!(
                implementation_maturity,
                ProviderImplementationMaturity::VerifiedFastPath
            );
        let supports_checkpoint_handoff = supports_checkpoint_handoff
            && matches!(
                implementation_maturity,
                ProviderImplementationMaturity::VerifiedFastPath
            );
        let supports_device_sampling = supports_device_sampling
            && matches!(
                implementation_maturity,
                ProviderImplementationMaturity::VerifiedFastPath
            );
        let fast_path_eligible = fast_path_eligible
            && matches!(
                implementation_maturity,
                ProviderImplementationMaturity::VerifiedFastPath
            );
        let mut descriptor = Self {
            provider,
            compatibility_class,
            optimization_profile,
            supports_decode_microbatch,
            supports_live_kv,
            kv_cache_layout,
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
            && self.supports_live_kv
            && self.verified_runtime.fast_path_serving
            && self.verified_runtime.live_kv
            && (!self.supports_decode_microbatch || self.verified_runtime.decode_microbatch)
            && (!self.supports_checkpoint_handoff || self.verified_runtime.checkpoint_handoff)
            && (!self.supports_device_sampling || self.verified_runtime.device_sampling)
    }

    pub fn production_readiness_summary(&self) -> String {
        if self.supports_production_serving() {
            format!(
                "provider {} satisfies fast-path serving requirements",
                self.provider.as_str()
            )
        } else {
            format!(
                "provider {} is not production serving ready (maturity={:?}, fast_path_eligible={}, supports_decode_microbatch={}, supports_live_kv={}, kv_cache_layout={:?}, supports_device_sampling={}, verified_fast_path_serving={}, verified_decode_microbatch={}, verified_live_kv={}, verified_checkpoint_handoff={}, verified_device_sampling={})",
                self.provider.as_str(),
                self.implementation_maturity,
                self.fast_path_eligible,
                self.supports_decode_microbatch,
                self.supports_live_kv,
                self.kv_cache_layout,
                self.supports_device_sampling,
                self.verified_runtime.fast_path_serving,
                self.verified_runtime.decode_microbatch,
                self.verified_runtime.live_kv,
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
        if self.supports_live_kv && !self.verified_runtime.live_kv {
            return Err(AgentError::Config(format!(
                "provider {} advertises live KV without runtime verification",
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
        self.supports_live_kv.hash(&mut hasher);
        self.kv_cache_layout.hash(&mut hasher);
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
        build_provider_info(
            ExecutionProviderKind::Rocm,
            runtime::probe_provider(ExecutionProviderKind::Rocm),
        ),
    ]
}

fn build_provider_info(
    kind: ExecutionProviderKind,
    probe: (bool, Option<String>),
) -> ExecutionProviderInfo {
    let (available, reason) = probe;
    let contract = if available {
        BackendContractDescriptor::for_provider(kind)
    } else {
        BackendContractDescriptor::runtime_unavailable(kind)
    };
    ExecutionProviderInfo {
        kind,
        available,
        reason,
        contract,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_parser_accepts_normalized_rocm_label() {
        assert_eq!(
            ExecutionProviderKind::from_str(" ROCM "),
            Some(ExecutionProviderKind::Rocm)
        );
    }

    #[test]
    fn rocm_contract_is_discrete_fast_path_contract() {
        let contract = BackendContractDescriptor::for_provider(ExecutionProviderKind::Rocm);

        assert_eq!(contract.provider, ExecutionProviderKind::Rocm);
        assert_eq!(
            contract.compatibility_class,
            ProviderCompatibilityClass::RocmFastPath
        );
        assert_eq!(contract.optimization_profile, "rocm_fused");
        assert_eq!(contract.memory_model, MemoryModel::DiscreteVram);
        assert_eq!(
            contract.kv_cache_layout,
            LiveKvCacheLayout::DeviceContiguousWindow
        );
        assert!(!contract.supports_decode_microbatch);
        assert!(!contract.supports_device_sampling);
        assert!(contract.supports_production_serving());
    }

    #[test]
    fn unavailable_rocm_contract_is_not_production_serving() {
        let contract = BackendContractDescriptor::runtime_unavailable(ExecutionProviderKind::Rocm);

        assert_eq!(
            contract.implementation_maturity,
            ProviderImplementationMaturity::RuntimeUnavailable
        );
        assert!(!contract.fast_path_eligible);
        assert!(!contract.supports_decode_microbatch);
        assert!(!contract.supports_live_kv);
        assert!(!contract.supports_device_sampling);
        assert!(!contract.supports_production_serving());
        contract.validate_runtime_consistency().unwrap();
    }

    #[test]
    fn requested_rocm_provider_fails_closed_when_runtime_is_unavailable() {
        let providers = vec![
            ExecutionProviderInfo {
                kind: ExecutionProviderKind::Cpu,
                available: true,
                reason: None,
                contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu),
            },
            ExecutionProviderInfo {
                kind: ExecutionProviderKind::Rocm,
                available: false,
                reason: Some("native HIP/ROCm tensor backend is not linked".to_string()),
                contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Rocm),
            },
        ];

        let error = resolve_requested_provider(Some(ExecutionProviderKind::Rocm), &providers)
            .expect_err("rocm must not silently fallback");
        assert!(error
            .to_string()
            .contains("native HIP/ROCm tensor backend is not linked"));
    }
}
