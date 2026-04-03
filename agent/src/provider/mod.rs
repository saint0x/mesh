use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProviderInfo {
    pub kind: ExecutionProviderKind,
    pub available: bool,
    pub reason: Option<String>,
}

pub fn detect_execution_providers() -> Vec<ExecutionProviderInfo> {
    let mut providers = vec![ExecutionProviderInfo {
        kind: ExecutionProviderKind::Cpu,
        available: true,
        reason: None,
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
        });
    }

    #[cfg(not(target_os = "macos"))]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Metal,
            available: false,
            reason: Some("metal provider is only available on macOS".to_string()),
        });
    }

    #[cfg(target_os = "linux")]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cuda,
            available: true,
            reason: None,
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        providers.push(ExecutionProviderInfo {
            kind: ExecutionProviderKind::Cuda,
            available: false,
            reason: Some("cuda provider is only available on Linux builds".to_string()),
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
