// Re-export DeviceCapabilities and Tier from agent crate for use in control plane
// This ensures consistency between agent and control plane

use serde::{Deserialize, Serialize};

/// Device hardware capabilities and tier classification
/// (Mirrors agent::device::DeviceCapabilities)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceCapabilities {
    pub cpu_cores: usize,
    pub ram_mb: usize,
    pub os: String,
    pub arch: String,
    pub has_gpu: bool,
    pub tier: Tier,
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

impl Tier {
    /// Get credit multiplier for this tier
    pub fn credit_multiplier(&self) -> f64 {
        match self {
            Tier::Tier0 => 1.0,
            Tier::Tier1 => 2.0,
            Tier::Tier2 => 4.0,
            Tier::Tier3 => 8.0,
            Tier::Tier4 => 16.0,
        }
    }
}
