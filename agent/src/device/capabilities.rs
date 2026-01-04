use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Device tier classification based on hardware specs.
///
/// Tiers determine credit multipliers for job execution:
/// - Tier0: 1.0x (basic devices)
/// - Tier1: 2.0x (entry-level compute)
/// - Tier2: 4.0x (mid-range compute)
/// - Tier3: 8.0x (high-end compute)
/// - Tier4: 16.0x (server-grade)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Tier0,
    Tier1,
    Tier2,
    Tier3,
    Tier4,
}

impl Tier {
    /// Classify device tier based on CPU cores and RAM.
    ///
    /// Classification logic:
    /// - Tier0: <4 cores OR <4GB RAM
    /// - Tier1: 4-7 cores AND 4-7GB RAM
    /// - Tier2: 8-15 cores AND 8-15GB RAM
    /// - Tier3: 16-31 cores AND 16-31GB RAM
    /// - Tier4: 32+ cores AND 32+ GB RAM
    pub fn from_specs(cpu_cores: usize, ram_gb: usize) -> Self {
        // Must meet BOTH CPU and RAM requirements for a tier
        match (cpu_cores, ram_gb) {
            (cores, ram) if cores >= 32 && ram >= 32 => Tier::Tier4,
            (cores, ram) if cores >= 16 && ram >= 16 => Tier::Tier3,
            (cores, ram) if cores >= 8 && ram >= 8 => Tier::Tier2,
            (cores, ram) if cores >= 4 && ram >= 4 => Tier::Tier1,
            _ => Tier::Tier0,
        }
    }

    /// Get the credit multiplier for this tier.
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

/// Device hardware capabilities.
///
/// Detected at runtime using the `sysinfo` crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Device tier classification
    pub tier: Tier,

    /// Number of CPU cores
    pub cpu_cores: usize,

    /// Total RAM in megabytes
    pub ram_mb: usize,

    /// Whether GPU is present (basic detection for MVP)
    pub gpu_present: bool,

    /// GPU VRAM in megabytes (if detectable)
    pub gpu_vram_mb: Option<usize>,

    /// Operating system
    pub os: String,

    /// CPU architecture
    pub arch: String,
}

impl DeviceCapabilities {
    /// Detect current device capabilities.
    ///
    /// This function queries the system for hardware information and
    /// classifies the device into a tier for credit calculation.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_agent::DeviceCapabilities;
    ///
    /// let caps = DeviceCapabilities::detect();
    /// println!("Device tier: {:?}", caps.tier);
    /// println!("Credit multiplier: {}", caps.tier.credit_multiplier());
    /// ```
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_cores = sys.cpus().len();
        let ram_bytes = sys.total_memory();
        let ram_mb = (ram_bytes / 1_048_576) as usize; // bytes to MB
        let ram_gb = ram_mb / 1024;

        let tier = Tier::from_specs(cpu_cores, ram_gb);

        // Basic GPU detection (enhanced detection deferred to Phase 1)
        let gpu_present = Self::detect_gpu();
        let gpu_vram_mb = None; // Will implement in Phase 1

        let os = Self::detect_os();
        let arch = Self::detect_arch();

        Self {
            tier,
            cpu_cores,
            ram_mb,
            gpu_present,
            gpu_vram_mb,
            os,
            arch,
        }
    }

    /// Detect if GPU is present (basic detection).
    ///
    /// For MVP, this is a simple heuristic. Will be enhanced in Phase 1.
    fn detect_gpu() -> bool {
        // Basic heuristic: assume GPU present on macOS (Metal)
        // and most modern desktop systems with >8GB RAM
        #[cfg(target_os = "macos")]
        {
            true
        }

        #[cfg(not(target_os = "macos"))]
        {
            let mut sys = System::new_all();
            sys.refresh_all();
            let ram_gb = (sys.total_memory() / 1_073_741_824) as usize;
            ram_gb > 8 // Heuristic: systems with >8GB likely have discrete GPU
        }
    }

    /// Detect operating system.
    fn detect_os() -> String {
        std::env::consts::OS.to_string()
    }

    /// Detect CPU architecture.
    fn detect_arch() -> String {
        std::env::consts::ARCH.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_classification() {
        // Tier 0 - insufficient CPU
        assert_eq!(Tier::from_specs(2, 8), Tier::Tier0);

        // Tier 0 - insufficient RAM
        assert_eq!(Tier::from_specs(8, 2), Tier::Tier0);

        // Tier 1
        assert_eq!(Tier::from_specs(4, 4), Tier::Tier1);
        assert_eq!(Tier::from_specs(6, 6), Tier::Tier1);

        // Tier 2
        assert_eq!(Tier::from_specs(8, 8), Tier::Tier2);
        assert_eq!(Tier::from_specs(12, 12), Tier::Tier2);

        // Tier 3
        assert_eq!(Tier::from_specs(16, 16), Tier::Tier3);
        assert_eq!(Tier::from_specs(24, 24), Tier::Tier3);

        // Tier 4
        assert_eq!(Tier::from_specs(32, 32), Tier::Tier4);
        assert_eq!(Tier::from_specs(64, 64), Tier::Tier4);
    }

    #[test]
    fn test_credit_multipliers() {
        assert_eq!(Tier::Tier0.credit_multiplier(), 1.0);
        assert_eq!(Tier::Tier1.credit_multiplier(), 2.0);
        assert_eq!(Tier::Tier2.credit_multiplier(), 4.0);
        assert_eq!(Tier::Tier3.credit_multiplier(), 8.0);
        assert_eq!(Tier::Tier4.credit_multiplier(), 16.0);
    }

    #[test]
    fn test_capabilities_detection() {
        let caps = DeviceCapabilities::detect();

        // Verify basic sanity checks
        assert!(caps.cpu_cores > 0, "CPU cores should be detected");
        assert!(caps.ram_mb > 0, "RAM should be detected");
        assert!(!caps.os.is_empty(), "OS should be detected");
        assert!(!caps.arch.is_empty(), "Architecture should be detected");

        // Verify tier is within valid range
        assert!(matches!(
            caps.tier,
            Tier::Tier0 | Tier::Tier1 | Tier::Tier2 | Tier::Tier3 | Tier::Tier4
        ));
    }

    #[test]
    fn test_tier_serialization() {
        let tier = Tier::Tier2;
        let json = serde_json::to_string(&tier).unwrap();
        assert_eq!(json, "\"tier2\"");

        let deserialized: Tier = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, tier);
    }

    #[test]
    fn test_capabilities_serialization() {
        let caps = DeviceCapabilities {
            tier: Tier::Tier2,
            cpu_cores: 8,
            ram_mb: 16384,
            gpu_present: true,
            gpu_vram_mb: Some(4096),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
        };

        let json = serde_json::to_string(&caps).unwrap();
        let deserialized: DeviceCapabilities = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.tier, caps.tier);
        assert_eq!(deserialized.cpu_cores, caps.cpu_cores);
        assert_eq!(deserialized.ram_mb, caps.ram_mb);
    }
}
