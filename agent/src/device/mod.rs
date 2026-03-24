mod capabilities;
pub mod keypair;

pub use capabilities::{DeviceCapabilities, Tier};

use crate::connectivity::NetworkConnectivity;
use crate::errors::{AgentError, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Device configuration containing identity and network settings.
///
/// This struct is serialized to TOML and saved at `~/.meshnet/device.toml`.
/// The Ed25519 keypair is encoded using multibase Base58BTC format.
///
/// # Examples
///
/// ```no_run
/// use agent::DeviceConfig;
///
/// // Generate new device configuration
/// let config = DeviceConfig::generate(
///     "my-laptop".to_string(),
///     "my-network".to_string(),
///     "https://control.mesh.example.com".to_string(),
/// );
///
/// // Save to default location (~/.meshnet/device.toml)
/// let path = DeviceConfig::default_path().unwrap();
/// config.save(&path).unwrap();
///
/// // Load from file
/// let loaded = DeviceConfig::load(&path).unwrap();
/// assert_eq!(config.device_id, loaded.device_id);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Unique device identifier (UUID v4)
    pub device_id: Uuid,

    /// Human-readable device name
    pub name: String,

    /// Ed25519 signing keypair (multibase Base58BTC encoded)
    #[serde(with = "keypair::keypair_serde")]
    pub keypair: SigningKey,

    /// Network identifier this device belongs to
    pub network_id: String,

    /// Control plane API URL
    pub control_plane_url: String,

    /// Network connectivity profile advertised by the control plane
    pub connectivity: NetworkConnectivity,

    /// Device hardware capabilities
    pub capabilities: DeviceCapabilities,

    /// Runtime governance configuration
    #[serde(default)]
    pub governance: GovernanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GovernanceConfig {
    /// Maximum number of jobs that may execute concurrently on this agent.
    pub max_concurrent_jobs: usize,

    /// Maximum number of admitted jobs allowed to wait in the local scheduler queue.
    pub max_pending_jobs: usize,

    /// Maximum number of concurrent jobs a single peer may hold on this agent.
    pub max_concurrent_jobs_per_peer: usize,

    /// Maximum job timeout this agent will admit for execution.
    pub max_job_timeout_ms: u64,

    /// Workloads this agent will admit in the production runtime path.
    pub allowed_workloads: Vec<String>,

    /// Workload-specific concurrency caps enforced within the local runner.
    pub workload_concurrency_limits: Vec<WorkloadConcurrencyLimit>,

    /// Relative peer weights for deterministic scheduler fairness under contention.
    pub peer_priority_weights: Vec<PeerPriorityWeight>,

    /// Relative workload weights for deterministic scheduler fairness under contention.
    pub workload_priority_weights: Vec<WorkloadPriorityWeight>,

    /// If non-empty, only these peers may submit mesh jobs to this agent.
    pub trusted_peer_ids: Vec<String>,

    /// These peers are always denied mesh job admission.
    pub blocked_peer_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkloadConcurrencyLimit {
    /// Workload identifier this cap applies to.
    pub workload_id: String,

    /// Maximum concurrent jobs of this workload allowed on the agent.
    pub max_concurrent_jobs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerPriorityWeight {
    /// Peer identifier this weight applies to.
    pub peer_id: String,

    /// Relative weight used by the scheduler. Higher means more preferred under contention.
    pub weight: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkloadPriorityWeight {
    /// Workload identifier this weight applies to.
    pub workload_id: String,

    /// Relative weight used by the scheduler. Higher means more preferred under contention.
    pub weight: u32,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 2,
            max_pending_jobs: 8,
            max_concurrent_jobs_per_peer: 1,
            max_job_timeout_ms: 300_000,
            allowed_workloads: vec!["embeddings".to_string(), "embeddings-v1".to_string()],
            workload_concurrency_limits: vec![
                WorkloadConcurrencyLimit {
                    workload_id: "embeddings".to_string(),
                    max_concurrent_jobs: 1,
                },
                WorkloadConcurrencyLimit {
                    workload_id: "embeddings-v1".to_string(),
                    max_concurrent_jobs: 1,
                },
            ],
            peer_priority_weights: Vec::new(),
            workload_priority_weights: vec![
                WorkloadPriorityWeight {
                    workload_id: "embeddings".to_string(),
                    weight: 100,
                },
                WorkloadPriorityWeight {
                    workload_id: "embeddings-v1".to_string(),
                    weight: 100,
                },
            ],
            trusted_peer_ids: Vec::new(),
            blocked_peer_ids: Vec::new(),
        }
    }
}

impl DeviceConfig {
    /// Generate new device configuration with auto-detected capabilities.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable device name
    /// * `network_id` - Network identifier to join
    /// * `control_plane_url` - Control plane API endpoint
    ///
    /// # Examples
    ///
    /// ```
    /// use agent::DeviceConfig;
    ///
    /// let config = DeviceConfig::generate(
    ///     "my-device".to_string(),
    ///     "test-network".to_string(),
    ///     "http://localhost:8080".to_string(),
    /// );
    ///
    /// println!("Device ID: {}", config.device_id);
    /// println!("Device tier: {:?}", config.capabilities.tier);
    /// ```
    pub fn generate(name: String, network_id: String, control_plane_url: String) -> Self {
        let device_id = Uuid::new_v4();
        let keypair = keypair::generate_keypair();
        let capabilities = DeviceCapabilities::detect();

        Self {
            device_id,
            name,
            keypair,
            network_id,
            control_plane_url,
            connectivity: NetworkConnectivity {
                preferred_path: crate::connectivity::ConnectivityPath::Direct,
                attachments: Vec::new(),
            },
            capabilities,
            governance: GovernanceConfig::default(),
        }
    }

    /// Get default configuration file path: `~/.meshnet/device.toml`
    ///
    /// # Errors
    ///
    /// Returns `AgentError::Config` if home directory cannot be determined.
    pub fn default_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| AgentError::Config("Could not determine home directory".to_string()))?;

        Ok(home.join(".meshnet").join("device.toml"))
    }

    /// Save configuration to file.
    ///
    /// Creates parent directories if they don't exist.
    /// Uses atomic write (temp file + rename) to prevent corruption.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save configuration
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Parent directory cannot be created
    /// - Serialization fails
    /// - File write fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use agent::DeviceConfig;
    /// use std::path::Path;
    ///
    /// let config = DeviceConfig::generate(
    ///     "my-device".to_string(),
    ///     "test-network".to_string(),
    ///     "http://localhost:8080".to_string(),
    /// );
    ///
    /// let path = Path::new("/tmp/device.toml");
    /// config.save(path).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                tracing::error!(
                    path = %parent.display(),
                    error = %e,
                    "Failed to create config directory"
                );
                e
            })?;
        }

        // Serialize to TOML
        let toml_string = toml::to_string_pretty(self)?;

        // Atomic write: write to temp file, then rename
        let temp_path = path.with_extension("toml.tmp");
        fs::write(&temp_path, &toml_string).map_err(|e| {
            tracing::error!(
                path = %temp_path.display(),
                error = %e,
                "Failed to write temp config file"
            );
            e
        })?;

        fs::rename(&temp_path, path).map_err(|e| {
            tracing::error!(
                from = %temp_path.display(),
                to = %path.display(),
                error = %e,
                "Failed to rename temp config file"
            );
            e
        })?;

        tracing::info!(path = %path.display(), "Device configuration saved");
        Ok(())
    }

    /// Load configuration from file.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load configuration from
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File doesn't exist
    /// - File cannot be read
    /// - TOML parsing fails
    /// - Keypair deserialization fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use agent::DeviceConfig;
    /// use std::path::Path;
    ///
    /// let path = Path::new("~/.meshnet/device.toml");
    /// let config = DeviceConfig::load(path).unwrap();
    ///
    /// println!("Loaded device: {}", config.name);
    /// ```
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| {
            tracing::error!(
                path = %path.display(),
                error = %e,
                "Failed to read config file"
            );
            e
        })?;

        let config: Self = toml::from_str(&content)?;

        tracing::info!(
            path = %path.display(),
            device_id = %config.device_id,
            "Device configuration loaded"
        );

        Ok(config)
    }

    /// Get public key as multibase-encoded string.
    ///
    /// Useful for displaying or transmitting the public key.
    ///
    /// # Examples
    ///
    /// ```
    /// use agent::DeviceConfig;
    ///
    /// let config = DeviceConfig::generate(
    ///     "my-device".to_string(),
    ///     "test-network".to_string(),
    ///     "http://localhost:8080".to_string(),
    /// );
    ///
    /// let pub_key_str = config.public_key_multibase();
    /// assert!(pub_key_str.starts_with('z'), "Should use Base58BTC encoding");
    /// ```
    pub fn public_key_multibase(&self) -> String {
        multibase::encode(
            multibase::Base::Base58Btc,
            keypair::public_key(&self.keypair).to_bytes(),
        )
    }

    /// Get default certificate file path: `~/.meshnet/device-cert.bin`
    pub fn default_certificate_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| AgentError::Config("Could not determine home directory".to_string()))?;

        let meshnet_dir = home.join(".meshnet");
        if !meshnet_dir.exists() {
            fs::create_dir_all(&meshnet_dir)?;
        }

        Ok(meshnet_dir.join("device-cert.bin"))
    }

    /// Save device certificate to file
    pub fn save_certificate(&self, certificate: &[u8]) -> Result<()> {
        let cert_path = Self::default_certificate_path()?;

        // Atomic write
        let temp_path = cert_path.with_extension("bin.tmp");
        fs::write(&temp_path, certificate)?;
        fs::rename(&temp_path, &cert_path)?;

        tracing::info!(path = %cert_path.display(), size = certificate.len(), "Certificate saved");
        Ok(())
    }

    /// Load device certificate from file
    pub fn load_certificate(&self) -> Result<Vec<u8>> {
        let cert_path = Self::default_certificate_path()?;

        if !cert_path.exists() {
            return Err(AgentError::Config(format!(
                "Certificate file not found: {}",
                cert_path.display()
            )));
        }

        let certificate = fs::read(&cert_path)?;
        tracing::info!(path = %cert_path.display(), size = certificate.len(), "Certificate loaded");
        Ok(certificate)
    }

    /// Check if device certificate exists
    pub fn has_certificate(&self) -> bool {
        Self::default_certificate_path()
            .ok()
            .map(|p| p.exists())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_config() {
        let config = DeviceConfig::generate(
            "test-device".to_string(),
            "test-network".to_string(),
            "http://localhost:8080".to_string(),
        );

        assert_eq!(config.name, "test-device");
        assert_eq!(config.network_id, "test-network");
        assert_eq!(config.control_plane_url, "http://localhost:8080");
        assert!(config.connectivity.attachments.is_empty());
        assert!(config.capabilities.cpu_cores > 0);
        assert_eq!(config.governance.max_concurrent_jobs, 2);
        assert_eq!(config.governance.max_pending_jobs, 8);
        assert_eq!(config.governance.max_concurrent_jobs_per_peer, 1);
        assert_eq!(config.governance.max_job_timeout_ms, 300_000);
        assert_eq!(
            config.governance.allowed_workloads,
            vec!["embeddings".to_string(), "embeddings-v1".to_string()]
        );
        assert_eq!(
            config.governance.workload_concurrency_limits,
            vec![
                WorkloadConcurrencyLimit {
                    workload_id: "embeddings".to_string(),
                    max_concurrent_jobs: 1
                },
                WorkloadConcurrencyLimit {
                    workload_id: "embeddings-v1".to_string(),
                    max_concurrent_jobs: 1
                }
            ]
        );
        assert!(config.governance.peer_priority_weights.is_empty());
        assert_eq!(
            config.governance.workload_priority_weights,
            vec![
                WorkloadPriorityWeight {
                    workload_id: "embeddings".to_string(),
                    weight: 100
                },
                WorkloadPriorityWeight {
                    workload_id: "embeddings-v1".to_string(),
                    weight: 100
                }
            ]
        );
        assert!(config.governance.trusted_peer_ids.is_empty());
        assert!(config.governance.blocked_peer_ids.is_empty());
    }

    #[test]
    fn test_load_legacy_config_defaults_governance() {
        let keypair = keypair::generate_keypair();
        let encoded_keypair = multibase::encode(multibase::Base::Base58Btc, keypair.to_bytes());
        let legacy_toml = format!(
            r#"
device_id = "11111111-1111-1111-1111-111111111111"
name = "legacy-device"
keypair = "{encoded_keypair}"
network_id = "test-network"
control_plane_url = "http://localhost:8080"

[connectivity]
preferred_path = "direct"
attachments = []

[capabilities]
tier = "tier2"
cpu_cores = 8
ram_mb = 16384
gpu_present = false
gpu_vram_mb = 0
os = "linux"
arch = "x86_64"
"#
        );

        let loaded: DeviceConfig = toml::from_str(&legacy_toml).unwrap();
        assert_eq!(loaded.governance.max_concurrent_jobs, 2);
        assert_eq!(loaded.governance.max_pending_jobs, 8);
        assert_eq!(loaded.governance.max_concurrent_jobs_per_peer, 1);
        assert_eq!(loaded.governance.max_job_timeout_ms, 300_000);
        assert_eq!(
            loaded.governance.allowed_workloads,
            vec!["embeddings".to_string(), "embeddings-v1".to_string()]
        );
        assert_eq!(
            loaded.governance.workload_concurrency_limits,
            vec![
                WorkloadConcurrencyLimit {
                    workload_id: "embeddings".to_string(),
                    max_concurrent_jobs: 1
                },
                WorkloadConcurrencyLimit {
                    workload_id: "embeddings-v1".to_string(),
                    max_concurrent_jobs: 1
                }
            ]
        );
        assert!(loaded.governance.peer_priority_weights.is_empty());
        assert_eq!(
            loaded.governance.workload_priority_weights,
            vec![
                WorkloadPriorityWeight {
                    workload_id: "embeddings".to_string(),
                    weight: 100
                },
                WorkloadPriorityWeight {
                    workload_id: "embeddings-v1".to_string(),
                    weight: 100
                }
            ]
        );
        assert!(loaded.governance.trusted_peer_ids.is_empty());
        assert!(loaded.governance.blocked_peer_ids.is_empty());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("device.toml");

        // Generate and save
        let original = DeviceConfig::generate(
            "test-device".to_string(),
            "test-network".to_string(),
            "http://localhost:8080".to_string(),
        );

        original.save(&config_path).expect("save should succeed");

        // Verify file exists
        assert!(config_path.exists(), "Config file should exist");

        // Load
        let loaded = DeviceConfig::load(&config_path).expect("load should succeed");

        // Verify all fields match
        assert_eq!(original.device_id, loaded.device_id);
        assert_eq!(original.name, loaded.name);
        assert_eq!(original.network_id, loaded.network_id);
        assert_eq!(original.control_plane_url, loaded.control_plane_url);
        assert_eq!(original.connectivity, loaded.connectivity);
        assert_eq!(
            original.governance.max_concurrent_jobs,
            loaded.governance.max_concurrent_jobs
        );
        assert_eq!(
            original.governance.max_pending_jobs,
            loaded.governance.max_pending_jobs
        );
        assert_eq!(
            original.governance.max_job_timeout_ms,
            loaded.governance.max_job_timeout_ms
        );
        assert_eq!(
            original.governance.max_concurrent_jobs_per_peer,
            loaded.governance.max_concurrent_jobs_per_peer
        );
        assert_eq!(
            original.governance.allowed_workloads,
            loaded.governance.allowed_workloads
        );
        assert_eq!(
            original.governance.workload_concurrency_limits,
            loaded.governance.workload_concurrency_limits
        );
        assert_eq!(
            original.governance.peer_priority_weights,
            loaded.governance.peer_priority_weights
        );
        assert_eq!(
            original.governance.workload_priority_weights,
            loaded.governance.workload_priority_weights
        );
        assert_eq!(
            original.governance.trusted_peer_ids,
            loaded.governance.trusted_peer_ids
        );
        assert_eq!(
            original.governance.blocked_peer_ids,
            loaded.governance.blocked_peer_ids
        );

        // CRITICAL: Verify keypair bytes are identical
        assert_eq!(
            original.keypair.to_bytes(),
            loaded.keypair.to_bytes(),
            "Keypair bytes must be identical after save/load"
        );

        // Verify capabilities
        assert_eq!(
            original.capabilities.cpu_cores,
            loaded.capabilities.cpu_cores
        );
        assert_eq!(original.capabilities.ram_mb, loaded.capabilities.ram_mb);
    }

    #[test]
    fn test_default_path() {
        let path = DeviceConfig::default_path().unwrap();
        assert!(path.to_string_lossy().contains(".meshnet"));
        assert!(path.to_string_lossy().ends_with("device.toml"));
    }

    #[test]
    fn test_public_key_multibase() {
        let config = DeviceConfig::generate(
            "test".to_string(),
            "test".to_string(),
            "http://localhost:8080".to_string(),
        );

        let pub_key = config.public_key_multibase();

        // Verify format
        assert!(pub_key.starts_with('z'), "Should use Base58BTC encoding");

        // Verify can be decoded
        let (base, decoded) = multibase::decode(&pub_key).unwrap();
        assert_eq!(base, multibase::Base::Base58Btc);
        assert_eq!(decoded.len(), 32, "Ed25519 public key is 32 bytes");
    }

    #[test]
    fn test_save_creates_parent_directory() {
        let temp_dir = TempDir::new().unwrap();
        let nested_path = temp_dir
            .path()
            .join("nested")
            .join("dir")
            .join("device.toml");

        let config = DeviceConfig::generate(
            "test".to_string(),
            "test".to_string(),
            "http://localhost:8080".to_string(),
        );

        config
            .save(&nested_path)
            .expect("save should create parent dirs");
        assert!(nested_path.exists());
    }

    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("device.toml");

        let config = DeviceConfig::generate(
            "test".to_string(),
            "test".to_string(),
            "http://localhost:8080".to_string(),
        );

        // Save
        config.save(&config_path).unwrap();

        // Verify no .tmp file left behind
        let temp_path = config_path.with_extension("toml.tmp");
        assert!(!temp_path.exists(), "Temp file should be cleaned up");
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = DeviceConfig::load(Path::new("/nonexistent/path/device.toml"));
        assert!(result.is_err(), "Loading nonexistent file should fail");
    }

    #[test]
    fn test_keypair_uniqueness() {
        let config1 = DeviceConfig::generate(
            "device1".to_string(),
            "test".to_string(),
            "http://localhost:8080".to_string(),
        );

        let config2 = DeviceConfig::generate(
            "device2".to_string(),
            "test".to_string(),
            "http://localhost:8080".to_string(),
        );

        // Device IDs should be different
        assert_ne!(config1.device_id, config2.device_id);

        // Keypairs should be different
        assert_ne!(config1.keypair.to_bytes(), config2.keypair.to_bytes());
    }
}
