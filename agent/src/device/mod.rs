mod capabilities;
pub mod keypair;

pub use capabilities::{DeviceCapabilities, Tier};

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

    /// Device hardware capabilities
    pub capabilities: DeviceCapabilities,
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
            capabilities,
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
        assert!(config.capabilities.cpu_cores > 0);
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
