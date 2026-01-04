use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use crate::errors::{RelayError, Result};

/// Main configuration for the relay server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub relay: RelayConfig,
    pub network: NetworkConfig,
    pub auth: AuthConfig,
    pub logging: LoggingConfig,
}

/// Relay-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayConfig {
    pub max_reservations: usize,
    pub max_reservations_per_peer: usize,
    pub max_circuits_per_peer: usize,
    pub max_circuit_duration_secs: u64,
    pub max_circuit_bytes: u64,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub tcp_listen_addr: String,
    pub quic_listen_addr: String,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub auth_token: String,
    pub auth_enabled: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub log_to_file: bool,
    pub log_file_path: String,
    pub log_format: String,
}

impl Config {
    /// Get default configuration file path: `~/.meshnet/relay.toml`
    #[allow(dead_code)]
    pub fn default_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| RelayError::Config("Cannot determine home directory".into()))?;
        Ok(home.join(".meshnet").join("relay.toml"))
    }

    /// Load configuration from file
    pub fn load(path: &Path) -> Result<Self> {
        tracing::info!(path = %path.display(), "Loading configuration");

        let content = std::fs::read_to_string(path).map_err(|e| {
            tracing::error!(path = %path.display(), error = %e, "Failed to read config file");
            e
        })?;

        let config: Config = toml::from_str(&content)?;

        config.validate()?;

        tracing::info!("Configuration loaded successfully");
        Ok(config)
    }

    /// Generate default configuration
    pub fn default() -> Self {
        Config {
            relay: RelayConfig {
                max_reservations: 100,
                max_reservations_per_peer: 5,
                max_circuits_per_peer: 16,
                max_circuit_duration_secs: 120,
                max_circuit_bytes: 10485760, // 10MB
            },
            network: NetworkConfig {
                tcp_listen_addr: "/ip4/0.0.0.0/tcp/4001".to_string(),
                quic_listen_addr: "/ip4/0.0.0.0/udp/4001/quic-v1".to_string(),
            },
            auth: AuthConfig {
                auth_token: "CHANGE_ME_IN_PRODUCTION".to_string(),
                auth_enabled: false,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                log_to_file: false,
                log_file_path: "~/.meshnet/logs/relay-server.log".to_string(),
                log_format: "pretty".to_string(),
            },
        }
    }

    /// Validate configuration
    fn validate(&self) -> Result<()> {
        // Validate auth token if auth is enabled
        if self.auth.auth_enabled && self.auth.auth_token == "CHANGE_ME_IN_PRODUCTION" {
            return Err(RelayError::Config(
                "CRITICAL: auth_token is default value. Generate a secure token before enabling auth!".into()
            ));
        }

        // Validate reservation limits
        if self.relay.max_reservations == 0 || self.relay.max_reservations > 10000 {
            return Err(RelayError::Config(
                "max_reservations must be between 1 and 10000".into()
            ));
        }

        if self.relay.max_reservations_per_peer == 0 {
            return Err(RelayError::Config(
                "max_reservations_per_peer must be at least 1".into()
            ));
        }

        if self.relay.max_circuits_per_peer == 0 {
            return Err(RelayError::Config(
                "max_circuits_per_peer must be at least 1".into()
            ));
        }

        // Validate multiaddrs
        self.network.tcp_listen_addr.parse::<libp2p::Multiaddr>()
            .map_err(|e| RelayError::Config(format!("Invalid TCP address: {}", e)))?;

        self.network.quic_listen_addr.parse::<libp2p::Multiaddr>()
            .map_err(|e| RelayError::Config(format!("Invalid QUIC address: {}", e)))?;

        // Validate log level
        match self.logging.level.to_lowercase().as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {},
            _ => return Err(RelayError::Config(
                "log level must be one of: trace, debug, info, warn, error".into()
            )),
        }

        // Validate log format
        match self.logging.log_format.as_str() {
            "pretty" | "json" => {},
            _ => return Err(RelayError::Config(
                "log_format must be 'pretty' or 'json'".into()
            )),
        }

        Ok(())
    }

    /// Save configuration to file (atomic write)
    pub fn save(&self, path: &Path) -> Result<()> {
        tracing::info!(path = %path.display(), "Saving configuration");

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
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
        std::fs::write(&temp_path, &toml_string).map_err(|e| {
            tracing::error!(
                path = %temp_path.display(),
                error = %e,
                "Failed to write temp config file"
            );
            e
        })?;

        std::fs::rename(&temp_path, path).map_err(|e| {
            tracing::error!(
                from = %temp_path.display(),
                to = %path.display(),
                error = %e,
                "Failed to rename temp config file"
            );
            e
        })?;

        tracing::info!(path = %path.display(), "Configuration saved successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = Config::default();

        assert_eq!(config.relay.max_reservations, 100);
        assert_eq!(config.relay.max_reservations_per_peer, 5);
        assert_eq!(config.network.tcp_listen_addr, "/ip4/0.0.0.0/tcp/4001");
        assert!(!config.auth.auth_enabled);
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn test_config_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_auth_token() {
        let mut config = Config::default();
        config.auth.auth_enabled = true;
        // Token is default value, should fail
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_reservation_limits() {
        let mut config = Config::default();
        config.relay.max_reservations = 0;
        assert!(config.validate().is_err());

        config.relay.max_reservations = 100;
        config.relay.max_reservations_per_peer = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_multiaddr() {
        let mut config = Config::default();
        config.network.tcp_listen_addr = "invalid-multiaddr".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("relay.toml");

        // Save default config
        let original = Config::default();
        original.save(&config_path).expect("save should succeed");

        // Verify file exists
        assert!(config_path.exists());

        // Load config
        let loaded = Config::load(&config_path).expect("load should succeed");

        // Verify values match
        assert_eq!(original.relay.max_reservations, loaded.relay.max_reservations);
        assert_eq!(original.network.tcp_listen_addr, loaded.network.tcp_listen_addr);
        assert_eq!(original.logging.level, loaded.logging.level);
    }

    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("relay.toml");

        let config = Config::default();
        config.save(&config_path).unwrap();

        // Verify no .tmp file left behind
        let temp_path = config_path.with_extension("toml.tmp");
        assert!(!temp_path.exists(), "Temp file should be cleaned up");
    }

    #[test]
    fn test_default_path() {
        let path = Config::default_path().unwrap();
        assert!(path.to_string_lossy().contains(".meshnet"));
        assert!(path.to_string_lossy().ends_with("relay.toml"));
    }
}
