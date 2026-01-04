use thiserror::Error;

/// Errors that can occur in the agent.
#[derive(Error, Debug)]
pub enum AgentError {
    /// IO error occurred (file operations, network, etc.)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Cryptography error (keypair generation, signing, etc.)
    #[error("Cryptography error: {0}")]
    Crypto(String),

    /// Configuration error (invalid config, missing fields, etc.)
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network error (connection failed, protocol error, etc.)
    #[error("Network error: {0}")]
    Network(String),
}

/// Result type alias for agent operations.
pub type Result<T> = std::result::Result<T, AgentError>;

// Implement From for TOML serialization errors
impl From<toml::ser::Error> for AgentError {
    fn from(e: toml::ser::Error) -> Self {
        AgentError::Serialization(e.to_string())
    }
}

impl From<toml::de::Error> for AgentError {
    fn from(e: toml::de::Error) -> Self {
        AgentError::Serialization(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AgentError::Config("Invalid device name".to_string());
        assert_eq!(err.to_string(), "Configuration error: Invalid device name");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let agent_err: AgentError = io_err.into();
        assert!(agent_err.to_string().contains("IO error"));
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_result() -> Result<i32> {
            Ok(42)
        }

        assert_eq!(returns_result().unwrap(), 42);
    }
}
