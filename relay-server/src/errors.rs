use thiserror::Error;

/// Errors that can occur in the relay server
#[derive(Error, Debug)]
pub enum RelayError {
    /// Configuration error occurred
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network error occurred
    #[allow(dead_code)]
    #[error("Network error: {0}")]
    Network(String),

    /// IO error occurred (file operations, network, etc.)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// TOML deserialization error
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    /// TOML serialization error
    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    /// Multiaddr parse error
    #[error("Multiaddr parse error: {0}")]
    MultiaddrParse(#[from] libp2p::multiaddr::Error),

    /// Transport error
    #[error("Transport error: {0}")]
    Transport(String),

    /// Tracing/logging error
    #[error("Tracing error: {0}")]
    Tracing(String),
}

/// Result type alias for relay operations
pub type Result<T> = std::result::Result<T, RelayError>;

// Implement From for tracing errors
impl From<tracing::subscriber::SetGlobalDefaultError> for RelayError {
    fn from(e: tracing::subscriber::SetGlobalDefaultError) -> Self {
        RelayError::Tracing(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RelayError::Config("Invalid port".to_string());
        assert_eq!(err.to_string(), "Configuration error: Invalid port");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let relay_err: RelayError = io_err.into();
        assert!(relay_err.to_string().contains("IO error"));
    }
}
