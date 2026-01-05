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

    /// HTTP client error
    #[error("HTTP error: {0}")]
    Http(String),

    /// Device registration error
    #[error("Registration error: {0}")]
    Registration(String),

    /// Job execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Resource management error
    #[error("Resource error: {0}")]
    Resource(String),

    /// Cooldown is active, unlock not allowed
    #[error("Cooldown active: {remaining_hours} hours remaining until unlock")]
    CooldownActive { remaining_hours: u64 },
}

/// Result type alias for agent operations.
pub type Result<T> = std::result::Result<T, AgentError>;

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context message to error
    fn context(self, msg: &str) -> Result<T>;

    /// Add context using a closure (for lazy evaluation)
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<AgentError>,
{
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| {
            let base: AgentError = e.into();
            tracing::error!("{}: {:?}", msg, base);
            base
        })
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let base: AgentError = e.into();
            let msg = f();
            tracing::error!("{}: {:?}", msg, base);
            base
        })
    }
}

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

// Implement From for JSON serialization errors
impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::Serialization(e.to_string())
    }
}

// Implement From for libp2p noise errors
impl From<libp2p::noise::Error> for AgentError {
    fn from(e: libp2p::noise::Error) -> Self {
        AgentError::Network(format!("Noise encryption error: {}", e))
    }
}

// Implement From for libp2p transport errors
impl From<libp2p::TransportError<std::io::Error>> for AgentError {
    fn from(e: libp2p::TransportError<std::io::Error>) -> Self {
        AgentError::Network(format!("Transport error: {}", e))
    }
}

// Implement From for libp2p dial errors
impl From<libp2p::swarm::DialError> for AgentError {
    fn from(e: libp2p::swarm::DialError) -> Self {
        AgentError::Network(format!("Dial error: {}", e))
    }
}

// Implement From for libp2p listen errors
impl From<libp2p::swarm::ListenError> for AgentError {
    fn from(e: libp2p::swarm::ListenError) -> Self {
        AgentError::Network(format!("Listen error: {}", e))
    }
}

// Implement From for executor errors
impl From<crate::executor::ExecutorError> for AgentError {
    fn from(e: crate::executor::ExecutorError) -> Self {
        AgentError::Execution(e.to_string())
    }
}

/// Pretty error display module for CLI
pub mod display {
    use super::AgentError;
    use colored::Colorize;

    /// Print error with colors and actionable suggestions
    pub fn print_error(err: &AgentError) {
        eprintln!("{} {}", "Error:".red().bold(), err);

        // Provide actionable suggestions based on error type
        match err {
            AgentError::Network(_) => {
                eprintln!("{}", "  → Check that the relay server is running".yellow());
                eprintln!("{}", "  → Verify network connectivity".yellow());
            }
            AgentError::Config(_) => {
                eprintln!(
                    "{}",
                    "  → Run 'mesh init' to generate a valid configuration".yellow()
                );
                eprintln!(
                    "{}",
                    "  → Check config file at ~/.meshnet/device.toml".yellow()
                );
            }
            AgentError::Registration(_) => {
                eprintln!(
                    "{}",
                    "  → Ensure control plane is running and accessible".yellow()
                );
                eprintln!(
                    "{}",
                    "  → Check the control plane URL in your config".yellow()
                );
            }
            AgentError::Execution(_) => {
                eprintln!("{}", "  → Check that the job payload is valid".yellow());
                eprintln!("{}", "  → Ensure the target peer is online".yellow());
            }
            AgentError::Http(_) => {
                eprintln!("{}", "  → Check network connectivity".yellow());
                eprintln!("{}", "  → Verify the control plane URL is correct".yellow());
            }
            AgentError::Resource(_) => {
                eprintln!("{}", "  → Check system memory availability".yellow());
                eprintln!("{}", "  → Ensure sufficient privileges for memory locking".yellow());
            }
            AgentError::CooldownActive { remaining_hours } => {
                eprintln!("{}", format!("  → Unlock will be available in {} hours", remaining_hours).yellow());
                eprintln!("{}", "  → Use 'mesh resource-status' to check lock status".yellow());
            }
            _ => {}
        }
    }

    /// Print error with full context chain (for verbose mode)
    pub fn print_error_verbose(err: &AgentError) {
        eprintln!("{} {}", "Error:".red().bold(), err);
        eprintln!("{}", "Context chain:".yellow());
        eprintln!("  {:#?}", err);
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
