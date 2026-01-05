pub mod api;
pub mod device;
pub mod errors;
pub mod executor;
pub mod network;
pub mod observability;
pub mod telemetry;

pub use api::RegistrationClient;
pub use device::{DeviceCapabilities, DeviceConfig, Tier};
pub use errors::{AgentError, Result};
pub use executor::{
    EmbeddingsExecutor, EmbeddingsInput, EmbeddingsOutput, ExecutorError, JobRunner, JobStats,
};
pub use network::{
    ConnectionInfo, ConnectionType, MeshEvent, MeshSwarm, MeshSwarmBuilder, MeshSwarmConfig,
};
pub use observability::{init_production_logging, init_simple_logging};
pub use telemetry::{calculate_credits, LedgerClient, LedgerEvent};
