pub mod api;
pub mod checkpoint;
pub mod device;
pub mod discovery;
pub mod errors;
pub mod executor;
pub mod inference;
pub mod model;
pub mod network;
pub mod observability;
pub mod pki;
pub mod resource_manager;
pub mod telemetry;

// Workaround: re-export scheduler for now (stub module)
pub mod scheduler;

pub use api::RegistrationClient;
pub use checkpoint::{Checkpoint, CheckpointConfig, CheckpointManager, CheckpointMetadata};
pub use device::{DeviceCapabilities, DeviceConfig, Tier};
pub use errors::{AgentError, Result};
pub use executor::{
    EmbeddingsExecutor, EmbeddingsInput, EmbeddingsOutput, ExecutorError, JobRunner, JobStats,
    Tensor, WorkerRing,
};
pub use inference::{
    GenerationConfig, InferenceConfig, InferenceCoordinator, InferenceJob, InferenceRequest,
    InferenceResult, InferenceStats,
};
pub use model::{ModelInfo, ShardAssignment, ShardInfo, ShardRegistry, ShardStatus};
pub use network::{
    AllReducePhase, ConnectionInfo, ConnectionType, MeshEvent, MeshSwarm, MeshSwarmBuilder,
    MeshSwarmConfig, RingConnections, TensorMessage,
};
pub use observability::{init_production_logging, init_simple_logging};
pub use resource_manager::{format_bytes, parse_memory_string, ResourceManager};
pub use telemetry::{calculate_credits, LedgerClient, LedgerEvent};
