pub mod api;
pub mod checkpoint;
pub mod connectivity;
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

pub use api::RegistrationClient;
pub use checkpoint::{Checkpoint, CheckpointConfig, CheckpointManager, CheckpointMetadata};
pub use connectivity::{
    build_direct_peer_candidates, build_direct_peer_candidates_from_records,
    load_direct_candidate_seed_addrs, load_direct_candidate_seed_records,
    load_observed_reachability_addrs, load_runtime_connectivity_state,
    persist_observed_reachability_addr, persist_runtime_connectivity_state,
    select_direct_dial_addrs, select_direct_dial_addrs_from_candidates, ConnectivityAttachment,
    ConnectivityAttachmentKind, ConnectivityPath, ConnectivityStatus, DeviceConnectivityState,
    DirectCandidateScope, DirectCandidateSeed, DirectCandidateSource, DirectCandidateTransport,
    DirectPeerCandidate, NetworkConnectivity,
};
pub use device::{DeviceCapabilities, DeviceConfig, Tier};
pub use errors::{AgentError, Result};
pub use executor::{
    AdmissionPolicy, EmbeddingsExecutor, EmbeddingsInput, EmbeddingsOutput, ExecutorError,
    JobRunner, JobStats, Tensor, WorkerRing,
};
pub use inference::{
    GenerationConfig, InferenceConfig, InferenceCoordinator, InferenceJob, InferenceRequest,
    InferenceResult, InferenceStats,
};
pub use model::{ModelInfo, ShardAssignment, ShardInfo, ShardRegistry, ShardStatus};
pub use network::{
    parse_data_plane_endpoint, AllReducePhase, ConnectionInfo, ConnectionType, MeshEvent,
    MeshSwarm, MeshSwarmBuilder, MeshSwarmConfig, RingConnections, TensorMessage, TensorPlane,
    TensorPlaneConfig,
};
pub use observability::{init_production_logging, init_simple_logging};
pub use resource_manager::{format_bytes, parse_memory_string, ResourceManager};
pub use telemetry::{calculate_credits, LedgerClient, LedgerEvent};
