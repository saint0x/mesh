//! Public `zip` engine boundary inside Mesh.
//!
//! Mesh embeds the `zip` inference engine as its serving runtime. This module
//! is the native boundary that the rest of Mesh should use when interacting
//! with the engine.
//!
//! The internal implementation continues to live under `inference/`, but that
//! directory is intentionally treated as a `zip` implementation detail rather
//! than Mesh's public architectural surface.

pub mod artifact_loader {
    pub use crate::inference::artifact_loader::*;
}

pub mod coordinator {
    pub use crate::inference::coordinator::*;
}

pub mod forward_pass {
    pub use crate::inference::forward_pass::*;
}

pub use crate::inference::{
    ArtifactShardLoader, BackendInstanceSpec, BackendMicrobatchExecutor,
    BackendOptimizationProfile, CandleExecutionBackend, DecodeBatchPlan, DecodeBatchPolicy,
    DecodeBatchSlot, DecodeBatchTargets, DecodeMicrobatchOutput, DecodeMicrobatchRequest,
    DecodeTask, EngineSessionState, ExecutionBackend, ExecutionPhase, ExecutorPhasePlan,
    FastPathBackendContext, FastPathBucketKey, FastPathExecutionPlan, FastPathInvariantError,
    FastPathPlanner, FastPathRuntime, ForwardPass, FusedKernelStage, GenerationConfig,
    GraphCaptureStrategy, HostKernel, InferenceConfig, InferenceCoordinator, InferenceJob,
    InferenceProgressUpdate, InferenceRequest, InferenceResult, InferenceRuntimeMode,
    InferenceStats, KVCache, KVCacheConfig, KvRuntimeContract, KvTransferPolicy, LayerKVCache,
    LayerWeights, LocalExecutorClass, LocalExecutorContract, ModelWeights, PrefillBucketStrategy,
    RuntimeMemoryBudget, SegmentExecutionResult, SessionAssignment, SessionEvictionReason,
    SessionEvictionState, SessionPauseReason, SessionPauseState, SessionRuntimeStatus, ShardLoader,
    Tensor1D, Tensor2D, TransportCapabilityTier, WorkspaceRequirements,
};
