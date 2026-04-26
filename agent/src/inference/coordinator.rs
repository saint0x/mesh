//! Inference coordinator for tensor-parallel distributed inference
//!
//! The InferenceCoordinator manages the lifecycle of inference jobs across
//! a ring of workers. It handles:
//!
//! - Receiving inference requests from the control plane
//! - Coordinating tensor-parallel forward passes across workers
//! - Managing ring all-reduce operations for each layer
//! - Checkpointing for fault tolerance
//! - Returning results to the control plane

use crate::api::types::PeerPunchPlan;
use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::WorkerRing;
use crate::model::registry::ShardRegistry;
use crate::model::shard::ShardAssignment;
use crate::network::{MeshSwarm, TensorPlane};
use libp2p::PeerId;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::artifact_loader::{ArtifactShardLoader, ShardLoader};
use super::backend::{
    BackendMicrobatchExecutor, CandleExecutionBackend, DecodeMicrobatchRequest, ExecutionBackend,
};
use super::engine::{
    DecodeBatchPlan, DecodeBatchPolicy, DecodeBatchSlot, DecodeTask, EngineSessionState,
    ExecutionPhase, InferenceRuntimeMode, RuntimeMemoryBudget, SessionEvictionReason,
    SessionEvictionState, SessionPauseReason, SessionPauseState, SessionRuntimeStatus,
    TransportCapabilityTier,
};
use super::forward_pass::ModelWeights;
use super::job::{
    DecodeBatchTargets, InferenceJob, InferenceProgressUpdate, InferenceRequest, InferenceResult,
    SegmentExecutionResult,
};
use super::stats::InferenceStats;

/// Configuration for the inference coordinator
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Total number of layers in the model
    pub total_layers: u32,

    /// Model identifier
    pub model_id: String,

    /// Timeout for ring all-reduce operations
    pub allreduce_timeout: Duration,

    /// Timeout for entire inference job
    pub job_timeout: Duration,

    /// Whether checkpointing is enabled
    pub checkpointing_enabled: bool,

    /// Directory for checkpoint storage
    pub checkpoint_dir: std::path::PathBuf,

    /// Maximum checkpoint recovery attempts allowed for a single job.
    pub recovery_max_attempts_per_job: u32,

    /// Minimum cooldown between checkpoint recovery attempts for the same job.
    pub recovery_cooldown: Duration,

    /// Maximum checkpoint loads allowed across the node in a rolling minute.
    pub recovery_max_checkpoint_loads_per_minute: u32,

    /// Maximum number of decode sessions admitted into a single microbatch.
    pub max_decode_batch_size: usize,

    /// Maximum aggregate KV-token footprint admitted into a single microbatch.
    pub max_decode_batch_kv_tokens: usize,

    /// Maximum number of active resident sessions before runtime eviction engages.
    pub max_active_sessions: usize,

    /// Maximum aggregate KV cache bytes allowed across active sessions.
    pub max_total_kv_cache_bytes: usize,

    /// Maximum aggregate runtime memory footprint allowed across active sessions.
    pub max_total_runtime_bytes: usize,

    /// How long an idle session may stay resident before eviction is allowed.
    pub idle_session_evict_after: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            total_layers: 70, // Default for Llama-70B
            model_id: "llama-70b".to_string(),
            allreduce_timeout: Duration::from_secs(30),
            job_timeout: Duration::from_secs(300), // 5 minutes
            checkpointing_enabled: true,
            checkpoint_dir: dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".meshnet")
                .join("checkpoints"),
            recovery_max_attempts_per_job: 2,
            recovery_cooldown: Duration::from_secs(5),
            recovery_max_checkpoint_loads_per_minute: 8,
            max_decode_batch_size: 4,
            max_decode_batch_kv_tokens: 16_384,
            max_active_sessions: 8,
            max_total_kv_cache_bytes: 512 * 1024 * 1024,
            max_total_runtime_bytes: 2 * 1024 * 1024 * 1024,
            idle_session_evict_after: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
enum RecoveryAllowance {
    Allowed,
    Cooldown,
    LoadBudget,
}

#[derive(Debug, Default)]
#[cfg_attr(not(test), allow(dead_code))]
struct RecoveryGovernor {
    last_attempts_by_job: HashMap<Uuid, Instant>,
    checkpoint_loads: VecDeque<Instant>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl RecoveryGovernor {
    fn allow_attempt(
        &mut self,
        job_id: Uuid,
        cooldown: Duration,
        max_checkpoint_loads_per_minute: u32,
        now: Instant,
    ) -> RecoveryAllowance {
        while let Some(oldest) = self.checkpoint_loads.front().copied() {
            if now.duration_since(oldest) < Duration::from_secs(60) {
                break;
            }
            self.checkpoint_loads.pop_front();
        }

        if let Some(last_attempt) = self.last_attempts_by_job.get(&job_id).copied() {
            if now.duration_since(last_attempt) < cooldown {
                return RecoveryAllowance::Cooldown;
            }
        }

        if self.checkpoint_loads.len() >= max_checkpoint_loads_per_minute as usize {
            return RecoveryAllowance::LoadBudget;
        }

        self.last_attempts_by_job.insert(job_id, now);
        self.checkpoint_loads.push_back(now);
        RecoveryAllowance::Allowed
    }
}

/// Ring position and shard information for this worker
#[derive(Debug, Clone)]
pub struct WorkerPosition {
    /// This worker's position in the ring (0-indexed)
    pub position: u32,

    /// Total number of workers in the ring
    pub total_workers: u32,

    /// Left neighbor peer ID
    pub left_neighbor: PeerId,

    /// Left neighbor control-plane listen addresses
    pub left_neighbor_addrs: Vec<libp2p::Multiaddr>,

    /// Explicit punched-path coordination plan for the left neighbor
    pub left_neighbor_punch_plan: Option<PeerPunchPlan>,

    /// Left neighbor dedicated tensor endpoint
    pub left_neighbor_tensor_addr: SocketAddr,

    /// Right neighbor peer ID
    pub right_neighbor: PeerId,

    /// Right neighbor control-plane listen addresses
    pub right_neighbor_addrs: Vec<libp2p::Multiaddr>,

    /// Explicit punched-path coordination plan for the right neighbor
    pub right_neighbor_punch_plan: Option<PeerPunchPlan>,

    /// Right neighbor dedicated tensor endpoint
    pub right_neighbor_tensor_addr: SocketAddr,

    /// Column range this worker is responsible for
    pub shard_column_range: (u32, u32),

    /// Model shard memory usage in bytes
    pub shard_memory_bytes: u64,
}

/// The main inference coordinator
pub struct InferenceCoordinator {
    /// Network swarm for P2P communication
    swarm: MeshSwarm,

    /// Dedicated tensor data plane for the hot inference path.
    tensor_plane: TensorPlane,

    /// Inference configuration
    config: InferenceConfig,

    /// This worker's position in the ring
    position: Option<WorkerPosition>,

    /// Inference statistics
    stats: Arc<InferenceStats>,

    /// Currently active inference job (if any)
    active_job: RwLock<Option<InferenceJob>>,

    /// Checkpoint manager (optional)
    checkpoint_manager: Option<Arc<crate::checkpoint::CheckpointManager>>,

    /// Shard registry for tracking model shard lifecycle
    shard_registry: Arc<ShardRegistry>,

    /// Production shard loader
    loader: Arc<dyn ShardLoader>,

    /// Cached weights per model (model_id -> weights)
    weight_cache: RwLock<HashMap<String, Arc<ModelWeights>>>,

    /// Engine-owned session runtime state.
    sessions: HashMap<Uuid, ActiveSession>,

    /// Fair decode-work queue populated by session owners and consumed by the runtime.
    decode_queue: VecDeque<DecodeTask>,

    /// Monotonic counter used to preserve queue fairness across re-enqueues.
    next_decode_fairness_epoch: u64,

    /// Governs checkpoint recovery cadence and node-level recovery load.
    #[allow(dead_code)]
    recovery_governor: RecoveryGovernor,
}

struct ActiveSession {
    engine_state: EngineSessionState,
    backend: Box<dyn ExecutionBackend>,
    job: InferenceJob,
    queued_for_decode: bool,
    decode_steps_served: u64,
    last_active_at: Instant,
}

struct DecodeStepOutcome {
    session_id: Uuid,
    completion_tokens: u32,
    execution_time_ms: u64,
    kv_cache_seq_len: u32,
    should_checkpoint: bool,
    completed: bool,
    time_to_first_token_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct DecodeBatchTelemetry {
    kv_cache_seq_len: u32,
    batch_size: u32,
    active_decode_sessions: u32,
    batch_kv_tokens: u32,
    deferred_decode_sessions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SessionMemoryFootprint {
    runtime_bytes: usize,
    kv_cache_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeMemorySnapshot {
    active_sessions: usize,
    total_runtime_bytes: usize,
    total_kv_cache_bytes: usize,
}

impl InferenceCoordinator {
    /// Create a new inference coordinator
    pub fn new(swarm: MeshSwarm, tensor_plane: TensorPlane, config: InferenceConfig) -> Self {
        // Initialize shard registry with default path
        let shard_registry =
            Arc::new(ShardRegistry::with_defaults().expect("Failed to create shard registry"));

        let loader: Arc<dyn ShardLoader> = Arc::new(ArtifactShardLoader::with_defaults());

        Self {
            swarm,
            tensor_plane,
            config,
            position: None,
            stats: Arc::new(InferenceStats::new()),
            active_job: RwLock::new(None),
            checkpoint_manager: None,
            shard_registry,
            loader,
            weight_cache: RwLock::new(HashMap::new()),
            sessions: HashMap::new(),
            decode_queue: VecDeque::new(),
            next_decode_fairness_epoch: 0,
            recovery_governor: RecoveryGovernor::default(),
        }
    }

    fn build_session_state(
        &self,
        request: &InferenceRequest,
        position: &WorkerPosition,
        provider: crate::provider::ExecutionProviderKind,
    ) -> EngineSessionState {
        EngineSessionState::new(
            request.session_id,
            ExecutionPhase::Prefill,
            request.executor_id.clone(),
            vec![request.executor_id.clone()],
            vec![request.executor_id.clone()],
            request.runtime_mode,
            provider,
            position.shard_column_range,
            TransportCapabilityTier::DirectTcp,
            self.runtime_memory_budget(),
            request.config.max_tokens,
            request.prompt_tokens.len(),
        )
    }

    fn runtime_memory_budget(&self) -> RuntimeMemoryBudget {
        RuntimeMemoryBudget {
            max_active_sessions: self.config.max_active_sessions.max(1),
            max_total_kv_cache_bytes: self.config.max_total_kv_cache_bytes.max(1),
            max_total_runtime_bytes: self.config.max_total_runtime_bytes.max(1),
        }
    }

    fn session_memory_footprint(session: &ActiveSession) -> SessionMemoryFootprint {
        let prompt_bytes = session.job.request.prompt_tokens.len() * std::mem::size_of::<u32>();
        let generated_bytes = session.job.generated_tokens.len() * std::mem::size_of::<u32>();
        let kv_cache_bytes = session.backend.cache_memory_usage();
        SessionMemoryFootprint {
            runtime_bytes: prompt_bytes
                .saturating_add(generated_bytes)
                .saturating_add(kv_cache_bytes),
            kv_cache_bytes,
        }
    }

    fn runtime_memory_snapshot(&self) -> RuntimeMemorySnapshot {
        let mut active_sessions = 0_usize;
        let mut total_runtime_bytes = 0_usize;
        let mut total_kv_cache_bytes = 0_usize;
        for session in self.sessions.values() {
            if matches!(
                session.engine_state.runtime_status,
                SessionRuntimeStatus::Evicted
            ) {
                continue;
            }
            active_sessions = active_sessions.saturating_add(1);
            let footprint = Self::session_memory_footprint(session);
            total_runtime_bytes = total_runtime_bytes.saturating_add(footprint.runtime_bytes);
            total_kv_cache_bytes = total_kv_cache_bytes.saturating_add(footprint.kv_cache_bytes);
        }
        RuntimeMemorySnapshot {
            active_sessions,
            total_runtime_bytes,
            total_kv_cache_bytes,
        }
    }

    async fn ensure_session_backend<'a>(
        &'a mut self,
        request: &InferenceRequest,
        position: &WorkerPosition,
    ) -> Result<&'a mut ActiveSession> {
        if !self.sessions.contains_key(&request.session_id) {
            let weights = self
                .get_or_load_weights(&request.model_id, position)
                .await?;
            let backend = CandleExecutionBackend::new(
                (*weights).clone(),
                position.position,
                position.shard_column_range.0 as usize,
                position.shard_column_range.1 as usize,
                position.total_workers,
                self.config.allreduce_timeout,
            )?;
            let provider = backend.provider_kind();
            let state = self.build_session_state(request, position, provider);
            self.sessions.insert(
                request.session_id,
                ActiveSession {
                    engine_state: state,
                    backend: Box::new(backend),
                    job: InferenceJob::new(request.clone(), self.config.total_layers),
                    queued_for_decode: false,
                    decode_steps_served: 0,
                    last_active_at: Instant::now(),
                },
            );
            self.enforce_runtime_memory_budget(Some(request.session_id))
                .await?;
        }

        self.sessions.get_mut(&request.session_id).ok_or_else(|| {
            AgentError::Execution("session backend vanished during initialization".to_string())
        })
    }

    /// Set the checkpoint manager
    pub fn with_checkpoint_manager(
        mut self,
        manager: Arc<crate::checkpoint::CheckpointManager>,
    ) -> Self {
        self.checkpoint_manager = Some(manager);
        self
    }

    /// Get reference to statistics
    pub fn stats(&self) -> &Arc<InferenceStats> {
        &self.stats
    }

    pub fn has_session(&self, session_id: Uuid) -> bool {
        self.sessions.contains_key(&session_id)
    }

    pub fn pause_local_session(
        &mut self,
        session_id: Uuid,
        reason: SessionPauseReason,
        detail: String,
    ) {
        self.pause_session(session_id, reason, detail);
    }

    fn decode_batch_policy(&self) -> DecodeBatchPolicy {
        DecodeBatchPolicy {
            max_batch_size: self.config.max_decode_batch_size.max(1),
            max_total_kv_tokens: self.config.max_decode_batch_kv_tokens.max(1),
        }
    }

    fn session_decode_kv_tokens(session: &ActiveSession) -> usize {
        session
            .backend
            .cache_seq_len()
            .max(session.backend.sequence_position())
            .max(session.job.request.prompt_tokens.len())
    }

    fn enqueue_decode_task(&mut self, session_id: Uuid) -> Result<()> {
        let session = self.sessions.get_mut(&session_id).ok_or_else(|| {
            AgentError::Execution(format!(
                "Session {} is not active for decode scheduling",
                session_id
            ))
        })?;

        if session.queued_for_decode || session.job.is_complete() {
            return Ok(());
        }

        if !session.job.has_decode_context() {
            return Err(AgentError::Execution(format!(
                "Session {} cannot enter decode scheduling before prefill samples a token",
                session_id
            )));
        }

        self.next_decode_fairness_epoch = self.next_decode_fairness_epoch.saturating_add(1);
        session.queued_for_decode = true;
        self.decode_queue.push_back(DecodeTask {
            session_id,
            fairness_epoch: self.next_decode_fairness_epoch,
        });
        Ok(())
    }

    fn pause_session(&mut self, session_id: Uuid, reason: SessionPauseReason, detail: String) {
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.queued_for_decode = false;
            session.engine_state.runtime_status = SessionRuntimeStatus::Paused;
            session.engine_state.paused = Some(SessionPauseState { reason, detail });
        }
        self.decode_queue
            .retain(|task| task.session_id != session_id);
    }

    fn resume_session(&mut self, session_id: Uuid) -> Result<()> {
        let session = self.sessions.get_mut(&session_id).ok_or_else(|| {
            AgentError::Execution(format!("Session {} is not present for resume", session_id))
        })?;
        session.engine_state.runtime_status = SessionRuntimeStatus::Active;
        session.engine_state.paused = None;
        session.engine_state.evicted = None;
        session.last_active_at = Instant::now();
        Ok(())
    }

    async fn evict_session(
        &mut self,
        session_id: Uuid,
        reason: SessionEvictionReason,
        detail: String,
    ) -> Result<()> {
        if let Some(session) = self.sessions.get(&session_id) {
            if self.config.checkpointing_enabled
                && !session.job.is_complete()
                && self.checkpoint_manager.is_some()
            {
                let snapshot = session.job.clone();
                self.checkpoint(&snapshot).await?;
            }
        }

        if let Some(mut session) = self.sessions.remove(&session_id) {
            session.queued_for_decode = false;
            session.engine_state.runtime_status = SessionRuntimeStatus::Evicted;
            session.engine_state.evicted = Some(SessionEvictionState { reason, detail });
        }
        self.decode_queue
            .retain(|task| task.session_id != session_id);
        Ok(())
    }

    async fn enforce_runtime_memory_budget(
        &mut self,
        protected_session_id: Option<Uuid>,
    ) -> Result<()> {
        loop {
            let snapshot = self.runtime_memory_snapshot();
            let over_active_sessions = snapshot.active_sessions > self.config.max_active_sessions;
            let over_kv_budget =
                snapshot.total_kv_cache_bytes > self.config.max_total_kv_cache_bytes;
            let over_runtime_budget =
                snapshot.total_runtime_bytes > self.config.max_total_runtime_bytes;
            if !over_active_sessions && !over_kv_budget && !over_runtime_budget {
                return Ok(());
            }

            let eviction_candidate = self.select_eviction_candidate(protected_session_id);
            let Some(session_id) = eviction_candidate else {
                return Err(AgentError::Execution(
                    "runtime memory budget exceeded but no evictable session was available"
                        .to_string(),
                ));
            };

            let detail = format!(
                "active_sessions={} runtime_bytes={} kv_bytes={}",
                snapshot.active_sessions,
                snapshot.total_runtime_bytes,
                snapshot.total_kv_cache_bytes
            );
            self.evict_session(
                session_id,
                SessionEvictionReason::RuntimeMemoryPressure,
                detail,
            )
            .await?;
        }
    }

    fn select_eviction_candidate(&self, protected_session_id: Option<Uuid>) -> Option<Uuid> {
        let mut candidates = self
            .sessions
            .iter()
            .filter(|(session_id, _)| Some(**session_id) != protected_session_id)
            .map(|(session_id, session)| {
                let footprint = Self::session_memory_footprint(session);
                let idle_for = session.last_active_at.elapsed();
                let priority = match session.engine_state.runtime_status {
                    SessionRuntimeStatus::Paused => 0_u8,
                    SessionRuntimeStatus::Active if session.job.is_complete() => 1_u8,
                    SessionRuntimeStatus::Active => 2_u8,
                    SessionRuntimeStatus::Evicted => 3_u8,
                };
                let mode_bias = match session.job.request.runtime_mode {
                    InferenceRuntimeMode::FitFirst => footprint.runtime_bytes,
                    InferenceRuntimeMode::ThroughputFirst => usize::MAX.saturating_sub(
                        (session.decode_steps_served as usize).saturating_mul(1024),
                    ),
                    InferenceRuntimeMode::LatencyFirst => footprint.kv_cache_bytes,
                    InferenceRuntimeMode::ResilientEdge => {
                        footprint.kv_cache_bytes / 2
                            + if matches!(
                                session.engine_state.runtime_status,
                                SessionRuntimeStatus::Paused
                            ) {
                                0
                            } else {
                                footprint.runtime_bytes
                            }
                    }
                };
                (
                    *session_id,
                    priority,
                    idle_for < self.config.idle_session_evict_after,
                    mode_bias,
                    footprint.runtime_bytes,
                    footprint.kv_cache_bytes,
                )
            })
            .collect::<Vec<_>>();

        candidates.sort_by_key(|candidate| {
            (
                candidate.1,
                candidate.2,
                std::cmp::Reverse(candidate.3),
                std::cmp::Reverse(candidate.4),
                std::cmp::Reverse(candidate.5),
            )
        });
        candidates.first().map(|candidate| candidate.0)
    }

    fn build_next_decode_batch(
        &mut self,
        primary_session_id: Uuid,
        targets: DecodeBatchTargets,
    ) -> DecodeBatchPlan {
        let policy = self.decode_batch_policy_for_targets(targets);
        let mut total_kv_tokens = 0_usize;
        let mut slots = Vec::new();
        let mut deferred = Vec::new();
        let mut deferred_for_capacity = 0_usize;
        let mut deferred_for_kv_budget = 0_usize;
        let mut queue = std::mem::take(&mut self.decode_queue)
            .into_iter()
            .collect::<Vec<_>>();
        queue.sort_by_key(|task| {
            self.sessions
                .get(&task.session_id)
                .map(|session| {
                    (
                        task.session_id != primary_session_id,
                        session.decode_steps_served,
                        task.fairness_epoch,
                    )
                })
                .unwrap_or((true, u64::MAX, task.fairness_epoch))
        });

        for task in queue {
            let Some(session) = self.sessions.get(&task.session_id) else {
                continue;
            };

            if session.job.is_complete() {
                continue;
            }

            let kv_tokens = Self::session_decode_kv_tokens(session);
            let batch_full = slots.len() >= policy.max_batch_size;
            let kv_exhausted = !slots.is_empty()
                && total_kv_tokens.saturating_add(kv_tokens) > policy.max_total_kv_tokens;
            if batch_full || kv_exhausted {
                if batch_full {
                    deferred_for_capacity = deferred_for_capacity.saturating_add(1);
                } else {
                    deferred_for_kv_budget = deferred_for_kv_budget.saturating_add(1);
                }
                deferred.push(task);
                continue;
            }

            slots.push(DecodeBatchSlot {
                session_id: task.session_id,
                fairness_epoch: task.fairness_epoch,
                kv_tokens,
            });
            total_kv_tokens = total_kv_tokens.saturating_add(kv_tokens);
        }

        for slot in &slots {
            if let Some(session) = self.sessions.get_mut(&slot.session_id) {
                session.queued_for_decode = false;
            }
        }

        self.decode_queue = deferred.iter().copied().collect();

        DecodeBatchPlan {
            slots,
            deferred,
            total_kv_tokens,
            deferred_for_capacity,
            deferred_for_kv_budget,
        }
    }

    fn active_decode_session_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|session| session.job.has_decode_context() && !session.job.is_complete())
            .count()
    }

    fn decode_batch_policy_for_targets(&self, targets: DecodeBatchTargets) -> DecodeBatchPolicy {
        let base = self.decode_batch_policy();
        let mut max_batch_size = base.max_batch_size;

        if let Some(target_batch_size) = targets.target_batch_size {
            max_batch_size = max_batch_size.min(target_batch_size.max(1) as usize);
        }
        if let Some(target_session_count) = targets.target_session_count {
            max_batch_size = max_batch_size.min(target_session_count.max(1) as usize);
        }

        DecodeBatchPolicy {
            max_batch_size: max_batch_size.max(1),
            max_total_kv_tokens: base.max_total_kv_tokens,
        }
    }

    pub async fn export_session_checkpoint_bytes(&mut self, session_id: Uuid) -> Result<Vec<u8>> {
        let Some(manager) = self.checkpoint_manager.clone() else {
            return Err(AgentError::Execution(
                "Checkpoint manager is not configured for session handoff export".to_string(),
            ));
        };
        let job_snapshot = self
            .sessions
            .get(&session_id)
            .ok_or_else(|| {
                AgentError::Execution(format!(
                    "Session {} is not active for checkpoint export",
                    session_id
                ))
            })?
            .job
            .clone();
        let kv_snapshot = self
            .sessions
            .get(&session_id)
            .map(|session| session.backend.export_kv_cache())
            .transpose()?
            .flatten();
        manager
            .save_checkpoint(&job_snapshot, kv_snapshot.as_ref())
            .await?;
        manager
            .export_latest_checkpoint_bytes(job_snapshot.request.job_id)
            .await?
            .ok_or_else(|| {
                AgentError::Execution(format!(
                    "Checkpoint export for job {} produced no bytes",
                    job_snapshot.request.job_id
                ))
            })
    }

    pub fn checkpoint_manager(&self) -> Option<Arc<crate::checkpoint::CheckpointManager>> {
        self.checkpoint_manager.clone()
    }

    /// Get reference to swarm
    pub fn swarm(&self) -> &MeshSwarm {
        &self.swarm
    }

    /// Get mutable reference to swarm
    pub fn swarm_mut(&mut self) -> &mut MeshSwarm {
        &mut self.swarm
    }

    pub fn tensor_plane_mut(&mut self) -> &mut TensorPlane {
        &mut self.tensor_plane
    }

    fn sync_tensor_plane_metrics(&self) {
        self.stats
            .update_tensor_plane_metrics(self.tensor_plane.metrics_snapshot());
    }

    /// Check if this worker has joined a ring
    pub fn is_in_ring(&self) -> bool {
        self.position.is_some()
    }

    /// Get worker position information
    pub fn worker_position(&self) -> Option<&WorkerPosition> {
        self.position.as_ref()
    }

    /// Get or load model weights for the current worker position
    ///
    /// This method checks the weight cache first. If weights are not cached,
    /// it loads them from the production shard loader.
    ///
    /// Weights are cached per model_id to avoid regenerating them for each token.
    async fn get_or_load_weights(
        &mut self,
        model_id: &str,
        position: &WorkerPosition,
    ) -> Result<Arc<ModelWeights>> {
        // Check cache first
        {
            let cache = self.weight_cache.read().await;
            if let Some(weights) = cache.get(model_id) {
                debug!(model_id = %model_id, "Using cached weights");
                return Ok(Arc::clone(weights));
            }
        }

        debug!(model_id = %model_id, "Cache miss, loading weights");

        // Create shard assignment for this worker
        let assignment = ShardAssignment::from_column_range(
            model_id.to_string(),
            position.position,
            position.total_workers,
            position.shard_column_range.0,
            position.shard_column_range.1,
        );

        // Assign shard to registry if not already assigned
        if self.shard_registry.get_shard(model_id).await.is_none() {
            self.shard_registry.assign_shard(assignment.clone()).await?;
        }

        // Load from the production shard loader.
        let weights = self
            .loader
            .load_shard(model_id, &assignment, &self.shard_registry)
            .await?;

        // Cache it
        let weights_arc = Arc::new(weights);
        {
            let mut cache = self.weight_cache.write().await;
            cache.insert(model_id.to_string(), Arc::clone(&weights_arc));
        }

        info!(
            model_id = %model_id,
            memory_mb = weights_arc.memory_usage() / 1_000_000,
            "Weights loaded and cached"
        );

        Ok(weights_arc)
    }

    /// Join the ring topology
    ///
    /// This sets up the worker's position and neighbors for ring all-reduce.
    #[instrument(skip(self))]
    pub fn join_ring(&mut self, position: WorkerPosition) -> Result<()> {
        info!(
            position = position.position,
            total_workers = position.total_workers,
            left = %position.left_neighbor,
            right = %position.right_neighbor,
            shard_range = ?position.shard_column_range,
            "Joining ring topology"
        );

        // Set ring neighbors on the swarm
        self.swarm.set_ring_neighbors(
            position.left_neighbor,
            &position.left_neighbor_addrs,
            position.left_neighbor_punch_plan.as_ref(),
            position.right_neighbor,
            &position.right_neighbor_addrs,
            position.right_neighbor_punch_plan.as_ref(),
        );

        self.position = Some(position);

        info!("Successfully joined ring");
        Ok(())
    }

    /// Leave the ring topology
    #[instrument(skip(self))]
    pub fn leave_ring(&mut self) {
        if self.position.is_some() {
            info!("Leaving ring topology");
            self.swarm.clear_ring_neighbors();
            self.position = None;
        }
    }

    /// Process an inference request
    ///
    /// This is the main entry point for inference. It:
    /// 1. Creates an InferenceJob from the request
    /// 2. Runs the token generation loop
    /// 3. Returns the result
    #[instrument(skip(self, request), fields(job_id = %request.job_id))]
    pub async fn process_inference(
        &mut self,
        request: InferenceRequest,
    ) -> Result<InferenceResult> {
        self.process_inference_with_progress(request, |_| async { Ok(()) })
            .await
    }

    #[instrument(skip(self, request, on_progress), fields(job_id = %request.job_id, session_id = %request.session_id, phase = ?request.phase))]
    pub async fn process_segment_with_progress<F, Fut>(
        &mut self,
        request: InferenceRequest,
        mut on_progress: F,
    ) -> Result<SegmentExecutionResult>
    where
        F: FnMut(InferenceProgressUpdate) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let position = self.position.clone().ok_or_else(|| {
            AgentError::Execution("Worker is not part of a ring topology".to_string())
        })?;

        info!(
            job_id = %request.job_id,
            session_id = %request.session_id,
            phase = ?request.phase,
            model = %request.model_id,
            prompt_tokens = request.prompt_tokens.len(),
            max_tokens = request.config.max_tokens,
            "Starting execution segment"
        );

        self.ensure_session_backend(&request, &position).await?;
        match request.phase {
            ExecutionPhase::Prefill => {
                let result = self
                    .run_prefill_segment(&request, &position, &mut on_progress)
                    .await?;
                Ok(result)
            }
            ExecutionPhase::Decode => {
                let result = self
                    .run_decode_segment(&request, &position, &mut on_progress)
                    .await?;
                Ok(result)
            }
        }
    }

    #[instrument(skip(self, request, on_progress), fields(job_id = %request.job_id))]
    pub async fn process_inference_with_progress<F, Fut>(
        &mut self,
        request: InferenceRequest,
        mut on_progress: F,
    ) -> Result<InferenceResult>
    where
        F: FnMut(InferenceProgressUpdate) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let position = self.position.clone().ok_or_else(|| {
            AgentError::Execution("Worker is not part of a ring topology".to_string())
        })?;

        info!(
            job_id = %request.job_id,
            model = %request.model_id,
            prompt_tokens = request.prompt_tokens.len(),
            max_tokens = request.config.max_tokens,
            "Starting inference job"
        );

        let start = Instant::now();
        self.ensure_session_backend(&request, &position).await?;

        let mut prefill_request = request.clone();
        prefill_request.phase = ExecutionPhase::Prefill;
        if let Err(error) = self
            .run_prefill_segment(&prefill_request, &position, &mut on_progress)
            .await
        {
            self.stats.record_failure();
            self.sessions.remove(&prefill_request.session_id);
            self.decode_queue
                .retain(|task| task.session_id != prefill_request.session_id);
            return Ok(InferenceResult::failure(
                prefill_request.job_id,
                error.to_string(),
                start.elapsed().as_millis() as u64,
            ));
        }

        let mut decode_request = request;
        decode_request.phase = ExecutionPhase::Decode;
        match self
            .run_decode_segment(&decode_request, &position, &mut on_progress)
            .await
        {
            Ok(SegmentExecutionResult::Completed(result)) => Ok(result),
            Ok(SegmentExecutionResult::PrefillComplete { .. }) => Err(AgentError::Execution(
                "decode execution unexpectedly returned a prefill result".to_string(),
            )),
            Err(error) => {
                self.stats.record_failure();
                self.sessions.remove(&decode_request.session_id);
                self.decode_queue
                    .retain(|task| task.session_id != decode_request.session_id);
                Ok(InferenceResult::failure(
                    decode_request.job_id,
                    error.to_string(),
                    start.elapsed().as_millis() as u64,
                ))
            }
        }
    }

    fn ensure_checkpoint_resume_safe(job: &InferenceJob, has_kv_state: bool) -> Result<()> {
        if has_kv_state || job.generated_tokens.is_empty() {
            return Ok(());
        }

        Err(AgentError::Execution(format!(
            "Checkpoint for job {} cannot safely resume decode without KV state",
            job.request.job_id
        )))
    }

    /// Create a checkpoint of the current inference state
    async fn checkpoint(&mut self, job: &InferenceJob) -> Result<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            debug!(
                job_id = %job.request.job_id,
                token_idx = job.current_token_idx,
                "Creating checkpoint"
            );

            let kv_snapshot = self
                .sessions
                .get(&job.request.session_id)
                .map(|session| session.backend.export_kv_cache())
                .transpose()?
                .flatten();
            manager.save_checkpoint(job, kv_snapshot.as_ref()).await?;
            self.stats.record_checkpoint();
        }
        Ok(())
    }

    /// Attempt to recover a job from checkpoint
    #[instrument(skip(self))]
    pub async fn recover_from_checkpoint(&mut self, job_id: Uuid) -> Result<Option<InferenceJob>> {
        if let Some(manager) = self.checkpoint_manager.clone() {
            if let Some(job) = manager.load_checkpoint(job_id).await? {
                if let Some(position) = self.position.clone() {
                    let session = self.ensure_session_backend(&job.request, &position).await?;
                    session.job = job.clone();

                    if let Some(kv_handoff) = manager.load_checkpoint_kv_handoff(job_id).await? {
                        let snapshot = kv_handoff.materialize_snapshot().map_err(|e| {
                            AgentError::Execution(format!(
                                "Checkpoint KV state for job {} is not locally materialized: {}",
                                job_id, e
                            ))
                        })?;
                        let snapshot = snapshot.ok_or_else(|| {
                            AgentError::Execution(format!(
                                "Checkpoint KV state for job {} requires external fetch",
                                job_id
                            ))
                        })?;
                        session.backend.import_kv_cache(&snapshot)?;
                    } else {
                        Self::ensure_checkpoint_resume_safe(&job, false)?;
                        session.backend.clear();
                    }
                }

                self.stats.record_recovery();
                info!(
                    job_id = %job_id,
                    token_idx = job.current_token_idx,
                    "Recovered from checkpoint"
                );
                return Ok(Some(job));
            }
        }
        Ok(None)
    }

    #[allow(dead_code)]
    async fn try_recover_job(&mut self, job_id: Uuid) -> Result<Option<InferenceJob>> {
        match self.recovery_governor.allow_attempt(
            job_id,
            self.config.recovery_cooldown,
            self.config.recovery_max_checkpoint_loads_per_minute,
            Instant::now(),
        ) {
            RecoveryAllowance::Allowed => {
                self.stats.record_recovery_attempt();
            }
            RecoveryAllowance::Cooldown => {
                self.stats.record_recovery_cooldown_rejection();
                warn!(
                    job_id = %job_id,
                    cooldown_ms = self.config.recovery_cooldown.as_millis(),
                    "Skipping checkpoint recovery because cooldown is still active"
                );
                return Ok(None);
            }
            RecoveryAllowance::LoadBudget => {
                self.stats.record_recovery_budget_rejection();
                warn!(
                    job_id = %job_id,
                    max_loads_per_minute = self.config.recovery_max_checkpoint_loads_per_minute,
                    "Skipping checkpoint recovery because node recovery load budget is exhausted"
                );
                return Ok(None);
            }
        }

        let recovered = self.recover_from_checkpoint(job_id).await?;
        if recovered.is_none() {
            self.stats.record_recovery_checkpoint_miss();
        }
        Ok(recovered)
    }

    async fn recover_session_from_checkpoint(
        &mut self,
        request: &InferenceRequest,
        position: &WorkerPosition,
    ) -> Result<bool> {
        let Some(manager) = self.checkpoint_manager.clone() else {
            return Ok(false);
        };

        let Some(recovered_job) = manager.load_checkpoint(request.job_id).await? else {
            return Ok(false);
        };

        if recovered_job.request.session_id != request.session_id {
            return Err(AgentError::Execution(format!(
                "Checkpoint session {} does not match requested session {}",
                recovered_job.request.session_id, request.session_id
            )));
        }

        let session = self
            .ensure_session_backend(&recovered_job.request, position)
            .await?;
        session.job = recovered_job.clone();

        if let Some(kv_handoff) = manager.load_checkpoint_kv_handoff(request.job_id).await? {
            let snapshot = kv_handoff.materialize_snapshot().map_err(|e| {
                AgentError::Execution(format!(
                    "Checkpoint KV state for job {} is not locally materialized: {}",
                    request.job_id, e
                ))
            })?;
            let snapshot = snapshot.ok_or_else(|| {
                AgentError::Execution(format!(
                    "Checkpoint KV state for job {} requires external fetch",
                    request.job_id
                ))
            })?;
            session.backend.import_kv_cache(&snapshot)?;
        } else {
            Self::ensure_checkpoint_resume_safe(&recovered_job, false)?;
            session.backend.clear();
        }

        Ok(true)
    }

    async fn run_prefill_segment<F, Fut>(
        &mut self,
        request: &InferenceRequest,
        position: &WorkerPosition,
        on_progress: &mut F,
    ) -> Result<SegmentExecutionResult>
    where
        F: FnMut(InferenceProgressUpdate) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        if request.prompt_tokens.is_empty() {
            return Err(AgentError::Execution(
                "Inference request must contain at least one prompt token".to_string(),
            ));
        }

        let segment_start = Instant::now();
        let job_snapshot = {
            let session = self.ensure_session_backend(request, position).await?;
            if !session.job.generated_tokens.is_empty() {
                return Err(AgentError::Execution(format!(
                    "Prefill segment for session {} was requested after generation already started",
                    request.session_id
                )));
            }
            session.job.clone()
        };
        {
            let mut active = self.active_job.write().await;
            *active = Some(job_snapshot);
        }

        let next_token = {
            let mut session = self.sessions.remove(&request.session_id).ok_or_else(|| {
                AgentError::Execution(
                    "session backend missing at prefill execution time".to_string(),
                )
            })?;
            let mut worker_ring = WorkerRing::new(
                position.position,
                position.total_workers,
                position.left_neighbor,
                position.right_neighbor,
                position.left_neighbor_tensor_addr,
                position.right_neighbor_tensor_addr,
                session.job.request.runtime_mode,
                session.backend.provider_kind(),
                self.tensor_plane_mut(),
            );
            worker_ring.prepare_serving_group_channels().await?;
            session.engine_state.assignment.phase = ExecutionPhase::Prefill;
            let logits = session
                .backend
                .prefill(&request.prompt_tokens, &mut worker_ring, request.job_id)
                .await?;
            let seed = request.job_id.as_u128() as u64 ^ session.backend.sequence_position() as u64;
            let next_token = session.backend.sample(
                &logits,
                request.config.temperature,
                request.config.top_p,
                seed,
            );
            self.sessions.insert(request.session_id, session);
            next_token
        };

        let execution_time_ms = segment_start.elapsed().as_millis() as u64;
        let completion_tokens = {
            let session = self.sessions.get_mut(&request.session_id).ok_or_else(|| {
                AgentError::Execution("session state vanished after prefill".to_string())
            })?;
            session.job.add_token(next_token);
            session.job.request.phase = ExecutionPhase::Decode;
            session.job.current_layer = self.config.total_layers;
            session.job.clone()
        };
        let kv_cache_seq_len = self
            .sessions
            .get(&request.session_id)
            .map(|session| session.backend.cache_seq_len() as u32)
            .unwrap_or(request.prompt_tokens.len() as u32);

        on_progress(InferenceProgressUpdate {
            phase: ExecutionPhase::Prefill,
            completion_tokens: completion_tokens.current_token_idx,
            execution_time_ms,
            time_to_first_token_ms: completion_tokens
                .time_to_first_token()
                .map(|d| d.as_millis() as u64),
            kv_cache_seq_len: Some(kv_cache_seq_len),
            batch_size: None,
            active_decode_sessions: None,
            batch_kv_tokens: None,
            deferred_decode_sessions: None,
        })
        .await?;

        if self.config.checkpointing_enabled {
            self.checkpoint(&completion_tokens).await?;
            if let Some(session) = self.sessions.get_mut(&request.session_id) {
                session.job.mark_checkpointed();
            }
        }

        {
            let mut active = self.active_job.write().await;
            *active = None;
        }

        self.record_generation_metrics(request.session_id, execution_time_ms);

        Ok(SegmentExecutionResult::PrefillComplete {
            job_id: request.job_id,
            session_id: request.session_id,
            completion_tokens: completion_tokens.current_token_idx,
            execution_time_ms,
            time_to_first_token_ms: completion_tokens
                .time_to_first_token()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(execution_time_ms),
            kv_cache_seq_len,
        })
    }

    async fn execute_decode_microbatch(
        &mut self,
        batch: DecodeBatchPlan,
        position: &WorkerPosition,
    ) -> Result<Vec<DecodeStepOutcome>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let mut batch_sessions = Vec::with_capacity(batch.slots.len());
        for slot in batch.slots {
            let mut session = self.sessions.remove(&slot.session_id).ok_or_else(|| {
                AgentError::Execution(format!(
                    "Decode session {} vanished after batch admission",
                    slot.session_id
                ))
            })?;
            let decode_token = session.job.last_generated_token().ok_or_else(|| {
                AgentError::Execution(format!(
                    "Decode session {} entered the queue without a sampled token",
                    slot.session_id
                ))
            })?;
            session.engine_state.assignment.phase = ExecutionPhase::Decode;
            batch_sessions.push((slot.session_id, session, decode_token));
        }

        let mut worker_ring = WorkerRing::new(
            position.position,
            position.total_workers,
            position.left_neighbor,
            position.right_neighbor,
            position.left_neighbor_tensor_addr,
            position.right_neighbor_tensor_addr,
            batch_sessions
                .first()
                .map(|(_, session, _)| session.job.request.runtime_mode)
                .unwrap_or_default(),
            batch_sessions
                .first()
                .map(|(_, session, _)| session.backend.provider_kind())
                .unwrap_or(crate::provider::ExecutionProviderKind::Cpu),
            self.tensor_plane_mut(),
        );
        worker_ring.prepare_serving_group_channels().await?;
        let mut requests = batch_sessions
            .iter_mut()
            .map(
                |(session_id, session, decode_token)| DecodeMicrobatchRequest {
                    session_id: *session_id,
                    job_id: session.job.request.job_id,
                    token: *decode_token,
                    backend: session.backend.as_mut(),
                },
            )
            .collect::<Vec<_>>();
        let outputs =
            BackendMicrobatchExecutor::decode_step_batch(&mut requests, &mut worker_ring).await?;

        let mut outcomes = Vec::with_capacity(outputs.len());
        let mut sessions_to_requeue = Vec::new();

        for ((session_id, mut session, _), output) in batch_sessions.into_iter().zip(outputs) {
            debug_assert_eq!(session_id, output.session_id);
            let seed = session.job.request.job_id.as_u128() as u64
                ^ session.backend.sequence_position() as u64;
            let sampled_token = session.backend.sample(
                &output.logits,
                session.job.request.config.temperature,
                session.job.request.config.top_p,
                seed,
            );
            session.job.add_token(sampled_token);
            session.job.current_layer = self.config.total_layers;
            session.decode_steps_served = session.decode_steps_served.saturating_add(1);

            let outcome = DecodeStepOutcome {
                session_id,
                completion_tokens: session.job.current_token_idx,
                execution_time_ms: output.execution_time_ms,
                kv_cache_seq_len: session.backend.cache_seq_len() as u32,
                should_checkpoint: self.config.checkpointing_enabled
                    && session.job.should_checkpoint(),
                completed: session.job.is_complete(),
                time_to_first_token_ms: session
                    .job
                    .time_to_first_token()
                    .map(|ttft| ttft.as_millis() as u64),
            };

            self.sessions.insert(session_id, session);
            if !outcome.completed {
                sessions_to_requeue.push(session_id);
            }
            outcomes.push(outcome);
        }

        for outcome in &outcomes {
            self.record_generation_metrics(outcome.session_id, outcome.execution_time_ms);
        }

        for session_id in sessions_to_requeue {
            self.enqueue_decode_task(session_id)?;
        }

        Ok(outcomes)
    }

    async fn run_scheduler_owned_decode_worker<F, Fut>(
        &mut self,
        request: &InferenceRequest,
        position: &WorkerPosition,
        on_progress: &mut F,
    ) -> Result<(InferenceResult, Option<DecodeBatchTelemetry>, u32)>
    where
        F: FnMut(InferenceProgressUpdate) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let segment_start = Instant::now();
        let progress_report_interval = request.config.progress_report_interval.max(1);
        let mut last_reported_completion_tokens = self
            .sessions
            .get(&request.session_id)
            .map(|session| session.job.current_token_idx)
            .unwrap_or(0);
        let mut last_batch_telemetry = None::<DecodeBatchTelemetry>;

        self.resume_session(request.session_id)?;
        self.enqueue_decode_task(request.session_id)?;

        loop {
            let Some(session) = self.sessions.get(&request.session_id) else {
                return Err(AgentError::Execution(
                    "decode session vanished during execution".to_string(),
                ));
            };
            if session.job.is_complete() {
                break;
            }
            if matches!(
                session.engine_state.runtime_status,
                SessionRuntimeStatus::Paused
            ) {
                let detail = session
                    .engine_state
                    .paused
                    .as_ref()
                    .map(|state| state.detail.clone())
                    .unwrap_or_else(|| "decode session paused".to_string());
                return Err(AgentError::Execution(detail));
            }
            if session.job.elapsed() > self.config.job_timeout {
                return Err(AgentError::Execution("Inference job timed out".to_string()));
            }

            self.enforce_runtime_memory_budget(Some(request.session_id))
                .await?;

            let batch =
                self.build_next_decode_batch(request.session_id, request.decode_batch_targets);
            if batch.is_empty() {
                return Err(AgentError::Execution(
                    "decode runtime had no admitted work despite active queued sessions"
                        .to_string(),
                ));
            }
            self.stats.record_decode_microbatch(
                batch.slots.len(),
                batch.total_kv_tokens,
                batch.deferred.len(),
                batch.deferred_for_capacity,
                batch.deferred_for_kv_budget,
            );
            let batch_size = u32::try_from(batch.slots.len()).unwrap_or(u32::MAX);
            let batch_kv_tokens = u32::try_from(batch.total_kv_tokens).unwrap_or(u32::MAX);
            let deferred_decode_sessions = u32::try_from(batch.deferred.len()).unwrap_or(u32::MAX);

            let outcomes = self.execute_decode_microbatch(batch, position).await?;
            let active_decode_sessions =
                u32::try_from(self.active_decode_session_count()).unwrap_or(u32::MAX);

            for outcome in outcomes {
                if outcome.should_checkpoint {
                    let checkpoint_snapshot = self
                        .sessions
                        .get(&outcome.session_id)
                        .map(|session| session.job.clone())
                        .ok_or_else(|| {
                            AgentError::Execution(
                                "decode session vanished before checkpointing".to_string(),
                            )
                        })?;
                    self.checkpoint(&checkpoint_snapshot).await?;
                    if let Some(session) = self.sessions.get_mut(&outcome.session_id) {
                        session.job.mark_checkpointed();
                    }
                }

                if let Some(session) = self.sessions.get_mut(&outcome.session_id) {
                    session.last_active_at = Instant::now();
                }

                if outcome.session_id != request.session_id {
                    continue;
                }

                last_batch_telemetry = Some(DecodeBatchTelemetry {
                    kv_cache_seq_len: outcome.kv_cache_seq_len,
                    batch_size,
                    active_decode_sessions,
                    batch_kv_tokens,
                    deferred_decode_sessions,
                });

                if outcome
                    .completion_tokens
                    .saturating_sub(last_reported_completion_tokens)
                    >= progress_report_interval
                {
                    on_progress(InferenceProgressUpdate {
                        phase: ExecutionPhase::Decode,
                        completion_tokens: outcome.completion_tokens,
                        execution_time_ms: segment_start.elapsed().as_millis() as u64,
                        time_to_first_token_ms: outcome.time_to_first_token_ms,
                        kv_cache_seq_len: Some(outcome.kv_cache_seq_len),
                        batch_size: Some(batch_size),
                        active_decode_sessions: Some(active_decode_sessions),
                        batch_kv_tokens: Some(batch_kv_tokens),
                        deferred_decode_sessions: Some(deferred_decode_sessions),
                    })
                    .await?;
                    last_reported_completion_tokens = outcome.completion_tokens;
                }
            }
        }

        let execution_time_ms = segment_start.elapsed().as_millis() as u64;
        let session = self.sessions.remove(&request.session_id).ok_or_else(|| {
            AgentError::Execution("decode session vanished before finalization".to_string())
        })?;
        let mut result = session.job.into_result();
        if execution_time_ms > result.execution_time_ms {
            result.execution_time_ms = execution_time_ms;
        }

        if result.completion_tokens > last_reported_completion_tokens {
            let telemetry = last_batch_telemetry;
            on_progress(InferenceProgressUpdate {
                phase: ExecutionPhase::Decode,
                completion_tokens: result.completion_tokens,
                execution_time_ms,
                time_to_first_token_ms: None,
                kv_cache_seq_len: telemetry.map(|value| value.kv_cache_seq_len),
                batch_size: telemetry.map(|value| value.batch_size),
                active_decode_sessions: telemetry.map(|value| value.active_decode_sessions),
                batch_kv_tokens: telemetry.map(|value| value.batch_kv_tokens),
                deferred_decode_sessions: telemetry.map(|value| value.deferred_decode_sessions),
            })
            .await?;
        }

        Ok((result, last_batch_telemetry, execution_time_ms as u32))
    }

    async fn run_decode_segment<F, Fut>(
        &mut self,
        request: &InferenceRequest,
        position: &WorkerPosition,
        on_progress: &mut F,
    ) -> Result<SegmentExecutionResult>
    where
        F: FnMut(InferenceProgressUpdate) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        if !self.sessions.contains_key(&request.session_id) {
            let recovered = self
                .recover_session_from_checkpoint(request, position)
                .await?;
            if recovered {
                info!(
                    job_id = %request.job_id,
                    session_id = %request.session_id,
                    "Recovered decode segment session state from checkpoint"
                );
            }
        }

        let job_snapshot = {
            let session = self.sessions.get(&request.session_id).ok_or_else(|| {
                AgentError::Execution(format!(
                    "Decode segment for session {} cannot start before prefill state exists",
                    request.session_id
                ))
            })?;
            if session.job.generated_tokens.is_empty() {
                return Err(AgentError::Execution(format!(
                    "Decode segment for session {} cannot start without a sampled prefill token",
                    request.session_id
                )));
            }
            session.job.clone()
        };
        {
            let mut active = self.active_job.write().await;
            *active = Some(job_snapshot);
        }
        let (result, _last_batch_telemetry, _execution_time_ms) = self
            .run_scheduler_owned_decode_worker(request, position, on_progress)
            .await?;

        {
            let mut active = self.active_job.write().await;
            *active = None;
        }

        self.stats.record_success(
            result.prompt_tokens,
            result.completion_tokens,
            result.execution_time_ms,
        );

        Ok(SegmentExecutionResult::Completed(result))
    }

    fn record_generation_metrics(&self, session_id: Uuid, elapsed_ms: u64) {
        self.stats.record_allreduce(elapsed_ms);
        let layer_count = self.config.total_layers;
        if let Some(session) = self.sessions.get(&session_id) {
            self.stats
                .record_allreduce_breakdown(session.backend.last_allreduce_metrics());
        }
        for _ in 0..layer_count {
            self.stats.record_layer();
        }
        self.sync_tensor_plane_metrics();
    }

    /// Run the coordinator event loop
    ///
    /// This listens for network events and processes them.
    #[instrument(skip(self), fields(peer_id = %self.swarm.local_peer_id()))]
    pub async fn run(mut self) -> Result<()> {
        use tokio::signal;

        info!("Starting inference coordinator");

        // Periodic stats saver
        let mut stats_interval = tokio::time::interval(Duration::from_secs(30));
        stats_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Handle network events
                event = self.swarm.next_event() => {
                    if let Some(event) = event {
                        if let Err(e) = self.handle_event(event).await {
                            error!(error = %e, "Error handling event");
                        }
                    } else {
                        warn!("Event stream ended");
                        break;
                    }
                }

                // Save stats periodically
                _ = stats_interval.tick() => {
                    debug!("Saving inference stats");
                    self.sync_tensor_plane_metrics();
                    if let Err(e) = self.stats.save_to_file() {
                        warn!(error = %e, "Failed to save stats");
                    }
                }

                // Handle shutdown signal
                _ = signal::ctrl_c() => {
                    info!("Received shutdown signal");
                    break;
                }
            }
        }

        // Final cleanup
        self.leave_ring();
        self.sync_tensor_plane_metrics();
        self.stats.print_summary();
        let _ = self.stats.save_to_file();

        Ok(())
    }

    /// Handle a mesh network event
    async fn handle_event(&mut self, event: crate::network::MeshEvent) -> Result<()> {
        use crate::network::MeshEvent;

        match event {
            MeshEvent::PeerDisconnected { peer_id } => {
                // Check if disconnected peer is a ring neighbor
                if let Some(pos) = &self.position {
                    if peer_id == pos.left_neighbor || peer_id == pos.right_neighbor {
                        error!(
                            peer_id = %peer_id,
                            "Ring neighbor disconnected! Ring topology broken."
                        );
                        let impacted_sessions = self
                            .sessions
                            .iter()
                            .filter_map(|(session_id, session)| {
                                if session.job.has_decode_context() && !session.job.is_complete() {
                                    Some(*session_id)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();
                        for session_id in impacted_sessions {
                            self.pause_session(
                                session_id,
                                SessionPauseReason::Failover,
                                format!("ring neighbor {} disconnected", peer_id),
                            );
                        }
                    }
                }
            }

            _ => {
                // Ignore other events
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::ring_allreduce::RingAllReduceMetrics;
    use crate::inference::job::InferenceRequest;
    use crate::inference::kv_cache::KVCacheSnapshot;
    use crate::inference::tensor_ops::Tensor1D;
    use crate::network::TensorPlaneConfig;
    use crate::provider::ExecutionProviderKind;

    struct TestBackend {
        cache_seq_len: usize,
        cache_memory_usage: usize,
        sequence_position: usize,
    }

    #[async_trait::async_trait]
    impl ExecutionBackend for TestBackend {
        fn provider_kind(&self) -> ExecutionProviderKind {
            ExecutionProviderKind::Cpu
        }

        fn optimization_profile(&self) -> crate::inference::BackendOptimizationProfile {
            crate::inference::BackendOptimizationProfile::CpuSerial
        }

        async fn prefill(
            &mut self,
            _tokens: &[u32],
            _worker_ring: &mut WorkerRing<'_>,
            _job_id: Uuid,
        ) -> Result<Tensor1D> {
            Ok(Tensor1D::zeros(1))
        }

        async fn decode_step(
            &mut self,
            _token: u32,
            _worker_ring: &mut WorkerRing<'_>,
            _job_id: Uuid,
        ) -> Result<Tensor1D> {
            Ok(Tensor1D::zeros(1))
        }

        fn sample(&self, _logits: &Tensor1D, _temperature: f32, _top_p: f32, _seed: u64) -> u32 {
            0
        }

        fn cache_seq_len(&self) -> usize {
            self.cache_seq_len
        }

        fn cache_memory_usage(&self) -> usize {
            self.cache_memory_usage
        }

        fn sequence_position(&self) -> usize {
            self.sequence_position
        }

        fn last_allreduce_metrics(&self) -> RingAllReduceMetrics {
            RingAllReduceMetrics::default()
        }

        fn export_kv_cache(&self) -> Result<Option<KVCacheSnapshot>> {
            Ok(None)
        }

        fn import_kv_cache(&mut self, _snapshot: &KVCacheSnapshot) -> Result<()> {
            Ok(())
        }

        fn clear(&mut self) {}
    }

    fn test_coordinator(config: InferenceConfig) -> InferenceCoordinator {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let swarm = MeshSwarm::builder(keypair).build().expect("test swarm");
        let tensor_plane = tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(TensorPlane::bind(TensorPlaneConfig::default()))
            .expect("tensor plane");
        InferenceCoordinator::new(swarm, tensor_plane, config)
    }

    fn insert_decode_session(
        coordinator: &mut InferenceCoordinator,
        session_id: Uuid,
        kv_tokens: usize,
        generated_token: u32,
    ) {
        let mut request = InferenceRequest::new(
            "test-network".to_string(),
            "llama-70b".to_string(),
            vec![1, 2, 3],
            "executor-1".to_string(),
        );
        request.session_id = session_id;
        request.phase = ExecutionPhase::Decode;
        request.config.max_tokens = 8;

        let mut job = InferenceJob::new(request.clone(), 70);
        job.add_token(generated_token);

        coordinator.sessions.insert(
            session_id,
            ActiveSession {
                engine_state: EngineSessionState::new(
                    session_id,
                    ExecutionPhase::Decode,
                    "worker-1".to_string(),
                    vec!["worker-1".to_string()],
                    vec!["worker-1".to_string()],
                    InferenceRuntimeMode::ThroughputFirst,
                    ExecutionProviderKind::Cpu,
                    (0, 1024),
                    TransportCapabilityTier::DirectPreferred,
                    coordinator.runtime_memory_budget(),
                    request.config.max_tokens,
                    request.prompt_tokens.len(),
                ),
                backend: Box::new(TestBackend {
                    cache_seq_len: kv_tokens,
                    cache_memory_usage: kv_tokens * 1024,
                    sequence_position: kv_tokens,
                }),
                job,
                queued_for_decode: false,
                decode_steps_served: 0,
                last_active_at: Instant::now(),
            },
        );
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.total_layers, 70);
        assert_eq!(config.model_id, "llama-70b");
        assert!(config.checkpointing_enabled);
        assert_eq!(config.recovery_max_attempts_per_job, 2);
        assert_eq!(config.recovery_cooldown, Duration::from_secs(5));
        assert_eq!(config.recovery_max_checkpoint_loads_per_minute, 8);
        assert_eq!(config.max_decode_batch_size, 4);
        assert_eq!(config.max_decode_batch_kv_tokens, 16_384);
    }

    #[test]
    fn test_worker_position() {
        let _peer_id = PeerId::random();
        let position = WorkerPosition {
            position: 3,
            total_workers: 10,
            left_neighbor: PeerId::random(),
            left_neighbor_addrs: vec![],
            left_neighbor_punch_plan: None,
            left_neighbor_tensor_addr: "127.0.0.1:5001".parse().unwrap(),
            right_neighbor: PeerId::random(),
            right_neighbor_addrs: vec![],
            right_neighbor_punch_plan: None,
            right_neighbor_tensor_addr: "127.0.0.1:5002".parse().unwrap(),
            shard_column_range: (2457, 3276),
            shard_memory_bytes: 7_000_000_000,
        };

        assert_eq!(position.position, 3);
        assert_eq!(position.total_workers, 10);
    }

    #[test]
    fn test_recovery_governor_rejects_attempts_inside_cooldown() {
        let mut governor = RecoveryGovernor::default();
        let job_id = Uuid::new_v4();
        let now = Instant::now();

        assert_eq!(
            governor.allow_attempt(job_id, Duration::from_secs(5), 8, now),
            RecoveryAllowance::Allowed
        );
        assert_eq!(
            governor.allow_attempt(
                job_id,
                Duration::from_secs(5),
                8,
                now + Duration::from_secs(1)
            ),
            RecoveryAllowance::Cooldown
        );
        assert_eq!(
            governor.allow_attempt(
                job_id,
                Duration::from_secs(5),
                8,
                now + Duration::from_secs(6)
            ),
            RecoveryAllowance::Allowed
        );
    }

    #[test]
    fn test_recovery_governor_rejects_when_load_budget_is_exhausted() {
        let mut governor = RecoveryGovernor::default();
        let now = Instant::now();

        assert_eq!(
            governor.allow_attempt(Uuid::new_v4(), Duration::ZERO, 2, now),
            RecoveryAllowance::Allowed
        );
        assert_eq!(
            governor.allow_attempt(
                Uuid::new_v4(),
                Duration::ZERO,
                2,
                now + Duration::from_secs(1)
            ),
            RecoveryAllowance::Allowed
        );
        assert_eq!(
            governor.allow_attempt(
                Uuid::new_v4(),
                Duration::ZERO,
                2,
                now + Duration::from_secs(2)
            ),
            RecoveryAllowance::LoadBudget
        );
        assert_eq!(
            governor.allow_attempt(
                Uuid::new_v4(),
                Duration::ZERO,
                2,
                now + Duration::from_secs(61)
            ),
            RecoveryAllowance::Allowed
        );
    }

    #[test]
    fn test_checkpoint_resume_is_safe_without_kv_before_decode_begins() {
        let request = InferenceRequest::new(
            "test-network".to_string(),
            "llama-70b".to_string(),
            vec![1, 2, 3],
            "executor-1".to_string(),
        );
        let job = InferenceJob::new(request, 70);

        assert!(InferenceCoordinator::ensure_checkpoint_resume_safe(&job, false).is_ok());
    }

    #[test]
    fn test_checkpoint_resume_requires_kv_after_decode_progress() {
        let request = InferenceRequest::new(
            "test-network".to_string(),
            "llama-70b".to_string(),
            vec![1, 2, 3],
            "executor-1".to_string(),
        );
        let mut job = InferenceJob::new(request, 70);
        job.add_token(42);

        let error = InferenceCoordinator::ensure_checkpoint_resume_safe(&job, false).unwrap_err();
        assert!(error
            .to_string()
            .contains("cannot safely resume decode without KV state"));
        assert!(InferenceCoordinator::ensure_checkpoint_resume_safe(&job, true).is_ok());
    }

    #[test]
    fn test_build_next_decode_batch_admits_multiple_sessions() {
        let mut coordinator = test_coordinator(InferenceConfig {
            max_decode_batch_size: 4,
            max_decode_batch_kv_tokens: 1024,
            ..InferenceConfig::default()
        });
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 8, 11);
        insert_decode_session(&mut coordinator, session_b, 12, 22);

        coordinator.enqueue_decode_task(session_a).unwrap();
        coordinator.enqueue_decode_task(session_b).unwrap();

        let batch = coordinator.build_next_decode_batch(session_b, DecodeBatchTargets::default());
        assert_eq!(batch.slots.len(), 2);
        assert_eq!(batch.total_kv_tokens, 20);
        assert!(batch.deferred.is_empty());
        assert_eq!(batch.deferred_for_capacity, 0);
        assert_eq!(batch.deferred_for_kv_budget, 0);
    }

    #[test]
    fn test_build_next_decode_batch_defers_sessions_when_kv_budget_is_exhausted() {
        let mut coordinator = test_coordinator(InferenceConfig {
            max_decode_batch_size: 4,
            max_decode_batch_kv_tokens: 10,
            ..InferenceConfig::default()
        });
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        let session_c = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 4, 11);
        insert_decode_session(&mut coordinator, session_b, 5, 22);
        insert_decode_session(&mut coordinator, session_c, 6, 33);

        coordinator.enqueue_decode_task(session_a).unwrap();
        coordinator.enqueue_decode_task(session_b).unwrap();
        coordinator.enqueue_decode_task(session_c).unwrap();

        let batch = coordinator.build_next_decode_batch(session_b, DecodeBatchTargets::default());
        assert_eq!(batch.slots.len(), 2);
        assert_eq!(batch.total_kv_tokens, 9);
        assert_eq!(batch.deferred.len(), 1);
        assert_eq!(batch.deferred_for_capacity, 0);
        assert_eq!(batch.deferred_for_kv_budget, 1);
        assert_eq!(batch.deferred[0].session_id, session_c);
        assert_eq!(coordinator.decode_queue.len(), 1);
        assert_eq!(coordinator.decode_queue[0].session_id, session_c);
    }

    #[test]
    fn test_build_next_decode_batch_prefers_less_served_sessions() {
        let mut coordinator = test_coordinator(InferenceConfig {
            max_decode_batch_size: 1,
            max_decode_batch_kv_tokens: 1024,
            ..InferenceConfig::default()
        });
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 4, 11);
        insert_decode_session(&mut coordinator, session_b, 4, 22);
        coordinator
            .sessions
            .get_mut(&session_a)
            .expect("session a")
            .decode_steps_served = 5;
        coordinator
            .sessions
            .get_mut(&session_b)
            .expect("session b")
            .decode_steps_served = 1;

        coordinator.enqueue_decode_task(session_a).unwrap();
        coordinator.enqueue_decode_task(session_b).unwrap();

        let batch = coordinator.build_next_decode_batch(session_b, DecodeBatchTargets::default());
        assert_eq!(batch.slots.len(), 1);
        assert_eq!(batch.slots[0].session_id, session_b);
        assert_eq!(batch.deferred.len(), 1);
        assert_eq!(batch.deferred[0].session_id, session_a);
        assert_eq!(batch.deferred_for_capacity, 1);
    }

    #[test]
    fn test_build_next_decode_batch_honors_scheduler_batch_targets() {
        let mut coordinator = test_coordinator(InferenceConfig {
            max_decode_batch_size: 4,
            max_decode_batch_kv_tokens: 1024,
            ..InferenceConfig::default()
        });
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        let session_c = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 4, 11);
        insert_decode_session(&mut coordinator, session_b, 4, 22);
        insert_decode_session(&mut coordinator, session_c, 4, 33);

        coordinator.enqueue_decode_task(session_a).unwrap();
        coordinator.enqueue_decode_task(session_b).unwrap();
        coordinator.enqueue_decode_task(session_c).unwrap();

        let batch = coordinator.build_next_decode_batch(
            session_a,
            DecodeBatchTargets {
                target_session_count: Some(2),
                target_batch_size: Some(2),
            },
        );
        assert_eq!(batch.slots.len(), 2);
        assert_eq!(batch.slots[0].session_id, session_a);
        assert_eq!(batch.deferred.len(), 1);
        assert_eq!(batch.deferred_for_capacity, 1);
    }

    #[test]
    fn test_build_next_decode_batch_can_force_single_session_decode() {
        let mut coordinator = test_coordinator(InferenceConfig {
            max_decode_batch_size: 4,
            max_decode_batch_kv_tokens: 1024,
            ..InferenceConfig::default()
        });
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 4, 11);
        insert_decode_session(&mut coordinator, session_b, 4, 22);
        coordinator
            .sessions
            .get_mut(&session_b)
            .expect("session b")
            .decode_steps_served = 0;
        coordinator
            .sessions
            .get_mut(&session_a)
            .expect("session a")
            .decode_steps_served = 10;

        coordinator.enqueue_decode_task(session_a).unwrap();
        coordinator.enqueue_decode_task(session_b).unwrap();

        let batch = coordinator.build_next_decode_batch(
            session_a,
            DecodeBatchTargets {
                target_session_count: Some(1),
                target_batch_size: Some(1),
            },
        );
        assert_eq!(batch.slots.len(), 1);
        assert_eq!(batch.slots[0].session_id, session_a);
        assert_eq!(batch.deferred.len(), 1);
        assert_eq!(batch.deferred[0].session_id, session_b);
    }

    #[test]
    fn test_enforce_runtime_memory_budget_evicts_unprotected_session() {
        let mut coordinator = test_coordinator(InferenceConfig {
            max_active_sessions: 1,
            max_total_kv_cache_bytes: 8 * 1024,
            max_total_runtime_bytes: 16 * 1024,
            ..InferenceConfig::default()
        });
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 6, 11);
        insert_decode_session(&mut coordinator, session_b, 6, 22);
        coordinator
            .sessions
            .get_mut(&session_a)
            .expect("session a")
            .last_active_at = Instant::now() - Duration::from_secs(120);

        tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(async {
                coordinator
                    .enforce_runtime_memory_budget(Some(session_b))
                    .await
                    .expect("memory budget enforcement should succeed");
            });

        assert!(!coordinator.sessions.contains_key(&session_a));
        assert!(coordinator.sessions.contains_key(&session_b));
    }

    #[test]
    fn test_pause_and_resume_session_updates_runtime_status() {
        let mut coordinator = test_coordinator(InferenceConfig::default());
        let session_id = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_id, 4, 11);
        coordinator.enqueue_decode_task(session_id).unwrap();

        coordinator.pause_session(
            session_id,
            SessionPauseReason::Failover,
            "neighbor lost".to_string(),
        );
        let paused = coordinator
            .sessions
            .get(&session_id)
            .expect("paused session");
        assert!(matches!(
            paused.engine_state.runtime_status,
            SessionRuntimeStatus::Paused
        ));
        assert!(paused.engine_state.paused.is_some());
        assert!(coordinator.decode_queue.is_empty());

        coordinator
            .resume_session(session_id)
            .expect("resume session");
        let resumed = coordinator
            .sessions
            .get(&session_id)
            .expect("resumed session");
        assert!(matches!(
            resumed.engine_state.runtime_status,
            SessionRuntimeStatus::Active
        ));
        assert!(resumed.engine_state.paused.is_none());
    }

    #[test]
    fn test_runtime_memory_snapshot_counts_runtime_and_kv_bytes() {
        let mut coordinator = test_coordinator(InferenceConfig::default());
        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();
        insert_decode_session(&mut coordinator, session_a, 4, 11);
        insert_decode_session(&mut coordinator, session_b, 8, 22);

        let snapshot = coordinator.runtime_memory_snapshot();
        assert_eq!(snapshot.active_sessions, 2);
        assert_eq!(snapshot.total_kv_cache_bytes, (4 + 8) * 1024);
        assert!(snapshot.total_runtime_bytes >= snapshot.total_kv_cache_bytes);
    }
}
