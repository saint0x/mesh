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
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::artifact_loader::{ArtifactShardLoader, ShardLoader};
use super::forward_pass::{ForwardPass, ModelWeights};
use super::job::{InferenceJob, InferenceRequest, InferenceResult};
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryAllowance {
    Allowed,
    Cooldown,
    LoadBudget,
}

#[derive(Debug, Default)]
struct RecoveryGovernor {
    last_attempts_by_job: HashMap<Uuid, Instant>,
    checkpoint_loads: VecDeque<Instant>,
}

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

    /// Shard loader (mock or real)
    loader: Arc<dyn ShardLoader>,

    /// Cached weights per model (model_id -> weights)
    weight_cache: RwLock<HashMap<String, Arc<ModelWeights>>>,

    /// Active forward pass state for the currently executing job
    active_forward_pass: Option<ForwardPass>,

    /// Governs checkpoint recovery cadence and node-level recovery load.
    recovery_governor: RecoveryGovernor,
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
            active_forward_pass: None,
            recovery_governor: RecoveryGovernor::default(),
        }
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
    /// it loads them using the shard loader (which simulates download/load
    /// in the mock implementation, or does real loading in production).
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
        let assignment = ShardAssignment::new(
            model_id.to_string(),
            position.position,
            position.total_workers,
        );

        // Assign shard to registry if not already assigned
        if self.shard_registry.get_shard(model_id).await.is_none() {
            self.shard_registry.assign_shard(assignment.clone()).await?;
        }

        // Load from loader (simulated download + load for mock, real for production)
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
        // Check if we're in a ring and clone position to avoid borrow issues
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

        // Create inference job
        let mut job = InferenceJob::new(request.clone(), self.config.total_layers);
        let mut recovered_from_checkpoint = false;
        let mut recovery_attempts = 0;

        // Store as active job
        {
            let mut active = self.active_job.write().await;
            *active = Some(job.clone());
        }

        // Run generation loop with bounded checkpoint recovery.
        let result = loop {
            match self.run_generation_loop(&mut job, &position).await {
                Ok(()) => break Ok(()),
                Err(error) => {
                    if !self.config.checkpointing_enabled
                        || recovery_attempts >= self.config.recovery_max_attempts_per_job
                    {
                        break Err(error);
                    }

                    match self.try_recover_job(job.request.job_id).await? {
                        Some(recovered_job) => {
                            recovery_attempts += 1;
                            recovered_from_checkpoint = true;
                            job = recovered_job;
                            let mut active = self.active_job.write().await;
                            *active = Some(job.clone());
                            continue;
                        }
                        None => break Err(error),
                    }
                }
            }
        };

        // Clear active job
        {
            let mut active = self.active_job.write().await;
            *active = None;
        }

        let execution_time_ms = start.elapsed().as_millis() as u64;
        self.active_forward_pass = None;

        match result {
            Ok(()) => {
                let mut result = job.into_result();
                if recovered_from_checkpoint {
                    result = result.with_recovery();
                }
                self.stats.record_success(
                    result.prompt_tokens,
                    result.completion_tokens,
                    execution_time_ms,
                );
                info!(
                    job_id = %request.job_id,
                    tokens = result.completion_tokens,
                    execution_time_ms = execution_time_ms,
                    "Inference job completed"
                );
                Ok(result)
            }
            Err(e) => {
                self.stats.record_failure();
                let result =
                    InferenceResult::failure(request.job_id, e.to_string(), execution_time_ms);
                error!(
                    job_id = %request.job_id,
                    error = %e,
                    "Inference job failed"
                );
                Ok(result)
            }
        }
    }

    /// Run the token generation loop
    ///
    /// This generates tokens one at a time until max_tokens is reached
    /// or a stop sequence is encountered.
    async fn run_generation_loop(
        &mut self,
        job: &mut InferenceJob,
        position: &WorkerPosition,
    ) -> Result<()> {
        let max_tokens = job.request.config.max_tokens;

        debug!(
            job_id = %job.request.job_id,
            max_tokens = max_tokens,
            "Starting token generation loop"
        );

        // Generate tokens
        while !job.is_complete() {
            // Check for timeout
            if job.elapsed() > self.config.job_timeout {
                return Err(AgentError::Execution("Inference job timed out".to_string()));
            }

            // Generate next token via tensor-parallel forward pass
            // NOTE: WorkerRing is created per-token inside generate_next_token
            let next_token = self.generate_next_token(job, position).await?;

            // Add token to job
            job.add_token(next_token);

            debug!(
                job_id = %job.request.job_id,
                token_idx = job.current_token_idx,
                token = next_token,
                progress = format!("{:.1}%", job.progress()),
                "Generated token"
            );

            // Check for stop sequences
            if self.should_stop(job) {
                debug!(job_id = %job.request.job_id, "Stop sequence encountered");
                break;
            }

            // Checkpoint if needed
            if self.config.checkpointing_enabled && job.should_checkpoint() {
                self.checkpoint(job).await?;
                job.mark_checkpointed();
            }
        }

        Ok(())
    }

    /// Generate the next token using tensor-parallel forward pass
    ///
    /// This is where the actual distributed computation happens:
    /// 1. Load model weights (cached across tokens)
    /// 2. Each worker computes partial matmul for their columns
    /// 3. Ring all-reduce combines results
    /// 4. All workers have identical activations
    /// 5. Final layer produces logits
    /// 6. Sample next token
    async fn generate_next_token(
        &mut self,
        job: &mut InferenceJob,
        position: &WorkerPosition,
    ) -> Result<u32> {
        let start = Instant::now();

        // Clone model_id to avoid borrow issues
        let model_id = self.config.model_id.clone();

        // Load weights using the verified artifact loader
        debug!(
            job_id = %job.request.job_id,
            position = position.position,
            model_id = %model_id,
            "Loading model weights"
        );

        let weights = self.get_or_load_weights(&model_id, position).await?;

        if self.active_forward_pass.is_none() {
            self.active_forward_pass = Some(ForwardPass::new(
                (*weights).clone(),
                position.shard_column_range.0 as usize,
                position.shard_column_range.1 as usize,
                position.total_workers,
            ));
        }

        let mut forward_pass = self
            .active_forward_pass
            .take()
            .expect("active forward pass must be initialized");

        // Build full token sequence (prompt + generated so far)
        let mut all_tokens = job.request.prompt_tokens.clone();
        all_tokens.extend(&job.generated_tokens);

        debug!(
            job_id = %job.request.job_id,
            total_tokens = all_tokens.len(),
            "Running forward pass with ring all-reduce"
        );

        // Run forward pass with ring all-reduce
        // WorkerRing created per-token (lightweight metadata structure)
        let next_token = {
            let mut worker_ring = WorkerRing::new(
                position.position,
                position.total_workers,
                position.left_neighbor,
                position.right_neighbor,
                position.left_neighbor_tensor_addr,
                position.right_neighbor_tensor_addr,
                self.tensor_plane_mut(),
            );

            forward_pass
                .generate_next_token(
                    &all_tokens,
                    &mut worker_ring,
                    job.request.job_id,
                    job.request.config.temperature,
                    job.request.config.top_p,
                )
                .await?
        };

        self.active_forward_pass = Some(forward_pass);

        let elapsed = start.elapsed().as_millis() as u64;
        debug!(
            job_id = %job.request.job_id,
            next_token = next_token,
            elapsed_ms = elapsed,
            "Generated next token"
        );

        // Record statistics
        self.stats.record_allreduce(elapsed);
        for _ in 0..weights.config.num_layers {
            self.stats.record_layer();
        }
        self.sync_tensor_plane_metrics();

        Ok(next_token)
    }

    /// Check if generation should stop
    fn should_stop(&self, job: &InferenceJob) -> bool {
        // Check stop sequences (would need to decode tokens to text)
        // For now, just check max tokens
        job.is_complete()
    }

    /// Create a checkpoint of the current inference state
    async fn checkpoint(&mut self, job: &InferenceJob) -> Result<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            debug!(
                job_id = %job.request.job_id,
                token_idx = job.current_token_idx,
                "Creating checkpoint"
            );

            let kv_cache = self.active_forward_pass.as_ref().map(|pass| &pass.kv_cache);
            manager.save_checkpoint(job, kv_cache).await?;
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
                    let weights = self
                        .get_or_load_weights(&job.request.model_id, &position)
                        .await?;

                    let mut forward_pass = ForwardPass::new(
                        (*weights).clone(),
                        position.shard_column_range.0 as usize,
                        position.shard_column_range.1 as usize,
                        position.total_workers,
                    );

                    if let Some(kv_cache) = manager.load_checkpoint_kv_cache(job_id).await? {
                        forward_pass.kv_cache = kv_cache;
                        forward_pass.position = job.current_token_idx as usize;
                    }

                    self.active_forward_pass = Some(forward_pass);
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
                        // In production, this would trigger recovery
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

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.total_layers, 70);
        assert_eq!(config.model_id, "llama-70b");
        assert!(config.checkpointing_enabled);
        assert_eq!(config.recovery_max_attempts_per_job, 2);
        assert_eq!(config.recovery_cooldown, Duration::from_secs(5));
        assert_eq!(config.recovery_max_checkpoint_loads_per_minute, 8);
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
            governor.allow_attempt(job_id, Duration::from_secs(5), 8, now + Duration::from_secs(1)),
            RecoveryAllowance::Cooldown
        );
        assert_eq!(
            governor.allow_attempt(job_id, Duration::from_secs(5), 8, now + Duration::from_secs(6)),
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
            governor.allow_attempt(Uuid::new_v4(), Duration::ZERO, 2, now + Duration::from_secs(1)),
            RecoveryAllowance::Allowed
        );
        assert_eq!(
            governor.allow_attempt(Uuid::new_v4(), Duration::ZERO, 2, now + Duration::from_secs(2)),
            RecoveryAllowance::LoadBudget
        );
        assert_eq!(
            governor.allow_attempt(Uuid::new_v4(), Duration::ZERO, 2, now + Duration::from_secs(61)),
            RecoveryAllowance::Allowed
        );
    }
}
