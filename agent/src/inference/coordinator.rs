//! Inference coordinator for pipeline-parallel distributed inference
//!
//! The InferenceCoordinator manages the lifecycle of inference jobs across
//! a pipeline of workers. It handles:
//!
//! - Receiving inference requests from the control plane
//! - Coordinating pipeline-parallel forward passes across workers
//! - Managing activation passing between pipeline stages
//! - Ring reconfiguration on worker failure
//! - Checkpointing for fault tolerance
//! - Returning results to the control plane

use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::WorkerRing;
use crate::inference::tensor_ops::Tensor2D;
use crate::model::registry::ShardRegistry;
use crate::model::shard::ShardAssignment;
use crate::network::MeshSwarm;
use libp2p::PeerId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::forward_pass::{ForwardPass, ModelWeights};
use super::job::{InferenceJob, InferenceRequest, InferenceResult};
use super::mock_loader::{MockShardLoader, ShardLoader};
use super::stats::InferenceStats;

/// Configuration for the inference coordinator
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Total number of layers in the model
    pub total_layers: u32,

    /// Model identifier
    pub model_id: String,

    /// Timeout for pipeline stage communication
    pub pipeline_timeout: Duration,

    /// Timeout for entire inference job
    pub job_timeout: Duration,

    /// Whether checkpointing is enabled
    pub checkpointing_enabled: bool,

    /// Directory for checkpoint storage
    pub checkpoint_dir: std::path::PathBuf,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            total_layers: 80, // LLaMA 70B
            model_id: "llama-70b".to_string(),
            pipeline_timeout: Duration::from_secs(30),
            job_timeout: Duration::from_secs(300), // 5 minutes
            checkpointing_enabled: true,
            checkpoint_dir: dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".meshnet")
                .join("checkpoints"),
        }
    }
}

/// Pipeline position and shard information for this worker
#[derive(Debug, Clone)]
pub struct WorkerPosition {
    /// This worker's position in the pipeline (0-indexed)
    pub position: u32,

    /// Total number of workers in the pipeline
    pub total_workers: u32,

    /// Left neighbor peer ID (previous stage, sends activations to us)
    pub left_neighbor: PeerId,

    /// Right neighbor peer ID (next stage, we send activations to)
    pub right_neighbor: PeerId,

    /// Layer range this worker is responsible for [start, end)
    pub shard_layer_range: (u32, u32),

    /// Model shard memory usage in bytes
    pub shard_memory_bytes: u64,
}

impl WorkerPosition {
    /// Whether this is the first stage in the pipeline (runs embedding)
    pub fn is_first_stage(&self) -> bool {
        self.position == 0
    }

    /// Whether this is the last stage in the pipeline (runs lm_head + sampling)
    pub fn is_last_stage(&self) -> bool {
        self.position == self.total_workers - 1
    }

    /// Number of layers this worker is responsible for
    pub fn num_layers(&self) -> u32 {
        self.shard_layer_range.1 - self.shard_layer_range.0
    }
}

/// The main inference coordinator
pub struct InferenceCoordinator {
    /// Network swarm for P2P communication
    swarm: MeshSwarm,

    /// Inference configuration
    config: InferenceConfig,

    /// This worker's position in the pipeline
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
}

impl InferenceCoordinator {
    /// Create a new inference coordinator
    pub fn new(swarm: MeshSwarm, config: InferenceConfig) -> Self {
        // Initialize shard registry with default path
        let shard_registry = Arc::new(
            ShardRegistry::with_defaults()
                .expect("Failed to create shard registry"),
        );

        // Initialize mock loader (replace with real loader for production)
        let loader: Arc<dyn ShardLoader> = Arc::new(MockShardLoader::with_defaults());

        Self {
            swarm,
            config,
            position: None,
            stats: Arc::new(InferenceStats::new()),
            active_job: RwLock::new(None),
            checkpoint_manager: None,
            shard_registry,
            loader,
            weight_cache: RwLock::new(HashMap::new()),
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

    /// Check if this worker has joined a pipeline
    pub fn is_in_ring(&self) -> bool {
        self.position.is_some()
    }

    /// Get worker position information
    pub fn worker_position(&self) -> Option<&WorkerPosition> {
        self.position.as_ref()
    }

    /// Get or load model weights for the current worker position
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

        // Create shard assignment for this worker (layer-based)
        let assignment = ShardAssignment::with_layers(
            model_id.to_string(),
            position.position,
            position.total_workers,
            self.config.total_layers,
        );

        // Assign shard to registry if not already assigned
        if self.shard_registry.get_shard(model_id).await.is_none() {
            self.shard_registry
                .assign_shard(assignment.clone())
                .await?;
        }

        // Load from loader
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
            layers = format!("{}-{}", position.shard_layer_range.0, position.shard_layer_range.1),
            "Weights loaded and cached"
        );

        Ok(weights_arc)
    }

    /// Join the pipeline topology
    ///
    /// This sets up the worker's position and neighbors for pipeline parallelism.
    #[instrument(skip(self))]
    pub fn join_ring(&mut self, position: WorkerPosition) -> Result<()> {
        info!(
            position = position.position,
            total_workers = position.total_workers,
            left = %position.left_neighbor,
            right = %position.right_neighbor,
            layer_range = ?position.shard_layer_range,
            is_first = position.is_first_stage(),
            is_last = position.is_last_stage(),
            "Joining pipeline topology"
        );

        // Set ring neighbors on the swarm
        self.swarm
            .set_ring_neighbors(position.left_neighbor, position.right_neighbor);

        self.position = Some(position);

        info!("Successfully joined pipeline");
        Ok(())
    }

    /// Leave the pipeline topology
    #[instrument(skip(self))]
    pub fn leave_ring(&mut self) {
        if self.position.is_some() {
            info!("Leaving pipeline topology");
            self.swarm.clear_ring_neighbors();
            self.position = None;
        }
    }

    /// Process an inference request
    ///
    /// This is the main entry point for inference. It:
    /// 1. Creates an InferenceJob from the request
    /// 2. Runs the token generation loop using pipeline parallelism
    /// 3. Returns the result
    #[instrument(skip(self, request), fields(job_id = %request.job_id))]
    pub async fn process_inference(&mut self, request: InferenceRequest) -> Result<InferenceResult> {
        // Check if we're in a pipeline and clone position to avoid borrow issues
        let position = self.position.clone().ok_or_else(|| {
            AgentError::Execution("Worker is not part of a pipeline topology".to_string())
        })?;

        info!(
            job_id = %request.job_id,
            model = %request.model_id,
            prompt_tokens = request.prompt_tokens.len(),
            max_tokens = request.config.max_tokens,
            stage = format!("{}/{}", position.position, position.total_workers),
            layers = format!("{}-{}", position.shard_layer_range.0, position.shard_layer_range.1),
            "Starting inference job"
        );

        let start = Instant::now();

        // Create inference job
        let mut job = InferenceJob::new(request.clone(), self.config.total_layers);

        // Store as active job
        {
            let mut active = self.active_job.write().await;
            *active = Some(job.clone());
        }

        // Run generation loop
        let result = self.run_generation_loop(&mut job, &position).await;

        // Clear active job
        {
            let mut active = self.active_job.write().await;
            *active = None;
        }

        let execution_time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(()) => {
                let result = job.into_result();
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
                let result = InferenceResult::failure(request.job_id, e.to_string(), execution_time_ms);
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
    /// Pipeline parallelism flow per token:
    ///
    /// ```text
    /// Stage 0 (first):  embed(tokens) → forward_stage() → send_activations()
    /// Stage k (middle):                  recv_activations() → forward_stage() → send_activations()
    /// Stage N-1 (last):                  recv_activations() → forward_stage() → logits → sample → broadcast_token()
    /// All non-last:                      recv_token_broadcast()
    /// ```
    async fn run_generation_loop(
        &mut self,
        job: &mut InferenceJob,
        position: &WorkerPosition,
    ) -> Result<()> {
        let max_tokens = job.request.config.max_tokens;

        debug!(
            job_id = %job.request.job_id,
            max_tokens = max_tokens,
            stage = position.position,
            "Starting pipeline token generation loop"
        );

        // Generate tokens
        while !job.is_complete() {
            // Check for timeout
            if job.elapsed() > self.config.job_timeout {
                return Err(AgentError::Execution("Inference job timed out".to_string()));
            }

            // Generate next token via pipeline-parallel forward pass
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

    /// Generate the next token using pipeline-parallel forward pass
    ///
    /// Pipeline flow:
    /// - First stage: embed tokens, run local layers, send activations to next stage
    /// - Middle stages: receive activations, run local layers, send to next stage
    /// - Last stage: receive activations, run local layers, compute logits, sample, broadcast token
    /// - All stages: receive the broadcasted token
    async fn generate_next_token(
        &mut self,
        job: &mut InferenceJob,
        position: &WorkerPosition,
    ) -> Result<u32> {
        let start = Instant::now();
        let model_id = self.config.model_id.clone();
        let pipeline_timeout = self.config.pipeline_timeout;
        let job_id = job.request.job_id;
        let token_idx = job.current_token_idx as u32;

        // Load weights (cached across tokens)
        let weights = self.get_or_load_weights(&model_id, position).await?;

        // Create WorkerRing for pipeline communication
        let mut worker_ring = WorkerRing::new(
            position.position,
            position.total_workers,
            position.left_neighbor,
            position.right_neighbor,
            self.swarm_mut(),
        );

        // Create ForwardPass for this worker's pipeline stage
        let mut forward_pass = ForwardPass::new_pipeline_stage(
            (*weights).clone(),
            position.shard_layer_range.0 as usize,
            position.shard_layer_range.1 as usize,
            position.position,
            position.total_workers,
        );

        // Build full token sequence (prompt + generated so far)
        let mut all_tokens = job.request.prompt_tokens.clone();
        all_tokens.extend(&job.generated_tokens);

        debug!(
            job_id = %job_id,
            stage = position.position,
            total_tokens = all_tokens.len(),
            "Running pipeline forward pass"
        );

        let next_token = if position.is_first_stage() {
            // === FIRST STAGE ===
            // 1. Embed tokens into hidden states
            // 2. Run local layers
            // 3. Send activations to next stage
            // 4. Wait for token broadcast from last stage

            let embedded = forward_pass.embed(&all_tokens)?;
            let hidden = forward_pass.forward_stage(&embedded)?;

            if position.total_workers > 1 {
                // Send activations to next stage
                let shape = vec![hidden.rows, hidden.cols];
                worker_ring
                    .send_activations(
                        job_id,
                        position.shard_layer_range.1 - 1,
                        token_idx,
                        hidden.data.clone(),
                        shape,
                    )
                    .await?;

                // Wait for token broadcast from last stage
                worker_ring.recv_token_broadcast(job_id, token_idx, pipeline_timeout).await?
            } else {
                // Single worker: we are both first and last stage
                let logits = forward_pass.compute_logits(&hidden)?;
                let seed = job_id.as_u128() as u64 ^ token_idx as u64;
                forward_pass.sample(&logits, job.request.config.temperature, job.request.config.top_p, seed)
            }
        } else if position.is_last_stage() {
            // === LAST STAGE ===
            // 1. Receive activations from previous stage
            // 2. Run local layers
            // 3. Compute logits and sample
            // 4. Broadcast token to all stages

            let (recv_data, recv_shape, _seq) = worker_ring
                .recv_activations(job_id, token_idx, pipeline_timeout)
                .await?;

            // Reconstruct Tensor2D from received activation data
            let rows = recv_shape.first().copied().unwrap_or(1);
            let cols = recv_shape.get(1).copied().unwrap_or(recv_data.len());
            let input_hidden = Tensor2D::new(recv_data, rows, cols)?;

            let hidden = forward_pass.forward_stage(&input_hidden)?;

            // Compute logits and sample
            let logits = forward_pass.compute_logits(&hidden)?;
            let seed = job_id.as_u128() as u64 ^ token_idx as u64;
            let token = forward_pass.sample(
                &logits,
                job.request.config.temperature,
                job.request.config.top_p,
                seed,
            );

            // Broadcast token to all other stages
            worker_ring
                .broadcast_token(job_id, token_idx, token)
                .await?;

            token
        } else {
            // === MIDDLE STAGE ===
            // 1. Receive activations from previous stage
            // 2. Run local layers
            // 3. Send activations to next stage
            // 4. Wait for token broadcast from last stage

            let (recv_data, recv_shape, _seq) = worker_ring
                .recv_activations(job_id, token_idx, pipeline_timeout)
                .await?;

            let rows = recv_shape.first().copied().unwrap_or(1);
            let cols = recv_shape.get(1).copied().unwrap_or(recv_data.len());
            let input_hidden = Tensor2D::new(recv_data, rows, cols)?;

            let hidden = forward_pass.forward_stage(&input_hidden)?;

            // Send to next stage
            let shape = vec![hidden.rows, hidden.cols];
            worker_ring
                .send_activations(
                    job_id,
                    position.shard_layer_range.1 - 1,
                    token_idx,
                    hidden.data.clone(),
                    shape,
                )
                .await?;

            // Wait for token broadcast
            worker_ring.recv_token_broadcast(job_id, token_idx, pipeline_timeout).await?
        };

        let elapsed = start.elapsed().as_millis() as u64;
        debug!(
            job_id = %job_id,
            next_token = next_token,
            elapsed_ms = elapsed,
            stage = position.position,
            "Pipeline stage completed"
        );

        // Record statistics
        self.stats.record_allreduce(elapsed);
        for _ in 0..position.num_layers() {
            self.stats.record_layer();
        }

        Ok(next_token)
    }

    /// Check if generation should stop
    fn should_stop(&self, job: &InferenceJob) -> bool {
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

            manager.save_checkpoint(job).await?;
            self.stats.record_checkpoint();
        }
        Ok(())
    }

    /// Attempt to recover a job from checkpoint
    #[instrument(skip(self))]
    pub async fn recover_from_checkpoint(
        &mut self,
        job_id: Uuid,
    ) -> Result<Option<InferenceJob>> {
        if let Some(ref manager) = self.checkpoint_manager {
            if let Some(job) = manager.load_checkpoint(job_id).await? {
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

    /// Run the coordinator event loop
    #[instrument(skip(self), fields(peer_id = %self.swarm.local_peer_id()))]
    pub async fn run(mut self) -> Result<()> {
        use tokio::signal;

        info!("Starting inference coordinator (pipeline-parallel mode)");

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
        self.stats.print_summary();
        let _ = self.stats.save_to_file();

        Ok(())
    }

    /// Handle a mesh network event
    async fn handle_event(&mut self, event: crate::network::MeshEvent) -> Result<()> {
        use crate::network::MeshEvent;

        match event {
            MeshEvent::TensorReceived {
                peer_id,
                tensor,
                channel,
            } => {
                debug!(
                    peer_id = %peer_id,
                    job_id = %tensor.job_id,
                    layer = tensor.layer_idx,
                    phase = ?tensor.phase,
                    stage = tensor.sender_stage,
                    seq = tensor.sequence_num,
                    "Received pipeline message"
                );

                // Verify checksum on received activation data
                if !tensor.verify_checksum() {
                    error!(
                        peer_id = %peer_id,
                        job_id = %tensor.job_id,
                        "DIVERGENCE DETECTED: activation checksum mismatch!"
                    );
                }

                // Acknowledge
                let ack = tensor.clone();
                self.swarm.respond_to_tensor(channel, ack)?;
            }

            MeshEvent::TensorSendFailed { peer_id, error } => {
                warn!(
                    peer_id = %peer_id,
                    error = %error,
                    "Failed to send pipeline activation"
                );
            }

            MeshEvent::PeerDisconnected { peer_id } => {
                if let Some(pos) = &self.position {
                    if peer_id == pos.left_neighbor || peer_id == pos.right_neighbor {
                        error!(
                            peer_id = %peer_id,
                            stage = pos.position,
                            "Pipeline neighbor disconnected! Triggering reconfiguration."
                        );

                        // Abort active job — pipeline is broken
                        {
                            let mut active = self.active_job.write().await;
                            if let Some(ref job) = *active {
                                error!(
                                    job_id = %job.request.job_id,
                                    "Aborting active job due to pipeline break"
                                );
                            }
                            *active = None;
                        }

                        // In production: notify control plane to reconfigure pipeline
                        // The control plane will reassign layers and re-form the pipeline
                        warn!("Pipeline broken — waiting for control plane to reconfigure");
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
        assert_eq!(config.total_layers, 80);
        assert_eq!(config.model_id, "llama-70b");
        assert!(config.checkpointing_enabled);
    }

    #[test]
    fn test_worker_position() {
        let position = WorkerPosition {
            position: 1,
            total_workers: 4,
            left_neighbor: PeerId::random(),
            right_neighbor: PeerId::random(),
            shard_layer_range: (20, 40),
            shard_memory_bytes: 7_000_000_000,
        };

        assert_eq!(position.position, 1);
        assert_eq!(position.total_workers, 4);
        assert_eq!(position.num_layers(), 20);
        assert!(!position.is_first_stage());
        assert!(!position.is_last_stage());
    }

    #[test]
    fn test_worker_position_first_stage() {
        let position = WorkerPosition {
            position: 0,
            total_workers: 3,
            left_neighbor: PeerId::random(),
            right_neighbor: PeerId::random(),
            shard_layer_range: (0, 27),
            shard_memory_bytes: 7_000_000_000,
        };

        assert!(position.is_first_stage());
        assert!(!position.is_last_stage());
    }

    #[test]
    fn test_worker_position_last_stage() {
        let position = WorkerPosition {
            position: 2,
            total_workers: 3,
            left_neighbor: PeerId::random(),
            right_neighbor: PeerId::random(),
            shard_layer_range: (54, 80),
            shard_memory_bytes: 7_000_000_000,
        };

        assert!(!position.is_first_stage());
        assert!(position.is_last_stage());
    }
}
