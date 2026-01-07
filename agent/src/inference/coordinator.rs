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

use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::WorkerRing;
use crate::network::MeshSwarm;
use libp2p::PeerId;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::forward_pass::{ForwardPass, ModelConfig};
use super::job::{InferenceJob, InferenceRequest, InferenceResult};
use super::mock_validation; // MOCK: For validation only - TODO: Remove when using real weights
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
        }
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

    /// Right neighbor peer ID
    pub right_neighbor: PeerId,

    /// Column range this worker is responsible for
    pub shard_column_range: (u32, u32),

    /// Model shard memory usage in bytes
    pub shard_memory_bytes: u64,
}

/// The main inference coordinator
pub struct InferenceCoordinator {
    /// Network swarm for P2P communication
    swarm: MeshSwarm,

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
}

impl InferenceCoordinator {
    /// Create a new inference coordinator
    pub fn new(swarm: MeshSwarm, config: InferenceConfig) -> Self {
        Self {
            swarm,
            config,
            position: None,
            stats: Arc::new(InferenceStats::new()),
            active_job: RwLock::new(None),
            checkpoint_manager: None,
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

    /// Check if this worker has joined a ring
    pub fn is_in_ring(&self) -> bool {
        self.position.is_some()
    }

    /// Get worker position information
    pub fn worker_position(&self) -> Option<&WorkerPosition> {
        self.position.as_ref()
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
        self.swarm
            .set_ring_neighbors(position.left_neighbor, position.right_neighbor);

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
    pub async fn process_inference(&mut self, request: InferenceRequest) -> Result<InferenceResult> {
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

        // Store as active job
        {
            let mut active = self.active_job.write().await;
            *active = Some(job.clone());
        }

        // Run generation loop
        // NOTE: WorkerRing is created per-token inside generate_next_token()
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
    /// MOCK: This implementation uses Xavier-initialized mock weights for validation.
    /// TODO: Replace with safetensors weight loading when ready.
    /// See agent/src/inference/mock_validation.rs for mock implementation details.
    ///
    /// This is where the actual distributed computation happens:
    /// 1. Each worker computes partial matmul for their columns
    /// 2. Ring all-reduce combines results
    /// 3. All workers have identical activations
    /// 4. Final layer produces logits
    /// 5. Sample next token
    async fn generate_next_token(
        &mut self,
        job: &mut InferenceJob,
        position: &WorkerPosition,
    ) -> Result<u32> {
        // MOCK: Using Xavier-initialized mock weights for validation
        // TODO: Replace with safetensors weight loading when ready
        // See agent/src/inference/mock_validation.rs for mock implementation details

        let start = Instant::now();

        // Create WorkerRing for this token generation
        // NOTE: WorkerRing is lightweight (just metadata), so creating per-token is fine
        let mut worker_ring = WorkerRing::new(
            position.position,
            position.total_workers,
            position.left_neighbor,
            position.right_neighbor,
            self.swarm_mut(),
        );

        // MOCK: Generate Xavier-initialized weights for validation
        // TODO: Replace this with actual safetensors loading:
        //   let weights = load_weights_from_safetensors(&self.config.model_path, position.shard_column_range)?;
        let config = ModelConfig::default();
        let shard_cols =
            (position.shard_column_range.1 - position.shard_column_range.0) as usize;

        debug!(
            job_id = %job.request.job_id,
            position = position.position,
            shard_cols = shard_cols,
            "Generating mock weights for validation"
        );

        // MOCK: Generate mock weights (deterministic from fixed seed)
        let mock_weights = mock_validation::generate_mock_weights(&config, shard_cols, 12345);

        // Create ForwardPass with mock weights
        // NOTE: ForwardPass is created per-token (KV cache/position changes)
        // TODO: In production, consider caching weights and reusing ForwardPass
        let mut forward_pass = ForwardPass::new(
            mock_weights,
            position.shard_column_range.0 as usize,
            position.shard_column_range.1 as usize,
            position.total_workers,
        );

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
        let next_token = forward_pass
            .generate_next_token(
                &all_tokens,
                &mut worker_ring,
                job.request.job_id,
                job.request.config.temperature,
                job.request.config.top_p,
            )
            .await?;

        let elapsed = start.elapsed().as_millis() as u64;
        debug!(
            job_id = %job.request.job_id,
            next_token = next_token,
            elapsed_ms = elapsed,
            "Generated next token"
        );

        // Record statistics
        self.stats.record_allreduce(elapsed);
        for _ in 0..config.num_layers {
            self.stats.record_layer();
        }

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
                // Handle tensor message from ring neighbor
                debug!(
                    peer_id = %peer_id,
                    job_id = %tensor.job_id,
                    layer = tensor.layer_idx,
                    phase = ?tensor.phase,
                    "Received tensor from ring neighbor"
                );

                // Acknowledge the tensor (echo back)
                let ack = tensor.clone();
                self.swarm.respond_to_tensor(channel, ack)?;
            }

            MeshEvent::TensorSendFailed { peer_id, error } => {
                warn!(
                    peer_id = %peer_id,
                    error = %error,
                    "Failed to send tensor"
                );
            }

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
    }

    #[test]
    fn test_worker_position() {
        let peer_id = PeerId::random();
        let position = WorkerPosition {
            position: 3,
            total_workers: 10,
            left_neighbor: PeerId::random(),
            right_neighbor: PeerId::random(),
            shard_column_range: (2457, 3276),
            shard_memory_bytes: 7_000_000_000,
        };

        assert_eq!(position.position, 3);
        assert_eq!(position.total_workers, 10);
    }
}
