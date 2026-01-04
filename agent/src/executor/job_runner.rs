//! Job execution runner
//!
//! This module provides the core job execution loop that integrates the mesh network
//! with the workload executors. It handles:
//!
//! - Receiving job requests from the network
//! - Dispatching jobs to the appropriate executor
//! - Sending job results back to requesters
//! - Tracking job statistics and state
//! - Graceful shutdown handling
//!
//! ## Architecture
//!
//! ```text
//! Network Events → JobRunner → Executor → Network Response
//!      ↓                                         ↑
//!  JobReceived                             JobResult
//! ```
//!
//! The JobRunner acts as the bridge between the networking layer (MeshSwarm)
//! and the execution layer (EmbeddingsExecutor). It receives MeshEvents from
//! the swarm, executes jobs using the executor, and sends results back through
//! the swarm.

use crate::executor::{EmbeddingsExecutor, EmbeddingsInput, EmbeddingsOutput};
use crate::network::{JobEnvelope, JobResult, MeshEvent, MeshSwarm};
use crate::errors::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::signal;
use tracing::{debug, error, info, instrument, warn};

/// Job execution statistics
///
/// Tracks metrics about job execution for monitoring and debugging.
#[derive(Debug, Default)]
pub struct JobStats {
    /// Total number of jobs successfully completed
    pub jobs_completed: AtomicU64,

    /// Total number of jobs that failed
    pub jobs_failed: AtomicU64,

    /// Total execution time across all jobs (milliseconds)
    pub total_execution_time_ms: AtomicU64,

    /// Number of jobs currently being executed
    pub active_jobs: AtomicU64,
}

impl JobStats {
    /// Create new job statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful job completion
    pub fn record_success(&self, execution_time_ms: u64) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);
    }

    /// Record a job failure
    pub fn record_failure(&self, execution_time_ms: u64) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);
    }

    /// Increment active job count
    pub fn start_job(&self) {
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active job count
    pub fn finish_job(&self) {
        self.active_jobs.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get total jobs processed (completed + failed)
    pub fn total_jobs(&self) -> u64 {
        self.jobs_completed.load(Ordering::Relaxed) + self.jobs_failed.load(Ordering::Relaxed)
    }

    /// Get average execution time (milliseconds)
    pub fn avg_execution_time_ms(&self) -> f64 {
        let total_jobs = self.total_jobs();
        if total_jobs == 0 {
            return 0.0;
        }

        let total_time = self.total_execution_time_ms.load(Ordering::Relaxed);
        total_time as f64 / total_jobs as f64
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total_jobs = self.total_jobs();
        if total_jobs == 0 {
            return 0.0;
        }

        let completed = self.jobs_completed.load(Ordering::Relaxed);
        completed as f64 / total_jobs as f64
    }

    /// Get number of active jobs
    pub fn active_jobs(&self) -> u64 {
        self.active_jobs.load(Ordering::Relaxed)
    }

    /// Print statistics summary
    pub fn print_summary(&self) {
        let total = self.total_jobs();
        let completed = self.jobs_completed.load(Ordering::Relaxed);
        let failed = self.jobs_failed.load(Ordering::Relaxed);
        let active = self.active_jobs();
        let avg_time = self.avg_execution_time_ms();
        let success_rate = self.success_rate() * 100.0;

        info!(
            total_jobs = total,
            completed = completed,
            failed = failed,
            active = active,
            avg_execution_time_ms = format!("{:.2}", avg_time),
            success_rate = format!("{:.1}%", success_rate),
            "Job statistics"
        );
    }
}

/// Job execution runner
///
/// Manages the event loop that receives jobs from the network, executes them,
/// and sends results back. This is the main component that integrates the
/// networking and execution layers.
pub struct JobRunner {
    swarm: MeshSwarm,
    executor: EmbeddingsExecutor,
    stats: Arc<JobStats>,
    shutdown_on_idle: bool,
}

impl JobRunner {
    /// Create a new job runner
    ///
    /// # Arguments
    /// * `swarm` - Mesh network swarm for communication
    /// * `executor` - Executor for processing jobs
    ///
    /// # Example
    /// ```no_run
    /// use agent::{MeshSwarmBuilder, EmbeddingsExecutor, JobRunner};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let keypair = libp2p::identity::Keypair::generate_ed25519();
    /// let swarm = MeshSwarmBuilder::new(keypair).build()?;
    /// let executor = EmbeddingsExecutor::new()?;
    ///
    /// let runner = JobRunner::new(swarm, executor);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(swarm: MeshSwarm, executor: EmbeddingsExecutor) -> Self {
        Self {
            swarm,
            executor,
            stats: Arc::new(JobStats::new()),
            shutdown_on_idle: false,
        }
    }

    /// Enable shutdown when idle (for testing/ephemeral jobs)
    pub fn with_shutdown_on_idle(mut self) -> Self {
        self.shutdown_on_idle = true;
        self
    }

    /// Get reference to job statistics
    pub fn stats(&self) -> &Arc<JobStats> {
        &self.stats
    }

    /// Get reference to the swarm
    pub fn swarm(&self) -> &MeshSwarm {
        &self.swarm
    }

    /// Get mutable reference to the swarm
    pub fn swarm_mut(&mut self) -> &mut MeshSwarm {
        &mut self.swarm
    }

    /// Run the job execution loop
    ///
    /// This is the main event loop that processes network events and executes jobs.
    /// It runs until a shutdown signal (Ctrl+C) is received or an error occurs.
    ///
    /// # Returns
    /// * `Ok(())` - Clean shutdown
    /// * `Err(AgentError)` - Fatal error occurred
    ///
    /// # Example
    /// ```no_run
    /// # use agent::{MeshSwarmBuilder, EmbeddingsExecutor, JobRunner};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let keypair = libp2p::identity::Keypair::generate_ed25519();
    /// # let swarm = MeshSwarmBuilder::new(keypair).build()?;
    /// # let executor = EmbeddingsExecutor::new()?;
    /// let runner = JobRunner::new(swarm, executor);
    /// runner.run().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), fields(local_peer = %self.swarm.local_peer_id()))]
    pub async fn run(mut self) -> Result<()> {
        info!("Starting job execution loop");

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

                // Handle shutdown signal
                _ = signal::ctrl_c() => {
                    info!("Received shutdown signal (Ctrl+C)");
                    break;
                }
            }
        }

        // Print final statistics
        info!("Shutting down job runner");
        self.stats.print_summary();

        Ok(())
    }

    /// Handle a mesh network event
    ///
    /// Processes different types of network events, with special handling for
    /// job-related events (JobReceived, JobCompleted).
    #[instrument(skip(self, event), fields(event_type = std::any::type_name::<MeshEvent>()))]
    async fn handle_event(&mut self, event: MeshEvent) -> Result<()> {
        match event {
            MeshEvent::JobReceived {
                peer_id,
                job,
                channel,
            } => {
                self.handle_job_received(peer_id, job, channel).await?;
            }

            MeshEvent::JobCompleted { peer_id, result } => {
                self.handle_job_completed(peer_id, result).await?;
            }

            MeshEvent::JobSendFailed {
                peer_id,
                job_id,
                error,
            } => {
                warn!(
                    peer_id = %peer_id,
                    job_id = %job_id,
                    error = %error,
                    "Job send failed"
                );
            }

            MeshEvent::PeerConnected {
                peer_id,
                connection_info,
            } => {
                info!(
                    peer_id = %peer_id,
                    connection_type = ?connection_info.connection_type,
                    "Peer connected"
                );
            }

            MeshEvent::PeerDisconnected { peer_id } => {
                info!(peer_id = %peer_id, "Peer disconnected");
            }

            MeshEvent::RelayConnected {
                relay_peer_id,
                relay_addr,
            } => {
                info!(
                    relay_peer_id = %relay_peer_id,
                    relay_addr = %relay_addr,
                    "Connected to relay server"
                );
            }

            MeshEvent::ReservationAccepted {
                relay_peer_id,
                renewal_timeout,
            } => {
                info!(
                    relay_peer_id = %relay_peer_id,
                    renewal_timeout_secs = renewal_timeout.as_secs(),
                    "Relay reservation accepted"
                );
            }

            MeshEvent::ReservationDenied { relay_peer_id } => {
                warn!(
                    relay_peer_id = %relay_peer_id,
                    "Relay reservation denied"
                );
            }

            _ => {
                debug!(event = ?event, "Unhandled event");
            }
        }

        Ok(())
    }

    /// Handle a received job request
    ///
    /// Executes the job and sends the result back to the requester.
    #[instrument(
        skip(self, job, channel),
        fields(
            peer_id = %peer_id,
            job_id = %job.job_id,
            workload = %job.workload_id
        )
    )]
    async fn handle_job_received(
        &mut self,
        peer_id: libp2p::PeerId,
        job: JobEnvelope,
        channel: libp2p::request_response::ResponseChannel<JobResult>,
    ) -> Result<()> {
        info!("Received job from peer");
        self.stats.start_job();

        let start = Instant::now();
        let job_id = job.job_id;

        // Execute job and create result
        let result = self.execute_job(job).await;

        // Update statistics
        let execution_time_ms = start.elapsed().as_millis() as u64;
        if result.success {
            self.stats.record_success(execution_time_ms);
        } else {
            self.stats.record_failure(execution_time_ms);
        }
        self.stats.finish_job();

        // Send response
        debug!(
            job_id = %job_id,
            success = result.success,
            execution_time_ms = execution_time_ms,
            "Sending job result"
        );

        self.swarm.respond_to_job(channel, result)?;

        Ok(())
    }

    /// Handle a completed job result
    ///
    /// Processes results received from peers (when this node submitted a job).
    #[instrument(
        skip(self, result),
        fields(
            peer_id = %peer_id,
            job_id = %result.job_id
        )
    )]
    async fn handle_job_completed(
        &mut self,
        peer_id: libp2p::PeerId,
        result: JobResult,
    ) -> Result<()> {
        info!(
            success = result.success,
            execution_time_ms = result.execution_time_ms,
            "Job completed"
        );

        if let Some(result_bytes) = &result.result {
            match EmbeddingsOutput::from_cbor(result_bytes) {
                Ok(output) => {
                    info!(
                        dimensions = output.dimensions,
                        model = %output.model,
                        "Received embedding result"
                    );

                    // Print summary of embedding (first few values)
                    let preview: Vec<f32> = output.embedding.iter().take(5).copied().collect();
                    println!(
                        "\n✓ Job {} completed in {}ms",
                        result.job_id, result.execution_time_ms
                    );
                    println!(
                        "  Embedding: [{:.3}, {:.3}, {:.3}, ...] ({} dims)",
                        preview.get(0).unwrap_or(&0.0),
                        preview.get(1).unwrap_or(&0.0),
                        preview.get(2).unwrap_or(&0.0),
                        output.dimensions
                    );
                }
                Err(e) => {
                    warn!(error = %e, "Failed to deserialize embedding output");
                }
            }
        } else if let Some(error) = &result.error {
            error!(error = %error, "Job failed with error");
            println!("\n✗ Job {} failed: {}", result.job_id, error);
        }

        Ok(())
    }

    /// Execute a job using the appropriate executor
    ///
    /// Deserializes the job payload, executes it using the executor,
    /// and serializes the result.
    #[instrument(skip(self, job), fields(job_id = %job.job_id, workload = %job.workload_id))]
    async fn execute_job(&self, job: JobEnvelope) -> JobResult {
        let start = Instant::now();
        let job_id = job.job_id;

        // Route to appropriate executor based on workload_id
        let result = match job.workload_id.as_str() {
            "embeddings-v1" | "embeddings" => self.execute_embeddings_job(job).await,
            unknown => {
                warn!(workload = unknown, "Unknown workload type");
                Err(crate::errors::AgentError::Execution(format!(
                    "Unknown workload type: {}",
                    unknown
                )))
            }
        };

        // Convert to JobResult
        let execution_time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(output_bytes) => {
                info!(
                    job_id = %job_id,
                    execution_time_ms = execution_time_ms,
                    "Job executed successfully"
                );

                JobResult {
                    job_id,
                    success: true,
                    result: Some(output_bytes),
                    error: None,
                    execution_time_ms,
                }
            }
            Err(e) => {
                error!(
                    job_id = %job_id,
                    error = %e,
                    execution_time_ms = execution_time_ms,
                    "Job execution failed"
                );

                JobResult {
                    job_id,
                    success: false,
                    result: None,
                    error: Some(e.to_string()),
                    execution_time_ms,
                }
            }
        }
    }

    /// Execute an embeddings workload
    async fn execute_embeddings_job(&self, job: JobEnvelope) -> Result<Vec<u8>> {
        // Deserialize input
        let input = EmbeddingsInput::from_cbor(&job.payload)?;

        debug!(text_len = input.text.len(), "Executing embeddings job");

        // Execute with timeout
        let output = self
            .executor
            .execute_with_timeout(&input, job.timeout_ms)
            .await?;

        // Serialize output
        let output_bytes = output.to_cbor()?;

        Ok(output_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_stats_creation() {
        let stats = JobStats::new();
        assert_eq!(stats.total_jobs(), 0);
        assert_eq!(stats.avg_execution_time_ms(), 0.0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_job_stats_record_success() {
        let stats = JobStats::new();

        stats.record_success(100);
        stats.record_success(200);

        assert_eq!(stats.total_jobs(), 2);
        assert_eq!(stats.jobs_completed.load(Ordering::Relaxed), 2);
        assert_eq!(stats.jobs_failed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.avg_execution_time_ms(), 150.0);
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[test]
    fn test_job_stats_record_failure() {
        let stats = JobStats::new();

        stats.record_success(100);
        stats.record_failure(200);

        assert_eq!(stats.total_jobs(), 2);
        assert_eq!(stats.jobs_completed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.jobs_failed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.success_rate(), 0.5);
    }

    #[test]
    fn test_job_stats_active_jobs() {
        let stats = JobStats::new();

        stats.start_job();
        stats.start_job();
        assert_eq!(stats.active_jobs(), 2);

        stats.finish_job();
        assert_eq!(stats.active_jobs(), 1);

        stats.finish_job();
        assert_eq!(stats.active_jobs(), 0);
    }
}
