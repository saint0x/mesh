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

use crate::errors::Result;
use crate::executor::{EmbeddingsExecutor, EmbeddingsInput, EmbeddingsOutput};
use crate::network::{JobEnvelope, JobResult, MeshEvent, MeshSwarm};
use libp2p::PeerId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::signal;
use tokio::task::JoinSet;
use tracing::{debug, error, info, instrument, warn};

/// Job execution statistics
///
/// Tracks metrics about job execution for monitoring and debugging.
#[derive(Debug)]
pub struct JobStats {
    /// Total number of jobs successfully completed
    pub jobs_completed: AtomicU64,

    /// Total number of jobs that failed
    pub jobs_failed: AtomicU64,

    /// Total execution time across all jobs (milliseconds)
    pub total_execution_time_ms: AtomicU64,

    /// Number of jobs currently being executed
    pub active_jobs: AtomicU64,

    /// Start time for uptime tracking
    pub start_time: Instant,

    /// Number of direct peer connections established
    pub direct_peer_connections: AtomicU64,

    /// Number of relayed peer connections established
    pub relayed_peer_connections: AtomicU64,

    /// Number of times relay fallback was used after a direct dial failed
    pub relay_fallbacks: AtomicU64,

    /// Number of successful direct connection upgrades
    pub direct_upgrade_successes: AtomicU64,

    /// Number of failed direct connection upgrades
    pub direct_upgrade_failures: AtomicU64,

    /// Number of external/public address candidates discovered
    pub external_addr_candidates: AtomicU64,

    /// Number of external/public addresses confirmed
    pub external_addr_confirmed: AtomicU64,

    /// Number of explicit punch-path attempts initiated
    pub punch_path_attempts: AtomicU64,

    /// Number of direct peer connections established while a punch-path plan was active
    pub punch_assisted_direct_peer_connections: AtomicU64,

    /// Number of successful direct upgrades after an explicit punch-path plan
    pub punch_assisted_upgrade_successes: AtomicU64,

    /// Number of failed direct upgrades after an explicit punch-path plan
    pub punch_assisted_upgrade_failures: AtomicU64,

    /// Number of jobs rejected because the runner was already at capacity
    pub backpressure_rejections: AtomicU64,
}

impl Default for JobStats {
    fn default() -> Self {
        Self::new()
    }
}

impl JobStats {
    /// Create new job statistics tracker
    pub fn new() -> Self {
        Self {
            jobs_completed: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
            active_jobs: AtomicU64::new(0),
            start_time: Instant::now(),
            direct_peer_connections: AtomicU64::new(0),
            relayed_peer_connections: AtomicU64::new(0),
            relay_fallbacks: AtomicU64::new(0),
            direct_upgrade_successes: AtomicU64::new(0),
            direct_upgrade_failures: AtomicU64::new(0),
            external_addr_candidates: AtomicU64::new(0),
            external_addr_confirmed: AtomicU64::new(0),
            punch_path_attempts: AtomicU64::new(0),
            punch_assisted_direct_peer_connections: AtomicU64::new(0),
            punch_assisted_upgrade_successes: AtomicU64::new(0),
            punch_assisted_upgrade_failures: AtomicU64::new(0),
            backpressure_rejections: AtomicU64::new(0),
        }
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

    pub fn record_direct_peer_connection(&self) {
        self.direct_peer_connections.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_relayed_peer_connection(&self) {
        self.relayed_peer_connections
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_relay_fallback(&self) {
        self.relay_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_direct_upgrade_success(&self) {
        self.direct_upgrade_successes
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_direct_upgrade_failure(&self) {
        self.direct_upgrade_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_external_addr_candidate(&self) {
        self.external_addr_candidates
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_external_addr_confirmed(&self) {
        self.external_addr_confirmed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_punch_path_attempt(&self) {
        self.punch_path_attempts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_punch_assisted_direct_peer_connection(&self) {
        self.punch_assisted_direct_peer_connections
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_punch_assisted_upgrade_success(&self) {
        self.punch_assisted_upgrade_successes
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_punch_assisted_upgrade_failure(&self) {
        self.punch_assisted_upgrade_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_backpressure_rejection(&self) {
        self.backpressure_rejections.fetch_add(1, Ordering::Relaxed);
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

    /// Get agent uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Format uptime as human-readable string
    pub fn uptime_string(&self) -> String {
        let seconds = self.uptime_seconds();
        let days = seconds / 86400;
        let hours = (seconds % 86400) / 3600;
        let minutes = (seconds % 3600) / 60;
        let secs = seconds % 60;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, secs)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, secs)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }

    /// Print statistics summary (for logging)
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
            uptime = %self.uptime_string(),
            "Job statistics"
        );
    }

    /// Display formatted metrics summary with colors (for CLI)
    pub fn display(&self) {
        use colored::Colorize;

        println!("\n{}", "Agent Metrics".bold().cyan());
        println!("{}", "=============".cyan());

        println!("\n{}", "Job Statistics:".bold());
        println!("  Total Jobs:       {}", self.total_jobs());
        println!(
            "  Completed:        {}",
            self.jobs_completed
                .load(Ordering::Relaxed)
                .to_string()
                .green()
        );
        println!(
            "  Failed:           {}",
            self.jobs_failed.load(Ordering::Relaxed).to_string().red()
        );
        println!("  Active:           {}", self.active_jobs());
        println!(
            "  Rejected:         {}",
            self.backpressure_rejections.load(Ordering::Relaxed)
        );
        println!("  Success Rate:     {:.1}%", self.success_rate() * 100.0);

        println!("\n{}", "Performance:".bold());
        println!("  Avg Execution:    {:.2}ms", self.avg_execution_time_ms());
        println!(
            "  Total CPU Time:   {}ms",
            self.total_execution_time_ms.load(Ordering::Relaxed)
        );

        println!("\n{}", "System:".bold());
        println!("  Uptime:           {}", self.uptime_string());
        println!("\n{}", "Connectivity:".bold());
        println!(
            "  Direct Peers:     {}",
            self.direct_peer_connections.load(Ordering::Relaxed)
        );
        println!(
            "  Relayed Peers:    {}",
            self.relayed_peer_connections.load(Ordering::Relaxed)
        );
        println!(
            "  Relay Fallbacks:  {}",
            self.relay_fallbacks.load(Ordering::Relaxed)
        );
        println!(
            "  DCUTR Successes:  {}",
            self.direct_upgrade_successes.load(Ordering::Relaxed)
        );
        println!(
            "  DCUTR Failures:   {}",
            self.direct_upgrade_failures.load(Ordering::Relaxed)
        );
        println!(
            "  Ext Candidates:   {}",
            self.external_addr_candidates.load(Ordering::Relaxed)
        );
        println!(
            "  Ext Confirmed:    {}",
            self.external_addr_confirmed.load(Ordering::Relaxed)
        );
        println!(
            "  Punch Attempts:   {}",
            self.punch_path_attempts.load(Ordering::Relaxed)
        );
        println!(
            "  Punch Direct:     {}",
            self.punch_assisted_direct_peer_connections
                .load(Ordering::Relaxed)
        );
        println!(
            "  Punch Upgrades:   {}",
            self.punch_assisted_upgrade_successes
                .load(Ordering::Relaxed)
        );
        println!(
            "  Punch Failures:   {}",
            self.punch_assisted_upgrade_failures.load(Ordering::Relaxed)
        );
        println!();
    }

    /// Save statistics to file for `mesh metrics` command
    pub fn save_to_file(&self) -> Result<()> {
        use std::fs;
        use std::path::PathBuf;

        let stats_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".meshnet")
            .join("stats.json");

        // Ensure directory exists
        if let Some(parent) = stats_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let stats_json = serde_json::json!({
            "total_jobs": self.total_jobs(),
            "completed": self.jobs_completed.load(Ordering::Relaxed),
            "failed": self.jobs_failed.load(Ordering::Relaxed),
            "active": self.active_jobs(),
            "rejected": self.backpressure_rejections.load(Ordering::Relaxed),
            "success_rate": self.success_rate() * 100.0,
            "avg_execution_time_ms": self.avg_execution_time_ms(),
            "total_execution_time_ms": self.total_execution_time_ms.load(Ordering::Relaxed),
            "uptime": self.uptime_string(),
            "connectivity": {
                "direct_peer_connections": self.direct_peer_connections.load(Ordering::Relaxed),
                "relayed_peer_connections": self.relayed_peer_connections.load(Ordering::Relaxed),
                "relay_fallbacks": self.relay_fallbacks.load(Ordering::Relaxed),
                "direct_upgrade_successes": self.direct_upgrade_successes.load(Ordering::Relaxed),
                "direct_upgrade_failures": self.direct_upgrade_failures.load(Ordering::Relaxed),
                "external_addr_candidates": self.external_addr_candidates.load(Ordering::Relaxed),
                "external_addr_confirmed": self.external_addr_confirmed.load(Ordering::Relaxed),
                "punch_path_attempts": self.punch_path_attempts.load(Ordering::Relaxed),
                "punch_assisted_direct_peer_connections": self.punch_assisted_direct_peer_connections.load(Ordering::Relaxed),
                "punch_assisted_upgrade_successes": self.punch_assisted_upgrade_successes.load(Ordering::Relaxed),
                "punch_assisted_upgrade_failures": self.punch_assisted_upgrade_failures.load(Ordering::Relaxed),
            },
            "last_updated": chrono::Local::now().to_rfc3339(),
        });

        fs::write(&stats_path, serde_json::to_string_pretty(&stats_json)?)?;
        Ok(())
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
    max_concurrent_jobs: usize,
    ledger_client: Option<crate::telemetry::LedgerClient>,
    device_id: Option<uuid::Uuid>,
    network_id: Option<String>,
    tier: Option<crate::device::Tier>,
    punch_attempt_state: HashMap<PeerId, bool>,
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
            max_concurrent_jobs: 2,
            ledger_client: None,
            device_id: None,
            network_id: None,
            tier: None,
            punch_attempt_state: HashMap::new(),
        }
    }

    /// Enable shutdown when idle (for testing/ephemeral jobs)
    pub fn with_shutdown_on_idle(mut self) -> Self {
        self.shutdown_on_idle = true;
        self
    }

    pub fn with_max_concurrent_jobs(mut self, max_concurrent_jobs: usize) -> Self {
        self.max_concurrent_jobs = max_concurrent_jobs.max(1);
        self
    }

    /// Add ledger tracking to job runner
    pub fn with_ledger(
        mut self,
        ledger_client: crate::telemetry::LedgerClient,
        device_id: uuid::Uuid,
        network_id: String,
        tier: crate::device::Tier,
    ) -> Self {
        self.ledger_client = Some(ledger_client);
        self.device_id = Some(device_id);
        self.network_id = Some(network_id);
        self.tier = Some(tier);
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
        let mut in_flight_jobs = JoinSet::new();

        // Periodic stats saver (every 30 seconds)
        let mut stats_interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        stats_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Handle network events
                event = self.swarm.next_event() => {
                    if let Some(event) = event {
                        if let Err(e) = self.handle_event(event, &mut in_flight_jobs).await {
                            error!(error = %e, "Error handling event");
                        }
                    } else {
                        warn!("Event stream ended");
                        break;
                    }
                }

                // Save stats periodically
                _ = stats_interval.tick() => {
                    debug!("Saving stats to file");
                    if let Err(e) = self.stats.save_to_file() {
                        warn!(error = %e, "Failed to save stats");
                    }
                }

                // Handle shutdown signal
                _ = signal::ctrl_c() => {
                    info!("Received shutdown signal (Ctrl+C)");
                    break;
                }

                Some(joined_job) = in_flight_jobs.join_next() => {
                    match joined_job {
                        Ok(completed) => {
                            self.finish_completed_job(completed)?;
                        }
                        Err(e) => {
                            error!(error = %e, "Job task panicked");
                            self.stats.finish_job();
                        }
                    }
                }
            }
        }

        // Save final statistics
        info!("Shutting down job runner");
        self.stats.print_summary();
        let _ = self.stats.save_to_file();

        Ok(())
    }

    /// Handle a mesh network event
    ///
    /// Processes different types of network events, with special handling for
    /// job-related events (JobReceived, JobCompleted).
    #[instrument(skip(self, event), fields(event_type = std::any::type_name::<MeshEvent>()))]
    async fn handle_event(
        &mut self,
        event: MeshEvent,
        in_flight_jobs: &mut JoinSet<CompletedJob>,
    ) -> Result<()> {
        match event {
            MeshEvent::JobReceived {
                peer_id,
                job,
                channel,
            } => {
                self.handle_job_received(peer_id, job, channel, in_flight_jobs)
                    .await?;
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
                if matches!(
                    connection_info.connection_type,
                    crate::network::ConnectionType::Direct
                ) {
                    if let Some(recorded_direct) = self.punch_attempt_state.get_mut(&peer_id) {
                        if !*recorded_direct {
                            self.stats.record_punch_assisted_direct_peer_connection();
                            *recorded_direct = true;
                        }
                    }
                }
                match connection_info.connection_type {
                    crate::network::ConnectionType::Direct => {
                        self.stats.record_direct_peer_connection()
                    }
                    crate::network::ConnectionType::Relayed => {
                        self.stats.record_relayed_peer_connection()
                    }
                    crate::network::ConnectionType::Unknown => {}
                }
                info!(
                    peer_id = %peer_id,
                    connection_type = ?connection_info.connection_type,
                    "Peer connected"
                );
            }

            MeshEvent::PeerDisconnected { peer_id } => {
                self.punch_attempt_state.remove(&peer_id);
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

            MeshEvent::RelayFallbackToPeer { peer_id } => {
                self.stats.record_relay_fallback();
                self.punch_attempt_state.remove(&peer_id);
                warn!(peer_id = %peer_id, "Relay fallback used for peer");
            }

            MeshEvent::DirectConnectionUpgraded { peer_id } => {
                self.stats.record_direct_upgrade_success();
                if self.punch_attempt_state.remove(&peer_id).is_some() {
                    self.stats.record_punch_assisted_upgrade_success();
                }
                info!(peer_id = %peer_id, "Direct connection upgrade succeeded");
            }

            MeshEvent::DirectConnectionUpgradeFailed { peer_id, error } => {
                self.stats.record_direct_upgrade_failure();
                if self.punch_attempt_state.remove(&peer_id).is_some() {
                    self.stats.record_punch_assisted_upgrade_failure();
                }
                warn!(peer_id = %peer_id, error = %error, "Direct connection upgrade failed");
            }

            MeshEvent::ExternalAddrCandidateDiscovered { address } => {
                self.stats.record_external_addr_candidate();
                debug!(address = %address, "External address candidate discovered");
            }

            MeshEvent::ExternalAddrConfirmed { address } => {
                self.stats.record_external_addr_confirmed();
                info!(address = %address, "External address confirmed");
            }

            MeshEvent::PunchPathAttemptInitiated {
                peer_id,
                reason,
                candidate_count,
                relay_rendezvous_required,
            } => {
                self.stats.record_punch_path_attempt();
                self.punch_attempt_state.insert(peer_id, false);
                info!(
                    peer_id = %peer_id,
                    reason = %reason,
                    candidate_count = candidate_count,
                    relay_rendezvous_required = relay_rendezvous_required,
                    "Explicit punched-path attempt initiated"
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
        in_flight_jobs: &mut JoinSet<CompletedJob>,
    ) -> Result<()> {
        info!("Received job from peer");
        if self.stats.active_jobs() as usize >= self.max_concurrent_jobs {
            self.stats.record_backpressure_rejection();
            warn!(
                peer_id = %peer_id,
                job_id = %job.job_id,
                active_jobs = self.stats.active_jobs(),
                max_concurrent_jobs = self.max_concurrent_jobs,
                "Rejecting job because runner is at capacity"
            );
            self.swarm.respond_to_job(
                channel,
                JobResult {
                    job_id: job.job_id,
                    success: false,
                    result: None,
                    error: Some(format!(
                        "agent at capacity: {} active jobs (limit {})",
                        self.stats.active_jobs(),
                        self.max_concurrent_jobs
                    )),
                    execution_time_ms: 0,
                },
            )?;
            return Ok(());
        }

        self.stats.start_job();
        let executor = self.executor.clone();
        let ledger_client = self.ledger_client.clone();
        let device_id = self.device_id;
        let network_id = self.network_id.clone();
        let tier = self.tier;
        in_flight_jobs.spawn(async move {
            execute_job_task(
                executor,
                ledger_client,
                device_id,
                network_id,
                tier,
                peer_id,
                job,
                channel,
            )
            .await
        });

        Ok(())
    }

    fn finish_completed_job(&mut self, completed: CompletedJob) -> Result<()> {
        if completed.result.success {
            self.stats.record_success(completed.execution_time_ms);
        } else {
            self.stats.record_failure(completed.execution_time_ms);
        }
        self.stats.finish_job();

        debug!(
            job_id = %completed.job_id,
            peer_id = %completed.peer_id,
            success = completed.result.success,
            execution_time_ms = completed.execution_time_ms,
            "Sending completed job result"
        );
        self.swarm
            .respond_to_job(completed.channel, completed.result)?;
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
                        preview.first().unwrap_or(&0.0),
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
    #[instrument(skip(executor, job), fields(job_id = %job.job_id, workload = %job.workload_id))]
    async fn execute_job_with_executor(
        executor: &EmbeddingsExecutor,
        job: JobEnvelope,
    ) -> JobResult {
        let start = Instant::now();
        let job_id = job.job_id;

        // Route to appropriate executor based on workload_id
        let result = match job.workload_id.as_str() {
            "embeddings-v1" | "embeddings" => {
                execute_embeddings_job_with_executor(executor, job).await
            }
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
}

struct CompletedJob {
    peer_id: libp2p::PeerId,
    job_id: uuid::Uuid,
    channel: libp2p::request_response::ResponseChannel<JobResult>,
    result: JobResult,
    execution_time_ms: u64,
}

async fn execute_job_task(
    executor: EmbeddingsExecutor,
    ledger_client: Option<crate::telemetry::LedgerClient>,
    device_id: Option<uuid::Uuid>,
    network_id: Option<String>,
    tier: Option<crate::device::Tier>,
    peer_id: libp2p::PeerId,
    job: JobEnvelope,
    channel: libp2p::request_response::ResponseChannel<JobResult>,
) -> CompletedJob {
    let job_id = job.job_id;
    let result = JobRunner::execute_job_with_executor(&executor, job).await;
    let execution_time_ms = result.execution_time_ms;

    if let (Some(client), Some(device_id), Some(network_id), Some(tier)) =
        (ledger_client, device_id, network_id, tier)
    {
        let credits = crate::telemetry::calculate_credits(execution_time_ms, &tier);
        let event = crate::telemetry::LedgerEvent {
            network_id,
            event_type: if result.success {
                "job_completed".to_string()
            } else {
                "job_failed".to_string()
            },
            job_id: Some(job_id),
            device_id,
            credits_amount: if result.success { Some(credits) } else { None },
            metadata: serde_json::json!({
                "execution_time_ms": execution_time_ms,
                "workload": "embeddings",
            }),
        };
        client.send_event_async(event);
    }

    CompletedJob {
        peer_id,
        job_id,
        channel,
        result,
        execution_time_ms,
    }
}

async fn execute_embeddings_job_with_executor(
    executor: &EmbeddingsExecutor,
    job: JobEnvelope,
) -> Result<Vec<u8>> {
    // Deserialize input
    let input = EmbeddingsInput::from_cbor(&job.payload)?;

    debug!(text_len = input.text.len(), "Executing embeddings job");

    // Execute with timeout
    let output = executor
        .execute_with_timeout(&input, job.timeout_ms)
        .await?;

    // Serialize output
    let output_bytes = output.to_cbor()?;

    Ok(output_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{ConnectionInfo, ConnectionType};
    use libp2p::Multiaddr;

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

    #[test]
    fn test_job_stats_connectivity_metrics() {
        let stats = JobStats::new();

        stats.record_direct_peer_connection();
        stats.record_relayed_peer_connection();
        stats.record_relay_fallback();
        stats.record_direct_upgrade_success();
        stats.record_direct_upgrade_failure();
        stats.record_external_addr_candidate();
        stats.record_external_addr_confirmed();
        stats.record_punch_path_attempt();
        stats.record_punch_assisted_direct_peer_connection();
        stats.record_punch_assisted_upgrade_success();
        stats.record_punch_assisted_upgrade_failure();

        assert_eq!(stats.direct_peer_connections.load(Ordering::Relaxed), 1);
        assert_eq!(stats.relayed_peer_connections.load(Ordering::Relaxed), 1);
        assert_eq!(stats.relay_fallbacks.load(Ordering::Relaxed), 1);
        assert_eq!(stats.direct_upgrade_successes.load(Ordering::Relaxed), 1);
        assert_eq!(stats.direct_upgrade_failures.load(Ordering::Relaxed), 1);
        assert_eq!(stats.external_addr_candidates.load(Ordering::Relaxed), 1);
        assert_eq!(stats.external_addr_confirmed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.punch_path_attempts.load(Ordering::Relaxed), 1);
        assert_eq!(
            stats
                .punch_assisted_direct_peer_connections
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .punch_assisted_upgrade_successes
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .punch_assisted_upgrade_failures
                .load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_job_stats_backpressure_metrics() {
        let stats = JobStats::new();
        stats.record_backpressure_rejection();
        stats.record_backpressure_rejection();
        assert_eq!(stats.backpressure_rejections.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_with_max_concurrent_jobs_clamps_to_one() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let swarm = MeshSwarm::builder(keypair).build().unwrap();
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let runner = JobRunner::new(swarm, executor).with_max_concurrent_jobs(0);

        assert_eq!(runner.max_concurrent_jobs, 1);
    }

    #[tokio::test]
    async fn test_handle_event_records_connectivity_path_metrics() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let swarm = MeshSwarm::builder(keypair).build().unwrap();
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let mut runner = JobRunner::new(swarm, executor);
        let mut in_flight_jobs = JoinSet::new();

        runner
            .handle_event(
                MeshEvent::PeerConnected {
                    peer_id: libp2p::PeerId::random(),
                    connection_info: ConnectionInfo {
                        connection_type: ConnectionType::Direct,
                        remote_addr: "/ip4/34.120.0.10/tcp/4001".parse::<Multiaddr>().unwrap(),
                        num_established: 1,
                    },
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        runner
            .handle_event(
                MeshEvent::PeerConnected {
                    peer_id: libp2p::PeerId::random(),
                    connection_info: ConnectionInfo {
                        connection_type: ConnectionType::Relayed,
                        remote_addr: "/dns4/relay.mesh.example/tcp/4001/p2p-circuit"
                            .parse::<Multiaddr>()
                            .unwrap(),
                        num_established: 1,
                    },
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        runner
            .handle_event(
                MeshEvent::RelayFallbackToPeer {
                    peer_id: libp2p::PeerId::random(),
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        assert_eq!(
            runner.stats.direct_peer_connections.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            runner
                .stats
                .relayed_peer_connections
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(runner.stats.relay_fallbacks.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_handle_event_records_upgrade_and_external_addr_metrics() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let swarm = MeshSwarm::builder(keypair).build().unwrap();
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let mut runner = JobRunner::new(swarm, executor);
        let mut in_flight_jobs = JoinSet::new();

        runner
            .handle_event(
                MeshEvent::DirectConnectionUpgraded {
                    peer_id: libp2p::PeerId::random(),
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        runner
            .handle_event(
                MeshEvent::DirectConnectionUpgradeFailed {
                    peer_id: libp2p::PeerId::random(),
                    error: "timeout".to_string(),
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        runner
            .handle_event(
                MeshEvent::ExternalAddrCandidateDiscovered {
                    address: "/ip4/34.120.0.10/tcp/4001".parse::<Multiaddr>().unwrap(),
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        runner
            .handle_event(
                MeshEvent::ExternalAddrConfirmed {
                    address: "/ip4/34.120.0.10/tcp/4001".parse::<Multiaddr>().unwrap(),
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        assert_eq!(
            runner
                .stats
                .direct_upgrade_successes
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            runner.stats.direct_upgrade_failures.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            runner
                .stats
                .external_addr_candidates
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            runner.stats.external_addr_confirmed.load(Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn test_handle_event_records_punch_assisted_metrics() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let swarm = MeshSwarm::builder(keypair).build().unwrap();
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let mut runner = JobRunner::new(swarm, executor);
        let peer_id = libp2p::PeerId::random();
        let mut in_flight_jobs = JoinSet::new();

        runner
            .handle_event(
                MeshEvent::PunchPathAttemptInitiated {
                    peer_id,
                    reason: "RelayPath".to_string(),
                    candidate_count: 2,
                    relay_rendezvous_required: true,
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();
        runner
            .handle_event(
                MeshEvent::PeerConnected {
                    peer_id,
                    connection_info: ConnectionInfo {
                        connection_type: ConnectionType::Direct,
                        remote_addr: "/ip4/34.120.0.10/tcp/4001".parse::<Multiaddr>().unwrap(),
                        num_established: 1,
                    },
                },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();
        runner
            .handle_event(
                MeshEvent::DirectConnectionUpgraded { peer_id },
                &mut in_flight_jobs,
            )
            .await
            .unwrap();

        assert_eq!(runner.stats.punch_path_attempts.load(Ordering::Relaxed), 1);
        assert_eq!(
            runner
                .stats
                .punch_assisted_direct_peer_connections
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            runner
                .stats
                .punch_assisted_upgrade_successes
                .load(Ordering::Relaxed),
            1
        );
    }
}
