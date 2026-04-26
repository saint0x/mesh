//! Inference statistics tracking
//!
//! This module provides metrics tracking for distributed inference operations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::info;

use crate::executor::ring_allreduce::RingAllReduceMetrics;
use crate::network::TensorPlaneMetricsSnapshot;

/// Statistics for inference operations
#[derive(Debug)]
pub struct InferenceStats {
    /// Total number of inference jobs completed
    pub jobs_completed: AtomicU64,

    /// Total number of inference jobs failed
    pub jobs_failed: AtomicU64,

    /// Total tokens generated across all jobs
    pub total_tokens_generated: AtomicU64,

    /// Total prompt tokens processed
    pub total_prompt_tokens: AtomicU64,

    /// Total inference time in milliseconds
    pub total_inference_time_ms: AtomicU64,

    /// Total all-reduce time in milliseconds
    pub total_allreduce_time_ms: AtomicU64,

    /// Number of checkpoints created
    pub checkpoints_created: AtomicU64,

    /// Number of recoveries from checkpoint
    pub checkpoint_recoveries: AtomicU64,

    /// Number of checkpoint recovery attempts initiated.
    pub recovery_attempts: AtomicU64,

    /// Number of checkpoint recovery attempts skipped due to cooldown.
    pub recovery_cooldown_rejections: AtomicU64,

    /// Number of checkpoint recovery attempts skipped due to node-level load budget.
    pub recovery_budget_rejections: AtomicU64,

    /// Number of checkpoint recovery attempts that found no usable checkpoint.
    pub recovery_checkpoint_misses: AtomicU64,

    /// Start time for uptime tracking
    pub start_time: Instant,

    /// Number of ring all-reduce operations performed
    pub allreduce_operations: AtomicU64,

    /// Total layers processed (for averaging)
    pub total_layers_processed: AtomicU64,

    /// Total tensor-plane bytes sent across ring traffic.
    pub tensor_bytes_sent: AtomicU64,

    /// Total tensor-plane bytes received across ring traffic.
    pub tensor_bytes_received: AtomicU64,

    pub tensor_reduce_scatter_bytes_sent: AtomicU64,
    pub tensor_reduce_scatter_bytes_received: AtomicU64,
    pub tensor_all_gather_bytes_sent: AtomicU64,
    pub tensor_all_gather_bytes_received: AtomicU64,
    pub tensor_barrier_bytes_sent: AtomicU64,
    pub tensor_barrier_bytes_received: AtomicU64,

    /// Number of outbound sends that had to wait for byte-budget permits.
    pub tensor_outbound_backpressure_wait_count: AtomicU64,

    /// Total time spent waiting on outbound tensor-plane byte-budget permits.
    pub tensor_outbound_backpressure_wait_ms: AtomicU64,

    /// Number of outbound sends delayed by the tensor-plane bandwidth governor.
    pub tensor_outbound_bandwidth_wait_count: AtomicU64,

    /// Total time spent waiting on the tensor-plane bandwidth governor.
    pub tensor_outbound_bandwidth_wait_ms: AtomicU64,

    pub tensor_send_count: AtomicU64,
    pub tensor_send_latency_ms: AtomicU64,
    pub tensor_receive_count: AtomicU64,
    pub tensor_receive_latency_ms: AtomicU64,
    pub tensor_receive_queue_wait_ms: AtomicU64,
    pub tensor_send_timeout_count: AtomicU64,
    pub tensor_receive_timeout_count: AtomicU64,

    /// Number of inbound tensor messages rejected because the bounded queue was full.
    pub tensor_inbound_queue_full_rejections: AtomicU64,

    /// Number of inbound tensor messages rejected because the queued-byte budget was exhausted.
    pub tensor_inbound_byte_budget_rejections: AtomicU64,

    /// Number of tensor messages rejected because they exceeded the message-size budget.
    pub tensor_oversized_message_rejections: AtomicU64,

    /// Current inbound queued tensor bytes waiting for consumption.
    pub tensor_current_inbound_queued_bytes: AtomicU64,

    pub tensor_peak_inbound_queued_bytes: AtomicU64,

    /// Current outbound in-flight tensor bytes waiting to complete.
    pub tensor_current_outbound_inflight_bytes: AtomicU64,

    pub tensor_peak_outbound_inflight_bytes: AtomicU64,
    pub tensor_current_outbound_connections: AtomicU64,

    pub total_reduce_scatter_time_ms: AtomicU64,
    pub total_all_gather_time_ms: AtomicU64,
    pub total_allreduce_send_wait_ms: AtomicU64,
    pub total_allreduce_receive_wait_ms: AtomicU64,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceStats {
    /// Create new inference statistics tracker
    pub fn new() -> Self {
        Self {
            jobs_completed: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_inference_time_ms: AtomicU64::new(0),
            total_allreduce_time_ms: AtomicU64::new(0),
            checkpoints_created: AtomicU64::new(0),
            checkpoint_recoveries: AtomicU64::new(0),
            recovery_attempts: AtomicU64::new(0),
            recovery_cooldown_rejections: AtomicU64::new(0),
            recovery_budget_rejections: AtomicU64::new(0),
            recovery_checkpoint_misses: AtomicU64::new(0),
            start_time: Instant::now(),
            allreduce_operations: AtomicU64::new(0),
            total_layers_processed: AtomicU64::new(0),
            tensor_bytes_sent: AtomicU64::new(0),
            tensor_bytes_received: AtomicU64::new(0),
            tensor_reduce_scatter_bytes_sent: AtomicU64::new(0),
            tensor_reduce_scatter_bytes_received: AtomicU64::new(0),
            tensor_all_gather_bytes_sent: AtomicU64::new(0),
            tensor_all_gather_bytes_received: AtomicU64::new(0),
            tensor_barrier_bytes_sent: AtomicU64::new(0),
            tensor_barrier_bytes_received: AtomicU64::new(0),
            tensor_outbound_backpressure_wait_count: AtomicU64::new(0),
            tensor_outbound_backpressure_wait_ms: AtomicU64::new(0),
            tensor_outbound_bandwidth_wait_count: AtomicU64::new(0),
            tensor_outbound_bandwidth_wait_ms: AtomicU64::new(0),
            tensor_send_count: AtomicU64::new(0),
            tensor_send_latency_ms: AtomicU64::new(0),
            tensor_receive_count: AtomicU64::new(0),
            tensor_receive_latency_ms: AtomicU64::new(0),
            tensor_receive_queue_wait_ms: AtomicU64::new(0),
            tensor_send_timeout_count: AtomicU64::new(0),
            tensor_receive_timeout_count: AtomicU64::new(0),
            tensor_inbound_queue_full_rejections: AtomicU64::new(0),
            tensor_inbound_byte_budget_rejections: AtomicU64::new(0),
            tensor_oversized_message_rejections: AtomicU64::new(0),
            tensor_current_inbound_queued_bytes: AtomicU64::new(0),
            tensor_current_outbound_inflight_bytes: AtomicU64::new(0),
            tensor_peak_inbound_queued_bytes: AtomicU64::new(0),
            tensor_peak_outbound_inflight_bytes: AtomicU64::new(0),
            tensor_current_outbound_connections: AtomicU64::new(0),
            total_reduce_scatter_time_ms: AtomicU64::new(0),
            total_all_gather_time_ms: AtomicU64::new(0),
            total_allreduce_send_wait_ms: AtomicU64::new(0),
            total_allreduce_receive_wait_ms: AtomicU64::new(0),
        }
    }

    /// Record a successful inference job
    pub fn record_success(
        &self,
        prompt_tokens: u32,
        completion_tokens: u32,
        inference_time_ms: u64,
    ) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.total_inference_time_ms
            .fetch_add(inference_time_ms, Ordering::Relaxed);
    }

    /// Record a failed inference job
    pub fn record_failure(&self) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an all-reduce operation
    pub fn record_allreduce(&self, duration_ms: u64) {
        self.allreduce_operations.fetch_add(1, Ordering::Relaxed);
        self.total_allreduce_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Record a layer processed
    pub fn record_layer(&self) {
        self.total_layers_processed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_allreduce_breakdown(&self, metrics: RingAllReduceMetrics) {
        self.total_reduce_scatter_time_ms
            .fetch_add(metrics.reduce_scatter_step_time_ms, Ordering::Relaxed);
        self.total_all_gather_time_ms
            .fetch_add(metrics.all_gather_step_time_ms, Ordering::Relaxed);
        self.total_allreduce_send_wait_ms
            .fetch_add(metrics.send_wait_time_ms, Ordering::Relaxed);
        self.total_allreduce_receive_wait_ms
            .fetch_add(metrics.receive_wait_time_ms, Ordering::Relaxed);
    }

    /// Record a checkpoint creation
    pub fn record_checkpoint(&self) {
        self.checkpoints_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a recovery from checkpoint
    pub fn record_recovery(&self) {
        self.checkpoint_recoveries.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_recovery_attempt(&self) {
        self.recovery_attempts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_recovery_cooldown_rejection(&self) {
        self.recovery_cooldown_rejections
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_recovery_budget_rejection(&self) {
        self.recovery_budget_rejections
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_recovery_checkpoint_miss(&self) {
        self.recovery_checkpoint_misses
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_tensor_plane_metrics(&self, snapshot: TensorPlaneMetricsSnapshot) {
        self.tensor_bytes_sent
            .store(snapshot.bytes_sent, Ordering::Relaxed);
        self.tensor_bytes_received
            .store(snapshot.bytes_received, Ordering::Relaxed);
        self.tensor_reduce_scatter_bytes_sent
            .store(snapshot.reduce_scatter_bytes_sent, Ordering::Relaxed);
        self.tensor_reduce_scatter_bytes_received
            .store(snapshot.reduce_scatter_bytes_received, Ordering::Relaxed);
        self.tensor_all_gather_bytes_sent
            .store(snapshot.all_gather_bytes_sent, Ordering::Relaxed);
        self.tensor_all_gather_bytes_received
            .store(snapshot.all_gather_bytes_received, Ordering::Relaxed);
        self.tensor_barrier_bytes_sent
            .store(snapshot.barrier_bytes_sent, Ordering::Relaxed);
        self.tensor_barrier_bytes_received
            .store(snapshot.barrier_bytes_received, Ordering::Relaxed);
        self.tensor_outbound_backpressure_wait_count
            .store(snapshot.outbound_backpressure_wait_count, Ordering::Relaxed);
        self.tensor_outbound_backpressure_wait_ms
            .store(snapshot.outbound_backpressure_wait_ms, Ordering::Relaxed);
        self.tensor_outbound_bandwidth_wait_count
            .store(snapshot.outbound_bandwidth_wait_count, Ordering::Relaxed);
        self.tensor_outbound_bandwidth_wait_ms
            .store(snapshot.outbound_bandwidth_wait_ms, Ordering::Relaxed);
        self.tensor_send_count
            .store(snapshot.send_count, Ordering::Relaxed);
        self.tensor_send_latency_ms
            .store(snapshot.send_latency_ms, Ordering::Relaxed);
        self.tensor_receive_count
            .store(snapshot.receive_count, Ordering::Relaxed);
        self.tensor_receive_latency_ms
            .store(snapshot.receive_latency_ms, Ordering::Relaxed);
        self.tensor_receive_queue_wait_ms
            .store(snapshot.receive_queue_wait_ms, Ordering::Relaxed);
        self.tensor_send_timeout_count
            .store(snapshot.send_timeout_count, Ordering::Relaxed);
        self.tensor_receive_timeout_count
            .store(snapshot.receive_timeout_count, Ordering::Relaxed);
        self.tensor_inbound_queue_full_rejections
            .store(snapshot.inbound_queue_full_rejections, Ordering::Relaxed);
        self.tensor_inbound_byte_budget_rejections
            .store(snapshot.inbound_byte_budget_rejections, Ordering::Relaxed);
        self.tensor_oversized_message_rejections
            .store(snapshot.oversized_message_rejections, Ordering::Relaxed);
        self.tensor_current_inbound_queued_bytes
            .store(snapshot.current_inbound_queued_bytes, Ordering::Relaxed);
        self.tensor_peak_inbound_queued_bytes
            .store(snapshot.peak_inbound_queued_bytes, Ordering::Relaxed);
        self.tensor_current_outbound_inflight_bytes
            .store(snapshot.current_outbound_inflight_bytes, Ordering::Relaxed);
        self.tensor_peak_outbound_inflight_bytes
            .store(snapshot.peak_outbound_inflight_bytes, Ordering::Relaxed);
        self.tensor_current_outbound_connections
            .store(snapshot.current_outbound_connections, Ordering::Relaxed);
    }

    /// Get total jobs (completed + failed)
    pub fn total_jobs(&self) -> u64 {
        self.jobs_completed.load(Ordering::Relaxed) + self.jobs_failed.load(Ordering::Relaxed)
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_jobs();
        if total == 0 {
            return 0.0;
        }
        self.jobs_completed.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get average tokens per second
    pub fn avg_tokens_per_second(&self) -> f64 {
        let total_time_s = self.total_inference_time_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        if total_time_s == 0.0 {
            return 0.0;
        }
        self.total_tokens_generated.load(Ordering::Relaxed) as f64 / total_time_s
    }

    /// Get average all-reduce latency in milliseconds
    pub fn avg_allreduce_latency_ms(&self) -> f64 {
        let ops = self.allreduce_operations.load(Ordering::Relaxed);
        if ops == 0 {
            return 0.0;
        }
        self.total_allreduce_time_ms.load(Ordering::Relaxed) as f64 / ops as f64
    }

    /// Get average inference time per job in milliseconds
    pub fn avg_inference_time_ms(&self) -> f64 {
        let jobs = self.jobs_completed.load(Ordering::Relaxed);
        if jobs == 0 {
            return 0.0;
        }
        self.total_inference_time_ms.load(Ordering::Relaxed) as f64 / jobs as f64
    }

    /// Get uptime in seconds
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

    /// Print statistics summary
    pub fn print_summary(&self) {
        let jobs_completed = self.jobs_completed.load(Ordering::Relaxed);
        let jobs_failed = self.jobs_failed.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        let checkpoints = self.checkpoints_created.load(Ordering::Relaxed);
        let recoveries = self.checkpoint_recoveries.load(Ordering::Relaxed);
        let recovery_attempts = self.recovery_attempts.load(Ordering::Relaxed);

        info!(
            jobs_completed = jobs_completed,
            jobs_failed = jobs_failed,
            total_tokens = total_tokens,
            success_rate = format!("{:.1}%", self.success_rate() * 100.0),
            avg_tokens_per_second = format!("{:.2}", self.avg_tokens_per_second()),
            avg_allreduce_latency_ms = format!("{:.2}", self.avg_allreduce_latency_ms()),
            checkpoints_created = checkpoints,
            checkpoint_recoveries = recoveries,
            recovery_attempts = recovery_attempts,
            tensor_bytes_sent = self.tensor_bytes_sent.load(Ordering::Relaxed),
            tensor_bytes_received = self.tensor_bytes_received.load(Ordering::Relaxed),
            tensor_reduce_scatter_bytes_sent = self
                .tensor_reduce_scatter_bytes_sent
                .load(Ordering::Relaxed),
            tensor_all_gather_bytes_sent = self
                .tensor_all_gather_bytes_sent
                .load(Ordering::Relaxed),
            tensor_backpressure_waits = self
                .tensor_outbound_backpressure_wait_count
                .load(Ordering::Relaxed),
            tensor_bandwidth_waits = self
                .tensor_outbound_bandwidth_wait_count
                .load(Ordering::Relaxed),
            tensor_queue_rejections = self
                .tensor_inbound_queue_full_rejections
                .load(Ordering::Relaxed),
            tensor_byte_budget_rejections = self
                .tensor_inbound_byte_budget_rejections
                .load(Ordering::Relaxed),
            uptime = %self.uptime_string(),
            "Inference statistics"
        );
    }

    /// Display formatted metrics (for CLI)
    pub fn display(&self) {
        use colored::Colorize;

        println!("\n{}", "Inference Metrics".bold().cyan());
        println!("{}", "=================".cyan());

        println!("\n{}", "Job Statistics:".bold());
        println!(
            "  Completed:           {}",
            self.jobs_completed
                .load(Ordering::Relaxed)
                .to_string()
                .green()
        );
        println!(
            "  Failed:              {}",
            self.jobs_failed.load(Ordering::Relaxed).to_string().red()
        );
        println!("  Success Rate:        {:.1}%", self.success_rate() * 100.0);

        println!("\n{}", "Token Statistics:".bold());
        println!(
            "  Tokens Generated:    {}",
            self.total_tokens_generated.load(Ordering::Relaxed)
        );
        println!(
            "  Prompt Tokens:       {}",
            self.total_prompt_tokens.load(Ordering::Relaxed)
        );
        println!("  Avg Tokens/sec:      {:.2}", self.avg_tokens_per_second());

        println!("\n{}", "Ring Performance:".bold());
        println!(
            "  All-Reduce Ops:      {}",
            self.allreduce_operations.load(Ordering::Relaxed)
        );
        println!(
            "  Avg Latency:         {:.2}ms",
            self.avg_allreduce_latency_ms()
        );
        println!(
            "  Layers Processed:    {}",
            self.total_layers_processed.load(Ordering::Relaxed)
        );
        println!(
            "  Tensor Bytes Sent:   {}",
            self.tensor_bytes_sent.load(Ordering::Relaxed)
        );
        println!(
            "  Tensor Bytes Recv:   {}",
            self.tensor_bytes_received.load(Ordering::Relaxed)
        );
        println!(
            "  Send Waits:          {}",
            self.tensor_outbound_backpressure_wait_count
                .load(Ordering::Relaxed)
        );
        println!(
            "  Send Wait Time:      {}ms",
            self.tensor_outbound_backpressure_wait_ms
                .load(Ordering::Relaxed)
        );
        println!(
            "  Bandwidth Waits:     {}",
            self.tensor_outbound_bandwidth_wait_count
                .load(Ordering::Relaxed)
        );
        println!(
            "  Bandwidth Wait Time: {}ms",
            self.tensor_outbound_bandwidth_wait_ms
                .load(Ordering::Relaxed)
        );
        println!(
            "  Inbound Queue Drops: {}",
            self.tensor_inbound_queue_full_rejections
                .load(Ordering::Relaxed)
        );
        println!(
            "  Byte Budget Drops:   {}",
            self.tensor_inbound_byte_budget_rejections
                .load(Ordering::Relaxed)
        );
        println!(
            "  Oversize Drops:      {}",
            self.tensor_oversized_message_rejections
                .load(Ordering::Relaxed)
        );
        println!(
            "  Open Connections:    {}",
            self.tensor_current_outbound_connections
                .load(Ordering::Relaxed)
        );

        println!("\n{}", "Fault Tolerance:".bold());
        println!(
            "  Checkpoints:         {}",
            self.checkpoints_created.load(Ordering::Relaxed)
        );
        println!(
            "  Recoveries:          {}",
            self.checkpoint_recoveries.load(Ordering::Relaxed)
        );
        println!(
            "  Recovery Attempts:   {}",
            self.recovery_attempts.load(Ordering::Relaxed)
        );
        println!(
            "  Recovery Cooldowns:  {}",
            self.recovery_cooldown_rejections.load(Ordering::Relaxed)
        );
        println!(
            "  Recovery Budget Hit: {}",
            self.recovery_budget_rejections.load(Ordering::Relaxed)
        );
        println!(
            "  Recovery Misses:     {}",
            self.recovery_checkpoint_misses.load(Ordering::Relaxed)
        );

        println!("\n{}", "System:".bold());
        println!("  Uptime:              {}", self.uptime_string());
        println!();
    }

    /// Serialize to JSON for persistence
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        let insert_u64 =
            |map: &mut serde_json::Map<String, serde_json::Value>, key: &str, value: u64| {
                map.insert(key.to_string(), serde_json::Value::from(value));
            };

        insert_u64(
            &mut map,
            "jobs_completed",
            self.jobs_completed.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "jobs_failed",
            self.jobs_failed.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_tokens_generated",
            self.total_tokens_generated.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_prompt_tokens",
            self.total_prompt_tokens.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_inference_time_ms",
            self.total_inference_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_allreduce_time_ms",
            self.total_allreduce_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "allreduce_operations",
            self.allreduce_operations.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_layers_processed",
            self.total_layers_processed.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "checkpoints_created",
            self.checkpoints_created.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "checkpoint_recoveries",
            self.checkpoint_recoveries.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "recovery_attempts",
            self.recovery_attempts.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "recovery_cooldown_rejections",
            self.recovery_cooldown_rejections.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "recovery_budget_rejections",
            self.recovery_budget_rejections.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "recovery_checkpoint_misses",
            self.recovery_checkpoint_misses.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_bytes_sent",
            self.tensor_bytes_sent.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_bytes_received",
            self.tensor_bytes_received.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_reduce_scatter_bytes_sent",
            self.tensor_reduce_scatter_bytes_sent
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_reduce_scatter_bytes_received",
            self.tensor_reduce_scatter_bytes_received
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_all_gather_bytes_sent",
            self.tensor_all_gather_bytes_sent.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_all_gather_bytes_received",
            self.tensor_all_gather_bytes_received
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_barrier_bytes_sent",
            self.tensor_barrier_bytes_sent.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_barrier_bytes_received",
            self.tensor_barrier_bytes_received.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_outbound_backpressure_wait_count",
            self.tensor_outbound_backpressure_wait_count
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_outbound_backpressure_wait_ms",
            self.tensor_outbound_backpressure_wait_ms
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_outbound_bandwidth_wait_count",
            self.tensor_outbound_bandwidth_wait_count
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_outbound_bandwidth_wait_ms",
            self.tensor_outbound_bandwidth_wait_ms
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_send_count",
            self.tensor_send_count.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_send_latency_ms",
            self.tensor_send_latency_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_receive_count",
            self.tensor_receive_count.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_receive_latency_ms",
            self.tensor_receive_latency_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_receive_queue_wait_ms",
            self.tensor_receive_queue_wait_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_send_timeout_count",
            self.tensor_send_timeout_count.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_receive_timeout_count",
            self.tensor_receive_timeout_count.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_inbound_queue_full_rejections",
            self.tensor_inbound_queue_full_rejections
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_inbound_byte_budget_rejections",
            self.tensor_inbound_byte_budget_rejections
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_oversized_message_rejections",
            self.tensor_oversized_message_rejections
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_current_inbound_queued_bytes",
            self.tensor_current_inbound_queued_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_peak_inbound_queued_bytes",
            self.tensor_peak_inbound_queued_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_current_outbound_inflight_bytes",
            self.tensor_current_outbound_inflight_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_peak_outbound_inflight_bytes",
            self.tensor_peak_outbound_inflight_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_current_outbound_connections",
            self.tensor_current_outbound_connections
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_reduce_scatter_time_ms",
            self.total_reduce_scatter_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_all_gather_time_ms",
            self.total_all_gather_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_allreduce_send_wait_ms",
            self.total_allreduce_send_wait_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_allreduce_receive_wait_ms",
            self.total_allreduce_receive_wait_ms.load(Ordering::Relaxed),
        );

        map.insert(
            "success_rate".to_string(),
            serde_json::Value::from(self.success_rate()),
        );
        map.insert(
            "avg_tokens_per_second".to_string(),
            serde_json::Value::from(self.avg_tokens_per_second()),
        );
        map.insert(
            "avg_allreduce_latency_ms".to_string(),
            serde_json::Value::from(self.avg_allreduce_latency_ms()),
        );
        map.insert(
            "uptime".to_string(),
            serde_json::Value::from(self.uptime_string()),
        );
        map.insert(
            "last_updated".to_string(),
            serde_json::Value::from(chrono::Local::now().to_rfc3339()),
        );

        serde_json::Value::Object(map)
    }

    /// Save statistics to file
    pub fn save_to_file(&self) -> std::io::Result<()> {
        let stats_path = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".meshnet")
            .join("inference_stats.json");

        if let Some(parent) = stats_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.to_json())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        std::fs::write(&stats_path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_new() {
        let stats = InferenceStats::new();
        assert_eq!(stats.total_jobs(), 0);
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.avg_tokens_per_second(), 0.0);
    }

    #[test]
    fn test_stats_record_success() {
        let stats = InferenceStats::new();

        stats.record_success(10, 50, 1000);
        stats.record_success(20, 100, 2000);

        assert_eq!(stats.jobs_completed.load(Ordering::Relaxed), 2);
        assert_eq!(stats.total_tokens_generated.load(Ordering::Relaxed), 150);
        assert_eq!(stats.total_prompt_tokens.load(Ordering::Relaxed), 30);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.avg_tokens_per_second(), 50.0); // 150 tokens / 3 seconds
    }

    #[test]
    fn test_stats_record_failure() {
        let stats = InferenceStats::new();

        stats.record_success(10, 50, 1000);
        stats.record_failure();

        assert_eq!(stats.total_jobs(), 2);
        assert_eq!(stats.success_rate(), 0.5);
    }

    #[test]
    fn test_stats_allreduce() {
        let stats = InferenceStats::new();

        stats.record_allreduce(10);
        stats.record_allreduce(20);
        stats.record_allreduce(30);

        assert_eq!(stats.allreduce_operations.load(Ordering::Relaxed), 3);
        assert_eq!(stats.avg_allreduce_latency_ms(), 20.0);
    }

    #[test]
    fn test_stats_checkpoints() {
        let stats = InferenceStats::new();

        stats.record_checkpoint();
        stats.record_checkpoint();
        stats.record_recovery();
        stats.record_recovery_attempt();
        stats.record_recovery_cooldown_rejection();
        stats.record_recovery_budget_rejection();
        stats.record_recovery_checkpoint_miss();

        assert_eq!(stats.checkpoints_created.load(Ordering::Relaxed), 2);
        assert_eq!(stats.checkpoint_recoveries.load(Ordering::Relaxed), 1);
        assert_eq!(stats.recovery_attempts.load(Ordering::Relaxed), 1);
        assert_eq!(
            stats.recovery_cooldown_rejections.load(Ordering::Relaxed),
            1
        );
        assert_eq!(stats.recovery_budget_rejections.load(Ordering::Relaxed), 1);
        assert_eq!(stats.recovery_checkpoint_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats_to_json() {
        let stats = InferenceStats::new();
        stats.record_success(10, 50, 1000);

        let json = stats.to_json();

        assert_eq!(json["jobs_completed"], 1);
        assert_eq!(json["total_tokens_generated"], 50);
    }

    #[test]
    fn test_stats_update_tensor_plane_metrics() {
        let stats = InferenceStats::new();
        stats.update_tensor_plane_metrics(TensorPlaneMetricsSnapshot {
            bytes_sent: 128,
            bytes_received: 256,
            reduce_scatter_bytes_sent: 32,
            reduce_scatter_bytes_received: 64,
            all_gather_bytes_sent: 48,
            all_gather_bytes_received: 96,
            barrier_bytes_sent: 8,
            barrier_bytes_received: 16,
            outbound_backpressure_wait_count: 3,
            outbound_backpressure_wait_ms: 42,
            outbound_bandwidth_wait_count: 4,
            outbound_bandwidth_wait_ms: 55,
            send_count: 6,
            send_latency_ms: 77,
            receive_count: 9,
            receive_latency_ms: 88,
            receive_queue_wait_ms: 21,
            send_timeout_count: 1,
            receive_timeout_count: 2,
            inbound_queue_full_rejections: 5,
            inbound_byte_budget_rejections: 7,
            oversized_message_rejections: 11,
            current_inbound_queued_bytes: 13,
            peak_inbound_queued_bytes: 19,
            current_outbound_inflight_bytes: 17,
            peak_outbound_inflight_bytes: 23,
            current_outbound_connections: 2,
            latency_critical_send_count: 1,
            interactive_send_count: 2,
            bulk_send_count: 3,
        });

        assert_eq!(stats.tensor_bytes_sent.load(Ordering::Relaxed), 128);
        assert_eq!(stats.tensor_bytes_received.load(Ordering::Relaxed), 256);
        assert_eq!(
            stats
                .tensor_outbound_backpressure_wait_count
                .load(Ordering::Relaxed),
            3
        );
        assert_eq!(
            stats
                .tensor_outbound_bandwidth_wait_count
                .load(Ordering::Relaxed),
            4
        );
        assert_eq!(
            stats
                .tensor_inbound_queue_full_rejections
                .load(Ordering::Relaxed),
            5
        );
        assert_eq!(
            stats
                .tensor_inbound_byte_budget_rejections
                .load(Ordering::Relaxed),
            7
        );
    }
}
