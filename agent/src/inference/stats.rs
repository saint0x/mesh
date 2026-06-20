//! Inference statistics tracking
//!
//! This module provides metrics tracking for distributed inference operations.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, OnceLock, Weak,
};
use std::time::Instant;
use tracing::info;

use crate::executor::ring_allreduce::RingAllReduceMetrics;
use crate::network::TensorPlaneMetricsSnapshot;

use super::fast_path::{FastPathExecutionPlan, GraphCaptureStrategy};

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

    /// Number of decode microbatches executed by the runtime.
    pub decode_microbatches_executed: AtomicU64,

    /// Total number of decode sessions admitted across all microbatches.
    pub decode_sessions_batched: AtomicU64,

    /// Number of decode microbatches that contained more than one session.
    pub decode_multi_session_microbatches: AtomicU64,

    /// Peak decode batch size observed by the runtime.
    pub decode_batch_size_peak: AtomicU64,

    /// Total KV-token footprint admitted across decode microbatches.
    pub decode_batch_kv_tokens_total: AtomicU64,

    /// Total number of queued decode sessions deferred from admitted batches.
    pub decode_batch_deferred_sessions: AtomicU64,

    /// Number of decode sessions deferred because the batch was already full.
    pub decode_batch_capacity_deferrals: AtomicU64,

    /// Number of decode sessions deferred because the KV-token budget would be exceeded.
    pub decode_batch_kv_budget_deferrals: AtomicU64,

    /// Number of decode sessions deferred to preserve fast-path bucket or cohort cohesion.
    pub decode_batch_guardrail_deferrals: AtomicU64,

    /// Number of prefill executions that resolved an explicit fast-path bucket plan.
    pub prefill_fast_path_plans: AtomicU64,

    /// Number of decode microbatches that resolved an explicit fast-path bucket plan.
    pub decode_fast_path_plans: AtomicU64,

    /// Number of fast-path executions that reused a previously reserved arena.
    pub fast_path_arena_reuses: AtomicU64,

    /// Number of fast-path executions that were layout validated without graph replay.
    pub fast_path_layout_validated_plans: AtomicU64,

    /// Number of fast-path executions that used replay-preferred bucket strategies.
    pub fast_path_replay_preferred_plans: AtomicU64,

    /// Peak decode bucket batch-size ceiling admitted by the planner.
    pub fast_path_decode_bucket_batch_ceiling_peak: AtomicU64,

    /// Peak decode bucket KV-token ceiling admitted by the planner.
    pub fast_path_decode_bucket_token_ceiling_peak: AtomicU64,

    /// Peak prefill token ceiling admitted by the planner.
    pub fast_path_prefill_token_ceiling_peak: AtomicU64,

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
    pub tensor_bulk_transfer_bytes_sent: AtomicU64,
    pub tensor_bulk_transfer_bytes_received: AtomicU64,
    pub tensor_checkpoint_bytes_sent: AtomicU64,
    pub tensor_checkpoint_bytes_received: AtomicU64,

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
    pub tensor_connection_refresh_attempt_count: AtomicU64,
    pub tensor_connection_refresh_success_count: AtomicU64,
    pub tensor_connection_evict_count: AtomicU64,

    pub total_reduce_scatter_time_ms: AtomicU64,
    pub total_all_gather_time_ms: AtomicU64,
    pub total_allreduce_send_wait_ms: AtomicU64,
    pub total_allreduce_receive_wait_ms: AtomicU64,
    pub total_collective_operations: AtomicU64,
    pub total_collective_worker_participants: AtomicU64,
    pub total_pairwise_fast_path_operations: AtomicU64,
    pub total_larger_ring_operations: AtomicU64,
    pub total_collective_bytes_sent: AtomicU64,
    pub total_collective_bytes_received: AtomicU64,
    pub total_collective_host_materializations: AtomicU64,
    pub total_collective_host_materialization_bytes: AtomicU64,
    pub total_device_resident_collectives: AtomicU64,

    /// Number of accelerated collective submissions that staged through host memory.
    pub collective_host_stage_to_host_events: AtomicU64,

    /// Number of bytes copied from accelerated device memory into host-resident collective buffers.
    pub collective_host_stage_to_host_bytes: AtomicU64,

    /// Number of accelerated collective results restored from host-resident buffers back onto device.
    pub collective_host_restore_to_device_events: AtomicU64,

    /// Number of bytes restored from host-resident collective buffers back onto device.
    pub collective_host_restore_to_device_bytes: AtomicU64,

    /// Number of KV snapshot payloads exported by the runtime.
    pub kv_snapshot_exports: AtomicU64,

    /// Total bytes serialized into KV snapshot payloads.
    pub kv_snapshot_export_bytes: AtomicU64,

    /// Number of KV snapshot payloads materialized back into runtime cache state.
    pub kv_snapshot_materializations: AtomicU64,

    /// Total bytes materialized back into runtime cache state from KV snapshot payloads.
    pub kv_snapshot_materialized_bytes: AtomicU64,

    /// Number of token rows successfully sampled on device.
    pub device_sampling_requests: AtomicU64,

    /// Number of token rows that attempted device sampling but fell back before completion.
    pub device_sampling_fallback_requests: AtomicU64,

    /// Number of token rows sampled through host logic.
    pub host_sampling_requests: AtomicU64,

    /// Total time spent in all sampling paths.
    pub total_sampling_time_ms: AtomicU64,

    /// Total time spent in device sampling paths.
    pub device_sampling_time_ms: AtomicU64,

    /// Total time spent in host sampling paths.
    pub host_sampling_time_ms: AtomicU64,

    /// Number of active KV view accesses served from the existing cache.
    pub device_kv_active_view_cache_hits: AtomicU64,

    /// Number of active KV view accesses that had to rebuild the view.
    pub device_kv_active_view_cache_misses: AtomicU64,

    /// Number of head-expanded KV view accesses served from the existing cache.
    pub device_kv_head_view_cache_hits: AtomicU64,

    /// Number of head-expanded KV view accesses that had to rebuild the view.
    pub device_kv_head_view_cache_misses: AtomicU64,

    /// Number of selected-head KV view accesses served from the existing cache.
    pub device_kv_selected_head_view_cache_hits: AtomicU64,

    /// Number of selected-head KV view accesses that had to rebuild the view.
    pub device_kv_selected_head_view_cache_misses: AtomicU64,
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
            decode_microbatches_executed: AtomicU64::new(0),
            decode_sessions_batched: AtomicU64::new(0),
            decode_multi_session_microbatches: AtomicU64::new(0),
            decode_batch_size_peak: AtomicU64::new(0),
            decode_batch_kv_tokens_total: AtomicU64::new(0),
            decode_batch_deferred_sessions: AtomicU64::new(0),
            decode_batch_capacity_deferrals: AtomicU64::new(0),
            decode_batch_kv_budget_deferrals: AtomicU64::new(0),
            decode_batch_guardrail_deferrals: AtomicU64::new(0),
            prefill_fast_path_plans: AtomicU64::new(0),
            decode_fast_path_plans: AtomicU64::new(0),
            fast_path_arena_reuses: AtomicU64::new(0),
            fast_path_layout_validated_plans: AtomicU64::new(0),
            fast_path_replay_preferred_plans: AtomicU64::new(0),
            fast_path_decode_bucket_batch_ceiling_peak: AtomicU64::new(0),
            fast_path_decode_bucket_token_ceiling_peak: AtomicU64::new(0),
            fast_path_prefill_token_ceiling_peak: AtomicU64::new(0),
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
            tensor_bulk_transfer_bytes_sent: AtomicU64::new(0),
            tensor_bulk_transfer_bytes_received: AtomicU64::new(0),
            tensor_checkpoint_bytes_sent: AtomicU64::new(0),
            tensor_checkpoint_bytes_received: AtomicU64::new(0),
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
            tensor_connection_refresh_attempt_count: AtomicU64::new(0),
            tensor_connection_refresh_success_count: AtomicU64::new(0),
            tensor_connection_evict_count: AtomicU64::new(0),
            total_reduce_scatter_time_ms: AtomicU64::new(0),
            total_all_gather_time_ms: AtomicU64::new(0),
            total_allreduce_send_wait_ms: AtomicU64::new(0),
            total_allreduce_receive_wait_ms: AtomicU64::new(0),
            total_collective_operations: AtomicU64::new(0),
            total_collective_worker_participants: AtomicU64::new(0),
            total_pairwise_fast_path_operations: AtomicU64::new(0),
            total_larger_ring_operations: AtomicU64::new(0),
            total_collective_bytes_sent: AtomicU64::new(0),
            total_collective_bytes_received: AtomicU64::new(0),
            total_collective_host_materializations: AtomicU64::new(0),
            total_collective_host_materialization_bytes: AtomicU64::new(0),
            total_device_resident_collectives: AtomicU64::new(0),
            collective_host_stage_to_host_events: AtomicU64::new(0),
            collective_host_stage_to_host_bytes: AtomicU64::new(0),
            collective_host_restore_to_device_events: AtomicU64::new(0),
            collective_host_restore_to_device_bytes: AtomicU64::new(0),
            kv_snapshot_exports: AtomicU64::new(0),
            kv_snapshot_export_bytes: AtomicU64::new(0),
            kv_snapshot_materializations: AtomicU64::new(0),
            kv_snapshot_materialized_bytes: AtomicU64::new(0),
            device_sampling_requests: AtomicU64::new(0),
            device_sampling_fallback_requests: AtomicU64::new(0),
            host_sampling_requests: AtomicU64::new(0),
            total_sampling_time_ms: AtomicU64::new(0),
            device_sampling_time_ms: AtomicU64::new(0),
            host_sampling_time_ms: AtomicU64::new(0),
            device_kv_active_view_cache_hits: AtomicU64::new(0),
            device_kv_active_view_cache_misses: AtomicU64::new(0),
            device_kv_head_view_cache_hits: AtomicU64::new(0),
            device_kv_head_view_cache_misses: AtomicU64::new(0),
            device_kv_selected_head_view_cache_hits: AtomicU64::new(0),
            device_kv_selected_head_view_cache_misses: AtomicU64::new(0),
        }
    }

    pub fn install_as_runtime_collector(stats: &Arc<Self>) {
        let mut slot = active_runtime_stats_slot()
            .lock()
            .expect("active runtime stats mutex poisoned");
        *slot = Arc::downgrade(stats);
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

    pub fn record_decode_microbatch(
        &self,
        batch_size: usize,
        kv_tokens: usize,
        deferred_sessions: usize,
        deferred_for_capacity: usize,
        deferred_for_kv_budget: usize,
        deferred_for_guardrail: usize,
    ) {
        let batch_size = batch_size as u64;
        self.decode_microbatches_executed
            .fetch_add(1, Ordering::Relaxed);
        self.decode_sessions_batched
            .fetch_add(batch_size, Ordering::Relaxed);
        if batch_size > 1 {
            self.decode_multi_session_microbatches
                .fetch_add(1, Ordering::Relaxed);
        }
        self.decode_batch_kv_tokens_total
            .fetch_add(kv_tokens as u64, Ordering::Relaxed);
        self.decode_batch_deferred_sessions
            .fetch_add(deferred_sessions as u64, Ordering::Relaxed);
        self.decode_batch_capacity_deferrals
            .fetch_add(deferred_for_capacity as u64, Ordering::Relaxed);
        self.decode_batch_kv_budget_deferrals
            .fetch_add(deferred_for_kv_budget as u64, Ordering::Relaxed);
        self.decode_batch_guardrail_deferrals
            .fetch_add(deferred_for_guardrail as u64, Ordering::Relaxed);

        let mut current_peak = self.decode_batch_size_peak.load(Ordering::Relaxed);
        while batch_size > current_peak {
            match self.decode_batch_size_peak.compare_exchange_weak(
                current_peak,
                batch_size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => current_peak = observed,
            }
        }
    }

    pub fn record_prefill_fast_path_plan(
        &self,
        plan: &FastPathExecutionPlan,
        reused_existing_arena: bool,
    ) {
        self.prefill_fast_path_plans.fetch_add(1, Ordering::Relaxed);
        if reused_existing_arena {
            self.fast_path_arena_reuses.fetch_add(1, Ordering::Relaxed);
        }
        self.record_capture_strategy(plan.capture_strategy);
        self.update_peak(
            &self.fast_path_prefill_token_ceiling_peak,
            plan.bucket.token_ceiling as u64,
        );
    }

    pub fn record_decode_fast_path_plan(
        &self,
        plan: &FastPathExecutionPlan,
        reused_existing_arena: bool,
    ) {
        self.decode_fast_path_plans.fetch_add(1, Ordering::Relaxed);
        if reused_existing_arena {
            self.fast_path_arena_reuses.fetch_add(1, Ordering::Relaxed);
        }
        self.record_capture_strategy(plan.capture_strategy);
        self.update_peak(
            &self.fast_path_decode_bucket_batch_ceiling_peak,
            plan.bucket.batch_size_ceiling as u64,
        );
        self.update_peak(
            &self.fast_path_decode_bucket_token_ceiling_peak,
            plan.bucket.token_ceiling as u64,
        );
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
        self.total_collective_operations
            .fetch_add(metrics.collective_operations, Ordering::Relaxed);
        self.total_collective_worker_participants
            .fetch_add(metrics.collective_worker_participants, Ordering::Relaxed);
        self.total_pairwise_fast_path_operations
            .fetch_add(metrics.pairwise_fast_path_operations, Ordering::Relaxed);
        self.total_larger_ring_operations
            .fetch_add(metrics.larger_ring_operations, Ordering::Relaxed);
        self.total_collective_bytes_sent
            .fetch_add(metrics.bytes_sent, Ordering::Relaxed);
        self.total_collective_bytes_received
            .fetch_add(metrics.bytes_received, Ordering::Relaxed);
        self.total_collective_host_materializations
            .fetch_add(metrics.host_materialization_count, Ordering::Relaxed);
        self.total_collective_host_materialization_bytes
            .fetch_add(metrics.host_materialization_bytes, Ordering::Relaxed);
        self.total_device_resident_collectives
            .fetch_add(metrics.device_resident_collective_count, Ordering::Relaxed);
    }

    pub fn record_collective_host_stage_to_host(&self, bytes: u64) {
        self.collective_host_stage_to_host_events
            .fetch_add(1, Ordering::Relaxed);
        self.collective_host_stage_to_host_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_collective_host_restore_to_device(&self, bytes: u64) {
        self.collective_host_restore_to_device_events
            .fetch_add(1, Ordering::Relaxed);
        self.collective_host_restore_to_device_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_kv_snapshot_export(&self, bytes: u64) {
        self.kv_snapshot_exports.fetch_add(1, Ordering::Relaxed);
        self.kv_snapshot_export_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_kv_snapshot_materialization(&self, bytes: u64) {
        self.kv_snapshot_materializations
            .fetch_add(1, Ordering::Relaxed);
        self.kv_snapshot_materialized_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_device_sampling(&self, requests: u64, duration_ms: u64) {
        self.device_sampling_requests
            .fetch_add(requests, Ordering::Relaxed);
        self.device_sampling_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
        self.total_sampling_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    pub fn record_device_sampling_fallback(&self, requests: u64) {
        self.device_sampling_fallback_requests
            .fetch_add(requests, Ordering::Relaxed);
    }

    pub fn record_host_sampling(&self, requests: u64, duration_ms: u64) {
        self.host_sampling_requests
            .fetch_add(requests, Ordering::Relaxed);
        self.host_sampling_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
        self.total_sampling_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    pub fn record_device_kv_active_view_cache_hit(&self) {
        self.device_kv_active_view_cache_hits
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_device_kv_active_view_cache_miss(&self) {
        self.device_kv_active_view_cache_misses
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_device_kv_head_view_cache_hit(&self) {
        self.device_kv_head_view_cache_hits
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_device_kv_head_view_cache_miss(&self) {
        self.device_kv_head_view_cache_misses
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_device_kv_selected_head_view_cache_hit(&self) {
        self.device_kv_selected_head_view_cache_hits
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_device_kv_selected_head_view_cache_miss(&self) {
        self.device_kv_selected_head_view_cache_misses
            .fetch_add(1, Ordering::Relaxed);
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

    fn record_capture_strategy(&self, strategy: GraphCaptureStrategy) {
        match strategy {
            GraphCaptureStrategy::Unsupported | GraphCaptureStrategy::LayoutValidated => {
                self.fast_path_layout_validated_plans
                    .fetch_add(1, Ordering::Relaxed);
            }
            GraphCaptureStrategy::ReplayPreferred => {
                self.fast_path_replay_preferred_plans
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn update_peak(&self, metric: &AtomicU64, observed: u64) {
        let mut current_peak = metric.load(Ordering::Relaxed);
        while observed > current_peak {
            match metric.compare_exchange_weak(
                current_peak,
                observed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(latest) => current_peak = latest,
            }
        }
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
        self.tensor_bulk_transfer_bytes_sent
            .store(snapshot.bulk_transfer_bytes_sent, Ordering::Relaxed);
        self.tensor_bulk_transfer_bytes_received
            .store(snapshot.bulk_transfer_bytes_received, Ordering::Relaxed);
        self.tensor_checkpoint_bytes_sent
            .store(snapshot.checkpoint_bytes_sent, Ordering::Relaxed);
        self.tensor_checkpoint_bytes_received
            .store(snapshot.checkpoint_bytes_received, Ordering::Relaxed);
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
        self.tensor_connection_refresh_attempt_count
            .store(snapshot.connection_refresh_attempt_count, Ordering::Relaxed);
        self.tensor_connection_refresh_success_count
            .store(snapshot.connection_refresh_success_count, Ordering::Relaxed);
        self.tensor_connection_evict_count
            .store(snapshot.connection_evict_count, Ordering::Relaxed);
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

    pub fn avg_decode_batch_size(&self) -> f64 {
        let microbatches = self.decode_microbatches_executed.load(Ordering::Relaxed);
        if microbatches == 0 {
            return 0.0;
        }
        self.decode_sessions_batched.load(Ordering::Relaxed) as f64 / microbatches as f64
    }

    pub fn fast_path_decode_plan_rate(&self) -> f64 {
        let microbatches = self.decode_microbatches_executed.load(Ordering::Relaxed);
        if microbatches == 0 {
            return 0.0;
        }
        self.decode_fast_path_plans.load(Ordering::Relaxed) as f64 / microbatches as f64
    }

    pub fn multi_session_batch_rate(&self) -> f64 {
        let microbatches = self.decode_microbatches_executed.load(Ordering::Relaxed);
        if microbatches == 0 {
            return 0.0;
        }
        self.decode_multi_session_microbatches
            .load(Ordering::Relaxed) as f64
            / microbatches as f64
    }

    pub fn avg_deferred_sessions_per_microbatch(&self) -> f64 {
        let microbatches = self.decode_microbatches_executed.load(Ordering::Relaxed);
        if microbatches == 0 {
            return 0.0;
        }
        self.decode_batch_deferred_sessions.load(Ordering::Relaxed) as f64 / microbatches as f64
    }

    pub fn allreduce_send_wait_share(&self) -> f64 {
        let total = self.total_allreduce_time_ms.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_allreduce_send_wait_ms.load(Ordering::Relaxed) as f64 / total as f64
    }

    pub fn allreduce_receive_wait_share(&self) -> f64 {
        let total = self.total_allreduce_time_ms.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_allreduce_receive_wait_ms.load(Ordering::Relaxed) as f64 / total as f64
    }

    pub fn collective_transport_share_of_runtime(&self) -> f64 {
        let total_runtime = self.total_inference_time_ms.load(Ordering::Relaxed);
        if total_runtime == 0 {
            return 0.0;
        }
        let collective_ms = self.total_reduce_scatter_time_ms.load(Ordering::Relaxed)
            + self.total_all_gather_time_ms.load(Ordering::Relaxed)
            + self.total_allreduce_send_wait_ms.load(Ordering::Relaxed)
            + self.total_allreduce_receive_wait_ms.load(Ordering::Relaxed);
        collective_ms as f64 / total_runtime as f64
    }

    pub fn avg_collective_workers(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_collective_worker_participants
            .load(Ordering::Relaxed) as f64
            / operations as f64
    }

    pub fn larger_ring_collective_rate(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_larger_ring_operations.load(Ordering::Relaxed) as f64 / operations as f64
    }

    pub fn pairwise_fast_path_collective_rate(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_pairwise_fast_path_operations
            .load(Ordering::Relaxed) as f64
            / operations as f64
    }

    pub fn avg_collective_send_bytes(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_collective_bytes_sent.load(Ordering::Relaxed) as f64 / operations as f64
    }

    pub fn avg_collective_receive_bytes(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_collective_bytes_received.load(Ordering::Relaxed) as f64 / operations as f64
    }

    pub fn device_resident_collective_rate(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_device_resident_collectives
            .load(Ordering::Relaxed) as f64
            / operations as f64
    }

    pub fn collective_host_materialization_rate(&self) -> f64 {
        let operations = self.total_collective_operations.load(Ordering::Relaxed);
        if operations == 0 {
            return 0.0;
        }
        self.total_collective_host_materializations
            .load(Ordering::Relaxed) as f64
            / operations as f64
    }

    pub fn collective_host_staging_events(&self) -> u64 {
        self.collective_host_stage_to_host_events
            .load(Ordering::Relaxed)
            + self
                .collective_host_restore_to_device_events
                .load(Ordering::Relaxed)
    }

    pub fn collective_host_staging_bytes(&self) -> u64 {
        self.collective_host_stage_to_host_bytes
            .load(Ordering::Relaxed)
            + self
                .collective_host_restore_to_device_bytes
                .load(Ordering::Relaxed)
    }

    pub fn sampling_share_of_runtime(&self) -> f64 {
        let total_runtime = self.total_inference_time_ms.load(Ordering::Relaxed);
        if total_runtime == 0 {
            return 0.0;
        }
        self.total_sampling_time_ms.load(Ordering::Relaxed) as f64 / total_runtime as f64
    }

    pub fn device_sampling_fallback_rate(&self) -> f64 {
        let attempted = self.device_sampling_requests.load(Ordering::Relaxed)
            + self
                .device_sampling_fallback_requests
                .load(Ordering::Relaxed);
        if attempted == 0 {
            return 0.0;
        }
        self.device_sampling_fallback_requests
            .load(Ordering::Relaxed) as f64
            / attempted as f64
    }

    pub fn avg_generated_token_latency_ms(&self) -> f64 {
        let generated_tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        if generated_tokens == 0 {
            return 0.0;
        }
        self.total_inference_time_ms.load(Ordering::Relaxed) as f64 / generated_tokens as f64
    }

    pub fn kv_snapshot_avg_export_bytes(&self) -> f64 {
        let exports = self.kv_snapshot_exports.load(Ordering::Relaxed);
        if exports == 0 {
            return 0.0;
        }
        self.kv_snapshot_export_bytes.load(Ordering::Relaxed) as f64 / exports as f64
    }

    pub fn kv_snapshot_avg_materialized_bytes(&self) -> f64 {
        let materializations = self.kv_snapshot_materializations.load(Ordering::Relaxed);
        if materializations == 0 {
            return 0.0;
        }
        self.kv_snapshot_materialized_bytes.load(Ordering::Relaxed) as f64 / materializations as f64
    }

    pub fn device_kv_active_view_cache_hit_rate(&self) -> f64 {
        let hits = self
            .device_kv_active_view_cache_hits
            .load(Ordering::Relaxed);
        let misses = self
            .device_kv_active_view_cache_misses
            .load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    pub fn device_kv_head_view_cache_hit_rate(&self) -> f64 {
        let hits = self.device_kv_head_view_cache_hits.load(Ordering::Relaxed);
        let misses = self
            .device_kv_head_view_cache_misses
            .load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    pub fn device_kv_selected_head_view_cache_hit_rate(&self) -> f64 {
        let hits = self
            .device_kv_selected_head_view_cache_hits
            .load(Ordering::Relaxed);
        let misses = self
            .device_kv_selected_head_view_cache_misses
            .load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }

    pub fn collective_wait_share_of_collective_runtime(&self) -> f64 {
        let total_collective = self.total_reduce_scatter_time_ms.load(Ordering::Relaxed)
            + self.total_all_gather_time_ms.load(Ordering::Relaxed)
            + self.total_allreduce_send_wait_ms.load(Ordering::Relaxed)
            + self.total_allreduce_receive_wait_ms.load(Ordering::Relaxed);
        if total_collective == 0 {
            return 0.0;
        }
        let total_wait = self.total_allreduce_send_wait_ms.load(Ordering::Relaxed)
            + self.total_allreduce_receive_wait_ms.load(Ordering::Relaxed);
        total_wait as f64 / total_collective as f64
    }

    pub fn recovery_success_rate(&self) -> f64 {
        let attempts = self.recovery_attempts.load(Ordering::Relaxed);
        if attempts == 0 {
            return 0.0;
        }
        self.checkpoint_recoveries.load(Ordering::Relaxed) as f64 / attempts as f64
    }

    pub fn recovery_rejection_rate(&self) -> f64 {
        let attempts = self.recovery_attempts.load(Ordering::Relaxed);
        if attempts == 0 {
            return 0.0;
        }
        let rejections = self.recovery_cooldown_rejections.load(Ordering::Relaxed)
            + self.recovery_budget_rejections.load(Ordering::Relaxed);
        rejections as f64 / attempts as f64
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
        let decode_microbatches = self.decode_microbatches_executed.load(Ordering::Relaxed);

        info!(
            jobs_completed = jobs_completed,
            jobs_failed = jobs_failed,
            total_tokens = total_tokens,
            success_rate = format!("{:.1}%", self.success_rate() * 100.0),
            avg_tokens_per_second = format!("{:.2}", self.avg_tokens_per_second()),
            avg_allreduce_latency_ms = format!("{:.2}", self.avg_allreduce_latency_ms()),
            decode_microbatches_executed = decode_microbatches,
            avg_decode_batch_size = format!("{:.2}", self.avg_decode_batch_size()),
            fast_path_decode_plan_rate = format!("{:.3}", self.fast_path_decode_plan_rate()),
            multi_session_batch_rate = format!("{:.3}", self.multi_session_batch_rate()),
            avg_deferred_sessions_per_microbatch = format!(
                "{:.2}",
                self.avg_deferred_sessions_per_microbatch()
            ),
            allreduce_send_wait_share = format!("{:.3}", self.allreduce_send_wait_share()),
            allreduce_receive_wait_share = format!("{:.3}", self.allreduce_receive_wait_share()),
            collective_wait_share_of_collective_runtime = format!(
                "{:.3}",
                self.collective_wait_share_of_collective_runtime()
            ),
            collective_transport_share_of_runtime = format!(
                "{:.3}",
                self.collective_transport_share_of_runtime()
            ),
            avg_collective_workers = format!("{:.2}", self.avg_collective_workers()),
            larger_ring_collective_rate = format!("{:.3}", self.larger_ring_collective_rate()),
            pairwise_fast_path_collective_rate = format!(
                "{:.3}",
                self.pairwise_fast_path_collective_rate()
            ),
            device_resident_collective_rate = format!(
                "{:.3}",
                self.device_resident_collective_rate()
            ),
            collective_host_materialization_rate = format!(
                "{:.3}",
                self.collective_host_materialization_rate()
            ),
            collective_host_staging_events = self.collective_host_staging_events(),
            collective_host_staging_bytes = self.collective_host_staging_bytes(),
            decode_multi_session_microbatches = self
                .decode_multi_session_microbatches
                .load(Ordering::Relaxed),
            decode_batch_size_peak = self.decode_batch_size_peak.load(Ordering::Relaxed),
            decode_batch_kv_tokens_total = self
                .decode_batch_kv_tokens_total
                .load(Ordering::Relaxed),
            decode_batch_deferred_sessions = self
                .decode_batch_deferred_sessions
                .load(Ordering::Relaxed),
            decode_batch_capacity_deferrals = self
                .decode_batch_capacity_deferrals
                .load(Ordering::Relaxed),
            decode_batch_kv_budget_deferrals = self
                .decode_batch_kv_budget_deferrals
                .load(Ordering::Relaxed),
            decode_batch_guardrail_deferrals = self
                .decode_batch_guardrail_deferrals
                .load(Ordering::Relaxed),
            prefill_fast_path_plans = self.prefill_fast_path_plans.load(Ordering::Relaxed),
            decode_fast_path_plans = self.decode_fast_path_plans.load(Ordering::Relaxed),
            fast_path_arena_reuses = self.fast_path_arena_reuses.load(Ordering::Relaxed),
            fast_path_layout_validated_plans = self
                .fast_path_layout_validated_plans
                .load(Ordering::Relaxed),
            fast_path_replay_preferred_plans = self
                .fast_path_replay_preferred_plans
                .load(Ordering::Relaxed),
            fast_path_decode_bucket_batch_ceiling_peak = self
                .fast_path_decode_bucket_batch_ceiling_peak
                .load(Ordering::Relaxed),
            fast_path_decode_bucket_token_ceiling_peak = self
                .fast_path_decode_bucket_token_ceiling_peak
                .load(Ordering::Relaxed),
            fast_path_prefill_token_ceiling_peak = self
                .fast_path_prefill_token_ceiling_peak
                .load(Ordering::Relaxed),
            checkpoints_created = checkpoints,
            checkpoint_recoveries = recoveries,
            recovery_attempts = recovery_attempts,
            recovery_success_rate = format!("{:.3}", self.recovery_success_rate()),
            recovery_rejection_rate = format!("{:.3}", self.recovery_rejection_rate()),
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
            tensor_connection_refresh_attempts = self
                .tensor_connection_refresh_attempt_count
                .load(Ordering::Relaxed),
            tensor_connection_refresh_successes = self
                .tensor_connection_refresh_success_count
                .load(Ordering::Relaxed),
            tensor_connection_evicts = self
                .tensor_connection_evict_count
                .load(Ordering::Relaxed),
            kv_snapshot_exports = self.kv_snapshot_exports.load(Ordering::Relaxed),
            kv_snapshot_export_bytes = self.kv_snapshot_export_bytes.load(Ordering::Relaxed),
            kv_snapshot_materializations = self.kv_snapshot_materializations.load(Ordering::Relaxed),
            kv_snapshot_materialized_bytes = self
                .kv_snapshot_materialized_bytes
                .load(Ordering::Relaxed),
            kv_snapshot_avg_export_bytes = format!("{:.2}", self.kv_snapshot_avg_export_bytes()),
            kv_snapshot_avg_materialized_bytes = format!(
                "{:.2}",
                self.kv_snapshot_avg_materialized_bytes()
            ),
            device_sampling_requests = self.device_sampling_requests.load(Ordering::Relaxed),
            device_sampling_fallback_requests = self
                .device_sampling_fallback_requests
                .load(Ordering::Relaxed),
            host_sampling_requests = self.host_sampling_requests.load(Ordering::Relaxed),
            total_sampling_time_ms = self.total_sampling_time_ms.load(Ordering::Relaxed),
            sampling_share_of_runtime = format!("{:.3}", self.sampling_share_of_runtime()),
            device_sampling_fallback_rate = format!(
                "{:.3}",
                self.device_sampling_fallback_rate()
            ),
            avg_generated_token_latency_ms = format!(
                "{:.3}",
                self.avg_generated_token_latency_ms()
            ),
            device_kv_active_view_cache_hit_rate = format!(
                "{:.3}",
                self.device_kv_active_view_cache_hit_rate()
            ),
            device_kv_head_view_cache_hit_rate = format!(
                "{:.3}",
                self.device_kv_head_view_cache_hit_rate()
            ),
            device_kv_selected_head_view_cache_hit_rate = format!(
                "{:.3}",
                self.device_kv_selected_head_view_cache_hit_rate()
            ),
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
        println!(
            "  Conn Refreshes:      {}",
            self.tensor_connection_refresh_attempt_count
                .load(Ordering::Relaxed)
        );
        println!(
            "  Conn Refresh OK:     {}",
            self.tensor_connection_refresh_success_count
                .load(Ordering::Relaxed)
        );
        println!(
            "  Conn Evictions:      {}",
            self.tensor_connection_evict_count.load(Ordering::Relaxed)
        );
        println!(
            "  Avg Coll Workers:    {:.2}",
            self.avg_collective_workers()
        );
        println!(
            "  Larger-Ring Rate:    {:.1}%",
            self.larger_ring_collective_rate() * 100.0
        );
        println!(
            "  Pairwise Fast-Path:  {:.1}%",
            self.pairwise_fast_path_collective_rate() * 100.0
        );
        println!(
            "  Device Resident:    {:.1}%",
            self.device_resident_collective_rate() * 100.0
        );
        println!(
            "  Host Materialized:  {:.1}%",
            self.collective_host_materialization_rate() * 100.0
        );
        println!(
            "  Coll Wait Share:     {:.1}%",
            self.collective_wait_share_of_collective_runtime() * 100.0
        );
        println!(
            "  Host Staging Events: {}",
            self.collective_host_staging_events()
        );
        println!(
            "  Host Staging Bytes:  {}",
            self.collective_host_staging_bytes()
        );

        println!("\n{}", "Decode Batching:".bold());
        println!(
            "  Microbatches:        {}",
            self.decode_microbatches_executed.load(Ordering::Relaxed)
        );
        println!("  Avg Batch Size:      {:.2}", self.avg_decode_batch_size());
        println!(
            "  Fast-Path Plan Rate: {:.1}%",
            self.fast_path_decode_plan_rate() * 100.0
        );
        println!(
            "  Multi-Session Rate:  {:.1}%",
            self.multi_session_batch_rate() * 100.0
        );
        println!(
            "  Avg Deferred/Btch:   {:.2}",
            self.avg_deferred_sessions_per_microbatch()
        );
        println!(
            "  Peak Batch Size:     {}",
            self.decode_batch_size_peak.load(Ordering::Relaxed)
        );
        println!(
            "  Multi-session Batches:{}",
            self.decode_multi_session_microbatches
                .load(Ordering::Relaxed)
        );
        println!(
            "  KV Tokens Admitted:  {}",
            self.decode_batch_kv_tokens_total.load(Ordering::Relaxed)
        );
        println!(
            "  Deferred Sessions:   {}",
            self.decode_batch_deferred_sessions.load(Ordering::Relaxed)
        );
        println!(
            "  Capacity Deferrals:  {}",
            self.decode_batch_capacity_deferrals.load(Ordering::Relaxed)
        );
        println!(
            "  KV Budget Deferrals: {}",
            self.decode_batch_kv_budget_deferrals
                .load(Ordering::Relaxed)
        );
        println!(
            "  Prefill Plans:       {}",
            self.prefill_fast_path_plans.load(Ordering::Relaxed)
        );
        println!(
            "  Decode Plans:        {}",
            self.decode_fast_path_plans.load(Ordering::Relaxed)
        );
        println!(
            "  Arena Reuses:        {}",
            self.fast_path_arena_reuses.load(Ordering::Relaxed)
        );
        println!(
            "  Layout Validations:  {}",
            self.fast_path_layout_validated_plans
                .load(Ordering::Relaxed)
        );
        println!(
            "  Replay-Preferred:    {}",
            self.fast_path_replay_preferred_plans
                .load(Ordering::Relaxed)
        );
        println!(
            "  Peak Decode Bucket:  b{} / kv{}",
            self.fast_path_decode_bucket_batch_ceiling_peak
                .load(Ordering::Relaxed),
            self.fast_path_decode_bucket_token_ceiling_peak
                .load(Ordering::Relaxed)
        );
        println!(
            "  Peak Prefill Bucket: {}",
            self.fast_path_prefill_token_ceiling_peak
                .load(Ordering::Relaxed)
        );

        println!("\n{}", "KV Residency:".bold());
        println!(
            "  Snapshot Exports:    {}",
            self.kv_snapshot_exports.load(Ordering::Relaxed)
        );
        println!(
            "  Snapshot Export Bytes:{}",
            self.kv_snapshot_export_bytes.load(Ordering::Relaxed)
        );
        println!(
            "  Materializations:    {}",
            self.kv_snapshot_materializations.load(Ordering::Relaxed)
        );
        println!(
            "  Materialized Bytes:  {}",
            self.kv_snapshot_materialized_bytes.load(Ordering::Relaxed)
        );
        println!(
            "  Avg Export Bytes:    {:.2}",
            self.kv_snapshot_avg_export_bytes()
        );
        println!(
            "  Avg Materialized:    {:.2}",
            self.kv_snapshot_avg_materialized_bytes()
        );

        println!("\n{}", "Sampling:".bold());
        println!(
            "  Device Requests:     {}",
            self.device_sampling_requests.load(Ordering::Relaxed)
        );
        println!(
            "  Device Fallbacks:    {}",
            self.device_sampling_fallback_requests
                .load(Ordering::Relaxed)
        );
        println!(
            "  Host Requests:       {}",
            self.host_sampling_requests.load(Ordering::Relaxed)
        );
        println!(
            "  Sampling Time:       {}ms",
            self.total_sampling_time_ms.load(Ordering::Relaxed)
        );
        println!(
            "  Sampling Share:      {:.1}%",
            self.sampling_share_of_runtime() * 100.0
        );
        println!(
            "  Device Fallback Rate:{:.1}%",
            self.device_sampling_fallback_rate() * 100.0
        );
        println!(
            "  Token Latency:       {:.3}ms",
            self.avg_generated_token_latency_ms()
        );

        println!("\n{}", "KV View Reuse:".bold());
        println!(
            "  Active View Hit Rate:{:.1}%",
            self.device_kv_active_view_cache_hit_rate() * 100.0
        );
        println!(
            "  Head View Hit Rate:  {:.1}%",
            self.device_kv_head_view_cache_hit_rate() * 100.0
        );
        println!(
            "  Selected View Hit:   {:.1}%",
            self.device_kv_selected_head_view_cache_hit_rate() * 100.0
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
            "  Recovery Success:    {:.1}%",
            self.recovery_success_rate() * 100.0
        );
        println!(
            "  Recovery Attempts:   {}",
            self.recovery_attempts.load(Ordering::Relaxed)
        );
        println!(
            "  Recovery Reject Rate:{:.1}%",
            self.recovery_rejection_rate() * 100.0
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
            "decode_microbatches_executed",
            self.decode_microbatches_executed.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_sessions_batched",
            self.decode_sessions_batched.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_multi_session_microbatches",
            self.decode_multi_session_microbatches
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_batch_size_peak",
            self.decode_batch_size_peak.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_batch_kv_tokens_total",
            self.decode_batch_kv_tokens_total.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_batch_deferred_sessions",
            self.decode_batch_deferred_sessions.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_batch_capacity_deferrals",
            self.decode_batch_capacity_deferrals.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_batch_kv_budget_deferrals",
            self.decode_batch_kv_budget_deferrals
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_batch_guardrail_deferrals",
            self.decode_batch_guardrail_deferrals
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "prefill_fast_path_plans",
            self.prefill_fast_path_plans.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "decode_fast_path_plans",
            self.decode_fast_path_plans.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "fast_path_arena_reuses",
            self.fast_path_arena_reuses.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "fast_path_layout_validated_plans",
            self.fast_path_layout_validated_plans
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "fast_path_replay_preferred_plans",
            self.fast_path_replay_preferred_plans
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "fast_path_decode_bucket_batch_ceiling_peak",
            self.fast_path_decode_bucket_batch_ceiling_peak
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "fast_path_decode_bucket_token_ceiling_peak",
            self.fast_path_decode_bucket_token_ceiling_peak
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "fast_path_prefill_token_ceiling_peak",
            self.fast_path_prefill_token_ceiling_peak
                .load(Ordering::Relaxed),
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
            "tensor_bulk_transfer_bytes_sent",
            self.tensor_bulk_transfer_bytes_sent.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_bulk_transfer_bytes_received",
            self.tensor_bulk_transfer_bytes_received
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_checkpoint_bytes_sent",
            self.tensor_checkpoint_bytes_sent.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_checkpoint_bytes_received",
            self.tensor_checkpoint_bytes_received
                .load(Ordering::Relaxed),
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
            "tensor_connection_refresh_attempt_count",
            self.tensor_connection_refresh_attempt_count
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_connection_refresh_success_count",
            self.tensor_connection_refresh_success_count
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "tensor_connection_evict_count",
            self.tensor_connection_evict_count.load(Ordering::Relaxed),
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
        insert_u64(
            &mut map,
            "total_collective_operations",
            self.total_collective_operations.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_collective_worker_participants",
            self.total_collective_worker_participants
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_pairwise_fast_path_operations",
            self.total_pairwise_fast_path_operations
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_larger_ring_operations",
            self.total_larger_ring_operations.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_collective_bytes_sent",
            self.total_collective_bytes_sent.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_collective_bytes_received",
            self.total_collective_bytes_received.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_collective_host_materializations",
            self.total_collective_host_materializations
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_collective_host_materialization_bytes",
            self.total_collective_host_materialization_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_device_resident_collectives",
            self.total_device_resident_collectives
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "collective_host_stage_to_host_events",
            self.collective_host_stage_to_host_events
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "collective_host_stage_to_host_bytes",
            self.collective_host_stage_to_host_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "collective_host_restore_to_device_events",
            self.collective_host_restore_to_device_events
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "collective_host_restore_to_device_bytes",
            self.collective_host_restore_to_device_bytes
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "collective_host_staging_events",
            self.collective_host_staging_events(),
        );
        insert_u64(
            &mut map,
            "collective_host_staging_bytes",
            self.collective_host_staging_bytes(),
        );
        insert_u64(
            &mut map,
            "kv_snapshot_exports",
            self.kv_snapshot_exports.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "kv_snapshot_export_bytes",
            self.kv_snapshot_export_bytes.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "kv_snapshot_materializations",
            self.kv_snapshot_materializations.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "kv_snapshot_materialized_bytes",
            self.kv_snapshot_materialized_bytes.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_sampling_requests",
            self.device_sampling_requests.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_sampling_fallback_requests",
            self.device_sampling_fallback_requests
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "host_sampling_requests",
            self.host_sampling_requests.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "total_sampling_time_ms",
            self.total_sampling_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_sampling_time_ms",
            self.device_sampling_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "host_sampling_time_ms",
            self.host_sampling_time_ms.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_kv_active_view_cache_hits",
            self.device_kv_active_view_cache_hits
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_kv_active_view_cache_misses",
            self.device_kv_active_view_cache_misses
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_kv_head_view_cache_hits",
            self.device_kv_head_view_cache_hits.load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_kv_head_view_cache_misses",
            self.device_kv_head_view_cache_misses
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_kv_selected_head_view_cache_hits",
            self.device_kv_selected_head_view_cache_hits
                .load(Ordering::Relaxed),
        );
        insert_u64(
            &mut map,
            "device_kv_selected_head_view_cache_misses",
            self.device_kv_selected_head_view_cache_misses
                .load(Ordering::Relaxed),
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
            "avg_decode_batch_size".to_string(),
            serde_json::Value::from(self.avg_decode_batch_size()),
        );
        map.insert(
            "fast_path_decode_plan_rate".to_string(),
            serde_json::Value::from(self.fast_path_decode_plan_rate()),
        );
        map.insert(
            "multi_session_batch_rate".to_string(),
            serde_json::Value::from(self.multi_session_batch_rate()),
        );
        map.insert(
            "avg_deferred_sessions_per_microbatch".to_string(),
            serde_json::Value::from(self.avg_deferred_sessions_per_microbatch()),
        );
        map.insert(
            "allreduce_send_wait_share".to_string(),
            serde_json::Value::from(self.allreduce_send_wait_share()),
        );
        map.insert(
            "allreduce_receive_wait_share".to_string(),
            serde_json::Value::from(self.allreduce_receive_wait_share()),
        );
        map.insert(
            "collective_transport_share_of_runtime".to_string(),
            serde_json::Value::from(self.collective_transport_share_of_runtime()),
        );
        map.insert(
            "avg_collective_workers".to_string(),
            serde_json::Value::from(self.avg_collective_workers()),
        );
        map.insert(
            "larger_ring_collective_rate".to_string(),
            serde_json::Value::from(self.larger_ring_collective_rate()),
        );
        map.insert(
            "pairwise_fast_path_collective_rate".to_string(),
            serde_json::Value::from(self.pairwise_fast_path_collective_rate()),
        );
        map.insert(
            "avg_collective_send_bytes".to_string(),
            serde_json::Value::from(self.avg_collective_send_bytes()),
        );
        map.insert(
            "avg_collective_receive_bytes".to_string(),
            serde_json::Value::from(self.avg_collective_receive_bytes()),
        );
        map.insert(
            "device_resident_collective_rate".to_string(),
            serde_json::Value::from(self.device_resident_collective_rate()),
        );
        map.insert(
            "collective_host_materialization_rate".to_string(),
            serde_json::Value::from(self.collective_host_materialization_rate()),
        );
        map.insert(
            "collective_wait_share_of_collective_runtime".to_string(),
            serde_json::Value::from(self.collective_wait_share_of_collective_runtime()),
        );
        map.insert(
            "sampling_share_of_runtime".to_string(),
            serde_json::Value::from(self.sampling_share_of_runtime()),
        );
        map.insert(
            "device_sampling_fallback_rate".to_string(),
            serde_json::Value::from(self.device_sampling_fallback_rate()),
        );
        map.insert(
            "avg_generated_token_latency_ms".to_string(),
            serde_json::Value::from(self.avg_generated_token_latency_ms()),
        );
        map.insert(
            "kv_snapshot_avg_export_bytes".to_string(),
            serde_json::Value::from(self.kv_snapshot_avg_export_bytes()),
        );
        map.insert(
            "kv_snapshot_avg_materialized_bytes".to_string(),
            serde_json::Value::from(self.kv_snapshot_avg_materialized_bytes()),
        );
        map.insert(
            "device_kv_active_view_cache_hit_rate".to_string(),
            serde_json::Value::from(self.device_kv_active_view_cache_hit_rate()),
        );
        map.insert(
            "device_kv_head_view_cache_hit_rate".to_string(),
            serde_json::Value::from(self.device_kv_head_view_cache_hit_rate()),
        );
        map.insert(
            "device_kv_selected_head_view_cache_hit_rate".to_string(),
            serde_json::Value::from(self.device_kv_selected_head_view_cache_hit_rate()),
        );
        map.insert(
            "release_gate_metrics".to_string(),
            serde_json::json!({
                "avg_generated_token_latency_ms": self.avg_generated_token_latency_ms(),
                "collective_wait_share_of_collective_runtime": self.collective_wait_share_of_collective_runtime(),
                "collective_transport_share_of_runtime": self.collective_transport_share_of_runtime(),
                "device_resident_collective_rate": self.device_resident_collective_rate(),
                "collective_host_materialization_rate": self.collective_host_materialization_rate(),
                "collective_host_staging_events": self.collective_host_staging_events(),
                "collective_host_staging_bytes": self.collective_host_staging_bytes(),
                "avg_decode_batch_size": self.avg_decode_batch_size(),
                "multi_session_batch_rate": self.multi_session_batch_rate(),
                "kv_snapshot_avg_export_bytes": self.kv_snapshot_avg_export_bytes(),
                "kv_snapshot_avg_materialized_bytes": self.kv_snapshot_avg_materialized_bytes(),
                "sampling_share_of_runtime": self.sampling_share_of_runtime(),
                "device_sampling_fallback_rate": self.device_sampling_fallback_rate(),
                "device_kv_active_view_cache_hit_rate": self.device_kv_active_view_cache_hit_rate(),
                "device_kv_head_view_cache_hit_rate": self.device_kv_head_view_cache_hit_rate(),
                "device_kv_selected_head_view_cache_hit_rate": self.device_kv_selected_head_view_cache_hit_rate(),
            }),
        );
        map.insert(
            "recovery_success_rate".to_string(),
            serde_json::Value::from(self.recovery_success_rate()),
        );
        map.insert(
            "recovery_rejection_rate".to_string(),
            serde_json::Value::from(self.recovery_rejection_rate()),
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

static ACTIVE_RUNTIME_STATS: OnceLock<Mutex<Weak<InferenceStats>>> = OnceLock::new();

fn active_runtime_stats_slot() -> &'static Mutex<Weak<InferenceStats>> {
    ACTIVE_RUNTIME_STATS.get_or_init(|| Mutex::new(Weak::new()))
}

fn with_active_runtime_stats<T>(f: impl FnOnce(&InferenceStats) -> T) -> Option<T> {
    let stats = active_runtime_stats_slot()
        .lock()
        .expect("active runtime stats mutex poisoned")
        .upgrade()?;
    Some(f(&stats))
}

pub(crate) fn record_runtime_collective_host_stage(bytes: u64) {
    let _ = with_active_runtime_stats(|stats| stats.record_collective_host_stage_to_host(bytes));
}

pub(crate) fn record_runtime_collective_host_restore(bytes: u64) {
    let _ =
        with_active_runtime_stats(|stats| stats.record_collective_host_restore_to_device(bytes));
}

pub(crate) fn record_runtime_kv_snapshot_export(bytes: u64) {
    let _ = with_active_runtime_stats(|stats| stats.record_kv_snapshot_export(bytes));
}

pub(crate) fn record_runtime_kv_snapshot_materialization(bytes: u64) {
    let _ = with_active_runtime_stats(|stats| stats.record_kv_snapshot_materialization(bytes));
}

pub(crate) fn record_runtime_device_sampling(requests: u64, duration_ms: u64) {
    let _ = with_active_runtime_stats(|stats| stats.record_device_sampling(requests, duration_ms));
}

pub(crate) fn record_runtime_device_sampling_fallback(requests: u64) {
    let _ = with_active_runtime_stats(|stats| stats.record_device_sampling_fallback(requests));
}

pub(crate) fn record_runtime_host_sampling(requests: u64, duration_ms: u64) {
    let _ = with_active_runtime_stats(|stats| stats.record_host_sampling(requests, duration_ms));
}

pub(crate) fn record_runtime_device_kv_active_view_cache_hit() {
    let _ = with_active_runtime_stats(|stats| stats.record_device_kv_active_view_cache_hit());
}

pub(crate) fn record_runtime_device_kv_active_view_cache_miss() {
    let _ = with_active_runtime_stats(|stats| stats.record_device_kv_active_view_cache_miss());
}

pub(crate) fn record_runtime_device_kv_head_view_cache_hit() {
    let _ = with_active_runtime_stats(|stats| stats.record_device_kv_head_view_cache_hit());
}

pub(crate) fn record_runtime_device_kv_head_view_cache_miss() {
    let _ = with_active_runtime_stats(|stats| stats.record_device_kv_head_view_cache_miss());
}

pub(crate) fn record_runtime_device_kv_selected_head_view_cache_hit() {
    let _ =
        with_active_runtime_stats(|stats| stats.record_device_kv_selected_head_view_cache_hit());
}

pub(crate) fn record_runtime_device_kv_selected_head_view_cache_miss() {
    let _ =
        with_active_runtime_stats(|stats| stats.record_device_kv_selected_head_view_cache_miss());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ExecutionProviderKind;

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
    fn test_stats_decode_microbatch_metrics() {
        let stats = InferenceStats::new();

        stats.record_decode_microbatch(1, 128, 0, 0, 0, 0);
        stats.record_decode_microbatch(3, 512, 2, 1, 1, 1);

        assert_eq!(
            stats.decode_microbatches_executed.load(Ordering::Relaxed),
            2
        );
        assert_eq!(stats.decode_sessions_batched.load(Ordering::Relaxed), 4);
        assert_eq!(
            stats
                .decode_multi_session_microbatches
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(stats.decode_batch_size_peak.load(Ordering::Relaxed), 3);
        assert_eq!(
            stats.decode_batch_kv_tokens_total.load(Ordering::Relaxed),
            640
        );
        assert_eq!(
            stats.decode_batch_deferred_sessions.load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            stats
                .decode_batch_capacity_deferrals
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .decode_batch_kv_budget_deferrals
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .decode_batch_guardrail_deferrals
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(stats.avg_decode_batch_size(), 2.0);
        assert_eq!(stats.fast_path_decode_plan_rate(), 0.0);
        assert_eq!(stats.multi_session_batch_rate(), 0.5);
        assert_eq!(stats.avg_deferred_sessions_per_microbatch(), 1.0);
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
        assert_eq!(stats.recovery_success_rate(), 1.0);
        assert_eq!(stats.recovery_rejection_rate(), 2.0);
    }

    #[test]
    fn test_stats_to_json() {
        let stats = InferenceStats::new();
        stats.record_success(10, 50, 1000);
        stats.record_allreduce_breakdown(RingAllReduceMetrics {
            reduce_scatter_step_time_ms: 10,
            all_gather_step_time_ms: 12,
            send_wait_time_ms: 4,
            receive_wait_time_ms: 6,
            collective_operations: 2,
            collective_worker_participants: 5,
            pairwise_fast_path_operations: 1,
            larger_ring_operations: 1,
            bytes_sent: 128,
            bytes_received: 160,
            host_materialization_count: 1,
            host_materialization_bytes: 96,
            device_resident_collective_count: 1,
        });
        stats.record_collective_host_stage_to_host(64);
        stats.record_collective_host_restore_to_device(64);
        stats.record_kv_snapshot_export(128);
        stats.record_kv_snapshot_materialization(96);
        stats.record_device_sampling(2, 7);
        stats.record_device_sampling_fallback(1);
        stats.record_host_sampling(1, 5);
        stats.record_device_kv_active_view_cache_miss();
        stats.record_device_kv_active_view_cache_hit();
        stats.record_device_kv_head_view_cache_miss();
        stats.record_device_kv_selected_head_view_cache_miss();

        let json = stats.to_json();

        assert_eq!(json["jobs_completed"], 1);
        assert_eq!(json["total_tokens_generated"], 50);
        assert_eq!(json["avg_decode_batch_size"], 0.0);
        assert_eq!(json["fast_path_decode_plan_rate"], 0.0);
        assert_eq!(json["collective_host_staging_events"], 2);
        assert_eq!(json["collective_host_staging_bytes"], 128);
        assert_eq!(json["kv_snapshot_exports"], 1);
        assert_eq!(json["kv_snapshot_materializations"], 1);
        assert_eq!(json["device_sampling_requests"], 2);
        assert_eq!(json["device_sampling_fallback_requests"], 1);
        assert_eq!(json["host_sampling_requests"], 1);
        assert_eq!(json["total_collective_host_materializations"], 1);
        assert_eq!(json["total_device_resident_collectives"], 1);
        assert_eq!(json["device_kv_active_view_cache_hits"], 1);
        assert_eq!(json["device_kv_active_view_cache_misses"], 1);
        assert_eq!(json["device_resident_collective_rate"], 0.5);
        assert_eq!(json["collective_host_materialization_rate"], 0.5);
        assert_eq!(
            json["release_gate_metrics"]["device_kv_active_view_cache_hit_rate"],
            0.5
        );
    }

    #[test]
    fn test_stats_record_fast_path_plans() {
        let stats = InferenceStats::new();
        let plan = FastPathExecutionPlan {
            bucket: super::super::fast_path::FastPathBucketKey {
                phase: super::super::engine::ExecutionPhase::Decode,
                provider: ExecutionProviderKind::Cuda,
                optimization_profile: super::super::engine::BackendOptimizationProfile::CudaFused,
                batch_size_ceiling: 4,
                token_ceiling: 8_192,
            },
            bucket_label: "decode-b4-kv8192-cuda".to_string(),
            actual_batch_size: 3,
            actual_token_count: 4_096,
            max_sequence_len: 2_048,
            metadata: super::super::fast_path::BucketMetadataLayout {
                version: 1,
                total_bytes: 512,
                layout_hash: 99,
                fields: Vec::new(),
            },
            workspace: super::super::fast_path::WorkspaceRequirements {
                bytes: 4_096,
                alignment_bytes: 256,
            },
            capture_strategy: GraphCaptureStrategy::ReplayPreferred,
            prefill_strategy: None,
        };

        stats.record_prefill_fast_path_plan(&plan, false);
        stats.record_decode_fast_path_plan(&plan, true);

        assert_eq!(stats.prefill_fast_path_plans.load(Ordering::Relaxed), 1);
        assert_eq!(stats.decode_fast_path_plans.load(Ordering::Relaxed), 1);
        assert_eq!(stats.fast_path_decode_plan_rate(), 0.0);
        assert_eq!(stats.fast_path_arena_reuses.load(Ordering::Relaxed), 1);
        assert_eq!(
            stats
                .fast_path_replay_preferred_plans
                .load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            stats
                .fast_path_decode_bucket_batch_ceiling_peak
                .load(Ordering::Relaxed),
            4
        );
    }

    #[test]
    fn test_stats_transport_attribution_metrics() {
        let stats = InferenceStats::new();
        stats.record_success(10, 20, 100);
        stats.record_allreduce(40);
        stats.record_allreduce_breakdown(RingAllReduceMetrics {
            reduce_scatter_step_time_ms: 10,
            all_gather_step_time_ms: 12,
            send_wait_time_ms: 4,
            receive_wait_time_ms: 6,
            collective_operations: 2,
            collective_worker_participants: 5,
            pairwise_fast_path_operations: 1,
            larger_ring_operations: 1,
            bytes_sent: 128,
            bytes_received: 160,
            host_materialization_count: 0,
            host_materialization_bytes: 0,
            device_resident_collective_count: 0,
        });

        assert!((stats.allreduce_send_wait_share() - 0.1).abs() < f64::EPSILON);
        assert!((stats.allreduce_receive_wait_share() - 0.15).abs() < f64::EPSILON);
        assert!((stats.collective_transport_share_of_runtime() - 0.32).abs() < f64::EPSILON);
        assert!((stats.avg_collective_workers() - 2.5).abs() < f64::EPSILON);
        assert!((stats.larger_ring_collective_rate() - 0.5).abs() < f64::EPSILON);
        assert!((stats.pairwise_fast_path_collective_rate() - 0.5).abs() < f64::EPSILON);
        assert!((stats.avg_collective_send_bytes() - 64.0).abs() < f64::EPSILON);
        assert!((stats.avg_collective_receive_bytes() - 80.0).abs() < f64::EPSILON);
        assert!(
            (stats.collective_wait_share_of_collective_runtime() - (10.0 / 32.0)).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_stats_runtime_overhead_metrics() {
        let stats = InferenceStats::new();
        stats.record_success(16, 32, 200);
        stats.record_allreduce_breakdown(RingAllReduceMetrics {
            reduce_scatter_step_time_ms: 30,
            all_gather_step_time_ms: 20,
            send_wait_time_ms: 10,
            receive_wait_time_ms: 20,
            collective_operations: 4,
            collective_worker_participants: 10,
            pairwise_fast_path_operations: 1,
            larger_ring_operations: 3,
            bytes_sent: 1024,
            bytes_received: 2048,
            host_materialization_count: 1,
            host_materialization_bytes: 64,
            device_resident_collective_count: 3,
        });
        stats.record_collective_host_stage_to_host(64);
        stats.record_collective_host_restore_to_device(32);
        stats.record_kv_snapshot_export(256);
        stats.record_kv_snapshot_materialization(128);
        stats.record_device_sampling(3, 9);
        stats.record_device_sampling_fallback(1);
        stats.record_host_sampling(1, 5);
        stats.record_device_kv_active_view_cache_miss();
        stats.record_device_kv_active_view_cache_hit();
        stats.record_device_kv_active_view_cache_hit();
        stats.record_device_kv_head_view_cache_miss();
        stats.record_device_kv_head_view_cache_hit();
        stats.record_device_kv_selected_head_view_cache_miss();
        stats.record_device_kv_selected_head_view_cache_hit();

        assert_eq!(stats.collective_host_staging_events(), 2);
        assert_eq!(stats.collective_host_staging_bytes(), 96);
        assert_eq!(
            stats
                .total_collective_host_materializations
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .total_device_resident_collectives
                .load(Ordering::Relaxed),
            3
        );
        assert_eq!(stats.kv_snapshot_exports.load(Ordering::Relaxed), 1);
        assert_eq!(stats.kv_snapshot_export_bytes.load(Ordering::Relaxed), 256);
        assert_eq!(
            stats.kv_snapshot_materializations.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats.kv_snapshot_materialized_bytes.load(Ordering::Relaxed),
            128
        );
        assert_eq!(stats.total_sampling_time_ms.load(Ordering::Relaxed), 14);
        assert!((stats.sampling_share_of_runtime() - 0.07).abs() < f64::EPSILON);
        assert!((stats.device_sampling_fallback_rate() - 0.25).abs() < f64::EPSILON);
        assert!((stats.avg_generated_token_latency_ms() - 6.25).abs() < f64::EPSILON);
        assert!((stats.device_resident_collective_rate() - 0.75).abs() < f64::EPSILON);
        assert!((stats.collective_host_materialization_rate() - 0.25).abs() < f64::EPSILON);
        assert!((stats.kv_snapshot_avg_export_bytes() - 256.0).abs() < f64::EPSILON);
        assert!((stats.kv_snapshot_avg_materialized_bytes() - 128.0).abs() < f64::EPSILON);
        assert!((stats.device_kv_active_view_cache_hit_rate() - (2.0 / 3.0)).abs() < f64::EPSILON);
        assert!((stats.device_kv_head_view_cache_hit_rate() - 0.5).abs() < f64::EPSILON);
        assert!((stats.device_kv_selected_head_view_cache_hit_rate() - 0.5).abs() < f64::EPSILON);
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
            bulk_transfer_bytes_sent: 12,
            bulk_transfer_bytes_received: 14,
            checkpoint_bytes_sent: 18,
            checkpoint_bytes_received: 20,
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
            connection_refresh_attempt_count: 29,
            connection_refresh_success_count: 31,
            connection_evict_count: 37,
            latency_critical_send_count: 1,
            interactive_send_count: 2,
            bulk_send_count: 3,
        });

        assert_eq!(stats.tensor_bytes_sent.load(Ordering::Relaxed), 128);
        assert_eq!(stats.tensor_bytes_received.load(Ordering::Relaxed), 256);
        assert_eq!(
            stats
                .tensor_bulk_transfer_bytes_sent
                .load(Ordering::Relaxed),
            12
        );
        assert_eq!(
            stats
                .tensor_checkpoint_bytes_received
                .load(Ordering::Relaxed),
            20
        );
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
        assert_eq!(
            stats
                .tensor_connection_refresh_attempt_count
                .load(Ordering::Relaxed),
            29
        );
        assert_eq!(
            stats
                .tensor_connection_refresh_success_count
                .load(Ordering::Relaxed),
            31
        );
        assert_eq!(
            stats.tensor_connection_evict_count.load(Ordering::Relaxed),
            37
        );
    }
}
