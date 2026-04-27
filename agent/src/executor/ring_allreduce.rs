//! Ring All-Reduce Implementation for Distributed AI Training
//!
//! This module implements the Ring All-Reduce algorithm as described in:
//! "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
//! Reference: <https://arxiv.org/abs/1802.05799>
//!
//! The algorithm consists of two phases:
//! 1. **Reduce-Scatter**: Each worker ends up with a partial sum of one chunk
//! 2. **All-Gather**: Each worker collects all the partial sums
//!
//! This is the same algorithm used by NCCL (NVIDIA) and Horovod (Uber) for
//! bandwidth-optimal gradient aggregation in distributed training.

use crate::errors::{AgentError, Result};
use crate::inference::InferenceRuntimeMode;
use crate::network::{
    CollectiveLane, ServingFrame, ServingReceiveSpec, ServingSessionTransport, TensorPlane,
};
use crate::provider::ExecutionProviderKind;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;
use tracing::{debug, info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingAllReduceMetrics {
    pub reduce_scatter_step_time_ms: u64,
    pub all_gather_step_time_ms: u64,
    pub send_wait_time_ms: u64,
    pub receive_wait_time_ms: u64,
}

impl RingAllReduceMetrics {
    pub fn accumulate(&mut self, other: RingAllReduceMetrics) {
        self.reduce_scatter_step_time_ms += other.reduce_scatter_step_time_ms;
        self.all_gather_step_time_ms += other.all_gather_step_time_ms;
        self.send_wait_time_ms += other.send_wait_time_ms;
        self.receive_wait_time_ms += other.receive_wait_time_ms;
    }
}

/// Tensor data structure for ring all-reduce operations
///
/// Represents a multi-dimensional array of f32 values commonly used
/// for gradients and model parameters in distributed training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    /// Flattened tensor data
    pub data: Vec<f32>,
    /// Shape of the tensor (e.g., [100] for 1D, [10, 10] for 2D)
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with the given data and shape
    ///
    /// # Panics
    /// Panics if the data length doesn't match the product of the shape dimensions.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape product {}",
            data.len(),
            expected_len
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    /// Create a tensor filled with a constant value
    pub fn filled(shape: Vec<usize>, value: f32) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![value; len],
            shape,
        }
    }

    /// Split tensor into n chunks for ring all-reduce
    ///
    /// Each chunk will have approximately equal size. The last chunk
    /// may have fewer elements if the data doesn't divide evenly.
    pub fn chunk(&self, n: usize) -> Vec<Tensor> {
        assert!(n > 0, "Number of chunks must be positive");

        let chunk_size = self.data.len().div_ceil(n);

        self.data
            .chunks(chunk_size)
            .map(|chunk| Tensor::new(chunk.to_vec(), vec![chunk.len()]))
            .collect()
    }

    /// Element-wise addition of two tensors
    ///
    /// # Errors
    /// Returns an error if the tensors have different sizes.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.data.len() != other.data.len() {
            return Err(AgentError::Execution(format!(
                "Tensor size mismatch: {} vs {}",
                self.data.len(),
                other.data.len()
            )));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor::new(data, self.shape.clone()))
    }

    /// Concatenate multiple tensors into a single tensor
    pub fn concat(tensors: Vec<Tensor>) -> Tensor {
        let mut data = Vec::new();
        for t in tensors {
            data.extend(t.data);
        }
        let len = data.len();
        Tensor::new(data, vec![len])
    }

    /// Get the total number of elements in the tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn chunk_ranges(&self, n: usize) -> Vec<std::ops::Range<usize>> {
        assert!(n > 0, "Number of chunks must be positive");
        let chunk_size = self.data.len().div_ceil(n);
        let mut ranges = Vec::with_capacity(n);
        let mut start = 0usize;
        for _ in 0..n {
            let end = if chunk_size == 0 {
                start
            } else {
                (start + chunk_size).min(self.data.len())
            };
            ranges.push(start..end);
            start = end;
        }
        ranges
    }
}

/// Worker's position in the ring topology
///
/// Each worker knows its position, the total number of workers,
/// and its left/right neighbors in the ring.
///
/// NOTE: WorkerRing borrows the tensor data plane instead of owning it, allowing
/// the coordinator to maintain ownership for the entire inference job.
pub struct WorkerRing<'a> {
    /// This worker's position in the ring (0 to total_workers-1)
    pub my_position: u32,
    /// Total number of workers in the ring
    pub total_workers: u32,
    /// Peer ID of the left neighbor (sends to us in scatter phase)
    pub left_neighbor: PeerId,
    /// Peer ID of the right neighbor (we send to them in scatter phase)
    pub right_neighbor: PeerId,
    /// Left neighbor data-plane socket address.
    pub left_tensor_addr: SocketAddr,
    /// Right neighbor data-plane socket address.
    pub right_tensor_addr: SocketAddr,
    /// Dedicated tensor plane for network communication (borrowed).
    pub tensor_plane: &'a mut TensorPlane,
    /// Long-lived collective-native serving transport.
    serving_transport: Option<ServingSessionTransport>,
    /// Runtime mode for transport fallback and scheduling intent.
    runtime_mode: InferenceRuntimeMode,
    /// Provider kind used by the local execution backend.
    provider: ExecutionProviderKind,
    /// Provider/runtime specialized collective transport plan.
    collective_profile: CollectiveOptimizationProfile,
    /// Timings captured during the most recent all-reduce call.
    last_run_metrics: RingAllReduceMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveOptimizationProfile {
    GenericStable,
    CpuLowFanout,
    MetalBalanced,
    CudaHighThroughput,
}

impl CollectiveOptimizationProfile {
    fn for_provider_and_mode(
        provider: ExecutionProviderKind,
        runtime_mode: InferenceRuntimeMode,
    ) -> Self {
        match (provider, runtime_mode) {
            (ExecutionProviderKind::Cuda, InferenceRuntimeMode::ThroughputFirst) => {
                Self::CudaHighThroughput
            }
            (ExecutionProviderKind::Metal, InferenceRuntimeMode::LatencyFirst)
            | (ExecutionProviderKind::Metal, InferenceRuntimeMode::ThroughputFirst) => {
                Self::MetalBalanced
            }
            (ExecutionProviderKind::Cpu, _) | (_, InferenceRuntimeMode::FitFirst) => {
                Self::CpuLowFanout
            }
            _ => Self::GenericStable,
        }
    }
}

impl<'a> WorkerRing<'a> {
    /// Create a new worker ring
    ///
    /// # Arguments
    /// * `my_position` - This worker's position in the ring (0-indexed)
    /// * `total_workers` - Total number of workers in the ring
    /// * `left_neighbor` - Peer ID of the left neighbor
    /// * `right_neighbor` - Peer ID of the right neighbor
    /// * `swarm` - Mutable borrow of mesh swarm for network communication
    pub fn new(
        my_position: u32,
        total_workers: u32,
        left_neighbor: PeerId,
        right_neighbor: PeerId,
        left_tensor_addr: SocketAddr,
        right_tensor_addr: SocketAddr,
        runtime_mode: InferenceRuntimeMode,
        provider: ExecutionProviderKind,
        tensor_plane: &'a mut TensorPlane,
    ) -> Self {
        let collective_profile =
            CollectiveOptimizationProfile::for_provider_and_mode(provider, runtime_mode);
        Self {
            my_position,
            total_workers,
            left_neighbor,
            right_neighbor,
            left_tensor_addr,
            right_tensor_addr,
            tensor_plane,
            serving_transport: None,
            runtime_mode,
            provider,
            collective_profile,
            last_run_metrics: RingAllReduceMetrics::default(),
        }
    }

    pub async fn prepare_serving_group_channels(&mut self) -> Result<()> {
        self.serving_transport = Some(
            self.tensor_plane
                .serving_transport_for_neighbors(
                    self.left_tensor_addr,
                    self.right_tensor_addr,
                    self.runtime_mode,
                    self.provider,
                )
                .await?,
        );
        Ok(())
    }

    /// Perform ring all-reduce on a tensor
    ///
    /// This implements the bandwidth-optimal ring all-reduce algorithm:
    /// 1. Split tensor into n chunks (n = number of workers)
    /// 2. Reduce-scatter: n-1 steps, each worker accumulates one chunk
    /// 3. All-gather: n-1 steps, each worker gets all accumulated chunks
    ///
    /// # Arguments
    /// * `partial_result` - This worker's local tensor (e.g., gradient)
    /// * `job_id` - Unique identifier for this all-reduce operation
    /// * `layer_idx` - Layer index for multi-layer aggregation
    ///
    /// # Returns
    /// The fully reduced tensor (sum of all workers' partial results)
    pub async fn ring_all_reduce(
        &mut self,
        partial_result: Tensor,
        job_id: Uuid,
        layer_idx: u32,
    ) -> Result<Tensor> {
        let n = self.total_workers as usize;
        let my_pos = self.my_position as usize;
        let original_shape = partial_result.shape.clone();
        let mut run_metrics = RingAllReduceMetrics::default();
        let chunk_ranges = partial_result.chunk_ranges(n);
        let mut work_buffer = partial_result.data;

        info!(
            "Starting ring all-reduce: n={}, my_pos={}, chunks={}",
            n,
            my_pos,
            chunk_ranges.len()
        );

        // PHASE 1: Reduce-Scatter (n-1 steps)
        // After this phase, worker i has the complete sum of chunk (i+1) % n
        for step in 0..(n - 1) {
            // In step k, worker i:
            // - Sends chunk[(i - k) % n] to right neighbor
            // - Receives from left neighbor into chunk[(i - k - 1) % n]
            let send_idx = (my_pos + n - step) % n;
            let recv_idx = (my_pos + n - step - 1) % n;

            debug!(
                "Reduce-scatter step {}/{}: send_idx={}, recv_idx={}",
                step + 1,
                n - 1,
                send_idx,
                recv_idx
            );

            let send_range = &chunk_ranges[send_idx];
            let send_shape = [send_range.len()];

            let step_started = std::time::Instant::now();
            let (recv_msg, send_wait_ms, receive_wait_ms) = self
                .send_chunk_to_right_recv_from_left(
                    CollectiveLane::ReduceScatter,
                    job_id,
                    layer_idx,
                    step as u32,
                    recv_idx as u32,
                    &work_buffer[send_range.start..send_range.end],
                    &send_shape,
                )
                .await?;
            run_metrics.reduce_scatter_step_time_ms += step_started.elapsed().as_millis() as u64;
            run_metrics.send_wait_time_ms += send_wait_ms;
            run_metrics.receive_wait_time_ms += receive_wait_ms;

            let recv_range = &chunk_ranges[recv_idx];
            if recv_msg.chunk_data.len() != recv_range.len() {
                return Err(AgentError::Execution(format!(
                    "Received reduce-scatter chunk len {} but expected {}",
                    recv_msg.chunk_data.len(),
                    recv_range.len()
                )));
            }
            for (dst, src) in work_buffer[recv_range.start..recv_range.end]
                .iter_mut()
                .zip(recv_msg.chunk_data.iter())
            {
                *dst += *src;
            }
        }

        info!("Reduce-scatter complete");

        // PHASE 2: All-Gather (n-1 steps)
        // After this phase, all workers have all the complete sums
        for step in 0..(n - 1) {
            // In step k, worker i:
            // - Sends the chunk that was accumulated in the previous all-gather step
            // - Or in step 0, sends the chunk accumulated in reduce-scatter
            let send_idx = (my_pos + n - step + 1) % n;
            let recv_idx = (my_pos + n - step) % n;

            debug!(
                "All-gather step {}/{}: send_idx={}, recv_idx={}",
                step + 1,
                n - 1,
                send_idx,
                recv_idx
            );

            let send_range = &chunk_ranges[send_idx];
            let send_shape = [send_range.len()];

            let step_started = std::time::Instant::now();
            let (recv_msg, send_wait_ms, receive_wait_ms) = self
                .send_chunk_to_right_recv_from_left(
                    CollectiveLane::AllGather,
                    job_id,
                    layer_idx,
                    step as u32,
                    recv_idx as u32,
                    &work_buffer[send_range.start..send_range.end],
                    &send_shape,
                )
                .await?;
            run_metrics.all_gather_step_time_ms += step_started.elapsed().as_millis() as u64;
            run_metrics.send_wait_time_ms += send_wait_ms;
            run_metrics.receive_wait_time_ms += receive_wait_ms;

            let recv_range = &chunk_ranges[recv_idx];
            if recv_msg.chunk_data.len() != recv_range.len() {
                return Err(AgentError::Execution(format!(
                    "Received all-gather chunk len {} but expected {}",
                    recv_msg.chunk_data.len(),
                    recv_range.len()
                )));
            }
            work_buffer[recv_range.start..recv_range.end].copy_from_slice(&recv_msg.chunk_data);
        }

        info!("All-gather complete");

        self.last_run_metrics = run_metrics;
        Ok(Tensor::new(work_buffer, original_shape))
    }

    /// Ring all-reduce with timeout
    ///
    /// Wraps `ring_all_reduce` with a timeout to handle stalled workers.
    pub async fn ring_all_reduce_with_timeout(
        &mut self,
        partial_result: Tensor,
        job_id: Uuid,
        layer_idx: u32,
        timeout: Duration,
    ) -> Result<Tensor> {
        tokio::time::timeout(
            timeout,
            self.ring_all_reduce(partial_result, job_id, layer_idx),
        )
        .await
        .map_err(|_| {
            AgentError::Execution(format!("Ring all-reduce timed out after {:?}", timeout))
        })?
    }

    /// Barrier synchronization - wait for all workers to reach this point
    ///
    /// Uses a ring-based barrier where each worker sends a message to its
    /// right neighbor and waits for a message from its left neighbor.
    pub async fn barrier_sync(&mut self, job_id: Uuid, layer_idx: u32) -> Result<()> {
        let (received, _, _) = self
            .send_chunk_to_right_recv_from_left(
                CollectiveLane::Control,
                job_id,
                layer_idx,
                0,
                0,
                &[self.my_position as f32],
                &[1],
            )
            .await?;

        // Verify received from left neighbor
        let expected_pos = (self.my_position + self.total_workers - 1) % self.total_workers;
        if received.chunk_data[0] as u32 != expected_pos {
            warn!(
                "Barrier sync received from unexpected peer: got {}, expected {}",
                received.chunk_data[0] as u32, expected_pos
            );
        }

        debug!("Barrier sync complete for layer {}", layer_idx);
        Ok(())
    }

    fn serving_transport(&self) -> Result<&ServingSessionTransport> {
        self.serving_transport.as_ref().ok_or_else(|| {
            AgentError::Execution(
                "Serving session transport was used before preparation".to_string(),
            )
        })
    }

    async fn send_chunk_to_right_recv_from_left(
        &mut self,
        lane: CollectiveLane,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        chunk_data: &[f32],
        chunk_shape: &[usize],
    ) -> Result<(ServingFrame, u64, u64)> {
        let expected_sender_position =
            (self.my_position + self.total_workers - 1) % self.total_workers;
        let transport = self.serving_transport()?.clone();
        let send_started = std::time::Instant::now();
        let recv_started = std::time::Instant::now();
        let (send_result, inbound) = tokio::join!(
            async {
                match lane {
                    CollectiveLane::ReduceScatter => {
                        transport
                            .send_reduce_scatter_chunk(
                                collective_id,
                                layer_idx,
                                step,
                                slot,
                                0,
                                self.my_position,
                                chunk_data,
                                chunk_shape,
                            )
                            .await
                    }
                    CollectiveLane::AllGather => {
                        transport
                            .send_all_gather_chunk(
                                collective_id,
                                layer_idx,
                                step,
                                slot,
                                0,
                                self.my_position,
                                chunk_data,
                                chunk_shape,
                            )
                            .await
                    }
                    CollectiveLane::Control => {
                        transport
                            .send_control(
                                collective_id,
                                layer_idx,
                                step,
                                slot,
                                0,
                                self.my_position,
                                chunk_data,
                                chunk_shape,
                            )
                            .await
                    }
                    other => Err(AgentError::Execution(format!(
                        "Unsupported ring collective lane {:?}",
                        other
                    ))),
                }
            },
            transport.recv_frame(ServingReceiveSpec {
                collective_id,
                lane,
                layer_idx,
                step,
                slot,
                stream_id: 0,
                expected_sender_position,
            })
        );
        send_result?;
        let inbound = inbound?;

        Ok((
            inbound,
            send_started.elapsed().as_millis() as u64,
            recv_started.elapsed().as_millis() as u64,
        ))
    }

    pub fn last_run_metrics(&self) -> RingAllReduceMetrics {
        self.last_run_metrics
    }

    pub fn collective_optimization_profile(&self) -> CollectiveOptimizationProfile {
        self.collective_profile
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::provider::ExecutionProviderKind;

    // ============== Tensor Tests ==============

    #[test]
    fn test_tensor_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];
        let tensor = Tensor::new(data.clone(), shape.clone());
        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(vec![3, 4]);
        assert_eq!(tensor.data.len(), 12);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_filled() {
        let tensor = Tensor::filled(vec![2, 2], 5.0);
        assert_eq!(tensor.data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_tensor_new_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![4]; // Doesn't match data length
        Tensor::new(data, shape);
    }

    #[test]
    fn test_tensor_chunk_even() {
        let tensor = Tensor::new((0..10).map(|i| i as f32).collect(), vec![10]);
        let chunks = tensor.chunk(5);

        assert_eq!(chunks.len(), 5);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 2);
        }

        // Verify data is preserved
        let concat = Tensor::concat(chunks);
        assert_eq!(concat.data, tensor.data);
    }

    #[test]
    fn test_tensor_chunk_uneven() {
        let tensor = Tensor::new((0..10).map(|i| i as f32).collect(), vec![10]);
        let chunks = tensor.chunk(3);

        assert_eq!(chunks.len(), 3);
        // Ceiling division: 10/3 = 4, so first chunks have 4 elements
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[1].len(), 4);
        assert_eq!(chunks[2].len(), 2);
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_add_size_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0], vec![2]);
        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_concat() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2]);
        let t3 = Tensor::new(vec![5.0], vec![1]);

        let result = Tensor::concat(vec![t1, t2, t3]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_reconstruct_full_tensor_preserves_original_shape() {
        let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let chunks = original.chunk(2);
        let reconstructed = Tensor::new(Tensor::concat(chunks).data, original.shape.clone());
        assert_eq!(reconstructed.shape, vec![2, 2]);
        assert_eq!(reconstructed.data, original.data);
    }

    #[test]
    fn test_tensor_len_and_is_empty() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(tensor.len(), 3);
        assert!(!tensor.is_empty());

        let empty = Tensor::new(vec![], vec![0]);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_collective_optimization_profile_tracks_provider_and_runtime_mode() {
        assert_eq!(
            CollectiveOptimizationProfile::for_provider_and_mode(
                ExecutionProviderKind::Cuda,
                InferenceRuntimeMode::ThroughputFirst,
            ),
            CollectiveOptimizationProfile::CudaHighThroughput
        );
        assert_eq!(
            CollectiveOptimizationProfile::for_provider_and_mode(
                ExecutionProviderKind::Metal,
                InferenceRuntimeMode::LatencyFirst,
            ),
            CollectiveOptimizationProfile::MetalBalanced
        );
        assert_eq!(
            CollectiveOptimizationProfile::for_provider_and_mode(
                ExecutionProviderKind::Cpu,
                InferenceRuntimeMode::ResilientEdge,
            ),
            CollectiveOptimizationProfile::CpuLowFanout
        );
    }

    // ============== Ring All-Reduce Logic Tests ==============

    /// Simulates the ring all-reduce algorithm without network
    /// to verify algorithmic correctness
    fn simulate_ring_allreduce(partial_results: Vec<Tensor>) -> Vec<Tensor> {
        let n = partial_results.len();
        assert!(n > 1, "Need at least 2 workers");

        // Each worker has its own set of chunks
        let mut all_chunks: Vec<Vec<Tensor>> = partial_results.iter().map(|t| t.chunk(n)).collect();

        // Phase 1: Reduce-Scatter
        for step in 0..(n - 1) {
            // Create a copy of chunks to send (to avoid borrow issues)
            let chunks_to_send: Vec<Tensor> = (0..n)
                .map(|worker| {
                    let send_idx = (worker + n - step) % n;
                    all_chunks[worker][send_idx].clone()
                })
                .collect();

            // Each worker receives from its left neighbor
            for worker in 0..n {
                let recv_idx = (worker + n - step - 1) % n;
                let left = (worker + n - 1) % n;

                // Receive chunk from left neighbor
                let received = chunks_to_send[left].clone();

                // Accumulate
                all_chunks[worker][recv_idx] = all_chunks[worker][recv_idx].add(&received).unwrap();
            }
        }

        // Phase 2: All-Gather
        for step in 0..(n - 1) {
            // Create a copy of chunks to send
            let chunks_to_send: Vec<Tensor> = (0..n)
                .map(|worker| {
                    let send_idx = (worker + n - step + 1) % n;
                    all_chunks[worker][send_idx].clone()
                })
                .collect();

            // Each worker receives from its left neighbor
            for worker in 0..n {
                let recv_idx = (worker + n - step) % n;
                let left = (worker + n - 1) % n;

                // Copy received chunk (no accumulation in all-gather)
                all_chunks[worker][recv_idx] = chunks_to_send[left].clone();
            }
        }

        // Concatenate each worker's chunks
        all_chunks.into_iter().map(Tensor::concat).collect()
    }

    #[test]
    fn test_ring_allreduce_3_workers() {
        // 3 workers, each with value [i+1, i+1, i+1]
        let partial_results: Vec<Tensor> = (0..3)
            .map(|i| Tensor::filled(vec![6], (i + 1) as f32))
            .collect();

        let results = simulate_ring_allreduce(partial_results);

        // Sum should be 1+2+3 = 6 for each element
        let expected_sum = 6.0;
        for (i, result) in results.iter().enumerate() {
            for &value in &result.data {
                assert!(
                    (value - expected_sum).abs() < 0.001,
                    "Worker {} result mismatch: got {}, expected {}",
                    i,
                    value,
                    expected_sum
                );
            }
        }

        // All workers should have identical results
        for i in 1..results.len() {
            assert_eq!(
                results[0].data, results[i].data,
                "Worker 0 and {} have different results",
                i
            );
        }
    }

    #[test]
    fn test_ring_allreduce_5_workers() {
        let partial_results: Vec<Tensor> = (0..5)
            .map(|i| Tensor::filled(vec![10], (i + 1) as f32))
            .collect();

        let results = simulate_ring_allreduce(partial_results);

        // Sum should be 1+2+3+4+5 = 15
        let expected_sum = 15.0;
        for (i, result) in results.iter().enumerate() {
            for &value in &result.data {
                assert!(
                    (value - expected_sum).abs() < 0.001,
                    "Worker {} result mismatch: got {}, expected {}",
                    i,
                    value,
                    expected_sum
                );
            }
        }

        // All workers should have identical results
        for i in 1..results.len() {
            assert_eq!(results[0].data, results[i].data);
        }
    }

    #[test]
    fn test_ring_allreduce_10_workers() {
        let partial_results: Vec<Tensor> = (0..10)
            .map(|i| Tensor::filled(vec![100], (i + 1) as f32))
            .collect();

        let results = simulate_ring_allreduce(partial_results);

        // Sum should be 1+2+...+10 = 55
        let expected_sum: f32 = (1..=10).sum::<u32>() as f32;
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.data.len(), 100);
            for &value in &result.data {
                assert!(
                    (value - expected_sum).abs() < 0.001,
                    "Worker {} result mismatch: got {}, expected {}",
                    i,
                    value,
                    expected_sum
                );
            }
        }

        // All workers should have identical results
        for i in 1..results.len() {
            assert_eq!(results[0].data, results[i].data);
        }
    }

    #[test]
    fn test_ring_allreduce_numerical_stability() {
        // Test with small values that could cause precision issues
        let partial_results: Vec<Tensor> = (0..5)
            .map(|i| Tensor::filled(vec![100], (i + 1) as f32 * 0.0001))
            .collect();

        let results = simulate_ring_allreduce(partial_results);

        // Sum should be (1+2+3+4+5) * 0.0001 = 0.0015
        let expected_sum = 0.0015;
        for &value in &results[0].data {
            assert!(
                (value - expected_sum).abs() < 0.00001,
                "Numerical stability issue: got {}, expected {}",
                value,
                expected_sum
            );
        }
    }

    #[test]
    fn test_ring_allreduce_with_different_values() {
        // Workers have different values per element
        // Use 6 elements for 3 workers (divisible)
        let partial_results: Vec<Tensor> = vec![
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]),
            Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![6]),
            Tensor::new(vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0], vec![6]),
        ];

        let results = simulate_ring_allreduce(partial_results);

        // Expected sums: [1+7+13, 2+8+14, 3+9+15, 4+10+16, 5+11+17, 6+12+18]
        //              = [21, 24, 27, 30, 33, 36]
        let expected = [21.0, 24.0, 27.0, 30.0, 33.0, 36.0];

        for result in &results {
            for (i, &value) in result.data.iter().enumerate() {
                assert!(
                    (value - expected[i]).abs() < 0.001,
                    "Element {} mismatch: got {}, expected {}",
                    i,
                    value,
                    expected[i]
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_property_identical_output() {
        // Property: All workers produce identical output
        for n in 2..=8 {
            // Tensor size must be divisible by n for ring all-reduce
            let tensor_size = n * 10;
            let partial_results: Vec<Tensor> = (0..n)
                .map(|i| Tensor::filled(vec![tensor_size], (i + 1) as f32))
                .collect();

            let results = simulate_ring_allreduce(partial_results);

            for i in 1..results.len() {
                assert_eq!(
                    results[0].data, results[i].data,
                    "Property violated: workers 0 and {} differ for n={}",
                    i, n
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_property_sum_preservation() {
        // Property: Sum of partial results equals final result
        for n in 2..=8 {
            // Tensor size must be divisible by n for ring all-reduce
            let tensor_size = n * 10;
            let partial_results: Vec<Tensor> = (0..n)
                .map(|i| Tensor::filled(vec![tensor_size], (i + 1) as f32))
                .collect();

            // Calculate expected sum
            let expected_sum: f32 = (1..=n).sum::<usize>() as f32;

            let results = simulate_ring_allreduce(partial_results);

            // Verify sum is correct
            for &value in &results[0].data {
                assert!(
                    (value - expected_sum).abs() < 0.001,
                    "Sum preservation violated for n={}: got {}, expected {}",
                    n,
                    value,
                    expected_sum
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_stress_100_iterations() {
        // Stress test: Run 100 iterations without divergence
        for iteration in 0..100 {
            let n = 4; // Fixed number of workers for stress test
            let partial_results: Vec<Tensor> = (0..n)
                .map(|i| Tensor::filled(vec![100], ((i as f32) + 1.0) * (iteration as f32 + 1.0)))
                .collect();

            let results = simulate_ring_allreduce(partial_results);

            // Expected sum: (1+2+3+4) * (iteration+1) = 10 * (iteration+1)
            let expected_sum = 10.0 * (iteration as f32 + 1.0);

            for &value in &results[0].data {
                assert!(
                    (value - expected_sum).abs() < 0.01,
                    "Stress test failed at iteration {}: got {}, expected {}",
                    iteration,
                    value,
                    expected_sum
                );
            }

            // All workers should have identical results
            for i in 1..results.len() {
                assert_eq!(
                    results[0].data, results[i].data,
                    "Stress test: workers differ at iteration {}",
                    iteration
                );
            }
        }
    }

    // ============== AllReducePhase Tests ==============
    // Note: Phase serialization is exercised through tensor message transport tests.
}
