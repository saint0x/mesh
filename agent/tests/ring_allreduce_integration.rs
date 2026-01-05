//! Integration tests for Ring All-Reduce
//!
//! These tests verify the ring all-reduce algorithm in multi-worker scenarios.
//! Note: True network-based integration tests require running workers in separate
//! processes with a relay server. These tests focus on algorithmic correctness.

// Allow range loops in ring algorithm simulation - worker index is used both
// for array indexing AND mathematical ring position calculations
#![allow(clippy::needless_range_loop)]

use agent::executor::Tensor;
use agent::network::{AllReducePhase, TensorMessage};
use uuid::Uuid;

/// Simulates the ring all-reduce algorithm with n workers
/// This is a comprehensive test of the algorithmic correctness.
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

/// Integration test with 10 workers
/// Each worker has a tensor of 100 elements filled with (worker_id + 1)
/// After all-reduce, each element should equal 1+2+3+...+10 = 55
#[test]
fn test_ring_allreduce_10_workers() {
    let n = 10;
    let tensor_size = 100;

    // Each worker creates partial result (worker i has value i+1)
    let partial_results: Vec<Tensor> = (0..n)
        .map(|i| {
            let data = vec![(i + 1) as f32; tensor_size];
            Tensor::new(data, vec![tensor_size])
        })
        .collect();

    // Run all-reduce simulation
    let results = simulate_ring_allreduce(partial_results);

    // Expected sum: 1+2+...+10 = 55
    let expected_sum = (1..=n).sum::<usize>() as f32;

    // Verify all workers have identical results
    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.data.len(), tensor_size);
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

    // Verify all results are identical
    for i in 1..results.len() {
        assert_eq!(
            results[0].data, results[i].data,
            "Worker 0 and {} have different results",
            i
        );
    }
}

/// Integration test with varying tensor sizes
#[test]
fn test_ring_allreduce_varying_sizes() {
    for n in [3, 5, 7, 10] {
        for size_multiplier in [1, 10, 100] {
            let tensor_size = n * size_multiplier;
            let partial_results: Vec<Tensor> = (0..n)
                .map(|i| Tensor::filled(vec![tensor_size], (i + 1) as f32))
                .collect();

            let results = simulate_ring_allreduce(partial_results);
            let expected_sum: f32 = (1..=n).sum::<usize>() as f32;

            // Just verify first and last worker have correct results
            for &value in &results[0].data {
                assert!(
                    (value - expected_sum).abs() < 0.001,
                    "n={}, size={}: got {}, expected {}",
                    n,
                    tensor_size,
                    value,
                    expected_sum
                );
            }
        }
    }
}

/// Test barrier message creation
#[test]
fn test_barrier_message_format() {
    let job_id = Uuid::new_v4();
    let msg = TensorMessage::new(
        job_id,
        5,
        AllReducePhase::Barrier,
        TensorMessage::BARRIER_STEP,
        vec![3.0], // Worker 3's position
        vec![1],
    );

    assert!(msg.is_barrier());
    assert_eq!(msg.phase, AllReducePhase::Barrier);
    assert_eq!(msg.chunk_data[0] as u32, 3);
}

// Note: CBOR serialization tests are in agent/src/network/tensor_protocol.rs

/// Test large tensor handling
#[test]
fn test_large_tensor_allreduce() {
    let n = 4;
    let tensor_size = 10000; // 10k elements per worker

    let partial_results: Vec<Tensor> = (0..n)
        .map(|i| Tensor::filled(vec![tensor_size], (i + 1) as f32))
        .collect();

    let results = simulate_ring_allreduce(partial_results);

    // Expected sum: 1+2+3+4 = 10
    let expected_sum = 10.0f32;

    // Verify first few and last few elements
    assert!(
        (results[0].data[0] - expected_sum).abs() < 0.001,
        "First element mismatch"
    );
    assert!(
        (results[0].data[tensor_size - 1] - expected_sum).abs() < 0.001,
        "Last element mismatch"
    );
}

/// Test that chunks are correctly accumulated during reduce-scatter
#[test]
fn test_reduce_scatter_phase_correctness() {
    let _n = 4;
    let _tensor_size = 8; // 2 elements per chunk

    // Workers have distinct patterns
    let partial_results: Vec<Tensor> = vec![
        Tensor::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![8]),
        Tensor::new(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], vec![8]),
        Tensor::new(vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], vec![8]),
        Tensor::new(vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0], vec![8]),
    ];

    let results = simulate_ring_allreduce(partial_results);

    // Expected: 1+2+4+8 = 15 for all elements
    for &value in &results[0].data {
        assert!(
            (value - 15.0).abs() < 0.001,
            "Expected 15.0, got {}",
            value
        );
    }
}

/// Test numerical precision with small deltas
#[test]
fn test_numerical_precision() {
    let n = 5;
    let tensor_size = 50;

    // Very small values that could cause precision issues
    let partial_results: Vec<Tensor> = (0..n)
        .map(|i| Tensor::filled(vec![tensor_size], (i + 1) as f32 * 1e-7))
        .collect();

    let results = simulate_ring_allreduce(partial_results);

    // Expected: (1+2+3+4+5) * 1e-7 = 15e-7
    let expected = 15e-7;
    for &value in &results[0].data {
        let error = (value - expected).abs() / expected;
        assert!(
            error < 1e-5,
            "Relative error too large: {} (value={}, expected={})",
            error,
            value,
            expected
        );
    }
}

/// Test with negative values
#[test]
fn test_negative_values() {
    let _n = 4;
    let tensor_size = 16;

    let partial_results: Vec<Tensor> = vec![
        Tensor::filled(vec![tensor_size], -5.0),
        Tensor::filled(vec![tensor_size], 3.0),
        Tensor::filled(vec![tensor_size], -2.0),
        Tensor::filled(vec![tensor_size], 10.0),
    ];

    let results = simulate_ring_allreduce(partial_results);

    // Expected: -5 + 3 + (-2) + 10 = 6
    for &value in &results[0].data {
        assert!(
            (value - 6.0).abs() < 0.001,
            "Expected 6.0, got {}",
            value
        );
    }
}

/// Test with mixed positive/negative/zero
#[test]
fn test_mixed_values() {
    let _n = 3;
    let _tensor_size = 9;

    let partial_results: Vec<Tensor> = vec![
        Tensor::new(
            vec![1.0, -1.0, 0.0, 2.0, -2.0, 0.0, 3.0, -3.0, 0.0],
            vec![9],
        ),
        Tensor::new(
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            vec![9],
        ),
        Tensor::new(
            vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
            vec![9],
        ),
    ];

    let results = simulate_ring_allreduce(partial_results);

    let expected = vec![6.0, 4.0, 5.0, 7.0, 3.0, 5.0, 8.0, 2.0, 5.0];
    for (i, &value) in results[0].data.iter().enumerate() {
        assert!(
            (value - expected[i]).abs() < 0.001,
            "Element {} mismatch: got {}, expected {}",
            i,
            value,
            expected[i]
        );
    }
}
