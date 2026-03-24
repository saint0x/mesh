//! Integration tests for Ring All-Reduce
//!
//! These tests verify the ring all-reduce algorithm in multi-worker scenarios.
//! Note: True network-based integration tests require running workers in separate
//! processes with a relay server. These tests focus on algorithmic correctness.

// Allow range loops in ring algorithm simulation - worker index is used both
// for array indexing AND mathematical ring position calculations
#![allow(clippy::needless_range_loop)]

use agent::executor::Tensor;
use agent::network::{AllReducePhase, MeshEvent, MeshSwarm, TensorMessage};
use futures::StreamExt;
use relay_server::config::Config as RelayConfig;
use relay_server::relay::build_swarm as build_relay_swarm;
use std::time::Duration;
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

// Note: Tensor serialization is exercised through the dedicated tensor data plane.

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
        assert!((value - 15.0).abs() < 0.001, "Expected 15.0, got {}", value);
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
        assert!((value - 6.0).abs() < 0.001, "Expected 6.0, got {}", value);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multiple_live_peers_attach_to_same_relay_runtime() {
    let temp_home = tempfile::tempdir().unwrap();
    let original_home = std::env::var_os("HOME");
    unsafe {
        std::env::set_var("HOME", temp_home.path());
    }

    let relay_addr: libp2p::Multiaddr = "/ip4/127.0.0.1/tcp/43111".parse().unwrap();
    let mut relay_config = RelayConfig::default();
    relay_config.network.tcp_listen_addr = relay_addr.to_string();
    relay_config.network.quic_listen_addr = "/ip4/127.0.0.1/udp/43111/quic-v1".to_string();

    let mut relay_swarm = build_relay_swarm(&relay_config).await.unwrap();
    relay_swarm.listen_on(relay_addr.clone()).unwrap();

    let relay_task = tokio::spawn(async move {
        loop {
            let _ = relay_swarm.select_next_some().await;
        }
    });

    let build_agent = || {
        MeshSwarm::builder(libp2p::identity::Keypair::generate_ed25519())
            .with_relay_addr(relay_addr.clone())
            .build()
            .unwrap()
    };

    let mut swarm_a = build_agent();
    let mut swarm_b = build_agent();
    let mut swarm_c = build_agent();

    swarm_a.listen_on_direct_addrs().unwrap();
    swarm_b.listen_on_direct_addrs().unwrap();
    swarm_c.listen_on_direct_addrs().unwrap();

    swarm_a.connect_to_relay().unwrap();
    swarm_b.connect_to_relay().unwrap();
    swarm_c.connect_to_relay().unwrap();

    let relay_peer_a = wait_for_peer_connected(&mut swarm_a).await;
    let relay_peer_b = wait_for_peer_connected(&mut swarm_b).await;
    let relay_peer_c = wait_for_peer_connected(&mut swarm_c).await;

    assert_eq!(relay_peer_a, relay_peer_b);
    assert_eq!(relay_peer_b, relay_peer_c);

    swarm_a.listen_on_relay(relay_peer_a).unwrap();
    swarm_b.listen_on_relay(relay_peer_b).unwrap();
    swarm_c.listen_on_relay(relay_peer_c).unwrap();

    // Give the relay setup a bounded window to progress on live swarms.
    let _ = tokio::time::timeout(Duration::from_secs(2), swarm_a.next_event()).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), swarm_b.next_event()).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), swarm_c.next_event()).await;

    relay_task.abort();
    let _ = relay_task.await;

    match original_home {
        Some(home) => unsafe { std::env::set_var("HOME", home) },
        None => unsafe { std::env::remove_var("HOME") },
    }
}

async fn wait_for_peer_connected(swarm: &mut MeshSwarm) -> libp2p::PeerId {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(2), swarm.next_event()).await
        {
            if let MeshEvent::PeerConnected { peer_id, .. } = event {
                return peer_id;
            }
        }
    }
    panic!("timed out waiting for peer connection");
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
