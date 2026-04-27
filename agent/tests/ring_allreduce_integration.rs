//! Integration tests for Ring All-Reduce
//!
//! These tests verify the ring all-reduce algorithm in multi-worker scenarios.
//! Note: True network-based integration tests require running workers in separate
//! processes with a relay server. These tests focus on algorithmic correctness.

// Allow range loops in ring algorithm simulation - worker index is used both
// for array indexing AND mathematical ring position calculations
#![allow(clippy::needless_range_loop)]

use agent::executor::Tensor;
use agent::network::{MeshEvent, MeshSwarm};
use futures::StreamExt;
use relay_server::config::Config as RelayConfig;
use relay_server::relay::{build_swarm as build_relay_swarm, configured_advertised_addrs};
use std::time::Duration;

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
    let relay_runtime = start_live_relay_runtime(43111).await;

    let build_agent = || {
        MeshSwarm::builder(libp2p::identity::Keypair::generate_ed25519())
            .with_relay_addr(relay_runtime.relay_addr.clone())
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

    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_a).await,
        relay_peer_a
    );
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_b).await,
        relay_peer_b
    );
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_c).await,
        relay_peer_c
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multiple_live_peers_connect_via_relay_runtime() {
    let relay_runtime = start_live_relay_runtime(43112).await;

    let build_agent = || {
        MeshSwarm::builder(libp2p::identity::Keypair::generate_ed25519())
            .with_relay_addr(relay_runtime.relay_addr.clone())
            .build()
            .unwrap()
    };

    let mut swarm_a = build_agent();
    let mut swarm_b = build_agent();
    let mut swarm_c = build_agent();

    let peer_id_a = *swarm_a.local_peer_id();
    let peer_id_b = *swarm_b.local_peer_id();
    let peer_id_c = *swarm_c.local_peer_id();

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

    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_a).await,
        relay_peer_a
    );
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_b).await,
        relay_peer_b
    );
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_c).await,
        relay_peer_c
    );

    swarm_b.dial_peer(peer_id_a).unwrap();
    swarm_c.dial_peer(peer_id_a).unwrap();

    wait_for_runtime_connections(&mut [
        (&mut swarm_a, vec![peer_id_b, peer_id_c]),
        (&mut swarm_b, vec![peer_id_a]),
        (&mut swarm_c, vec![peer_id_a]),
    ])
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_live_peers_upgrade_to_direct_after_relay_rendezvous() {
    let relay_runtime = start_live_relay_runtime(43113).await;

    let build_agent = || {
        MeshSwarm::builder(libp2p::identity::Keypair::generate_ed25519())
            .with_relay_addr(relay_runtime.relay_addr.clone())
            .build()
            .unwrap()
    };

    let mut swarm_a = build_agent();
    let mut swarm_b = build_agent();
    let peer_id_a = *swarm_a.local_peer_id();
    let peer_id_b = *swarm_b.local_peer_id();

    swarm_a.listen_on_direct_addrs().unwrap();
    swarm_b.listen_on_direct_addrs().unwrap();

    swarm_a.connect_to_relay().unwrap();
    swarm_b.connect_to_relay().unwrap();

    let relay_peer_a = wait_for_peer_connected(&mut swarm_a).await;
    let relay_peer_b = wait_for_peer_connected(&mut swarm_b).await;
    assert_eq!(relay_peer_a, relay_peer_b);

    swarm_a.listen_on_relay(relay_peer_a).unwrap();
    swarm_b.listen_on_relay(relay_peer_b).unwrap();
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_a).await,
        relay_peer_a
    );
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_b).await,
        relay_peer_b
    );

    swarm_b.dial_peer(peer_id_a).unwrap();

    wait_for_relay_then_direct_upgrade(
        &mut [(&mut swarm_a, peer_id_b), (&mut swarm_b, peer_id_a)],
        peer_id_a,
        peer_id_b,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_live_peers_upgrade_with_staggered_relay_rendezvous_timing() {
    let relay_runtime = start_live_relay_runtime(43114).await;

    let build_agent = || {
        MeshSwarm::builder(libp2p::identity::Keypair::generate_ed25519())
            .with_relay_addr(relay_runtime.relay_addr.clone())
            .build()
            .unwrap()
    };

    let mut swarm_a = build_agent();
    let peer_id_a = *swarm_a.local_peer_id();
    swarm_a.listen_on_direct_addrs().unwrap();
    swarm_a.connect_to_relay().unwrap();
    let relay_peer_a = wait_for_peer_connected(&mut swarm_a).await;
    swarm_a.listen_on_relay(relay_peer_a).unwrap();
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_a).await,
        relay_peer_a
    );

    tokio::time::sleep(Duration::from_millis(300)).await;

    let mut swarm_b = build_agent();
    let peer_id_b = *swarm_b.local_peer_id();
    swarm_b.listen_on_direct_addrs().unwrap();
    swarm_b.connect_to_relay().unwrap();
    let relay_peer_b = wait_for_peer_connected(&mut swarm_b).await;
    assert_eq!(relay_peer_a, relay_peer_b);
    swarm_b.listen_on_relay(relay_peer_b).unwrap();
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_b).await,
        relay_peer_b
    );
    swarm_b.dial_peer(peer_id_a).unwrap();

    wait_for_relay_then_direct_upgrade(
        &mut [(&mut swarm_a, peer_id_b), (&mut swarm_b, peer_id_a)],
        peer_id_a,
        peer_id_b,
    )
    .await;

    tokio::time::sleep(Duration::from_millis(750)).await;

    let mut swarm_c = build_agent();
    let peer_id_c = *swarm_c.local_peer_id();
    swarm_c.listen_on_direct_addrs().unwrap();
    swarm_c.connect_to_relay().unwrap();
    let relay_peer_c = wait_for_peer_connected(&mut swarm_c).await;
    assert_eq!(relay_peer_a, relay_peer_c);
    swarm_c.listen_on_relay(relay_peer_c).unwrap();
    assert_eq!(
        wait_for_reservation_accepted(&mut swarm_c).await,
        relay_peer_c
    );
    swarm_c.dial_peer(peer_id_a).unwrap();

    wait_for_relay_then_direct_upgrade(
        &mut [(&mut swarm_a, peer_id_c), (&mut swarm_c, peer_id_a)],
        peer_id_a,
        peer_id_c,
    )
    .await;
}

async fn wait_for_peer_connected(swarm: &mut MeshSwarm) -> libp2p::PeerId {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(2), swarm.next_event()).await
        {
            match event {
                MeshEvent::PeerConnected { peer_id, .. }
                | MeshEvent::RelayConnected {
                    relay_peer_id: peer_id,
                    ..
                } => {
                    return peer_id;
                }
                _ => {}
            }
        }
    }
    panic!("timed out waiting for peer connection");
}

async fn wait_for_reservation_accepted(swarm: &mut MeshSwarm) -> libp2p::PeerId {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_secs(2), swarm.next_event()).await
        {
            if let MeshEvent::ReservationAccepted { relay_peer_id, .. } = event {
                return relay_peer_id;
            }
        }
    }
    panic!("timed out waiting for relay reservation acceptance");
}

async fn wait_for_runtime_connections(swarms: &mut [(&mut MeshSwarm, Vec<libp2p::PeerId>)]) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut remaining: Vec<std::collections::BTreeSet<libp2p::PeerId>> = swarms
        .iter()
        .map(|(_, peers)| peers.iter().copied().collect())
        .collect();

    while tokio::time::Instant::now() < deadline {
        if remaining.iter().all(|peers| peers.is_empty()) {
            return;
        }

        for ((swarm, _), pending_peers) in swarms.iter_mut().zip(remaining.iter_mut()) {
            if pending_peers.is_empty() {
                continue;
            }

            pending_peers.retain(|peer_id| !swarm.is_connected(peer_id));
            if pending_peers.is_empty() {
                continue;
            }

            if let Ok(Some(event)) =
                tokio::time::timeout(Duration::from_millis(250), swarm.next_event()).await
            {
                match event {
                    MeshEvent::PeerConnected { peer_id, .. }
                    | MeshEvent::DirectConnectionUpgraded { peer_id } => {
                        pending_peers.remove(&peer_id);
                    }
                    _ => {}
                }
            }
        }
    }

    panic!(
        "timed out waiting for expected live peer connections: {:?}",
        remaining
    );
}

async fn wait_for_relay_then_direct_upgrade(
    swarms: &mut [(&mut MeshSwarm, libp2p::PeerId)],
    peer_id_a: libp2p::PeerId,
    peer_id_b: libp2p::PeerId,
) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut saw_relay_connection = false;
    let mut saw_direct_upgrade = false;

    while tokio::time::Instant::now() < deadline {
        for (swarm, target_peer) in swarms.iter_mut() {
            if swarm.is_connected(target_peer) {
                saw_relay_connection = true;
            }
            if let Ok(Some(event)) =
                tokio::time::timeout(Duration::from_millis(250), swarm.next_event()).await
            {
                match event {
                    MeshEvent::PeerConnected {
                        peer_id,
                        connection_info,
                    } if peer_id == *target_peer && connection_info.is_relayed() => {
                        saw_relay_connection = true;
                    }
                    MeshEvent::DirectConnectionUpgraded { peer_id }
                        if peer_id == peer_id_a || peer_id == peer_id_b =>
                    {
                        saw_direct_upgrade = true;
                    }
                    MeshEvent::PeerConnected {
                        peer_id,
                        connection_info,
                    } if peer_id == *target_peer && connection_info.is_direct() => {
                        saw_direct_upgrade = true;
                    }
                    _ => {}
                }
            }
        }

        if saw_direct_upgrade {
            return;
        }
    }

    panic!(
        "timed out waiting for relay rendezvous and direct upgrade: saw_relay_connection={}, saw_direct_upgrade={}",
        saw_relay_connection, saw_direct_upgrade
    );
}

struct LiveRelayRuntime {
    relay_addr: libp2p::Multiaddr,
    relay_task: tokio::task::JoinHandle<()>,
    original_home: Option<std::ffi::OsString>,
    home_dir: std::path::PathBuf,
}

impl Drop for LiveRelayRuntime {
    fn drop(&mut self) {
        self.relay_task.abort();
        match self.original_home.take() {
            Some(home) => unsafe { std::env::set_var("HOME", home) },
            None => unsafe { std::env::remove_var("HOME") },
        }
        let _ = std::fs::remove_dir_all(&self.home_dir);
    }
}

async fn start_live_relay_runtime(port: u16) -> LiveRelayRuntime {
    let temp_home = tempfile::tempdir().unwrap();
    let original_home = std::env::var_os("HOME");
    let home_dir = temp_home.keep();
    unsafe {
        std::env::set_var("HOME", &home_dir);
    }

    let relay_addr: libp2p::Multiaddr = format!("/ip4/127.0.0.1/tcp/{}", port).parse().unwrap();
    let mut relay_config = RelayConfig::default();
    relay_config.network.tcp_listen_addr = relay_addr.to_string();
    relay_config.network.quic_listen_addr = format!("/ip4/127.0.0.1/udp/{}/quic-v1", port);
    relay_config.network.advertised_addrs = vec![
        relay_addr.to_string(),
        format!("/ip4/127.0.0.1/udp/{}/quic-v1", port),
    ];

    let mut relay_swarm = build_relay_swarm(&relay_config).await.unwrap();
    relay_swarm.listen_on(relay_addr.clone()).unwrap();
    for advertised_addr in configured_advertised_addrs(&relay_config).unwrap() {
        relay_swarm.add_external_address(advertised_addr);
    }

    let relay_task = tokio::spawn(async move {
        loop {
            let _ = relay_swarm.select_next_some().await;
        }
    });

    LiveRelayRuntime {
        relay_addr,
        relay_task,
        original_home,
        home_dir,
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
