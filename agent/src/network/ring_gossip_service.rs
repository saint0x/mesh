use crate::network::{MemberStatus, RingGossipMessage, RingMember, RingState, RingTopology};
use crate::pki::{DeviceKeyPair, NodeId, PoolId};
use anyhow::Result;
use libp2p::PeerId;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{interval, Duration};

/// Ring gossip service for P2P ring formation
pub struct RingGossipService {
    /// Pool this ring belongs to
    pool_id: PoolId,
    /// My node ID
    node_id: NodeId,
    /// My libp2p PeerID
    peer_id: PeerId,
    /// Device keypair for signing gossip
    device_keypair: DeviceKeyPair,
    /// Shared ring state
    ring_state: Arc<RwLock<RingState>>,
    /// Channel to send ring gossip messages to beacon broadcaster
    gossip_tx: mpsc::Sender<RingGossipMessage>,
    /// Channel to receive ring gossip from beacon listener
    gossip_rx: mpsc::Receiver<RingGossipMessage>,
    /// Channel to broadcast topology changes
    topology_tx: broadcast::Sender<RingTopology>,
    /// Last topology version notified
    last_notified_version: u64,
}

impl RingGossipService {
    /// Create a new ring gossip service
    pub fn new(
        pool_id: PoolId,
        node_id: NodeId,
        peer_id: PeerId,
        device_keypair: DeviceKeyPair,
        gossip_tx: mpsc::Sender<RingGossipMessage>,
        gossip_rx: mpsc::Receiver<RingGossipMessage>,
    ) -> (Self, Arc<RwLock<RingState>>, broadcast::Receiver<RingTopology>) {
        let ring_state = Arc::new(RwLock::new(RingState::new(pool_id)));
        let (topology_tx, topology_rx) = broadcast::channel(16);

        let service = Self {
            pool_id,
            node_id,
            peer_id,
            device_keypair,
            ring_state: ring_state.clone(),
            gossip_tx,
            gossip_rx,
            topology_tx,
            last_notified_version: 0,
        };

        (service, ring_state, topology_rx)
    }

    /// Run the ring gossip service
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(
            pool_id = %self.pool_id,
            node_id = %self.node_id,
            "Starting ring gossip service"
        );

        // Join ring immediately
        self.announce_presence().await?;

        // Periodic tasks
        let mut announce_interval = interval(Duration::from_secs(5));
        let mut convergence_check = interval(Duration::from_secs(2));

        loop {
            tokio::select! {
                _ = announce_interval.tick() => {
                    if let Err(e) = self.announce_presence().await {
                        tracing::error!(error = %e, "Failed to announce presence");
                    }
                }

                _ = convergence_check.tick() => {
                    if let Err(e) = self.check_convergence().await {
                        tracing::error!(error = %e, "Failed to check convergence");
                    }
                }

                Some(gossip) = self.gossip_rx.recv() => {
                    if let Err(e) = self.handle_gossip(gossip).await {
                        tracing::error!(error = %e, "Failed to handle gossip");
                    }
                }
            }
        }
    }

    /// Announce presence in ring
    async fn announce_presence(&mut self) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        // Update our own member info in ring state
        let member = RingMember {
            node_id: self.node_id,
            peer_id: Some(self.peer_id),
            lan_addr: None, // Will be populated by beacon discovery
            last_seen: now,
            status: MemberStatus::Active,
        };

        {
            let mut state = self.ring_state.write().await;
            state.add_member(member);
        }

        // Broadcast ring state via gossip
        self.broadcast_ring_state().await?;

        Ok(())
    }

    /// Broadcast current ring state
    async fn broadcast_ring_state(&self) -> Result<()> {
        let state = self.ring_state.read().await.clone();

        let gossip = RingGossipMessage::new(self.pool_id, state, &self.device_keypair);

        self.gossip_tx.send(gossip).await?;

        let member_count = self.ring_state.read().await.members.len();
        tracing::trace!(
            pool_id = %self.pool_id,
            version = self.last_notified_version,
            members = member_count,
            "Ring state broadcast"
        );

        Ok(())
    }

    /// Handle incoming ring gossip
    async fn handle_gossip(&mut self, gossip: RingGossipMessage) -> Result<()> {
        // Ignore our own gossip
        if gossip.sender_node_id == self.node_id {
            return Ok(());
        }

        // Verify gossip is for our pool
        if gossip.pool_id != self.pool_id {
            tracing::trace!("Ignoring gossip from different pool");
            return Ok(());
        }

        // TODO: Verify signature in Phase 2
        // if !gossip.verify(&sender_device_pubkey) {
        //     tracing::warn!("Invalid gossip signature");
        //     return Ok(());
        // }

        // Merge ring state
        let changed = {
            let mut state = self.ring_state.write().await;
            state.merge(&gossip.ring_state)
        };

        if changed {
            {
                let state = self.ring_state.read().await;
                tracing::debug!(
                    sender = %gossip.sender_node_id,
                    version = gossip.version,
                    members = state.members.len(),
                    "Ring state updated from gossip"
                );
            }

            // Check if we should notify about topology change
            self.check_convergence().await?;
        }

        Ok(())
    }

    /// Check if ring has converged and notify if topology changed
    async fn check_convergence(&mut self) -> Result<()> {
        let state = self.ring_state.read().await;

        // Need at least 2 members for a ring
        if state.members.len() < 2 {
            tracing::trace!("Ring too small (need 2+ members)");
            return Ok(());
        }

        // Check if we're in the ring
        if state.get_ring_position(&self.node_id).is_none() {
            tracing::trace!("We're not in ring yet");
            return Ok(());
        }

        // Only notify if version changed
        if state.version <= self.last_notified_version {
            return Ok(());
        }

        // Calculate topology
        let topology = match RingTopology::from_ring_state(&state, &self.node_id) {
            Ok(t) => t,
            Err(e) => {
                tracing::trace!(error = %e, "Failed to calculate topology");
                return Ok(());
            }
        };

        // Notify InferenceCoordinator
        match self.topology_tx.send(topology.clone()) {
            Ok(_) => {
                self.last_notified_version = state.version;
                tracing::info!(
                    position = topology.my_position,
                    shard_range = format!("[{}, {})", topology.shard_start, topology.shard_end),
                    left_neighbor = %topology.left_neighbor,
                    right_neighbor = %topology.right_neighbor,
                    ring_size = topology.ring_size,
                    version = state.version,
                    "Ring topology converged"
                );
            }
            Err(_) => {
                // No receivers, that's okay
            }
        }

        Ok(())
    }

    /// Get current ring state (for testing)
    pub async fn get_ring_state(&self) -> RingState {
        self.ring_state.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pki::DeviceKeyPair;

    #[tokio::test]
    async fn test_ring_gossip_service_startup() {
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        let peer_id = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_unused) = mpsc::channel(10);
        let (_gossip_tx_unused, gossip_rx) = mpsc::channel(10);

        let (service, ring_state, _topology_rx) = RingGossipService::new(
            pool_id,
            node_id,
            peer_id,
            device.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Announce presence
        let mut service = service;
        service.announce_presence().await.unwrap();

        // Check ring state contains us
        let state = ring_state.read().await;
        assert_eq!(state.members.len(), 1);
        assert!(state.get_ring_position(&node_id).is_some());
    }

    #[tokio::test]
    async fn test_ring_gossip_merge() {
        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let node1 = device1.node_id();
        let node2 = device2.node_id();
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        // Service 1
        let (gossip_tx1, _gossip_rx_for_tx1) = mpsc::channel(10);
        let (gossip_tx_to_service1, gossip_rx1) = mpsc::channel(10);
        let (mut service1, ring_state1, _) = RingGossipService::new(
            pool_id,
            node1,
            peer1,
            device1.clone(),
            gossip_tx1,
            gossip_rx1,
        );

        // Service 2
        let (gossip_tx2, _gossip_rx_for_tx2) = mpsc::channel(10);
        let (_gossip_tx_to_service2, gossip_rx2) = mpsc::channel(10);
        let (mut service2, ring_state2, _) = RingGossipService::new(
            pool_id,
            node2,
            peer2,
            device2.clone(),
            gossip_tx2,
            gossip_rx2,
        );

        // Both announce presence
        service1.announce_presence().await.unwrap();
        service2.announce_presence().await.unwrap();

        // Create gossip from service2 state
        let state2 = ring_state2.read().await.clone();
        let gossip2 = RingGossipMessage::new(pool_id, state2, &device2);

        // Send gossip2 to service1
        gossip_tx_to_service1.send(gossip2).await.unwrap();

        // Receive and handle gossip (fix borrow conflict)
        let received_gossip = service1.gossip_rx.recv().await.unwrap();
        service1.handle_gossip(received_gossip).await.unwrap();

        // Service1 should now have both members
        let state1 = ring_state1.read().await;
        assert_eq!(state1.members.len(), 2);
        assert!(state1.get_ring_position(&node1).is_some());
        assert!(state1.get_ring_position(&node2).is_some());
    }

    // =========================================================================
    // Phase 3: Service Layer Tests (23-31)
    // =========================================================================

    /// Test 23: Announce presence should update ring state with own info
    #[tokio::test]
    async fn test_service_announce_presence_updates_ring() {
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        let peer_id = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, ring_state, _) = RingGossipService::new(
            pool_id,
            node_id,
            peer_id,
            device.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Ring should be empty initially
        assert_eq!(ring_state.read().await.members.len(), 0);

        // Announce presence
        service.announce_presence().await.unwrap();

        // Ring should contain us now
        let state = ring_state.read().await;
        assert_eq!(state.members.len(), 1);

        // Verify our member info
        let member = state.members.values().next().unwrap();
        assert_eq!(member.node_id, node_id);
        assert_eq!(member.peer_id, Some(peer_id));
        assert_eq!(member.status, MemberStatus::Active);
    }

    /// Test 24: Service should ignore gossip from itself
    #[tokio::test]
    async fn test_service_handle_gossip_ignores_self() {
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        let peer_id = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, ring_state, _) = RingGossipService::new(
            pool_id,
            node_id,
            peer_id,
            device.clone(),
            gossip_tx,
            gossip_rx,
        );

        service.announce_presence().await.unwrap();
        let initial_version = ring_state.read().await.version;

        // Create gossip from our own state (simulating receiving our own broadcast)
        let our_state = ring_state.read().await.clone();
        let our_gossip = RingGossipMessage::new(pool_id, our_state, &device);

        // Handle our own gossip
        service.handle_gossip(our_gossip).await.unwrap();

        // Ring state should be unchanged (version unchanged)
        let final_version = ring_state.read().await.version;
        assert_eq!(final_version, initial_version, "Should ignore self gossip");
    }

    /// Test 25: Service should ignore gossip from different pool
    #[tokio::test]
    async fn test_service_handle_gossip_different_pool() {
        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let node1 = device1.node_id();
        let peer1 = PeerId::random();
        let pool_a = PoolId::from_bytes([1u8; 32]);
        let pool_b = PoolId::from_bytes([2u8; 32]);

        // Service 1 on pool_a
        let (gossip_tx1, _gossip_rx_for_tx1) = mpsc::channel(10);
        let (_gossip_tx_for_rx1, gossip_rx1) = mpsc::channel(10);
        let (mut service1, ring_state1, _) = RingGossipService::new(
            pool_a,
            node1,
            peer1,
            device1.clone(),
            gossip_tx1,
            gossip_rx1,
        );

        service1.announce_presence().await.unwrap();
        let initial_count = ring_state1.read().await.members.len();
        let initial_version = ring_state1.read().await.version;

        // Create gossip from pool_b
        let mut other_ring = RingState::new(pool_b);
        other_ring.add_member(RingMember {
            node_id: device2.node_id(),
            peer_id: None,
            lan_addr: None,
            last_seen: 1000,
            status: MemberStatus::Active,
        });
        let other_gossip = RingGossipMessage::new(pool_b, other_ring, &device2);

        // Handle gossip from different pool
        service1.handle_gossip(other_gossip).await.unwrap();

        // Ring should be unchanged
        let final_count = ring_state1.read().await.members.len();
        let final_version = ring_state1.read().await.version;
        assert_eq!(final_count, initial_count, "Should ignore different pool gossip");
        assert_eq!(final_version, initial_version, "Version should not change");
    }

    /// Test 26: Handling gossip should trigger convergence check
    #[tokio::test]
    async fn test_handle_gossip_triggers_convergence_check() {
        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let node1 = device1.node_id();
        let node2 = device2.node_id();
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        // Service 1
        let (gossip_tx1, _gossip_rx_for_tx1) = mpsc::channel(10);
        let (_gossip_tx_for_rx1, gossip_rx1) = mpsc::channel(10);
        let (mut service1, ring_state1, mut topology_rx1) = RingGossipService::new(
            pool_id,
            node1,
            peer1,
            device1.clone(),
            gossip_tx1,
            gossip_rx1,
        );

        // Service 2
        let (gossip_tx2, _gossip_rx_for_tx2) = mpsc::channel(10);
        let (_gossip_tx_for_rx2, gossip_rx2) = mpsc::channel(10);
        let (mut service2, ring_state2, _) = RingGossipService::new(
            pool_id,
            node2,
            peer2,
            device2.clone(),
            gossip_tx2,
            gossip_rx2,
        );

        // Both announce presence
        service1.announce_presence().await.unwrap();
        service2.announce_presence().await.unwrap();

        // Create gossip from service2
        let state2 = ring_state2.read().await.clone();
        let gossip2 = RingGossipMessage::new(pool_id, state2, &device2);

        // Service1 handles gossip2 → should trigger convergence
        service1.handle_gossip(gossip2).await.unwrap();

        // Check that ring was updated
        let state1 = ring_state1.read().await;
        assert_eq!(state1.members.len(), 2, "Ring should have 2 members after merge");

        // Check that topology was broadcast (convergence triggered)
        let topology = topology_rx1.try_recv();
        assert!(topology.is_ok(), "Convergence should broadcast topology");

        let topo = topology.unwrap();
        assert_eq!(topo.ring_size, 2);
        assert_eq!(topo.pool_id, pool_id);
    }

    /// Test 27: Convergence requires at least 2 members
    #[tokio::test]
    async fn test_convergence_requires_two_members() {
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        let peer_id = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, _ring_state, mut topology_rx) = RingGossipService::new(
            pool_id,
            node_id,
            peer_id,
            device.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Announce presence (adds self to ring)
        service.announce_presence().await.unwrap();

        // Check convergence with only 1 member
        service.check_convergence().await.unwrap();

        // Should NOT broadcast topology (need 2+ members)
        let result = topology_rx.try_recv();
        assert!(
            result.is_err(),
            "Should not broadcast topology with <2 members"
        );
    }

    /// Test 28: Convergence requires self to be in ring
    #[tokio::test]
    async fn test_convergence_requires_self_in_ring() {
        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let device3 = DeviceKeyPair::generate();
        let node1 = device1.node_id();
        let peer1 = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, ring_state, mut topology_rx) = RingGossipService::new(
            pool_id,
            node1,
            peer1,
            device1.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Manually add other members to ring (but not ourselves)
        {
            let mut state = ring_state.write().await;
            state.add_member(RingMember {
                node_id: device2.node_id(),
                peer_id: None,
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
            state.add_member(RingMember {
                node_id: device3.node_id(),
                peer_id: None,
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
        }

        // Check convergence (we're not in ring)
        service.check_convergence().await.unwrap();

        // Should NOT broadcast topology (we're not in ring)
        let result = topology_rx.try_recv();
        assert!(result.is_err(), "Should not broadcast if self not in ring");
    }

    /// Test 29: Convergence tracks version to avoid duplicate notifications
    #[tokio::test]
    async fn test_convergence_version_tracking() {
        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let node1 = device1.node_id();
        let node2 = device2.node_id();
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, ring_state, mut topology_rx) = RingGossipService::new(
            pool_id,
            node1,
            peer1,
            device1.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Add both members to ring
        service.announce_presence().await.unwrap();
        {
            let mut state = ring_state.write().await;
            state.add_member(RingMember {
                node_id: node2,
                peer_id: Some(peer2),
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
        }

        let version1 = ring_state.read().await.version;

        // Check convergence → should notify
        service.check_convergence().await.unwrap();
        let result1 = topology_rx.try_recv();
        assert!(result1.is_ok(), "First convergence should notify");

        // Check convergence again with same version → should NOT notify
        service.check_convergence().await.unwrap();
        let result2 = topology_rx.try_recv();
        assert!(
            result2.is_err(),
            "Second convergence with same version should not notify"
        );

        // Update ring version by adding a new member
        {
            let mut state = ring_state.write().await;
            let device3 = DeviceKeyPair::generate();
            state.add_member(RingMember {
                node_id: device3.node_id(),
                peer_id: None,
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
        }

        let version2 = ring_state.read().await.version;
        assert!(version2 > version1, "Version should increment");

        // Check convergence with new version → should notify again
        service.check_convergence().await.unwrap();
        let result3 = topology_rx.try_recv();
        assert!(result3.is_ok(), "Convergence with new version should notify");
    }

    /// Test 30: Convergence calculates correct topology
    #[tokio::test]
    async fn test_convergence_topology_calculation() {
        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let device3 = DeviceKeyPair::generate();
        let node1 = device1.node_id();
        let node2 = device2.node_id();
        let node3 = device3.node_id();
        let peer1 = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, ring_state, mut topology_rx) = RingGossipService::new(
            pool_id,
            node1,
            peer1,
            device1.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Create 3-node ring
        service.announce_presence().await.unwrap();
        {
            let mut state = ring_state.write().await;
            state.add_member(RingMember {
                node_id: node2,
                peer_id: None,
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
            state.add_member(RingMember {
                node_id: node3,
                peer_id: None,
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
        }

        // Trigger convergence
        service.check_convergence().await.unwrap();

        // Verify topology
        let topology = topology_rx.try_recv().unwrap();
        assert_eq!(topology.ring_size, 3);
        assert_eq!(topology.pool_id, pool_id);

        // Verify shard range is valid (no gaps, covers 0-8192)
        assert!(topology.shard_start < topology.shard_end);
        assert!(topology.shard_end <= 8192);

        // Verify neighbors are different from self
        assert_ne!(topology.left_neighbor, node1);
        assert_ne!(topology.right_neighbor, node1);
        assert_ne!(topology.left_neighbor, topology.right_neighbor);
    }

    /// Test 31: Gossip merge with no changes should skip convergence
    #[tokio::test]
    async fn test_gossip_merge_no_change_skips_convergence() {
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        let peer_id = PeerId::random();
        let pool_id = PoolId::from_bytes([1u8; 32]);

        let (gossip_tx, _gossip_rx_for_tx) = mpsc::channel(10);
        let (_gossip_tx_for_rx, gossip_rx) = mpsc::channel(10);

        let (mut service, ring_state, mut topology_rx) = RingGossipService::new(
            pool_id,
            node_id,
            peer_id,
            device.clone(),
            gossip_tx,
            gossip_rx,
        );

        // Announce presence
        service.announce_presence().await.unwrap();

        // Drain any existing topology notifications
        while topology_rx.try_recv().is_ok() {}

        // Create stale gossip (older timestamp)
        let stale_state = ring_state.read().await.clone();
        let stale_gossip = RingGossipMessage {
            pool_id,
            ring_state: stale_state,
            sender_node_id: node_id,
            version: 1,
            signature: [0u8; 64],
        };

        // Create a different sender to avoid self-ignore
        let device2 = DeviceKeyPair::generate();
        let mut stale_gossip_from_other = stale_gossip.clone();
        stale_gossip_from_other.sender_node_id = device2.node_id();

        // Handle stale gossip (should not change ring)
        service.handle_gossip(stale_gossip_from_other).await.unwrap();

        // Should NOT broadcast topology (no changes)
        let result = topology_rx.try_recv();
        assert!(
            result.is_err(),
            "Should not notify when merge detects no changes"
        );
    }
}
