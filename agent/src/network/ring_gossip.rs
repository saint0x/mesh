use crate::pki::{DeviceKeyPair, NodeId, PoolId};
use anyhow::{anyhow, Result};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};

/// Ring gossip message for P2P ring formation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingGossipMessage {
    /// Pool this ring belongs to
    pub pool_id: PoolId,
    /// Current ring state
    pub ring_state: RingState,
    /// Sender's node ID
    pub sender_node_id: NodeId,
    /// Lamport timestamp for conflict resolution
    pub version: u64,
    /// Signature by device private key (hex encoded for serde)
    #[serde(serialize_with = "hex_64_serialize", deserialize_with = "hex_64_deserialize")]
    pub signature: [u8; 64],
}

fn hex_64_serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&hex::encode(bytes))
}

fn hex_64_deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let s = String::deserialize(deserializer)?;
    let bytes = hex::decode(&s).map_err(D::Error::custom)?;
    if bytes.len() != 64 {
        return Err(D::Error::custom(format!("expected 64 bytes, got {}", bytes.len())));
    }
    let mut array = [0u8; 64];
    array.copy_from_slice(&bytes);
    Ok(array)
}

impl RingGossipMessage {
    /// Create a new ring gossip message
    pub fn new(
        pool_id: PoolId,
        ring_state: RingState,
        device_keypair: &DeviceKeyPair,
    ) -> Self {
        let sender_node_id = device_keypair.node_id();
        let version = ring_state.version;

        // Sign the message
        let mut payload = Vec::new();
        payload.extend_from_slice(pool_id.as_bytes());
        payload.extend_from_slice(&version.to_le_bytes());

        let signature = device_keypair.sign(&payload);

        Self {
            pool_id,
            ring_state,
            sender_node_id,
            version,
            signature,
        }
    }

    /// Verify gossip message signature (Phase 2 - TODO)
    pub fn verify(&self, _sender_device_pubkey: &[u8; 32]) -> bool {
        // TODO: Verify signature in Phase 2
        true
    }
}

/// Ring state for P2P ring formation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingState {
    /// Pool this ring belongs to
    pub pool_id: PoolId,
    /// Ring members sorted by hash(node_id)
    pub members: BTreeMap<u64, RingMember>,
    /// Lamport timestamp
    pub version: u64,
}

/// Ring member information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingMember {
    /// Node ID (cryptographic identity)
    pub node_id: NodeId,
    /// Libp2p PeerId for P2P connections (hex encoded)
    #[serde(with = "peer_id_serde")]
    pub peer_id: Option<PeerId>,
    /// LAN address (if discovered via beacon)
    pub lan_addr: Option<SocketAddr>,
    /// Last seen timestamp (Unix seconds)
    pub last_seen: u64,
    /// Member status
    pub status: MemberStatus,
}

/// Custom serde for Option<PeerId>
mod peer_id_serde {
    use libp2p::PeerId;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(peer_id: &Option<PeerId>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match peer_id {
            Some(id) => serializer.serialize_str(&id.to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<PeerId>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let opt_str: Option<String> = Option::deserialize(deserializer)?;
        match opt_str {
            Some(s) => {
                let peer_id = s.parse::<PeerId>().map_err(D::Error::custom)?;
                Ok(Some(peer_id))
            }
            None => Ok(None),
        }
    }
}

/// Member status in ring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberStatus {
    /// Announced presence, not yet stable
    Joining,
    /// Stable member participating in ring
    Active,
    /// Graceful shutdown announced
    Leaving,
    /// Detected as offline by gossip timeout
    Failed,
}

impl RingState {
    /// Create a new empty ring state
    pub fn new(pool_id: PoolId) -> Self {
        Self {
            pool_id,
            members: BTreeMap::new(),
            version: 0,
        }
    }

    /// Add or update a member
    pub fn add_member(&mut self, member: RingMember) {
        let hash = Self::hash_node_id(&member.node_id);
        self.members.insert(hash, member);
        self.version += 1;
    }

    /// Remove a member
    pub fn remove_member(&mut self, node_id: &NodeId) {
        let hash = Self::hash_node_id(node_id);
        self.members.remove(&hash);
        self.version += 1;
    }

    /// Update member status
    pub fn update_member_status(&mut self, node_id: &NodeId, status: MemberStatus) -> bool {
        let hash = Self::hash_node_id(node_id);
        if let Some(member) = self.members.get_mut(&hash) {
            member.status = status;
            member.last_seen = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            self.version += 1;
            true
        } else {
            false
        }
    }

    /// Get ring position for a node
    pub fn get_ring_position(&self, node_id: &NodeId) -> Option<usize> {
        let hash = Self::hash_node_id(node_id);
        self.members.keys().position(|&k| k == hash)
    }

    /// Get left and right neighbors in ring
    pub fn get_neighbors(&self, node_id: &NodeId) -> Option<(NodeId, NodeId)> {
        let position = self.get_ring_position(node_id)?;
        let members: Vec<_> = self.members.values().collect();
        let n = members.len();

        if n < 2 {
            return None; // Need at least 2 nodes for neighbors
        }

        let left = members[(position + n - 1) % n].node_id;
        let right = members[(position + 1) % n].node_id;

        Some((left, right))
    }

    /// Calculate shard range for a node (8192 columns divided evenly)
    pub fn calculate_shard_range(&self, node_id: &NodeId) -> Option<(u32, u32)> {
        let position = self.get_ring_position(node_id)?;
        let n = self.members.len() as u32;

        const TOTAL_COLUMNS: u32 = 8192;
        let columns_per_shard = TOTAL_COLUMNS / n;
        let start = position as u32 * columns_per_shard;
        let end = if position as u32 == n - 1 {
            TOTAL_COLUMNS // Last shard gets remainder
        } else {
            start + columns_per_shard
        };

        Some((start, end))
    }

    /// Merge another ring state (last-write-wins based on last_seen)
    pub fn merge(&mut self, other: &RingState) -> bool {
        if other.pool_id != self.pool_id {
            return false;
        }

        let mut changed = false;

        // Merge members (last-write-wins based on last_seen)
        for (hash, member) in &other.members {
            match self.members.get(hash) {
                Some(existing) if existing.last_seen < member.last_seen => {
                    self.members.insert(*hash, member.clone());
                    changed = true;
                }
                None => {
                    self.members.insert(*hash, member.clone());
                    changed = true;
                }
                _ => {}
            }
        }

        // Remove stale members (timeout: 60 seconds)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.members.retain(|_, member| {
            let keep = now - member.last_seen < 60;
            if !keep {
                changed = true;
            }
            keep
        });

        if changed {
            self.version = std::cmp::max(self.version, other.version) + 1;
        }

        changed
    }

    /// Check if ring is stable (all members Active)
    pub fn is_stable(&self) -> bool {
        self.members.values().all(|m| m.status == MemberStatus::Active)
    }

    /// Get count of active members
    pub fn active_member_count(&self) -> usize {
        self.members.values().filter(|m| m.status == MemberStatus::Active).count()
    }

    /// Hash node ID for consistent ring ordering
    fn hash_node_id(node_id: &NodeId) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        node_id.as_bytes().hash(&mut hasher);
        hasher.finish()
    }
}

/// Ring topology snapshot for InferenceCoordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingTopology {
    /// Pool ID
    pub pool_id: PoolId,
    /// My position in ring (0-indexed)
    pub my_position: u32,
    /// Left neighbor node ID
    pub left_neighbor: NodeId,
    /// Right neighbor node ID
    pub right_neighbor: NodeId,
    /// My shard column range [start, end)
    pub shard_start: u32,
    pub shard_end: u32,
    /// Total ring size
    pub ring_size: usize,
    /// All ring members (for logging/debugging)
    pub members: Vec<RingMember>,
}

impl RingTopology {
    /// Create from ring state
    pub fn from_ring_state(ring_state: &RingState, my_node_id: &NodeId) -> Result<Self> {
        let position = ring_state
            .get_ring_position(my_node_id)
            .ok_or_else(|| anyhow!("Node not in ring"))?;

        let (left, right) = ring_state
            .get_neighbors(my_node_id)
            .ok_or_else(|| anyhow!("Failed to get neighbors"))?;

        let (shard_start, shard_end) = ring_state
            .calculate_shard_range(my_node_id)
            .ok_or_else(|| anyhow!("Failed to calculate shard range"))?;

        Ok(Self {
            pool_id: ring_state.pool_id,
            my_position: position as u32,
            left_neighbor: left,
            right_neighbor: right,
            shard_start,
            shard_end,
            ring_size: ring_state.members.len(),
            members: ring_state.members.values().cloned().collect(),
        })
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::pki::DeviceKeyPair;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Create a test pool ID with a specific seed
    pub fn create_test_pool_id() -> PoolId {
        PoolId::from_bytes([1u8; 32])
    }

    /// Create a test node with deterministic seed
    pub fn create_test_node(seed: u64) -> (DeviceKeyPair, NodeId) {
        // For testing, we'll use a simple approach - generate and cache
        // In real property-based tests, quickcheck will generate these
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        (device, node_id)
    }

    /// Create a test ring member with specified parameters
    pub fn create_test_member(
        node_id: NodeId,
        last_seen: u64,
        status: MemberStatus,
    ) -> RingMember {
        RingMember {
            node_id,
            peer_id: None,
            lan_addr: None,
            last_seen,
            status,
        }
    }

    /// Create a ring with N members for testing
    pub fn create_ring_with_members(
        pool_id: PoolId,
        count: usize,
        base_timestamp: u64,
    ) -> (RingState, Vec<NodeId>) {
        let mut ring = RingState::new(pool_id);
        let mut node_ids = Vec::new();

        for i in 0..count {
            let device = DeviceKeyPair::generate();
            let node_id = device.node_id();
            node_ids.push(node_id);

            // Use timestamps in the past to avoid overflow in merge's staleness check
            ring.add_member(create_test_member(
                node_id,
                base_timestamp.saturating_sub(i as u64),
                MemberStatus::Active,
            ));
        }

        (ring, node_ids)
    }

    /// Get current timestamp in Unix seconds
    pub fn timestamp_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Get timestamp with offset (positive = future, negative = past)
    pub fn timestamp_offset(offset_seconds: i64) -> u64 {
        let now = timestamp_now();
        if offset_seconds >= 0 {
            now + offset_seconds as u64
        } else {
            now - (-offset_seconds) as u64
        }
    }

    /// Assert that a ring has complete shard coverage with no overlaps
    pub fn assert_shard_coverage(ring: &RingState) -> bool {
        if ring.members.is_empty() {
            return true;
        }

        let mut ranges: Vec<(u32, u32)> = ring
            .members
            .values()
            .map(|m| ring.calculate_shard_range(&m.node_id).unwrap())
            .collect();

        ranges.sort_by_key(|r| r.0);

        // Check first starts at 0
        if ranges[0].0 != 0 {
            return false;
        }

        // Check last ends at 8192
        if ranges.last().unwrap().1 != 8192 {
            return false;
        }

        // Check no gaps or overlaps
        for i in 0..ranges.len() - 1 {
            if ranges[i].1 != ranges[i + 1].0 {
                return false;
            }
        }

        true
    }

    /// Assert that no shards overlap
    pub fn assert_no_shard_overlap(ring: &RingState) -> bool {
        if ring.members.len() < 2 {
            return true;
        }

        let mut ranges: Vec<(u32, u32)> = ring
            .members
            .values()
            .map(|m| ring.calculate_shard_range(&m.node_id).unwrap())
            .collect();

        ranges.sort_by_key(|r| r.0);

        for i in 0..ranges.len() - 1 {
            let (_, end_i) = ranges[i];
            let (start_j, _) = ranges[i + 1];
            if end_i > start_j {
                return false; // Overlap detected
            }
        }

        true
    }

    /// Find a member by NodeId in the ring
    pub fn find_member<'a>(ring: &'a RingState, node_id: &NodeId) -> Option<&'a RingMember> {
        ring.members.values().find(|m| m.node_id == *node_id)
    }

    /// Check if ring contains a member with given NodeId
    pub fn contains_member(ring: &RingState, node_id: &NodeId) -> bool {
        ring.members.values().any(|m| m.node_id == *node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_utils::*;
    use crate::pki::DeviceKeyPair;

    #[test]
    fn test_ring_state_add_remove() {
        let pool_id = PoolId::from_bytes([1u8; 32]);
        let mut ring_state = RingState::new(pool_id);

        let device1 = DeviceKeyPair::generate();
        let node1 = device1.node_id();

        let member = RingMember {
            node_id: node1,
            peer_id: None,
            lan_addr: None,
            last_seen: 1000,
            status: MemberStatus::Active,
        };

        ring_state.add_member(member);
        assert_eq!(ring_state.members.len(), 1);
        assert_eq!(ring_state.version, 1);

        ring_state.remove_member(&node1);
        assert_eq!(ring_state.members.len(), 0);
        assert_eq!(ring_state.version, 2);
    }

    #[test]
    fn test_ring_neighbors() {
        let pool_id = PoolId::from_bytes([1u8; 32]);
        let mut ring_state = RingState::new(pool_id);

        let device1 = DeviceKeyPair::generate();
        let device2 = DeviceKeyPair::generate();
        let device3 = DeviceKeyPair::generate();

        let node1 = device1.node_id();
        let node2 = device2.node_id();
        let node3 = device3.node_id();

        for node_id in [node1, node2, node3] {
            ring_state.add_member(RingMember {
                node_id,
                peer_id: None,
                lan_addr: None,
                last_seen: 1000,
                status: MemberStatus::Active,
            });
        }

        // Each node should have 2 neighbors
        let (left, right) = ring_state.get_neighbors(&node1).unwrap();
        assert_ne!(left, node1);
        assert_ne!(right, node1);
        assert_ne!(left, right);
    }

    #[test]
    fn test_shard_calculation() {
        let pool_id = PoolId::from_bytes([1u8; 32]);
        let mut ring_state = RingState::new(pool_id);

        // Add 4 nodes
        for i in 0..4 {
            let device = DeviceKeyPair::generate();
            ring_state.add_member(RingMember {
                node_id: device.node_id(),
                peer_id: None,
                lan_addr: None,
                last_seen: 1000 + i,
                status: MemberStatus::Active,
            });
        }

        // Check shard ranges
        let mut total_columns = 0;
        for member in ring_state.members.values() {
            let (start, end) = ring_state.calculate_shard_range(&member.node_id).unwrap();
            total_columns += end - start;
        }

        assert_eq!(total_columns, 8192, "Total columns should be 8192");
    }

    #[test]
    fn test_ring_merge() {
        let pool_id = PoolId::from_bytes([1u8; 32]);
        let mut ring1 = RingState::new(pool_id);
        let mut ring2 = RingState::new(pool_id);

        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();

        // Use current timestamps to avoid stale member removal
        let now = timestamp_now();

        // Ring 1 has member with older timestamp
        ring1.add_member(RingMember {
            node_id,
            peer_id: None,
            lan_addr: None,
            last_seen: now - 10,
            status: MemberStatus::Active,
        });

        // Ring 2 has same member with newer timestamp
        ring2.add_member(RingMember {
            node_id,
            peer_id: None,
            lan_addr: None,
            last_seen: now,
            status: MemberStatus::Active,
        });

        // Merge ring2 into ring1
        let changed = ring1.merge(&ring2);
        assert!(changed);

        // Should have newer timestamp
        let member = ring1.members.values().next().unwrap();
        assert_eq!(member.last_seen, now);
    }

    // ========================================================================
    // PHASE 1: Core Invariants (Tests 1-12)
    // ========================================================================

    // Group 1.1: Hash & Ordering (3 tests)

    #[test]
    fn test_hash_node_id_determinism() {
        // Create 5 different NodeIds
        let devices: Vec<_> = (0..5).map(|_| DeviceKeyPair::generate()).collect();

        for device in &devices {
            let node_id = device.node_id();

            // Hash the same node_id 100 times
            let first_hash = RingState::hash_node_id(&node_id);
            for _ in 0..100 {
                let hash = RingState::hash_node_id(&node_id);
                assert_eq!(
                    hash, first_hash,
                    "Hash should be deterministic for same node_id"
                );
            }
        }
    }

    #[test]
    fn test_hash_node_id_collision_resistance() {
        use std::collections::HashSet;

        // Generate 1000 random NodeIds and hash them
        let mut hashes = HashSet::new();

        for _ in 0..1000 {
            let device = DeviceKeyPair::generate();
            let node_id = device.node_id();
            let hash = RingState::hash_node_id(&node_id);
            hashes.insert(hash);
        }

        // All hashes should be unique (no collisions)
        assert_eq!(
            hashes.len(),
            1000,
            "Expected 1000 unique hashes, got {} (collisions detected)",
            hashes.len()
        );
    }

    #[test]
    fn test_ring_ordering_stability() {
        let pool_id = create_test_pool_id();
        // Use current timestamp to avoid stale member removal
        let now = timestamp_now();
        let (mut ring, node_ids) = create_ring_with_members(pool_id, 5, now);

        // Get initial member order
        let initial_order: Vec<_> = ring.members.keys().copied().collect();

        // Create a second ring with the SAME members (same node_ids)
        let mut other_ring = RingState::new(pool_id);
        for node_id in &node_ids {
            other_ring.add_member(create_test_member(*node_id, now, MemberStatus::Active));
        }

        // Merge the same ring 10 times - order should never change
        for i in 0..10 {
            ring.merge(&other_ring);

            let current_order: Vec<_> = ring.members.keys().copied().collect();
            assert_eq!(
                current_order, initial_order,
                "Member order changed after merge #{}", i + 1
            );
        }

        // Verify all original members still present
        assert_eq!(ring.members.len(), 5);
        for node_id in &node_ids {
            assert!(
                ring.members.values().any(|m| &m.node_id == node_id),
                "Node {:?} missing after merges",
                node_id
            );
        }
    }

    // Group 1.2: Shard Range Edge Cases (7 tests)

    #[test]
    fn test_shard_range_single_node() {
        let pool_id = create_test_pool_id();
        let (ring, node_ids) = create_ring_with_members(pool_id, 1, 1000);

        let (start, end) = ring.calculate_shard_range(&node_ids[0]).unwrap();
        assert_eq!(start, 0, "Single node should start at column 0");
        assert_eq!(end, 8192, "Single node should end at column 8192");
        assert_eq!(end - start, 8192, "Single node should get all 8192 columns");
    }

    #[test]
    fn test_shard_range_two_nodes() {
        let pool_id = create_test_pool_id();
        let (ring, node_ids) = create_ring_with_members(pool_id, 2, 1000);

        // Get shard ranges for both nodes
        let mut ranges: Vec<_> = node_ids
            .iter()
            .map(|node_id| ring.calculate_shard_range(node_id).unwrap())
            .collect();

        ranges.sort_by_key(|r| r.0);

        // First node: 0-4096
        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges[0].1, 4096);

        // Second node: 4096-8192
        assert_eq!(ranges[1].0, 4096);
        assert_eq!(ranges[1].1, 8192);

        // Verify full coverage
        assert!(assert_shard_coverage(&ring), "Two nodes should have full coverage");
        assert!(assert_no_shard_overlap(&ring), "Two nodes should not overlap");
    }

    #[test]
    fn test_shard_range_three_nodes() {
        let pool_id = create_test_pool_id();
        let (ring, node_ids) = create_ring_with_members(pool_id, 3, 1000);

        // 8192 / 3 = 2730 remainder 2
        // Last shard gets the remainder
        let mut ranges: Vec<_> = node_ids
            .iter()
            .map(|node_id| ring.calculate_shard_range(node_id).unwrap())
            .collect();

        ranges.sort_by_key(|r| r.0);

        // Calculate expected distribution
        let columns_per_shard = 8192 / 3; // 2730

        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges[0].1, columns_per_shard);

        assert_eq!(ranges[1].0, columns_per_shard);
        assert_eq!(ranges[1].1, columns_per_shard * 2);

        assert_eq!(ranges[2].0, columns_per_shard * 2);
        assert_eq!(ranges[2].1, 8192); // Last gets remainder

        // Verify total coverage
        let total: u32 = ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total, 8192, "Three nodes should cover all 8192 columns");

        assert!(assert_shard_coverage(&ring), "Three nodes should have full coverage");
    }

    #[test]
    fn test_shard_range_prime_nodes() {
        // Test with prime numbers 5 and 7
        for n in [5, 7] {
            let pool_id = create_test_pool_id();
            let (ring, node_ids) = create_ring_with_members(pool_id, n, 1000);

            let mut ranges: Vec<_> = node_ids
                .iter()
                .map(|node_id| ring.calculate_shard_range(node_id).unwrap())
                .collect();

            ranges.sort_by_key(|r| r.0);

            // Verify invariants
            assert_eq!(ranges[0].0, 0, "First shard should start at 0");
            assert_eq!(ranges.last().unwrap().1, 8192, "Last shard should end at 8192");

            // Verify full coverage and no overlaps
            assert!(
                assert_shard_coverage(&ring),
                "Ring with {} nodes should have full coverage",
                n
            );
            assert!(
                assert_no_shard_overlap(&ring),
                "Ring with {} nodes should have no overlaps",
                n
            );

            // Verify total
            let total: u32 = ranges.iter().map(|(s, e)| e - s).sum();
            assert_eq!(total, 8192, "Ring with {} nodes should sum to 8192", n);
        }
    }

    #[test]
    fn test_shard_range_exact_divisor() {
        // Test with n=8192 (each node gets exactly 1 column)
        let pool_id = create_test_pool_id();
        let mut ring = RingState::new(pool_id);

        // Create 8192 nodes (lightweight test - just checking calculation)
        for i in 0..8192 {
            let device = DeviceKeyPair::generate();
            let node_id = device.node_id();

            ring.add_member(create_test_member(node_id, 1000 + i, MemberStatus::Active));
        }

        // Get nodes by their actual ring position
        let sorted_members: Vec<_> = ring.members.values().collect();

        // Sample positions: 0, 4095, 8191
        for (pos, expected_start) in [(0, 0), (4095, 4095), (8191, 8191)] {
            let node_id = &sorted_members[pos].node_id;
            let (start, end) = ring.calculate_shard_range(node_id).unwrap();
            assert_eq!(start, expected_start, "Position {} should start at {}", pos, expected_start);
            assert_eq!(end, expected_start + 1, "Position {} should end at {}", pos, expected_start + 1);
            assert_eq!(end - start, 1, "Position {} should have exactly 1 column", pos);
        }

        // Verify full coverage
        assert!(assert_shard_coverage(&ring), "8192 nodes should have full coverage");
    }

    #[test]
    fn test_shard_range_overflow() {
        // Test with n=8193 (more nodes than columns)
        let pool_id = create_test_pool_id();
        let mut ring = RingState::new(pool_id);

        for i in 0..8193 {
            let device = DeviceKeyPair::generate();
            let node_id = device.node_id();

            ring.add_member(create_test_member(node_id, 1000 + i, MemberStatus::Active));
        }

        // 8192 / 8193 = 0 columns per shard
        // Last shard gets all remaining columns (8192)

        // Get nodes by their actual ring position
        let sorted_members: Vec<_> = ring.members.values().collect();

        // Sample positions
        let (start_0, end_0) = ring.calculate_shard_range(&sorted_members[0].node_id).unwrap();
        assert_eq!(start_0, 0);
        assert_eq!(end_0, 0, "With overflow, most shards have 0 columns");

        let (start_4096, end_4096) = ring.calculate_shard_range(&sorted_members[4096].node_id).unwrap();
        assert_eq!(start_4096, 0);
        assert_eq!(end_4096, 0);

        // Last position gets all columns
        let (start_last, end_last) = ring.calculate_shard_range(&sorted_members[8192].node_id).unwrap();
        assert_eq!(start_last, 0);
        assert_eq!(end_last, 8192, "Last shard with overflow should get all remaining columns");

        // Verify no panic and deterministic behavior
        assert!(assert_shard_coverage(&ring), "Overflow case should still have valid coverage");
    }

    #[test]
    fn test_shard_range_property_based() {
        use quickcheck::{quickcheck, TestResult};

        fn prop_shard_coverage(n: u32) -> TestResult {
            // Discard invalid inputs
            if n == 0 || n > 10000 {
                return TestResult::discard();
            }

            let pool_id = PoolId::from_bytes([1u8; 32]);
            let mut ring = RingState::new(pool_id);

            // Create n members
            for i in 0..n {
                let device = DeviceKeyPair::generate();
                ring.add_member(RingMember {
                    node_id: device.node_id(),
                    peer_id: None,
                    lan_addr: None,
                    last_seen: 1000 + i as u64,
                    status: MemberStatus::Active,
                });
            }

            // Collect all shard ranges
            let mut ranges: Vec<(u32, u32)> = ring
                .members
                .values()
                .map(|m| ring.calculate_shard_range(&m.node_id).unwrap())
                .collect();

            ranges.sort_by_key(|r| r.0);

            // Verify invariants
            let first_starts_at_zero = ranges[0].0 == 0;
            let last_ends_at_8192 = ranges.last().unwrap().1 == 8192;

            // Check no gaps or overlaps
            let mut no_gaps_or_overlaps = true;
            for i in 0..ranges.len() - 1 {
                if ranges[i].1 != ranges[i + 1].0 {
                    no_gaps_or_overlaps = false;
                    break;
                }
            }

            // Total coverage
            let total: u32 = ranges.iter().map(|(s, e)| e - s).sum();
            let full_coverage = total == 8192;

            TestResult::from_bool(
                first_starts_at_zero
                    && last_ends_at_8192
                    && no_gaps_or_overlaps
                    && full_coverage
            )
        }

        quickcheck(prop_shard_coverage as fn(u32) -> TestResult);
    }

    // Group 1.3: Version & Serialization (2 tests)

    #[test]
    fn test_merge_version_monotonicity() {
        let pool_id = create_test_pool_id();

        let mut ring1 = RingState::new(pool_id);
        ring1.version = 5;

        let mut ring2 = RingState::new(pool_id);
        ring2.version = 3;

        // Add a new member to ring2 (forces merge to succeed)
        let device = DeviceKeyPair::generate();
        ring2.add_member(create_test_member(device.node_id(), 2000, MemberStatus::Active));

        // Merge ring2 into ring1
        let changed = ring1.merge(&ring2);

        assert!(changed, "Merge should succeed with new member");
        assert_eq!(
            ring1.version, 6,
            "Version should be max(5, 3) + 1 = 6"
        );
    }

    #[test]
    fn test_ring_gossip_message_serde_roundtrip() {
        let pool_id = create_test_pool_id();
        let (ring_state, _) = create_ring_with_members(pool_id, 3, 1000);
        let device = DeviceKeyPair::generate();

        // Create RingGossipMessage
        let original = RingGossipMessage::new(pool_id, ring_state.clone(), &device);

        // Serialize to CBOR
        let mut serialized = Vec::new();
        ciborium::ser::into_writer(&original, &mut serialized).unwrap();

        // Deserialize from CBOR
        let deserialized: RingGossipMessage =
            ciborium::de::from_reader(&serialized[..]).unwrap();

        // Verify fields match
        assert_eq!(deserialized.pool_id, original.pool_id);
        assert_eq!(deserialized.sender_node_id, original.sender_node_id);
        assert_eq!(deserialized.version, original.version);
        assert_eq!(deserialized.signature, original.signature);
        assert_eq!(deserialized.ring_state.members.len(), original.ring_state.members.len());
    }

    // =========================================================================
    // Phase 2: Merge Semantics Tests (13-22)
    // =========================================================================

    /// Test 13: Merge with newer member info should update (last-write-wins)
    #[test]
    fn test_merge_last_write_wins_newer() {
        let pool_id = create_test_pool_id();
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();

        let now = timestamp_now();
        let mut ring1 = RingState::new(pool_id);
        let mut ring2 = RingState::new(pool_id);

        // Ring1: node_A at t=(now-30), status=Joining
        ring1.add_member(create_test_member(node_id, now - 30, MemberStatus::Joining));

        // Ring2: node_A at t=(now-10), status=Active (newer)
        ring2.add_member(create_test_member(node_id, now - 10, MemberStatus::Active));

        let initial_version = ring1.version;

        // Merge ring2 into ring1
        let changed = ring1.merge(&ring2);

        // Should detect change
        assert!(changed, "Merge should detect newer member info");

        // Should use newer info (last-write-wins)
        let member = find_member(&ring1, &node_id).unwrap();
        assert_eq!(member.last_seen, now - 10, "Should use newer last_seen");
        assert_eq!(member.status, MemberStatus::Active, "Should use newer status");

        // Version should increment
        assert_eq!(ring1.version, initial_version + 1);
    }

    /// Test 14: Merge with older member info should be ignored
    #[test]
    fn test_merge_last_write_wins_older() {
        let pool_id = create_test_pool_id();
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();

        let now = timestamp_now();
        let mut ring1 = RingState::new(pool_id);
        let mut ring2 = RingState::new(pool_id);

        // Ring1: node_A at t=(now-10), status=Active
        ring1.add_member(create_test_member(node_id, now - 10, MemberStatus::Active));

        // Ring2: node_A at t=(now-30), status=Joining (older)
        ring2.add_member(create_test_member(node_id, now - 30, MemberStatus::Joining));

        let initial_version = ring1.version;

        // Merge ring2 into ring1
        let changed = ring1.merge(&ring2);

        // Should not detect change (older info ignored)
        assert!(!changed, "Merge should ignore older member info");

        // Should keep newer info
        let member = find_member(&ring1, &node_id).unwrap();
        assert_eq!(member.last_seen, now - 10, "Should keep newer last_seen");
        assert_eq!(member.status, MemberStatus::Active, "Should keep newer status");

        // Version should not change
        assert_eq!(ring1.version, initial_version);
    }

    /// Test 15: Merge with different pool IDs should be rejected
    #[test]
    fn test_merge_different_pool_ids() {
        let pool_a = PoolId::from_bytes([1u8; 32]);
        let pool_b = PoolId::from_bytes([2u8; 32]);

        let now = timestamp_now();
        let (mut ring1, _) = create_ring_with_members(pool_a, 3, now);
        let (ring2, _) = create_ring_with_members(pool_b, 3, now);

        let initial_version = ring1.version;
        let initial_count = ring1.members.len();

        // Merge should fail for different pool IDs
        let changed = ring1.merge(&ring2);

        assert!(!changed, "Merge should reject different pool IDs");
        assert_eq!(ring1.members.len(), initial_count, "Ring1 should be unchanged");
        assert_eq!(ring1.version, initial_version, "Version should not change");
    }

    /// Test 16: Stale members (>60s old) should be removed during merge
    #[test]
    fn test_merge_stale_member_timeout() {
        let pool_id = create_test_pool_id();
        let device_a = DeviceKeyPair::generate();
        let device_b = DeviceKeyPair::generate();
        let node_a = device_a.node_id();
        let node_b = device_b.node_id();

        let now = timestamp_now();
        let mut ring = RingState::new(pool_id);

        // node_A: 70 seconds old (stale)
        ring.add_member(create_test_member(node_a, now - 70, MemberStatus::Active));

        // node_B: 10 seconds old (fresh)
        ring.add_member(create_test_member(node_b, now - 10, MemberStatus::Active));

        assert_eq!(ring.members.len(), 2, "Should start with 2 members");

        // Merge with empty ring to trigger stale check
        let empty_ring = RingState::new(pool_id);
        let changed = ring.merge(&empty_ring);

        // Should detect stale member removal
        assert!(changed, "Merge should detect stale member removal");

        // node_A should be removed (>60s old)
        assert!(!contains_member(&ring, &node_a), "Stale member should be removed");

        // node_B should be retained (<60s old)
        assert!(contains_member(&ring, &node_b), "Fresh member should be retained");

        assert_eq!(ring.members.len(), 1, "Should have 1 member after cleanup");
    }

    /// Test 17: Member at exactly 59 seconds should NOT be removed (boundary test)
    #[test]
    fn test_merge_stale_boundary_59_seconds() {
        let pool_id = create_test_pool_id();
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();

        let now = timestamp_now();
        let mut ring = RingState::new(pool_id);

        // Member at exactly 59 seconds old (boundary - should be kept)
        ring.add_member(create_test_member(node_id, now - 59, MemberStatus::Active));

        assert_eq!(ring.members.len(), 1);

        // Merge with empty ring to trigger stale check
        let empty_ring = RingState::new(pool_id);
        let changed = ring.merge(&empty_ring);

        // Should NOT detect change (member still valid)
        assert!(!changed, "Merge should not remove member at 59s boundary");

        // Member should be retained
        assert!(contains_member(&ring, &node_id), "Member at 59s should be retained");
        assert_eq!(ring.members.len(), 1);
    }

    /// Test 18: Member at exactly 60 seconds should be removed (boundary test)
    #[test]
    fn test_merge_stale_boundary_60_seconds() {
        let pool_id = create_test_pool_id();
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();

        let now = timestamp_now();
        let mut ring = RingState::new(pool_id);

        // Member at exactly 60 seconds old (boundary - should be removed)
        ring.add_member(create_test_member(node_id, now - 60, MemberStatus::Active));

        assert_eq!(ring.members.len(), 1);

        // Merge with empty ring to trigger stale check
        let empty_ring = RingState::new(pool_id);
        let changed = ring.merge(&empty_ring);

        // Should detect stale member removal
        assert!(changed, "Merge should remove member at 60s boundary");

        // Member should be removed
        assert!(!contains_member(&ring, &node_id), "Member at 60s should be removed");
        assert_eq!(ring.members.len(), 0);
    }

    /// Test 19: Update member status should succeed for existing member
    #[test]
    fn test_update_member_status_success() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (mut ring, node_ids) = create_ring_with_members(pool_id, 3, now);

        let node_id = node_ids[0];
        let initial_version = ring.version;

        // Update status: Active → Joining
        let result = ring.update_member_status(&node_id, MemberStatus::Joining);

        assert!(result, "Update should succeed for existing member");

        // Verify status changed
        let member = find_member(&ring, &node_id).unwrap();
        assert_eq!(member.status, MemberStatus::Joining);

        // Version should increment
        assert_eq!(ring.version, initial_version + 1);
    }

    /// Test 20: Update status for non-existent member should fail
    #[test]
    fn test_update_member_status_nonexistent() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (mut ring, _) = create_ring_with_members(pool_id, 3, now);

        // Create a node that's not in the ring
        let nonexistent_device = DeviceKeyPair::generate();
        let nonexistent_node = nonexistent_device.node_id();

        let initial_version = ring.version;

        // Try to update non-existent member
        let result = ring.update_member_status(&nonexistent_node, MemberStatus::Joining);

        assert!(!result, "Update should fail for non-existent member");

        // Version should not change
        assert_eq!(ring.version, initial_version);
    }

    /// Test 21: is_stable() should return true when all members are Active
    #[test]
    fn test_is_stable_all_active() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (ring, _) = create_ring_with_members(pool_id, 5, now);

        // All members created with Active status
        assert!(ring.is_stable(), "Ring with all Active members should be stable");
    }

    /// Test 22: is_stable() should return false when any member is not Active
    #[test]
    fn test_is_stable_one_joining() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (mut ring, node_ids) = create_ring_with_members(pool_id, 5, now);

        // Change one member to Joining status
        ring.update_member_status(&node_ids[2], MemberStatus::Joining);

        assert!(!ring.is_stable(), "Ring with Joining member should not be stable");
    }

    // =========================================================================
    // Phase 4: Integration & Wraparound Tests (32-35)
    // =========================================================================

    /// Test 32: Neighbors at position 0 should wrap around to last position
    #[test]
    fn test_neighbors_wraparound_position_zero() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (ring, node_ids) = create_ring_with_members(pool_id, 5, now);

        // Find the node at position 0 (first in sorted order)
        let sorted_members: Vec<_> = ring.members.values().collect();
        let node_at_pos_0 = sorted_members[0].node_id;
        let node_at_pos_4 = sorted_members[4].node_id;
        let node_at_pos_1 = sorted_members[1].node_id;

        // Get neighbors for position 0
        let (left, right) = ring.get_neighbors(&node_at_pos_0).unwrap();

        // Left neighbor should be position 4 (wraps around)
        assert_eq!(left, node_at_pos_4, "Left neighbor of position 0 should be position 4");

        // Right neighbor should be position 1
        assert_eq!(right, node_at_pos_1, "Right neighbor of position 0 should be position 1");
    }

    /// Test 33: Neighbors at last position should wrap around to position 0
    #[test]
    fn test_neighbors_wraparound_last_position() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (ring, node_ids) = create_ring_with_members(pool_id, 7, now);

        // Find the node at last position (position 6 in 7-node ring)
        let sorted_members: Vec<_> = ring.members.values().collect();
        let node_at_pos_6 = sorted_members[6].node_id;
        let node_at_pos_5 = sorted_members[5].node_id;
        let node_at_pos_0 = sorted_members[0].node_id;

        // Get neighbors for last position
        let (left, right) = ring.get_neighbors(&node_at_pos_6).unwrap();

        // Left neighbor should be position 5
        assert_eq!(left, node_at_pos_5, "Left neighbor of position 6 should be position 5");

        // Right neighbor should be position 0 (wraps around)
        assert_eq!(right, node_at_pos_0, "Right neighbor of position 6 should wrap to position 0");
    }

    /// Test 34: Single node ring has no neighbors
    #[test]
    fn test_neighbors_single_node() {
        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (ring, node_ids) = create_ring_with_members(pool_id, 1, now);

        let node_id = node_ids[0];

        // Single node has no neighbors (need 2+ members)
        let result = ring.get_neighbors(&node_id);
        assert!(result.is_none(), "Single node should have no neighbors");
    }

    /// Test 35: Neighbor relationships are symmetric (property-based)
    #[quickcheck_macros::quickcheck]
    fn test_neighbors_symmetry_property(n: u8) -> quickcheck::TestResult {
        // Valid range: 2-100 nodes
        if n < 2 || n > 100 {
            return quickcheck::TestResult::discard();
        }

        let pool_id = create_test_pool_id();
        let now = timestamp_now();
        let (ring, node_ids) = create_ring_with_members(pool_id, n as usize, now);

        // Verify symmetry: If A's right neighbor is B, then B's left neighbor is A
        let sorted_members: Vec<_> = ring.members.values().collect();

        for i in 0..sorted_members.len() {
            let node_a = sorted_members[i].node_id;
            let (left_of_a, right_of_a) = match ring.get_neighbors(&node_a) {
                Some(neighbors) => neighbors,
                None => return quickcheck::TestResult::error("get_neighbors failed"),
            };

            // Check right neighbor symmetry
            let (left_of_right, _) = match ring.get_neighbors(&right_of_a) {
                Some(neighbors) => neighbors,
                None => return quickcheck::TestResult::error("get_neighbors failed for right neighbor"),
            };

            if left_of_right != node_a {
                return quickcheck::TestResult::failed();
            }

            // Check left neighbor symmetry
            let (_, right_of_left) = match ring.get_neighbors(&left_of_a) {
                Some(neighbors) => neighbors,
                None => return quickcheck::TestResult::error("get_neighbors failed for left neighbor"),
            };

            if right_of_left != node_a {
                return quickcheck::TestResult::failed();
            }
        }

        quickcheck::TestResult::passed()
    }

    /// Test 36: RingMember with Option<PeerId> serializes correctly
    #[test]
    fn test_ring_member_peer_id_serde() {
        let device = DeviceKeyPair::generate();
        let node_id = device.node_id();
        let peer_id = PeerId::random();

        // Member with Some(PeerId)
        let member_with_peer = RingMember {
            node_id,
            peer_id: Some(peer_id),
            lan_addr: None,
            last_seen: 1000,
            status: MemberStatus::Active,
        };

        // Member with None
        let member_without_peer = RingMember {
            node_id,
            peer_id: None,
            lan_addr: None,
            last_seen: 1000,
            status: MemberStatus::Active,
        };

        // Serialize/deserialize member with PeerId
        let json_with = serde_json::to_string(&member_with_peer).unwrap();
        let decoded_with: RingMember = serde_json::from_str(&json_with).unwrap();
        assert_eq!(decoded_with.peer_id, Some(peer_id));
        assert_eq!(decoded_with.node_id, node_id);

        // Serialize/deserialize member without PeerId
        let json_without = serde_json::to_string(&member_without_peer).unwrap();
        let decoded_without: RingMember = serde_json::from_str(&json_without).unwrap();
        assert_eq!(decoded_without.peer_id, None);
        assert_eq!(decoded_without.node_id, node_id);
    }
}
