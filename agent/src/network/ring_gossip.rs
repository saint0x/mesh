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
mod tests {
    use super::*;
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

        // Ring 1 has member with last_seen = 1000
        ring1.add_member(RingMember {
            node_id,
            peer_id: None,
            lan_addr: None,
            last_seen: 1000,
            status: MemberStatus::Active,
        });

        // Ring 2 has same member with last_seen = 2000
        ring2.add_member(RingMember {
            node_id,
            peer_id: None,
            lan_addr: None,
            last_seen: 2000,
            status: MemberStatus::Active,
        });

        // Merge ring2 into ring1
        let changed = ring1.merge(&ring2);
        assert!(changed);

        // Should have newer timestamp
        let member = ring1.members.values().next().unwrap();
        assert_eq!(member.last_seen, 2000);
    }
}
