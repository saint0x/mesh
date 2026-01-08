use crate::pki::{
    DeviceKeyPair, DiscoveredPeer, DiscoveryMethod, NodeId, PeerCache, PoolConfig, PoolId,
    PoolMembershipCert,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, Duration};

/// Multicast settings for LAN discovery
pub const BEACON_MULTICAST_ADDR: &str = "239.192.0.1";
pub const BEACON_MULTICAST_PORT: u16 = 42424;
pub const BEACON_INTERVAL_SECS: u64 = 5;
pub const STALE_PEER_THRESHOLD_SECS: u64 = 30;

/// Pool beacon packet (UDP multicast)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolBeacon {
    /// Protocol version (v1)
    pub version: u8,
    /// Pool identifier
    pub pool_id: PoolId,
    /// Node identifier
    pub node_id: NodeId,
    /// QUIC listener port (future use)
    pub quic_port: u16,
    /// First 8 bytes of device pubkey (fingerprint)
    pub pubkey_fingerprint: [u8; 8],
    /// Hash of membership cert
    pub membership_cert_hash: [u8; 32],
    /// Capability bitmap (future use)
    pub capabilities_bitmap: u32,
    /// Unix timestamp
    pub timestamp: u64,
    /// Signature by device private key (v2+, hex encoded for serde)
    #[serde(with = "hex_array")]
    pub signature: [u8; 64],
}

mod hex_array {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = hex::encode(bytes);
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
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
}

impl PoolBeacon {
    /// Create a new beacon
    pub fn new(
        pool_config: &PoolConfig,
        membership_cert: &PoolMembershipCert,
        device_keypair: &DeviceKeyPair,
        quic_port: u16,
    ) -> Self {
        let node_id = device_keypair.node_id();

        // Compute cert hash
        let cert_hash = membership_cert.hash();

        // Get pubkey fingerprint
        let mut fingerprint = [0u8; 8];
        fingerprint.copy_from_slice(&device_keypair.public[..8]);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Construct signing payload
        let mut payload = Vec::new();
        payload.push(1); // version
        payload.extend_from_slice(pool_config.pool_id.as_bytes());
        payload.extend_from_slice(node_id.as_bytes());
        payload.extend_from_slice(&quic_port.to_le_bytes());
        payload.extend_from_slice(&fingerprint);
        payload.extend_from_slice(&cert_hash);
        payload.extend_from_slice(&0u32.to_le_bytes()); // capabilities
        payload.extend_from_slice(&timestamp.to_le_bytes());

        let signature = device_keypair.sign(&payload);

        PoolBeacon {
            version: 1,
            pool_id: pool_config.pool_id,
            node_id,
            quic_port,
            pubkey_fingerprint: fingerprint,
            membership_cert_hash: cert_hash,
            capabilities_bitmap: 0,
            timestamp,
            signature,
        }
    }

    /// Verify beacon signature (Phase 2)
    pub fn verify(&self, _device_pubkey: &[u8; 32]) -> bool {
        // TODO: Implement signature verification in Phase 2
        // For Phase 1, trust LAN beacons
        true
    }
}

/// Beacon broadcaster
pub struct BeaconBroadcaster {
    socket: Arc<UdpSocket>,
    pools: Vec<(PoolConfig, PoolMembershipCert)>,
    device_keypair: DeviceKeyPair,
    quic_port: u16,
}

impl BeaconBroadcaster {
    /// Create a new beacon broadcaster
    pub async fn new(
        pools: Vec<(PoolConfig, PoolMembershipCert)>,
        device_keypair: DeviceKeyPair,
        quic_port: u16,
    ) -> Result<Self> {
        // Create UDP socket
        let socket = UdpSocket::bind("0.0.0.0:0").await?;

        // Disable multicast loopback (don't receive our own beacons)
        socket.set_multicast_loop_v4(false)?;

        tracing::info!(
            pools = pools.len(),
            interval_secs = BEACON_INTERVAL_SECS,
            "Beacon broadcaster initialized"
        );

        Ok(BeaconBroadcaster {
            socket: Arc::new(socket),
            pools,
            device_keypair,
            quic_port,
        })
    }

    /// Run beacon broadcaster loop
    pub async fn run(self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(BEACON_INTERVAL_SECS));
        let target_addr: SocketAddr = format!("{}:{}", BEACON_MULTICAST_ADDR, BEACON_MULTICAST_PORT).parse()?;

        loop {
            interval.tick().await;

            for (pool_config, membership_cert) in &self.pools {
                // Create beacon
                let beacon = PoolBeacon::new(
                    pool_config,
                    membership_cert,
                    &self.device_keypair,
                    self.quic_port,
                );

                // Serialize to CBOR
                let mut packet = Vec::new();
                match ciborium::ser::into_writer(&beacon, &mut packet) {
                    Ok(_) => {
                        // Send beacon
                        match self.socket.send_to(&packet, target_addr).await {
                            Ok(sent) => {
                                tracing::trace!(
                                    pool_id = %pool_config.pool_id,
                                    node_id = %beacon.node_id,
                                    bytes = sent,
                                    "Beacon sent"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    pool_id = %pool_config.pool_id,
                                    error = %e,
                                    "Failed to send beacon"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            pool_id = %pool_config.pool_id,
                            error = %e,
                            "Failed to serialize beacon"
                        );
                    }
                }
            }
        }
    }
}

/// Beacon listener
pub struct BeaconListener {
    socket: Arc<UdpSocket>,
    my_pools: HashMap<PoolId, (PoolConfig, PoolMembershipCert)>,
    my_node_id: NodeId,
    discovered_tx: mpsc::Sender<DiscoveredPeer>,
    peer_cache: Arc<RwLock<HashMap<PoolId, PeerCache>>>,
}

impl BeaconListener {
    /// Create a new beacon listener
    pub async fn new(
        pools: Vec<(PoolConfig, PoolMembershipCert)>,
        device_keypair: &DeviceKeyPair,
    ) -> Result<(Self, mpsc::Receiver<DiscoveredPeer>)> {
        // Create UDP socket
        let socket = UdpSocket::bind(format!("0.0.0.0:{}", BEACON_MULTICAST_PORT)).await?;

        // Join multicast group
        socket.join_multicast_v4(
            BEACON_MULTICAST_ADDR.parse()?,
            Ipv4Addr::new(0, 0, 0, 0),
        )?;

        // Build pool map
        let mut my_pools = HashMap::new();
        let mut peer_cache_map = HashMap::new();

        for (config, cert) in pools {
            let pool_id = config.pool_id;
            my_pools.insert(pool_id, (config, cert));

            // Load peer cache
            match PeerCache::load(&pool_id) {
                Ok(cache) => {
                    peer_cache_map.insert(pool_id, cache);
                }
                Err(_) => {
                    peer_cache_map.insert(pool_id, PeerCache::default());
                }
            }
        }

        let my_node_id = device_keypair.node_id();
        let (discovered_tx, discovered_rx) = mpsc::channel(100);

        tracing::info!(
            pools = my_pools.len(),
            multicast_addr = BEACON_MULTICAST_ADDR,
            multicast_port = BEACON_MULTICAST_PORT,
            node_id = %my_node_id,
            "Beacon listener initialized"
        );

        Ok((
            BeaconListener {
                socket: Arc::new(socket),
                my_pools,
                my_node_id,
                discovered_tx,
                peer_cache: Arc::new(RwLock::new(peer_cache_map)),
            },
            discovered_rx,
        ))
    }

    /// Run beacon listener loop
    pub async fn run(mut self) -> Result<()> {
        let mut buf = vec![0u8; 2048];

        // Spawn peer cache persistence task
        let peer_cache = self.peer_cache.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                interval.tick().await;

                let cache_map = peer_cache.read().await;
                for (pool_id, cache) in cache_map.iter() {
                    if let Err(e) = cache.save(pool_id) {
                        tracing::warn!(pool_id = %pool_id, error = %e, "Failed to save peer cache");
                    }
                }
            }
        });

        loop {
            // Receive beacon
            match self.socket.recv_from(&mut buf).await {
                Ok((len, sender_addr)) => {
                    // Parse beacon
                    match ciborium::de::from_reader::<PoolBeacon, _>(&buf[..len]) {
                        Ok(beacon) => {
                            self.handle_beacon(beacon, sender_addr).await;
                        }
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                sender = %sender_addr,
                                "Failed to parse beacon"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to receive beacon");
                }
            }
        }
    }

    async fn handle_beacon(&mut self, beacon: PoolBeacon, sender_addr: SocketAddr) {
        // Ignore our own beacons
        if beacon.node_id == self.my_node_id {
            return;
        }

        // Check if we belong to this pool
        if !self.my_pools.contains_key(&beacon.pool_id) {
            tracing::trace!(
                pool_id = %beacon.pool_id,
                "Ignoring beacon from unknown pool"
            );
            return;
        }

        // Create discovered peer
        let lan_addr = format!("{}:{}", sender_addr.ip(), beacon.quic_port);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let peer = DiscoveredPeer {
            pool_id: beacon.pool_id,
            node_id: beacon.node_id,
            lan_addr: lan_addr.clone(),
            discovery_method: DiscoveryMethod::LAN,
            last_seen: now,
        };

        // Update peer cache
        let mut cache_map = self.peer_cache.write().await;
        if let Some(cache) = cache_map.get_mut(&beacon.pool_id) {
            cache.upsert_peer(peer.clone());
            cache.remove_stale(STALE_PEER_THRESHOLD_SECS);
        }

        // Send to discovered channel
        if let Err(e) = self.discovered_tx.try_send(peer) {
            tracing::debug!(
                pool_id = %beacon.pool_id,
                node_id = %beacon.node_id,
                error = %e,
                "Failed to send discovered peer to channel"
            );
        } else {
            tracing::info!(
                pool_id = %beacon.pool_id,
                node_id = %beacon.node_id,
                lan_addr = %lan_addr,
                "LAN peer discovered"
            );
        }
    }

    /// Get peer cache for a pool
    pub async fn get_peer_cache(&self, pool_id: &PoolId) -> Option<PeerCache> {
        let cache_map = self.peer_cache.read().await;
        cache_map.get(pool_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pki::PoolRootKeyPair;

    #[test]
    fn test_create_beacon() {
        let device = DeviceKeyPair::generate();
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        let cert = crate::pki::PoolMembershipCert::new(
            device.public,
            &pool_root,
            crate::pki::MembershipRole::Member,
            u64::MAX,
        );

        let pool_config = PoolConfig {
            pool_id,
            name: "Test Pool".to_string(),
            pool_root_pubkey: pool_root.public,
            beacon_config: crate::pki::BeaconConfig::default(),
            role: crate::pki::MembershipRole::Member,
            expires_at: u64::MAX,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, 4001);

        assert_eq!(beacon.version, 1);
        assert_eq!(beacon.pool_id, pool_id);
        assert_eq!(beacon.node_id, device.node_id());
        assert_eq!(beacon.quic_port, 4001);
    }

    #[test]
    fn test_beacon_serialization() {
        let device = DeviceKeyPair::generate();
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        let cert = crate::pki::PoolMembershipCert::new(
            device.public,
            &pool_root,
            crate::pki::MembershipRole::Member,
            u64::MAX,
        );

        let pool_config = PoolConfig {
            pool_id,
            name: "Test Pool".to_string(),
            pool_root_pubkey: pool_root.public,
            beacon_config: crate::pki::BeaconConfig::default(),
            role: crate::pki::MembershipRole::Member,
            expires_at: u64::MAX,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, 4001);

        // Serialize to CBOR
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&beacon, &mut bytes).unwrap();

        // Deserialize
        let decoded: PoolBeacon = ciborium::de::from_reader(&bytes[..]).unwrap();

        assert_eq!(decoded.pool_id, beacon.pool_id);
        assert_eq!(decoded.node_id, beacon.node_id);
        assert_eq!(decoded.quic_port, beacon.quic_port);
    }
}
