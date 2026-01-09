use crate::network::RingGossipMessage;
use crate::pki::{
    CertSigningRequest, DeviceKeyPair, DiscoveredPeer, DiscoveryMethod, NodeId, PeerCache,
    PoolConfig, PoolId, PoolMembershipCert,
};
use anyhow::Result;
use libp2p::PeerId;
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

/// Capability flags for beacon capabilities_bitmap
pub const CAPABILITY_CAN_SIGN_CERTS: u32 = 0x0001;  // Admin can sign member certificates
pub const CAPABILITY_ACCEPTING_JOINS: u32 = 0x0002;  // Admin is accepting new pool joins
pub const CAPABILITY_HAS_RELAY: u32 = 0x0004;        // Node has relay server running
pub const CAPABILITY_HAS_CONTROL_PLANE: u32 = 0x0008; // Node has control plane running

/// Beacon message types for LAN discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeaconMessage {
    /// Regular pool presence announcement
    PoolBeacon(PoolBeacon),
    /// Certificate signing request from new member
    CertRequest(CertSigningRequest),
    /// Signed certificate response from admin
    CertResponse(PoolMembershipCert),
    /// Ring topology gossip
    RingGossip(RingGossipMessage),
}

/// Pool beacon packet (UDP multicast)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolBeacon {
    /// Protocol version (v1)
    pub version: u8,
    /// Pool identifier
    pub pool_id: PoolId,
    /// Node identifier
    pub node_id: NodeId,
    /// Libp2p PeerID (for P2P ring connections)
    #[serde(with = "peer_id_opt_serde")]
    pub peer_id: Option<PeerId>,
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

mod peer_id_opt_serde {
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

impl PoolBeacon {
    /// Create a new beacon
    pub fn new(
        pool_config: &PoolConfig,
        membership_cert: &PoolMembershipCert,
        device_keypair: &DeviceKeyPair,
        peer_id: Option<PeerId>,
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

        // Set capabilities based on role
        let mut capabilities = 0u32;
        if pool_config.role == crate::pki::MembershipRole::Admin {
            capabilities |= CAPABILITY_CAN_SIGN_CERTS;
            capabilities |= CAPABILITY_ACCEPTING_JOINS;
        }
        // TODO: Set CAPABILITY_HAS_RELAY and CAPABILITY_HAS_CONTROL_PLANE from config

        // Construct signing payload
        let mut payload = Vec::new();
        payload.push(1); // version
        payload.extend_from_slice(pool_config.pool_id.as_bytes());
        payload.extend_from_slice(node_id.as_bytes());
        payload.extend_from_slice(&quic_port.to_le_bytes());
        payload.extend_from_slice(&fingerprint);
        payload.extend_from_slice(&cert_hash);
        payload.extend_from_slice(&capabilities.to_le_bytes());
        payload.extend_from_slice(&timestamp.to_le_bytes());

        let signature = device_keypair.sign(&payload);

        PoolBeacon {
            version: 1,
            pool_id: pool_config.pool_id,
            node_id,
            peer_id,
            quic_port,
            pubkey_fingerprint: fingerprint,
            membership_cert_hash: cert_hash,
            capabilities_bitmap: capabilities,
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

    /// Check if this beacon advertises the capability to sign certificates (admin)
    pub fn can_sign_certs(&self) -> bool {
        self.capabilities_bitmap & CAPABILITY_CAN_SIGN_CERTS != 0
    }

    /// Check if this beacon is accepting new pool joins
    pub fn is_accepting_joins(&self) -> bool {
        self.capabilities_bitmap & CAPABILITY_ACCEPTING_JOINS != 0
    }

    /// Check if this node has a relay server
    pub fn has_relay(&self) -> bool {
        self.capabilities_bitmap & CAPABILITY_HAS_RELAY != 0
    }

    /// Check if this node has a control plane
    pub fn has_control_plane(&self) -> bool {
        self.capabilities_bitmap & CAPABILITY_HAS_CONTROL_PLANE != 0
    }
}

/// Beacon broadcaster
pub struct BeaconBroadcaster {
    socket: Arc<UdpSocket>,
    pools: Vec<(PoolConfig, PoolMembershipCert)>,
    device_keypair: DeviceKeyPair,
    peer_id: Option<PeerId>,
    quic_port: u16,
    ring_gossip_rx: mpsc::Receiver<RingGossipMessage>,
}

impl BeaconBroadcaster {
    /// Create a new beacon broadcaster
    pub async fn new(
        pools: Vec<(PoolConfig, PoolMembershipCert)>,
        device_keypair: DeviceKeyPair,
        peer_id: Option<PeerId>,
        quic_port: u16,
        ring_gossip_rx: mpsc::Receiver<RingGossipMessage>,
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
            peer_id,
            quic_port,
            ring_gossip_rx,
        })
    }

    /// Run beacon broadcaster loop
    pub async fn run(mut self) -> Result<()> {
        let mut pool_beacon_interval = interval(Duration::from_secs(BEACON_INTERVAL_SECS));
        let target_addr: SocketAddr = format!("{}:{}", BEACON_MULTICAST_ADDR, BEACON_MULTICAST_PORT).parse()?;

        loop {
            tokio::select! {
                _ = pool_beacon_interval.tick() => {
                    // Broadcast pool presence beacons
                    for (pool_config, membership_cert) in &self.pools {
                        let beacon = PoolBeacon::new(
                            pool_config,
                            membership_cert,
                            &self.device_keypair,
                            self.peer_id,
                            self.quic_port,
                        );

                        let message = BeaconMessage::PoolBeacon(beacon.clone());
                        self.broadcast_message(&message, &target_addr).await;
                    }
                }

                Some(ring_gossip) = self.ring_gossip_rx.recv() => {
                    // Broadcast ring gossip
                    let message = BeaconMessage::RingGossip(ring_gossip);
                    self.broadcast_message(&message, &target_addr).await;
                }
            }
        }
    }

    async fn broadcast_message(&self, message: &BeaconMessage, target_addr: &SocketAddr) {
        let mut packet = Vec::new();
        match ciborium::ser::into_writer(message, &mut packet) {
            Ok(_) => {
                match self.socket.send_to(&packet, target_addr).await {
                    Ok(sent) => {
                        tracing::trace!(bytes = sent, "Beacon message broadcast");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to broadcast message");
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to serialize message");
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
    ring_gossip_tx: Option<mpsc::Sender<RingGossipMessage>>,
    /// Track recently processed certificate requests by device_pubkey to prevent spam
    cert_request_cache: HashMap<[u8; 32], std::time::Instant>,
    /// Track recently processed certificate responses by device_pubkey
    cert_response_cache: HashMap<[u8; 32], std::time::Instant>,
}

impl BeaconListener {
    /// Create a new beacon listener
    pub async fn new(
        pools: Vec<(PoolConfig, PoolMembershipCert)>,
        device_keypair: &DeviceKeyPair,
    ) -> Result<(Self, mpsc::Receiver<DiscoveredPeer>, mpsc::Receiver<RingGossipMessage>)> {
        // === PRODUCTION MULTICAST SOCKET CONFIGURATION FOR macOS ===
        //
        // CRITICAL: macOS requires SO_REUSEPORT for multicast UDP sockets.
        // Unlike Linux (which uses SO_REUSEADDR), macOS will return "Address already
        // in use" if SO_REUSEPORT is not set when multiple sockets bind to the same port.
        //
        // References:
        // - https://lore.kernel.org/all/20220502003830.31062-1-cheptsov@ispras.ru/T/
        // - https://github.com/openframeworks/openFrameworks/issues/2937
        //
        // MULTICAST BEHAVIOR:
        // - Each socket with SO_REUSEPORT that joins a multicast group receives
        //   a COPY of every multicast packet (this is by design, not a bug)
        // - Deduplication must be handled at application layer (via cert_request_cache)
        // - IP_MULTICAST_LOOP disabled to prevent receiving our own broadcasts
        //
        // CROSS-PLATFORM:
        // - SO_REUSEADDR works on both Linux and macOS for binding
        // - SO_REUSEPORT required on macOS, optional on Linux
        // - socket2 crate provides consistent API across platforms

        let recv_addr: std::net::SocketAddr = format!("0.0.0.0:{}", BEACON_MULTICAST_PORT)
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid multicast address: {}", e))?;

        // Create raw socket with socket2 for pre-bind configuration
        let socket = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::DGRAM,
            Some(socket2::Protocol::UDP),
        )?;

        // Set SO_REUSEADDR (allows rebinding after crash/restart)
        socket.set_reuse_address(true)?;

        // Set SO_REUSEPORT (REQUIRED on macOS for multicast)
        #[cfg(unix)]
        socket.set_reuse_port(true)?;

        // Set nonblocking mode for tokio compatibility
        socket.set_nonblocking(true)?;

        // Bind to multicast port
        socket.bind(&recv_addr.into())?;

        // Convert socket2 → std::net → tokio (standard pattern for socket options)
        let std_socket: std::net::UdpSocket = socket.into();
        let socket = UdpSocket::from_std(std_socket)?;

        // Disable multicast loopback (don't receive our own broadcasts)
        // This prevents the sender from receiving its own multicast packets
        socket.set_multicast_loop_v4(false)?;

        // Join the multicast group (must be done AFTER bind)
        socket.join_multicast_v4(
            BEACON_MULTICAST_ADDR.parse()?,
            Ipv4Addr::new(0, 0, 0, 0),  // Listen on all interfaces
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
        let (ring_gossip_tx, ring_gossip_rx) = mpsc::channel(100);

        tracing::info!(
            pools = my_pools.len(),
            multicast_addr = BEACON_MULTICAST_ADDR,
            multicast_port = BEACON_MULTICAST_PORT,
            node_id = %my_node_id,
            "Beacon listener initialized with SO_REUSEPORT (macOS-compatible)"
        );

        Ok((
            BeaconListener {
                socket: Arc::new(socket),
                my_pools,
                my_node_id,
                discovered_tx,
                peer_cache: Arc::new(RwLock::new(peer_cache_map)),
                ring_gossip_tx: Some(ring_gossip_tx),
                cert_request_cache: HashMap::new(),
                cert_response_cache: HashMap::new(),
            },
            discovered_rx,
            ring_gossip_rx,
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

        // Track last cache cleanup time
        let mut last_cache_cleanup = std::time::Instant::now();

        loop {
            // Periodically clean up old certificate cache entries (every 60 seconds)
            if last_cache_cleanup.elapsed() > Duration::from_secs(60) {
                let before_req = self.cert_request_cache.len();
                let before_resp = self.cert_response_cache.len();

                // Remove entries older than 5 minutes
                self.cert_request_cache.retain(|_, last_seen| {
                    last_seen.elapsed() < Duration::from_secs(300)
                });
                self.cert_response_cache.retain(|_, last_seen| {
                    last_seen.elapsed() < Duration::from_secs(300)
                });

                let after_req = self.cert_request_cache.len();
                let after_resp = self.cert_response_cache.len();

                if before_req != after_req || before_resp != after_resp {
                    tracing::debug!(
                        request_cache_removed = before_req - after_req,
                        response_cache_removed = before_resp - after_resp,
                        "Cleaned up certificate cache"
                    );
                }

                last_cache_cleanup = std::time::Instant::now();
            }

            // Receive beacon
            match self.socket.recv_from(&mut buf).await {
                Ok((len, sender_addr)) => {
                    // Parse beacon message
                    match ciborium::de::from_reader::<BeaconMessage, _>(&buf[..len]) {
                        Ok(message) => {
                            match message {
                                BeaconMessage::PoolBeacon(beacon) => {
                                    self.handle_pool_beacon(beacon, sender_addr).await;
                                }
                                BeaconMessage::CertRequest(request) => {
                                    self.handle_cert_request(request, sender_addr).await;
                                }
                                BeaconMessage::CertResponse(cert) => {
                                    self.handle_cert_response(cert, sender_addr).await;
                                }
                                BeaconMessage::RingGossip(gossip) => {
                                    self.handle_ring_gossip(gossip).await;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!(
                                error = %e,
                                sender = %sender_addr,
                                "Failed to parse beacon message"
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

    async fn handle_pool_beacon(&mut self, beacon: PoolBeacon, sender_addr: SocketAddr) {
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

    /// Handle certificate signing request (admins only)
    async fn handle_cert_request(&mut self, request: CertSigningRequest, sender_addr: SocketAddr) {
        // 1. Check if we're admin for this pool (fast path rejection)
        let admin_pool = self.my_pools.get(&request.pool_id);
        let is_admin = admin_pool
            .map(|(config, _)| config.role == crate::pki::MembershipRole::Admin)
            .unwrap_or(false);

        if !is_admin {
            tracing::trace!("Ignoring CSR - not admin for this pool");
            return;
        }

        // 2. Check deduplication BEFORE logging (prevents spam)
        // This prevents certificate spam when Device 2 retries every 2 seconds
        // and when multiple multicast copies arrive (macOS SO_REUSEPORT behavior)
        let request_key = request.device_pubkey;
        if let Some(&last_processed) = self.cert_request_cache.get(&request_key) {
            if last_processed.elapsed() < std::time::Duration::from_secs(30) {
                // Only log duplicates at DEBUG level
                tracing::debug!(
                    device_pubkey = %hex::encode(&request_key[..8]),
                    sender = %sender_addr,
                    elapsed_ms = last_processed.elapsed().as_millis(),
                    "Ignoring duplicate certificate request (already processed)"
                );
                return;
            }
        }

        // 3. NOW log at INFO level (only for NEW requests that will be processed)
        tracing::info!(
            pool_id = %request.pool_id,
            node_id = %request.node_id,
            requested_role = ?request.requested_role,
            sender = %sender_addr,
            "Certificate signing request received (processing new request)"
        );

        // Verify CSR
        if let Err(e) = request.verify() {
            tracing::warn!(error = %e, "Invalid CSR signature");
            return;
        }

        // Check if CSR is recent
        if !request.is_recent() {
            tracing::warn!("CSR is too old (>5 minutes)");
            return;
        }

        // TODO: For Phase 1, we'll auto-approve Member role requests
        // In Phase 2, add authorization policy check here
        if request.requested_role == crate::pki::MembershipRole::Admin {
            tracing::warn!("Cannot auto-approve Admin role request");
            return;
        }

        // Load pool root keypair to sign cert
        let pool_dir = crate::pki::PoolConfig::pool_dir(&request.pool_id)
            .expect("Failed to get pool dir");
        let root_keypair_path = pool_dir.join("pool-root-private");

        let pool_root = match std::fs::read(&root_keypair_path) {
            Ok(bytes) => {
                if bytes.len() != 32 {
                    tracing::error!("Invalid pool root keypair size");
                    return;
                }
                let mut array = [0u8; 32];
                array.copy_from_slice(&bytes);
                match crate::pki::PoolRootKeyPair::from_private_bytes(array) {
                    Ok(kp) => kp,
                    Err(e) => {
                        tracing::error!(error = %e, "Failed to load pool root keypair");
                        return;
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to read pool root keypair");
                return;
            }
        };

        // Sign certificate (1 year expiry)
        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 365 * 24 * 60 * 60;

        let cert = PoolMembershipCert::new(
            request.device_pubkey,
            &pool_root,
            request.requested_role,
            expires_at,
        );

        // Mark this request as processed to prevent duplicate signing
        self.cert_request_cache.insert(request_key, std::time::Instant::now());

        tracing::info!(
            pool_id = %request.pool_id,
            node_id = %request.node_id,
            role = ?request.requested_role,
            "Signed membership certificate (first time)"
        );

        // Broadcast signed cert via LAN beacon
        let message = BeaconMessage::CertResponse(cert);
        let mut packet = Vec::new();
        if let Err(e) = ciborium::ser::into_writer(&message, &mut packet) {
            tracing::error!(error = %e, "Failed to serialize cert response");
            return;
        }

        let target_addr: SocketAddr = format!("{}:{}", BEACON_MULTICAST_ADDR, BEACON_MULTICAST_PORT)
            .parse()
            .unwrap();

        if let Err(e) = self.socket.send_to(&packet, target_addr).await {
            tracing::error!(error = %e, "Failed to send cert response");
        } else {
            tracing::info!(
                pool_id = %request.pool_id,
                node_id = %request.node_id,
                "Cert response broadcast via LAN"
            );
        }
    }

    /// Handle certificate response (members waiting for cert)
    async fn handle_cert_response(&mut self, cert: PoolMembershipCert, sender_addr: SocketAddr) {
        tracing::info!(
            pool_id = %cert.pool_id,
            node_id = %cert.node_id(),
            role = ?cert.role,
            sender = %sender_addr,
            "Certificate response received"
        );

        // Check if this cert is for us
        let my_device_pubkey = self.my_pools.values().next().map(|(_config, my_cert)| my_cert.device_pubkey);
        let is_for_me = my_device_pubkey.map(|pk| pk == cert.device_pubkey).unwrap_or(false);

        if !is_for_me {
            tracing::trace!("Cert not for us, ignoring");
            return;
        }

        // Get pool root pubkey
        let pool_root_pubkey = self
            .my_pools
            .get(&cert.pool_id)
            .map(|(config, _)| config.pool_root_pubkey);

        let pool_root_pubkey = match pool_root_pubkey {
            Some(pk) => pk,
            None => {
                tracing::warn!("Received cert for unknown pool");
                return;
            }
        };

        // Verify cert
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Err(e) = cert.verify(&pool_root_pubkey, current_time) {
            tracing::error!(error = %e, "Invalid certificate signature");
            return;
        }

        // Save cert to disk
        let pool_dir = match crate::pki::PoolConfig::pool_dir(&cert.pool_id) {
            Ok(dir) => dir,
            Err(e) => {
                tracing::error!(error = %e, "Failed to get pool dir");
                return;
            }
        };

        let cert_path = pool_dir.join("membership.cert");
        let cert_json = match serde_json::to_string_pretty(&cert) {
            Ok(json) => json,
            Err(e) => {
                tracing::error!(error = %e, "Failed to serialize cert");
                return;
            }
        };

        if let Err(e) = std::fs::write(&cert_path, cert_json) {
            tracing::error!(error = %e, "Failed to save cert");
            return;
        }

        tracing::info!(
            pool_id = %cert.pool_id,
            node_id = %cert.node_id(),
            role = ?cert.role,
            "Membership certificate saved"
        );

        // TODO: Phase 2 - Signal pool-join command that cert is ready via channel
    }

    /// Handle ring gossip message
    async fn handle_ring_gossip(&mut self, gossip: RingGossipMessage) {
        // Forward to ring gossip service
        if let Some(tx) = &self.ring_gossip_tx {
            if let Err(e) = tx.try_send(gossip) {
                tracing::debug!(error = %e, "Failed to forward ring gossip");
            } else {
                tracing::trace!("Ring gossip forwarded to service");
            }
        }
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

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, None, 4001);

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

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, None, 4001);

        // Serialize to CBOR
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&beacon, &mut bytes).unwrap();

        // Deserialize
        let decoded: PoolBeacon = ciborium::de::from_reader(&bytes[..]).unwrap();

        assert_eq!(decoded.pool_id, beacon.pool_id);
        assert_eq!(decoded.node_id, beacon.node_id);
        assert_eq!(decoded.quic_port, beacon.quic_port);
    }

    #[test]
    fn test_admin_beacon_capabilities() {
        let device = DeviceKeyPair::generate();
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        // Create admin certificate
        let cert = crate::pki::PoolMembershipCert::new(
            device.public,
            &pool_root,
            crate::pki::MembershipRole::Admin,  // Admin role
            u64::MAX,
        );

        let pool_config = PoolConfig {
            pool_id,
            name: "Test Pool".to_string(),
            pool_root_pubkey: pool_root.public,
            beacon_config: crate::pki::BeaconConfig::default(),
            role: crate::pki::MembershipRole::Admin,  // Admin role
            expires_at: u64::MAX,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, None, 4001);

        // Verify admin capabilities
        assert!(beacon.can_sign_certs(), "Admin beacon should have can_sign_certs=true");
        assert!(beacon.is_accepting_joins(), "Admin beacon should have is_accepting_joins=true");
        assert_eq!(beacon.capabilities_bitmap & CAPABILITY_CAN_SIGN_CERTS, CAPABILITY_CAN_SIGN_CERTS);
        assert_eq!(beacon.capabilities_bitmap & CAPABILITY_ACCEPTING_JOINS, CAPABILITY_ACCEPTING_JOINS);
    }

    #[test]
    fn test_member_beacon_capabilities() {
        let device = DeviceKeyPair::generate();
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        // Create member certificate
        let cert = crate::pki::PoolMembershipCert::new(
            device.public,
            &pool_root,
            crate::pki::MembershipRole::Member,  // Member role
            u64::MAX,
        );

        let pool_config = PoolConfig {
            pool_id,
            name: "Test Pool".to_string(),
            pool_root_pubkey: pool_root.public,
            beacon_config: crate::pki::BeaconConfig::default(),
            role: crate::pki::MembershipRole::Member,  // Member role
            expires_at: u64::MAX,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, None, 4001);

        // Verify member does NOT have signing capabilities
        assert!(!beacon.can_sign_certs(), "Member beacon should have can_sign_certs=false");
        assert!(!beacon.is_accepting_joins(), "Member beacon should have is_accepting_joins=false");
        assert_eq!(beacon.capabilities_bitmap & CAPABILITY_CAN_SIGN_CERTS, 0);
        assert_eq!(beacon.capabilities_bitmap & CAPABILITY_ACCEPTING_JOINS, 0);
    }

    #[test]
    fn test_beacon_capability_serialization() {
        let device = DeviceKeyPair::generate();
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        // Create admin beacon with capabilities
        let cert = crate::pki::PoolMembershipCert::new(
            device.public,
            &pool_root,
            crate::pki::MembershipRole::Admin,
            u64::MAX,
        );

        let pool_config = PoolConfig {
            pool_id,
            name: "Test Pool".to_string(),
            pool_root_pubkey: pool_root.public,
            beacon_config: crate::pki::BeaconConfig::default(),
            role: crate::pki::MembershipRole::Admin,
            expires_at: u64::MAX,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        let beacon = PoolBeacon::new(&pool_config, &cert, &device, None, 4001);

        // Serialize to CBOR
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&beacon, &mut bytes).unwrap();

        // Deserialize
        let decoded: PoolBeacon = ciborium::de::from_reader(&bytes[..]).unwrap();

        // Verify capabilities are preserved
        assert_eq!(decoded.capabilities_bitmap, beacon.capabilities_bitmap);
        assert_eq!(decoded.can_sign_certs(), beacon.can_sign_certs());
        assert_eq!(decoded.is_accepting_joins(), beacon.is_accepting_joins());
        assert!(decoded.can_sign_certs(), "Deserialized admin beacon should still have can_sign_certs=true");
        assert!(decoded.is_accepting_joins(), "Deserialized admin beacon should still have is_accepting_joins=true");
    }
}
