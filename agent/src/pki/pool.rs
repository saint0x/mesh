use super::types::{DeviceKeyPair, MembershipRole, NodeId, PoolId, PoolMembershipCert, PoolRootKeyPair};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Beacon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaconConfig {
    pub enabled: bool,
    pub multicast_addr: String,
    pub multicast_port: u16,
    pub interval_secs: u64,
}

impl Default for BeaconConfig {
    fn default() -> Self {
        BeaconConfig {
            enabled: true,
            multicast_addr: "239.192.0.1".to_string(),
            multicast_port: 42424,
            interval_secs: 5,
        }
    }
}

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub pool_id: PoolId,
    pub name: String,
    pub pool_root_pubkey: [u8; 32],
    pub beacon_config: BeaconConfig,
    pub role: MembershipRole,
    pub expires_at: u64,
    pub created_at: String,
}

impl PoolConfig {
    /// Create a new pool (admin role, self-issued cert)
    pub fn create_pool(name: String, device: &DeviceKeyPair) -> Result<(Self, PoolRootKeyPair, PoolMembershipCert)> {
        // Generate pool root keypair (CLIENT-SIDE, never sent to platform)
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        // Admin role never expires (or set to far future)
        let expires_at = u64::MAX;

        // Issue self-signed membership cert as admin
        let membership_cert = PoolMembershipCert::new(
            device.public,
            &pool_root,
            MembershipRole::Admin,
            expires_at,
        );

        let created_at = chrono::Utc::now().to_rfc3339();

        let config = PoolConfig {
            pool_id,
            name,
            pool_root_pubkey: pool_root.public,
            beacon_config: BeaconConfig::default(),
            role: MembershipRole::Admin,
            expires_at,
            created_at,
        };

        Ok((config, pool_root, membership_cert))
    }

    /// Join an existing pool (member role)
    pub fn join_pool(
        pool_id: PoolId,
        name: String,
        pool_root_pubkey: [u8; 32],
        membership_cert: PoolMembershipCert,
    ) -> Result<Self> {
        let created_at = chrono::Utc::now().to_rfc3339();

        Ok(PoolConfig {
            pool_id,
            name,
            pool_root_pubkey,
            beacon_config: BeaconConfig::default(),
            role: membership_cert.role,
            expires_at: membership_cert.expires_at,
            created_at,
        })
    }

    /// Get pool directory path: ~/.meshnet/pools/{pool_id}/
    pub fn pool_dir(pool_id: &PoolId) -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("Could not determine home directory"))?;

        Ok(home.join(".meshnet").join("pools").join(pool_id.to_hex()))
    }

    /// Get config file path: ~/.meshnet/pools/{pool_id}/config.toml
    pub fn config_path(pool_id: &PoolId) -> Result<PathBuf> {
        Ok(Self::pool_dir(pool_id)?.join("config.toml"))
    }

    /// Get membership cert path: ~/.meshnet/pools/{pool_id}/membership.cert
    pub fn membership_cert_path(pool_id: &PoolId) -> Result<PathBuf> {
        Ok(Self::pool_dir(pool_id)?.join("membership.cert"))
    }

    /// Get pool root pubkey path: ~/.meshnet/pools/{pool_id}/pool-root-pubkey
    pub fn pool_root_pubkey_path(pool_id: &PoolId) -> Result<PathBuf> {
        Ok(Self::pool_dir(pool_id)?.join("pool-root-pubkey"))
    }

    /// Get peer cache path: ~/.meshnet/pools/{pool_id}/peer_cache.json
    pub fn peer_cache_path(pool_id: &PoolId) -> Result<PathBuf> {
        Ok(Self::pool_dir(pool_id)?.join("peer_cache.json"))
    }

    /// Get pool root private key path (admin only): ~/.meshnet/pools/{pool_id}/pool-root-private
    pub fn pool_root_private_path(pool_id: &PoolId) -> Result<PathBuf> {
        Ok(Self::pool_dir(pool_id)?.join("pool-root-private"))
    }

    /// Save pool configuration and membership cert
    pub fn save(&self, membership_cert: &PoolMembershipCert, pool_root: Option<&PoolRootKeyPair>) -> Result<()> {
        let pool_dir = Self::pool_dir(&self.pool_id)?;
        fs::create_dir_all(&pool_dir)?;

        // Save config.toml (atomic write)
        let config_path = Self::config_path(&self.pool_id)?;
        let toml_string = toml::to_string_pretty(self)?;
        let temp_path = config_path.with_extension("toml.tmp");
        fs::write(&temp_path, &toml_string)?;
        fs::rename(&temp_path, &config_path)?;

        // Save membership.cert (CBOR binary, atomic write)
        let cert_path = Self::membership_cert_path(&self.pool_id)?;
        let mut cert_bytes = Vec::new();
        ciborium::ser::into_writer(membership_cert, &mut cert_bytes)?;
        let temp_path = cert_path.with_extension("cert.tmp");
        fs::write(&temp_path, &cert_bytes)?;
        fs::rename(&temp_path, &cert_path)?;

        // Save pool-root-pubkey (hex encoded)
        let pubkey_path = Self::pool_root_pubkey_path(&self.pool_id)?;
        fs::write(&pubkey_path, hex::encode(self.pool_root_pubkey))?;

        // Save pool-root-private if admin
        if let Some(root) = pool_root {
            let private_path = Self::pool_root_private_path(&self.pool_id)?;
            let private_bytes = root.private;
            let temp_path = private_path.with_extension("tmp");
            fs::write(&temp_path, &private_bytes)?;
            fs::rename(&temp_path, &private_path)?;

            tracing::info!(
                pool_id = %self.pool_id,
                pool_dir = %pool_dir.display(),
                "Pool configuration saved (admin with private key)"
            );
        } else {
            tracing::info!(
                pool_id = %self.pool_id,
                pool_dir = %pool_dir.display(),
                "Pool configuration saved (member)"
            );
        }

        Ok(())
    }

    /// Load pool configuration and membership cert
    pub fn load(pool_id: &PoolId) -> Result<(Self, PoolMembershipCert)> {
        let config_path = Self::config_path(pool_id)?;
        let cert_path = Self::membership_cert_path(pool_id)?;

        // Load config
        let content = fs::read_to_string(&config_path)?;
        let config: PoolConfig = toml::from_str(&content)?;

        // Load membership cert
        let cert_bytes = fs::read(&cert_path)?;
        let membership_cert: PoolMembershipCert = ciborium::de::from_reader(&cert_bytes[..])?;

        tracing::info!(
            pool_id = %pool_id,
            role = %config.role,
            "Pool configuration loaded"
        );

        Ok((config, membership_cert))
    }

    /// Load pool root keypair (admin only)
    pub fn load_pool_root(pool_id: &PoolId) -> Result<PoolRootKeyPair> {
        let private_path = Self::pool_root_private_path(pool_id)?;

        if !private_path.exists() {
            return Err(anyhow!("Pool root private key not found (not admin?)"));
        }

        let private_bytes = fs::read(&private_path)?;
        if private_bytes.len() != 32 {
            return Err(anyhow!("Invalid private key length"));
        }

        let mut private = [0u8; 32];
        private.copy_from_slice(&private_bytes);

        PoolRootKeyPair::from_private_bytes(private)
    }

    /// List all pools
    pub fn list_pools() -> Result<Vec<(PoolId, PoolConfig, PoolMembershipCert)>> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("Could not determine home directory"))?;

        let pools_dir = home.join(".meshnet").join("pools");

        if !pools_dir.exists() {
            return Ok(Vec::new());
        }

        let mut pools = Vec::new();

        for entry in fs::read_dir(&pools_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // Try to parse directory name as pool ID
                if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Ok(pool_id) = PoolId::from_hex(dir_name) {
                        // Try to load pool config
                        if let Ok((config, cert)) = Self::load(&pool_id) {
                            pools.push((pool_id, config, cert));
                        }
                    }
                }
            }
        }

        Ok(pools)
    }

    /// Check if membership cert is valid
    pub fn is_cert_valid(&self, cert: &PoolMembershipCert) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        cert.is_valid(&self.pool_root_pubkey, now)
    }

    /// Get days until expiration (None if never expires)
    pub fn days_until_expiry(&self) -> Option<i64> {
        if self.expires_at == u64::MAX {
            return None;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let remaining_secs = self.expires_at.saturating_sub(now);
        Some((remaining_secs / 86400) as i64)
    }
}

/// Discovered peer info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPeer {
    pub pool_id: PoolId,
    pub node_id: NodeId,
    pub lan_addr: String,
    pub discovery_method: DiscoveryMethod,
    pub last_seen: u64,
}

/// Discovery method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    LAN,
    Cached,
    Rendezvous,
    Relay,
}

/// Peer cache for storing discovered peers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PeerCache {
    pub peers: Vec<DiscoveredPeer>,
}

impl PeerCache {
    /// Load peer cache from file
    pub fn load(pool_id: &PoolId) -> Result<Self> {
        let cache_path = PoolConfig::peer_cache_path(pool_id)?;

        if !cache_path.exists() {
            return Ok(PeerCache::default());
        }

        let content = fs::read_to_string(&cache_path)?;
        let cache: PeerCache = serde_json::from_str(&content)?;

        Ok(cache)
    }

    /// Save peer cache to file
    pub fn save(&self, pool_id: &PoolId) -> Result<()> {
        let cache_path = PoolConfig::peer_cache_path(pool_id)?;
        let json_string = serde_json::to_string_pretty(self)?;

        let temp_path = cache_path.with_extension("json.tmp");
        fs::write(&temp_path, &json_string)?;
        fs::rename(&temp_path, &cache_path)?;

        Ok(())
    }

    /// Add or update a peer
    pub fn upsert_peer(&mut self, peer: DiscoveredPeer) {
        // Remove old entry if exists
        self.peers.retain(|p| !(p.pool_id == peer.pool_id && p.node_id == peer.node_id));

        // Add new entry
        self.peers.push(peer);
    }

    /// Get peers for a pool
    pub fn get_peers(&self, pool_id: &PoolId) -> Vec<&DiscoveredPeer> {
        self.peers
            .iter()
            .filter(|p| p.pool_id == *pool_id)
            .collect()
    }

    /// Remove stale peers (older than threshold)
    pub fn remove_stale(&mut self, threshold_secs: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.peers.retain(|p| now - p.last_seen < threshold_secs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_pool() {
        let device = DeviceKeyPair::generate();
        let (config, pool_root, cert) = PoolConfig::create_pool("Test Pool".to_string(), &device).unwrap();

        assert_eq!(config.name, "Test Pool");
        assert_eq!(config.role, MembershipRole::Admin);
        assert_eq!(config.pool_id, pool_root.pool_id());
        assert_eq!(cert.device_pubkey, device.public);
        assert_eq!(cert.role, MembershipRole::Admin);
    }

    #[test]
    fn test_pool_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let device = DeviceKeyPair::generate();

        // Override home directory for test
        std::env::set_var("HOME", temp_dir.path());

        let (config, pool_root, cert) = PoolConfig::create_pool("Test Pool".to_string(), &device).unwrap();

        // Save
        config.save(&cert, Some(&pool_root)).unwrap();

        // Load
        let (loaded_config, loaded_cert) = PoolConfig::load(&config.pool_id).unwrap();

        assert_eq!(config.pool_id, loaded_config.pool_id);
        assert_eq!(config.name, loaded_config.name);
        assert_eq!(config.role, loaded_config.role);
        assert_eq!(cert.device_pubkey, loaded_cert.device_pubkey);
    }

    #[test]
    fn test_list_pools() {
        let temp_dir = TempDir::new().unwrap();
        let device = DeviceKeyPair::generate();

        std::env::set_var("HOME", temp_dir.path());

        // Create and save multiple pools
        let (config1, root1, cert1) = PoolConfig::create_pool("Pool 1".to_string(), &device).unwrap();
        config1.save(&cert1, Some(&root1)).unwrap();

        let (config2, root2, cert2) = PoolConfig::create_pool("Pool 2".to_string(), &device).unwrap();
        config2.save(&cert2, Some(&root2)).unwrap();

        // List pools
        let pools = PoolConfig::list_pools().unwrap();
        assert_eq!(pools.len(), 2);

        let pool_ids: Vec<PoolId> = pools.iter().map(|(id, _, _)| *id).collect();
        assert!(pool_ids.contains(&config1.pool_id));
        assert!(pool_ids.contains(&config2.pool_id));
    }

    #[test]
    fn test_peer_cache() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("HOME", temp_dir.path());

        let pool_id = PoolId::from_bytes([1u8; 32]);
        let node_id = NodeId::from_bytes([2u8; 32]);

        // Create cache
        let mut cache = PeerCache::default();

        let peer = DiscoveredPeer {
            pool_id,
            node_id,
            lan_addr: "192.168.1.100:4001".to_string(),
            discovery_method: DiscoveryMethod::LAN,
            last_seen: 1234567890,
        };

        cache.upsert_peer(peer);

        // Save and load
        fs::create_dir_all(PoolConfig::pool_dir(&pool_id).unwrap()).unwrap();
        cache.save(&pool_id).unwrap();
        let loaded_cache = PeerCache::load(&pool_id).unwrap();

        assert_eq!(loaded_cache.peers.len(), 1);
        assert_eq!(loaded_cache.peers[0].node_id, node_id);
    }
}
