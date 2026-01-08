use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Cryptographic pool identifier derived from pool root public key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolId([u8; 32]);

impl PoolId {
    /// Compute pool ID from pool root public key
    pub fn from_pubkey(pubkey: &[u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"meshnet-pool-v1");
        hasher.update(pubkey);
        let hash = hasher.finalize();
        PoolId(hash.into())
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        PoolId(bytes)
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string
    pub fn from_hex(s: &str) -> Result<Self> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 32 {
            return Err(anyhow!("Invalid pool ID length: expected 32 bytes"));
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(&bytes);
        Ok(PoolId(array))
    }
}

impl fmt::Display for PoolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.to_hex()[..16])
    }
}

/// Cryptographic node identifier derived from device public key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId([u8; 32]);

impl NodeId {
    /// Compute node ID from device public key
    pub fn from_pubkey(pubkey: &[u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"meshnet-node-v1");
        hasher.update(pubkey);
        let hash = hasher.finalize();
        NodeId(hash.into())
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        NodeId(bytes)
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string
    pub fn from_hex(s: &str) -> Result<Self> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 32 {
            return Err(anyhow!("Invalid node ID length: expected 32 bytes"));
        }
        let mut array = [0u8; 32];
        array.copy_from_slice(&bytes);
        Ok(NodeId(array))
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.to_hex()[..16])
    }
}

/// Device keypair for node identity
#[derive(Clone)]
pub struct DeviceKeyPair {
    pub public: [u8; 32],
    pub private: [u8; 32],
    keypair: SigningKey,
}

impl DeviceKeyPair {
    /// Generate a new random device keypair
    pub fn generate() -> Self {
        let mut rng = OsRng;
        let keypair = SigningKey::generate(&mut rng);
        let public = keypair.verifying_key().to_bytes();
        let private = keypair.to_bytes();

        DeviceKeyPair {
            public,
            private,
            keypair,
        }
    }

    /// Create from existing private key bytes
    pub fn from_private_bytes(private: [u8; 32]) -> Result<Self> {
        let keypair = SigningKey::from_bytes(&private);
        let public = keypair.verifying_key().to_bytes();

        Ok(DeviceKeyPair {
            public,
            private,
            keypair,
        })
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.keypair.sign(message).to_bytes()
    }

    /// Verify a signature on a message
    pub fn verify(&self, message: &[u8], signature: &[u8; 64]) -> bool {
        let verifying_key = self.keypair.verifying_key();
        let sig = Signature::from_bytes(signature);
        verifying_key.verify(message, &sig).is_ok()
    }

    /// Get node ID for this device
    pub fn node_id(&self) -> NodeId {
        NodeId::from_pubkey(&self.public)
    }
}

/// Pool root keypair (held by pool admin only)
#[derive(Clone)]
pub struct PoolRootKeyPair {
    pub public: [u8; 32],
    pub private: [u8; 32],
    keypair: SigningKey,
}

impl PoolRootKeyPair {
    /// Generate a new random pool root keypair
    pub fn generate() -> Self {
        let mut rng = OsRng;
        let keypair = SigningKey::generate(&mut rng);
        let public = keypair.verifying_key().to_bytes();
        let private = keypair.to_bytes();

        PoolRootKeyPair {
            public,
            private,
            keypair,
        }
    }

    /// Create from existing private key bytes
    pub fn from_private_bytes(private: [u8; 32]) -> Result<Self> {
        let keypair = SigningKey::from_bytes(&private);
        let public = keypair.verifying_key().to_bytes();

        Ok(PoolRootKeyPair {
            public,
            private,
            keypair,
        })
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.keypair.sign(message).to_bytes()
    }

    /// Get pool ID for this pool
    pub fn pool_id(&self) -> PoolId {
        PoolId::from_pubkey(&self.public)
    }
}

/// Membership role in a pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MembershipRole {
    /// Regular member (can participate in pool)
    Member,
    /// Admin (can issue membership certificates)
    Admin,
}

impl fmt::Display for MembershipRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MembershipRole::Member => write!(f, "member"),
            MembershipRole::Admin => write!(f, "admin"),
        }
    }
}

/// Pool membership certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolMembershipCert {
    /// Device public key of the member
    pub device_pubkey: [u8; 32],
    /// Pool this cert is for
    pub pool_id: PoolId,
    /// Role in the pool
    pub role: MembershipRole,
    /// Expiration timestamp (Unix seconds)
    pub expires_at: u64,
    /// Signature by pool root keypair (hex encoded for serde)
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

impl PoolMembershipCert {
    /// Create and sign a new membership certificate
    pub fn new(
        device_pubkey: [u8; 32],
        pool_root: &PoolRootKeyPair,
        role: MembershipRole,
        expires_at: u64,
    ) -> Self {
        let pool_id = pool_root.pool_id();

        // Construct signing payload
        let mut payload = Vec::new();
        payload.extend_from_slice(&device_pubkey);
        payload.extend_from_slice(pool_id.as_bytes());
        payload.push(match role {
            MembershipRole::Member => 0,
            MembershipRole::Admin => 1,
        });
        payload.extend_from_slice(&expires_at.to_le_bytes());

        let signature = pool_root.sign(&payload);

        PoolMembershipCert {
            device_pubkey,
            pool_id,
            role,
            expires_at,
            signature,
        }
    }

    /// Verify certificate signature and expiry
    pub fn verify(&self, pool_root_pubkey: &[u8; 32], current_time: u64) -> Result<()> {
        // Check expiry
        if current_time >= self.expires_at {
            return Err(anyhow!("Certificate expired"));
        }

        // Check pool ID matches
        let expected_pool_id = PoolId::from_pubkey(pool_root_pubkey);
        if self.pool_id != expected_pool_id {
            return Err(anyhow!("Pool ID mismatch"));
        }

        // Verify signature
        let mut payload = Vec::new();
        payload.extend_from_slice(&self.device_pubkey);
        payload.extend_from_slice(self.pool_id.as_bytes());
        payload.push(match self.role {
            MembershipRole::Member => 0,
            MembershipRole::Admin => 1,
        });
        payload.extend_from_slice(&self.expires_at.to_le_bytes());

        let verifying_key = VerifyingKey::from_bytes(pool_root_pubkey)
            .map_err(|e| anyhow!("Invalid pool root pubkey: {}", e))?;

        let signature = Signature::from_bytes(&self.signature);

        verifying_key
            .verify(&payload, &signature)
            .map_err(|e| anyhow!("Invalid signature: {}", e))?;

        Ok(())
    }

    /// Check if certificate is valid (signature + not expired)
    pub fn is_valid(&self, pool_root_pubkey: &[u8; 32], current_time: u64) -> bool {
        self.verify(pool_root_pubkey, current_time).is_ok()
    }

    /// Get node ID for this certificate's device
    pub fn node_id(&self) -> NodeId {
        NodeId::from_pubkey(&self.device_pubkey)
    }

    /// Compute certificate hash
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.device_pubkey);
        hasher.update(self.pool_id.as_bytes());
        hasher.update(&[match self.role {
            MembershipRole::Member => 0,
            MembershipRole::Admin => 1,
        }]);
        hasher.update(&self.expires_at.to_le_bytes());
        hasher.update(&self.signature);
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_id_from_pubkey() {
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = PoolId::from_pubkey(&pool_root.public);

        // Pool ID should be deterministic
        let pool_id2 = PoolId::from_pubkey(&pool_root.public);
        assert_eq!(pool_id, pool_id2);

        // Should match pool_root.pool_id()
        assert_eq!(pool_id, pool_root.pool_id());
    }

    #[test]
    fn test_node_id_from_pubkey() {
        let device = DeviceKeyPair::generate();
        let node_id = NodeId::from_pubkey(&device.public);

        // Node ID should be deterministic
        let node_id2 = NodeId::from_pubkey(&device.public);
        assert_eq!(node_id, node_id2);

        // Should match device.node_id()
        assert_eq!(node_id, device.node_id());
    }

    #[test]
    fn test_pool_id_hex_roundtrip() {
        let pool_root = PoolRootKeyPair::generate();
        let pool_id = pool_root.pool_id();

        let hex = pool_id.to_hex();
        let parsed = PoolId::from_hex(&hex).unwrap();

        assert_eq!(pool_id, parsed);
    }

    #[test]
    fn test_device_keypair_sign_verify() {
        let device = DeviceKeyPair::generate();
        let message = b"hello world";

        let signature = device.sign(message);
        assert!(device.verify(message, &signature));

        // Wrong message should fail
        assert!(!device.verify(b"wrong message", &signature));
    }

    #[test]
    fn test_membership_cert_create_and_verify() {
        let pool_root = PoolRootKeyPair::generate();
        let device = DeviceKeyPair::generate();

        let expires_at = 1735152000; // Some future timestamp
        let cert = PoolMembershipCert::new(
            device.public,
            &pool_root,
            MembershipRole::Member,
            expires_at,
        );

        // Should verify successfully before expiry
        let current_time = expires_at - 1000;
        assert!(cert.verify(&pool_root.public, current_time).is_ok());
        assert!(cert.is_valid(&pool_root.public, current_time));

        // Should fail after expiry
        let current_time = expires_at + 1;
        assert!(cert.verify(&pool_root.public, current_time).is_err());
        assert!(!cert.is_valid(&pool_root.public, current_time));
    }

    #[test]
    fn test_membership_cert_wrong_pool() {
        let pool_root1 = PoolRootKeyPair::generate();
        let pool_root2 = PoolRootKeyPair::generate();
        let device = DeviceKeyPair::generate();

        let expires_at = 1735152000;
        let cert = PoolMembershipCert::new(
            device.public,
            &pool_root1,
            MembershipRole::Member,
            expires_at,
        );

        // Should fail with wrong pool root pubkey
        let current_time = expires_at - 1000;
        assert!(cert.verify(&pool_root2.public, current_time).is_err());
    }
}
