use libp2p::PeerId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Token-based authentication for relay clients
///
/// **NOTE:** This is a placeholder implementation for MVP.
/// In production (Phase 1 deployment), this will be replaced with:
/// - mTLS certificate validation
/// - Control plane integration for device verification
/// - Network membership checks
pub struct TokenAuth {
    expected_token: String,
    authenticated_peers: HashMap<PeerId, Instant>,
    auth_timeout: Duration,
}

impl TokenAuth {
    /// Create new token authenticator
    pub fn new(expected_token: String) -> Self {
        Self {
            expected_token,
            authenticated_peers: HashMap::new(),
            auth_timeout: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Authenticate a peer with a token
    ///
    /// Returns `true` if authentication succeeds, `false` otherwise.
    pub fn authenticate(&mut self, peer_id: PeerId, token: &str) -> bool {
        if token == self.expected_token {
            let expiry = Instant::now() + self.auth_timeout;
            self.authenticated_peers.insert(peer_id, expiry);

            tracing::info!(
                peer_id = %peer_id,
                expiry_secs = self.auth_timeout.as_secs(),
                "Peer authenticated successfully"
            );

            true
        } else {
            tracing::warn!(
                peer_id = %peer_id,
                "Authentication failed: invalid token"
            );

            false
        }
    }

    /// Check if a peer is authenticated
    pub fn is_authenticated(&mut self, peer_id: &PeerId) -> bool {
        if let Some(expiry) = self.authenticated_peers.get(peer_id) {
            if Instant::now() < *expiry {
                return true;
            }

            // Auth expired
            tracing::info!(peer_id = %peer_id, "Authentication expired");
            self.authenticated_peers.remove(peer_id);
        }

        false
    }

    /// Remove expired authentications
    #[allow(dead_code)]
    pub fn cleanup_expired(&mut self) {
        let now = Instant::now();
        let before_count = self.authenticated_peers.len();

        self.authenticated_peers.retain(|peer_id, expiry| {
            if now >= *expiry {
                tracing::debug!(peer_id = %peer_id, "Removed expired authentication");
                false
            } else {
                true
            }
        });

        let removed = before_count - self.authenticated_peers.len();
        if removed > 0 {
            tracing::info!(removed = removed, "Cleaned up expired authentications");
        }
    }

    /// Get count of authenticated peers
    #[allow(dead_code)]
    pub fn authenticated_count(&self) -> usize {
        self.authenticated_peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity::Keypair;

    fn test_peer_id() -> PeerId {
        Keypair::generate_ed25519().public().to_peer_id()
    }

    #[test]
    fn test_authenticate_success() {
        let mut auth = TokenAuth::new("test-token".to_string());
        let peer_id = test_peer_id();

        assert!(auth.authenticate(peer_id, "test-token"));
        assert!(auth.is_authenticated(&peer_id));
    }

    #[test]
    fn test_authenticate_failure() {
        let mut auth = TokenAuth::new("correct-token".to_string());
        let peer_id = test_peer_id();

        assert!(!auth.authenticate(peer_id, "wrong-token"));
        assert!(!auth.is_authenticated(&peer_id));
    }

    #[test]
    fn test_auth_expiry() {
        let mut auth = TokenAuth::new("test-token".to_string());
        auth.auth_timeout = Duration::from_millis(100);

        let peer_id = test_peer_id();
        auth.authenticate(peer_id, "test-token");
        assert!(auth.is_authenticated(&peer_id));

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(150));
        assert!(!auth.is_authenticated(&peer_id));
    }

    #[test]
    fn test_cleanup_expired() {
        let mut auth = TokenAuth::new("test-token".to_string());
        auth.auth_timeout = Duration::from_millis(100);

        let peer1 = test_peer_id();
        let peer2 = test_peer_id();

        auth.authenticate(peer1, "test-token");
        auth.authenticate(peer2, "test-token");

        assert_eq!(auth.authenticated_count(), 2);

        std::thread::sleep(Duration::from_millis(150));
        auth.cleanup_expired();

        assert_eq!(auth.authenticated_count(), 0);
    }
}
