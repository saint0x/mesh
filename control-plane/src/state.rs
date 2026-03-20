use crate::api::error::{ApiError, ApiResult};
use crate::db::Database;
use crate::services::certificate::ControlPlaneKeypair;
use crate::services::ring_manager::RingTopologyManager;
use crate::services::topology_notifier::TopologyNotifier;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Default model ID for ring topology
const DEFAULT_MODEL_ID: &str = "default-model";

/// Axum application state shared across all request handlers
#[derive(Clone)]
pub struct AppState {
    /// Database connection pool
    pub db: Database,
    /// Control plane keypair for signing certificates
    pub keypair: Arc<ControlPlaneKeypair>,
    /// Ring managers per network (lazily created)
    ring_managers: Arc<RwLock<HashMap<String, Arc<RingTopologyManager>>>>,
    /// Topology notifier for worker notifications
    pub topology_notifier: Arc<TopologyNotifier>,
    /// Relay addresses returned to agents during registration
    pub relay_addresses: Arc<Vec<String>>,
}

impl AppState {
    /// Create new application state
    pub fn new(db: Database, keypair: Arc<ControlPlaneKeypair>) -> Self {
        Self::with_relay_addresses(
            db,
            keypair,
            default_relay_addresses(),
        )
    }

    /// Create new application state with explicit relay addresses
    pub fn with_relay_addresses(
        db: Database,
        keypair: Arc<ControlPlaneKeypair>,
        relay_addresses: Vec<String>,
    ) -> Self {
        let topology_notifier = Arc::new(TopologyNotifier::new(Arc::new(db.clone())));
        Self {
            db,
            keypair,
            ring_managers: Arc::new(RwLock::new(HashMap::new())),
            topology_notifier,
            relay_addresses: Arc::new(relay_addresses),
        }
    }

    /// Get or create a ring manager for a network
    pub fn get_ring_manager(&self, network_id: &str) -> ApiResult<Arc<RingTopologyManager>> {
        // First try read lock
        {
            let managers = self.ring_managers.read().map_err(|_| {
                ApiError::Internal("Failed to acquire ring_managers read lock".to_string())
            })?;

            if let Some(manager) = managers.get(network_id) {
                return Ok(manager.clone());
            }
        }

        // Need to create new manager, acquire write lock
        let mut managers = self.ring_managers.write().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_managers write lock".to_string())
        })?;

        // Double-check after acquiring write lock
        if let Some(manager) = managers.get(network_id) {
            return Ok(manager.clone());
        }

        // Create new manager
        let manager = Arc::new(RingTopologyManager::new(
            Arc::new(self.db.clone()),
            DEFAULT_MODEL_ID.to_string(),
        ));

        // Load existing topology from database
        manager.load_from_db(network_id)?;

        managers.insert(network_id.to_string(), manager.clone());

        Ok(manager)
    }
}

fn default_relay_addresses() -> Vec<String> {
    std::env::var("MESHNET_RELAY_ADDRESSES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|addresses| !addresses.is_empty())
        .unwrap_or_default()
}
