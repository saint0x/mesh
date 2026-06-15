use crate::api::error::{execute_with_db_lock_retry, ApiError, ApiResult};
use crate::connectivity::InferenceSchedulingPolicy;
use crate::db::Database;
use crate::services::certificate::ControlPlaneKeypair;
use crate::services::network_service;
use crate::services::ring_manager::RingTopologyManager;
use crate::services::topology_notifier::TopologyNotifier;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Axum application state shared across all request handlers
#[derive(Clone)]
pub struct AppState {
    /// Database connection pool
    pub db: Database,
    /// Control plane keypair for signing certificates
    pub keypair: Arc<ControlPlaneKeypair>,
    /// Ring managers per network (lazily created)
    ring_managers: Arc<RwLock<HashMap<String, Arc<RingTopologyManager>>>>,
    /// Stable network-level scheduling config cached out of the SQLite hot path.
    network_scheduling_policies: Arc<RwLock<HashMap<String, InferenceSchedulingPolicy>>>,
    /// Topology notifier for worker notifications
    pub topology_notifier: Arc<TopologyNotifier>,
    /// SQLite is a single-writer database; serialize inference mutations explicitly.
    pub inference_write_gate: Arc<Mutex<()>>,
}

impl AppState {
    /// Create new application state
    pub fn new(db: Database, keypair: Arc<ControlPlaneKeypair>) -> Self {
        let topology_notifier = Arc::new(TopologyNotifier::new(Arc::new(db.clone())));
        let inference_write_gate = db.write_gate();
        Self {
            db,
            keypair,
            ring_managers: Arc::new(RwLock::new(HashMap::new())),
            network_scheduling_policies: Arc::new(RwLock::new(HashMap::new())),
            topology_notifier,
            inference_write_gate,
        }
    }

    pub fn get_network_scheduling_policy(
        &self,
        network_id: &str,
    ) -> ApiResult<InferenceSchedulingPolicy> {
        if let Some(policy) = self.load_cached_network_scheduling_policy(network_id)? {
            return Ok(policy);
        }

        let _write_guard = self
            .inference_write_gate
            .lock()
            .map_err(|_| ApiError::Internal("Inference write gate lock poisoned".to_string()))?;

        if let Some(policy) = self.load_cached_network_scheduling_policy(network_id)? {
            return Ok(policy);
        }

        let policy = execute_with_db_lock_retry(|| {
            Ok(network_service::load_network_settings(&self.db, network_id)?.scheduling_policy)
        })?;
        let mut policies = self.network_scheduling_policies.write().map_err(|_| {
            ApiError::Internal(
                "Failed to acquire network_scheduling_policies write lock".to_string(),
            )
        })?;
        let entry = policies
            .entry(network_id.to_string())
            .or_insert_with(|| policy.clone());
        Ok(entry.clone())
    }

    fn load_cached_network_scheduling_policy(
        &self,
        network_id: &str,
    ) -> ApiResult<Option<InferenceSchedulingPolicy>> {
        let policies = self.network_scheduling_policies.read().map_err(|_| {
            ApiError::Internal(
                "Failed to acquire network_scheduling_policies read lock".to_string(),
            )
        })?;
        Ok(policies.get(network_id).cloned())
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
        let manager = Arc::new(RingTopologyManager::new(Arc::new(self.db.clone())));

        // Load existing topology from database
        manager.load_from_db(network_id)?;

        managers.insert(network_id.to_string(), manager.clone());

        Ok(manager)
    }
}
