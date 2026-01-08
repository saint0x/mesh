use crate::api::error::{ApiError, ApiResult};
use crate::db::Database;
use crate::services::certificate::ControlPlaneKeypair;
use crate::services::ring_manager::RingTopologyManager;
use crate::services::topology_notifier::TopologyNotifier;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// Default model ID for ring topology
const DEFAULT_MODEL_ID: &str = "default-model";

/// Distributed inference job to be distributed to workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedInferenceJob {
    /// Unique job ID
    pub job_id: String,
    /// Network ID this job belongs to
    pub network_id: String,
    /// Model ID to use
    pub model_id: String,
    /// Tokenized prompt
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Unix timestamp when job was created
    pub created_at: u64,
}

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
    /// Job queue for distributed inference (network_id -> job queue)
    pub job_queues: Arc<RwLock<HashMap<String, VecDeque<DistributedInferenceJob>>>>,
}

impl AppState {
    /// Create new application state
    pub fn new(db: Database, keypair: Arc<ControlPlaneKeypair>) -> Self {
        let topology_notifier = Arc::new(TopologyNotifier::new(Arc::new(db.clone())));
        Self {
            db,
            keypair,
            ring_managers: Arc::new(RwLock::new(HashMap::new())),
            topology_notifier,
            job_queues: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enqueue an inference job for a network
    pub fn enqueue_job(&self, job: DistributedInferenceJob) -> ApiResult<()> {
        let mut queues = self.job_queues.write().map_err(|_| {
            ApiError::Internal("Failed to acquire job_queues write lock".to_string())
        })?;

        queues
            .entry(job.network_id.clone())
            .or_insert_with(VecDeque::new)
            .push_back(job);

        Ok(())
    }

    /// Dequeue the next inference job for a network (FIFO)
    pub fn dequeue_job(&self, network_id: &str) -> ApiResult<Option<DistributedInferenceJob>> {
        let mut queues = self.job_queues.write().map_err(|_| {
            ApiError::Internal("Failed to acquire job_queues write lock".to_string())
        })?;

        Ok(queues.get_mut(network_id).and_then(|q| q.pop_front()))
    }

    /// Get the number of pending jobs for a network
    pub fn pending_job_count(&self, network_id: &str) -> ApiResult<usize> {
        let queues = self.job_queues.read().map_err(|_| {
            ApiError::Internal("Failed to acquire job_queues read lock".to_string())
        })?;

        Ok(queues.get(network_id).map(|q| q.len()).unwrap_or(0))
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
