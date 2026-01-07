//! Topology Notification System
//!
//! This module provides worker notification capabilities for topology changes.
//! Workers are notified when:
//! - A new worker joins the ring
//! - A worker leaves/fails
//! - Shard assignments change
//! - Ring needs rebalancing
//!
//! ## Notification Methods
//!
//! 1. **HTTP Webhook**: Push notifications to worker's callback URL
//! 2. **Polling with Version**: Workers poll and check version number
//!
//! ## Handoff Protocol
//!
//! When topology changes require shard redistribution:
//! 1. Control plane marks handoff as PENDING
//! 2. Source worker is notified to PREPARE handoff
//! 3. Target worker is notified to RECEIVE handoff
//! 4. Source streams data to target
//! 5. Target confirms receipt
//! 6. Control plane marks handoff as COMPLETE

use crate::api::error::{ApiError, ApiResult};
use crate::db::Database;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Topology change event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TopologyEventType {
    /// New worker joined the ring
    WorkerJoined,
    /// Worker left or failed
    WorkerLeft,
    /// Shard assignment changed
    ShardReassigned,
    /// Ring topology reconfigured
    RingReconfigured,
    /// Handoff initiated
    HandoffStarted,
    /// Handoff completed
    HandoffCompleted,
}

/// Topology change notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyNotification {
    /// Unique notification ID
    pub notification_id: String,
    /// Network this applies to
    pub network_id: String,
    /// Type of change
    pub event_type: TopologyEventType,
    /// Current topology version (monotonically increasing)
    pub topology_version: u64,
    /// Affected device (if applicable)
    pub affected_device: Option<String>,
    /// New position (for join events)
    pub new_position: Option<u32>,
    /// Shard range (start, end)
    pub shard_range: Option<(u32, u32)>,
    /// Left neighbor device ID
    pub left_neighbor: Option<String>,
    /// Right neighbor device ID
    pub right_neighbor: Option<String>,
    /// Timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TopologyNotification {
    pub fn new(network_id: String, event_type: TopologyEventType, version: u64) -> Self {
        Self {
            notification_id: uuid::Uuid::new_v4().to_string(),
            network_id,
            event_type,
            topology_version: version,
            affected_device: None,
            new_position: None,
            shard_range: None,
            left_neighbor: None,
            right_neighbor: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_device(mut self, device_id: String) -> Self {
        self.affected_device = Some(device_id);
        self
    }

    pub fn with_position(mut self, position: u32) -> Self {
        self.new_position = Some(position);
        self
    }

    pub fn with_shard(mut self, start: u32, end: u32) -> Self {
        self.shard_range = Some((start, end));
        self
    }

    pub fn with_neighbors(mut self, left: String, right: String) -> Self {
        self.left_neighbor = Some(left);
        self.right_neighbor = Some(right);
        self
    }
}

/// Shard handoff status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HandoffStatus {
    /// Handoff is pending
    Pending,
    /// Source is preparing data
    Preparing,
    /// Data is being transferred
    Transferring,
    /// Target is verifying
    Verifying,
    /// Handoff completed successfully
    Completed,
    /// Handoff failed
    Failed,
    /// Handoff was cancelled
    Cancelled,
}

/// Shard handoff record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHandoff {
    /// Unique handoff ID
    pub handoff_id: String,
    /// Network ID
    pub network_id: String,
    /// Source device (giving up shard)
    pub source_device: String,
    /// Target device (receiving shard)
    pub target_device: String,
    /// Column range being transferred
    pub column_range: (u32, u32),
    /// Model ID
    pub model_id: String,
    /// Current status
    pub status: HandoffStatus,
    /// Bytes transferred so far
    pub bytes_transferred: u64,
    /// Total bytes to transfer
    pub total_bytes: u64,
    /// Start timestamp
    pub started_at: u64,
    /// Completion timestamp
    pub completed_at: Option<u64>,
    /// Error message if failed
    pub error: Option<String>,
}

impl ShardHandoff {
    pub fn new(
        network_id: String,
        source_device: String,
        target_device: String,
        column_range: (u32, u32),
        model_id: String,
    ) -> Self {
        Self {
            handoff_id: uuid::Uuid::new_v4().to_string(),
            network_id,
            source_device,
            target_device,
            column_range,
            model_id,
            status: HandoffStatus::Pending,
            bytes_transferred: 0,
            total_bytes: 0,
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            completed_at: None,
            error: None,
        }
    }

    /// Progress as percentage (0.0 - 1.0)
    pub fn progress(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.bytes_transferred as f32 / self.total_bytes as f32
        }
    }
}

/// Worker callback registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCallback {
    /// Device ID
    pub device_id: String,
    /// Callback URL for HTTP webhook
    pub callback_url: Option<String>,
    /// Last topology version seen
    pub last_seen_version: u64,
    /// Registration timestamp
    pub registered_at: u64,
    /// Last notification sent
    pub last_notified_at: Option<u64>,
}

/// Topology Notifier service
pub struct TopologyNotifier {
    db: Arc<Database>,
    /// Current topology version per network
    versions: RwLock<HashMap<String, u64>>,
    /// Registered worker callbacks
    callbacks: RwLock<HashMap<String, WorkerCallback>>,
    /// Active handoffs
    handoffs: RwLock<HashMap<String, ShardHandoff>>,
    /// HTTP client for webhooks
    http_client: reqwest::Client,
    /// Notification channel for async processing
    notification_tx: Option<mpsc::Sender<TopologyNotification>>,
}

impl TopologyNotifier {
    /// Create a new TopologyNotifier
    pub fn new(db: Arc<Database>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            db,
            versions: RwLock::new(HashMap::new()),
            callbacks: RwLock::new(HashMap::new()),
            handoffs: RwLock::new(HashMap::new()),
            http_client,
            notification_tx: None,
        }
    }

    /// Initialize notification processing
    pub fn with_channel(mut self, tx: mpsc::Sender<TopologyNotification>) -> Self {
        self.notification_tx = Some(tx);
        self
    }

    /// Get current topology version for a network
    pub fn get_version(&self, network_id: &str) -> u64 {
        self.versions
            .read()
            .ok()
            .and_then(|v| v.get(network_id).copied())
            .unwrap_or(0)
    }

    /// Increment topology version and return new version
    pub fn increment_version(&self, network_id: &str) -> u64 {
        let mut versions = self.versions.write().expect("Lock poisoned");
        let version = versions.entry(network_id.to_string()).or_insert(0);
        *version += 1;
        *version
    }

    /// Register a worker callback
    pub fn register_callback(
        &self,
        device_id: String,
        callback_url: Option<String>,
    ) -> ApiResult<()> {
        let callback = WorkerCallback {
            device_id: device_id.clone(),
            callback_url,
            last_seen_version: 0,
            registered_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            last_notified_at: None,
        };

        let mut callbacks = self
            .callbacks
            .write()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;
        callbacks.insert(device_id.clone(), callback);

        info!(device_id = %device_id, "Worker callback registered");
        Ok(())
    }

    /// Unregister a worker callback
    pub fn unregister_callback(&self, device_id: &str) -> ApiResult<()> {
        let mut callbacks = self
            .callbacks
            .write()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;
        callbacks.remove(device_id);

        info!(device_id = %device_id, "Worker callback unregistered");
        Ok(())
    }

    /// Notify all workers in a network of a topology change
    pub async fn notify_network(
        &self,
        notification: TopologyNotification,
    ) -> ApiResult<Vec<String>> {
        let callbacks = self
            .callbacks
            .read()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?
            .clone();

        let mut notified = Vec::new();
        let mut failed = Vec::new();

        for (device_id, callback) in callbacks.iter() {
            if let Some(url) = &callback.callback_url {
                match self.send_webhook(url, &notification).await {
                    Ok(()) => {
                        notified.push(device_id.clone());
                        debug!(device_id = %device_id, "Notification sent");
                    }
                    Err(e) => {
                        warn!(device_id = %device_id, error = %e, "Failed to send notification");
                        failed.push(device_id.clone());
                    }
                }
            }
        }

        // Also send to channel if configured
        if let Some(tx) = &self.notification_tx {
            let _ = tx.send(notification.clone()).await;
        }

        info!(
            notified = notified.len(),
            failed = failed.len(),
            event_type = ?notification.event_type,
            "Network notification sent"
        );

        Ok(notified)
    }

    /// Send HTTP webhook notification
    async fn send_webhook(&self, url: &str, notification: &TopologyNotification) -> ApiResult<()> {
        let response = self
            .http_client
            .post(url)
            .json(notification)
            .send()
            .await
            .map_err(|e| ApiError::Internal(format!("Webhook failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(ApiError::Internal(format!(
                "Webhook returned {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Notify a specific worker
    pub async fn notify_worker(
        &self,
        device_id: &str,
        notification: TopologyNotification,
    ) -> ApiResult<()> {
        let callbacks = self
            .callbacks
            .read()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;

        if let Some(callback) = callbacks.get(device_id) {
            if let Some(url) = &callback.callback_url {
                self.send_webhook(url, &notification).await?;
            }
        }

        Ok(())
    }

    // ==================== Handoff Management ====================

    /// Create a new shard handoff
    pub fn create_handoff(
        &self,
        network_id: String,
        source_device: String,
        target_device: String,
        column_range: (u32, u32),
        model_id: String,
    ) -> ApiResult<ShardHandoff> {
        let handoff = ShardHandoff::new(
            network_id,
            source_device,
            target_device,
            column_range,
            model_id,
        );

        let mut handoffs = self
            .handoffs
            .write()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;

        let handoff_id = handoff.handoff_id.clone();
        handoffs.insert(handoff_id.clone(), handoff.clone());

        info!(
            handoff_id = %handoff_id,
            source = %handoff.source_device,
            target = %handoff.target_device,
            columns = ?handoff.column_range,
            "Handoff created"
        );

        Ok(handoff)
    }

    /// Update handoff status
    pub fn update_handoff_status(
        &self,
        handoff_id: &str,
        status: HandoffStatus,
        bytes_transferred: Option<u64>,
        error: Option<String>,
    ) -> ApiResult<()> {
        let mut handoffs = self
            .handoffs
            .write()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;

        let handoff = handoffs
            .get_mut(handoff_id)
            .ok_or_else(|| ApiError::NotFound(format!("Handoff {} not found", handoff_id)))?;

        handoff.status = status;
        if let Some(bytes) = bytes_transferred {
            handoff.bytes_transferred = bytes;
        }
        if error.is_some() {
            handoff.error = error;
        }
        if status == HandoffStatus::Completed || status == HandoffStatus::Failed {
            handoff.completed_at = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
        }

        info!(
            handoff_id = %handoff_id,
            status = ?status,
            progress = handoff.progress(),
            "Handoff status updated"
        );

        Ok(())
    }

    /// Get handoff by ID
    pub fn get_handoff(&self, handoff_id: &str) -> ApiResult<Option<ShardHandoff>> {
        let handoffs = self
            .handoffs
            .read()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;

        Ok(handoffs.get(handoff_id).cloned())
    }

    /// List active handoffs for a network
    pub fn list_active_handoffs(&self, network_id: &str) -> ApiResult<Vec<ShardHandoff>> {
        let handoffs = self
            .handoffs
            .read()
            .map_err(|_| ApiError::Internal("Lock poisoned".to_string()))?;

        let active: Vec<ShardHandoff> = handoffs
            .values()
            .filter(|h| {
                h.network_id == network_id
                    && !matches!(
                        h.status,
                        HandoffStatus::Completed | HandoffStatus::Failed | HandoffStatus::Cancelled
                    )
            })
            .cloned()
            .collect();

        Ok(active)
    }

    /// Cancel a handoff
    pub fn cancel_handoff(&self, handoff_id: &str) -> ApiResult<()> {
        self.update_handoff_status(
            handoff_id,
            HandoffStatus::Cancelled,
            None,
            Some("Cancelled by user".to_string()),
        )
    }

    // ==================== Worker Topology Polling ====================

    /// Get notifications since a specific version
    pub fn get_notifications_since(
        &self,
        network_id: &str,
        _since_version: u64,
    ) -> ApiResult<(u64, Vec<TopologyNotification>)> {
        let current_version = self.get_version(network_id);

        // In a full implementation, we'd store notifications in DB
        // For now, just return current version (worker should re-fetch topology)
        Ok((current_version, Vec::new()))
    }
}

/// Create topology change notifications for common events
impl TopologyNotifier {
    /// Create notification for worker join
    pub fn worker_joined_notification(
        &self,
        network_id: &str,
        device_id: &str,
        position: u32,
        shard_range: (u32, u32),
        left: &str,
        right: &str,
    ) -> TopologyNotification {
        let version = self.increment_version(network_id);
        TopologyNotification::new(
            network_id.to_string(),
            TopologyEventType::WorkerJoined,
            version,
        )
        .with_device(device_id.to_string())
        .with_position(position)
        .with_shard(shard_range.0, shard_range.1)
        .with_neighbors(left.to_string(), right.to_string())
    }

    /// Create notification for worker leave
    pub fn worker_left_notification(
        &self,
        network_id: &str,
        device_id: &str,
    ) -> TopologyNotification {
        let version = self.increment_version(network_id);
        TopologyNotification::new(
            network_id.to_string(),
            TopologyEventType::WorkerLeft,
            version,
        )
        .with_device(device_id.to_string())
    }

    /// Create notification for ring reconfiguration
    pub fn ring_reconfigured_notification(&self, network_id: &str) -> TopologyNotification {
        let version = self.increment_version(network_id);
        TopologyNotification::new(
            network_id.to_string(),
            TopologyEventType::RingReconfigured,
            version,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::create_test_db;

    fn create_test_notifier() -> TopologyNotifier {
        let db = create_test_db();
        TopologyNotifier::new(Arc::new(db))
    }

    #[test]
    fn test_version_increment() {
        let notifier = create_test_notifier();

        assert_eq!(notifier.get_version("test-network"), 0);

        let v1 = notifier.increment_version("test-network");
        assert_eq!(v1, 1);

        let v2 = notifier.increment_version("test-network");
        assert_eq!(v2, 2);

        // Different network starts at 0
        assert_eq!(notifier.get_version("other-network"), 0);
    }

    #[test]
    fn test_callback_registration() {
        let notifier = create_test_notifier();

        notifier
            .register_callback("device-1".to_string(), Some("http://localhost:8080".to_string()))
            .unwrap();

        let callbacks = notifier.callbacks.read().unwrap();
        assert!(callbacks.contains_key("device-1"));
        assert_eq!(
            callbacks.get("device-1").unwrap().callback_url,
            Some("http://localhost:8080".to_string())
        );
    }

    #[test]
    fn test_callback_unregistration() {
        let notifier = create_test_notifier();

        notifier
            .register_callback("device-1".to_string(), None)
            .unwrap();
        notifier.unregister_callback("device-1").unwrap();

        let callbacks = notifier.callbacks.read().unwrap();
        assert!(!callbacks.contains_key("device-1"));
    }

    #[test]
    fn test_handoff_lifecycle() {
        let notifier = create_test_notifier();

        // Create handoff
        let handoff = notifier
            .create_handoff(
                "test-network".to_string(),
                "source-device".to_string(),
                "target-device".to_string(),
                (0, 1024),
                "llama-70b".to_string(),
            )
            .unwrap();

        assert_eq!(handoff.status, HandoffStatus::Pending);

        // Update to transferring
        notifier
            .update_handoff_status(
                &handoff.handoff_id,
                HandoffStatus::Transferring,
                Some(500),
                None,
            )
            .unwrap();

        let updated = notifier.get_handoff(&handoff.handoff_id).unwrap().unwrap();
        assert_eq!(updated.status, HandoffStatus::Transferring);
        assert_eq!(updated.bytes_transferred, 500);

        // Complete
        notifier
            .update_handoff_status(&handoff.handoff_id, HandoffStatus::Completed, None, None)
            .unwrap();

        let completed = notifier.get_handoff(&handoff.handoff_id).unwrap().unwrap();
        assert_eq!(completed.status, HandoffStatus::Completed);
        assert!(completed.completed_at.is_some());
    }

    #[test]
    fn test_notification_creation() {
        let notifier = create_test_notifier();

        let notification = notifier.worker_joined_notification(
            "test-network",
            "device-1",
            0,
            (0, 4096),
            "device-3",
            "device-2",
        );

        assert_eq!(notification.event_type, TopologyEventType::WorkerJoined);
        assert_eq!(notification.topology_version, 1);
        assert_eq!(notification.affected_device, Some("device-1".to_string()));
        assert_eq!(notification.new_position, Some(0));
        assert_eq!(notification.shard_range, Some((0, 4096)));
    }

    #[test]
    fn test_active_handoffs_filter() {
        let notifier = create_test_notifier();

        // Create multiple handoffs
        let h1 = notifier
            .create_handoff(
                "test-network".to_string(),
                "s1".to_string(),
                "t1".to_string(),
                (0, 100),
                "model".to_string(),
            )
            .unwrap();

        let h2 = notifier
            .create_handoff(
                "test-network".to_string(),
                "s2".to_string(),
                "t2".to_string(),
                (100, 200),
                "model".to_string(),
            )
            .unwrap();

        // Complete one
        notifier
            .update_handoff_status(&h1.handoff_id, HandoffStatus::Completed, None, None)
            .unwrap();

        // List active
        let active = notifier.list_active_handoffs("test-network").unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].handoff_id, h2.handoff_id);
    }
}
