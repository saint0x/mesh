use axum::{
    extract::{Path, Query, State},
    Json,
};
use rusqlite::OptionalExtension;
use serde::Deserialize;
use tracing::instrument;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    CreateHandoffRequest, CreateHandoffResponse, HandoffInfo, ListHandoffsResponse,
    RegisterCallbackRequest, RegisterCallbackResponse, RingJoinRequest, RingJoinResponse,
    RingLeaveResponse, RingTopologyResponse, ShardInfo, TopologyVersionRequest,
    TopologyVersionResponse, UpdateHandoffRequest, WorkerInfo,
};
use crate::services::ring_manager::Worker;
use crate::services::topology_notifier::HandoffStatus;
use crate::state::AppState;

/// Query parameters for topology endpoint
#[derive(Debug, Deserialize)]
pub struct TopologyQuery {
    /// Network ID to query topology for
    pub network_id: String,
}

/// Join the ring topology
///
/// POST /api/ring/join
/// Request: { device_id, network_id, contributed_memory }
/// Response: { position, shard, left_neighbor, right_neighbor }
#[instrument(skip(state))]
pub async fn join_ring(
    State(state): State<AppState>,
    Json(req): Json<RingJoinRequest>,
) -> ApiResult<Json<RingJoinResponse>> {
    // Validate request
    if req.device_id.is_empty() {
        return Err(ApiError::BadRequest("device_id cannot be empty".to_string()));
    }
    if req.network_id.is_empty() {
        return Err(ApiError::BadRequest("network_id cannot be empty".to_string()));
    }

    // Get or create ring manager for this network
    let ring_manager = state.get_ring_manager(&req.network_id)?;

    // Create worker from request
    let worker = Worker {
        device_id: req.device_id.clone(),
        network_id: req.network_id.clone(),
        contributed_memory: req.contributed_memory,
        ring_position: None,
        status: "online".to_string(),
    };

    // Execute blocking database operation in thread pool
    let position = tokio::task::spawn_blocking(move || ring_manager.add_worker(worker))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    // Send topology notification for worker join
    let notifier = state.topology_notifier.clone();
    let notification = notifier.worker_joined_notification(
        &req.network_id,
        &req.device_id,
        position.position,
        position.shard.column_range,
        &position.left_neighbor,
        &position.right_neighbor,
    );

    // Send notification asynchronously (don't block response)
    tokio::spawn(async move {
        let _ = notifier.notify_network(notification).await;
    });

    Ok(Json(RingJoinResponse {
        success: true,
        position: position.position,
        shard: ShardInfo {
            model_id: position.shard.model_id,
            column_start: position.shard.column_range.0,
            column_end: position.shard.column_range.1,
            estimated_memory: position.shard.estimated_memory,
        },
        left_neighbor: position.left_neighbor,
        right_neighbor: position.right_neighbor,
    }))
}

/// Get ring topology for a network
///
/// GET /api/ring/topology?network_id=xxx
/// Response: { workers: [{ device_id, position, shard, neighbors }], ring_stable: bool }
#[instrument(skip(state))]
pub async fn get_topology(
    State(state): State<AppState>,
    Query(query): Query<TopologyQuery>,
) -> ApiResult<Json<RingTopologyResponse>> {
    // Validate request
    if query.network_id.is_empty() {
        return Err(ApiError::BadRequest("network_id cannot be empty".to_string()));
    }

    // Check if network exists
    let db = state.db.clone();
    let network_id = query.network_id.clone();

    let network_exists = tokio::task::spawn_blocking(move || {
        let conn = db.get_conn()?;
        let exists: Option<String> = conn
            .query_row(
                "SELECT network_id FROM networks WHERE network_id = ?",
                rusqlite::params![&network_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        Ok::<_, ApiError>(exists.is_some())
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    if !network_exists {
        return Err(ApiError::NotFound(format!(
            "Network {} not found",
            query.network_id
        )));
    }

    // Get ring manager for this network
    let ring_manager = state.get_ring_manager(&query.network_id)?;
    let network_id = query.network_id.clone();

    // Execute blocking database operation in thread pool
    let topology = tokio::task::spawn_blocking(move || ring_manager.get_topology(&network_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    // Convert to API response types
    let workers: Vec<WorkerInfo> = topology
        .workers
        .into_iter()
        .map(|w| WorkerInfo {
            device_id: w.device_id,
            position: w.position,
            shard: ShardInfo {
                model_id: w.shard.model_id,
                column_start: w.shard.column_range.0,
                column_end: w.shard.column_range.1,
                estimated_memory: w.shard.estimated_memory,
            },
            left_neighbor: w.left_neighbor,
            right_neighbor: w.right_neighbor,
        })
        .collect();

    Ok(Json(RingTopologyResponse {
        workers,
        ring_stable: topology.ring_stable,
    }))
}

/// Leave the ring topology
///
/// DELETE /api/ring/leave/:device_id
/// Response: { success: bool }
#[instrument(skip(state))]
pub async fn leave_ring(
    State(state): State<AppState>,
    Path(device_id): Path<String>,
) -> ApiResult<Json<RingLeaveResponse>> {
    // Validate request
    if device_id.is_empty() {
        return Err(ApiError::BadRequest("device_id cannot be empty".to_string()));
    }

    // First, find which network this device belongs to
    let db = state.db.clone();
    let device_id_clone = device_id.clone();

    let network_id: Option<String> = tokio::task::spawn_blocking(move || {
        let conn = db.get_conn()?;
        let network: Option<String> = conn
            .query_row(
                "SELECT network_id FROM devices WHERE device_id = ?",
                rusqlite::params![&device_id_clone],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        Ok::<_, ApiError>(network)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    let network_id = network_id.ok_or_else(|| {
        ApiError::NotFound(format!("Device {} not found", device_id))
    })?;

    // Get ring manager for this network
    let ring_manager = state.get_ring_manager(&network_id)?;

    // Execute blocking database operation in thread pool
    let device_id_clone = device_id.clone();
    tokio::task::spawn_blocking(move || ring_manager.handle_worker_failure(device_id_clone))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    // Send topology notification for worker leave
    let notifier = state.topology_notifier.clone();
    let notification = notifier.worker_left_notification(&network_id, &device_id);

    // Send notification asynchronously (don't block response)
    tokio::spawn(async move {
        let _ = notifier.notify_network(notification).await;
    });

    Ok(Json(RingLeaveResponse { success: true }))
}

// ==================== Handoff Management Endpoints ====================

/// Create a new shard handoff
///
/// POST /api/ring/handoff
#[instrument(skip(state))]
pub async fn create_handoff(
    State(state): State<AppState>,
    Json(req): Json<CreateHandoffRequest>,
) -> ApiResult<Json<CreateHandoffResponse>> {
    // Validate request
    if req.source_device.is_empty() || req.target_device.is_empty() {
        return Err(ApiError::BadRequest(
            "source_device and target_device are required".to_string(),
        ));
    }
    if req.column_start >= req.column_end {
        return Err(ApiError::BadRequest(
            "column_start must be less than column_end".to_string(),
        ));
    }

    let handoff = state.topology_notifier.create_handoff(
        req.network_id,
        req.source_device,
        req.target_device,
        (req.column_start, req.column_end),
        req.model_id,
    )?;

    Ok(Json(CreateHandoffResponse {
        success: true,
        handoff_id: handoff.handoff_id,
        status: format!("{:?}", handoff.status).to_lowercase(),
    }))
}

/// Get handoff status
///
/// GET /api/ring/handoff/:handoff_id
#[instrument(skip(state))]
pub async fn get_handoff(
    State(state): State<AppState>,
    Path(handoff_id): Path<String>,
) -> ApiResult<Json<HandoffInfo>> {
    let handoff = state
        .topology_notifier
        .get_handoff(&handoff_id)?
        .ok_or_else(|| ApiError::NotFound(format!("Handoff {} not found", handoff_id)))?;

    let progress = handoff.progress();
    Ok(Json(HandoffInfo {
        handoff_id: handoff.handoff_id,
        network_id: handoff.network_id,
        source_device: handoff.source_device,
        target_device: handoff.target_device,
        column_start: handoff.column_range.0,
        column_end: handoff.column_range.1,
        model_id: handoff.model_id,
        status: format!("{:?}", handoff.status).to_lowercase(),
        bytes_transferred: handoff.bytes_transferred,
        total_bytes: handoff.total_bytes,
        progress,
        started_at: handoff.started_at,
        completed_at: handoff.completed_at,
        error: handoff.error,
    }))
}

/// Update handoff status
///
/// PATCH /api/ring/handoff/:handoff_id
#[instrument(skip(state))]
pub async fn update_handoff(
    State(state): State<AppState>,
    Path(handoff_id): Path<String>,
    Json(req): Json<UpdateHandoffRequest>,
) -> ApiResult<Json<HandoffInfo>> {
    // Parse status
    let status = match req.status.to_lowercase().as_str() {
        "pending" => HandoffStatus::Pending,
        "preparing" => HandoffStatus::Preparing,
        "transferring" => HandoffStatus::Transferring,
        "verifying" => HandoffStatus::Verifying,
        "completed" => HandoffStatus::Completed,
        "failed" => HandoffStatus::Failed,
        "cancelled" => HandoffStatus::Cancelled,
        _ => return Err(ApiError::BadRequest(format!("Invalid status: {}", req.status))),
    };

    // Update handoff with bytes if provided
    if let Some(total) = req.total_bytes {
        // Get handoff and update total_bytes
        if let Some(mut handoff) = state.topology_notifier.get_handoff(&handoff_id)? {
            handoff.total_bytes = total;
        }
    }

    state
        .topology_notifier
        .update_handoff_status(&handoff_id, status, req.bytes_transferred, req.error)?;

    // Return updated handoff
    get_handoff(State(state), Path(handoff_id)).await
}

/// List active handoffs for a network
///
/// GET /api/ring/handoffs?network_id=xxx
#[derive(Debug, Deserialize)]
pub struct HandoffsQuery {
    pub network_id: String,
}

#[instrument(skip(state))]
pub async fn list_handoffs(
    State(state): State<AppState>,
    Query(query): Query<HandoffsQuery>,
) -> ApiResult<Json<ListHandoffsResponse>> {
    let handoffs = state
        .topology_notifier
        .list_active_handoffs(&query.network_id)?;

    let handoff_infos: Vec<HandoffInfo> = handoffs
        .into_iter()
        .map(|h| {
            let progress = h.progress();
            HandoffInfo {
                handoff_id: h.handoff_id,
                network_id: h.network_id,
                source_device: h.source_device,
                target_device: h.target_device,
                column_start: h.column_range.0,
                column_end: h.column_range.1,
                model_id: h.model_id,
                status: format!("{:?}", h.status).to_lowercase(),
                bytes_transferred: h.bytes_transferred,
                total_bytes: h.total_bytes,
                progress,
                started_at: h.started_at,
                completed_at: h.completed_at,
                error: h.error,
            }
        })
        .collect();

    Ok(Json(ListHandoffsResponse {
        handoffs: handoff_infos,
    }))
}

/// Cancel a handoff
///
/// DELETE /api/ring/handoff/:handoff_id
#[instrument(skip(state))]
pub async fn cancel_handoff(
    State(state): State<AppState>,
    Path(handoff_id): Path<String>,
) -> ApiResult<Json<HandoffInfo>> {
    state.topology_notifier.cancel_handoff(&handoff_id)?;
    get_handoff(State(state), Path(handoff_id)).await
}

// ==================== Worker Callback Endpoints ====================

/// Register worker callback for topology notifications
///
/// POST /api/ring/callback
#[instrument(skip(state))]
pub async fn register_callback(
    State(state): State<AppState>,
    Json(req): Json<RegisterCallbackRequest>,
) -> ApiResult<Json<RegisterCallbackResponse>> {
    if req.device_id.is_empty() {
        return Err(ApiError::BadRequest("device_id is required".to_string()));
    }

    state
        .topology_notifier
        .register_callback(req.device_id, req.callback_url)?;

    Ok(Json(RegisterCallbackResponse { success: true }))
}

/// Unregister worker callback
///
/// DELETE /api/ring/callback/:device_id
#[instrument(skip(state))]
pub async fn unregister_callback(
    State(state): State<AppState>,
    Path(device_id): Path<String>,
) -> ApiResult<Json<RegisterCallbackResponse>> {
    state.topology_notifier.unregister_callback(&device_id)?;
    Ok(Json(RegisterCallbackResponse { success: true }))
}

/// Check topology version (for polling)
///
/// POST /api/ring/version
#[instrument(skip(state))]
pub async fn check_topology_version(
    State(state): State<AppState>,
    Json(req): Json<TopologyVersionRequest>,
) -> ApiResult<Json<TopologyVersionResponse>> {
    let current = state.topology_notifier.get_version(&req.network_id);
    let has_updates = current > req.since_version;

    Ok(Json(TopologyVersionResponse {
        current_version: current,
        has_updates,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::device_service;
    use std::sync::Arc;

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            cpu_cores: 8,
            ram_mb: 16384,
            os: "macos".into(),
            arch: "aarch64".into(),
            has_gpu: false,
            tier: Tier::Tier2,
        }
    }

    fn register_test_device(db: &crate::db::Database, device_id: &str, network_id: &str) {
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        // Generate unique public key based on device_id hash
        let mut public_key = [0u8; 32];
        let hash = device_id.as_bytes();
        for (i, byte) in hash.iter().cycle().take(32).enumerate() {
            public_key[i] = *byte;
        }
        device_service::register_device(
            db,
            &keypair,
            device_id.to_string(),
            network_id.to_string(),
            "Test Device".to_string(),
            public_key.to_vec(),
            test_capabilities(),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_join_ring_handler() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let device_id = "test-device-1";
        let network_id = "test-network";

        // Register device first
        register_test_device(&db, device_id, network_id);

        let request = RingJoinRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
        };

        let result = join_ring(
            axum::extract::State(state),
            axum::Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.success);
        assert_eq!(response.position, 0);
        assert_eq!(response.left_neighbor, device_id);
        assert_eq!(response.right_neighbor, device_id);
    }

    #[tokio::test]
    async fn test_join_ring_nonexistent_device() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db, keypair);

        let request = RingJoinRequest {
            device_id: "nonexistent".to_string(),
            network_id: "test-network".to_string(),
            contributed_memory: 8_000_000_000,
        };

        let result = join_ring(
            axum::extract::State(state),
            axum::Json(request),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_get_topology_handler() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let network_id = "test-network";

        // Register and join 2 devices
        for i in 0..2 {
            let device_id = format!("device-{}", i);
            register_test_device(&db, &device_id, network_id);

            let join_request = RingJoinRequest {
                device_id,
                network_id: network_id.to_string(),
                contributed_memory: 8_000_000_000,
            };

            let _ = join_ring(
                axum::extract::State(state.clone()),
                axum::Json(join_request),
            )
            .await
            .unwrap();
        }

        // Query topology
        let result = get_topology(
            axum::extract::State(state),
            axum::extract::Query(TopologyQuery {
                network_id: network_id.to_string(),
            }),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.ring_stable);
        assert_eq!(response.workers.len(), 2);
    }

    #[tokio::test]
    async fn test_get_topology_nonexistent_network() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db, keypair);

        let result = get_topology(
            axum::extract::State(state),
            axum::extract::Query(TopologyQuery {
                network_id: "nonexistent".to_string(),
            }),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_leave_ring_handler() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let device_id = "device-to-leave";
        let network_id = "test-network";

        // Register and join device
        register_test_device(&db, device_id, network_id);

        let join_request = RingJoinRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
        };

        let _ = join_ring(
            axum::extract::State(state.clone()),
            axum::Json(join_request),
        )
        .await
        .unwrap();

        // Leave ring
        let result = leave_ring(
            axum::extract::State(state.clone()),
            axum::extract::Path(device_id.to_string()),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.success);

        // Verify device is no longer in ring
        let topology = get_topology(
            axum::extract::State(state),
            axum::extract::Query(TopologyQuery {
                network_id: network_id.to_string(),
            }),
        )
        .await
        .unwrap()
        .0;

        assert!(topology.workers.is_empty());
        assert!(!topology.ring_stable);
    }

    #[tokio::test]
    async fn test_leave_ring_nonexistent_device() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db, keypair);

        let result = leave_ring(
            axum::extract::State(state),
            axum::extract::Path("nonexistent".to_string()),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    #[tokio::test]
    async fn test_concurrent_ring_joins() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let network_id = "test-network";

        // Register 5 devices
        for i in 0..5 {
            register_test_device(&db, &format!("device-{}", i), network_id);
        }

        // Join concurrently
        let mut handles = Vec::new();
        for i in 0..5 {
            let state = state.clone();
            let network_id = network_id.to_string();
            let handle = tokio::spawn(async move {
                let request = RingJoinRequest {
                    device_id: format!("device-{}", i),
                    network_id,
                    contributed_memory: 8_000_000_000,
                };
                join_ring(axum::extract::State(state), axum::Json(request)).await
            });
            handles.push(handle);
        }

        // Wait for all joins
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .collect();

        // All should complete (some may succeed, some may fail due to race conditions)
        let succeeded: Vec<_> = results
            .into_iter()
            .filter_map(|r| r.ok())
            .filter_map(|r| r.ok())
            .collect();

        // At least some should succeed
        assert!(!succeeded.is_empty());

        // Verify topology
        let topology = get_topology(
            axum::extract::State(state),
            axum::extract::Query(TopologyQuery {
                network_id: network_id.to_string(),
            }),
        )
        .await
        .unwrap()
        .0;

        // Should have workers in the ring
        assert!(!topology.workers.is_empty());
    }
}
