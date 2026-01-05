use axum::{
    extract::{Path, Query, State},
    Json,
};
use rusqlite::OptionalExtension;
use serde::Deserialize;
use tracing::instrument;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    RingJoinRequest, RingJoinResponse, RingLeaveResponse, RingTopologyResponse, ShardInfo,
    WorkerInfo,
};
use crate::services::ring_manager::Worker;
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
    tokio::task::spawn_blocking(move || ring_manager.handle_worker_failure(device_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(RingLeaveResponse { success: true }))
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
