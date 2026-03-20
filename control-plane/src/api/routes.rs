use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use tracing::{info, instrument};

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    CreateNetworkRequest, CreateNetworkResponse, HeartbeatRequest, HeartbeatResponse,
    ListNetworksResponse, RegisterDeviceRequest, RegisterDeviceResponse, RingPositionInfo,
    ShardInfo,
};
use crate::services::{device_service, network_service};
use crate::services::ring_manager::Worker;
use crate::state::AppState;

/// Health check endpoint
#[instrument]
pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "healthy")
}

/// Register a new device
#[instrument(skip(state, req))]
pub async fn register_device(
    State(state): State<AppState>,
    Json(req): Json<RegisterDeviceRequest>,
) -> ApiResult<Json<RegisterDeviceResponse>> {
    // Extract ring join info before moving req
    let device_id = req.device_id.clone();
    let network_id = req.network_id.clone();
    let contributed_memory = req.contributed_memory;

    // Execute blocking database operation in thread pool
    let db = state.db.clone();
    let keypair = state.keypair.clone();
    let relay_addresses = (*state.relay_addresses).clone();

    let result = tokio::task::spawn_blocking(move || {
        device_service::register_device(
            &db,
            &keypair,
            req.device_id,
            req.network_id,
            req.name,
            req.public_key,
            req.capabilities,
            relay_addresses,
        )
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    // If contributed_memory is provided, automatically join the ring
    let ring_position = if let Some(memory) = contributed_memory {
        // Get or create ring manager for this network
        let ring_manager = state.get_ring_manager(&network_id)?;

        // Create worker from registration
        let worker = Worker {
            device_id: device_id.clone(),
            network_id: network_id.clone(),
            contributed_memory: memory,
            ring_position: None,
            status: "online".to_string(),
        };

        // Join the ring
        let position = tokio::task::spawn_blocking(move || ring_manager.add_worker(worker))
            .await
            .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

        info!(
            device_id = %device_id,
            network_id = %network_id,
            ring_position = position.position,
            "Device automatically joined ring after registration"
        );

        Some(RingPositionInfo {
            position: position.position,
            shard: ShardInfo {
                model_id: position.shard.model_id,
                column_start: position.shard.column_range.0,
                column_end: position.shard.column_range.1,
                estimated_memory: position.shard.estimated_memory,
            },
            left_neighbor: position.left_neighbor,
            right_neighbor: position.right_neighbor,
        })
    } else {
        None
    };

    Ok(Json(RegisterDeviceResponse {
        success: true,
        certificate: Some(result.0),
        relay_addresses: result.1,
        message: Some("Device registered successfully".to_string()),
        ring_position,
    }))
}

#[instrument(skip(state, req))]
pub async fn create_network(
    State(state): State<AppState>,
    Json(req): Json<CreateNetworkRequest>,
) -> ApiResult<Json<CreateNetworkResponse>> {
    let db = state.db.clone();
    let network = tokio::task::spawn_blocking(move || {
        network_service::create_network(
            &db,
            req.network_id,
            req.name,
            req.owner_user_id,
            req.settings,
        )
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(CreateNetworkResponse {
        success: true,
        network,
    }))
}

#[instrument(skip(state))]
pub async fn list_networks(
    State(state): State<AppState>,
) -> ApiResult<Json<ListNetworksResponse>> {
    let db = state.db.clone();
    let networks = tokio::task::spawn_blocking(move || network_service::list_networks(&db))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(ListNetworksResponse {
        success: true,
        networks,
    }))
}

/// Update device heartbeat
#[instrument(skip(state))]
pub async fn heartbeat(
    State(state): State<AppState>,
    Path(device_id): Path<String>,
    Json(_req): Json<HeartbeatRequest>,
) -> ApiResult<Json<HeartbeatResponse>> {
    // Execute blocking database operation in thread pool
    let db = state.db.clone();

    let last_seen = tokio::task::spawn_blocking(move || {
        device_service::update_heartbeat(&db, device_id)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(HeartbeatResponse {
        success: true,
        last_seen,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::network_service;
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

    #[tokio::test]
    async fn test_register_device_handler() {
        // Test using the handler function directly instead of full HTTP stack
        let db = crate::db::create_test_db();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            None,
        )
        .unwrap();

        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::with_relay_addresses(
            db,
            keypair,
            vec!["/dns4/relay.mesh.example/tcp/4001".to_string()],
        );

        let request = RegisterDeviceRequest {
            device_id: "test-device-1".to_string(),
            network_id: "test-network".to_string(),
            name: "Test Device".to_string(),
            public_key: vec![42u8; 32],
            capabilities: test_capabilities(),
            contributed_memory: None,
        };

        // Call handler directly
        let result = register_device(
            axum::extract::State(state),
            axum::Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.success);
        assert!(response.certificate.is_some());
        assert!(!response.relay_addresses.is_empty());
    }

    #[tokio::test]
    async fn test_heartbeat_handler() {
        // Setup database with a registered device
        let db = crate::db::create_test_db();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            None,
        )
        .unwrap();

        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());

        // Register a device first
        let device_id = "test-device-2";
        device_service::register_device(
            &db,
            &keypair,
            device_id.to_string(),
            "test-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
            test_capabilities(),
            vec!["/ip4/127.0.0.1/tcp/4001".to_string()],
        )
        .unwrap();

        let state = AppState::with_relay_addresses(
            db,
            keypair,
            vec!["/dns4/relay.mesh.example/tcp/4001".to_string()],
        );

        // Send heartbeat
        let result = heartbeat(
            axum::extract::State(state),
            axum::extract::Path(device_id.to_string()),
            axum::Json(HeartbeatRequest {}),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.success);
        assert!(!response.last_seen.is_empty());
    }

    #[tokio::test]
    async fn test_heartbeat_nonexistent_device() {
        let db = crate::db::create_test_db();

        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::with_relay_addresses(
            db,
            keypair,
            vec!["/dns4/relay.mesh.example/tcp/4001".to_string()],
        );

        let result = heartbeat(
            axum::extract::State(state),
            axum::extract::Path("nonexistent-device".to_string()),
            axum::Json(HeartbeatRequest {}),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_health_check() {
        let _response = health_check().await;
        // Health check returns a tuple of (StatusCode, &str)
        // We just verify it completes without error
    }

    #[tokio::test]
    async fn test_register_device_with_ring_join() {
        // Test registration with automatic ring joining
        let db = crate::db::create_test_db();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            None,
        )
        .unwrap();

        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::with_relay_addresses(
            db,
            keypair,
            vec!["/dns4/relay.mesh.example/tcp/4001".to_string()],
        );

        let request = RegisterDeviceRequest {
            device_id: "test-device-ring".to_string(),
            network_id: "test-network".to_string(),
            name: "Test Device".to_string(),
            public_key: vec![43u8; 32], // Different key to avoid conflicts
            capabilities: test_capabilities(),
            contributed_memory: Some(8_000_000_000),
        };

        // Call handler directly
        let result = register_device(
            axum::extract::State(state),
            axum::Json(request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.success);
        assert!(response.certificate.is_some());
        assert!(!response.relay_addresses.is_empty());

        // Should have ring position info
        assert!(response.ring_position.is_some());
        let ring_pos = response.ring_position.unwrap();
        assert_eq!(ring_pos.position, 0); // First worker
        assert_eq!(ring_pos.left_neighbor, "test-device-ring");
        assert_eq!(ring_pos.right_neighbor, "test-device-ring");
    }

    #[tokio::test]
    async fn test_create_and_list_networks_handlers() {
        let db = crate::db::create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db, keypair);

        let create_result = create_network(
            axum::extract::State(state.clone()),
            axum::Json(CreateNetworkRequest {
                network_id: "test-network".to_string(),
                name: "Test Network".to_string(),
                owner_user_id: "owner-1".to_string(),
                settings: Some(serde_json::json!({"region": "us-east-1"})),
            }),
        )
        .await;

        assert!(create_result.is_ok());
        assert_eq!(create_result.unwrap().0.network.network_id, "test-network");

        let list_result = list_networks(axum::extract::State(state)).await;
        assert!(list_result.is_ok());

        let response = list_result.unwrap().0;
        assert!(response.success);
        assert_eq!(response.networks.len(), 1);
        assert_eq!(response.networks[0].network_id, "test-network");
    }
}
