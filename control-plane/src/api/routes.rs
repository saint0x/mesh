use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    HeartbeatRequest, HeartbeatResponse, RegisterDeviceRequest, RegisterDeviceResponse,
};
use crate::services::device_service;
use crate::state::AppState;
use tracing::instrument;

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
    // Execute blocking database operation in thread pool
    let db = state.db.clone();
    let keypair = state.keypair.clone();

    let result = tokio::task::spawn_blocking(move || {
        device_service::register_device(
            &db,
            &keypair,
            req.device_id,
            req.network_id,
            req.name,
            req.public_key,
            req.capabilities,
        )
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(RegisterDeviceResponse {
        success: true,
        certificate: Some(result.0),
        relay_addresses: result.1,
        message: Some("Device registered successfully".to_string()),
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

        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db, keypair);

        let request = RegisterDeviceRequest {
            device_id: "test-device-1".to_string(),
            network_id: "test-network".to_string(),
            name: "Test Device".to_string(),
            public_key: vec![42u8; 32],
            capabilities: test_capabilities(),
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
        )
        .unwrap();

        let state = AppState::new(db, keypair);

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
        let state = AppState::new(db, keypair);

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
        let response = health_check().await;
        // Health check returns a tuple of (StatusCode, &str)
        // We just verify it completes without error
    }
}
