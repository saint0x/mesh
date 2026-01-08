//! Distributed inference API endpoints
//!
//! This module provides endpoints for submitting and managing distributed inference jobs.

use axum::{extract::State, Json};
use rusqlite::OptionalExtension;
use tracing::{info, instrument};
use uuid::Uuid;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{SubmitInferenceRequest, SubmitInferenceResponse};
use crate::state::AppState;

/// Submit a distributed inference job
///
/// POST /api/inference/submit
/// Request: { device_id, network_id, model_id, prompt, max_tokens, temperature, top_p }
/// Response: { success, job_id, completion, completion_tokens, execution_time_ms, error }
#[instrument(skip(state))]
pub async fn submit_inference(
    State(state): State<AppState>,
    Json(req): Json<SubmitInferenceRequest>,
) -> ApiResult<Json<SubmitInferenceResponse>> {
    // Validate request
    if req.device_id.is_empty() {
        return Err(ApiError::BadRequest("device_id cannot be empty".to_string()));
    }
    if req.network_id.is_empty() {
        return Err(ApiError::BadRequest("network_id cannot be empty".to_string()));
    }
    if req.model_id.is_empty() {
        return Err(ApiError::BadRequest("model_id cannot be empty".to_string()));
    }
    if req.prompt.is_empty() {
        return Err(ApiError::BadRequest("prompt cannot be empty".to_string()));
    }
    if req.max_tokens == 0 || req.max_tokens > 2048 {
        return Err(ApiError::BadRequest(
            "max_tokens must be between 1 and 2048".to_string(),
        ));
    }

    info!(
        device_id = %req.device_id,
        network_id = %req.network_id,
        model_id = %req.model_id,
        max_tokens = req.max_tokens,
        "Received inference request"
    );

    // Check that the ring exists and is stable
    let ring_manager = state.get_ring_manager(&req.network_id)?;
    let network_id = req.network_id.clone();

    let topology = tokio::task::spawn_blocking(move || ring_manager.get_topology(&network_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    if topology.workers.is_empty() {
        return Err(ApiError::BadRequest(
            "No workers in ring - cannot execute inference".to_string(),
        ));
    }

    if !topology.ring_stable {
        return Err(ApiError::BadRequest(
            "Ring is not stable - wait for topology to stabilize".to_string(),
        ));
    }

    // Verify device is registered in this network
    let db = state.db.clone();
    let device_id = req.device_id.clone();
    let network_id = req.network_id.clone();

    let device_exists = tokio::task::spawn_blocking(move || {
        let conn = db.get_conn()?;
        let exists: Option<String> = conn
            .query_row(
                "SELECT device_id FROM devices WHERE device_id = ? AND network_id = ?",
                rusqlite::params![&device_id, &network_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        Ok::<_, ApiError>(exists.is_some())
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    if !device_exists {
        return Err(ApiError::NotFound(format!(
            "Device {} not found in network {}",
            req.device_id, req.network_id
        )));
    }

    // Generate job ID
    let job_id = Uuid::new_v4();

    info!(
        job_id = %job_id,
        workers = topology.workers.len(),
        "Distributing inference job to workers"
    );

    // For MVP: Simple tokenization (char codes as tokens)
    // In production, this would use a real tokenizer (tiktoken, sentencepiece, etc.)
    let prompt_tokens: Vec<u32> = req
        .prompt
        .chars()
        .map(|c| c as u32)
        .collect();

    if prompt_tokens.is_empty() {
        return Err(ApiError::BadRequest("Prompt tokenization failed".to_string()));
    }

    info!(
        job_id = %job_id,
        prompt_len = req.prompt.len(),
        token_count = prompt_tokens.len(),
        "Tokenized prompt"
    );

    // TODO: Distribute job to workers
    // For now, we'll send notifications via the topology notifier
    // Workers should poll for jobs or listen to callbacks

    let worker_count = topology.workers.len();

    // Create distributed inference job
    use std::time::{SystemTime, UNIX_EPOCH};
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let job = crate::state::DistributedInferenceJob {
        job_id: job_id.to_string(),
        network_id: req.network_id.clone(),
        model_id: req.model_id.clone(),
        prompt_tokens: prompt_tokens.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        created_at,
    };

    // Enqueue job for workers to poll
    state.enqueue_job(job)?;

    info!(
        job_id = %job_id,
        worker_count,
        "Inference job queued successfully"
    );

    Ok(Json(SubmitInferenceResponse {
        success: true,
        job_id: job_id.to_string(),
        completion: None,
        completion_tokens: 0,
        execution_time_ms: 0,
        error: Some(format!(
            "Job queued successfully. {} workers in ring will poll and process the job.",
            worker_count
        )),
    }))
}

/// Poll for the next inference job in the network queue
///
/// GET /api/inference/poll/:network_id
/// Response: { job_id, model_id, prompt_tokens, max_tokens, temperature, top_p } or 204 No Content
#[instrument(skip(state))]
pub async fn poll_inference_job(
    State(state): State<AppState>,
    axum::extract::Path(network_id): axum::extract::Path<String>,
) -> ApiResult<axum::response::Response> {
    use axum::response::IntoResponse;

    // Validate network_id
    if network_id.is_empty() {
        return Err(ApiError::BadRequest("network_id cannot be empty".to_string()));
    }

    // Try to dequeue a job
    match state.dequeue_job(&network_id)? {
        Some(job) => {
            info!(
                job_id = %job.job_id,
                network_id = %network_id,
                "Worker polling - job dispatched"
            );

            Ok((axum::http::StatusCode::OK, axum::Json(job)).into_response())
        }
        None => {
            // No jobs available
            Ok(axum::http::StatusCode::NO_CONTENT.into_response())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ring::join_ring;
    use crate::api::types::RingJoinRequest;
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
    async fn test_submit_inference_success() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let device_id = "test-device-1";
        let network_id = "test-network";

        // Register device and join ring
        register_test_device(&db, device_id, network_id);

        let join_request = RingJoinRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
        };

        let _ = join_ring(axum::extract::State(state.clone()), axum::Json(join_request))
            .await
            .unwrap();

        // Submit inference request
        let inference_request = SubmitInferenceRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            model_id: "llama-70b".to_string(),
            prompt: "Hello, world!".to_string(),
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = submit_inference(
            axum::extract::State(state),
            axum::Json(inference_request),
        )
        .await;

        assert!(result.is_ok());
        let response = result.unwrap().0;
        assert!(response.success);
        assert!(!response.job_id.is_empty());
    }

    #[tokio::test]
    async fn test_submit_inference_no_workers() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let device_id = "test-device-1";
        let network_id = "test-network";

        // Register device but don't join ring
        register_test_device(&db, device_id, network_id);

        let inference_request = SubmitInferenceRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            model_id: "llama-70b".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = submit_inference(
            axum::extract::State(state),
            axum::Json(inference_request),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::BadRequest(_)));
    }

    #[tokio::test]
    async fn test_submit_inference_invalid_device() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db, keypair);

        let inference_request = SubmitInferenceRequest {
            device_id: "nonexistent".to_string(),
            network_id: "test-network".to_string(),
            model_id: "llama-70b".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = submit_inference(
            axum::extract::State(state),
            axum::Json(inference_request),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_submit_inference_empty_prompt() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let device_id = "test-device-1";
        let network_id = "test-network";

        register_test_device(&db, device_id, network_id);

        let join_request = RingJoinRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
        };

        let _ = join_ring(axum::extract::State(state.clone()), axum::Json(join_request))
            .await
            .unwrap();

        let inference_request = SubmitInferenceRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            model_id: "llama-70b".to_string(),
            prompt: "".to_string(),
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = submit_inference(
            axum::extract::State(state),
            axum::Json(inference_request),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::BadRequest(_)));
    }

    #[tokio::test]
    async fn test_submit_inference_max_tokens_validation() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let device_id = "test-device-1";
        let network_id = "test-network";

        register_test_device(&db, device_id, network_id);

        let join_request = RingJoinRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
        };

        let _ = join_ring(axum::extract::State(state.clone()), axum::Json(join_request))
            .await
            .unwrap();

        // Test max_tokens = 0
        let inference_request = SubmitInferenceRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            model_id: "llama-70b".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 0,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = submit_inference(
            axum::extract::State(state.clone()),
            axum::Json(inference_request),
        )
        .await;

        assert!(result.is_err());

        // Test max_tokens > 2048
        let inference_request = SubmitInferenceRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            model_id: "llama-70b".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 3000,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = submit_inference(
            axum::extract::State(state),
            axum::Json(inference_request),
        )
        .await;

        assert!(result.is_err());
    }
}
