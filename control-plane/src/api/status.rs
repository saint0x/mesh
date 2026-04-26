use axum::{
    extract::{Path, State},
    Json,
};
use tracing::instrument;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{JobSchedulerStatusResponse, NetworkSchedulerStatusResponse};
use crate::state::AppState;

/// Get operator-facing scheduler status for a network.
#[instrument(skip(state))]
pub async fn get_network_scheduler_status(
    State(state): State<AppState>,
    Path(network_id): Path<String>,
) -> ApiResult<Json<NetworkSchedulerStatusResponse>> {
    if network_id.is_empty() {
        return Err(ApiError::BadRequest(
            "network_id cannot be empty".to_string(),
        ));
    }

    let db = state.db.clone();
    let response =
        tokio::task::spawn_blocking(move || db.load_network_scheduler_status(&network_id))
            .await
            .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(response))
}

/// Get operator-facing scheduler status for a single job.
#[instrument(skip(state))]
pub async fn get_job_scheduler_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> ApiResult<Json<JobSchedulerStatusResponse>> {
    if job_id.is_empty() {
        return Err(ApiError::BadRequest("job_id cannot be empty".to_string()));
    }

    let db = state.db.clone();
    let response = tokio::task::spawn_blocking(move || db.load_job_scheduler_status(&job_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(response))
}
