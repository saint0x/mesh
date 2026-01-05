use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::state::AppState;
use crate::api::error::ApiError;

/// Request to create a ledger event
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateLedgerEventRequest {
    pub network_id: String,
    pub event_type: String,
    pub job_id: Option<Uuid>,
    pub device_id: Uuid,
    pub credits_amount: Option<f64>,
    pub metadata: serde_json::Value,
}

/// Response after creating a ledger event
#[derive(Debug, Serialize)]
pub struct CreateLedgerEventResponse {
    pub event_id: i64,
    pub message: String,
}

/// Create a new ledger event
///
/// Records job execution events, credit transfers, and other network activity
/// in the immutable append-only ledger.
pub async fn create_ledger_event(
    State(state): State<AppState>,
    Json(event): Json<CreateLedgerEventRequest>,
) -> Result<Json<CreateLedgerEventResponse>, ApiError> {
    tracing::info!(
        network_id = %event.network_id,
        event_type = %event.event_type,
        device_id = %event.device_id,
        job_id = ?event.job_id,
        credits = ?event.credits_amount,
        "Creating ledger event"
    );

    // Insert ledger event into database
    let event_id = tokio::task::spawn_blocking({
        let db = state.db.clone();
        let event = event.clone();
        move || -> Result<i64, ApiError> {
            let conn = db.get_conn()
                .map_err(|e| ApiError::Internal(format!("Failed to get connection: {}", e)))?;

            let event_id: i64 = conn.query_row(
                "INSERT INTO ledger_events (
                    network_id,
                    event_type,
                    job_id,
                    device_id,
                    credits_amount,
                    metadata,
                    created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))
                RETURNING id",
                rusqlite::params![
                    event.network_id,
                    event.event_type,
                    event.job_id.map(|id: Uuid| id.to_string()),
                    event.device_id.to_string(),
                    event.credits_amount,
                    event.metadata.to_string(),
                ],
                |row: &rusqlite::Row| row.get(0),
            )
            .map_err(|e| ApiError::Internal(format!("Failed to insert ledger event: {}", e)))?;

            Ok(event_id)
        }
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    tracing::debug!(event_id = event_id, "Ledger event created");

    Ok(Json(CreateLedgerEventResponse {
        event_id,
        message: "Ledger event created successfully".to_string(),
    }))
}

// Note: GET endpoint for ledger events deferred to future iteration
// Can be added when web dashboard needs to query events

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_ledger_event_request_deserialize() {
        let json = r#"{
            "network_id": "test-net",
            "event_type": "job_completed",
            "job_id": "123e4567-e89b-12d3-a456-426614174000",
            "device_id": "123e4567-e89b-12d3-a456-426614174001",
            "credits_amount": 10.5,
            "metadata": {"duration_ms": 250}
        }"#;

        let request: CreateLedgerEventRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.network_id, "test-net");
        assert_eq!(request.event_type, "job_completed");
        assert_eq!(request.credits_amount, Some(10.5));
    }

    #[test]
    fn test_create_ledger_event_request_without_optional_fields() {
        let json = r#"{
            "network_id": "test-net",
            "event_type": "job_started",
            "job_id": null,
            "device_id": "123e4567-e89b-12d3-a456-426614174001",
            "credits_amount": null,
            "metadata": {}
        }"#;

        let request: CreateLedgerEventRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.network_id, "test-net");
        assert!(request.job_id.is_none());
        assert!(request.credits_amount.is_none());
    }
}
