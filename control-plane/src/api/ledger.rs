use crate::api::error::ApiError;
use crate::state::AppState;
use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    pub event_id: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListLedgerEventsQuery {
    pub network_id: Option<String>,
    pub job_id: Option<Uuid>,
    pub device_id: Option<Uuid>,
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LedgerEventRecord {
    pub event_id: String,
    pub network_id: String,
    pub event_type: String,
    pub job_id: Option<Uuid>,
    pub device_id: Option<Uuid>,
    pub credits_amount: Option<f64>,
    pub metadata: serde_json::Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListLedgerEventsResponse {
    pub events: Vec<LedgerEventRecord>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LedgerSummaryQuery {
    pub network_id: String,
    pub job_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LedgerSummaryResponse {
    pub network_id: String,
    pub job_id: Option<Uuid>,
    pub total_events: u64,
    pub total_credits_earned: f64,
    pub total_credits_burned: f64,
    pub total_jobs_started: u64,
    pub total_jobs_completed: u64,
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
    let event_id = Uuid::new_v4().to_string();
    let persisted_event_id = event_id.clone();
    tokio::task::spawn_blocking({
        let db = state.db.clone();
        let event = event.clone();
        move || -> Result<(), ApiError> {
            let conn = db
                .get_conn()
                .map_err(|e| ApiError::Internal(format!("Failed to get connection: {}", e)))?;

            conn.execute(
                "INSERT INTO ledger_events (
                    event_id,
                    network_id,
                    event_type,
                    job_id,
                    device_id,
                    credits_amount,
                    metadata,
                    timestamp
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, datetime('now'))",
                rusqlite::params![
                    persisted_event_id,
                    event.network_id,
                    event.event_type,
                    event.job_id.map(|id: Uuid| id.to_string()),
                    event.device_id.to_string(),
                    event.credits_amount,
                    event.metadata.to_string(),
                ],
            )
            .map_err(|e| ApiError::Internal(format!("Failed to insert ledger event: {}", e)))?;
            Ok(())
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

pub async fn list_ledger_events(
    State(state): State<AppState>,
    Query(query): Query<ListLedgerEventsQuery>,
) -> Result<Json<ListLedgerEventsResponse>, ApiError> {
    let db = state.db.clone();
    let events = tokio::task::spawn_blocking(move || -> Result<Vec<LedgerEventRecord>, ApiError> {
        let conn = db
            .get_conn()
            .map_err(|e| ApiError::Internal(format!("Failed to get connection: {}", e)))?;

        let limit = query.limit.unwrap_or(100).clamp(1, 1000);
        let mut stmt = conn
            .prepare(
                r#"
                SELECT event_id, network_id, event_type, job_id, device_id, credits_amount, metadata, timestamp
                FROM ledger_events
                WHERE (?1 IS NULL OR network_id = ?1)
                  AND (?2 IS NULL OR job_id = ?2)
                  AND (?3 IS NULL OR device_id = ?3)
                ORDER BY timestamp DESC, event_id DESC
                LIMIT ?4
                "#,
            )
            .map_err(|e| ApiError::Internal(format!("Failed to prepare ledger query: {}", e)))?;

        let rows = stmt
            .query_map(
                rusqlite::params![
                    query.network_id,
                    query.job_id.map(|id| id.to_string()),
                    query.device_id.map(|id| id.to_string()),
                    i64::from(limit),
                ],
                |row| {
                    let job_id: Option<String> = row.get(3)?;
                    let device_id: Option<String> = row.get(4)?;
                    let metadata_str: String = row.get(6)?;
                    let metadata = serde_json::from_str(&metadata_str).map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            6,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;

                    Ok(LedgerEventRecord {
                        event_id: row.get(0)?,
                        network_id: row.get(1)?,
                        event_type: row.get(2)?,
                        job_id: job_id.and_then(|id| Uuid::parse_str(&id).ok()),
                        device_id: device_id.and_then(|id| Uuid::parse_str(&id).ok()),
                        credits_amount: row.get(5)?,
                        metadata,
                        created_at: row.get(7)?,
                    })
                },
            )
            .map_err(|e| ApiError::Internal(format!("Failed to query ledger events: {}", e)))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ApiError::Internal(format!("Failed to collect ledger events: {}", e)))?;

        Ok(rows)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(ListLedgerEventsResponse { events }))
}

pub async fn get_ledger_summary(
    State(state): State<AppState>,
    Query(query): Query<LedgerSummaryQuery>,
) -> Result<Json<LedgerSummaryResponse>, ApiError> {
    let db = state.db.clone();
    let summary = tokio::task::spawn_blocking(move || -> Result<LedgerSummaryResponse, ApiError> {
        let conn = db
            .get_conn()
            .map_err(|e| ApiError::Internal(format!("Failed to get connection: {}", e)))?;

        let job_id = query.job_id.map(|id| id.to_string());
        let mut stmt = conn
            .prepare(
                r#"
                SELECT
                    COUNT(*) AS total_events,
                    COALESCE(SUM(CASE WHEN event_type = 'credits_earned' THEN COALESCE(credits_amount, 0) ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type = 'credits_burned' THEN ABS(COALESCE(credits_amount, 0)) ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type = 'job_started' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type = 'job_completed' THEN 1 ELSE 0 END), 0)
                FROM ledger_events
                WHERE network_id = ?1
                  AND (?2 IS NULL OR job_id = ?2)
                "#,
            )
            .map_err(|e| ApiError::Internal(format!("Failed to prepare ledger summary: {}", e)))?;

        let summary = stmt
            .query_row(rusqlite::params![&query.network_id, job_id], |row| {
                Ok(LedgerSummaryResponse {
                    network_id: query.network_id.clone(),
                    job_id: query.job_id,
                    total_events: row.get::<_, i64>(0)? as u64,
                    total_credits_earned: row.get(1)?,
                    total_credits_burned: row.get(2)?,
                    total_jobs_started: row.get::<_, i64>(3)? as u64,
                    total_jobs_completed: row.get::<_, i64>(4)? as u64,
                })
            })
            .map_err(|e| ApiError::Internal(format!("Failed to query ledger summary: {}", e)))?;

        Ok(summary)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(summary))
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

    #[test]
    fn test_list_ledger_events_query_deserialize() {
        let query =
            serde_urlencoded::from_str::<ListLedgerEventsQuery>("network_id=test-net&limit=25")
                .unwrap();

        assert_eq!(query.network_id.as_deref(), Some("test-net"));
        assert_eq!(query.limit, Some(25));
    }
}
