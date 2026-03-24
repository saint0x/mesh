use axum::{
    extract::{Path, State},
    Json,
};
use rusqlite::{params, OptionalExtension};
use time::{Duration, OffsetDateTime};
use tracing::{info, instrument};
use uuid::Uuid;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    AcknowledgeInferenceAssignmentRequest, ClaimInferenceAssignmentRequest,
    ClaimInferenceAssignmentResponse, InferenceAssignment, InferenceJobAssignmentStatus,
    InferenceJobStatusResponse, ReportInferenceAssignmentRequest, SubmitInferenceRequest,
    SubmitInferenceResponse,
};
use crate::state::AppState;

const ASSIGNMENT_LEASE_SECS: i64 = 60;

#[derive(Clone)]
struct PersistedAssignment {
    assignment: InferenceAssignment,
}

#[derive(Clone)]
struct PersistedJobStatus {
    job_id: String,
    network_id: String,
    model_id: String,
    status: String,
    completion: Option<String>,
    completion_tokens: u32,
    execution_time_ms: u64,
    error: Option<String>,
    assignments: Vec<InferenceJobAssignmentStatus>,
}

#[instrument(skip(state))]
pub async fn submit_inference(
    State(state): State<AppState>,
    Json(req): Json<SubmitInferenceRequest>,
) -> ApiResult<Json<SubmitInferenceResponse>> {
    validate_submit_request(&req)?;

    info!(
        device_id = %req.device_id,
        network_id = %req.network_id,
        model_id = %req.model_id,
        max_tokens = req.max_tokens,
        "Received inference request"
    );

    let ring_manager = state.get_ring_manager(&req.network_id)?;
    let topology_network_id = req.network_id.clone();
    let topology =
        tokio::task::spawn_blocking(move || ring_manager.get_topology(&topology_network_id))
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

    let prompt_tokens: Vec<u32> = req.prompt.chars().map(|c| c as u32).collect();
    if prompt_tokens.is_empty() {
        return Err(ApiError::BadRequest(
            "Prompt tokenization failed".to_string(),
        ));
    }

    let db = state.db.clone();
    let request = req.clone();
    let workers = topology.workers.clone();
    let job_id = Uuid::new_v4().to_string();

    let persisted_job_id = job_id.clone();
    tokio::task::spawn_blocking(move || {
        let mut conn = db.get_conn()?;

        let device_exists: Option<String> = conn
            .query_row(
                "SELECT device_id FROM devices WHERE device_id = ? AND network_id = ?",
                params![&request.device_id, &request.network_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        if device_exists.is_none() {
            return Err(ApiError::NotFound(format!(
                "Device {} not found in network {}",
                request.device_id, request.network_id
            )));
        }

        let now = now_rfc3339()?;
        let prompt_tokens_json = serde_json::to_string(&prompt_tokens)
            .map_err(|e| ApiError::Internal(format!("Failed to serialize prompt tokens: {}", e)))?;

        let tx = conn
            .transaction()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        tx.execute(
            r#"
            INSERT INTO inference_jobs (
                job_id, network_id, submitted_by_device_id, model_id, prompt, prompt_tokens,
                max_tokens, temperature, top_p, status, ring_worker_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'dispatched', ?, ?, ?)
            "#,
            params![
                &persisted_job_id,
                &request.network_id,
                &request.device_id,
                &request.model_id,
                &request.prompt,
                &prompt_tokens_json,
                request.max_tokens,
                request.temperature,
                request.top_p,
                workers.len() as i64,
                &now,
                &now
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        for worker in &workers {
            let assignment_id = Uuid::new_v4().to_string();
            tx.execute(
                r#"
                INSERT INTO inference_job_assignments (
                    assignment_id, job_id, network_id, device_id, ring_position, status, assigned_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                "#,
                params![
                    &assignment_id,
                    &persisted_job_id,
                    &request.network_id,
                    &worker.device_id,
                    worker.position as i64,
                    &now
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }

        tx.commit()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        Ok::<_, ApiError>(())
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    info!(
        job_id = %job_id,
        workers = topology.workers.len(),
        "Inference job dispatched"
    );

    Ok(Json(SubmitInferenceResponse {
        success: true,
        job_id,
        completion: None,
        completion_tokens: 0,
        execution_time_ms: 0,
        error: None,
    }))
}

#[instrument(skip(state))]
pub async fn claim_inference_assignment(
    State(state): State<AppState>,
    Json(req): Json<ClaimInferenceAssignmentRequest>,
) -> ApiResult<Json<ClaimInferenceAssignmentResponse>> {
    if req.device_id.is_empty() || req.network_id.is_empty() {
        return Err(ApiError::BadRequest(
            "device_id and network_id must be provided".to_string(),
        ));
    }

    let db = state.db.clone();
    let req_clone = req.clone();
    let assignment = tokio::task::spawn_blocking(move || claim_assignment(&db, &req_clone))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(ClaimInferenceAssignmentResponse {
        success: true,
        assignment: assignment.map(|record| record.assignment),
    }))
}

#[instrument(skip(state))]
pub async fn acknowledge_inference_assignment(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<AcknowledgeInferenceAssignmentRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    if req.device_id.is_empty() || job_id.is_empty() {
        return Err(ApiError::BadRequest(
            "device_id and job_id must be provided".to_string(),
        ));
    }

    let db = state.db.clone();
    tokio::task::spawn_blocking(move || acknowledge_assignment(&db, &job_id, &req.device_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(serde_json::json!({ "success": true })))
}

#[instrument(skip(state))]
pub async fn report_inference_result(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<ReportInferenceAssignmentRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    if req.device_id.is_empty() || job_id.is_empty() {
        return Err(ApiError::BadRequest(
            "device_id and job_id must be provided".to_string(),
        ));
    }

    let db = state.db.clone();
    let request = req.clone();
    tokio::task::spawn_blocking(move || report_assignment_result(&db, &job_id, &request))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(serde_json::json!({ "success": true })))
}

#[instrument(skip(state))]
pub async fn get_inference_job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> ApiResult<Json<InferenceJobStatusResponse>> {
    if job_id.is_empty() {
        return Err(ApiError::BadRequest("job_id cannot be empty".to_string()));
    }

    let db = state.db.clone();
    let status = tokio::task::spawn_blocking(move || load_job_status(&db, &job_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(InferenceJobStatusResponse {
        success: true,
        job_id: status.job_id,
        network_id: status.network_id,
        model_id: status.model_id,
        status: status.status,
        completion: status.completion,
        completion_tokens: status.completion_tokens,
        execution_time_ms: status.execution_time_ms,
        error: status.error,
        assignments: status.assignments,
    }))
}

fn validate_submit_request(req: &SubmitInferenceRequest) -> ApiResult<()> {
    if req.device_id.is_empty() {
        return Err(ApiError::BadRequest(
            "device_id cannot be empty".to_string(),
        ));
    }
    if req.network_id.is_empty() {
        return Err(ApiError::BadRequest(
            "network_id cannot be empty".to_string(),
        ));
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
    Ok(())
}

fn claim_assignment(
    db: &crate::db::Database,
    req: &ClaimInferenceAssignmentRequest,
) -> ApiResult<Option<PersistedAssignment>> {
    let mut conn = db.get_conn()?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let now = OffsetDateTime::now_utc();
    let now_str = format_time(now)?;
    let lease_expires = format_time(now + Duration::seconds(ASSIGNMENT_LEASE_SECS))?;

    let row = tx
        .query_row(
            r#"
            SELECT
                a.assignment_id, a.job_id, a.network_id, a.device_id, a.ring_position,
                j.model_id, j.prompt_tokens, j.max_tokens, j.temperature, j.top_p
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            WHERE a.device_id = ?
              AND a.network_id = ?
              AND (
                    a.status = 'pending'
                    OR (a.status = 'leased' AND (a.lease_expires_at IS NULL OR a.lease_expires_at <= ?))
                  )
              AND j.status IN ('dispatched', 'running')
            ORDER BY
                (
                    SELECT COUNT(*)
                    FROM inference_job_assignments submitter_assignments
                    INNER JOIN inference_jobs submitter_jobs
                        ON submitter_jobs.job_id = submitter_assignments.job_id
                    WHERE submitter_jobs.network_id = a.network_id
                      AND submitter_jobs.submitted_by_device_id = j.submitted_by_device_id
                      AND submitter_assignments.status IN ('leased', 'acknowledged')
                ) ASC,
                (
                    SELECT COUNT(*)
                    FROM inference_job_assignments job_assignments
                    WHERE job_assignments.job_id = a.job_id
                      AND job_assignments.status IN ('leased', 'acknowledged')
                ) ASC,
                j.created_at ASC,
                a.assigned_at ASC,
                a.assignment_id ASC
            LIMIT 1
            "#,
            params![&req.device_id, &req.network_id, &now_str],
            |row| {
                let prompt_tokens_json: String = row.get(6)?;
                let prompt_tokens = serde_json::from_str::<Vec<u32>>(&prompt_tokens_json)
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            6,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;

                Ok(PersistedAssignment {
                    assignment: InferenceAssignment {
                        assignment_id: row.get(0)?,
                        job_id: row.get(1)?,
                        network_id: row.get(2)?,
                        device_id: row.get(3)?,
                        ring_position: row.get::<_, i64>(4)? as u32,
                        model_id: row.get(5)?,
                        prompt_tokens,
                        max_tokens: row.get::<_, i64>(7)? as u32,
                        temperature: row.get(8)?,
                        top_p: row.get(9)?,
                        lease_expires_at: lease_expires.clone(),
                    },
                })
            },
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if let Some(assignment) = &row {
        tx.execute(
            r#"
            UPDATE inference_job_assignments
            SET status = 'leased', lease_expires_at = ?
            WHERE assignment_id = ?
            "#,
            params![&lease_expires, &assignment.assignment.assignment_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    }

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(row)
}

fn acknowledge_assignment(
    db: &crate::db::Database,
    job_id: &str,
    device_id: &str,
) -> ApiResult<()> {
    let conn = db.get_conn()?;
    let now = now_rfc3339()?;

    let updated = conn
        .execute(
            r#"
            UPDATE inference_job_assignments
            SET status = 'acknowledged', acknowledged_at = ?, lease_expires_at = NULL
            WHERE job_id = ? AND device_id = ? AND status = 'leased'
            "#,
            params![&now, job_id, device_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if updated == 0 {
        return Err(ApiError::Conflict(format!(
            "No leased assignment found for job {} and device {}",
            job_id, device_id
        )));
    }

    conn.execute(
        r#"
        UPDATE inference_jobs
        SET status = CASE WHEN status = 'dispatched' THEN 'running' ELSE status END,
            started_at = COALESCE(started_at, ?),
            updated_at = ?
        WHERE job_id = ?
        "#,
        params![&now, &now, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(())
}

fn report_assignment_result(
    db: &crate::db::Database,
    job_id: &str,
    req: &ReportInferenceAssignmentRequest,
) -> ApiResult<()> {
    let mut conn = db.get_conn()?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let now = now_rfc3339()?;
    let assignment_status = if req.success { "completed" } else { "failed" };
    let rows = tx
        .execute(
            r#"
            UPDATE inference_job_assignments
            SET status = ?, completed_at = ?, lease_expires_at = NULL, failure_reason = ?
            WHERE job_id = ? AND device_id = ?
            "#,
            params![
                assignment_status,
                &now,
                req.error.as_deref(),
                job_id,
                &req.device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if rows == 0 {
        return Err(ApiError::NotFound(format!(
            "Assignment not found for job {} and device {}",
            job_id, req.device_id
        )));
    }

    let assignment_states = load_assignment_states(&tx, job_id)?;
    let failed = assignment_states
        .iter()
        .find(|(_, status, _)| status == "failed");
    let all_completed = assignment_states
        .iter()
        .all(|(_, status, _)| status == "completed");

    let (job_status, completion, completion_tokens, execution_time_ms, error, completed_at) =
        if let Some((_, _, failure_reason)) = failed {
            (
                "failed",
                None,
                0_i64,
                req.execution_time_ms as i64,
                failure_reason.clone().or_else(|| req.error.clone()),
                Some(now.clone()),
            )
        } else if all_completed {
            (
                "completed",
                req.completion.clone(),
                req.completion_tokens.unwrap_or(0) as i64,
                req.execution_time_ms as i64,
                None,
                Some(now.clone()),
            )
        } else {
            ("running", None, 0_i64, 0_i64, None, None)
        };

    tx.execute(
        r#"
        UPDATE inference_jobs
        SET status = ?, completion = COALESCE(?, completion), completion_tokens = CASE WHEN ? > 0 THEN ? ELSE completion_tokens END,
            execution_time_ms = CASE WHEN ? > 0 THEN ? ELSE execution_time_ms END,
            error = COALESCE(?, error), completed_at = COALESCE(?, completed_at), updated_at = ?
        WHERE job_id = ?
        "#,
        params![
            job_status,
            completion.as_deref(),
            completion_tokens,
            completion_tokens,
            execution_time_ms,
            execution_time_ms,
            error.as_deref(),
            completed_at.as_deref(),
            &now,
            job_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(())
}

fn load_job_status(db: &crate::db::Database, job_id: &str) -> ApiResult<PersistedJobStatus> {
    let conn = db.get_conn()?;

    let job = conn
        .query_row(
            r#"
            SELECT job_id, network_id, model_id, status, completion, completion_tokens, execution_time_ms, error
            FROM inference_jobs
            WHERE job_id = ?
            "#,
            params![job_id],
            |row| {
                Ok(PersistedJobStatus {
                    job_id: row.get(0)?,
                    network_id: row.get(1)?,
                    model_id: row.get(2)?,
                    status: row.get(3)?,
                    completion: row.get(4)?,
                    completion_tokens: row.get::<_, i64>(5)? as u32,
                    execution_time_ms: row.get::<_, i64>(6)? as u64,
                    error: row.get(7)?,
                    assignments: Vec::new(),
                })
            },
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .ok_or_else(|| ApiError::NotFound(format!("Inference job {} not found", job_id)))?;

    let mut stmt = conn
        .prepare(
            r#"
            SELECT device_id, ring_position, status, failure_reason
            FROM inference_job_assignments
            WHERE job_id = ?
            ORDER BY ring_position ASC
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let assignments = stmt
        .query_map(params![job_id], |row| {
            Ok(InferenceJobAssignmentStatus {
                device_id: row.get(0)?,
                ring_position: row.get::<_, i64>(1)? as u32,
                status: row.get(2)?,
                failure_reason: row.get(3)?,
            })
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(PersistedJobStatus { assignments, ..job })
}

fn load_assignment_states(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> ApiResult<Vec<(String, String, Option<String>)>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT device_id, status, failure_reason
            FROM inference_job_assignments
            WHERE job_id = ?
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let rows = stmt
        .query_map(params![job_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(rows)
}

fn now_rfc3339() -> ApiResult<String> {
    format_time(OffsetDateTime::now_utc())
}

fn format_time(ts: OffsetDateTime) -> ApiResult<String> {
    ts.format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ring::join_ring;
    use crate::api::types::RingJoinRequest;
    use crate::connectivity::{
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath, NetworkConnectivity,
    };
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

    fn test_connectivity() -> NetworkConnectivity {
        NetworkConnectivity {
            preferred_path: ConnectivityPath::Relayed,
            attachments: vec![ConnectivityAttachment {
                kind: ConnectivityAttachmentKind::Libp2pRelay,
                endpoint: "/dns4/relay.mesh.example/tcp/4001".to_string(),
                priority: 0,
            }],
        }
    }

    fn register_test_device(db: &crate::db::Database, device_id: &str, network_id: &str) {
        let _ = crate::services::network_service::create_network(
            db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
        );
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
            format!("test-peer-inf-{}", device_id),
            test_capabilities(),
        )
        .unwrap();
    }

    async fn joined_state(device_ids: &[&str], network_id: &str) -> AppState {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        for (idx, device_id) in device_ids.iter().enumerate() {
            register_test_device(&db, device_id, network_id);
            let join_request = RingJoinRequest {
                device_id: (*device_id).to_string(),
                network_id: network_id.to_string(),
                contributed_memory: 8_000_000_000 + idx as u64,
            };

            let _ = join_ring(State(state.clone()), Json(join_request))
                .await
                .unwrap();
        }

        state
    }

    #[tokio::test]
    async fn test_submit_claim_ack_complete_flow() {
        let network_id = "test-network";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "hello".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0;

        let assignment = claim.assignment.expect("expected assignment");
        assert_eq!(assignment.job_id, submit.job_id);

        let _ = acknowledge_inference_assignment(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(AcknowledgeInferenceAssignmentRequest {
                device_id: "worker-1".into(),
            }),
        )
        .await
        .unwrap();

        let _ = report_inference_result(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(ReportInferenceAssignmentRequest {
                device_id: "worker-1".into(),
                success: true,
                completion: Some("partial".into()),
                completion_tokens: Some(2),
                execution_time_ms: 100,
                error: None,
            }),
        )
        .await
        .unwrap();

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
            .await
            .unwrap()
            .0;

        assert_eq!(status.status, "running");
        assert_eq!(status.assignments.len(), 2);
    }

    #[tokio::test]
    async fn test_claim_assignment_prefers_less_served_submitter() {
        let network_id = "test-network-fairness";
        let state = joined_state(&["worker-1", "worker-2", "worker-3"], network_id).await;

        let submit_a = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-a".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let submit_b = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-b".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let first_claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-3".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment
        .expect("expected first assignment");
        assert_eq!(first_claim.job_id, submit_a.job_id);

        let second_claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment
        .expect("expected second assignment");
        assert_eq!(second_claim.job_id, submit_b.job_id);
    }

    #[tokio::test]
    async fn test_claim_assignment_prefers_less_served_job_before_created_at() {
        let network_id = "test-network-job-fairness";
        let state = joined_state(&["worker-1", "worker-2", "worker-3"], network_id).await;

        let submit_a = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-a".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let submit_b = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-b".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let first_claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-3".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment
        .expect("expected first assignment");
        assert_eq!(first_claim.job_id, submit_a.job_id);

        let second_claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment
        .expect("expected second assignment");
        assert_eq!(second_claim.job_id, submit_b.job_id);
    }

    #[tokio::test]
    async fn test_submit_inference_requires_ring() {
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        register_test_device(&db, "test-device-1", "test-network");

        let result = submit_inference(
            State(state),
            Json(SubmitInferenceRequest {
                device_id: "test-device-1".into(),
                network_id: "test-network".into(),
                model_id: "llama-70b".into(),
                prompt: "hello".into(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await;

        assert!(result.is_err());
    }
}
