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
    InferenceJobStatusResponse, ReportInferenceAssignmentProgressRequest,
    ReportInferenceAssignmentRequest, SubmitInferenceRequest, SubmitInferenceResponse,
};
use crate::connectivity::InferenceSchedulingPolicy;
use crate::consumption_policy::{
    compute_consumption_components, quote_consumption, ConsumptionQuoteInput,
};
use crate::credit_policy::{
    compute_credit_policy, AssignmentCreditInput, AssignmentCreditOutput, CreditPolicyInput,
};
use crate::device::{DeviceCapabilities, Tier};
use crate::model_assets;
use crate::provider::ExecutionProviderKind;
use crate::services::network_service;
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
    reserved_credits: f64,
    settled_credits: f64,
    released_credits: f64,
    available_completion_tokens: u32,
    model_size_factor: f64,
    error: Option<String>,
    assignments: Vec<InferenceJobAssignmentStatus>,
}

#[derive(Clone)]
struct PersistedJobContext {
    network_id: String,
    model_id: String,
    submitted_by_device_id: String,
    ring_worker_count: u32,
    prompt_tokens: u32,
    reserved_credits: f64,
    total_model_bytes: u64,
    total_columns: u32,
}

#[derive(Clone)]
struct PersistedCompletedAssignmentCreditInput {
    device_id: String,
    execution_provider: String,
    execution_time_ms: u64,
    reported_completion_tokens: u32,
    assigned_capacity_units: u32,
    shard_column_start: u32,
    shard_column_end: u32,
    available_memory_bytes: u64,
}

#[derive(Clone, Copy)]
struct PersistedJobSettlementState {
    settled_credits: f64,
    released_credits: f64,
    accounted_completion_tokens: u32,
    prompt_credits_accounted: bool,
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

    let prompt_tokens = model_assets::tokenize_prompt(&req.model_id, &req.prompt)?;

    let db = state.db.clone();
    let request = req.clone();
    let workers = topology.workers.clone();
    let job_id = Uuid::new_v4().to_string();

    let persisted_job_id = job_id.clone();
    let reservation = tokio::task::spawn_blocking(move || {
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
        let manifest = model_assets::load_model_manifest(&request.model_id)?;
        let consumption_quote = quote_consumption(ConsumptionQuoteInput {
            prompt_tokens: prompt_tokens.len() as u32,
            requested_completion_tokens: request.max_tokens,
            total_model_bytes: manifest.total_model_bytes,
        });

        let scheduling_policy =
            network_service::load_network_settings(&db, &request.network_id)?.scheduling_policy;

        let tx = conn
            .transaction()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let available_credits =
            load_device_available_credits(&tx, &request.network_id, &request.device_id)?;
        if available_credits + f64::EPSILON < consumption_quote.total_credits {
            return Err(ApiError::Conflict(format!(
                "Device {} has insufficient credits: available {:.3}, required {:.3}",
                request.device_id, available_credits, consumption_quote.total_credits
            )));
        }

        tx.execute(
            r#"
            INSERT INTO inference_jobs (
                job_id, network_id, submitted_by_device_id, model_id, prompt, prompt_tokens,
                max_tokens, temperature, top_p, status, ring_worker_count, created_at, updated_at,
                reserved_credits, available_completion_tokens, model_size_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'dispatched', ?, ?, ?, ?, ?, ?)
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
                &now,
                consumption_quote.total_credits,
                request.max_tokens,
                consumption_quote.model_size_factor,
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        insert_ledger_event(
            &tx,
            &request.network_id,
            "credits_reserved",
            Some(&persisted_job_id),
            Some(&request.device_id),
            None,
            serde_json::json!({
                "credit_model": "consumption_v1",
                "model_id": request.model_id,
                "prompt_tokens": prompt_tokens.len(),
                "requested_completion_tokens": request.max_tokens,
                "reserved_credits": consumption_quote.total_credits,
                "model_size_factor": consumption_quote.model_size_factor,
            }),
        )?;

        insert_ledger_event(
            &tx,
            &request.network_id,
            "job_started",
            Some(&persisted_job_id),
            Some(&request.device_id),
            None,
            serde_json::json!({
                "model_id": request.model_id,
                "ring_worker_count": workers.len(),
                "max_tokens": request.max_tokens,
                "reserved_credits": consumption_quote.total_credits,
                "model_size_factor": consumption_quote.model_size_factor,
            }),
        )?;

        for worker in &workers {
            let assignment_id = Uuid::new_v4().to_string();
            let assignment_metadata = load_device_assignment_metadata(
                &tx,
                &request.network_id,
                &worker.device_id,
                &scheduling_policy,
            )?;
            tx.execute(
                r#"
                INSERT INTO inference_job_assignments (
                    assignment_id, job_id, network_id, device_id, ring_position, status, assigned_at,
                    shard_column_start, shard_column_end, assigned_capacity_units, execution_provider
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                "#,
                params![
                    &assignment_id,
                    &persisted_job_id,
                    &request.network_id,
                    &worker.device_id,
                    worker.position as i64,
                    &now,
                    worker.shard.column_range.0 as i64,
                    worker.shard.column_range.1 as i64,
                    assignment_metadata.assigned_capacity_units as i64,
                    assignment_metadata.execution_provider,
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }

        tx.commit()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        Ok::<_, ApiError>((
            consumption_quote.total_credits,
            request.max_tokens,
            consumption_quote.model_size_factor,
        ))
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
        reserved_credits: reservation.0,
        available_completion_tokens: reservation.1,
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
pub async fn report_inference_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<ReportInferenceAssignmentProgressRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    if req.device_id.is_empty() || job_id.is_empty() {
        return Err(ApiError::BadRequest(
            "device_id and job_id must be provided".to_string(),
        ));
    }

    let db = state.db.clone();
    let request = req.clone();
    tokio::task::spawn_blocking(move || report_assignment_progress(&db, &job_id, &request))
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
        reserved_credits: status.reserved_credits,
        settled_credits: status.settled_credits,
        released_credits: status.released_credits,
        available_completion_tokens: status.available_completion_tokens,
        model_size_factor: status.model_size_factor,
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

fn execution_provider_label(provider: ExecutionProviderKind) -> &'static str {
    match provider {
        ExecutionProviderKind::Cpu => "cpu",
        ExecutionProviderKind::Metal => "metal",
        ExecutionProviderKind::Cuda => "cuda",
    }
}

fn capacity_units_for_tier(policy: &InferenceSchedulingPolicy, tier: Tier) -> u32 {
    match tier {
        Tier::Tier0 => policy.tier_capacity_units.tier0,
        Tier::Tier1 => policy.tier_capacity_units.tier1,
        Tier::Tier2 => policy.tier_capacity_units.tier2,
        Tier::Tier3 => policy.tier_capacity_units.tier3,
        Tier::Tier4 => policy.tier_capacity_units.tier4,
    }
}

fn load_device_assignment_metadata(
    conn: &rusqlite::Transaction<'_>,
    network_id: &str,
    device_id: &str,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> ApiResult<PersistedAssignmentAccountingMetadata> {
    let capabilities_json: String = conn
        .query_row(
            "SELECT capabilities FROM devices WHERE network_id = ? AND device_id = ?",
            params![network_id, device_id],
            |row| row.get(0),
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let capabilities: DeviceCapabilities = serde_json::from_str(&capabilities_json)
        .map_err(|e| ApiError::Internal(format!("Failed to parse device capabilities: {}", e)))?;

    Ok(PersistedAssignmentAccountingMetadata {
        assigned_capacity_units: capacity_units_for_tier(scheduling_policy, capabilities.tier),
        execution_provider: execution_provider_label(capabilities.default_execution_provider)
            .to_string(),
    })
}

#[derive(Clone)]
struct PersistedAssignmentAccountingMetadata {
    assigned_capacity_units: u32,
    execution_provider: String,
}

fn claim_assignment(
    db: &crate::db::Database,
    req: &ClaimInferenceAssignmentRequest,
) -> ApiResult<Option<PersistedAssignment>> {
    let scheduling_policy =
        network_service::load_network_settings(db, &req.network_id)?.scheduling_policy;
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
                j.model_id, j.prompt_tokens, j.max_tokens, j.temperature, j.top_p,
                j.reserved_credits, j.available_completion_tokens, j.model_size_factor
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
                CASE
                    WHEN (
                        SELECT COUNT(*)
                        FROM inference_jobs active_jobs
                        WHERE active_jobs.network_id = a.network_id
                          AND active_jobs.submitted_by_device_id = j.submitted_by_device_id
                          AND active_jobs.status IN ('dispatched', 'running')
                          AND EXISTS (
                              SELECT 1
                              FROM inference_job_assignments active_assignments
                              WHERE active_assignments.job_id = active_jobs.job_id
                                AND active_assignments.status IN ('leased', 'acknowledged')
                    )
                    ) >= ?
                    THEN 1
                    ELSE 0
                END ASC,
                CASE
                    WHEN (
                        SELECT COUNT(*)
                        FROM inference_jobs active_model_jobs
                        WHERE active_model_jobs.network_id = a.network_id
                          AND active_model_jobs.model_id = j.model_id
                          AND active_model_jobs.status IN ('dispatched', 'running')
                          AND EXISTS (
                              SELECT 1
                              FROM inference_job_assignments active_model_assignments
                              WHERE active_model_assignments.job_id = active_model_jobs.job_id
                                AND active_model_assignments.status IN ('leased', 'acknowledged')
                          )
                    ) >= (
                        SELECT MAX(
                            1,
                            COUNT(*) / ?
                        )
                        FROM devices pool_workers
                        WHERE pool_workers.network_id = a.network_id
                          AND pool_workers.ring_position IS NOT NULL
                          AND pool_workers.status = 'online'
                    )
                    THEN 1
                    ELSE 0
                END ASC,
                CASE
                    WHEN (
                        SELECT COALESCE(SUM(
                            CASE json_extract(active_capacity_devices.capabilities, '$.tier')
                                WHEN 'Tier0' THEN ?
                                WHEN 'Tier1' THEN ?
                                WHEN 'Tier2' THEN ?
                                WHEN 'Tier3' THEN ?
                                WHEN 'Tier4' THEN ?
                                ELSE ?
                            END
                        ), 0)
                        FROM inference_job_assignments active_capacity_assignments
                        INNER JOIN inference_jobs active_capacity_jobs
                            ON active_capacity_jobs.job_id = active_capacity_assignments.job_id
                        INNER JOIN devices active_capacity_devices
                            ON active_capacity_devices.device_id = active_capacity_assignments.device_id
                           AND active_capacity_devices.network_id = active_capacity_assignments.network_id
                        WHERE active_capacity_jobs.network_id = a.network_id
                          AND active_capacity_jobs.model_id = j.model_id
                          AND active_capacity_assignments.status IN ('leased', 'acknowledged')
                    ) >= (
                        SELECT MAX(
                            1,
                            COALESCE(SUM(
                                CASE json_extract(pool_capacity_workers.capabilities, '$.tier')
                                    WHEN 'Tier0' THEN ?
                                    WHEN 'Tier1' THEN ?
                                    WHEN 'Tier2' THEN ?
                                    WHEN 'Tier3' THEN ?
                                    WHEN 'Tier4' THEN ?
                                    ELSE ?
                                END
                            ), 0) / ?
                        )
                        FROM devices pool_capacity_workers
                        WHERE pool_capacity_workers.network_id = a.network_id
                          AND pool_capacity_workers.ring_position IS NOT NULL
                          AND pool_capacity_workers.status = 'online'
                    )
                    THEN 1
                    ELSE 0
                END ASC,
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
            params![
                &req.device_id,
                &req.network_id,
                &now_str,
                i64::from(scheduling_policy.submitter_active_job_soft_cap),
                i64::from(scheduling_policy.model_active_job_soft_cap_divisor),
                i64::from(scheduling_policy.tier_capacity_units.tier0),
                i64::from(scheduling_policy.tier_capacity_units.tier1),
                i64::from(scheduling_policy.tier_capacity_units.tier2),
                i64::from(scheduling_policy.tier_capacity_units.tier3),
                i64::from(scheduling_policy.tier_capacity_units.tier4),
                i64::from(scheduling_policy.tier_capacity_units.tier0),
                i64::from(scheduling_policy.tier_capacity_units.tier0),
                i64::from(scheduling_policy.tier_capacity_units.tier1),
                i64::from(scheduling_policy.tier_capacity_units.tier2),
                i64::from(scheduling_policy.tier_capacity_units.tier3),
                i64::from(scheduling_policy.tier_capacity_units.tier4),
                i64::from(scheduling_policy.tier_capacity_units.tier0),
                i64::from(scheduling_policy.capacity_unit_soft_cap_divisor)
            ],
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
                        reserved_credits: row.get(10)?,
                        available_completion_tokens: row.get::<_, i64>(11)? as u32,
                        model_size_factor: row.get(12)?,
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
            SET status = ?, completed_at = ?, lease_expires_at = NULL, failure_reason = ?,
                execution_time_ms = ?,
                reported_completion_tokens = MAX(reported_completion_tokens, ?)
            WHERE job_id = ? AND device_id = ? AND status IN ('leased', 'acknowledged')
            "#,
            params![
                assignment_status,
                &now,
                req.error.as_deref(),
                req.execution_time_ms as i64,
                req.completion_tokens.unwrap_or(0) as i64,
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
    let job_context = load_job_context(&tx, job_id)?;
    let settlement_state = load_job_settlement_state(&tx, job_id)?;

    let failed = assignment_states
        .iter()
        .find(|(_, status, _)| status == "failed");
    let all_completed = assignment_states
        .iter()
        .all(|(_, status, _)| status == "completed");

    let authoritative_execution_time_ms =
        compute_job_execution_time_ms(&tx, job_id, &now).unwrap_or(req.execution_time_ms);

    let (
        job_status,
        completion,
        completion_tokens,
        execution_time_ms,
        settled_credits,
        released_credits,
        error,
        completed_at,
    ) = if let Some((_, _, failure_reason)) = failed {
        if settlement_state.released_credits <= f64::EPSILON {
            insert_ledger_event(
                &tx,
                &job_context.network_id,
                "credits_released",
                Some(job_id),
                Some(&job_context.submitted_by_device_id),
                None,
                serde_json::json!({
                    "credit_model": "consumption_v1",
                    "model_id": job_context.model_id,
                    "reserved_credits": job_context.reserved_credits,
                    "release_reason": "job_failed",
                }),
            )?;
        }
        (
            "failed",
            None,
            0_i64,
            authoritative_execution_time_ms as i64,
            settlement_state.settled_credits,
            if settlement_state.released_credits > f64::EPSILON {
                settlement_state.released_credits
            } else {
                job_context.reserved_credits
            },
            failure_reason.clone().or_else(|| req.error.clone()),
            Some(now.clone()),
        )
    } else if all_completed {
        reconcile_realtime_job_accounting(&tx, job_id, &job_context, &settlement_state)?;

        let refreshed_settlement_state = load_job_settlement_state(&tx, job_id)?;
        if refreshed_settlement_state.released_credits <= f64::EPSILON {
            insert_ledger_event(
                &tx,
                &job_context.network_id,
                "credits_released",
                Some(job_id),
                Some(&job_context.submitted_by_device_id),
                None,
                serde_json::json!({
                    "credit_model": "consumption_v1",
                    "model_id": job_context.model_id,
                    "reserved_credits": job_context.reserved_credits,
                    "release_reason": "job_settlement",
                }),
            )?;
        }
        insert_ledger_event(
            &tx,
            &job_context.network_id,
            "job_completed",
            Some(job_id),
            Some(&job_context.submitted_by_device_id),
            None,
            serde_json::json!({
                "credit_model": "realtime_pipeline",
                "model_id": job_context.model_id,
                "ring_worker_count": job_context.ring_worker_count,
                "execution_time_ms": authoritative_execution_time_ms,
                "reserved_credits": job_context.reserved_credits,
                "settled_credits": refreshed_settlement_state.settled_credits,
                "accounted_completion_tokens": refreshed_settlement_state.accounted_completion_tokens,
            }),
        )?;
        (
            "completed",
            req.completion.clone(),
            req.completion_tokens.unwrap_or(0) as i64,
            authoritative_execution_time_ms as i64,
            refreshed_settlement_state.settled_credits,
            job_context.reserved_credits,
            None,
            Some(now.clone()),
        )
    } else {
        ("running", None, 0_i64, 0_i64, 0.0_f64, 0.0_f64, None, None)
    };

    tx.execute(
        r#"
        UPDATE inference_jobs
        SET status = ?, completion = COALESCE(?, completion), completion_tokens = CASE WHEN ? > 0 THEN ? ELSE completion_tokens END,
            execution_time_ms = CASE WHEN ? > 0 THEN ? ELSE execution_time_ms END,
            settled_credits = CASE WHEN ? > 0 THEN ? ELSE settled_credits END,
            released_credits = CASE WHEN ? > 0 THEN ? ELSE released_credits END,
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
            settled_credits,
            settled_credits,
            released_credits,
            released_credits,
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

fn report_assignment_progress(
    db: &crate::db::Database,
    job_id: &str,
    req: &ReportInferenceAssignmentProgressRequest,
) -> ApiResult<()> {
    let mut conn = db.get_conn()?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let updated = tx
        .execute(
            r#"
            UPDATE inference_job_assignments
            SET reported_completion_tokens = MAX(reported_completion_tokens, ?),
                execution_time_ms = MAX(execution_time_ms, ?)
            WHERE job_id = ? AND device_id = ? AND status IN ('acknowledged', 'completed')
            "#,
            params![
                req.completion_tokens as i64,
                req.execution_time_ms as i64,
                job_id,
                &req.device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if updated == 0 {
        return Err(ApiError::NotFound(format!(
            "Assignment not found for progress report on job {} and device {}",
            job_id, req.device_id
        )));
    }

    let job_context = load_job_context(&tx, job_id)?;
    let settlement_state = load_job_settlement_state(&tx, job_id)?;
    reconcile_realtime_job_accounting(&tx, job_id, &job_context, &settlement_state)?;

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn load_job_status(db: &crate::db::Database, job_id: &str) -> ApiResult<PersistedJobStatus> {
    let conn = db.get_conn()?;

    let job = conn
        .query_row(
            r#"
            SELECT job_id, network_id, model_id, status, completion, completion_tokens, execution_time_ms,
                   reserved_credits, settled_credits, released_credits, available_completion_tokens,
                   model_size_factor, error
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
                    reserved_credits: row.get(7)?,
                    settled_credits: row.get(8)?,
                    released_credits: row.get(9)?,
                    available_completion_tokens: row.get::<_, i64>(10)? as u32,
                    model_size_factor: row.get(11)?,
                    error: row.get(12)?,
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
            SELECT device_id, ring_position, status, failure_reason,
                   shard_column_start, shard_column_end, assigned_capacity_units,
                   execution_provider, execution_time_ms
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
                shard_column_start: row.get::<_, i64>(4)? as u32,
                shard_column_end: row.get::<_, i64>(5)? as u32,
                assigned_capacity_units: row.get::<_, i64>(6)? as u32,
                execution_provider: row.get(7)?,
                execution_time_ms: row.get::<_, i64>(8)? as u64,
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

fn load_job_context(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> ApiResult<PersistedJobContext> {
    conn.query_row(
        r#"
        SELECT network_id, model_id, submitted_by_device_id, ring_worker_count, prompt_tokens,
               reserved_credits
        FROM inference_jobs
        WHERE job_id = ?
        "#,
        params![job_id],
        |row| {
            let prompt_tokens_json: String = row.get(4)?;
            let prompt_tokens =
                serde_json::from_str::<Vec<u32>>(&prompt_tokens_json).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            let manifest =
                model_assets::load_model_manifest(&row.get::<_, String>(1)?).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        1,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::other(e.to_string())),
                    )
                })?;
            Ok(PersistedJobContext {
                network_id: row.get(0)?,
                model_id: row.get(1)?,
                submitted_by_device_id: row.get(2)?,
                ring_worker_count: row.get::<_, i64>(3)? as u32,
                prompt_tokens: prompt_tokens.len() as u32,
                reserved_credits: row.get(5)?,
                total_model_bytes: manifest.total_model_bytes,
                total_columns: manifest.tensor_parallelism_dim,
            })
        },
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_completed_assignment_credit_inputs(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> ApiResult<Vec<PersistedCompletedAssignmentCreditInput>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT
                a.device_id,
                a.execution_provider,
                a.execution_time_ms,
                a.reported_completion_tokens,
                a.assigned_capacity_units,
                a.shard_column_start,
                a.shard_column_end,
                d.capabilities,
                d.contributed_memory
            FROM inference_job_assignments a
            INNER JOIN devices d
                ON d.device_id = a.device_id
               AND d.network_id = a.network_id
            WHERE a.job_id = ?
              AND a.status IN ('acknowledged', 'completed')
            ORDER BY a.ring_position ASC, a.assignment_id ASC
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let rows = stmt
        .query_map(params![job_id], |row| {
            let capabilities_json: String = row.get(7)?;
            let capabilities: DeviceCapabilities = serde_json::from_str(&capabilities_json)
                .map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        7,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            let contributed_memory =
                row.get::<_, Option<i64>>(8)?.unwrap_or_default().max(0) as u64;
            let available_memory_bytes = if contributed_memory > 0 {
                contributed_memory.max(1)
            } else {
                ((capabilities.ram_mb + capabilities.gpu_vram_mb.unwrap_or_default()) as u64
                    * 1024
                    * 1024)
                    .max(1)
            };

            Ok(PersistedCompletedAssignmentCreditInput {
                device_id: row.get(0)?,
                execution_provider: row.get(1)?,
                execution_time_ms: row.get::<_, i64>(2)? as u64,
                reported_completion_tokens: row.get::<_, i64>(3)? as u32,
                assigned_capacity_units: row.get::<_, i64>(4)? as u32,
                shard_column_start: row.get::<_, i64>(5)? as u32,
                shard_column_end: row.get::<_, i64>(6)? as u32,
                available_memory_bytes,
            })
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_job_settlement_state(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> ApiResult<PersistedJobSettlementState> {
    conn.query_row(
        r#"
        SELECT settled_credits, released_credits
             , accounted_completion_tokens, prompt_credits_accounted
        FROM inference_jobs
        WHERE job_id = ?
        "#,
        params![job_id],
        |row| {
            Ok(PersistedJobSettlementState {
                settled_credits: row.get(0)?,
                released_credits: row.get(1)?,
                accounted_completion_tokens: row.get::<_, i64>(2)? as u32,
                prompt_credits_accounted: row.get::<_, i64>(3)? != 0,
            })
        },
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn reconcile_realtime_job_accounting(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
    job_context: &PersistedJobContext,
    settlement_state: &PersistedJobSettlementState,
) -> ApiResult<()> {
    let assignment_inputs = load_completed_assignment_credit_inputs(conn, job_id)?;
    if assignment_inputs.len() < job_context.ring_worker_count as usize {
        return Ok(());
    }

    let frontier_completion_tokens = assignment_inputs
        .iter()
        .map(|assignment| assignment.reported_completion_tokens)
        .min()
        .unwrap_or_default();

    let prompt_tokens_delta = if settlement_state.prompt_credits_accounted {
        0
    } else {
        job_context.prompt_tokens
    };
    let completion_tokens_delta =
        frontier_completion_tokens.saturating_sub(settlement_state.accounted_completion_tokens);

    if prompt_tokens_delta == 0 && completion_tokens_delta == 0 {
        return Ok(());
    }

    let credit_plan = calculate_realtime_credit_plan(
        job_context,
        &assignment_inputs,
        prompt_tokens_delta,
        completion_tokens_delta,
    );
    let consumption = compute_consumption_components(
        prompt_tokens_delta,
        completion_tokens_delta,
        job_context.total_model_bytes,
    );

    for (assignment, input) in credit_plan.assignments.iter().zip(assignment_inputs.iter()) {
        insert_ledger_event(
            conn,
            &job_context.network_id,
            "credits_earned",
            Some(job_id),
            Some(&assignment.device_id),
            Some(assignment.credits),
            realtime_earned_credit_metadata(
                job_context,
                assignment,
                input,
                prompt_tokens_delta,
                completion_tokens_delta,
                frontier_completion_tokens,
            ),
        )?;
    }

    insert_ledger_event(
        conn,
        &job_context.network_id,
        "credits_burned",
        Some(job_id),
        Some(&job_context.submitted_by_device_id),
        Some(-consumption.total_credits),
        serde_json::json!({
            "credit_model": "realtime_consumption",
            "model_id": job_context.model_id,
            "prompt_tokens_delta": prompt_tokens_delta,
            "completion_tokens_delta": completion_tokens_delta,
            "frontier_completion_tokens": frontier_completion_tokens,
            "prompt_credits": consumption.prompt_credits,
            "completion_credits": consumption.completion_credits,
            "settled_credits_delta": consumption.total_credits,
            "model_size_factor": consumption.model_size_factor,
        }),
    )?;

    conn.execute(
        r#"
        UPDATE inference_jobs
        SET completion_tokens = MAX(completion_tokens, ?),
            accounted_completion_tokens = ?,
            prompt_credits_accounted = CASE WHEN ? > 0 THEN 1 ELSE prompt_credits_accounted END,
            settled_credits = settled_credits + ?,
            updated_at = ?
        WHERE job_id = ?
        "#,
        params![
            frontier_completion_tokens as i64,
            frontier_completion_tokens as i64,
            prompt_tokens_delta as i64,
            consumption.total_credits,
            &now_rfc3339()?,
            job_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(())
}

fn calculate_realtime_credit_plan(
    job_context: &PersistedJobContext,
    assignments: &[PersistedCompletedAssignmentCreditInput],
    prompt_tokens: u32,
    completion_tokens: u32,
) -> crate::credit_policy::CreditPolicyOutput {
    let inputs = assignments
        .iter()
        .map(|assignment| AssignmentCreditInput {
            device_id: assignment.device_id.clone(),
            execution_time_ms: assignment.execution_time_ms.max(1),
            assigned_capacity_units: assignment.assigned_capacity_units,
            shard_column_start: assignment.shard_column_start,
            shard_column_end: assignment.shard_column_end,
            available_memory_bytes: assignment.available_memory_bytes,
        })
        .collect::<Vec<_>>();

    compute_credit_policy(CreditPolicyInput {
        prompt_tokens,
        completion_tokens,
        total_model_bytes: job_context.total_model_bytes,
        total_columns: job_context.total_columns,
        assignments: inputs,
    })
}

fn realtime_earned_credit_metadata(
    job_context: &PersistedJobContext,
    assignment: &AssignmentCreditOutput,
    assignment_input: &PersistedCompletedAssignmentCreditInput,
    prompt_tokens_delta: u32,
    completion_tokens_delta: u32,
    frontier_completion_tokens: u32,
) -> serde_json::Value {
    serde_json::json!({
        "credit_model": "realtime_contribution",
        "model_id": job_context.model_id,
        "execution_provider": assignment_input.execution_provider,
        "execution_time_ms": assignment_input.execution_time_ms,
        "reported_completion_tokens": assignment_input.reported_completion_tokens,
        "assigned_capacity_units": assignment_input.assigned_capacity_units,
        "shard_column_start": assignment_input.shard_column_start,
        "shard_column_end": assignment_input.shard_column_end,
        "prompt_tokens_delta": prompt_tokens_delta,
        "completion_tokens_delta": completion_tokens_delta,
        "frontier_completion_tokens": frontier_completion_tokens,
        "compute_share": assignment.compute_share,
        "throughput_multiplier": assignment.throughput_multiplier,
        "resource_pressure_multiplier": assignment.resource_pressure_multiplier,
        "normalized_contribution_share": assignment.normalized_contribution_share,
        "measured_service_rate": assignment.measured_service_rate,
        "reference_service_rate": assignment.reference_service_rate,
        "memory_pressure": assignment.memory_pressure,
    })
}

fn compute_job_execution_time_ms(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
    completed_at: &str,
) -> ApiResult<u64> {
    conn.query_row(
        r#"
        SELECT COALESCE(
            MAX(0, CAST(ROUND((julianday(?1) - julianday(started_at)) * 86400000.0) AS INTEGER)),
            0
        )
        FROM inference_jobs
        WHERE job_id = ?2
          AND started_at IS NOT NULL
        "#,
        params![completed_at, job_id],
        |row| row.get::<_, i64>(0),
    )
    .map(|value| value.max(0) as u64)
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_device_available_credits(
    conn: &rusqlite::Transaction<'_>,
    network_id: &str,
    device_id: &str,
) -> ApiResult<f64> {
    conn.query_row(
        r#"
        SELECT
            COALESCE((
                SELECT SUM(COALESCE(credits_amount, 0))
                FROM ledger_events
                WHERE network_id = ?1
                  AND device_id = ?2
            ), 0)
            -
            COALESCE((
                SELECT SUM(MAX(reserved_credits - released_credits, 0))
                FROM inference_jobs
                WHERE network_id = ?1
                  AND submitted_by_device_id = ?2
                  AND reserved_credits > released_credits
            ), 0)
        "#,
        params![network_id, device_id],
        |row| row.get::<_, f64>(0),
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn insert_ledger_event(
    conn: &rusqlite::Transaction<'_>,
    network_id: &str,
    event_type: &str,
    job_id: Option<&str>,
    device_id: Option<&str>,
    credits_amount: Option<f64>,
    metadata: serde_json::Value,
) -> ApiResult<()> {
    let event_id = Uuid::new_v4().to_string();
    conn.execute(
        r#"
        INSERT INTO ledger_events (
            event_id,
            network_id,
            event_type,
            job_id,
            device_id,
            credits_amount,
            metadata,
            timestamp
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, datetime('now'))
        "#,
        params![
            event_id,
            network_id,
            event_type,
            job_id,
            device_id,
            credits_amount,
            metadata.to_string(),
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
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
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath,
        InferenceSchedulingPolicy, NetworkConnectivity, TierCapacityUnits,
    };
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::provider::{ExecutionProviderInfo, ExecutionProviderKind};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::device_service;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn ensure_test_models() {
        for model_id in [
            "test-model",
            "llama-70b",
            "model-x",
            "model-y",
            "tinyllama-1.1b",
        ] {
            crate::model_assets::testsupport::ensure_test_model(model_id, 8192);
        }
    }

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            tier: Tier::Tier2,
            cpu_cores: 8,
            ram_mb: 16384,
            gpu_present: false,
            gpu_vram_mb: None,
            os: "macos".into(),
            arch: "aarch64".into(),
            execution_providers: vec![
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cpu,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Metal,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cuda,
                    available: false,
                    reason: Some("cuda provider is only available on Linux builds".into()),
                },
            ],
            default_execution_provider: ExecutionProviderKind::Metal,
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
        register_test_device_with_capabilities(db, device_id, network_id, test_capabilities());
    }

    fn register_test_device_with_capabilities(
        db: &crate::db::Database,
        device_id: &str,
        network_id: &str,
        capabilities: DeviceCapabilities,
    ) {
        let _ = crate::services::network_service::create_network(
            db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
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
            capabilities,
        )
        .unwrap();
    }

    fn seed_device_credits(
        db: &crate::db::Database,
        network_id: &str,
        device_id: &str,
        credits: f64,
    ) {
        let conn = db.get_conn().unwrap();
        conn.execute(
            r#"
            INSERT INTO ledger_events (
                event_id, network_id, event_type, job_id, device_id, credits_amount, metadata, timestamp
            ) VALUES (?1, ?2, 'credits_earned', NULL, ?3, ?4, ?5, datetime('now'))
            "#,
            params![
                Uuid::new_v4().to_string(),
                network_id,
                device_id,
                credits,
                serde_json::json!({
                    "credit_model": "bootstrap_test_funds",
                    "reason": "seeded inference submitter credits for tests",
                })
                .to_string(),
            ],
        )
        .unwrap();
    }

    async fn joined_state(device_ids: &[&str], network_id: &str) -> AppState {
        joined_state_with_policy(device_ids, network_id, InferenceSchedulingPolicy::default()).await
    }

    async fn joined_state_with_policy(
        device_ids: &[&str],
        network_id: &str,
        scheduling_policy: InferenceSchedulingPolicy,
    ) -> AppState {
        ensure_test_models();
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let _ = crate::services::network_service::create_network(
            &db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            scheduling_policy,
        );

        for (idx, device_id) in device_ids.iter().enumerate() {
            register_test_device(&db, device_id, network_id);
            seed_device_credits(&db, network_id, device_id, 10_000.0);
            let join_request = RingJoinRequest {
                device_id: (*device_id).to_string(),
                network_id: network_id.to_string(),
                model_id: "test-model".to_string(),
                contributed_memory: 8_000_000_000 + idx as u64,
            };

            let _ = join_ring(State(state.clone()), Json(join_request))
                .await
                .unwrap();
        }

        state
    }

    async fn joined_state_with_device_capabilities(
        devices: &[(&str, DeviceCapabilities)],
        network_id: &str,
        scheduling_policy: InferenceSchedulingPolicy,
    ) -> AppState {
        ensure_test_models();
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);

        let _ = crate::services::network_service::create_network(
            &db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            scheduling_policy,
        );

        for (idx, (device_id, capabilities)) in devices.iter().enumerate() {
            register_test_device_with_capabilities(
                &db,
                device_id,
                network_id,
                capabilities.clone(),
            );
            seed_device_credits(&db, network_id, device_id, 10_000.0);
            let join_request = RingJoinRequest {
                device_id: (*device_id).to_string(),
                network_id: network_id.to_string(),
                model_id: "test-model".to_string(),
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
    async fn test_inference_completion_records_ledger_events_and_credits() {
        let network_id = "test-network-ledger";
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

        for device_id in ["worker-1", "worker-2"] {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    network_id: network_id.into(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_result(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(ReportInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    success: true,
                    completion: Some(format!("completion-from-{}", device_id)),
                    completion_tokens: Some(4),
                    execution_time_ms: 500,
                    error: None,
                }),
            )
            .await
            .unwrap();
        }

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
            .await
            .unwrap()
            .0;
        assert_eq!(status.status, "completed");

        let events = crate::api::ledger::list_ledger_events(
            State(state.clone()),
            axum::extract::Query(crate::api::ledger::ListLedgerEventsQuery {
                network_id: Some(network_id.into()),
                job_id: Some(Uuid::parse_str(&submit.job_id).unwrap()),
                device_id: None,
                limit: Some(20),
            }),
        )
        .await
        .unwrap()
        .0;
        assert!(events
            .events
            .iter()
            .any(|event| event.event_type == "job_started"));
        assert!(events
            .events
            .iter()
            .any(|event| event.event_type == "job_completed"));
        assert_eq!(
            events
                .events
                .iter()
                .filter(|event| event.event_type == "credits_earned")
                .count(),
            2
        );
        assert!(events
            .events
            .iter()
            .any(|event| event.event_type == "credits_burned"));

        let summary = crate::api::ledger::get_ledger_summary(
            State(state),
            axum::extract::Query(crate::api::ledger::LedgerSummaryQuery {
                network_id: network_id.into(),
                job_id: Some(Uuid::parse_str(&submit.job_id).unwrap()),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(summary.total_jobs_started, 1);
        assert_eq!(summary.total_jobs_completed, 1);
        assert!(summary.total_credits_earned > 0.0);
        assert!(summary.total_credits_burned > 0.0);
        assert!((summary.total_credits_earned - summary.total_credits_burned).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_submit_inference_rejects_underfunded_submitter() {
        ensure_test_models();
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);
        let network_id = "test-network-underfunded";

        let _ = crate::services::network_service::create_network(
            &db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
        );

        register_test_device(&db, "worker-1", network_id);
        register_test_device(&db, "worker-2", network_id);

        for (idx, device_id) in ["worker-1", "worker-2"].into_iter().enumerate() {
            let _ = join_ring(
                State(state.clone()),
                Json(RingJoinRequest {
                    device_id: device_id.to_string(),
                    network_id: network_id.to_string(),
                    model_id: "llama-70b".to_string(),
                    contributed_memory: 8_000_000_000 + idx as u64,
                }),
            )
            .await
            .unwrap();
        }

        let result = submit_inference(
            State(state),
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
        .await;

        match result {
            Err(ApiError::Conflict(message)) => {
                assert!(message.contains("insufficient credits"));
            }
            other => panic!("expected insufficient-credit conflict, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_job_completion_releases_reservation_and_burns_actual_consumption() {
        let network_id = "test-network-consumption-settlement";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "reservation".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        assert!(submit.reserved_credits > 0.0);
        assert_eq!(submit.available_completion_tokens, 16);

        for device_id in ["worker-1", "worker-2"] {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    network_id: network_id.into(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected assignment");

            assert_eq!(claim.available_completion_tokens, 16);

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_result(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(ReportInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    success: true,
                    completion: Some(format!("completion-from-{}", device_id)),
                    completion_tokens: Some(4),
                    execution_time_ms: 500,
                    error: None,
                }),
            )
            .await
            .unwrap();
        }

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
            .await
            .unwrap()
            .0;
        assert_eq!(status.status, "completed");
        assert!(status.settled_credits > 0.0);
        assert_eq!(status.released_credits, submit.reserved_credits);
        assert!(status.reserved_credits >= status.settled_credits);

        let events = crate::api::ledger::list_ledger_events(
            State(state),
            axum::extract::Query(crate::api::ledger::ListLedgerEventsQuery {
                network_id: Some(network_id.into()),
                job_id: Some(Uuid::parse_str(&submit.job_id).unwrap()),
                device_id: None,
                limit: Some(20),
            }),
        )
        .await
        .unwrap()
        .0;

        assert!(events
            .events
            .iter()
            .any(|event| event.event_type == "credits_reserved"));
        assert!(events
            .events
            .iter()
            .any(|event| event.event_type == "credits_released"));
        assert!(events
            .events
            .iter()
            .any(|event| event.event_type == "credits_burned"));
    }

    #[tokio::test]
    async fn test_realtime_progress_advances_frontier_only_after_all_assignments_report() {
        let network_id = "test-network-realtime-progress";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "streaming".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        for device_id in ["worker-1", "worker-2"] {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    network_id: network_id.into(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();
        }

        let _ = report_inference_progress(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(ReportInferenceAssignmentProgressRequest {
                device_id: "worker-1".into(),
                completion_tokens: 3,
                execution_time_ms: 120,
            }),
        )
        .await
        .unwrap();

        let status_after_first_progress =
            get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
                .await
                .unwrap()
                .0;
        assert_eq!(status_after_first_progress.completion_tokens, 0);
        assert!(status_after_first_progress.settled_credits > 0.0);

        let _ = report_inference_progress(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(ReportInferenceAssignmentProgressRequest {
                device_id: "worker-2".into(),
                completion_tokens: 3,
                execution_time_ms: 140,
            }),
        )
        .await
        .unwrap();

        let status_after_frontier =
            get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
                .await
                .unwrap()
                .0;
        assert_eq!(status_after_frontier.completion_tokens, 3);
        assert!(status_after_frontier.settled_credits > 0.0);

        let events = crate::api::ledger::list_ledger_events(
            State(state),
            axum::extract::Query(crate::api::ledger::ListLedgerEventsQuery {
                network_id: Some(network_id.into()),
                job_id: Some(Uuid::parse_str(&submit.job_id).unwrap()),
                device_id: None,
                limit: Some(20),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(
            events
                .events
                .iter()
                .filter(|event| event.event_type == "credits_earned")
                .count(),
            4
        );
        assert_eq!(
            events
                .events
                .iter()
                .filter(|event| event.event_type == "credits_burned")
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn test_open_reservation_reduces_available_balance_until_release() {
        ensure_test_models();
        let db = create_test_db();
        let keypair = Arc::new(ControlPlaneKeypair::load_or_generate().unwrap());
        let state = AppState::new(db.clone(), keypair);
        let network_id = "test-network-open-reservation";

        let _ = crate::services::network_service::create_network(
            &db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
        );

        register_test_device(&db, "worker-1", network_id);
        register_test_device(&db, "worker-2", network_id);

        for (idx, device_id) in ["worker-1", "worker-2"].into_iter().enumerate() {
            let _ = join_ring(
                State(state.clone()),
                Json(RingJoinRequest {
                    device_id: device_id.to_string(),
                    network_id: network_id.to_string(),
                    model_id: "llama-70b".to_string(),
                    contributed_memory: 8_000_000_000 + idx as u64,
                }),
            )
            .await
            .unwrap();
        }

        let manifest = crate::model_assets::load_model_manifest("llama-70b").unwrap();
        let prompt_tokens = crate::model_assets::tokenize_prompt("llama-70b", "reservation")
            .unwrap()
            .len() as u32;
        let reserved_credits = quote_consumption(ConsumptionQuoteInput {
            prompt_tokens,
            requested_completion_tokens: 16,
            total_model_bytes: manifest.total_model_bytes,
        })
        .total_credits;
        let settled_credits =
            compute_consumption_components(prompt_tokens, 4, manifest.total_model_bytes)
                .total_credits;
        seed_device_credits(
            &db,
            network_id,
            "worker-1",
            reserved_credits + settled_credits,
        );

        let first_submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "reservation".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let second_submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "reservation".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await;
        assert!(matches!(second_submit, Err(ApiError::Conflict(_))));

        for device_id in ["worker-1", "worker-2"] {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    network_id: network_id.into(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_result(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(ReportInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    success: true,
                    completion: Some(format!("completion-from-{}", device_id)),
                    completion_tokens: Some(4),
                    execution_time_ms: 300,
                    error: None,
                }),
            )
            .await
            .unwrap();
        }

        let released_status =
            get_inference_job_status(State(state.clone()), Path(first_submit.job_id.clone()))
                .await
                .unwrap()
                .0;
        assert_eq!(
            released_status.released_credits,
            first_submit.reserved_credits
        );

        let third_submit = submit_inference(
            State(state),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "reservation".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await;
        assert!(third_submit.is_ok());
    }

    #[tokio::test]
    async fn test_credit_policy_rewards_faster_contributor_for_equal_assignment_work() {
        let network_id = "test-network-credit-throughput";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "throughput".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        for (device_id, execution_time_ms) in [("worker-1", 200_u64), ("worker-2", 800_u64)] {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    network_id: network_id.into(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_result(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(ReportInferenceAssignmentRequest {
                    device_id: device_id.into(),
                    success: true,
                    completion: Some(format!("completion-from-{}", device_id)),
                    completion_tokens: Some(4),
                    execution_time_ms,
                    error: None,
                }),
            )
            .await
            .unwrap();
        }

        let events = crate::api::ledger::list_ledger_events(
            State(state.clone()),
            axum::extract::Query(crate::api::ledger::ListLedgerEventsQuery {
                network_id: Some(network_id.into()),
                job_id: Some(Uuid::parse_str(&submit.job_id).unwrap()),
                device_id: None,
                limit: Some(20),
            }),
        )
        .await
        .unwrap()
        .0;

        let earned = events
            .events
            .iter()
            .filter(|event| event.event_type == "credits_earned")
            .collect::<Vec<_>>();
        assert_eq!(earned.len(), 2);

        let by_device = earned
            .iter()
            .map(|event| {
                (
                    event
                        .metadata
                        .get("execution_time_ms")
                        .and_then(|value| value.as_u64())
                        .expect("expected execution_time_ms metadata"),
                    event.credits_amount.expect("expected earned credits"),
                )
            })
            .collect::<HashMap<_, _>>();

        let fast_credits = by_device.get(&200).copied().expect("expected fast credits");
        let slow_credits = by_device.get(&800).copied().expect("expected slow credits");
        assert!(fast_credits > slow_credits);

        let summary = crate::api::ledger::get_ledger_summary(
            State(state),
            axum::extract::Query(crate::api::ledger::LedgerSummaryQuery {
                network_id: network_id.into(),
                job_id: Some(Uuid::parse_str(&submit.job_id).unwrap()),
            }),
        )
        .await
        .unwrap()
        .0;
        assert!((summary.total_credits_earned - summary.total_credits_burned).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_job_status_exposes_assignment_accounting_metadata() {
        let network_id = "test-network-assignment-metadata";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "metadata".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let status = get_inference_job_status(State(state), Path(submit.job_id))
            .await
            .unwrap()
            .0;

        assert_eq!(status.assignments.len(), 2);
        assert!(status
            .assignments
            .iter()
            .all(|assignment| assignment.execution_provider.is_some()));
        assert!(status
            .assignments
            .iter()
            .all(|assignment| assignment.assigned_capacity_units >= 1));
        assert_eq!(status.assignments[0].shard_column_start, 0);
        assert_eq!(
            status
                .assignments
                .last()
                .expect("expected final assignment")
                .shard_column_end,
            8192
        );
    }

    #[tokio::test]
    async fn test_completed_job_uses_authoritative_wall_clock_execution_time() {
        let network_id = "test-network-authoritative-wall-clock";
        let state = joined_state(&["worker-1"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "timing".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let started_at = format_time(OffsetDateTime::now_utc() - Duration::seconds(5)).unwrap();
        let conn = state.db.get_conn().unwrap();
        conn.execute(
            "UPDATE inference_jobs SET started_at = ? WHERE job_id = ?",
            params![started_at, &submit.job_id],
        )
        .unwrap();

        let claim = claim_inference_assignment(
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
        .expect("expected assignment");

        let _ = acknowledge_inference_assignment(
            State(state.clone()),
            Path(claim.job_id.clone()),
            Json(AcknowledgeInferenceAssignmentRequest {
                device_id: "worker-1".into(),
            }),
        )
        .await
        .unwrap();

        let _ = report_inference_result(
            State(state.clone()),
            Path(claim.job_id.clone()),
            Json(ReportInferenceAssignmentRequest {
                device_id: "worker-1".into(),
                success: true,
                completion: Some("done".into()),
                completion_tokens: Some(1),
                execution_time_ms: 25,
                error: None,
            }),
        )
        .await
        .unwrap();

        let status = get_inference_job_status(State(state), Path(submit.job_id))
            .await
            .unwrap()
            .0;

        assert_eq!(status.status, "completed");
        assert!(status.execution_time_ms >= 4_000);
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
    async fn test_claim_assignment_applies_submitter_soft_cap_before_second_active_job() {
        let network_id = "test-network-submitter-cap";
        let state = joined_state(&["worker-1", "worker-2", "worker-3"], network_id).await;

        let submit_a1 = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-a1".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let submit_a2 = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-a2".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let submit_b1 = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "job-b1".into(),
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
        assert_eq!(first_claim.job_id, submit_a1.job_id);

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
        assert_eq!(second_claim.job_id, submit_b1.job_id);

        let third_claim = claim_inference_assignment(
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
        .expect("expected third assignment");
        assert_eq!(third_claim.job_id, submit_a2.job_id);
    }

    #[tokio::test]
    async fn test_claim_assignment_applies_model_soft_cap_before_second_active_job() {
        let network_id = "test-network-model-cap";
        let state = joined_state(&["worker-1", "worker-2", "worker-3"], network_id).await;

        let model_x_first = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "job-x1".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let model_x_second = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "job-x2".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let model_y = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-3".into(),
                network_id: network_id.into(),
                model_id: "model-y".into(),
                prompt: "job-y1".into(),
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
        assert_eq!(first_claim.job_id, model_x_first.job_id);

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
        assert_eq!(second_claim.job_id, model_y.job_id);

        let third_claim = claim_inference_assignment(
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
        .expect("expected third assignment");
        assert_eq!(third_claim.job_id, model_x_second.job_id);
    }

    #[tokio::test]
    async fn test_claim_assignment_allows_same_model_progress_when_no_competing_model_waits() {
        let network_id = "test-network-model-routing";
        let state = joined_state(&["worker-1", "worker-2", "worker-3"], network_id).await;

        let model_x_first = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "job-x1".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let model_x_second = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "job-x2".into(),
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
        assert_eq!(first_claim.job_id, model_x_first.job_id);

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
        assert_eq!(second_claim.job_id, model_x_second.job_id);
    }

    #[tokio::test]
    async fn test_claim_assignment_respects_custom_model_soft_cap_divisor() {
        let network_id = "test-network-custom-model-divisor";
        let state = joined_state_with_policy(
            &["worker-1", "worker-2", "worker-3"],
            network_id,
            InferenceSchedulingPolicy {
                submitter_active_job_soft_cap: 1,
                model_active_job_soft_cap_divisor: 1,
                capacity_unit_soft_cap_divisor: 2,
                tier_capacity_units: TierCapacityUnits::default(),
            },
        )
        .await;

        let model_x_first = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "job-x1".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let model_x_second = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "job-x2".into(),
                max_tokens: 16,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let model_y = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-3".into(),
                network_id: network_id.into(),
                model_id: "model-y".into(),
                prompt: "job-y1".into(),
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
        assert_eq!(first_claim.job_id, model_x_first.job_id);

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
        assert_eq!(second_claim.job_id, model_x_second.job_id);
        assert_ne!(second_claim.job_id, model_y.job_id);
    }

    #[tokio::test]
    async fn test_claim_assignment_prefers_lower_capacity_model_when_model_counts_tie() {
        let network_id = "test-network-capacity-units";
        let scheduling_policy = InferenceSchedulingPolicy {
            submitter_active_job_soft_cap: 8,
            model_active_job_soft_cap_divisor: 1,
            capacity_unit_soft_cap_divisor: 2,
            tier_capacity_units: TierCapacityUnits {
                tier0: 1,
                tier1: 2,
                tier2: 4,
                tier3: 8,
                tier4: 16,
            },
        };
        let state = joined_state_with_device_capabilities(
            &[
                (
                    "worker-1",
                    DeviceCapabilities {
                        tier: Tier::Tier4,
                        ..test_capabilities()
                    },
                ),
                (
                    "worker-2",
                    DeviceCapabilities {
                        tier: Tier::Tier0,
                        cpu_cores: 2,
                        ram_mb: 2048,
                        ..test_capabilities()
                    },
                ),
                (
                    "worker-3",
                    DeviceCapabilities {
                        tier: Tier::Tier0,
                        cpu_cores: 2,
                        ram_mb: 2048,
                        ..test_capabilities()
                    },
                ),
            ],
            network_id,
            scheduling_policy,
        )
        .await;

        let model_x_first = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "x1".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let claim_x_first = claim_inference_assignment(
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
        .unwrap();
        assert_eq!(claim_x_first.job_id, model_x_first.job_id);
        let _ = acknowledge_inference_assignment(
            State(state.clone()),
            Path(model_x_first.job_id.clone()),
            Json(AcknowledgeInferenceAssignmentRequest {
                device_id: "worker-1".into(),
            }),
        )
        .await
        .unwrap();

        let model_y_first = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "model-y".into(),
                prompt: "y1".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let claim_y_first = claim_inference_assignment(
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
        .unwrap();
        assert_eq!(claim_y_first.job_id, model_y_first.job_id);
        let _ = acknowledge_inference_assignment(
            State(state.clone()),
            Path(model_y_first.job_id.clone()),
            Json(AcknowledgeInferenceAssignmentRequest {
                device_id: "worker-2".into(),
            }),
        )
        .await
        .unwrap();

        let model_x_second = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "model-x".into(),
                prompt: "x2".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let model_y_second = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
                model_id: "model-y".into(),
                prompt: "y2".into(),
                max_tokens: 8,
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
                device_id: "worker-3".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment
        .unwrap();

        assert_eq!(claim.job_id, model_y_second.job_id);
        assert_ne!(claim.job_id, model_x_second.job_id);
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
