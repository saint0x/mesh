use axum::{
    extract::{Path, State},
    Json,
};
use rusqlite::{params, OptionalExtension};
use std::collections::{HashMap, HashSet};
use time::{Duration, OffsetDateTime};
use tracing::{info, instrument};
use uuid::Uuid;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    AcknowledgeInferenceAssignmentRequest, ClaimInferenceAssignmentRequest,
    ClaimInferenceAssignmentResponse, DownloadInferenceSessionCheckpointResponse,
    InferenceExecutionLease, InferenceExecutionPlan, InferenceJobAssignmentStatus,
    InferenceJobStatusResponse, InferenceSessionCheckpointPayload,
    InferenceSessionCheckpointStatus, InferenceSessionLease, InferenceSessionStatus,
    ReportInferenceAssignmentProgressRequest, ReportInferenceAssignmentRequest,
    SubmitInferenceRequest, SubmitInferenceResponse, UploadInferenceSessionCheckpointRequest,
};
use crate::connectivity::InferenceSchedulingPolicy;
use crate::consumption_policy::{
    compute_consumption_components, quote_consumption, ConsumptionQuoteInput,
};
use crate::credit_policy::{
    compute_credit_policy, AssignmentCreditInput, AssignmentCreditOutput, CreditPolicyInput,
};
use crate::device::DeviceCapabilities;
use crate::model_assets;
use crate::services::{
    device_metadata_from_capabilities, network_service, refresh_decode_plan_for_job,
    select_claim_assignment_id, ExecutionPlanner, PlannerDeviceMetadata,
};
use crate::state::AppState;

const ASSIGNMENT_LEASE_SECS: i64 = 60;

#[derive(Clone)]
struct PersistedAssignment {
    assignment: PersistedLeaseRecord,
    checkpoint: Option<PersistedSessionCheckpointStatus>,
}

#[derive(Clone)]
struct PersistedLeaseRecord {
    lease_id: String,
    job_id: String,
    network_id: String,
    device_id: String,
    model_id: String,
    reserved_credits: f64,
    available_completion_tokens: u32,
    model_size_factor: f64,
    lease_expires_at: String,
    execution_plan_json: String,
    active_segment_id: String,
    session_id: String,
    session_status: String,
    session_active_segment_id: Option<String>,
    kv_owner_device_id: String,
    kv_transfer_policy: String,
    kv_sequence_position: Option<u32>,
    latest_batch_size: Option<u32>,
    latest_active_decode_sessions: Option<u32>,
    latest_batch_kv_tokens: Option<u32>,
    latest_deferred_decode_sessions: Option<u32>,
    lease_target_session_count: Option<u32>,
    lease_target_batch_size: Option<u32>,
    kv_checkpoint_device_id: Option<String>,
    kv_checkpoint_created_at: Option<String>,
    session_updated_at: String,
    replica_status: String,
    replica_active_segment_id: Option<String>,
    replica_kv_sequence_position: Option<u32>,
    replica_checkpoint_created_at: Option<String>,
    replica_updated_at: String,
    replica_last_error: Option<String>,
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
    time_to_first_token_ms: Option<u64>,
    active_segment_id: Option<String>,
    reserved_credits: f64,
    settled_credits: f64,
    released_credits: f64,
    available_completion_tokens: u32,
    model_size_factor: f64,
    error: Option<String>,
    assignments: Vec<InferenceJobAssignmentStatus>,
    execution_plan: Option<InferenceExecutionPlan>,
    session: Option<PersistedSessionStatus>,
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
    execution_plan: Option<InferenceExecutionPlan>,
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

#[derive(Clone)]
struct PersistedSessionStatus {
    session_id: String,
    status: String,
    active_segment_id: Option<String>,
    kv_owner_device_id: String,
    kv_transfer_policy: String,
    kv_sequence_position: Option<u32>,
    latest_batch_size: Option<u32>,
    latest_active_decode_sessions: Option<u32>,
    latest_batch_kv_tokens: Option<u32>,
    latest_deferred_decode_sessions: Option<u32>,
    lease_target_session_count: Option<u32>,
    lease_target_batch_size: Option<u32>,
    kv_checkpoint_device_id: Option<String>,
    kv_checkpoint_created_at: Option<String>,
    updated_at: String,
    last_error: Option<String>,
    checkpoint: Option<PersistedSessionCheckpointStatus>,
    replicas: Vec<PersistedSessionReplicaStatus>,
    recent_decode_batches: Vec<PersistedDecodeBatchEvent>,
}

#[derive(Clone)]
struct PersistedSessionCheckpointStatus {
    checkpoint_id: String,
    source_device_id: String,
    source_segment_id: String,
    phase: String,
    kv_sequence_position: u32,
    size_bytes: u64,
    sha256: String,
    created_at: String,
}

#[derive(Clone)]
struct PersistedSessionReplicaStatus {
    device_id: String,
    status: String,
    active_segment_id: Option<String>,
    kv_sequence_position: Option<u32>,
    checkpoint_created_at: Option<String>,
    updated_at: String,
    last_error: Option<String>,
}

#[derive(Clone)]
struct PersistedDecodeBatchEvent {
    event_id: i64,
    session_id: String,
    job_id: String,
    network_id: String,
    device_id: String,
    segment_id: String,
    completion_tokens: u32,
    execution_time_ms: u64,
    batch_size: Option<u32>,
    active_decode_sessions: Option<u32>,
    batch_kv_tokens: Option<u32>,
    deferred_decode_sessions: Option<u32>,
    kv_cache_seq_len: Option<u32>,
    observed_at: String,
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
    let topology_for_submit = topology.clone();
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
        let device_metadata = topology_for_submit
            .workers
            .iter()
            .map(|worker| load_device_assignment_metadata_from_connection(
                &conn,
                &request.network_id,
                &worker.device_id,
                &scheduling_policy,
            ))
            .collect::<ApiResult<Vec<_>>>()?;
        let execution_plan = ExecutionPlanner::plan(
            &request,
            &prompt_tokens,
            &topology_for_submit,
            &scheduling_policy,
            &device_metadata,
        )?;
        let execution_plan_json = serde_json::to_string(&execution_plan)
            .map_err(|e| ApiError::Internal(format!("Failed to serialize execution plan: {}", e)))?;
        let initial_participants = execution_plan
            .segments
            .iter()
            .find(|segment| segment.segment_id == execution_plan.initial_segment_id)
            .map(|segment| {
                segment
                    .participant_device_ids
                    .iter()
                    .cloned()
                    .collect::<HashSet<_>>()
            })
            .ok_or_else(|| {
                ApiError::Internal(format!(
                    "Initial segment {} missing from execution plan {}",
                    execution_plan.initial_segment_id, execution_plan.plan_id
                ))
            })?;

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
                reserved_credits, available_completion_tokens, model_size_factor, execution_plan_json,
                active_segment_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'dispatched', ?, ?, ?, ?, ?, ?, ?, ?)
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
                execution_plan_json,
                execution_plan.initial_segment_id,
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let initial_segment = execution_plan
            .segments
            .iter()
            .find(|segment| segment.segment_id == execution_plan.initial_segment_id)
            .ok_or_else(|| {
                ApiError::Internal(format!(
                    "Initial segment {} missing from execution plan {}",
                    execution_plan.initial_segment_id, execution_plan.plan_id
                ))
            })?;
        let initial_group = execution_plan
            .execution_groups
            .iter()
            .find(|group| group.group_id == initial_segment.execution_group_id)
            .ok_or_else(|| {
                ApiError::Internal(format!(
                    "Execution group {} missing from execution plan {}",
                    initial_segment.execution_group_id, execution_plan.plan_id
                ))
            })?;
        tx.execute(
            r#"
            INSERT INTO inference_sessions (
                session_id, job_id, network_id, model_id, status, active_segment_id,
                kv_owner_device_id, kv_transfer_policy, kv_sequence_position,
                latest_batch_size, latest_active_decode_sessions,
                latest_batch_kv_tokens, latest_deferred_decode_sessions,
                kv_checkpoint_device_id, kv_checkpoint_created_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'prefill_pending', ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)
            "#,
            params![
                &initial_segment.session_id,
                &persisted_job_id,
                &request.network_id,
                &request.model_id,
                &initial_segment.segment_id,
                &initial_segment.kv_owner_device_id,
                serialize_kv_transfer_policy(initial_group.kv_transfer_policy)?,
                &now,
                &now,
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        for worker in &workers {
            tx.execute(
                r#"
                INSERT INTO inference_session_replicas (
                    session_id, device_id, job_id, status, active_segment_id, kv_sequence_position,
                    checkpoint_created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL)
                "#,
                params![
                    &initial_segment.session_id,
                    &worker.device_id,
                    &persisted_job_id,
                    if initial_participants.contains(&worker.device_id) {
                        "prefill_pending"
                    } else {
                        "waiting"
                    },
                    if initial_participants.contains(&worker.device_id) {
                        Some(initial_segment.segment_id.as_str())
                    } else {
                        None
                    },
                    &now,
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }
        sync_serving_groups(
            &tx,
            &persisted_job_id,
            &request.network_id,
            &request.model_id,
            &execution_plan,
            &now,
        )?;
        if let Some(decode_segment) = execution_plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, crate::api::types::ExecutionPhase::Decode))
        {
            upsert_decode_queue(
                &tx,
                &decode_segment.session_id,
                &persisted_job_id,
                &request.network_id,
                &decode_segment.segment_id,
                &decode_segment.execution_group_id,
                "blocked_on_prefill",
                None,
                None,
                None,
                None,
                None,
                None,
                &now,
            )?;
        }

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

        for (worker, assignment_metadata) in workers.iter().zip(device_metadata.iter()) {
            let assignment_id = Uuid::new_v4().to_string();
            tx.execute(
                r#"
                INSERT INTO inference_job_assignments (
                    assignment_id, job_id, network_id, device_id, ring_position, status, assigned_at,
                    shard_column_start, shard_column_end, assigned_capacity_units, execution_provider,
                    active_segment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
                params![
                    &assignment_id,
                    &persisted_job_id,
                    &request.network_id,
                    &worker.device_id,
                    worker.position as i64,
                    if initial_participants.contains(&worker.device_id) {
                        "pending"
                    } else {
                        "waiting"
                    },
                    &now,
                    worker.shard.column_range.0 as i64,
                    worker.shard.column_range.1 as i64,
                    assignment_metadata.assigned_capacity_units as i64,
                    assignment_metadata.execution_provider,
                    if initial_participants.contains(&worker.device_id) {
                        Some(execution_plan.initial_segment_id.as_str())
                    } else {
                        None
                    },
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }

        tx.commit()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        Ok::<_, ApiError>((
            consumption_quote.total_credits,
            request.max_tokens,
            execution_plan,
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
        execution_plan: Some(reservation.2),
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

    let execution_lease = assignment.map(build_execution_lease).transpose()?;

    Ok(Json(ClaimInferenceAssignmentResponse {
        success: true,
        assignment: execution_lease,
    }))
}

fn build_execution_lease(record: PersistedAssignment) -> ApiResult<InferenceExecutionLease> {
    let PersistedAssignment {
        assignment,
        checkpoint,
    } = record;
    let device_id = assignment.device_id.clone();
    let execution_plan = load_execution_plan_json(&assignment.execution_plan_json)?;
    let active_segment = execution_plan
        .segments
        .iter()
        .find(|segment| segment.segment_id == assignment.active_segment_id)
        .cloned()
        .ok_or_else(|| {
            ApiError::Internal(format!(
                "Active segment {} missing from plan for job {}",
                assignment.active_segment_id, assignment.job_id
            ))
        })?;
    Ok(InferenceExecutionLease {
        lease_id: assignment.lease_id,
        job_id: assignment.job_id.clone(),
        network_id: assignment.network_id,
        device_id,
        model_id: assignment.model_id.clone(),
        reserved_credits: assignment.reserved_credits,
        available_completion_tokens: assignment.available_completion_tokens,
        model_size_factor: assignment.model_size_factor,
        lease_expires_at: assignment.lease_expires_at,
        execution_plan,
        active_segment,
        session: InferenceSessionLease {
            session_id: assignment.session_id,
            status: assignment.session_status,
            active_segment_id: assignment.session_active_segment_id,
            kv_owner_device_id: assignment.kv_owner_device_id,
            kv_transfer_policy: parse_kv_transfer_policy(&assignment.kv_transfer_policy)?,
            kv_sequence_position: assignment.kv_sequence_position,
            latest_batch_size: assignment.latest_batch_size,
            latest_active_decode_sessions: assignment.latest_active_decode_sessions,
            latest_batch_kv_tokens: assignment.latest_batch_kv_tokens,
            latest_deferred_decode_sessions: assignment.latest_deferred_decode_sessions,
            lease_target_session_count: assignment.lease_target_session_count,
            lease_target_batch_size: assignment.lease_target_batch_size,
            kv_checkpoint_device_id: assignment.kv_checkpoint_device_id,
            kv_checkpoint_created_at: assignment.kv_checkpoint_created_at,
            updated_at: assignment.session_updated_at,
            checkpoint: checkpoint
                .map(convert_persisted_session_checkpoint_status)
                .transpose()?,
            local_replica: Some(crate::api::types::InferenceSessionReplicaStatus {
                device_id: assignment.device_id,
                status: assignment.replica_status,
                active_segment_id: assignment.replica_active_segment_id,
                kv_sequence_position: assignment.replica_kv_sequence_position,
                checkpoint_created_at: assignment.replica_checkpoint_created_at,
                updated_at: assignment.replica_updated_at,
                last_error: assignment.replica_last_error,
            }),
        },
    })
}

fn convert_persisted_session_checkpoint_status(
    checkpoint: PersistedSessionCheckpointStatus,
) -> ApiResult<InferenceSessionCheckpointStatus> {
    Ok(InferenceSessionCheckpointStatus {
        checkpoint_id: checkpoint.checkpoint_id,
        source_device_id: checkpoint.source_device_id,
        source_segment_id: checkpoint.source_segment_id,
        phase: parse_execution_phase(&checkpoint.phase)?,
        kv_sequence_position: checkpoint.kv_sequence_position,
        size_bytes: checkpoint.size_bytes,
        sha256: checkpoint.sha256,
        created_at: checkpoint.created_at,
    })
}

fn convert_persisted_decode_batch_event(
    event: PersistedDecodeBatchEvent,
) -> crate::api::types::DecodeBatchEventStatus {
    crate::api::types::DecodeBatchEventStatus {
        event_id: event.event_id,
        session_id: event.session_id,
        job_id: event.job_id,
        network_id: event.network_id,
        device_id: event.device_id,
        segment_id: event.segment_id,
        completion_tokens: event.completion_tokens,
        execution_time_ms: event.execution_time_ms,
        batch_size: event.batch_size,
        active_decode_sessions: event.active_decode_sessions,
        batch_kv_tokens: event.batch_kv_tokens,
        deferred_decode_sessions: event.deferred_decode_sessions,
        kv_cache_seq_len: event.kv_cache_seq_len,
        observed_at: event.observed_at,
    }
}

fn active_segment<'a>(
    plan: &'a InferenceExecutionPlan,
    segment_id: &str,
) -> ApiResult<&'a crate::api::types::ExecutionSegment> {
    plan.segments
        .iter()
        .find(|segment| segment.segment_id == segment_id)
        .ok_or_else(|| {
            ApiError::Internal(format!(
                "Active segment {} missing from execution plan {}",
                segment_id, plan.plan_id
            ))
        })
}

fn next_segment_id(plan: &InferenceExecutionPlan, segment_id: &str) -> Option<String> {
    let index = plan
        .segments
        .iter()
        .position(|segment| segment.segment_id == segment_id)?;
    plan.segments
        .get(index + 1)
        .map(|segment| segment.segment_id.clone())
}

fn participants_for_segment(
    plan: &InferenceExecutionPlan,
    segment_id: &str,
) -> ApiResult<HashSet<String>> {
    Ok(active_segment(plan, segment_id)?
        .participant_device_ids
        .iter()
        .cloned()
        .collect())
}

fn serialize_kv_transfer_policy(policy: crate::api::types::KvTransferPolicy) -> ApiResult<String> {
    serde_json::to_string(&policy)
        .map(|value| value.trim_matches('"').to_string())
        .map_err(|e| ApiError::Internal(format!("Failed to serialize kv transfer policy: {}", e)))
}

fn parse_kv_transfer_policy(value: &str) -> ApiResult<crate::api::types::KvTransferPolicy> {
    let normalized = if value.starts_with('"') {
        value.to_string()
    } else {
        format!("\"{}\"", value)
    };
    serde_json::from_str(&normalized)
        .map_err(|e| ApiError::Internal(format!("Failed to parse kv transfer policy: {}", e)))
}

fn serialize_execution_phase(phase: crate::api::types::ExecutionPhase) -> &'static str {
    match phase {
        crate::api::types::ExecutionPhase::Prefill => "prefill",
        crate::api::types::ExecutionPhase::Decode => "decode",
    }
}

fn parse_execution_phase(value: &str) -> ApiResult<crate::api::types::ExecutionPhase> {
    match value {
        "prefill" => Ok(crate::api::types::ExecutionPhase::Prefill),
        "decode" => Ok(crate::api::types::ExecutionPhase::Decode),
        other => Err(ApiError::Internal(format!(
            "Failed to parse execution phase: {}",
            other
        ))),
    }
}

fn sync_serving_groups(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
    network_id: &str,
    model_id: &str,
    plan: &InferenceExecutionPlan,
    now: &str,
) -> ApiResult<()> {
    let active_group_ids = plan
        .execution_groups
        .iter()
        .map(|group| group.group_id.as_str())
        .collect::<Vec<_>>();

    for group in &plan.execution_groups {
        let phase = serialize_execution_phase(group.phase);
        let participant_ids = plan
            .segments
            .iter()
            .find(|segment| segment.execution_group_id == group.group_id)
            .map(|segment| {
                segment
                    .participant_device_ids
                    .iter()
                    .cloned()
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        for member in &group.members {
            let status = match group.phase {
                crate::api::types::ExecutionPhase::Prefill => {
                    if plan.initial_segment_id.contains("prefill") {
                        "prefill_member"
                    } else {
                        "prefill_complete"
                    }
                }
                crate::api::types::ExecutionPhase::Decode => {
                    if participant_ids.contains(&member.device_id) {
                        "decode_member"
                    } else {
                        "standby"
                    }
                }
            };
            conn.execute(
                r#"
                INSERT INTO inference_serving_groups (
                    group_id, session_id, job_id, network_id, model_id, phase, device_id,
                    ring_position, shard_column_start, shard_column_end, assigned_capacity_units,
                    execution_provider, status, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(group_id, device_id) DO UPDATE SET
                    ring_position = excluded.ring_position,
                    shard_column_start = excluded.shard_column_start,
                    shard_column_end = excluded.shard_column_end,
                    assigned_capacity_units = excluded.assigned_capacity_units,
                    execution_provider = excluded.execution_provider,
                    status = excluded.status,
                    last_error = NULL,
                    updated_at = excluded.updated_at
                "#,
                params![
                    &group.group_id,
                    plan.segments
                        .iter()
                        .find(|segment| segment.execution_group_id == group.group_id)
                        .map(|segment| segment.session_id.as_str())
                        .unwrap_or_default(),
                    job_id,
                    network_id,
                    model_id,
                    phase,
                    &member.device_id,
                    i64::from(member.ring_position),
                    i64::from(member.shard.column_start),
                    i64::from(member.shard.column_end),
                    i64::from(member.assigned_capacity_units),
                    &member.execution_provider,
                    status,
                    now
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }
    }

    if !active_group_ids.is_empty() {
        let placeholders = std::iter::repeat_n("?", active_group_ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "UPDATE inference_serving_groups
             SET status = 'superseded', updated_at = ?
             WHERE job_id = ?
               AND group_id NOT IN ({})",
            placeholders
        );
        let mut values: Vec<&dyn rusqlite::ToSql> = vec![&now, &job_id];
        for group_id in &active_group_ids {
            values.push(group_id);
        }
        conn.execute(&sql, rusqlite::params_from_iter(values))
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    }

    Ok(())
}

fn upsert_decode_queue(
    conn: &rusqlite::Transaction<'_>,
    session_id: &str,
    job_id: &str,
    network_id: &str,
    segment_id: &str,
    group_id: &str,
    status: &str,
    ready_at: Option<&str>,
    lease_owner_device_id: Option<&str>,
    lease_expires_at: Option<&str>,
    lease_target_session_count: Option<u32>,
    lease_target_batch_size: Option<u32>,
    last_error: Option<&str>,
    now: &str,
) -> ApiResult<()> {
    conn.execute(
        r#"
        INSERT INTO inference_decode_queue (
            session_id, job_id, network_id, segment_id, group_id, status, ready_at,
            lease_owner_device_id, lease_expires_at, lease_target_session_count,
            lease_target_batch_size, last_error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            segment_id = excluded.segment_id,
            group_id = excluded.group_id,
            status = excluded.status,
            ready_at = excluded.ready_at,
            lease_owner_device_id = excluded.lease_owner_device_id,
            lease_expires_at = excluded.lease_expires_at,
            lease_target_session_count = excluded.lease_target_session_count,
            lease_target_batch_size = excluded.lease_target_batch_size,
            last_error = excluded.last_error,
            updated_at = excluded.updated_at
        "#,
        params![
            session_id,
            job_id,
            network_id,
            segment_id,
            group_id,
            status,
            ready_at,
            lease_owner_device_id,
            lease_expires_at,
            lease_target_session_count.map(i64::from),
            lease_target_batch_size.map(i64::from),
            last_error,
            now
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn compute_decode_lease_targets(
    conn: &rusqlite::Transaction<'_>,
    network_id: &str,
    group_id: &str,
) -> ApiResult<(u32, u32)> {
    let (target_session_count, observed_peak): (u32, Option<u32>) = conn
        .query_row(
            r#"
            SELECT
                COUNT(*),
                MAX(COALESCE(s.latest_active_decode_sessions, s.latest_batch_size))
            FROM inference_decode_queue dq
            LEFT JOIN inference_sessions s
              ON s.session_id = dq.session_id
            WHERE dq.network_id = ?1
              AND dq.group_id = ?2
              AND dq.status IN ('ready', 'leased', 'active')
            "#,
            params![network_id, group_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)? as u32,
                    row.get::<_, Option<i64>>(1)?.map(|v| v as u32),
                ))
            },
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let target_session_count = target_session_count.max(1);
    let target_batch_size = observed_peak
        .map(|peak| peak.min(target_session_count).max(1))
        .unwrap_or(target_session_count);
    Ok((target_session_count, target_batch_size))
}

fn load_latest_session_checkpoint_status(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> ApiResult<Option<PersistedSessionCheckpointStatus>> {
    conn.query_row(
        r#"
        SELECT checkpoint_id, source_device_id, source_segment_id, phase,
               kv_sequence_position, size_bytes, checkpoint_sha256, created_at
        FROM inference_session_checkpoints
        WHERE session_id = ?
        ORDER BY created_at DESC, checkpoint_id DESC
        LIMIT 1
        "#,
        params![session_id],
        |row| {
            Ok(PersistedSessionCheckpointStatus {
                checkpoint_id: row.get(0)?,
                source_device_id: row.get(1)?,
                source_segment_id: row.get(2)?,
                phase: row.get(3)?,
                kv_sequence_position: row.get::<_, i64>(4)? as u32,
                size_bytes: row.get::<_, i64>(5)? as u64,
                sha256: row.get(6)?,
                created_at: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_recent_decode_batch_events(
    conn: &rusqlite::Connection,
    session_id: &str,
    limit: usize,
) -> ApiResult<Vec<PersistedDecodeBatchEvent>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT event_id, session_id, job_id, network_id, device_id, segment_id,
                   completion_tokens, execution_time_ms, batch_size, active_decode_sessions,
                   batch_kv_tokens, deferred_decode_sessions, kv_cache_seq_len, observed_at
            FROM inference_decode_batch_events
            WHERE session_id = ?
            ORDER BY observed_at DESC, event_id DESC
            LIMIT ?
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let rows = stmt.query_map(params![session_id, limit as i64], |row| {
        Ok(PersistedDecodeBatchEvent {
            event_id: row.get(0)?,
            session_id: row.get(1)?,
            job_id: row.get(2)?,
            network_id: row.get(3)?,
            device_id: row.get(4)?,
            segment_id: row.get(5)?,
            completion_tokens: row.get::<_, i64>(6)? as u32,
            execution_time_ms: row.get::<_, i64>(7)? as u64,
            batch_size: row.get::<_, Option<i64>>(8)?.map(|v| v as u32),
            active_decode_sessions: row.get::<_, Option<i64>>(9)?.map(|v| v as u32),
            batch_kv_tokens: row.get::<_, Option<i64>>(10)?.map(|v| v as u32),
            deferred_decode_sessions: row.get::<_, Option<i64>>(11)?.map(|v| v as u32),
            kv_cache_seq_len: row.get::<_, Option<i64>>(12)?.map(|v| v as u32),
            observed_at: row.get(13)?,
        })
    })
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_latest_session_checkpoint_payload(
    db: &crate::db::Database,
    job_id: &str,
    session_id: &str,
) -> ApiResult<Option<InferenceSessionCheckpointPayload>> {
    let conn = db.get_conn()?;
    conn.query_row(
        r#"
        SELECT checkpoint_id, source_device_id, source_segment_id, phase,
               kv_sequence_position, size_bytes, checkpoint_sha256, created_at,
               checkpoint_bytes
        FROM inference_session_checkpoints
        WHERE job_id = ? AND session_id = ?
        ORDER BY created_at DESC, checkpoint_id DESC
        LIMIT 1
        "#,
        params![job_id, session_id],
        |row| {
            let metadata = PersistedSessionCheckpointStatus {
                checkpoint_id: row.get(0)?,
                source_device_id: row.get(1)?,
                source_segment_id: row.get(2)?,
                phase: row.get(3)?,
                kv_sequence_position: row.get::<_, i64>(4)? as u32,
                size_bytes: row.get::<_, i64>(5)? as u64,
                sha256: row.get(6)?,
                created_at: row.get(7)?,
            };
            let bytes: Vec<u8> = row.get(8)?;
            Ok(InferenceSessionCheckpointPayload {
                metadata: convert_persisted_session_checkpoint_status(metadata).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::other(e.to_string())),
                    )
                })?,
                checkpoint_hex: hex::encode(bytes),
            })
        },
    )
    .optional()
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
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

    let app_state = state.clone();
    let request = req.clone();
    tokio::task::spawn_blocking(move || report_assignment_progress(&app_state, &job_id, &request))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(serde_json::json!({ "success": true })))
}

#[instrument(skip(state))]
pub async fn upload_inference_session_checkpoint(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    Json(req): Json<UploadInferenceSessionCheckpointRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    if req.device_id.is_empty() || req.session_id.is_empty() || req.segment_id.is_empty() {
        return Err(ApiError::BadRequest(
            "device_id, session_id, and segment_id must be provided".to_string(),
        ));
    }

    let db = state.db.clone();
    let request = req.clone();
    tokio::task::spawn_blocking(move || store_session_checkpoint(&db, &job_id, &request))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(serde_json::json!({ "success": true })))
}

#[instrument(skip(state))]
pub async fn download_inference_session_checkpoint(
    State(state): State<AppState>,
    Path((job_id, session_id)): Path<(String, String)>,
) -> ApiResult<Json<DownloadInferenceSessionCheckpointResponse>> {
    if job_id.is_empty() || session_id.is_empty() {
        return Err(ApiError::BadRequest(
            "job_id and session_id must be provided".to_string(),
        ));
    }

    let db = state.db.clone();
    let checkpoint = tokio::task::spawn_blocking(move || {
        load_latest_session_checkpoint_payload(&db, &job_id, &session_id)
    })
    .await
    .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(DownloadInferenceSessionCheckpointResponse {
        success: true,
        checkpoint,
    }))
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
    let session = status
        .session
        .map(|session| {
            Ok::<InferenceSessionStatus, ApiError>(InferenceSessionStatus {
                session_id: session.session_id,
                status: session.status,
                active_segment_id: session.active_segment_id,
                kv_owner_device_id: session.kv_owner_device_id,
                kv_transfer_policy: parse_kv_transfer_policy(&session.kv_transfer_policy)?,
                kv_sequence_position: session.kv_sequence_position,
                latest_batch_size: session.latest_batch_size,
                latest_active_decode_sessions: session.latest_active_decode_sessions,
                latest_batch_kv_tokens: session.latest_batch_kv_tokens,
                latest_deferred_decode_sessions: session.latest_deferred_decode_sessions,
                lease_target_session_count: session.lease_target_session_count,
                lease_target_batch_size: session.lease_target_batch_size,
                kv_checkpoint_device_id: session.kv_checkpoint_device_id,
                kv_checkpoint_created_at: session.kv_checkpoint_created_at,
                updated_at: session.updated_at,
                last_error: session.last_error,
                checkpoint: session
                    .checkpoint
                    .map(convert_persisted_session_checkpoint_status)
                    .transpose()?,
                replicas: session
                    .replicas
                    .into_iter()
                    .map(|replica| crate::api::types::InferenceSessionReplicaStatus {
                        device_id: replica.device_id,
                        status: replica.status,
                        active_segment_id: replica.active_segment_id,
                        kv_sequence_position: replica.kv_sequence_position,
                        checkpoint_created_at: replica.checkpoint_created_at,
                        updated_at: replica.updated_at,
                        last_error: replica.last_error,
                    })
                    .collect(),
                recent_decode_batches: session
                    .recent_decode_batches
                    .into_iter()
                    .map(convert_persisted_decode_batch_event)
                    .collect(),
            })
        })
        .transpose()?;

    Ok(Json(InferenceJobStatusResponse {
        success: true,
        job_id: status.job_id,
        network_id: status.network_id,
        model_id: status.model_id,
        status: status.status,
        completion: status.completion,
        completion_tokens: status.completion_tokens,
        execution_time_ms: status.execution_time_ms,
        time_to_first_token_ms: status.time_to_first_token_ms,
        active_segment_id: status.active_segment_id,
        reserved_credits: status.reserved_credits,
        settled_credits: status.settled_credits,
        released_credits: status.released_credits,
        available_completion_tokens: status.available_completion_tokens,
        model_size_factor: status.model_size_factor,
        error: status.error,
        assignments: status.assignments,
        execution_plan: status.execution_plan,
        session,
    }))
}

#[instrument(skip(state))]
pub async fn cancel_inference_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    if job_id.is_empty() {
        return Err(ApiError::BadRequest("job_id cannot be empty".to_string()));
    }

    let db = state.db.clone();
    let result = tokio::task::spawn_blocking(move || cancel_job(&db, &job_id))
        .await
        .map_err(|e| ApiError::Internal(format!("Task join error: {}", e)))??;

    Ok(Json(serde_json::json!({
        "success": true,
        "job_id": result.job_id,
        "status": result.status,
        "released_credits": result.released_credits,
    })))
}

#[derive(Debug, Clone)]
struct CancelledJobResult {
    job_id: String,
    status: String,
    released_credits: f64,
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

fn cancel_job(db: &crate::db::Database, job_id: &str) -> ApiResult<CancelledJobResult> {
    let mut conn = db.get_conn()?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let job_context = load_job_context(&tx, job_id)?;
    let current_state = load_job_settlement_state(&tx, job_id)?;
    let current_status: String = tx
        .query_row(
            "SELECT status FROM inference_jobs WHERE job_id = ?",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if !matches!(current_status.as_str(), "dispatched" | "running") {
        return Err(ApiError::Conflict(format!(
            "Job {} cannot be cancelled from status {}",
            job_id, current_status
        )));
    }

    let now = now_rfc3339()?;
    let released_credits = if current_state.released_credits > f64::EPSILON {
        current_state.released_credits
    } else {
        job_context.reserved_credits
    };

    tx.execute(
        r#"
        UPDATE inference_job_assignments
        SET status = CASE
                WHEN status IN ('pending', 'leased', 'acknowledged') THEN 'cancelled'
                ELSE status
            END,
            completed_at = COALESCE(completed_at, ?),
            lease_expires_at = NULL,
            failure_reason = CASE
                WHEN status IN ('pending', 'leased', 'acknowledged') THEN 'cancelled'
                ELSE failure_reason
            END
        WHERE job_id = ?
        "#,
        params![&now, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    tx.execute(
        r#"
        UPDATE inference_jobs
        SET status = 'cancelled',
            completed_at = COALESCE(completed_at, ?),
            updated_at = ?,
            error = COALESCE(error, 'cancelled by operator'),
            released_credits = CASE
                WHEN released_credits > 0 THEN released_credits
                ELSE ?
            END
        WHERE job_id = ?
        "#,
        params![&now, &now, released_credits, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_sessions
        SET status = 'cancelled',
            active_segment_id = NULL,
            updated_at = ?,
            last_error = COALESCE(last_error, 'cancelled by operator')
        WHERE job_id = ?
        "#,
        params![&now, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_session_replicas
        SET status = 'cancelled',
            active_segment_id = NULL,
            updated_at = ?,
            last_error = COALESCE(last_error, 'cancelled by operator')
        WHERE job_id = ?
        "#,
        params![&now, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_serving_groups
        SET status = 'cancelled',
            updated_at = ?,
            last_error = COALESCE(last_error, 'cancelled by operator')
        WHERE job_id = ?
        "#,
        params![&now, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_decode_queue
        SET status = 'cancelled',
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            last_error = COALESCE(last_error, 'cancelled by operator'),
            updated_at = ?
        WHERE job_id = ?
        "#,
        params![&now, job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if current_state.released_credits <= f64::EPSILON {
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
                "release_reason": "job_cancelled",
            }),
        )?;
    }

    insert_ledger_event(
        &tx,
        &job_context.network_id,
        "job_cancelled",
        Some(job_id),
        Some(&job_context.submitted_by_device_id),
        None,
        serde_json::json!({
            "credit_model": "consumption_v1",
            "model_id": job_context.model_id,
            "reserved_credits_released": released_credits,
            "cancelled_by_device_id": job_context.submitted_by_device_id,
        }),
    )?;

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(CancelledJobResult {
        job_id: job_id.to_string(),
        status: "cancelled".to_string(),
        released_credits,
    })
}

fn load_device_assignment_metadata_from_connection(
    conn: &rusqlite::Connection,
    network_id: &str,
    device_id: &str,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> ApiResult<PlannerDeviceMetadata> {
    let capabilities_json: String = conn
        .query_row(
            "SELECT capabilities FROM devices WHERE network_id = ? AND device_id = ?",
            params![network_id, device_id],
            |row| row.get(0),
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let capabilities: DeviceCapabilities = serde_json::from_str(&capabilities_json)
        .map_err(|e| ApiError::Internal(format!("Failed to parse device capabilities: {}", e)))?;
    Ok(device_metadata_from_capabilities(
        scheduling_policy,
        &capabilities,
    ))
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

    let selected_assignment_id = select_claim_assignment_id(&tx, req, &scheduling_policy)?;

    let mut row = if let Some(selected_assignment_id) = selected_assignment_id {
        tx.query_row(
            r#"
            SELECT
                a.assignment_id, a.job_id, a.network_id, a.device_id, a.ring_position,
                j.model_id, j.prompt_tokens, j.max_tokens, j.temperature, j.top_p,
                j.reserved_credits, j.available_completion_tokens, j.model_size_factor,
                j.execution_plan_json, j.active_segment_id,
                s.session_id, s.status, s.active_segment_id, s.kv_owner_device_id,
                s.kv_transfer_policy, s.kv_sequence_position, s.latest_batch_size,
                s.latest_active_decode_sessions, s.latest_batch_kv_tokens,
                s.latest_deferred_decode_sessions, dq.lease_target_session_count,
                dq.lease_target_batch_size, s.kv_checkpoint_device_id,
                s.kv_checkpoint_created_at, s.updated_at,
                r.status, r.active_segment_id, r.kv_sequence_position,
                r.checkpoint_created_at, r.updated_at, r.last_error
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            INNER JOIN inference_sessions s ON s.job_id = j.job_id
            LEFT JOIN inference_decode_queue dq ON dq.session_id = s.session_id
            INNER JOIN inference_session_replicas r
                ON r.session_id = s.session_id AND r.device_id = a.device_id
            WHERE a.assignment_id = ?
            "#,
            params![selected_assignment_id],
            |row| {
                Ok(PersistedLeaseRecord {
                    lease_id: row.get(0)?,
                    job_id: row.get(1)?,
                    network_id: row.get(2)?,
                    device_id: row.get(3)?,
                    model_id: row.get(5)?,
                    reserved_credits: row.get(10)?,
                    available_completion_tokens: row.get::<_, i64>(11)? as u32,
                    model_size_factor: row.get(12)?,
                    lease_expires_at: lease_expires.clone(),
                    execution_plan_json: row.get(13)?,
                    active_segment_id: row.get(14)?,
                    session_id: row.get(15)?,
                    session_status: row.get(16)?,
                    session_active_segment_id: row.get(17)?,
                    kv_owner_device_id: row.get(18)?,
                    kv_transfer_policy: row.get(19)?,
                    kv_sequence_position: row.get::<_, Option<i64>>(20)?.map(|v| v as u32),
                    latest_batch_size: row.get::<_, Option<i64>>(21)?.map(|v| v as u32),
                    latest_active_decode_sessions: row.get::<_, Option<i64>>(22)?.map(|v| v as u32),
                    latest_batch_kv_tokens: row.get::<_, Option<i64>>(23)?.map(|v| v as u32),
                    latest_deferred_decode_sessions: row.get::<_, Option<i64>>(24)?.map(|v| v as u32),
                    lease_target_session_count: row.get::<_, Option<i64>>(25)?.map(|v| v as u32),
                    lease_target_batch_size: row.get::<_, Option<i64>>(26)?.map(|v| v as u32),
                    kv_checkpoint_device_id: row.get(27)?,
                    kv_checkpoint_created_at: row.get(28)?,
                    session_updated_at: row.get(29)?,
                    replica_status: row.get(30)?,
                    replica_active_segment_id: row.get(31)?,
                    replica_kv_sequence_position: row.get::<_, Option<i64>>(32)?.map(|v| v as u32),
                    replica_checkpoint_created_at: row.get(33)?,
                    replica_updated_at: row.get(34)?,
                    replica_last_error: row.get(35)?,
                })
            },
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
    } else {
        None
    };

    if let Some(assignment) = row.as_mut() {
        tx.execute(
            r#"
            UPDATE inference_job_assignments
            SET status = 'leased', lease_expires_at = ?
            WHERE assignment_id = ?
            "#,
            params![&lease_expires, &assignment.lease_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        let execution_plan = load_execution_plan_json(&assignment.execution_plan_json)?;
        let active = active_segment(&execution_plan, &assignment.active_segment_id)?;
        tx.execute(
            r#"
            UPDATE inference_serving_groups
            SET status = ?, updated_at = ?, last_error = NULL
            WHERE job_id = ? AND group_id = ? AND device_id = ?
            "#,
            params![
                match active.phase {
                    crate::api::types::ExecutionPhase::Prefill => "prefill_leased",
                    crate::api::types::ExecutionPhase::Decode => "decode_leased",
                },
                &now_str,
                &assignment.job_id,
                &active.execution_group_id,
                &assignment.device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        if matches!(active.phase, crate::api::types::ExecutionPhase::Decode) {
            let (lease_target_session_count, lease_target_batch_size) =
                compute_decode_lease_targets(&tx, &assignment.network_id, &active.execution_group_id)?;
            assignment.lease_target_session_count = Some(lease_target_session_count);
            assignment.lease_target_batch_size = Some(lease_target_batch_size);
            upsert_decode_queue(
                &tx,
                &assignment.session_id,
                &assignment.job_id,
                &assignment.network_id,
                &active.segment_id,
                &active.execution_group_id,
                "leased",
                Some(&now_str),
                Some(&assignment.device_id),
                Some(&lease_expires),
                Some(lease_target_session_count),
                Some(lease_target_batch_size),
                None,
                &now_str,
            )?;
        }
    }

    let row = if let Some(assignment) = row {
        let checkpoint = load_latest_session_checkpoint_status(&tx, &assignment.session_id)?;
        Some(PersistedAssignment {
            assignment,
            checkpoint,
        })
    } else {
        None
    };

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

    let (execution_plan_json, active_segment_id): (Option<String>, Option<String>) = conn
        .query_row(
            "SELECT execution_plan_json, active_segment_id FROM inference_jobs WHERE job_id = ?",
            params![job_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    if let (Some(execution_plan_json), Some(active_segment_id)) =
        (execution_plan_json, active_segment_id)
    {
        let execution_plan = load_execution_plan_json(&execution_plan_json)?;
        let phase = active_segment(&execution_plan, &active_segment_id)?.phase;
        let session_status = match phase {
            crate::api::types::ExecutionPhase::Prefill => "prefill_active",
            crate::api::types::ExecutionPhase::Decode => "decode_active",
        };
        conn.execute(
            r#"
            UPDATE inference_sessions
            SET status = ?, active_segment_id = ?, updated_at = ?
            WHERE job_id = ?
            "#,
            params![session_status, &active_segment_id, &now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        conn.execute(
            r#"
            UPDATE inference_session_replicas
            SET status = ?, active_segment_id = ?, updated_at = ?, last_error = NULL
            WHERE job_id = ? AND device_id = ?
            "#,
            params![session_status, &active_segment_id, &now, job_id, device_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        let group_id = active_segment(&execution_plan, &active_segment_id)?
            .execution_group_id
            .clone();
        conn.execute(
            r#"
            UPDATE inference_serving_groups
            SET status = ?, updated_at = ?, last_error = NULL
            WHERE job_id = ? AND group_id = ? AND device_id = ?
            "#,
            params![
                match phase {
                    crate::api::types::ExecutionPhase::Prefill => "prefill_active",
                    crate::api::types::ExecutionPhase::Decode => "decode_active",
                },
                &now,
                job_id,
                &group_id,
                device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        if matches!(phase, crate::api::types::ExecutionPhase::Decode) {
            conn.execute(
                r#"
                UPDATE inference_decode_queue
                SET status = 'active',
                    lease_owner_device_id = ?,
                    lease_expires_at = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE session_id = (
                    SELECT session_id FROM inference_sessions WHERE job_id = ?
                )
                "#,
                params![device_id, &now, job_id],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }
    }

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
    let prefill_completed_at = req.time_to_first_token_ms.map(|_| now.clone());
    let rows = tx
        .execute(
            r#"
            UPDATE inference_job_assignments
            SET status = ?, completed_at = ?, lease_expires_at = NULL, failure_reason = ?,
                execution_time_ms = ?, reported_completion_tokens = MAX(reported_completion_tokens, ?),
                active_segment_id = NULL, last_completed_segment_id = ?, segment_completed_at = ?
            WHERE job_id = ? AND device_id = ? AND active_segment_id = ? AND status IN ('leased', 'acknowledged')
            "#,
            params![
                assignment_status,
                &now,
                req.error.as_deref(),
                req.execution_time_ms as i64,
                req.completion_tokens.unwrap_or(0) as i64,
                &req.segment_id,
                &now,
                job_id,
                &req.device_id,
                &req.segment_id
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
    let relevant_participants = job_context
        .execution_plan
        .as_ref()
        .map(|plan| participants_for_segment(plan, &req.segment_id))
        .transpose()?
        .unwrap_or_else(|| {
            assignment_states
                .iter()
                .map(|(device_id, _, _)| device_id.clone())
                .collect()
        });
    let assignment_state_map = assignment_states
        .iter()
        .map(|(device_id, status, failure_reason)| {
            (
                device_id.clone(),
                (status.as_str(), failure_reason.as_ref().map(String::as_str)),
            )
        })
        .collect::<HashMap<_, _>>();

    let failed = relevant_participants.iter().find_map(|device_id| {
        assignment_state_map
            .get(device_id)
            .filter(|(status, _)| *status == "failed")
            .map(|(_, failure_reason)| ((*device_id).clone(), failure_reason.map(str::to_string)))
    });
    let all_completed = relevant_participants.iter().all(|device_id| {
        assignment_state_map
            .get(device_id)
            .map(|(status, _)| *status == "completed")
            .unwrap_or(false)
    });

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
    ) = if let Some((_, failure_reason)) = failed {
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
        reconcile_realtime_job_accounting(
            &tx,
            job_id,
            &job_context,
            &settlement_state,
            &relevant_participants,
        )?;

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
            time_to_first_token_ms = COALESCE(time_to_first_token_ms, ?),
            prefill_completed_at = COALESCE(prefill_completed_at, ?),
            active_segment_id = CASE
                WHEN ? IN ('completed', 'failed') THEN NULL
                ELSE active_segment_id
            END,
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
            req.time_to_first_token_ms.map(|v| v as i64),
            prefill_completed_at.as_deref(),
            job_status,
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
    tx.execute(
        r#"
        UPDATE inference_sessions
        SET status = ?,
            active_segment_id = CASE
                WHEN ? IN ('completed', 'failed') THEN NULL
                ELSE active_segment_id
            END,
            kv_sequence_position = COALESCE(?, kv_sequence_position),
            kv_checkpoint_device_id = CASE
                WHEN ? = 'completed' THEN NULL
                ELSE kv_checkpoint_device_id
            END,
            kv_checkpoint_created_at = CASE
                WHEN ? = 'completed' THEN NULL
                ELSE kv_checkpoint_created_at
            END,
            updated_at = ?,
            last_error = ?
        WHERE job_id = ?
        "#,
        params![
            if req.success && all_completed {
                "completed"
            } else if !req.success {
                "failed"
            } else {
                "decode_active"
            },
            job_status,
            req.kv_cache_seq_len.map(i64::from),
            job_status,
            job_status,
            &now,
            error.as_deref().or(req.error.as_deref()),
            job_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_session_replicas
        SET status = ?,
            active_segment_id = NULL,
            kv_sequence_position = COALESCE(?, kv_sequence_position),
            checkpoint_created_at = CASE
                WHEN ? = 'completed' THEN NULL
                ELSE checkpoint_created_at
            END,
            updated_at = ?,
            last_error = ?
        WHERE job_id = ? AND device_id = ?
        "#,
        params![
            if req.success { "completed" } else { "failed" },
            req.kv_cache_seq_len.map(i64::from),
            job_status,
            &now,
            error.as_deref().or(req.error.as_deref()),
            job_id,
            &req.device_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    if let Some(plan) = &job_context.execution_plan {
        if let Ok(segment) = active_segment(plan, &req.segment_id) {
            tx.execute(
                r#"
                UPDATE inference_serving_groups
                SET status = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE job_id = ? AND group_id = ? AND device_id = ?
                "#,
                params![
                    if req.success { "completed" } else { "failed" },
                    &now,
                    error.as_deref().or(req.error.as_deref()),
                    job_id,
                    &segment.execution_group_id,
                    &req.device_id
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }
    }

    if job_status == "completed" {
        tx.execute(
            r#"
            UPDATE inference_job_assignments
            SET status = 'completed',
                completed_at = COALESCE(completed_at, ?),
                active_segment_id = NULL,
                lease_expires_at = NULL,
                failure_reason = NULL
            WHERE job_id = ?
              AND status NOT IN ('completed', 'failed', 'cancelled')
            "#,
            params![&now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_session_replicas
            SET status = 'completed',
                active_segment_id = NULL,
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ?
              AND status != 'failed'
            "#,
            params![&now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_serving_groups
            SET status = 'completed',
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ?
            "#,
            params![&now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_decode_queue
            SET status = 'completed',
                lease_owner_device_id = NULL,
                lease_expires_at = NULL,
                last_error = NULL,
                updated_at = ?
            WHERE job_id = ?
            "#,
            params![&now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    } else if job_status == "failed" {
        tx.execute(
            r#"
            UPDATE inference_decode_queue
            SET status = 'failed',
                lease_owner_device_id = NULL,
                lease_expires_at = NULL,
                last_error = ?,
                updated_at = ?
            WHERE job_id = ?
            "#,
            params![error.as_deref().or(req.error.as_deref()), &now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    }

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(())
}

fn report_assignment_progress(
    state: &AppState,
    job_id: &str,
    req: &ReportInferenceAssignmentProgressRequest,
) -> ApiResult<()> {
    let scheduling_policy = {
        let db = state.db.clone();
        let conn = db.get_conn()?;
        let network_id: String = conn
            .query_row(
                "SELECT network_id FROM inference_jobs WHERE job_id = ?",
                params![job_id],
                |row| row.get(0),
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        network_service::load_network_settings(&db, &network_id)?.scheduling_policy
    };

    let mut conn = state.db.get_conn()?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let prefill_complete = matches!(
        req.event,
        crate::api::types::ProgressEventKind::PrefillComplete
    );
    let segment_completed_at = if prefill_complete {
        Some(now_rfc3339()?)
    } else {
        None
    };

    let updated = tx
        .execute(
            r#"
            UPDATE inference_job_assignments
            SET reported_completion_tokens = MAX(reported_completion_tokens, ?),
                execution_time_ms = MAX(execution_time_ms, ?),
                last_completed_segment_id = CASE
                    WHEN ? = 'prefill_complete' THEN ?
                    ELSE last_completed_segment_id
                END,
                segment_completed_at = CASE
                    WHEN ? = 'prefill_complete' THEN ?
                    ELSE segment_completed_at
                END,
                status = CASE
                    WHEN ? = 'prefill_complete' THEN 'waiting'
                    ELSE status
                END,
                active_segment_id = CASE
                    WHEN ? = 'prefill_complete' THEN NULL
                    ELSE active_segment_id
                END,
                lease_expires_at = CASE
                    WHEN ? = 'prefill_complete' THEN NULL
                    ELSE lease_expires_at
                END
            WHERE job_id = ? AND device_id = ? AND active_segment_id = ? AND status IN ('acknowledged', 'completed')
            "#,
            params![
                req.completion_tokens as i64,
                req.execution_time_ms as i64,
                if prefill_complete {
                    "prefill_complete"
                } else {
                    "decode_progress"
                },
                &req.segment_id,
                segment_completed_at.as_deref(),
                if prefill_complete {
                    "prefill_complete"
                } else {
                    "decode_progress"
                },
                if prefill_complete {
                    "prefill_complete"
                } else {
                    "decode_progress"
                },
                if prefill_complete {
                    "prefill_complete"
                } else {
                    "decode_progress"
                },
                if prefill_complete {
                    "prefill_complete"
                } else {
                    "decode_progress"
                },
                job_id,
                &req.device_id,
                &req.segment_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if updated == 0 {
        return Err(ApiError::NotFound(format!(
            "Assignment not found for progress report on job {} and device {}",
            job_id, req.device_id
        )));
    }

    if prefill_complete {
        let now = now_rfc3339()?;
        let job_context = load_job_context(&tx, job_id)?;
        let mut execution_plan = job_context.execution_plan.clone().ok_or_else(|| {
            ApiError::Internal(format!("Job {} is missing an execution plan", job_id))
        })?;
        let active = active_segment(&execution_plan, &req.segment_id)?;
        let participants = active
            .participant_device_ids
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let all_prefill_participants_completed = load_assignment_segment_completion(&tx, job_id)?
            .into_iter()
            .filter(|(device_id, _, _)| participants.contains(device_id))
            .all(|(_, last_completed_segment_id, _)| {
                last_completed_segment_id.as_deref() == Some(req.segment_id.as_str())
            });

        tx.execute(
            r#"
            UPDATE inference_jobs
            SET time_to_first_token_ms = COALESCE(time_to_first_token_ms, ?),
                prefill_completed_at = COALESCE(prefill_completed_at, ?),
                updated_at = ?
            WHERE job_id = ?
            "#,
            params![
                req.time_to_first_token_ms.map(|v| v as i64),
                &now,
                &now,
                job_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_sessions
            SET status = CASE
                    WHEN ? THEN 'decode_ready'
                    ELSE 'prefill_active'
                END,
                active_segment_id = CASE
                    WHEN ? THEN ?
                    ELSE active_segment_id
                END,
                kv_sequence_position = COALESCE(?, kv_sequence_position),
                latest_batch_size = COALESCE(?, latest_batch_size),
                latest_active_decode_sessions = COALESCE(?, latest_active_decode_sessions),
                latest_batch_kv_tokens = COALESCE(?, latest_batch_kv_tokens),
                latest_deferred_decode_sessions = COALESCE(?, latest_deferred_decode_sessions),
                kv_checkpoint_device_id = ?,
                kv_checkpoint_created_at = ?,
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ?
            "#,
            params![
                all_prefill_participants_completed,
                all_prefill_participants_completed,
                next_segment_id(&execution_plan, &req.segment_id).as_deref(),
                req.kv_cache_seq_len.map(i64::from),
                req.batch_size.map(i64::from),
                req.active_decode_sessions.map(i64::from),
                req.batch_kv_tokens.map(i64::from),
                req.deferred_decode_sessions.map(i64::from),
                &req.device_id,
                &now,
                &now,
                job_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_session_replicas
            SET status = 'prefill_complete',
                active_segment_id = NULL,
                kv_sequence_position = COALESCE(?, kv_sequence_position),
                checkpoint_created_at = ?,
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ? AND device_id = ?
            "#,
            params![
                req.kv_cache_seq_len.map(i64::from),
                &now,
                &now,
                job_id,
                &req.device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_serving_groups
            SET status = 'prefill_complete',
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ? AND group_id = ? AND device_id = ?
            "#,
            params![&now, job_id, &active.execution_group_id, &req.device_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        if all_prefill_participants_completed {
            if let Some(refreshed_plan) = refresh_decode_plan_for_job(
                &tx,
                &job_context.network_id,
                &execution_plan,
                &scheduling_policy,
            )? {
                let execution_plan_json = serde_json::to_string(&refreshed_plan).map_err(|e| {
                    ApiError::Internal(format!(
                        "Failed to serialize refreshed execution plan: {}",
                        e
                    ))
                })?;
                tx.execute(
                    r#"
                    UPDATE inference_jobs
                    SET execution_plan_json = ?, updated_at = ?
                    WHERE job_id = ?
                    "#,
                    params![execution_plan_json, &now, job_id],
                )
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                execution_plan = refreshed_plan;
            }
            sync_serving_groups(
                &tx,
                job_id,
                &job_context.network_id,
                &job_context.model_id,
                &execution_plan,
                &now,
            )?;

            if let Some(next_segment_id) = next_segment_id(&execution_plan, &req.segment_id) {
                let next_segment = active_segment(&execution_plan, &next_segment_id)?;
                let next_group = execution_plan
                    .execution_groups
                    .iter()
                    .find(|group| group.group_id == next_segment.execution_group_id)
                    .ok_or_else(|| {
                        ApiError::Internal(format!(
                            "Execution group {} missing from execution plan {}",
                            next_segment.execution_group_id, execution_plan.plan_id
                        ))
                    })?;
                let next_participants = next_segment
                    .participant_device_ids
                    .iter()
                    .cloned()
                    .collect::<HashSet<_>>();
                let decode_queue_status = if next_participants
                    .iter()
                    .all(|device_id| participants.contains(device_id))
                {
                    "ready"
                } else {
                    "blocked_on_transfer"
                };
                tx.execute(
                    r#"
                    UPDATE inference_jobs
                    SET active_segment_id = ?, updated_at = ?
                    WHERE job_id = ?
                    "#,
                    params![&next_segment_id, &now, job_id],
                )
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                upsert_decode_queue(
                    &tx,
                    &next_segment.session_id,
                    job_id,
                    &job_context.network_id,
                    &next_segment_id,
                    &next_group.group_id,
                    decode_queue_status,
                    if decode_queue_status == "ready" {
                        Some(&now)
                    } else {
                        None
                    },
                    None,
                    None,
                    None,
                    None,
                    None,
                    &now,
                )?;
                for device_id in &next_participants {
                    let group_member = next_group
                        .members
                        .iter()
                        .find(|member| &member.device_id == device_id)
                        .ok_or_else(|| {
                            ApiError::Internal(format!(
                                "Execution group {} missing device {} for job {}",
                                next_group.group_id, device_id, job_id
                            ))
                        })?;
                    let next_status = if participants.contains(device_id) {
                        "decode_ready"
                    } else {
                        "decode_pending_transfer"
                    };
                    let assignment_exists: Option<String> = tx
                        .query_row(
                            r#"
                            SELECT assignment_id
                            FROM inference_job_assignments
                            WHERE job_id = ? AND device_id = ?
                            "#,
                            params![job_id, device_id],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(|e| {
                            ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e)))
                        })?;
                    if assignment_exists.is_none() {
                        tx.execute(
                            r#"
                            INSERT INTO inference_job_assignments (
                                assignment_id, job_id, network_id, device_id, ring_position, status,
                                assigned_at, shard_column_start, shard_column_end,
                                assigned_capacity_units, execution_provider, active_segment_id
                            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                            "#,
                            params![
                                Uuid::new_v4().to_string(),
                                job_id,
                                &job_context.network_id,
                                device_id,
                                i64::from(group_member.ring_position),
                                &now,
                                i64::from(group_member.shard.column_start),
                                i64::from(group_member.shard.column_end),
                                i64::from(group_member.assigned_capacity_units),
                                &group_member.execution_provider,
                                &next_segment_id
                            ],
                        )
                        .map_err(|e| {
                            ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e)))
                        })?;
                    }
                    tx.execute(
                        r#"
                        UPDATE inference_job_assignments
                        SET status = CASE
                                WHEN status IN ('completed', 'failed', 'cancelled') THEN status
                                ELSE 'pending'
                            END,
                            active_segment_id = ?,
                            acknowledged_at = NULL,
                            lease_expires_at = NULL
                        WHERE job_id = ? AND device_id = ?
                        "#,
                        params![&next_segment_id, job_id, device_id],
                    )
                    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                    let replica_exists: Option<String> = tx
                        .query_row(
                            r#"
                            SELECT device_id
                            FROM inference_session_replicas
                            WHERE session_id = ? AND device_id = ?
                            "#,
                            params![&next_segment.session_id, device_id],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(|e| {
                            ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e)))
                        })?;
                    if replica_exists.is_none() {
                        tx.execute(
                            r#"
                            INSERT INTO inference_session_replicas (
                                session_id, device_id, job_id, status, active_segment_id,
                                kv_sequence_position, checkpoint_created_at, updated_at, last_error
                            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL)
                            "#,
                            params![
                                &next_segment.session_id,
                                device_id,
                                job_id,
                                next_status,
                                &next_segment_id,
                                &now
                            ],
                        )
                        .map_err(|e| {
                            ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e)))
                        })?;
                    }
                    tx.execute(
                        r#"
                        UPDATE inference_session_replicas
                        SET status = ?,
                            active_segment_id = ?,
                            updated_at = ?,
                            last_error = NULL
                        WHERE session_id = ? AND device_id = ?
                        "#,
                        params![
                            next_status,
                            &next_segment_id,
                            &now,
                            &next_segment.session_id,
                            device_id
                        ],
                    )
                    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                    tx.execute(
                        r#"
                        UPDATE inference_serving_groups
                        SET status = ?,
                            updated_at = ?,
                            last_error = NULL
                        WHERE job_id = ? AND group_id = ? AND device_id = ?
                        "#,
                        params![next_status, &now, job_id, &next_group.group_id, device_id],
                    )
                    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                }
            }
        }
    } else {
        let now = now_rfc3339()?;
        tx.execute(
            r#"
            UPDATE inference_sessions
            SET status = 'decode_active',
                kv_sequence_position = COALESCE(?, kv_sequence_position),
                latest_batch_size = COALESCE(?, latest_batch_size),
                latest_active_decode_sessions = COALESCE(?, latest_active_decode_sessions),
                latest_batch_kv_tokens = COALESCE(?, latest_batch_kv_tokens),
                latest_deferred_decode_sessions = COALESCE(?, latest_deferred_decode_sessions),
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ?
            "#,
            params![
                req.kv_cache_seq_len.map(i64::from),
                req.batch_size.map(i64::from),
                req.active_decode_sessions.map(i64::from),
                req.batch_kv_tokens.map(i64::from),
                req.deferred_decode_sessions.map(i64::from),
                &now,
                job_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_session_replicas
            SET status = 'decode_active',
                active_segment_id = ?,
                kv_sequence_position = COALESCE(?, kv_sequence_position),
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ? AND device_id = ?
            "#,
            params![
                &req.segment_id,
                req.kv_cache_seq_len.map(i64::from),
                &now,
                job_id,
                &req.device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            UPDATE inference_decode_queue
            SET status = 'active',
                ready_at = COALESCE(ready_at, ?),
                updated_at = ?,
                last_error = NULL
            WHERE job_id = ?
            "#,
            params![&now, &now, job_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    }

    let job_context = load_job_context(&tx, job_id)?;
    if !prefill_complete
        && (req.batch_size.is_some()
            || req.active_decode_sessions.is_some()
            || req.batch_kv_tokens.is_some()
            || req.deferred_decode_sessions.is_some())
    {
        let observed_at = now_rfc3339()?;
        let session_id: String = tx
            .query_row(
                "SELECT session_id FROM inference_sessions WHERE job_id = ?",
                params![job_id],
                |row| row.get(0),
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        tx.execute(
            r#"
            INSERT INTO inference_decode_batch_events (
                session_id, job_id, network_id, device_id, segment_id, completion_tokens,
                execution_time_ms, batch_size, active_decode_sessions, batch_kv_tokens,
                deferred_decode_sessions, kv_cache_seq_len, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            params![
                session_id,
                job_id,
                &job_context.network_id,
                &req.device_id,
                &req.segment_id,
                i64::from(req.completion_tokens),
                req.execution_time_ms as i64,
                req.batch_size.map(i64::from),
                req.active_decode_sessions.map(i64::from),
                req.batch_kv_tokens.map(i64::from),
                req.deferred_decode_sessions.map(i64::from),
                req.kv_cache_seq_len.map(i64::from),
                &observed_at,
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    }
    let settlement_state = load_job_settlement_state(&tx, job_id)?;
    let relevant_participants = job_context
        .execution_plan
        .as_ref()
        .map(|plan| participants_for_segment(plan, &req.segment_id))
        .transpose()?
        .unwrap_or_else(|| {
            load_assignment_states(&tx, job_id)
                .unwrap_or_default()
                .into_iter()
                .map(|(device_id, _, _)| device_id)
                .collect()
        });
    reconcile_realtime_job_accounting(
        &tx,
        job_id,
        &job_context,
        &settlement_state,
        &relevant_participants,
    )?;

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
                   time_to_first_token_ms, reserved_credits, settled_credits, released_credits,
                   available_completion_tokens, model_size_factor, error, execution_plan_json,
                   active_segment_id
            FROM inference_jobs
            WHERE job_id = ?
            "#,
            params![job_id],
            |row| {
                let execution_plan = row
                    .get::<_, Option<String>>(14)?
                    .as_deref()
                    .map(load_execution_plan_json)
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            14,
                            rusqlite::types::Type::Text,
                            Box::new(std::io::Error::other(e.to_string())),
                        )
                    })?;
                Ok(PersistedJobStatus {
                    job_id: row.get(0)?,
                    network_id: row.get(1)?,
                    model_id: row.get(2)?,
                    status: row.get(3)?,
                    completion: row.get(4)?,
                    completion_tokens: row.get::<_, i64>(5)? as u32,
                    execution_time_ms: row.get::<_, i64>(6)? as u64,
                    time_to_first_token_ms: row.get::<_, Option<i64>>(7)?.map(|v| v as u64),
                    active_segment_id: row.get(15)?,
                    reserved_credits: row.get(8)?,
                    settled_credits: row.get(9)?,
                    released_credits: row.get(10)?,
                    available_completion_tokens: row.get::<_, i64>(11)? as u32,
                    model_size_factor: row.get(12)?,
                    error: row.get(13)?,
                    assignments: Vec::new(),
                    execution_plan,
                    session: None,
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
    let session = conn
        .query_row(
            r#"
            SELECT s.session_id, s.status, s.active_segment_id, s.kv_owner_device_id, s.kv_transfer_policy,
                   s.kv_sequence_position, s.latest_batch_size, s.latest_active_decode_sessions,
                   s.latest_batch_kv_tokens, s.latest_deferred_decode_sessions,
                   dq.lease_target_session_count, dq.lease_target_batch_size,
                   s.kv_checkpoint_device_id, s.kv_checkpoint_created_at, s.updated_at, s.last_error
            FROM inference_sessions s
            LEFT JOIN inference_decode_queue dq ON dq.session_id = s.session_id
            WHERE s.job_id = ?
            "#,
            params![job_id],
            |row| {
                Ok(PersistedSessionStatus {
                    session_id: row.get(0)?,
                    status: row.get(1)?,
                    active_segment_id: row.get(2)?,
                    kv_owner_device_id: row.get(3)?,
                    kv_transfer_policy: row.get(4)?,
                    kv_sequence_position: row.get::<_, Option<i64>>(5)?.map(|v| v as u32),
                    latest_batch_size: row.get::<_, Option<i64>>(6)?.map(|v| v as u32),
                    latest_active_decode_sessions: row.get::<_, Option<i64>>(7)?.map(|v| v as u32),
                    latest_batch_kv_tokens: row.get::<_, Option<i64>>(8)?.map(|v| v as u32),
                    latest_deferred_decode_sessions: row.get::<_, Option<i64>>(9)?.map(|v| v as u32),
                    lease_target_session_count: row.get::<_, Option<i64>>(10)?.map(|v| v as u32),
                    lease_target_batch_size: row.get::<_, Option<i64>>(11)?.map(|v| v as u32),
                    kv_checkpoint_device_id: row.get(12)?,
                    kv_checkpoint_created_at: row.get(13)?,
                    updated_at: row.get(14)?,
                    last_error: row.get(15)?,
                    checkpoint: None,
                    replicas: Vec::new(),
                    recent_decode_batches: Vec::new(),
                })
            },
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let session = if let Some(mut session) = session {
        session.checkpoint = load_latest_session_checkpoint_status(&conn, &session.session_id)?;
        let mut stmt = conn
            .prepare(
                r#"
                SELECT device_id, status, active_segment_id, kv_sequence_position,
                       checkpoint_created_at, updated_at, last_error
                FROM inference_session_replicas
                WHERE session_id = ?
                ORDER BY device_id ASC
                "#,
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        session.replicas = stmt
            .query_map(params![&session.session_id], |row| {
                Ok(PersistedSessionReplicaStatus {
                    device_id: row.get(0)?,
                    status: row.get(1)?,
                    active_segment_id: row.get(2)?,
                    kv_sequence_position: row.get::<_, Option<i64>>(3)?.map(|v| v as u32),
                    checkpoint_created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                    last_error: row.get(6)?,
                })
            })
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        session.recent_decode_batches =
            load_recent_decode_batch_events(&conn, &session.session_id, 8)?;
        Some(session)
    } else {
        None
    };

    Ok(PersistedJobStatus {
        assignments,
        session,
        ..job
    })
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

fn store_session_checkpoint(
    db: &crate::db::Database,
    job_id: &str,
    req: &UploadInferenceSessionCheckpointRequest,
) -> ApiResult<()> {
    let mut conn = db.get_conn()?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let now = now_rfc3339()?;

    let phase = serialize_execution_phase(req.phase);
    let execution_plan_json: String = tx
        .query_row(
            "SELECT execution_plan_json FROM inference_jobs WHERE job_id = ?",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let execution_plan = load_execution_plan_json(&execution_plan_json)?;
    let source_segment = active_segment(&execution_plan, &req.segment_id)?;
    if source_segment.session_id != req.session_id {
        return Err(ApiError::Conflict(format!(
            "Checkpoint session {} does not match source segment session {}",
            req.session_id, source_segment.session_id
        )));
    }
    if source_segment.phase != req.phase {
        return Err(ApiError::Conflict(format!(
            "Checkpoint phase {:?} does not match source segment phase {:?}",
            req.phase, source_segment.phase
        )));
    }
    if !source_segment
        .participant_device_ids
        .iter()
        .any(|device_id| device_id == &req.device_id)
    {
        return Err(ApiError::Conflict(format!(
            "Device {} is not a participant in source segment {}",
            req.device_id, req.segment_id
        )));
    }
    let replica_exists: Option<String> = tx
        .query_row(
            r#"
            SELECT device_id
            FROM inference_session_replicas
            WHERE job_id = ? AND session_id = ? AND device_id = ?
            "#,
            params![job_id, &req.session_id, &req.device_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    if replica_exists.is_none() {
        return Err(ApiError::NotFound(format!(
            "Session replica not found for job {}, session {}, device {}",
            job_id, req.session_id, req.device_id
        )));
    }
    let source_assignment_completed: Option<String> = tx
        .query_row(
            r#"
            SELECT last_completed_segment_id
            FROM inference_job_assignments
            WHERE job_id = ? AND device_id = ?
            "#,
            params![job_id, &req.device_id],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .flatten();
    if source_assignment_completed.as_deref() != Some(req.segment_id.as_str()) {
        return Err(ApiError::Conflict(format!(
            "Device {} has not completed source segment {} for checkpoint export",
            req.device_id, req.segment_id
        )));
    }

    let checkpoint_bytes = hex::decode(&req.checkpoint_hex).map_err(|e| {
        ApiError::BadRequest(format!("checkpoint_hex must be valid hexadecimal: {}", e))
    })?;
    if checkpoint_bytes.is_empty() {
        return Err(ApiError::BadRequest(
            "checkpoint_hex must not be empty".to_string(),
        ));
    }

    let checkpoint_sha256 = {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&checkpoint_bytes);
        hex::encode(hasher.finalize())
    };
    let checkpoint_id = Uuid::new_v4().to_string();

    tx.execute(
        r#"
        INSERT INTO inference_session_checkpoints (
            checkpoint_id, session_id, job_id, source_device_id, source_segment_id, phase,
            kv_sequence_position, size_bytes, checkpoint_sha256, checkpoint_bytes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
        params![
            &checkpoint_id,
            &req.session_id,
            job_id,
            &req.device_id,
            &req.segment_id,
            phase,
            i64::from(req.kv_sequence_position),
            checkpoint_bytes.len() as i64,
            &checkpoint_sha256,
            checkpoint_bytes,
            &now,
            &now
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    tx.execute(
        r#"
        UPDATE inference_sessions
        SET kv_checkpoint_device_id = ?,
            kv_checkpoint_created_at = ?,
            kv_sequence_position = MAX(COALESCE(kv_sequence_position, 0), ?),
            updated_at = ?
        WHERE job_id = ? AND session_id = ?
        "#,
        params![
            &req.device_id,
            &now,
            i64::from(req.kv_sequence_position),
            &now,
            job_id,
            &req.session_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    tx.execute(
        r#"
        UPDATE inference_session_replicas
        SET status = CASE
                WHEN status = 'decode_pending_transfer' THEN 'decode_ready'
                ELSE status
            END,
            kv_sequence_position = COALESCE(kv_sequence_position, ?),
            checkpoint_created_at = COALESCE(checkpoint_created_at, ?),
            updated_at = ?,
            last_error = NULL
        WHERE session_id = ?
        "#,
        params![
            i64::from(req.kv_sequence_position),
            &now,
            &now,
            &req.session_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_serving_groups
        SET status = CASE
                WHEN status = 'decode_pending_transfer' THEN 'decode_ready'
                ELSE status
            END,
            updated_at = ?,
            last_error = NULL
        WHERE session_id = ? AND phase = 'decode'
        "#,
        params![&now, &req.session_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_decode_queue
        SET status = 'ready',
            ready_at = COALESCE(ready_at, ?),
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            last_error = NULL,
            updated_at = ?
        WHERE session_id = ?
        "#,
        params![&now, &now, &req.session_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn load_assignment_segment_completion(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> ApiResult<Vec<(String, Option<String>, Option<String>)>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT device_id, last_completed_segment_id, segment_completed_at
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

fn load_execution_plan_json(json: &str) -> ApiResult<InferenceExecutionPlan> {
    serde_json::from_str(json)
        .map_err(|e| ApiError::Internal(format!("Failed to parse execution plan: {}", e)))
}

fn load_job_context(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
) -> ApiResult<PersistedJobContext> {
    conn.query_row(
        r#"
        SELECT network_id, model_id, submitted_by_device_id, ring_worker_count, prompt_tokens,
               reserved_credits, execution_plan_json
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
            let execution_plan = row
                .get::<_, Option<String>>(6)?
                .as_deref()
                .map(load_execution_plan_json)
                .transpose()
                .map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        6,
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
                execution_plan,
            })
        },
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_completed_assignment_credit_inputs(
    conn: &rusqlite::Transaction<'_>,
    job_id: &str,
    device_scope: &HashSet<String>,
) -> ApiResult<Vec<PersistedCompletedAssignmentCreditInput>> {
    if device_scope.is_empty() {
        return Ok(Vec::new());
    }

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
        .map(|rows| {
            rows.into_iter()
                .filter(|row| device_scope.contains(&row.device_id))
                .collect()
        })
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
    device_scope: &HashSet<String>,
) -> ApiResult<()> {
    let assignment_inputs = load_completed_assignment_credit_inputs(conn, job_id, device_scope)?;
    if assignment_inputs.len() < device_scope.len() {
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
    use crate::api::types::{ExecutionPhase, KvTransferPolicy, ProgressEventKind, RingJoinRequest};
    use crate::connectivity::{
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath,
        DeviceConnectivityState, InferenceSchedulingPolicy, NetworkConnectivity, TierCapacityUnits,
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

    async fn drive_job_through_prefill_and_decode(
        state: &AppState,
        job_id: &str,
        network_id: &str,
        plan: &InferenceExecutionPlan,
        decode_results: &[(&str, u64, u32)],
    ) {
        let prefill_segment_id = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
            .map(|segment| segment.segment_id.clone())
            .expect("expected prefill segment id");
        let prefill_devices = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
            .map(|segment| segment.participant_device_ids.clone())
            .expect("expected prefill participants");
        for device_id in prefill_devices {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: device_id.clone(),
                    network_id: network_id.to_string(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected prefill assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.clone(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_progress(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id,
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 42,
                    time_to_first_token_ms: Some(42),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        let status = get_inference_job_status(State(state.clone()), Path(job_id.to_string()))
            .await
            .unwrap()
            .0;
        if status
            .session
            .as_ref()
            .map(|session| {
                session
                    .replicas
                    .iter()
                    .any(|replica| replica.status == "decode_pending_transfer")
            })
            .unwrap_or(false)
        {
            let session_id = status
                .session
                .as_ref()
                .map(|session| session.session_id.clone())
                .expect("expected session id");
            let _ = upload_inference_session_checkpoint(
                State(state.clone()),
                Path(job_id.to_string()),
                Json(UploadInferenceSessionCheckpointRequest {
                    device_id: plan.segments[0].participant_device_ids[0].clone(),
                    session_id,
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    kv_sequence_position: 1,
                    checkpoint_hex: "c0ffee".into(),
                }),
            )
            .await
            .unwrap();
        }

        for (device_id, execution_time_ms, completion_tokens) in decode_results {
            let claim = claim_inference_assignment(
                State(state.clone()),
                Json(ClaimInferenceAssignmentRequest {
                    device_id: (*device_id).to_string(),
                    network_id: network_id.to_string(),
                }),
            )
            .await
            .unwrap()
            .0
            .assignment
            .expect("expected decode assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: (*device_id).to_string(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_result(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(ReportInferenceAssignmentRequest {
                    device_id: (*device_id).to_string(),
                    segment_id: claim.active_segment.segment_id.clone(),
                    success: true,
                    completion: Some(format!("completion-from-{}", device_id)),
                    completion_tokens: Some(*completion_tokens),
                    execution_time_ms: *execution_time_ms,
                    time_to_first_token_ms: None,
                    kv_cache_seq_len: Some(*completion_tokens),
                    error: None,
                }),
            )
            .await
            .unwrap();
        }
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
                segment_id: assignment.active_segment.segment_id.clone(),
                success: true,
                completion: Some("partial".into()),
                completion_tokens: Some(2),
                execution_time_ms: 100,
                time_to_first_token_ms: None,
                kv_cache_seq_len: None,
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

        drive_job_through_prefill_and_decode(
            &state,
            &submit.job_id,
            network_id,
            submit
                .execution_plan
                .as_ref()
                .expect("expected execution plan"),
            &[("worker-1", 500, 4), ("worker-2", 500, 4)],
        )
        .await;

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

        drive_job_through_prefill_and_decode(
            &state,
            &submit.job_id,
            network_id,
            submit
                .execution_plan
                .as_ref()
                .expect("expected execution plan"),
            &[("worker-1", 500, 4), ("worker-2", 500, 4)],
        )
        .await;

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
        let prefill_segment_id = submit
            .execution_plan
            .as_ref()
            .and_then(|plan| {
                plan.segments
                    .iter()
                    .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
                    .map(|segment| segment.segment_id.clone())
            })
            .expect("expected prefill segment id");
        let decode_segment_id = submit
            .execution_plan
            .as_ref()
            .and_then(|plan| {
                plan.segments
                    .iter()
                    .find(|segment| matches!(segment.phase, ExecutionPhase::Decode))
                    .map(|segment| segment.segment_id.clone())
            })
            .expect("expected decode segment id");

        for device_id in ["worker-1", "worker-2"] {
            let prefill_claim = claim_inference_assignment(
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
                Path(prefill_claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_progress(
                State(state.clone()),
                Path(prefill_claim.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id.into(),
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 40,
                    time_to_first_token_ms: Some(40),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        for device_id in ["worker-1", "worker-2"] {
            let decode_claim = claim_inference_assignment(
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
            .expect("expected decode assignment");

            assert_eq!(decode_claim.active_segment.segment_id, decode_segment_id);
            assert_eq!(decode_claim.session.lease_target_session_count, Some(1));
            assert_eq!(decode_claim.session.lease_target_batch_size, Some(1));

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(decode_claim.job_id.clone()),
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
                segment_id: decode_segment_id.clone(),
                phase: ExecutionPhase::Decode,
                event: ProgressEventKind::DecodeProgress,
                completion_tokens: 3,
                execution_time_ms: 120,
                time_to_first_token_ms: None,
                kv_cache_seq_len: Some(2),
                batch_size: Some(2),
                active_decode_sessions: Some(2),
                batch_kv_tokens: Some(2),
                deferred_decode_sessions: Some(1),
            }),
        )
        .await
        .unwrap();

        let status_after_first_progress =
            get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
                .await
                .unwrap()
                .0;
        assert_eq!(status_after_first_progress.completion_tokens, 1);
        assert!(status_after_first_progress.settled_credits > 0.0);

        let _ = report_inference_progress(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(ReportInferenceAssignmentProgressRequest {
                device_id: "worker-2".into(),
                segment_id: decode_segment_id.clone(),
                phase: ExecutionPhase::Decode,
                event: ProgressEventKind::DecodeProgress,
                completion_tokens: 3,
                execution_time_ms: 140,
                time_to_first_token_ms: None,
                kv_cache_seq_len: Some(2),
                batch_size: Some(2),
                active_decode_sessions: Some(2),
                batch_kv_tokens: Some(2),
                deferred_decode_sessions: Some(1),
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
        assert_eq!(
            status_after_frontier
                .session
                .as_ref()
                .and_then(|session| session.latest_batch_size),
            Some(2)
        );
        assert_eq!(
            status_after_frontier
                .session
                .as_ref()
                .and_then(|session| session.latest_active_decode_sessions),
            Some(2)
        );
        assert_eq!(
            status_after_frontier
                .session
                .as_ref()
                .and_then(|session| session.latest_batch_kv_tokens),
            Some(2)
        );
        assert_eq!(
            status_after_frontier
                .session
                .as_ref()
                .and_then(|session| session.latest_deferred_decode_sessions),
            Some(1)
        );
        assert_eq!(
            status_after_frontier
                .session
                .as_ref()
                .map(|session| session.recent_decode_batches.len()),
            Some(2)
        );
        assert_eq!(
            status_after_frontier
                .session
                .as_ref()
                .and_then(|session| session.recent_decode_batches.first())
                .and_then(|event| event.batch_size),
            Some(2)
        );

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
    async fn test_prefill_progress_records_ttft() {
        let network_id = "test-network-prefill-ttft";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "prefill".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let prefill_segment_id = submit
            .execution_plan
            .as_ref()
            .and_then(|plan| {
                plan.segments
                    .iter()
                    .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
                    .map(|segment| segment.segment_id.clone())
            })
            .expect("expected prefill segment id");

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

            let _ = report_inference_progress(
                State(state.clone()),
                Path(submit.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id.into(),
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 42,
                    time_to_first_token_ms: Some(42),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id))
            .await
            .unwrap()
            .0;
        assert_eq!(status.time_to_first_token_ms, Some(42));
        assert_eq!(
            status
                .session
                .as_ref()
                .map(|session| session.status.as_str()),
            Some("decode_ready")
        );
        assert_eq!(
            status
                .session
                .as_ref()
                .and_then(|session| session.kv_sequence_position),
            Some(1)
        );
        assert_eq!(
            status
                .session
                .as_ref()
                .and_then(|session| session.kv_checkpoint_device_id.as_deref()),
            Some("worker-2")
        );
        assert_eq!(
            status
                .session
                .as_ref()
                .map(|session| session.replicas.len()),
            Some(2)
        );
        assert!(status
            .session
            .as_ref()
            .map(|session| session
                .replicas
                .iter()
                .all(|replica| replica.kv_sequence_position == Some(1)))
            .unwrap_or(false));
        assert!(status
            .active_segment_id
            .as_deref()
            .map(|segment_id| segment_id.contains("decode"))
            .unwrap_or(false));

        let decode_claim = claim_inference_assignment(
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
        .expect("expected decode assignment");
        assert_eq!(
            decode_claim
                .session
                .local_replica
                .as_ref()
                .map(|replica| replica.status.as_str()),
            Some("decode_ready")
        );
        assert_eq!(
            decode_claim
                .session
                .local_replica
                .as_ref()
                .and_then(|replica| replica.kv_sequence_position),
            Some(1)
        );
    }

    #[tokio::test]
    async fn test_uploaded_session_checkpoint_is_exposed_on_status_lease_and_download() {
        let network_id = "test-network-session-checkpoint-upload";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "handoff".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let prefill_segment_id = submit
            .execution_plan
            .as_ref()
            .and_then(|plan| {
                plan.segments
                    .iter()
                    .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
                    .map(|segment| segment.segment_id.clone())
            })
            .expect("expected prefill segment id");
        let session_id = submit
            .execution_plan
            .as_ref()
            .map(|plan| plan.segments[0].session_id.clone())
            .expect("expected session id");

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

            let _ = report_inference_progress(
                State(state.clone()),
                Path(submit.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id.into(),
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 42,
                    time_to_first_token_ms: Some(42),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        let _ = upload_inference_session_checkpoint(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(UploadInferenceSessionCheckpointRequest {
                device_id: "worker-2".into(),
                session_id: session_id.clone(),
                segment_id: prefill_segment_id.clone(),
                phase: ExecutionPhase::Prefill,
                kv_sequence_position: 1,
                checkpoint_hex: "c0ffee".into(),
            }),
        )
        .await
        .unwrap();

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
            .await
            .unwrap()
            .0;
        let session = status.session.expect("expected session");
        let checkpoint = session.checkpoint.expect("expected session checkpoint");
        assert_eq!(checkpoint.source_device_id, "worker-2");
        assert_eq!(checkpoint.source_segment_id, prefill_segment_id);
        assert_eq!(checkpoint.kv_sequence_position, 1);
        assert_eq!(checkpoint.sha256.len(), 64);

        let download = download_inference_session_checkpoint(
            State(state.clone()),
            Path((submit.job_id.clone(), session_id.clone())),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(
            download
                .checkpoint
                .as_ref()
                .map(|payload| payload.checkpoint_hex.as_str()),
            Some("c0ffee")
        );

        let decode_claim = claim_inference_assignment(
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
        .expect("expected decode assignment");
        assert_eq!(
            decode_claim
                .session
                .checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.source_device_id.as_str()),
            Some("worker-2")
        );
    }

    #[tokio::test]
    async fn test_session_checkpoint_upload_requires_completed_source_segment() {
        let network_id = "test-network-checkpoint-upload-validation";
        let state = joined_state(&["worker-1", "worker-2"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "handoff".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let plan = submit
            .execution_plan
            .clone()
            .expect("expected execution plan");
        let prefill_segment_id = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
            .map(|segment| segment.segment_id.clone())
            .expect("expected prefill segment id");
        let session_id = plan.segments[0].session_id.clone();

        let upload_err = upload_inference_session_checkpoint(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(UploadInferenceSessionCheckpointRequest {
                device_id: "worker-1".into(),
                session_id: session_id.clone(),
                segment_id: prefill_segment_id.clone(),
                phase: ExecutionPhase::Prefill,
                kv_sequence_position: 1,
                checkpoint_hex: "c0ffee".into(),
            }),
        )
        .await
        .expect_err("upload before prefill completion should be rejected");
        assert!(matches!(upload_err, ApiError::Conflict(_)));

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
            .expect("expected prefill assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_progress(
                State(state.clone()),
                Path(submit.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id.into(),
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 42,
                    time_to_first_token_ms: Some(42),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        let _ = upload_inference_session_checkpoint(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(UploadInferenceSessionCheckpointRequest {
                device_id: "worker-1".into(),
                session_id,
                segment_id: prefill_segment_id,
                phase: ExecutionPhase::Prefill,
                kv_sequence_position: 1,
                checkpoint_hex: "c0ffee".into(),
            }),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_prefill_handoff_replans_decode_group_from_live_pool() {
        let network_id = "test-network-live-decode-replan";
        let state = joined_state_with_device_capabilities(
            &[
                (
                    "worker-1",
                    DeviceCapabilities {
                        tier: Tier::Tier0,
                        cpu_cores: 2,
                        ram_mb: 2048,
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
            ],
            network_id,
            InferenceSchedulingPolicy::default(),
        )
        .await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "handoff".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;
        let plan = submit
            .execution_plan
            .clone()
            .expect("expected execution plan");
        let prefill_segment_id = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
            .map(|segment| segment.segment_id.clone())
            .expect("expected prefill segment id");
        let session_id = plan.segments[0].session_id.clone();
        let decode_segment_id = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Decode))
            .map(|segment| segment.segment_id.clone())
            .expect("expected decode segment id");

        register_test_device_with_capabilities(
            &state.db,
            "worker-3",
            network_id,
            DeviceCapabilities {
                tier: Tier::Tier4,
                cpu_cores: 32,
                ram_mb: 65536,
                ..test_capabilities()
            },
        );
        seed_device_credits(&state.db, network_id, "worker-3", 10_000.0);
        let _ = join_ring(
            State(state.clone()),
            Json(RingJoinRequest {
                device_id: "worker-3".into(),
                network_id: network_id.into(),
                model_id: "test-model".into(),
                contributed_memory: 32_000_000_000,
            }),
        )
        .await
        .unwrap();
        let conn = state.db.get_conn().unwrap();
        let direct_connectivity = serde_json::to_string(&DeviceConnectivityState {
            active_path: ConnectivityPath::Direct,
            active_endpoint: Some("tcp://127.0.0.1:9000".into()),
            status: crate::connectivity::ConnectivityStatus::Connected,
        })
        .unwrap();
        let direct_listen_addrs =
            serde_json::to_string(&vec!["dataplane://127.0.0.1:9000"]).unwrap();
        conn.execute(
            r#"
            UPDATE devices
            SET shard_column_start = 0,
                shard_column_end = 8192,
                contributed_memory = 32000000000,
                connectivity_state = ?,
                listen_addrs = ?
            WHERE network_id = ? AND device_id = 'worker-3'
            "#,
            params![direct_connectivity, direct_listen_addrs, network_id],
        )
        .unwrap();
        conn.execute(
            r#"
            UPDATE devices
            SET connectivity_state = ?,
                listen_addrs = ?
            WHERE network_id = ? AND device_id IN ('worker-1', 'worker-2')
            "#,
            params![
                serde_json::to_string(&DeviceConnectivityState {
                    active_path: ConnectivityPath::Direct,
                    active_endpoint: Some("tcp://127.0.0.1:9000".into()),
                    status: crate::connectivity::ConnectivityStatus::Connected,
                })
                .unwrap(),
                serde_json::to_string(&vec!["dataplane://127.0.0.1:9000"]).unwrap(),
                network_id
            ],
        )
        .unwrap();

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
            .expect("expected prefill assignment");
            assert_eq!(claim.active_segment.segment_id, prefill_segment_id);

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_progress(
                State(state.clone()),
                Path(submit.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id.into(),
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 42,
                    time_to_first_token_ms: Some(42),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
            .await
            .unwrap()
            .0;
        let refreshed_plan = status.execution_plan.expect("expected refreshed plan");
        let decode_segment = refreshed_plan
            .segments
            .iter()
            .find(|segment| segment.segment_id == decode_segment_id)
            .expect("expected decode segment");
        assert_eq!(
            decode_segment.participant_device_ids,
            vec!["worker-3".to_string()]
        );
        let worker3_replica = status
            .session
            .as_ref()
            .expect("expected session")
            .replicas
            .iter()
            .find(|replica| replica.device_id == "worker-3")
            .expect("expected worker-3 replica");
        assert_eq!(worker3_replica.status, "decode_pending_transfer");
        let blocked_claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-3".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment;
        assert!(blocked_claim.is_none());

        let _ = upload_inference_session_checkpoint(
            State(state.clone()),
            Path(submit.job_id.clone()),
            Json(UploadInferenceSessionCheckpointRequest {
                device_id: "worker-1".into(),
                session_id,
                segment_id: prefill_segment_id.clone(),
                phase: ExecutionPhase::Prefill,
                kv_sequence_position: 1,
                checkpoint_hex: "c0ffee".into(),
            }),
        )
        .await
        .unwrap();

        let decode_claim = claim_inference_assignment(
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
        .expect("expected decode assignment");
        assert_eq!(decode_claim.active_segment.segment_id, decode_segment_id);
        assert_eq!(
            decode_claim.active_segment.participant_device_ids,
            vec!["worker-3"]
        );
    }

    #[tokio::test]
    async fn test_decode_completion_uses_final_segment_participants_only() {
        let network_id = "test-network-final-segment-participants";
        let state = joined_state(&["worker-1", "worker-2", "worker-3"], network_id).await;

        let submit = submit_inference(
            State(state.clone()),
            Json(SubmitInferenceRequest {
                device_id: "worker-1".into(),
                network_id: network_id.into(),
                model_id: "llama-70b".into(),
                prompt: "handoff".into(),
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
            }),
        )
        .await
        .unwrap()
        .0;

        let mut plan = submit
            .execution_plan
            .clone()
            .expect("expected execution plan");
        let decode_group_id = plan
            .execution_groups
            .iter()
            .find(|group| matches!(group.phase, ExecutionPhase::Decode))
            .map(|group| group.group_id.clone())
            .expect("expected decode group");
        for group in &mut plan.execution_groups {
            if group.group_id == decode_group_id {
                group
                    .members
                    .retain(|member| member.device_id == "worker-1");
                group.total_capacity_units = group
                    .members
                    .iter()
                    .map(|member| member.assigned_capacity_units)
                    .sum();
                group.kv_transfer_policy = KvTransferPolicy::ExportOnHandoff;
            }
        }
        for segment in &mut plan.segments {
            if matches!(segment.phase, ExecutionPhase::Decode) {
                segment.participant_device_ids = vec!["worker-1".into()];
                segment.shard_owner_device_ids = vec!["worker-1".into()];
                segment.kv_owner_device_id = "worker-1".into();
            }
        }

        let conn = state.db.get_conn().unwrap();
        conn.execute(
            "UPDATE inference_jobs SET execution_plan_json = ? WHERE job_id = ?",
            params![serde_json::to_string(&plan).unwrap(), &submit.job_id],
        )
        .unwrap();
        conn.execute(
            r#"
            UPDATE devices
            SET shard_column_start = CASE
                    WHEN device_id = 'worker-1' THEN 0
                    WHEN device_id = 'worker-2' THEN 0
                    ELSE 4096
                END,
                shard_column_end = CASE
                    WHEN device_id = 'worker-1' THEN 8192
                    WHEN device_id = 'worker-2' THEN 4096
                    ELSE 8192
                END,
                contributed_memory = CASE
                    WHEN device_id = 'worker-1' THEN 64000000000
                    ELSE contributed_memory
                END,
                connectivity_state = ?,
                listen_addrs = ?
            WHERE network_id = ?
            "#,
            params![
                serde_json::to_string(&DeviceConnectivityState {
                    active_path: ConnectivityPath::Direct,
                    active_endpoint: Some("tcp://127.0.0.1:9000".into()),
                    status: crate::connectivity::ConnectivityStatus::Connected,
                })
                .unwrap(),
                serde_json::to_string(&vec!["dataplane://127.0.0.1:9000"]).unwrap(),
                network_id
            ],
        )
        .unwrap();

        let prefill_segment_id = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Prefill))
            .map(|segment| segment.segment_id.clone())
            .expect("expected prefill segment id");
        let decode_segment_id = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Decode))
            .map(|segment| segment.segment_id.clone())
            .expect("expected decode segment id");

        for device_id in ["worker-1", "worker-2", "worker-3"] {
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
            .expect("expected prefill assignment");

            let _ = acknowledge_inference_assignment(
                State(state.clone()),
                Path(claim.job_id.clone()),
                Json(AcknowledgeInferenceAssignmentRequest {
                    device_id: device_id.into(),
                }),
            )
            .await
            .unwrap();

            let _ = report_inference_progress(
                State(state.clone()),
                Path(submit.job_id.clone()),
                Json(ReportInferenceAssignmentProgressRequest {
                    device_id: device_id.into(),
                    segment_id: prefill_segment_id.clone(),
                    phase: ExecutionPhase::Prefill,
                    event: ProgressEventKind::PrefillComplete,
                    completion_tokens: 1,
                    execution_time_ms: 42,
                    time_to_first_token_ms: Some(42),
                    kv_cache_seq_len: Some(1),
                    batch_size: None,
                    active_decode_sessions: None,
                    batch_kv_tokens: None,
                    deferred_decode_sessions: None,
                }),
            )
            .await
            .unwrap();
        }

        let worker1_decode_claim = claim_inference_assignment(
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
        .expect("expected decode assignment");
        assert_eq!(
            worker1_decode_claim.active_segment.segment_id,
            decode_segment_id
        );

        let worker2_decode_claim = claim_inference_assignment(
            State(state.clone()),
            Json(ClaimInferenceAssignmentRequest {
                device_id: "worker-2".into(),
                network_id: network_id.into(),
            }),
        )
        .await
        .unwrap()
        .0
        .assignment;
        assert!(worker2_decode_claim.is_none());

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
                segment_id: decode_segment_id,
                success: true,
                completion: Some("done".into()),
                completion_tokens: Some(4),
                execution_time_ms: 100,
                time_to_first_token_ms: None,
                kv_cache_seq_len: Some(4),
                error: None,
            }),
        )
        .await
        .unwrap();

        let status = get_inference_job_status(State(state.clone()), Path(submit.job_id.clone()))
            .await
            .unwrap()
            .0;
        assert_eq!(status.status, "completed");
        assert_eq!(status.completion.as_deref(), Some("done"));
        assert!(status.active_segment_id.is_none());
        assert!(status
            .assignments
            .iter()
            .all(|assignment| assignment.status == "completed"));
        assert!(status
            .session
            .as_ref()
            .map(|session| session
                .replicas
                .iter()
                .all(|replica| replica.status == "completed"))
            .unwrap_or(false));
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

        drive_job_through_prefill_and_decode(
            &state,
            &first_submit.job_id,
            network_id,
            first_submit
                .execution_plan
                .as_ref()
                .expect("expected execution plan"),
            &[("worker-1", 300, 4), ("worker-2", 300, 4)],
        )
        .await;

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

        drive_job_through_prefill_and_decode(
            &state,
            &submit.job_id,
            network_id,
            submit
                .execution_plan
                .as_ref()
                .expect("expected execution plan"),
            &[("worker-1", 200, 4), ("worker-2", 800, 4)],
        )
        .await;

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
                segment_id: claim.active_segment.segment_id.clone(),
                success: true,
                completion: Some("done".into()),
                completion_tokens: Some(1),
                execution_time_ms: 25,
                time_to_first_token_ms: None,
                kv_cache_seq_len: None,
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
