use std::collections::HashMap;

use rusqlite::{params, OptionalExtension, Transaction};

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    ClaimInferenceAssignmentRequest, InferenceExecutionPlan, InferenceRuntimeMode, WorkClaimMode,
};
use crate::connectivity::{DeviceConnectivityState, InferenceSchedulingPolicy};
use crate::device::DeviceCapabilities;
use crate::services::planner::{
    device_metadata_from_capabilities, ExecutionPlanner, PlannerDeviceMetadata,
};
use crate::services::ring_manager::{ModelShard, RingTopology, WorkerTopologyInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicyMode {
    FitFirst,
    ThroughputFirst,
    LatencyFirst,
    ResilientEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchedulerPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerBlockedReason {
    QueueMissing,
    WaitingForTransfer,
    LeaseHeldByPeer,
    AlreadyRunning,
    NotEligible,
}

#[derive(Debug, Clone)]
pub struct SchedulerBlockedSession {
    pub session_id: String,
    pub job_id: String,
    pub assignment_id: String,
    pub reason: SchedulerBlockedReason,
}

#[derive(Debug, Clone)]
pub struct SchedulerDecision {
    pub selected_assignment_id: Option<String>,
    pub blocked: Vec<SchedulerBlockedSession>,
}

#[derive(Debug, Clone)]
struct SchedulerCandidate {
    assignment_id: String,
    job_id: String,
    session_id: String,
    model_id: String,
    runtime_mode: SchedulerPolicyMode,
    submitted_by_device_id: String,
    created_at: String,
    assigned_at: String,
    phase: SchedulerPhase,
    group_status: String,
    decode_queue_status: Option<String>,
    decode_ready_at: Option<String>,
    decode_lease_owner_device_id: Option<String>,
    decode_lease_expires_at: Option<String>,
    decode_updated_at: Option<String>,
    group_lease_owner_device_id: Option<String>,
    group_lease_expires_at: Option<String>,
    decode_lease_target_session_count: Option<u32>,
    decode_cohort_ready_sessions: u32,
    decode_cohort_blocked_sessions: u32,
    decode_cohort_oldest_ready_at: Option<String>,
    decode_cohort_leased_sessions: u32,
    decode_cohort_active_sessions: u32,
}

#[derive(Debug, Clone)]
struct RunnableCandidate {
    candidate: SchedulerCandidate,
    ready_at: String,
}

#[derive(Debug, Clone)]
struct SchedulerSnapshot {
    runnable: Vec<RunnableCandidate>,
    blocked: Vec<SchedulerBlockedSession>,
    active_jobs_by_submitter: HashMap<String, u32>,
    active_jobs_by_model: HashMap<String, u32>,
    active_capacity_by_model: HashMap<String, u32>,
    leased_assignments_by_submitter: HashMap<String, u32>,
    leased_assignments_by_job: HashMap<String, u32>,
    online_worker_count: u32,
    online_capacity_units: u32,
}

pub fn select_claim_assignment_id(
    conn: &Transaction<'_>,
    req: &ClaimInferenceAssignmentRequest,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> ApiResult<Option<String>> {
    Ok(
        schedule_claim_decision(conn, req, scheduling_policy, SchedulerPolicyMode::FitFirst)?
            .selected_assignment_id,
    )
}

pub fn schedule_claim_decision(
    conn: &Transaction<'_>,
    req: &ClaimInferenceAssignmentRequest,
    scheduling_policy: &InferenceSchedulingPolicy,
    mode: SchedulerPolicyMode,
) -> ApiResult<SchedulerDecision> {
    let now = now_rfc3339()?;
    reconcile_stale_scheduler_leases(conn, &req.network_id, &now)?;

    let snapshot = load_scheduler_snapshot(conn, req, scheduling_policy, &now)?;
    if snapshot.runnable.is_empty() {
        return Ok(SchedulerDecision {
            selected_assignment_id: None,
            blocked: snapshot.blocked,
        });
    }

    let model_soft_cap =
        (snapshot.online_worker_count / scheduling_policy.model_active_job_soft_cap_divisor).max(1);
    let capacity_soft_cap =
        (snapshot.online_capacity_units / scheduling_policy.capacity_unit_soft_cap_divisor).max(1);

    let selected = snapshot
        .runnable
        .into_iter()
        .min_by(|left, right| {
            rank_candidate(
                left,
                mode,
                scheduling_policy,
                model_soft_cap,
                capacity_soft_cap,
                &snapshot.active_jobs_by_submitter,
                &snapshot.active_jobs_by_model,
                &snapshot.active_capacity_by_model,
                &snapshot.leased_assignments_by_submitter,
                &snapshot.leased_assignments_by_job,
            )
            .cmp(&rank_candidate(
                right,
                mode,
                scheduling_policy,
                model_soft_cap,
                capacity_soft_cap,
                &snapshot.active_jobs_by_submitter,
                &snapshot.active_jobs_by_model,
                &snapshot.active_capacity_by_model,
                &snapshot.leased_assignments_by_submitter,
                &snapshot.leased_assignments_by_job,
            ))
        })
        .map(|candidate| candidate.candidate.assignment_id);

    Ok(SchedulerDecision {
        selected_assignment_id: selected,
        blocked: snapshot.blocked,
    })
}

pub fn refresh_decode_plan_for_job(
    conn: &Transaction<'_>,
    network_id: &str,
    execution_plan: &InferenceExecutionPlan,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> ApiResult<Option<InferenceExecutionPlan>> {
    let topology = load_live_ring_topology(conn, network_id)?;
    if topology.workers.is_empty() {
        return Ok(None);
    }

    let device_metadata = topology
        .workers
        .iter()
        .map(|worker| {
            load_device_assignment_metadata(conn, network_id, &worker.device_id, scheduling_policy)
        })
        .collect::<ApiResult<Vec<_>>>()?;

    match ExecutionPlanner::refresh_decode_plan(
        execution_plan,
        &topology,
        scheduling_policy,
        &device_metadata,
    ) {
        Ok(plan) => Ok(Some(plan)),
        Err(ApiError::Conflict(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

fn load_scheduler_snapshot(
    conn: &Transaction<'_>,
    req: &ClaimInferenceAssignmentRequest,
    scheduling_policy: &InferenceSchedulingPolicy,
    now: &str,
) -> ApiResult<SchedulerSnapshot> {
    let mut runnable = Vec::new();
    let mut blocked = Vec::new();
    for candidate in load_scheduler_candidates(conn, req)? {
        match classify_candidate(&candidate, &req.device_id, now) {
            Ok(ready_at) => runnable.push(RunnableCandidate {
                candidate,
                ready_at,
            }),
            Err(reason) => blocked.push(SchedulerBlockedSession {
                session_id: candidate.session_id.clone(),
                job_id: candidate.job_id.clone(),
                assignment_id: candidate.assignment_id.clone(),
                reason,
            }),
        }
    }

    Ok(SchedulerSnapshot {
        runnable,
        blocked,
        active_jobs_by_submitter: load_active_job_counts_by_submitter(conn, &req.network_id)?,
        active_jobs_by_model: load_active_job_counts_by_model(conn, &req.network_id)?,
        active_capacity_by_model: load_active_capacity_by_model(conn, &req.network_id)?,
        leased_assignments_by_submitter: load_leased_assignment_counts_by_submitter(
            conn,
            &req.network_id,
        )?,
        leased_assignments_by_job: load_leased_assignment_counts_by_job(conn, &req.network_id)?,
        online_worker_count: load_online_ring_worker_count(conn, &req.network_id)?,
        online_capacity_units: load_online_ring_capacity_units(
            conn,
            &req.network_id,
            scheduling_policy,
        )?,
    })
}

fn classify_candidate(
    candidate: &SchedulerCandidate,
    requesting_device_id: &str,
    now: &str,
) -> Result<String, SchedulerBlockedReason> {
    match candidate.phase {
        SchedulerPhase::Prefill => match candidate.group_status.as_str() {
            "prefill_member" | "prefill_leased" => Ok(candidate.created_at.clone()),
            "prefill_active" => Err(SchedulerBlockedReason::AlreadyRunning),
            _ => Err(SchedulerBlockedReason::NotEligible),
        },
        SchedulerPhase::Decode => {
            let queue_status = candidate
                .decode_queue_status
                .as_deref()
                .ok_or(SchedulerBlockedReason::QueueMissing)?;
            match queue_status {
                "blocked_on_transfer" => Err(SchedulerBlockedReason::WaitingForTransfer),
                "active" => Err(SchedulerBlockedReason::AlreadyRunning),
                "completed" | "failed" => Err(SchedulerBlockedReason::NotEligible),
                "leased" => {
                    let lease_owner_device_id = candidate
                        .group_lease_owner_device_id
                        .as_deref()
                        .or(candidate.decode_lease_owner_device_id.as_deref());
                    let lease_expires_at = candidate
                        .group_lease_expires_at
                        .as_deref()
                        .or(candidate.decode_lease_expires_at.as_deref());
                    let held_by_peer = lease_owner_device_id
                        .as_deref()
                        .map(|owner| owner != requesting_device_id)
                        .unwrap_or(false)
                        && lease_expires_at.map(|expiry| expiry > now).unwrap_or(false);
                    if held_by_peer {
                        Err(SchedulerBlockedReason::LeaseHeldByPeer)
                    } else if candidate.group_status == "decode_pending_transfer" {
                        Err(SchedulerBlockedReason::WaitingForTransfer)
                    } else if matches!(
                        candidate.group_status.as_str(),
                        "decode_ready" | "decode_member" | "decode_leased"
                    ) {
                        Ok(candidate
                            .decode_ready_at
                            .clone()
                            .or_else(|| candidate.decode_updated_at.clone())
                            .unwrap_or_else(|| candidate.created_at.clone()))
                    } else {
                        Err(SchedulerBlockedReason::NotEligible)
                    }
                }
                "ready" => {
                    let held_by_peer = candidate
                        .group_lease_owner_device_id
                        .as_deref()
                        .map(|owner| owner != requesting_device_id)
                        .unwrap_or(false)
                        && candidate
                            .group_lease_expires_at
                            .as_deref()
                            .map(|expiry| expiry > now)
                            .unwrap_or(false);
                    if held_by_peer {
                        return Err(SchedulerBlockedReason::LeaseHeldByPeer);
                    }
                    if candidate.group_status == "decode_pending_transfer" {
                        Err(SchedulerBlockedReason::WaitingForTransfer)
                    } else if matches!(
                        candidate.group_status.as_str(),
                        "decode_ready" | "decode_member" | "decode_leased"
                    ) {
                        Ok(candidate
                            .decode_ready_at
                            .clone()
                            .or_else(|| candidate.decode_updated_at.clone())
                            .unwrap_or_else(|| candidate.created_at.clone()))
                    } else {
                        Err(SchedulerBlockedReason::NotEligible)
                    }
                }
                _ => Err(SchedulerBlockedReason::NotEligible),
            }
        }
    }
}

fn rank_candidate(
    candidate: &RunnableCandidate,
    mode: SchedulerPolicyMode,
    scheduling_policy: &InferenceSchedulingPolicy,
    model_soft_cap: u32,
    capacity_soft_cap: u32,
    active_jobs_by_submitter: &HashMap<String, u32>,
    active_jobs_by_model: &HashMap<String, u32>,
    active_capacity_by_model: &HashMap<String, u32>,
    leased_assignments_by_submitter: &HashMap<String, u32>,
    leased_assignments_by_job: &HashMap<String, u32>,
) -> (u8, u8, u8, u8, u8, u32, u8, u32, u32, String, String, String) {
    let submitter_active_jobs = active_jobs_by_submitter
        .get(&candidate.candidate.submitted_by_device_id)
        .copied()
        .unwrap_or_default();
    let model_active_jobs = active_jobs_by_model
        .get(&candidate.candidate.model_id)
        .copied()
        .unwrap_or_default();
    let model_active_capacity = active_capacity_by_model
        .get(&candidate.candidate.model_id)
        .copied()
        .unwrap_or_default();
    let submitter_leased_assignments = leased_assignments_by_submitter
        .get(&candidate.candidate.submitted_by_device_id)
        .copied()
        .unwrap_or_default();
    let job_leased_assignments = leased_assignments_by_job
        .get(&candidate.candidate.job_id)
        .copied()
        .unwrap_or_default();

    let fairness_rank = (
        u8::from(submitter_active_jobs >= scheduling_policy.submitter_active_job_soft_cap),
        u8::from(model_active_jobs >= model_soft_cap),
        u8::from(model_active_capacity >= capacity_soft_cap),
    );
    let runtime_mode_priority = runtime_mode_priority(candidate.candidate.runtime_mode);
    let policy_rank = mode_rank(
        mode,
        candidate,
        submitter_leased_assignments,
        job_leased_assignments,
    );
    let session_order_time = decode_session_order_time(candidate);

    (
        fairness_rank.0,
        fairness_rank.1,
        fairness_rank.2,
        runtime_mode_priority,
        policy_rank.0,
        policy_rank.1,
        policy_rank.2,
        policy_rank.3,
        policy_rank.4,
        policy_rank.5,
        session_order_time,
        candidate.candidate.assignment_id.clone(),
    )
}

fn runtime_mode_priority(mode: SchedulerPolicyMode) -> u8 {
    match mode {
        SchedulerPolicyMode::LatencyFirst => 0,
        SchedulerPolicyMode::ResilientEdge => 1,
        SchedulerPolicyMode::ThroughputFirst => 2,
        SchedulerPolicyMode::FitFirst => 3,
    }
}

fn mode_rank(
    mode: SchedulerPolicyMode,
    candidate: &RunnableCandidate,
    submitter_leased_assignments: u32,
    job_leased_assignments: u32,
) -> (u8, u32, u8, u32, u32, String) {
    let decode_bias = u8::from(candidate.candidate.phase != SchedulerPhase::Decode);
    let prefill_bias = u8::from(candidate.candidate.phase != SchedulerPhase::Prefill);
    let decode_group_fill_rank = decode_group_fill_rank(candidate);
    let cohort_order_time = decode_group_order_time(candidate);
    match mode {
        SchedulerPolicyMode::FitFirst => (
            prefill_bias,
            decode_group_fill_rank,
            decode_bias,
            submitter_leased_assignments,
            job_leased_assignments,
            candidate.candidate.created_at.clone(),
        ),
        SchedulerPolicyMode::ThroughputFirst => (
            decode_bias,
            decode_group_fill_rank,
            0,
            job_leased_assignments,
            submitter_leased_assignments,
            cohort_order_time.clone(),
        ),
        SchedulerPolicyMode::LatencyFirst => (
            decode_bias,
            decode_group_fill_rank,
            0,
            submitter_leased_assignments,
            job_leased_assignments,
            cohort_order_time,
        ),
        SchedulerPolicyMode::ResilientEdge => (
            decode_bias,
            decode_group_fill_rank,
            u8::from(!matches!(
                candidate.candidate.group_status.as_str(),
                "decode_ready" | "prefill_member"
            )),
            submitter_leased_assignments.saturating_add(job_leased_assignments),
            0,
            decode_group_order_time(candidate),
        ),
    }
}

fn decode_group_order_time(candidate: &RunnableCandidate) -> String {
    if matches!(candidate.candidate.phase, SchedulerPhase::Decode)
        && candidate.candidate.decode_cohort_oldest_ready_at.is_some()
    {
        return candidate
            .candidate
            .decode_cohort_oldest_ready_at
            .clone()
            .unwrap_or_else(|| candidate.ready_at.clone());
    }
    candidate.ready_at.clone()
}

fn decode_session_order_time(candidate: &RunnableCandidate) -> String {
    if matches!(candidate.candidate.phase, SchedulerPhase::Decode) {
        return format!(
            "{}:{}",
            decode_session_lease_rank(candidate),
            candidate.ready_at
        );
    }
    candidate.candidate.assigned_at.clone()
}

fn decode_session_lease_rank(candidate: &RunnableCandidate) -> &'static str {
    if !matches!(candidate.candidate.phase, SchedulerPhase::Decode) {
        return "z";
    }
    match candidate.candidate.decode_queue_status.as_deref() {
        Some("leased") => "a",
        Some("ready") => "b",
        Some("active") => "c",
        _ => "z",
    }
}

fn decode_group_fill_rank(candidate: &RunnableCandidate) -> u32 {
    if !matches!(candidate.candidate.phase, SchedulerPhase::Decode) {
        return 0;
    }

    let is_owned = candidate
        .candidate
        .group_lease_owner_device_id
        .as_deref()
        .is_some();
    if !is_owned {
        let fresh_ready = candidate
            .candidate
            .decode_cohort_ready_sessions
            .max(1)
            .min(31);
        let fresh_blocked = candidate
            .candidate
            .decode_cohort_blocked_sessions
            .min(31);
        let fresh_target = candidate
            .candidate
            .decode_lease_target_session_count
            .unwrap_or(1)
            .min(31);
        let fresh_score = (fresh_ready << 10)
            .saturating_add((31u32.saturating_sub(fresh_blocked)) << 5)
            .saturating_add(fresh_target);
        return 1_999_999u32.saturating_sub(fresh_score);
    }

    let cohort_fill = candidate
        .candidate
        .decode_cohort_leased_sessions
        .saturating_add(candidate.candidate.decode_cohort_active_sessions);
    let target = candidate
        .candidate
        .decode_lease_target_session_count
        .unwrap_or(cohort_fill.max(1));
    if cohort_fill < target {
        let remaining_capacity = target.saturating_sub(cohort_fill);
        let remaining_rank = 31u32.saturating_sub(remaining_capacity.min(31));
        let fill_density_rank = 31u32.saturating_sub(cohort_fill.min(31));
        let ready_rank = 31u32.saturating_sub(candidate.candidate.decode_cohort_ready_sessions.min(31));
        let blocked_rank = candidate.candidate.decode_cohort_blocked_sessions.min(31);
        return (remaining_rank << 15)
            .saturating_add(fill_density_rank << 10)
            .saturating_add(ready_rank << 5)
            .saturating_add(blocked_rank);
    } else {
        2_000_000
    }
}

fn load_scheduler_candidates(
    conn: &Transaction<'_>,
    req: &ClaimInferenceAssignmentRequest,
) -> ApiResult<Vec<SchedulerCandidate>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT
                a.assignment_id,
                a.job_id,
                s.session_id,
                j.model_id,
                j.execution_plan_json,
                j.submitted_by_device_id,
                j.created_at,
                a.assigned_at,
                sg.group_id,
                sg.phase,
                s.status,
                sg.status,
                dq.status,
                dq.ready_at,
                dq.lease_owner_device_id,
                dq.lease_expires_at,
                dq.updated_at,
                sg.lease_owner_device_id,
                sg.lease_expires_at,
                dq.lease_target_session_count,
                COALESCE((
                    SELECT SUM(CASE WHEN peer.status = 'ready' THEN 1 ELSE 0 END)
                    FROM inference_decode_queue peer
                    WHERE peer.network_id = a.network_id
                      AND COALESCE(peer.batch_group_key, peer.group_id) =
                          COALESCE(dq.batch_group_key, dq.group_id)
                ), 0),
                COALESCE((
                    SELECT SUM(CASE WHEN peer.status = 'blocked_on_transfer' THEN 1 ELSE 0 END)
                    FROM inference_decode_queue peer
                    WHERE peer.network_id = a.network_id
                      AND COALESCE(peer.batch_group_key, peer.group_id) =
                          COALESCE(dq.batch_group_key, dq.group_id)
                ), 0),
                (
                    SELECT MIN(peer.ready_at)
                    FROM inference_decode_queue peer
                    WHERE peer.network_id = a.network_id
                      AND COALESCE(peer.batch_group_key, peer.group_id) =
                          COALESCE(dq.batch_group_key, dq.group_id)
                      AND peer.status = 'ready'
                ),
                COALESCE((
                    SELECT SUM(CASE WHEN peer.status = 'leased' THEN 1 ELSE 0 END)
                    FROM inference_decode_queue peer
                    WHERE peer.network_id = a.network_id
                      AND COALESCE(peer.batch_group_key, peer.group_id) =
                          COALESCE(dq.batch_group_key, dq.group_id)
                ), 0),
                COALESCE((
                    SELECT SUM(CASE WHEN peer.status = 'active' THEN 1 ELSE 0 END)
                    FROM inference_decode_queue peer
                    WHERE peer.network_id = a.network_id
                      AND COALESCE(peer.batch_group_key, peer.group_id) =
                          COALESCE(dq.batch_group_key, dq.group_id)
                ), 0)
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            INNER JOIN inference_sessions s ON s.job_id = j.job_id
            INNER JOIN inference_serving_groups sg
                ON sg.job_id = a.job_id AND sg.device_id = a.device_id
            LEFT JOIN inference_decode_queue dq
                ON dq.session_id = s.session_id AND sg.phase = 'decode'
            WHERE a.device_id = ?
              AND a.network_id = ?
              AND a.active_segment_id = j.active_segment_id
              AND a.status IN ('pending', 'leased')
              AND j.status IN ('dispatched', 'running')
              AND (
                    sg.status IN (
                        'prefill_member',
                        'prefill_leased',
                        'prefill_active',
                        'decode_member',
                        'decode_ready',
                        'decode_leased',
                        'decode_pending_transfer',
                        'decode_active'
                    )
                    OR dq.status IN ('ready', 'leased', 'active', 'blocked_on_transfer')
                  )
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let rows = stmt.query_map(params![&req.device_id, &req.network_id], |row| {
        let phase: String = row.get(9)?;
        Ok(SchedulerCandidate {
            assignment_id: row.get(0)?,
            job_id: row.get(1)?,
            session_id: row.get(2)?,
            model_id: row.get(3)?,
            runtime_mode: parse_runtime_mode(
                serde_json::from_str::<InferenceExecutionPlan>(&row.get::<_, String>(4)?)
                    .map_err(|err| to_from_sql_error(err.to_string()))?
                    .runtime_mode,
            ),
            submitted_by_device_id: row.get(5)?,
            created_at: row.get(6)?,
            assigned_at: row.get(7)?,
            phase: parse_scheduler_phase(&phase).map_err(to_from_sql_error)?,
            group_status: row.get(11)?,
            decode_queue_status: row.get(12)?,
            decode_ready_at: row.get(13)?,
            decode_lease_owner_device_id: row.get(14)?,
            decode_lease_expires_at: row.get(15)?,
            decode_updated_at: row.get(16)?,
            group_lease_owner_device_id: row.get(17)?,
            group_lease_expires_at: row.get(18)?,
            decode_lease_target_session_count: row.get::<_, Option<i64>>(19)?.map(|v| v as u32),
            decode_cohort_ready_sessions: row.get::<_, i64>(20)? as u32,
            decode_cohort_blocked_sessions: row.get::<_, i64>(21)? as u32,
            decode_cohort_oldest_ready_at: row.get(22)?,
            decode_cohort_leased_sessions: row.get::<_, i64>(23)? as u32,
            decode_cohort_active_sessions: row.get::<_, i64>(24)? as u32,
        })
    });
    let candidates = rows
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(candidates
        .into_iter()
        .filter(|candidate| claim_mode_matches(req.claim_mode, candidate.phase))
        .collect())
}

fn claim_mode_matches(mode: WorkClaimMode, phase: SchedulerPhase) -> bool {
    match mode {
        WorkClaimMode::Any => true,
        WorkClaimMode::Prefill => matches!(phase, SchedulerPhase::Prefill),
        WorkClaimMode::Decode => matches!(phase, SchedulerPhase::Decode),
    }
}

fn parse_scheduler_phase(value: &str) -> Result<SchedulerPhase, String> {
    match value {
        "prefill" => Ok(SchedulerPhase::Prefill),
        "decode" => Ok(SchedulerPhase::Decode),
        other => Err(format!("Unknown scheduler phase {}", other)),
    }
}

fn parse_runtime_mode(value: InferenceRuntimeMode) -> SchedulerPolicyMode {
    match value {
        InferenceRuntimeMode::FitFirst => SchedulerPolicyMode::FitFirst,
        InferenceRuntimeMode::ThroughputFirst => SchedulerPolicyMode::ThroughputFirst,
        InferenceRuntimeMode::LatencyFirst => SchedulerPolicyMode::LatencyFirst,
        InferenceRuntimeMode::ResilientEdge => SchedulerPolicyMode::ResilientEdge,
    }
}

fn to_from_sql_error(message: String) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        8,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::other(message)),
    )
}

fn reconcile_stale_scheduler_leases(
    conn: &Transaction<'_>,
    network_id: &str,
    now: &str,
) -> ApiResult<()> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT a.assignment_id, a.job_id, a.device_id, s.session_id, sg.phase, sg.group_id
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            INNER JOIN inference_sessions s ON s.job_id = j.job_id
            LEFT JOIN inference_serving_groups sg
                ON sg.job_id = a.job_id AND sg.device_id = a.device_id
            WHERE a.network_id = ?
              AND a.status = 'leased'
              AND a.lease_expires_at IS NOT NULL
              AND a.lease_expires_at <= ?
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let expired = stmt
        .query_map(params![network_id, now], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    for (assignment_id, job_id, device_id, session_id, phase, group_id) in expired {
        conn.execute(
            r#"
            UPDATE inference_job_assignments
            SET status = 'pending', lease_expires_at = NULL
            WHERE assignment_id = ?
            "#,
            params![assignment_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        match phase.as_deref() {
            Some("prefill") => {
                if let Some(group_id) = &group_id {
                    conn.execute(
                        r#"
                        UPDATE inference_serving_groups
                        SET status = 'prefill_member', updated_at = ?, last_error = NULL
                        WHERE job_id = ? AND group_id = ? AND device_id = ?
                          AND status = 'prefill_leased'
                        "#,
                        params![now, &job_id, group_id, &device_id],
                    )
                    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                }
            }
            Some("decode") => {
                if let Some(group_id) = &group_id {
                    conn.execute(
                        r#"
                        UPDATE inference_serving_groups
                        SET status = 'decode_ready', updated_at = ?, last_error = NULL
                        WHERE job_id = ? AND group_id = ? AND device_id = ?
                          AND status = 'decode_leased'
                        "#,
                        params![now, &job_id, group_id, &device_id],
                    )
                    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
                }
                conn.execute(
                    r#"
                    UPDATE inference_decode_queue
                    SET status = 'ready',
                        lease_owner_device_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?,
                        last_error = NULL
                    WHERE session_id = ?
                      AND status = 'leased'
                      AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                    "#,
                    params![now, &session_id, now],
                )
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn now_rfc3339() -> ApiResult<String> {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))
}

fn load_active_job_counts_by_submitter(
    conn: &Transaction<'_>,
    network_id: &str,
) -> ApiResult<HashMap<String, u32>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT j.submitted_by_device_id, COUNT(*)
            FROM inference_jobs j
            WHERE j.network_id = ?
              AND j.status IN ('dispatched', 'running')
              AND EXISTS (
                    SELECT 1
                    FROM inference_serving_groups sg
                    WHERE sg.job_id = j.job_id
                      AND sg.status IN ('prefill_leased', 'prefill_active', 'decode_leased', 'decode_active')
              )
            GROUP BY j.submitted_by_device_id
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    collect_count_map(&mut stmt, params![network_id])
}

fn load_active_job_counts_by_model(
    conn: &Transaction<'_>,
    network_id: &str,
) -> ApiResult<HashMap<String, u32>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT j.model_id, COUNT(*)
            FROM inference_jobs j
            WHERE j.network_id = ?
              AND j.status IN ('dispatched', 'running')
              AND EXISTS (
                    SELECT 1
                    FROM inference_serving_groups sg
                    WHERE sg.job_id = j.job_id
                      AND sg.status IN ('prefill_leased', 'prefill_active', 'decode_leased', 'decode_active')
              )
            GROUP BY j.model_id
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    collect_count_map(&mut stmt, params![network_id])
}

fn load_active_capacity_by_model(
    conn: &Transaction<'_>,
    network_id: &str,
) -> ApiResult<HashMap<String, u32>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT j.model_id, COALESCE(SUM(a.assigned_capacity_units), 0)
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            INNER JOIN inference_serving_groups sg
                ON sg.job_id = a.job_id AND sg.device_id = a.device_id
            WHERE j.network_id = ?
              AND j.status IN ('dispatched', 'running')
              AND sg.status IN ('prefill_leased', 'prefill_active', 'decode_leased', 'decode_active')
            GROUP BY j.model_id
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    collect_count_map(&mut stmt, params![network_id])
}

fn load_leased_assignment_counts_by_submitter(
    conn: &Transaction<'_>,
    network_id: &str,
) -> ApiResult<HashMap<String, u32>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT j.submitted_by_device_id, COUNT(*)
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            INNER JOIN inference_serving_groups sg
                ON sg.job_id = a.job_id AND sg.device_id = a.device_id
            WHERE j.network_id = ?
              AND sg.status IN ('prefill_leased', 'decode_leased', 'prefill_active', 'decode_active')
            GROUP BY j.submitted_by_device_id
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    collect_count_map(&mut stmt, params![network_id])
}

fn load_leased_assignment_counts_by_job(
    conn: &Transaction<'_>,
    network_id: &str,
) -> ApiResult<HashMap<String, u32>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT a.job_id, COUNT(*)
            FROM inference_job_assignments a
            INNER JOIN inference_serving_groups sg
                ON sg.job_id = a.job_id AND sg.device_id = a.device_id
            WHERE a.network_id = ?
              AND sg.status IN ('prefill_leased', 'decode_leased', 'prefill_active', 'decode_active')
            GROUP BY a.job_id
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    collect_count_map(&mut stmt, params![network_id])
}

fn load_online_ring_worker_count(conn: &Transaction<'_>, network_id: &str) -> ApiResult<u32> {
    conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM devices
        WHERE network_id = ?
          AND ring_position IS NOT NULL
          AND status = 'online'
        "#,
        params![network_id],
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count.max(0) as u32)
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

fn load_online_ring_capacity_units(
    conn: &Transaction<'_>,
    network_id: &str,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> ApiResult<u32> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT capabilities
            FROM devices
            WHERE network_id = ?
              AND ring_position IS NOT NULL
              AND status = 'online'
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let mut total = 0u32;
    for row in stmt
        .query_map(params![network_id], |row| row.get::<_, String>(0))
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
    {
        let capabilities_json =
            row.map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        let capabilities: DeviceCapabilities =
            serde_json::from_str(&capabilities_json).map_err(|e| {
                ApiError::Internal(format!("Failed to parse device capabilities: {}", e))
            })?;
        total = total.saturating_add(
            device_metadata_from_capabilities(scheduling_policy, &capabilities)
                .assigned_capacity_units,
        );
    }
    Ok(total.max(1))
}

fn collect_count_map<P: rusqlite::Params>(
    stmt: &mut rusqlite::Statement<'_>,
    params: P,
) -> ApiResult<HashMap<String, u32>> {
    stmt.query_map(params, |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
    .collect::<Result<Vec<_>, _>>()
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
    .map(|rows| {
        rows.into_iter()
            .map(|(key, value)| (key, value.max(0) as u32))
            .collect()
    })
}

fn load_live_ring_topology(conn: &Transaction<'_>, network_id: &str) -> ApiResult<RingTopology> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT device_id, peer_id, ring_position, status, contributed_memory, shard_model_id,
                   shard_column_start, shard_column_end, connectivity_state, listen_addrs,
                   direct_candidates, left_neighbor_id, right_neighbor_id
            FROM devices
            WHERE network_id = ?
              AND ring_position IS NOT NULL
              AND shard_model_id IS NOT NULL
              AND status = 'online'
            ORDER BY ring_position ASC
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let workers = stmt
        .query_map(params![network_id], |row| {
            let connectivity_state_json = row.get::<_, Option<String>>(8)?;
            let listen_addrs_json = row.get::<_, Option<String>>(9)?;
            let direct_candidates_json = row.get::<_, Option<String>>(10)?;
            Ok(WorkerTopologyInfo {
                device_id: row.get(0)?,
                peer_id: row.get(1)?,
                position: row.get::<_, i64>(2)? as u32,
                status: row.get(3)?,
                contributed_memory: row.get::<_, Option<i64>>(4)?.unwrap_or_default().max(0) as u64,
                shard: ModelShard {
                    model_id: row.get(5)?,
                    column_range: (
                        row.get::<_, Option<i64>>(6)?.unwrap_or_default().max(0) as u32,
                        row.get::<_, Option<i64>>(7)?.unwrap_or_default().max(0) as u32,
                    ),
                    estimated_memory: row.get::<_, Option<i64>>(4)?.unwrap_or_default().max(0)
                        as u64,
                },
                connectivity_state: connectivity_state_json
                    .as_deref()
                    .map(serde_json::from_str::<DeviceConnectivityState>)
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            8,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?,
                listen_addrs: listen_addrs_json
                    .as_deref()
                    .map(serde_json::from_str::<Vec<String>>)
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            9,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?
                    .unwrap_or_default(),
                direct_candidates: direct_candidates_json
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            10,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?
                    .unwrap_or_default(),
                left_neighbor: row.get::<_, Option<String>>(11)?.unwrap_or_default(),
                right_neighbor: row.get::<_, Option<String>>(12)?.unwrap_or_default(),
            })
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(RingTopology {
        ring_stable: !workers.is_empty(),
        workers,
        peer_punch_plans: Vec::new(),
    })
}

fn load_device_assignment_metadata(
    conn: &Transaction<'_>,
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
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .ok_or_else(|| {
            ApiError::NotFound(format!(
                "Device {} not found in network {}",
                device_id, network_id
            ))
        })?;
    let capabilities: DeviceCapabilities = serde_json::from_str(&capabilities_json)
        .map_err(|e| ApiError::Internal(format!("Failed to parse device capabilities: {}", e)))?;
    Ok(device_metadata_from_capabilities(
        scheduling_policy,
        &capabilities,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn base_candidate(phase: SchedulerPhase) -> SchedulerCandidate {
        SchedulerCandidate {
            assignment_id: "assignment-1".into(),
            job_id: "job-1".into(),
            session_id: "session-1".into(),
            model_id: "model-a".into(),
            runtime_mode: SchedulerPolicyMode::FitFirst,
            submitted_by_device_id: "submitter-a".into(),
            created_at: "2026-01-01T00:00:00Z".into(),
            assigned_at: "2026-01-01T00:00:00Z".into(),
            phase,
            group_status: "prefill_member".into(),
            decode_queue_status: None,
            decode_ready_at: None,
            decode_lease_owner_device_id: None,
            decode_lease_expires_at: None,
            decode_updated_at: None,
            group_lease_owner_device_id: None,
            group_lease_expires_at: None,
            decode_lease_target_session_count: None,
            decode_cohort_ready_sessions: 0,
            decode_cohort_blocked_sessions: 0,
            decode_cohort_oldest_ready_at: None,
            decode_cohort_leased_sessions: 0,
            decode_cohort_active_sessions: 0,
        }
    }

    #[test]
    fn blocked_decode_transfer_is_reported_from_scheduler_state() {
        let mut candidate = base_candidate(SchedulerPhase::Decode);
        candidate.group_status = "decode_pending_transfer".into();
        candidate.decode_queue_status = Some("blocked_on_transfer".into());

        let result = classify_candidate(&candidate, "worker-1", "2026-01-01T00:00:00Z");
        assert_eq!(result, Err(SchedulerBlockedReason::WaitingForTransfer));
    }

    #[test]
    fn queue_ordering_prefers_older_ready_decode_work_for_latency() {
        let mut older = base_candidate(SchedulerPhase::Decode);
        older.assignment_id = "older".into();
        older.group_status = "decode_ready".into();
        older.decode_queue_status = Some("ready".into());
        older.decode_ready_at = Some("2026-01-01T00:00:00Z".into());

        let mut newer = older.clone();
        newer.assignment_id = "newer".into();
        newer.decode_ready_at = Some("2026-01-01T00:01:00Z".into());

        let older = RunnableCandidate {
            ready_at: older.decode_ready_at.clone().unwrap(),
            candidate: older,
        };
        let newer = RunnableCandidate {
            ready_at: newer.decode_ready_at.clone().unwrap(),
            candidate: newer,
        };

        let left = mode_rank(SchedulerPolicyMode::LatencyFirst, &older, 0, 0);
        let right = mode_rank(SchedulerPolicyMode::LatencyFirst, &newer, 0, 0);
        assert!(left < right);
    }

    #[test]
    fn runtime_mode_priority_prefers_latency_before_fit_first() {
        let mut latency = base_candidate(SchedulerPhase::Decode);
        latency.assignment_id = "latency".into();
        latency.runtime_mode = SchedulerPolicyMode::LatencyFirst;
        latency.group_status = "decode_ready".into();
        latency.decode_queue_status = Some("ready".into());
        latency.decode_ready_at = Some("2026-01-01T00:01:00Z".into());

        let mut fit = latency.clone();
        fit.assignment_id = "fit".into();
        fit.runtime_mode = SchedulerPolicyMode::FitFirst;

        let latency = RunnableCandidate {
            ready_at: latency.decode_ready_at.clone().unwrap(),
            candidate: latency,
        };
        let fit = RunnableCandidate {
            ready_at: fit.decode_ready_at.clone().unwrap(),
            candidate: fit,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &latency,
            SchedulerPolicyMode::FitFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &fit,
            SchedulerPolicyMode::FitFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn fit_first_respects_submitter_soft_cap_before_arrival_order() {
        let mut capped = base_candidate(SchedulerPhase::Prefill);
        capped.assignment_id = "capped".into();
        capped.submitted_by_device_id = "submitter-a".into();
        capped.created_at = "2026-01-01T00:00:00Z".into();

        let mut uncapped = base_candidate(SchedulerPhase::Prefill);
        uncapped.assignment_id = "uncapped".into();
        uncapped.submitted_by_device_id = "submitter-b".into();
        uncapped.created_at = "2026-01-01T00:01:00Z".into();

        let capped = RunnableCandidate {
            ready_at: capped.created_at.clone(),
            candidate: capped,
        };
        let uncapped = RunnableCandidate {
            ready_at: uncapped.created_at.clone(),
            candidate: uncapped,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &capped,
            SchedulerPolicyMode::FitFirst,
            &policy,
            1,
            1,
            &HashMap::from([("submitter-a".into(), 1), ("submitter-b".into(), 0)]),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &uncapped,
            SchedulerPolicyMode::FitFirst,
            &policy,
            1,
            1,
            &HashMap::from([("submitter-a".into(), 1), ("submitter-b".into(), 0)]),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(right < left);
    }

    #[test]
    fn decode_ready_candidate_is_blocked_when_group_lease_is_held_by_peer() {
        let mut candidate = base_candidate(SchedulerPhase::Decode);
        candidate.group_status = "decode_ready".into();
        candidate.decode_queue_status = Some("ready".into());
        candidate.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        candidate.decode_updated_at = Some("2026-01-01T00:00:01Z".into());
        candidate.group_lease_owner_device_id = Some("worker-peer".into());
        candidate.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());

        let blocked = classify_candidate(&candidate, "worker-1", "2026-01-01T00:05:00Z");
        assert_eq!(blocked, Err(SchedulerBlockedReason::LeaseHeldByPeer));

        let same_owner = classify_candidate(&candidate, "worker-peer", "2026-01-01T00:05:00Z");
        assert_eq!(same_owner, Ok("2026-01-01T00:00:01Z".into()));
    }

    #[test]
    fn latency_prefers_filling_owned_decode_group_before_fresh_ready_group() {
        let mut owned = base_candidate(SchedulerPhase::Decode);
        owned.assignment_id = "owned".into();
        owned.group_status = "decode_leased".into();
        owned.decode_queue_status = Some("ready".into());
        owned.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        owned.group_lease_owner_device_id = Some("worker-1".into());
        owned.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());

        let mut fresh = owned.clone();
        fresh.assignment_id = "fresh".into();
        fresh.group_lease_owner_device_id = None;
        fresh.group_lease_expires_at = None;
        fresh.decode_ready_at = Some("2026-01-01T00:00:01Z".into());

        let owned = RunnableCandidate {
            ready_at: owned.decode_ready_at.clone().unwrap(),
            candidate: owned,
        };
        let fresh = RunnableCandidate {
            ready_at: fresh.decode_ready_at.clone().unwrap(),
            candidate: fresh,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &owned,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &fresh,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_continuing_owned_decode_group_until_target_is_satisfied() {
        let mut under_target = base_candidate(SchedulerPhase::Decode);
        under_target.assignment_id = "under-target".into();
        under_target.group_status = "decode_leased".into();
        under_target.decode_queue_status = Some("ready".into());
        under_target.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        under_target.group_lease_owner_device_id = Some("worker-1".into());
        under_target.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        under_target.decode_lease_target_session_count = Some(3);
        under_target.decode_cohort_leased_sessions = 1;
        under_target.decode_cohort_active_sessions = 1;

        let mut satisfied = under_target.clone();
        satisfied.assignment_id = "satisfied".into();
        satisfied.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        satisfied.decode_cohort_leased_sessions = 2;
        satisfied.decode_cohort_active_sessions = 1;

        let under_target = RunnableCandidate {
            ready_at: under_target.decode_ready_at.clone().unwrap(),
            candidate: under_target,
        };
        let satisfied = RunnableCandidate {
            ready_at: satisfied.decode_ready_at.clone().unwrap(),
            candidate: satisfied,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &under_target,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &satisfied,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_opens_fresh_ready_group_once_owned_group_reaches_target() {
        let mut satisfied = base_candidate(SchedulerPhase::Decode);
        satisfied.assignment_id = "satisfied".into();
        satisfied.group_status = "decode_leased".into();
        satisfied.decode_queue_status = Some("ready".into());
        satisfied.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        satisfied.group_lease_owner_device_id = Some("worker-1".into());
        satisfied.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        satisfied.decode_lease_target_session_count = Some(2);
        satisfied.decode_cohort_leased_sessions = 1;
        satisfied.decode_cohort_active_sessions = 1;

        let mut fresh = satisfied.clone();
        fresh.assignment_id = "fresh".into();
        fresh.group_status = "decode_ready".into();
        fresh.group_lease_owner_device_id = None;
        fresh.group_lease_expires_at = None;
        fresh.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        fresh.decode_cohort_leased_sessions = 0;
        fresh.decode_cohort_active_sessions = 0;

        let satisfied = RunnableCandidate {
            ready_at: satisfied.decode_ready_at.clone().unwrap(),
            candidate: satisfied,
        };
        let fresh = RunnableCandidate {
            ready_at: fresh.decode_ready_at.clone().unwrap(),
            candidate: fresh,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &fresh,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &satisfied,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_owned_group_with_more_remaining_pooled_capacity() {
        let mut more_remaining = base_candidate(SchedulerPhase::Decode);
        more_remaining.assignment_id = "more-remaining".into();
        more_remaining.group_status = "decode_leased".into();
        more_remaining.decode_queue_status = Some("ready".into());
        more_remaining.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        more_remaining.group_lease_owner_device_id = Some("worker-1".into());
        more_remaining.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        more_remaining.decode_lease_target_session_count = Some(5);
        more_remaining.decode_cohort_leased_sessions = 1;
        more_remaining.decode_cohort_active_sessions = 1;

        let mut less_remaining = more_remaining.clone();
        less_remaining.assignment_id = "less-remaining".into();
        less_remaining.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        less_remaining.decode_lease_target_session_count = Some(3);
        less_remaining.decode_cohort_leased_sessions = 1;
        less_remaining.decode_cohort_active_sessions = 1;

        let more_remaining = RunnableCandidate {
            ready_at: more_remaining.decode_ready_at.clone().unwrap(),
            candidate: more_remaining,
        };
        let less_remaining = RunnableCandidate {
            ready_at: less_remaining.decode_ready_at.clone().unwrap(),
            candidate: less_remaining,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &more_remaining,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &less_remaining,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_fresh_decode_group_with_larger_pooled_target() {
        let mut larger = base_candidate(SchedulerPhase::Decode);
        larger.assignment_id = "larger".into();
        larger.group_status = "decode_ready".into();
        larger.decode_queue_status = Some("ready".into());
        larger.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        larger.decode_lease_target_session_count = Some(4);
        larger.decode_cohort_ready_sessions = 1;
        larger.decode_cohort_blocked_sessions = 0;
        larger.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:02Z".into());

        let mut smaller = larger.clone();
        smaller.assignment_id = "smaller".into();
        smaller.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        smaller.decode_lease_target_session_count = Some(2);

        let larger = RunnableCandidate {
            ready_at: larger.decode_ready_at.clone().unwrap(),
            candidate: larger,
        };
        let smaller = RunnableCandidate {
            ready_at: smaller.decode_ready_at.clone().unwrap(),
            candidate: smaller,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &larger,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &smaller,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_fresh_decode_group_with_more_ready_runway() {
        let mut more_ready = base_candidate(SchedulerPhase::Decode);
        more_ready.assignment_id = "more-ready".into();
        more_ready.group_status = "decode_ready".into();
        more_ready.decode_queue_status = Some("ready".into());
        more_ready.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        more_ready.decode_lease_target_session_count = Some(4);
        more_ready.decode_cohort_ready_sessions = 3;
        more_ready.decode_cohort_blocked_sessions = 0;
        more_ready.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:02Z".into());

        let mut less_ready = more_ready.clone();
        less_ready.assignment_id = "less-ready".into();
        less_ready.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        less_ready.decode_cohort_ready_sessions = 1;

        let more_ready = RunnableCandidate {
            ready_at: more_ready.decode_ready_at.clone().unwrap(),
            candidate: more_ready,
        };
        let less_ready = RunnableCandidate {
            ready_at: less_ready.decode_ready_at.clone().unwrap(),
            candidate: less_ready,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &more_ready,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &less_ready,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_fresh_decode_group_with_less_transfer_debt_when_ready_runway_ties() {
        let mut less_blocked = base_candidate(SchedulerPhase::Decode);
        less_blocked.assignment_id = "less-blocked".into();
        less_blocked.group_status = "decode_ready".into();
        less_blocked.decode_queue_status = Some("ready".into());
        less_blocked.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        less_blocked.decode_lease_target_session_count = Some(4);
        less_blocked.decode_cohort_ready_sessions = 2;
        less_blocked.decode_cohort_blocked_sessions = 1;
        less_blocked.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:02Z".into());

        let mut more_blocked = less_blocked.clone();
        more_blocked.assignment_id = "more-blocked".into();
        more_blocked.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        more_blocked.decode_cohort_blocked_sessions = 3;

        let less_blocked = RunnableCandidate {
            ready_at: less_blocked.decode_ready_at.clone().unwrap(),
            candidate: less_blocked,
        };
        let more_blocked = RunnableCandidate {
            ready_at: more_blocked.decode_ready_at.clone().unwrap(),
            candidate: more_blocked,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &less_blocked,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &more_blocked,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_fresh_decode_cohort_with_older_cohort_ready_time_when_scores_tie() {
        let mut older_cohort = base_candidate(SchedulerPhase::Decode);
        older_cohort.assignment_id = "older-cohort".into();
        older_cohort.group_status = "decode_ready".into();
        older_cohort.decode_queue_status = Some("ready".into());
        older_cohort.decode_ready_at = Some("2026-01-01T00:00:05Z".into());
        older_cohort.decode_lease_target_session_count = Some(4);
        older_cohort.decode_cohort_ready_sessions = 2;
        older_cohort.decode_cohort_blocked_sessions = 1;
        older_cohort.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let mut newer_cohort = older_cohort.clone();
        newer_cohort.assignment_id = "newer-cohort".into();
        newer_cohort.decode_ready_at = Some("2026-01-01T00:00:02Z".into());
        newer_cohort.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:03Z".into());

        let older_cohort = RunnableCandidate {
            ready_at: older_cohort.decode_ready_at.clone().unwrap(),
            candidate: older_cohort,
        };
        let newer_cohort = RunnableCandidate {
            ready_at: newer_cohort.decode_ready_at.clone().unwrap(),
            candidate: newer_cohort,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &older_cohort,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &newer_cohort,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_oldest_ready_session_within_same_fresh_decode_cohort() {
        let mut older_session = base_candidate(SchedulerPhase::Decode);
        older_session.assignment_id = "older-session".into();
        older_session.group_status = "decode_ready".into();
        older_session.decode_queue_status = Some("ready".into());
        older_session.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        older_session.decode_lease_target_session_count = Some(4);
        older_session.decode_cohort_ready_sessions = 2;
        older_session.decode_cohort_blocked_sessions = 1;
        older_session.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let mut newer_session = older_session.clone();
        newer_session.assignment_id = "newer-session".into();
        newer_session.decode_ready_at = Some("2026-01-01T00:00:03Z".into());
        newer_session.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let older_session = RunnableCandidate {
            ready_at: older_session.decode_ready_at.clone().unwrap(),
            candidate: older_session,
        };
        let newer_session = RunnableCandidate {
            ready_at: newer_session.decode_ready_at.clone().unwrap(),
            candidate: newer_session,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &older_session,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &newer_session,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_draining_leased_decode_session_before_ready_sibling_in_owned_cohort() {
        let mut leased = base_candidate(SchedulerPhase::Decode);
        leased.assignment_id = "leased".into();
        leased.group_status = "decode_leased".into();
        leased.decode_queue_status = Some("leased".into());
        leased.decode_ready_at = Some("2026-01-01T00:00:03Z".into());
        leased.group_lease_owner_device_id = Some("worker-1".into());
        leased.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        leased.decode_lease_target_session_count = Some(3);
        leased.decode_cohort_leased_sessions = 2;
        leased.decode_cohort_active_sessions = 0;
        leased.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let mut ready = leased.clone();
        ready.assignment_id = "ready".into();
        ready.decode_queue_status = Some("ready".into());
        ready.decode_ready_at = Some("2026-01-01T00:00:01Z".into());

        let leased = RunnableCandidate {
            ready_at: leased.decode_ready_at.clone().unwrap(),
            candidate: leased,
        };
        let ready = RunnableCandidate {
            ready_at: ready.decode_ready_at.clone().unwrap(),
            candidate: ready,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &leased,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &ready,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_denser_owned_decode_cohort_when_remaining_capacity_ties() {
        let mut denser = base_candidate(SchedulerPhase::Decode);
        denser.assignment_id = "denser".into();
        denser.group_status = "decode_leased".into();
        denser.decode_queue_status = Some("leased".into());
        denser.decode_ready_at = Some("2026-01-01T00:00:03Z".into());
        denser.group_lease_owner_device_id = Some("worker-1".into());
        denser.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        denser.decode_lease_target_session_count = Some(5);
        denser.decode_cohort_leased_sessions = 2;
        denser.decode_cohort_active_sessions = 1;
        denser.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let mut thinner = denser.clone();
        thinner.assignment_id = "thinner".into();
        thinner.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        thinner.decode_lease_target_session_count = Some(3);
        thinner.decode_cohort_leased_sessions = 1;
        thinner.decode_cohort_active_sessions = 0;

        let denser = RunnableCandidate {
            ready_at: denser.decode_ready_at.clone().unwrap(),
            candidate: denser,
        };
        let thinner = RunnableCandidate {
            ready_at: thinner.decode_ready_at.clone().unwrap(),
            candidate: thinner,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &denser,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &thinner,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_owned_decode_cohort_with_less_blocked_transfer_debt_when_fill_ties() {
        let mut less_blocked = base_candidate(SchedulerPhase::Decode);
        less_blocked.assignment_id = "less-blocked".into();
        less_blocked.group_status = "decode_leased".into();
        less_blocked.decode_queue_status = Some("leased".into());
        less_blocked.decode_ready_at = Some("2026-01-01T00:00:03Z".into());
        less_blocked.group_lease_owner_device_id = Some("worker-1".into());
        less_blocked.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        less_blocked.decode_lease_target_session_count = Some(5);
        less_blocked.decode_cohort_leased_sessions = 2;
        less_blocked.decode_cohort_active_sessions = 1;
        less_blocked.decode_cohort_blocked_sessions = 1;
        less_blocked.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let mut more_blocked = less_blocked.clone();
        more_blocked.assignment_id = "more-blocked".into();
        more_blocked.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        more_blocked.decode_cohort_blocked_sessions = 4;

        let less_blocked = RunnableCandidate {
            ready_at: less_blocked.decode_ready_at.clone().unwrap(),
            candidate: less_blocked,
        };
        let more_blocked = RunnableCandidate {
            ready_at: more_blocked.decode_ready_at.clone().unwrap(),
            candidate: more_blocked,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &less_blocked,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &more_blocked,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn latency_prefers_owned_decode_cohort_with_more_ready_runway_when_fill_ties() {
        let mut more_ready = base_candidate(SchedulerPhase::Decode);
        more_ready.assignment_id = "more-ready".into();
        more_ready.group_status = "decode_leased".into();
        more_ready.decode_queue_status = Some("leased".into());
        more_ready.decode_ready_at = Some("2026-01-01T00:00:03Z".into());
        more_ready.group_lease_owner_device_id = Some("worker-1".into());
        more_ready.group_lease_expires_at = Some("2026-01-01T00:10:00Z".into());
        more_ready.decode_lease_target_session_count = Some(5);
        more_ready.decode_cohort_leased_sessions = 2;
        more_ready.decode_cohort_active_sessions = 1;
        more_ready.decode_cohort_ready_sessions = 3;
        more_ready.decode_cohort_blocked_sessions = 1;
        more_ready.decode_cohort_oldest_ready_at = Some("2026-01-01T00:00:01Z".into());

        let mut less_ready = more_ready.clone();
        less_ready.assignment_id = "less-ready".into();
        less_ready.decode_ready_at = Some("2026-01-01T00:00:01Z".into());
        less_ready.decode_cohort_ready_sessions = 1;

        let more_ready = RunnableCandidate {
            ready_at: more_ready.decode_ready_at.clone().unwrap(),
            candidate: more_ready,
        };
        let less_ready = RunnableCandidate {
            ready_at: less_ready.decode_ready_at.clone().unwrap(),
            candidate: less_ready,
        };

        let policy = InferenceSchedulingPolicy::default();
        let left = rank_candidate(
            &more_ready,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        let right = rank_candidate(
            &less_ready,
            SchedulerPolicyMode::LatencyFirst,
            &policy,
            8,
            8,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(left < right);
    }

    #[test]
    fn stale_decode_lease_is_recovered_into_ready_queue() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE inference_job_assignments (
                assignment_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                network_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_expires_at TEXT
            );
            CREATE TABLE inference_jobs (
                job_id TEXT PRIMARY KEY
            );
            CREATE TABLE inference_sessions (
                session_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL
            );
            CREATE TABLE inference_serving_groups (
                group_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_error TEXT
            );
            CREATE TABLE inference_decode_queue (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                lease_owner_device_id TEXT,
                lease_expires_at TEXT,
                updated_at TEXT NOT NULL,
                last_error TEXT
            );
            "#,
        )
        .unwrap();
        conn.execute("INSERT INTO inference_jobs (job_id) VALUES ('job-1')", [])
            .unwrap();
        conn.execute(
            "INSERT INTO inference_sessions (session_id, job_id) VALUES ('session-1', 'job-1')",
            [],
        )
        .unwrap();
        conn.execute(
            r#"
            INSERT INTO inference_job_assignments (
                assignment_id, job_id, network_id, device_id, status, lease_expires_at
            ) VALUES ('assignment-1', 'job-1', 'network-1', 'worker-1', 'leased', '2025-01-01T00:00:00Z')
            "#,
            [],
        )
        .unwrap();
        conn.execute(
            r#"
            INSERT INTO inference_serving_groups (
                group_id, job_id, device_id, phase, status, updated_at, last_error
            ) VALUES ('group-1', 'job-1', 'worker-1', 'decode', 'decode_leased', '2025-01-01T00:00:00Z', NULL)
            "#,
            [],
        )
        .unwrap();
        conn.execute(
            r#"
            INSERT INTO inference_decode_queue (
                session_id, status, lease_owner_device_id, lease_expires_at, updated_at, last_error
            ) VALUES ('session-1', 'leased', 'worker-1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z', NULL)
            "#,
            [],
        )
        .unwrap();

        let tx = conn.transaction().unwrap();
        reconcile_stale_scheduler_leases(&tx, "network-1", "2026-01-01T00:00:00Z").unwrap();
        tx.commit().unwrap();

        let assignment_status: String = conn
            .query_row(
                "SELECT status FROM inference_job_assignments WHERE assignment_id = 'assignment-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let group_status: String = conn
            .query_row(
                "SELECT status FROM inference_serving_groups WHERE group_id = 'group-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let queue_status: String = conn
            .query_row(
                "SELECT status FROM inference_decode_queue WHERE session_id = 'session-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(assignment_status, "pending");
        assert_eq!(group_status, "decode_ready");
        assert_eq!(queue_status, "ready");
    }
}
