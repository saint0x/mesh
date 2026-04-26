use std::collections::HashMap;

use rusqlite::{params, OptionalExtension, Transaction};

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{ClaimInferenceAssignmentRequest, InferenceExecutionPlan};
use crate::connectivity::{DeviceConnectivityState, InferenceSchedulingPolicy};
use crate::device::DeviceCapabilities;
use crate::services::planner::{
    device_metadata_from_capabilities, ExecutionPlanner, PlannerDeviceMetadata,
};
use crate::services::ring_manager::{ModelShard, RingTopology, WorkerTopologyInfo};

#[derive(Debug, Clone)]
struct ClaimCandidate {
    assignment_id: String,
    job_id: String,
    model_id: String,
    submitted_by_device_id: String,
    created_at: String,
    assigned_at: String,
}

pub fn select_claim_assignment_id(
    conn: &Transaction<'_>,
    req: &ClaimInferenceAssignmentRequest,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> ApiResult<Option<String>> {
    let candidates = load_claim_candidates(conn, req)?;
    if candidates.is_empty() {
        return Ok(None);
    }

    let active_jobs_by_submitter = load_active_job_counts_by_submitter(conn, &req.network_id)?;
    let active_jobs_by_model = load_active_job_counts_by_model(conn, &req.network_id)?;
    let active_capacity_by_model = load_active_capacity_by_model(conn, &req.network_id)?;
    let leased_assignments_by_submitter =
        load_leased_assignment_counts_by_submitter(conn, &req.network_id)?;
    let leased_assignments_by_job = load_leased_assignment_counts_by_job(conn, &req.network_id)?;
    let online_worker_count = load_online_ring_worker_count(conn, &req.network_id)?;
    let online_capacity_units =
        load_online_ring_capacity_units(conn, &req.network_id, scheduling_policy)?;

    let model_soft_cap =
        (online_worker_count / scheduling_policy.model_active_job_soft_cap_divisor).max(1);
    let capacity_soft_cap =
        (online_capacity_units / scheduling_policy.capacity_unit_soft_cap_divisor).max(1);

    let selected = candidates.into_iter().min_by(|left, right| {
        rank_candidate(
            left,
            scheduling_policy,
            model_soft_cap,
            capacity_soft_cap,
            &active_jobs_by_submitter,
            &active_jobs_by_model,
            &active_capacity_by_model,
            &leased_assignments_by_submitter,
            &leased_assignments_by_job,
        )
        .cmp(&rank_candidate(
            right,
            scheduling_policy,
            model_soft_cap,
            capacity_soft_cap,
            &active_jobs_by_submitter,
            &active_jobs_by_model,
            &active_capacity_by_model,
            &leased_assignments_by_submitter,
            &leased_assignments_by_job,
        ))
    });

    Ok(selected.map(|candidate| candidate.assignment_id))
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

fn rank_candidate(
    candidate: &ClaimCandidate,
    scheduling_policy: &InferenceSchedulingPolicy,
    model_soft_cap: u32,
    capacity_soft_cap: u32,
    active_jobs_by_submitter: &HashMap<String, u32>,
    active_jobs_by_model: &HashMap<String, u32>,
    active_capacity_by_model: &HashMap<String, u32>,
    leased_assignments_by_submitter: &HashMap<String, u32>,
    leased_assignments_by_job: &HashMap<String, u32>,
) -> (u8, u8, u8, u32, u32, String, String, String) {
    let submitter_active_jobs = active_jobs_by_submitter
        .get(&candidate.submitted_by_device_id)
        .copied()
        .unwrap_or_default();
    let model_active_jobs = active_jobs_by_model
        .get(&candidate.model_id)
        .copied()
        .unwrap_or_default();
    let model_active_capacity = active_capacity_by_model
        .get(&candidate.model_id)
        .copied()
        .unwrap_or_default();
    let submitter_leased_assignments = leased_assignments_by_submitter
        .get(&candidate.submitted_by_device_id)
        .copied()
        .unwrap_or_default();
    let job_leased_assignments = leased_assignments_by_job
        .get(&candidate.job_id)
        .copied()
        .unwrap_or_default();

    (
        u8::from(submitter_active_jobs >= scheduling_policy.submitter_active_job_soft_cap),
        u8::from(model_active_jobs >= model_soft_cap),
        u8::from(model_active_capacity >= capacity_soft_cap),
        submitter_leased_assignments,
        job_leased_assignments,
        candidate.created_at.clone(),
        candidate.assigned_at.clone(),
        candidate.assignment_id.clone(),
    )
}

fn load_claim_candidates(
    conn: &Transaction<'_>,
    req: &ClaimInferenceAssignmentRequest,
) -> ApiResult<Vec<ClaimCandidate>> {
    let now = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))?;
    let mut stmt = conn
        .prepare(
            r#"
            SELECT
                a.assignment_id,
                a.job_id,
                j.model_id,
                j.submitted_by_device_id,
                j.created_at,
                a.assigned_at
            FROM inference_job_assignments a
            INNER JOIN inference_jobs j ON j.job_id = a.job_id
            INNER JOIN inference_sessions s ON s.job_id = j.job_id
            INNER JOIN inference_session_replicas r
                ON r.session_id = s.session_id AND r.device_id = a.device_id
            LEFT JOIN inference_decode_queue dq
                ON dq.session_id = s.session_id
            WHERE a.device_id = ?
              AND a.network_id = ?
              AND a.active_segment_id = j.active_segment_id
              AND (
                    (
                        s.status IN ('prefill_pending', 'prefill_active')
                        AND r.status IN ('prefill_pending', 'prefill_active')
                    )
                    OR (
                        s.status IN ('decode_ready', 'decode_active')
                        AND r.status IN ('decode_ready', 'decode_active')
                        AND dq.status IN ('ready', 'leased', 'active')
                    )
                  )
              AND (
                    a.status = 'pending'
                    OR (a.status = 'leased' AND (a.lease_expires_at IS NULL OR a.lease_expires_at <= ?))
                  )
              AND j.status IN ('dispatched', 'running')
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let rows = stmt.query_map(params![&req.device_id, &req.network_id, &now], |row| {
        Ok(ClaimCandidate {
            assignment_id: row.get(0)?,
            job_id: row.get(1)?,
            model_id: row.get(2)?,
            submitted_by_device_id: row.get(3)?,
            created_at: row.get(4)?,
            assigned_at: row.get(5)?,
        })
    });
    let candidates = rows
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(candidates)
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
                    FROM inference_job_assignments a
                    WHERE a.job_id = j.job_id
                      AND a.status IN ('leased', 'acknowledged')
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
                    FROM inference_job_assignments a
                    WHERE a.job_id = j.job_id
                      AND a.status IN ('leased', 'acknowledged')
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
            WHERE j.network_id = ?
              AND j.status IN ('dispatched', 'running')
              AND a.status IN ('leased', 'acknowledged')
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
            WHERE j.network_id = ?
              AND a.status IN ('leased', 'acknowledged')
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
            WHERE a.network_id = ?
              AND a.status IN ('leased', 'acknowledged')
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
