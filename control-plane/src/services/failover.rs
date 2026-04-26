use std::collections::{BTreeSet, HashMap, HashSet};

use rusqlite::{params, OptionalExtension, Transaction};
use time::OffsetDateTime;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    ExecutionGroup, ExecutionGroupMember, ExecutionPhase, InferenceExecutionPlan,
    InferenceRuntimeMode,
};
use crate::db::Database;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceSessionRecord {
    pub session_id: String,
    pub job_id: String,
    pub network_id: String,
    pub model_id: String,
    pub status: String,
    pub active_segment_id: Option<String>,
    pub kv_owner_device_id: String,
    pub kv_transfer_policy: String,
    pub kv_sequence_position: Option<u32>,
    pub kv_checkpoint_device_id: Option<String>,
    pub kv_checkpoint_created_at: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionReplicaRecord {
    pub session_id: String,
    pub device_id: String,
    pub job_id: String,
    pub status: String,
    pub active_segment_id: Option<String>,
    pub kv_sequence_position: Option<u32>,
    pub checkpoint_created_at: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingGroupRecord {
    pub group_id: String,
    pub session_id: String,
    pub job_id: String,
    pub network_id: String,
    pub model_id: String,
    pub phase: ExecutionPhase,
    pub device_id: String,
    pub ring_position: u32,
    pub shard_column_start: u32,
    pub shard_column_end: u32,
    pub assigned_capacity_units: u32,
    pub execution_provider: String,
    pub status: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeQueueRecord {
    pub session_id: String,
    pub job_id: String,
    pub network_id: String,
    pub segment_id: String,
    pub group_id: String,
    pub status: String,
    pub ready_at: Option<String>,
    pub lease_owner_device_id: Option<String>,
    pub lease_expires_at: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailoverConfig {
    pub prefer_replacement: bool,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            prefer_replacement: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryState {
    Healthy,
    ResumeReady,
    Paused,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeMode {
    None,
    Immediate,
    RequiresCheckpointTransfer,
    ManualIntervention,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegroupStrategy {
    None,
    Replace {
        lost_device_ids: Vec<String>,
        replacement_device_ids: Vec<String>,
    },
    Shrink {
        removed_device_ids: Vec<String>,
    },
    Fail {
        lost_device_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceStatusUpdate {
    pub device_id: String,
    pub status: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PausePlan {
    pub required: bool,
    pub session_status: Option<String>,
    pub decode_queue_status: Option<String>,
    pub clear_lease: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumePlan {
    pub mode: ResumeMode,
    pub session_status: Option<String>,
    pub decode_queue_status: Option<String>,
    pub next_kv_owner_device_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailoverDecision {
    pub recovery_state: RecoveryState,
    pub strategy: RegroupStrategy,
    pub pause: PausePlan,
    pub resume: ResumePlan,
    pub target_group_id: Option<String>,
    pub target_segment_id: Option<String>,
    pub target_participant_device_ids: Vec<String>,
    pub replica_updates: Vec<DeviceStatusUpdate>,
    pub serving_group_updates: Vec<DeviceStatusUpdate>,
    pub reason: String,
}

pub struct FailoverEngine;

impl FailoverEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        plan: &InferenceExecutionPlan,
        session: &InferenceSessionRecord,
        replicas: &[SessionReplicaRecord],
        serving_groups: &[ServingGroupRecord],
        decode_queue: Option<&DecodeQueueRecord>,
        lost_device_ids: &[String],
        config: &FailoverConfig,
    ) -> ApiResult<FailoverDecision> {
        let active_segment_id = session.active_segment_id.as_ref().ok_or_else(|| {
            ApiError::Conflict(format!(
                "Session {} has no active segment for failover evaluation",
                session.session_id
            ))
        })?;
        let active_segment = plan
            .segments
            .iter()
            .find(|segment| &segment.segment_id == active_segment_id)
            .ok_or_else(|| {
                ApiError::Conflict(format!(
                    "Execution plan {} is missing active segment {}",
                    plan.plan_id, active_segment_id
                ))
            })?;
        let active_group = plan
            .execution_groups
            .iter()
            .find(|group| group.group_id == active_segment.execution_group_id)
            .ok_or_else(|| {
                ApiError::Conflict(format!(
                    "Execution plan {} is missing active group {}",
                    plan.plan_id, active_segment.execution_group_id
                ))
            })?;

        let lost_set = lost_device_ids.iter().cloned().collect::<HashSet<_>>();
        let active_participants = active_segment
            .participant_device_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let lost_active = active_participants
            .iter()
            .filter(|device_id| lost_set.contains(*device_id))
            .cloned()
            .collect::<Vec<_>>();

        if lost_active.is_empty() {
            return Ok(FailoverDecision {
                recovery_state: RecoveryState::Healthy,
                strategy: RegroupStrategy::None,
                pause: PausePlan {
                    required: false,
                    session_status: None,
                    decode_queue_status: None,
                    clear_lease: false,
                    reason: None,
                },
                resume: ResumePlan {
                    mode: ResumeMode::None,
                    session_status: None,
                    decode_queue_status: None,
                    next_kv_owner_device_id: None,
                },
                target_group_id: Some(active_group.group_id.clone()),
                target_segment_id: Some(active_segment.segment_id.clone()),
                target_participant_device_ids: active_segment.participant_device_ids.clone(),
                replica_updates: Vec::new(),
                serving_group_updates: Vec::new(),
                reason: "No active participant loss detected".to_string(),
            });
        }

        match active_segment.phase {
            ExecutionPhase::Prefill => Ok(Self::prefill_failure(
                session,
                replicas,
                serving_groups,
                &lost_active,
            )),
            ExecutionPhase::Decode => Self::decode_failover(
                active_group,
                active_segment,
                session,
                replicas,
                serving_groups,
                decode_queue,
                &lost_active,
                &lost_set,
                config,
            ),
        }
    }

    fn prefill_failure(
        session: &InferenceSessionRecord,
        replicas: &[SessionReplicaRecord],
        serving_groups: &[ServingGroupRecord],
        lost_active: &[String],
    ) -> FailoverDecision {
        let reason = format!(
            "Prefill participant loss ({}) cannot safely regroup without replaying the session",
            lost_active.join(", ")
        );

        FailoverDecision {
            recovery_state: RecoveryState::Failed,
            strategy: RegroupStrategy::Fail {
                lost_device_ids: lost_active.to_vec(),
            },
            pause: PausePlan {
                required: true,
                session_status: Some("failed".to_string()),
                decode_queue_status: Some("failed".to_string()),
                clear_lease: true,
                reason: Some(reason.clone()),
            },
            resume: ResumePlan {
                mode: ResumeMode::ManualIntervention,
                session_status: Some("failed".to_string()),
                decode_queue_status: Some("failed".to_string()),
                next_kv_owner_device_id: Some(session.kv_owner_device_id.clone()),
            },
            target_group_id: None,
            target_segment_id: session.active_segment_id.clone(),
            target_participant_device_ids: Vec::new(),
            replica_updates: lost_active
                .iter()
                .filter(|device_id| {
                    replicas
                        .iter()
                        .any(|replica| &replica.device_id == *device_id)
                })
                .map(|device_id| DeviceStatusUpdate {
                    device_id: device_id.clone(),
                    status: "failed".to_string(),
                    last_error: Some(reason.clone()),
                })
                .collect(),
            serving_group_updates: lost_active
                .iter()
                .filter(|device_id| {
                    serving_groups
                        .iter()
                        .any(|member| &member.device_id == *device_id)
                })
                .map(|device_id| DeviceStatusUpdate {
                    device_id: device_id.clone(),
                    status: "failed".to_string(),
                    last_error: Some(reason.clone()),
                })
                .collect(),
            reason,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_failover(
        active_group: &ExecutionGroup,
        active_segment: &crate::api::types::ExecutionSegment,
        session: &InferenceSessionRecord,
        replicas: &[SessionReplicaRecord],
        serving_groups: &[ServingGroupRecord],
        decode_queue: Option<&DecodeQueueRecord>,
        lost_active: &[String],
        lost_set: &HashSet<String>,
        config: &FailoverConfig,
    ) -> ApiResult<FailoverDecision> {
        let member_records = serving_groups
            .iter()
            .filter(|record| {
                record.group_id == active_group.group_id
                    && record.session_id == active_segment.session_id
                    && matches!(record.phase, ExecutionPhase::Decode)
            })
            .collect::<Vec<_>>();
        let member_record_map = member_records
            .iter()
            .map(|record| (record.device_id.as_str(), *record))
            .collect::<HashMap<_, _>>();
        let member_map = active_group
            .members
            .iter()
            .map(|member| (member.device_id.as_str(), member))
            .collect::<HashMap<_, _>>();
        let replica_map = replicas
            .iter()
            .map(|replica| (replica.device_id.as_str(), replica))
            .collect::<HashMap<_, _>>();

        let active_participants = active_segment
            .participant_device_ids
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let survivor_ids = active_participants
            .iter()
            .filter(|device_id| !lost_set.contains(*device_id))
            .cloned()
            .collect::<Vec<_>>();

        let required_span = coverage_target(&active_participants, &member_map)?;
        let active_count = active_participants.len();
        let available_members = active_group
            .members
            .iter()
            .filter(|member| !lost_set.contains(&member.device_id))
            .filter(|member| {
                member_record_map
                    .get(member.device_id.as_str())
                    .map(|record| record.status != "superseded" && record.status != "failed")
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();

        let survivor_only_target = find_covering_set(
            &available_members,
            &survivor_ids,
            required_span,
            Some(survivor_ids.len()),
        );
        let mut target = survivor_only_target
            .or_else(|| {
                if config.prefer_replacement {
                    find_covering_set(
                        &available_members,
                        &survivor_ids,
                        required_span,
                        Some(active_count),
                    )
                } else {
                    None
                }
            })
            .or_else(|| find_covering_set(&available_members, &survivor_ids, required_span, None));

        let Some(target_participants) = target.take() else {
            return Ok(Self::failed_decode_decision(
                session,
                replicas,
                member_records.as_slice(),
                lost_active,
                "Remaining decode members can no longer cover the active shard span",
            ));
        };

        let introduced = target_participants
            .iter()
            .filter(|device_id| !active_participants.contains(device_id))
            .cloned()
            .collect::<Vec<_>>();
        let next_kv_owner_device_id =
            select_next_kv_owner(&target_participants, session, &replica_map, &member_map)?;

        let requires_transfer = if introduced.is_empty() {
            false
        } else if session.kv_transfer_policy == "remote_access"
            && active_participants.contains(&session.kv_owner_device_id)
            && !lost_set.contains(&session.kv_owner_device_id)
        {
            false
        } else {
            true
        };

        if requires_transfer && !checkpoint_available(session, replicas, &target_participants) {
            return Ok(Self::failed_decode_decision(
                session,
                replicas,
                member_records.as_slice(),
                lost_active,
                "Decode regroup requires checkpoint transfer, but no checkpoint is available",
            ));
        }

        let strategy = if !introduced.is_empty() {
            RegroupStrategy::Replace {
                lost_device_ids: lost_active.to_vec(),
                replacement_device_ids: introduced.clone(),
            }
        } else {
            RegroupStrategy::Shrink {
                removed_device_ids: lost_active.to_vec(),
            }
        };
        let pause_required = session.status == "decode_active"
            || decode_queue
                .map(|queue| matches!(queue.status.as_str(), "leased" | "active"))
                .unwrap_or(false);
        let replica_updates = build_replica_updates(
            replicas,
            &active_participants,
            &target_participants,
            lost_active,
            requires_transfer,
        );
        let serving_group_updates = build_group_updates(
            member_records.as_slice(),
            &active_participants,
            &target_participants,
            lost_active,
            requires_transfer,
        );
        let pause_reason = if requires_transfer {
            Some("Pause decode and wait for checkpoint transfer to regroup members".to_string())
        } else if pause_required {
            Some("Pause decode, regroup survivors, and resume immediately".to_string())
        } else {
            None
        };
        let reason = if requires_transfer {
            format!(
                "Decode regroup replaces lost participants ({}) and pauses until checkpoint transfer completes",
                lost_active.join(", ")
            )
        } else if introduced.is_empty() {
            format!(
                "Decode can shrink to surviving participants after losing {}",
                lost_active.join(", ")
            )
        } else {
            format!(
                "Decode can replace lost participants ({}) without blocking on transfer",
                lost_active.join(", ")
            )
        };

        Ok(FailoverDecision {
            recovery_state: if requires_transfer {
                RecoveryState::Paused
            } else {
                RecoveryState::ResumeReady
            },
            strategy,
            pause: PausePlan {
                required: pause_required || requires_transfer,
                session_status: Some("decode_ready".to_string()),
                decode_queue_status: Some(if requires_transfer {
                    "blocked_on_transfer".to_string()
                } else {
                    "ready".to_string()
                }),
                clear_lease: pause_required || requires_transfer,
                reason: pause_reason,
            },
            resume: ResumePlan {
                mode: if requires_transfer {
                    ResumeMode::RequiresCheckpointTransfer
                } else {
                    ResumeMode::Immediate
                },
                session_status: Some("decode_ready".to_string()),
                decode_queue_status: Some("ready".to_string()),
                next_kv_owner_device_id: Some(next_kv_owner_device_id),
            },
            target_group_id: Some(active_group.group_id.clone()),
            target_segment_id: Some(active_segment.segment_id.clone()),
            target_participant_device_ids: target_participants,
            replica_updates,
            serving_group_updates,
            reason,
        })
    }

    fn failed_decode_decision(
        session: &InferenceSessionRecord,
        replicas: &[SessionReplicaRecord],
        serving_groups: &[&ServingGroupRecord],
        lost_active: &[String],
        detail: &str,
    ) -> FailoverDecision {
        let reason = format!("{}: {}", detail, lost_active.join(", "));
        FailoverDecision {
            recovery_state: RecoveryState::Failed,
            strategy: RegroupStrategy::Fail {
                lost_device_ids: lost_active.to_vec(),
            },
            pause: PausePlan {
                required: true,
                session_status: Some("failed".to_string()),
                decode_queue_status: Some("failed".to_string()),
                clear_lease: true,
                reason: Some(reason.clone()),
            },
            resume: ResumePlan {
                mode: ResumeMode::ManualIntervention,
                session_status: Some("failed".to_string()),
                decode_queue_status: Some("failed".to_string()),
                next_kv_owner_device_id: Some(session.kv_owner_device_id.clone()),
            },
            target_group_id: None,
            target_segment_id: session.active_segment_id.clone(),
            target_participant_device_ids: Vec::new(),
            replica_updates: lost_active
                .iter()
                .filter(|device_id| {
                    replicas
                        .iter()
                        .any(|replica| &replica.device_id == *device_id)
                })
                .map(|device_id| DeviceStatusUpdate {
                    device_id: device_id.clone(),
                    status: "failed".to_string(),
                    last_error: Some(reason.clone()),
                })
                .collect(),
            serving_group_updates: lost_active
                .iter()
                .filter(|device_id| {
                    serving_groups
                        .iter()
                        .any(|group| &group.device_id == *device_id)
                })
                .map(|device_id| DeviceStatusUpdate {
                    device_id: device_id.clone(),
                    status: "failed".to_string(),
                    last_error: Some(reason.clone()),
                })
                .collect(),
            reason,
        }
    }
}

#[derive(Debug, Clone)]
struct FailoverSessionContext {
    plan: InferenceExecutionPlan,
    session: InferenceSessionRecord,
    replicas: Vec<SessionReplicaRecord>,
    serving_groups: Vec<ServingGroupRecord>,
    decode_queue: Option<DecodeQueueRecord>,
    batch_group_key: Option<String>,
}

pub fn reconcile_failover_state(db: &Database, lost_device_ids: &[String]) -> ApiResult<usize> {
    let mut conn = db.get_conn().map_err(|e| ApiError::Database(Box::new(e)))?;
    let tx = conn
        .transaction()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let now = now_rfc3339()?;
    let mut changed_sessions = 0usize;

    for context in load_impacted_sessions(&tx, lost_device_ids)? {
        let config = failover_config_for_runtime_mode(context.plan.runtime_mode);
        let decision = FailoverEngine::evaluate(
            &context.plan,
            &context.session,
            &context.replicas,
            &context.serving_groups,
            context.decode_queue.as_ref(),
            lost_device_ids,
            &config,
        )?;
        if matches!(decision.recovery_state, RecoveryState::Healthy) {
            continue;
        }
        apply_failover_decision(&tx, &context, &decision, &now)?;
        changed_sessions += 1;
    }

    cleanup_terminal_session_state(&tx, &now)?;
    cleanup_orphaned_leases(&tx, &now)?;

    tx.commit()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(changed_sessions)
}

fn build_replica_updates(
    replicas: &[SessionReplicaRecord],
    active_participants: &[String],
    target_participants: &[String],
    lost_active: &[String],
    requires_transfer: bool,
) -> Vec<DeviceStatusUpdate> {
    let target_set = target_participants.iter().collect::<HashSet<_>>();
    let active_set = active_participants.iter().collect::<HashSet<_>>();
    let lost_set = lost_active.iter().collect::<HashSet<_>>();

    replicas
        .iter()
        .filter_map(|replica| {
            if lost_set.contains(&replica.device_id) {
                Some(DeviceStatusUpdate {
                    device_id: replica.device_id.clone(),
                    status: "failed".to_string(),
                    last_error: Some("participant_lost".to_string()),
                })
            } else if target_set.contains(&replica.device_id) {
                Some(DeviceStatusUpdate {
                    device_id: replica.device_id.clone(),
                    status: if active_set.contains(&replica.device_id) || !requires_transfer {
                        "decode_ready".to_string()
                    } else {
                        "decode_pending_transfer".to_string()
                    },
                    last_error: None,
                })
            } else {
                None
            }
        })
        .collect()
}

fn load_impacted_sessions(
    conn: &Transaction<'_>,
    lost_device_ids: &[String],
) -> ApiResult<Vec<FailoverSessionContext>> {
    if lost_device_ids.is_empty() {
        return Ok(Vec::new());
    }

    let placeholders = std::iter::repeat_n("?", lost_device_ids.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        r#"
        SELECT DISTINCT s.session_id
        FROM inference_sessions s
        INNER JOIN inference_jobs j ON j.job_id = s.job_id
        INNER JOIN inference_session_replicas r ON r.session_id = s.session_id
        WHERE r.device_id IN ({})
          AND s.active_segment_id IS NOT NULL
          AND s.status NOT IN ('completed', 'failed', 'cancelled')
          AND j.status IN ('dispatched', 'running')
        "#,
        placeholders
    );
    let params = lost_device_ids
        .iter()
        .map(|value| value as &dyn rusqlite::ToSql)
        .collect::<Vec<_>>();
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let session_ids = stmt
        .query_map(rusqlite::params_from_iter(params), |row| {
            row.get::<_, String>(0)
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    session_ids
        .into_iter()
        .map(|session_id| load_session_context(conn, &session_id))
        .collect()
}

fn load_session_context(
    conn: &Transaction<'_>,
    session_id: &str,
) -> ApiResult<FailoverSessionContext> {
    let (execution_plan_json, session) = conn
        .query_row(
            r#"
            SELECT j.execution_plan_json,
                   s.session_id, s.job_id, s.network_id, s.model_id, s.status,
                   s.active_segment_id, s.kv_owner_device_id, s.kv_transfer_policy,
                   s.kv_sequence_position, s.kv_checkpoint_device_id,
                   s.kv_checkpoint_created_at, s.last_error
            FROM inference_sessions s
            INNER JOIN inference_jobs j ON j.job_id = s.job_id
            WHERE s.session_id = ?
            "#,
            params![session_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    InferenceSessionRecord {
                        session_id: row.get(1)?,
                        job_id: row.get(2)?,
                        network_id: row.get(3)?,
                        model_id: row.get(4)?,
                        status: row.get(5)?,
                        active_segment_id: row.get(6)?,
                        kv_owner_device_id: row.get(7)?,
                        kv_transfer_policy: row.get(8)?,
                        kv_sequence_position: row
                            .get::<_, Option<i64>>(9)?
                            .map(|value| value as u32),
                        kv_checkpoint_device_id: row.get(10)?,
                        kv_checkpoint_created_at: row.get(11)?,
                        last_error: row.get(12)?,
                    },
                ))
            },
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    let plan = serde_json::from_str::<InferenceExecutionPlan>(&execution_plan_json)
        .map_err(|e| ApiError::Internal(format!("Invalid execution plan json: {}", e)))?;

    let replicas = {
        let mut stmt = conn
            .prepare(
                r#"
                SELECT session_id, device_id, job_id, status, active_segment_id,
                       kv_sequence_position, checkpoint_created_at, last_error
                FROM inference_session_replicas
                WHERE session_id = ?
                "#,
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        let rows = stmt.query_map(params![session_id], |row| {
            Ok(SessionReplicaRecord {
                session_id: row.get(0)?,
                device_id: row.get(1)?,
                job_id: row.get(2)?,
                status: row.get(3)?,
                active_segment_id: row.get(4)?,
                kv_sequence_position: row.get::<_, Option<i64>>(5)?.map(|value| value as u32),
                checkpoint_created_at: row.get(6)?,
                last_error: row.get(7)?,
            })
        });
        let collected = rows
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        collected
    };

    let serving_groups = {
        let mut stmt = conn
            .prepare(
                r#"
                SELECT group_id, session_id, job_id, network_id, model_id, phase, device_id,
                       ring_position, shard_column_start, shard_column_end,
                       assigned_capacity_units, execution_provider, status, last_error
                FROM inference_serving_groups
                WHERE session_id = ?
                "#,
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        let rows = stmt.query_map(params![session_id], |row| {
            let phase: String = row.get(5)?;
            Ok(ServingGroupRecord {
                group_id: row.get(0)?,
                session_id: row.get(1)?,
                job_id: row.get(2)?,
                network_id: row.get(3)?,
                model_id: row.get(4)?,
                phase: parse_phase(&phase)?,
                device_id: row.get(6)?,
                ring_position: row.get::<_, i64>(7)? as u32,
                shard_column_start: row.get::<_, i64>(8)? as u32,
                shard_column_end: row.get::<_, i64>(9)? as u32,
                assigned_capacity_units: row.get::<_, i64>(10)? as u32,
                execution_provider: row.get(11)?,
                status: row.get(12)?,
                last_error: row.get(13)?,
            })
        });
        let collected = rows
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        collected
    };

    let (decode_queue, batch_group_key) = conn
        .query_row(
            r#"
            SELECT session_id, job_id, network_id, segment_id, group_id, status,
                   ready_at, lease_owner_device_id, lease_expires_at, last_error,
                   batch_group_key
            FROM inference_decode_queue
            WHERE session_id = ?
            "#,
            params![session_id],
            |row| {
                Ok((
                    DecodeQueueRecord {
                        session_id: row.get(0)?,
                        job_id: row.get(1)?,
                        network_id: row.get(2)?,
                        segment_id: row.get(3)?,
                        group_id: row.get(4)?,
                        status: row.get(5)?,
                        ready_at: row.get(6)?,
                        lease_owner_device_id: row.get(7)?,
                        lease_expires_at: row.get(8)?,
                        last_error: row.get(9)?,
                    },
                    row.get::<_, Option<String>>(10)?,
                ))
            },
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?
        .map(|(queue, key)| (Some(queue), key))
        .unwrap_or((None, None));

    Ok(FailoverSessionContext {
        plan,
        session,
        replicas,
        serving_groups,
        decode_queue,
        batch_group_key,
    })
}

fn apply_failover_decision(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    decision: &FailoverDecision,
    now: &str,
) -> ApiResult<()> {
    let mut plan = context.plan.clone();
    update_plan_for_failover(&mut plan, decision)?;
    let execution_plan_json = serde_json::to_string(&plan)
        .map_err(|e| ApiError::Internal(format!("Failed to serialize regrouped plan: {}", e)))?;

    if matches!(decision.recovery_state, RecoveryState::Failed) {
        fail_session_job(tx, context, decision, &execution_plan_json, now)?;
        return Ok(());
    }

    let target_segment_id = decision
        .target_segment_id
        .as_deref()
        .or(context.session.active_segment_id.as_deref())
        .ok_or_else(|| ApiError::Conflict("Missing active segment for regroup".to_string()))?;
    let target_group_id = decision
        .target_group_id
        .as_deref()
        .ok_or_else(|| ApiError::Conflict("Missing target group for regroup".to_string()))?;
    let target_group = plan
        .execution_groups
        .iter()
        .find(|group| group.group_id == target_group_id)
        .ok_or_else(|| {
            ApiError::Conflict(format!(
                "Execution group {} missing after regroup",
                target_group_id
            ))
        })?;

    tx.execute(
        r#"
        UPDATE inference_jobs
        SET execution_plan_json = ?,
            active_segment_id = ?,
            updated_at = ?,
            error = NULL
        WHERE job_id = ?
        "#,
        params![
            execution_plan_json,
            target_segment_id,
            now,
            &context.session.job_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let next_kv_owner = decision
        .resume
        .next_kv_owner_device_id
        .as_deref()
        .unwrap_or(&context.session.kv_owner_device_id);
    tx.execute(
        r#"
        UPDATE inference_sessions
        SET status = ?,
            active_segment_id = ?,
            kv_owner_device_id = ?,
            updated_at = ?,
            last_error = ?
        WHERE session_id = ?
        "#,
        params![
            decision
                .resume
                .session_status
                .as_deref()
                .or(decision.pause.session_status.as_deref())
                .unwrap_or("decode_ready"),
            target_segment_id,
            next_kv_owner,
            now,
            decision.pause.reason.as_deref(),
            &context.session.session_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    apply_assignment_regroup_updates(
        tx,
        context,
        target_group,
        target_segment_id,
        &decision.target_participant_device_ids,
        now,
    )?;
    apply_replica_regroup_updates(tx, context, decision, target_segment_id, now)?;
    apply_serving_group_regroup_updates(tx, context, decision, now)?;
    upsert_decode_queue_regroup_state(
        tx,
        context,
        decision,
        target_segment_id,
        target_group_id,
        now,
    )?;
    record_regroup_scheduler_events(tx, context, decision, now)?;

    Ok(())
}

fn update_plan_for_failover(
    plan: &mut InferenceExecutionPlan,
    decision: &FailoverDecision,
) -> ApiResult<()> {
    let Some(segment_id) = decision.target_segment_id.as_deref() else {
        return Ok(());
    };
    let Some(segment) = plan
        .segments
        .iter_mut()
        .find(|segment| segment.segment_id == segment_id)
    else {
        return Err(ApiError::Conflict(format!(
            "Execution plan {} missing regroup target segment {}",
            plan.plan_id, segment_id
        )));
    };

    segment.participant_device_ids = decision.target_participant_device_ids.clone();
    segment.shard_owner_device_ids = decision.target_participant_device_ids.clone();
    if let Some(next_kv_owner) = &decision.resume.next_kv_owner_device_id {
        segment.kv_owner_device_id = next_kv_owner.clone();
    }
    Ok(())
}

fn fail_session_job(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    decision: &FailoverDecision,
    execution_plan_json: &str,
    now: &str,
) -> ApiResult<()> {
    tx.execute(
        r#"
        UPDATE inference_jobs
        SET status = 'failed',
            active_segment_id = NULL,
            updated_at = ?,
            completed_at = COALESCE(completed_at, ?),
            error = ?
        WHERE job_id = ?
        "#,
        params![now, now, &decision.reason, &context.session.job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_jobs
        SET execution_plan_json = ?
        WHERE job_id = ?
        "#,
        params![execution_plan_json, &context.session.job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_sessions
        SET status = 'failed',
            active_segment_id = NULL,
            updated_at = ?,
            last_error = ?
        WHERE session_id = ?
        "#,
        params![now, &decision.reason, &context.session.session_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_job_assignments
        SET status = CASE
                WHEN status IN ('completed', 'cancelled') THEN status
                ELSE 'failed'
            END,
            active_segment_id = NULL,
            lease_expires_at = NULL,
            failure_reason = COALESCE(failure_reason, ?)
        WHERE job_id = ?
        "#,
        params![&decision.reason, &context.session.job_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_session_replicas
        SET status = CASE
                WHEN status = 'completed' THEN status
                ELSE 'failed'
            END,
            active_segment_id = NULL,
            updated_at = ?,
            last_error = COALESCE(last_error, ?)
        WHERE session_id = ?
        "#,
        params![now, &decision.reason, &context.session.session_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_serving_groups
        SET status = CASE
                WHEN status = 'completed' THEN status
                ELSE 'failed'
            END,
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            updated_at = ?,
            last_error = COALESCE(last_error, ?)
        WHERE session_id = ?
        "#,
        params![now, &decision.reason, &context.session.session_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_decode_queue
        SET status = 'failed',
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            blocked_reason = 'participant_lost',
            last_error = ?,
            updated_at = ?
        WHERE session_id = ?
        "#,
        params![&decision.reason, now, &context.session.session_id],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    record_regroup_scheduler_events(tx, context, decision, now)?;
    Ok(())
}

fn apply_assignment_regroup_updates(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    target_group: &ExecutionGroup,
    target_segment_id: &str,
    target_participants: &[String],
    now: &str,
) -> ApiResult<()> {
    let target_set = target_participants.iter().collect::<HashSet<_>>();
    for member in &target_group.members {
        let exists: Option<String> = tx
            .query_row(
                r#"
                SELECT assignment_id
                FROM inference_job_assignments
                WHERE job_id = ? AND device_id = ?
                "#,
                params![&context.session.job_id, &member.device_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        if exists.is_none() {
            tx.execute(
                r#"
                INSERT INTO inference_job_assignments (
                    assignment_id, job_id, network_id, device_id, ring_position, status,
                    assigned_at, shard_column_start, shard_column_end,
                    assigned_capacity_units, execution_provider, active_segment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
                params![
                    uuid::Uuid::new_v4().to_string(),
                    &context.session.job_id,
                    &context.session.network_id,
                    &member.device_id,
                    i64::from(member.ring_position),
                    if target_set.contains(&member.device_id) {
                        "pending"
                    } else {
                        "waiting"
                    },
                    now,
                    i64::from(member.shard.column_start),
                    i64::from(member.shard.column_end),
                    i64::from(member.assigned_capacity_units),
                    &member.execution_provider,
                    if target_set.contains(&member.device_id) {
                        Some(target_segment_id)
                    } else {
                        None
                    }
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }
    }

    let placeholders = std::iter::repeat_n("?", target_participants.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        r#"
        UPDATE inference_job_assignments
        SET status = 'pending',
            active_segment_id = ?,
            acknowledged_at = NULL,
            lease_expires_at = NULL,
            failure_reason = NULL
        WHERE job_id = ?
          AND device_id IN ({})
          AND status NOT IN ('completed', 'failed', 'cancelled')
        "#,
        placeholders
    );
    let mut values: Vec<&dyn rusqlite::ToSql> = vec![&target_segment_id, &context.session.job_id];
    for device_id in target_participants {
        values.push(device_id);
    }
    tx.execute(&sql, rusqlite::params_from_iter(values))
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    tx.execute(
        r#"
        UPDATE inference_job_assignments
        SET status = CASE
                WHEN status = 'completed' THEN status
                WHEN status = 'failed' THEN status
                ELSE 'waiting'
            END,
            active_segment_id = CASE
                WHEN status IN ('completed', 'failed', 'cancelled') THEN active_segment_id
                ELSE NULL
            END,
            acknowledged_at = NULL,
            lease_expires_at = NULL
        WHERE job_id = ?
          AND active_segment_id = ?
          AND device_id NOT IN (
              SELECT value FROM json_each(?)
          )
        "#,
        params![
            &context.session.job_id,
            target_segment_id,
            serde_json::to_string(target_participants).map_err(|e| {
                ApiError::Internal(format!(
                    "Failed to serialize regroup participant set: {}",
                    e
                ))
            })?
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn apply_replica_regroup_updates(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    decision: &FailoverDecision,
    target_segment_id: &str,
    now: &str,
) -> ApiResult<()> {
    for update in &decision.replica_updates {
        let exists: Option<String> = tx
            .query_row(
                "SELECT device_id FROM inference_session_replicas WHERE session_id = ? AND device_id = ?",
                params![&context.session.session_id, &update.device_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        if exists.is_none() {
            tx.execute(
                r#"
                INSERT INTO inference_session_replicas (
                    session_id, device_id, job_id, status, active_segment_id,
                    kv_sequence_position, checkpoint_created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                "#,
                params![
                    &context.session.session_id,
                    &update.device_id,
                    &context.session.job_id,
                    &update.status,
                    if update.status == "failed" {
                        None::<&str>
                    } else {
                        Some(target_segment_id)
                    },
                    now,
                    update.last_error.as_deref()
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        } else {
            tx.execute(
                r#"
                UPDATE inference_session_replicas
                SET status = ?,
                    active_segment_id = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE session_id = ? AND device_id = ?
                "#,
                params![
                    &update.status,
                    if update.status == "failed" {
                        None::<&str>
                    } else {
                        Some(target_segment_id)
                    },
                    now,
                    update.last_error.as_deref(),
                    &context.session.session_id,
                    &update.device_id
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }
    }
    Ok(())
}

fn apply_serving_group_regroup_updates(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    decision: &FailoverDecision,
    now: &str,
) -> ApiResult<()> {
    for update in &decision.serving_group_updates {
        tx.execute(
            r#"
            UPDATE inference_serving_groups
            SET status = ?,
                lease_owner_device_id = NULL,
                lease_expires_at = NULL,
                updated_at = ?,
                last_error = ?
            WHERE session_id = ? AND device_id = ? AND phase = 'decode'
            "#,
            params![
                &update.status,
                now,
                update.last_error.as_deref(),
                &context.session.session_id,
                &update.device_id
            ],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    }
    Ok(())
}

fn upsert_decode_queue_regroup_state(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    decision: &FailoverDecision,
    target_segment_id: &str,
    target_group_id: &str,
    now: &str,
) -> ApiResult<()> {
    let queue_status = decision
        .pause
        .decode_queue_status
        .as_deref()
        .or(decision.resume.decode_queue_status.as_deref())
        .unwrap_or("ready");
    tx.execute(
        r#"
        UPDATE inference_decode_queue
        SET segment_id = ?,
            group_id = ?,
            status = ?,
            ready_at = CASE
                WHEN ? = 'ready' THEN COALESCE(ready_at, ?)
                ELSE ready_at
            END,
            blocked_reason = CASE
                WHEN ? = 'blocked_on_transfer' THEN 'participant_lost'
                ELSE NULL
            END,
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            last_error = ?,
            updated_at = ?
        WHERE session_id = ?
        "#,
        params![
            target_segment_id,
            target_group_id,
            queue_status,
            queue_status,
            now,
            queue_status,
            decision.pause.reason.as_deref(),
            now,
            &context.session.session_id
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn record_regroup_scheduler_events(
    tx: &Transaction<'_>,
    context: &FailoverSessionContext,
    decision: &FailoverDecision,
    now: &str,
) -> ApiResult<()> {
    let event_kind = match (&decision.strategy, &decision.resume.mode) {
        (RegroupStrategy::Replace { .. }, ResumeMode::RequiresCheckpointTransfer) => {
            "decode_regroup_transfer"
        }
        (RegroupStrategy::Replace { .. }, _) => "decode_regroup_replace",
        (RegroupStrategy::Shrink { .. }, _) => "decode_regroup_shrink",
        (RegroupStrategy::Fail { .. }, _) => "decode_regroup_failed",
        _ => "decode_regroup_unchanged",
    };
    tx.execute(
        r#"
        INSERT INTO inference_scheduler_events (
            network_id, job_id, session_id, device_id, segment_id, group_id, batch_group_key,
            event_kind, queue_status, detail, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
        params![
            &context.session.network_id,
            &context.session.job_id,
            &context.session.session_id,
            decision.resume.next_kv_owner_device_id.as_deref(),
            decision.target_segment_id.as_deref(),
            decision.target_group_id.as_deref(),
            context.batch_group_key.as_deref(),
            event_kind,
            decision
                .pause
                .decode_queue_status
                .as_deref()
                .or(decision.resume.decode_queue_status.as_deref()),
            Some(decision.reason.as_str()),
            now
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn cleanup_terminal_session_state(tx: &Transaction<'_>, now: &str) -> ApiResult<()> {
    tx.execute(
        r#"
        UPDATE inference_sessions
        SET status = (
                SELECT status FROM inference_jobs j WHERE j.job_id = inference_sessions.job_id
            ),
            active_segment_id = NULL,
            updated_at = ?
        WHERE status IN ('completed', 'failed', 'cancelled')
          AND active_segment_id IS NOT NULL
        "#,
        params![now],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_job_assignments
        SET status = CASE
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_job_assignments.job_id
                ) = 'completed' THEN 'completed'
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_job_assignments.job_id
                ) = 'cancelled' THEN 'cancelled'
                ELSE 'failed'
            END,
            active_segment_id = NULL,
            lease_expires_at = NULL
        WHERE job_id IN (
            SELECT job_id FROM inference_jobs WHERE status IN ('completed', 'failed', 'cancelled')
        )
          AND (active_segment_id IS NOT NULL OR lease_expires_at IS NOT NULL)
        "#,
        [],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_session_replicas
        SET status = CASE
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_session_replicas.job_id
                ) = 'completed' THEN 'completed'
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_session_replicas.job_id
                ) = 'cancelled' THEN 'cancelled'
                ELSE 'failed'
            END,
            active_segment_id = NULL,
            updated_at = ?
        WHERE job_id IN (
            SELECT job_id FROM inference_jobs WHERE status IN ('completed', 'failed', 'cancelled')
        )
          AND active_segment_id IS NOT NULL
        "#,
        params![now],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_serving_groups
        SET status = CASE
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_serving_groups.job_id
                ) = 'completed' THEN 'completed'
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_serving_groups.job_id
                ) = 'cancelled' THEN 'cancelled'
                ELSE 'failed'
            END,
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            updated_at = ?
        WHERE job_id IN (
            SELECT job_id FROM inference_jobs WHERE status IN ('completed', 'failed', 'cancelled')
        )
          AND (lease_owner_device_id IS NOT NULL OR lease_expires_at IS NOT NULL)
        "#,
        params![now],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_decode_queue
        SET status = CASE
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_decode_queue.job_id
                ) = 'completed' THEN 'completed'
                WHEN (
                    SELECT status FROM inference_jobs j WHERE j.job_id = inference_decode_queue.job_id
                ) = 'cancelled' THEN 'cancelled'
                ELSE 'failed'
            END,
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            updated_at = ?
        WHERE job_id IN (
            SELECT job_id FROM inference_jobs WHERE status IN ('completed', 'failed', 'cancelled')
        )
          AND (lease_owner_device_id IS NOT NULL OR lease_expires_at IS NOT NULL)
        "#,
        params![now],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn cleanup_orphaned_leases(tx: &Transaction<'_>, now: &str) -> ApiResult<()> {
    tx.execute(
        r#"
        UPDATE inference_job_assignments
        SET status = 'pending',
            lease_expires_at = NULL
        WHERE status = 'leased'
          AND device_id IN (SELECT device_id FROM devices WHERE status = 'offline')
          AND job_id IN (SELECT job_id FROM inference_jobs WHERE status IN ('dispatched', 'running'))
        "#,
        [],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_decode_queue
        SET status = CASE
                WHEN status = 'leased' THEN 'ready'
                ELSE status
            END,
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            updated_at = ?
        WHERE lease_owner_device_id IN (SELECT device_id FROM devices WHERE status = 'offline')
        "#,
        params![now],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    tx.execute(
        r#"
        UPDATE inference_serving_groups
        SET status = CASE
                WHEN status = 'decode_leased' THEN 'decode_ready'
                WHEN status = 'prefill_leased' THEN 'prefill_member'
                ELSE status
            END,
            lease_owner_device_id = NULL,
            lease_expires_at = NULL,
            updated_at = ?
        WHERE lease_owner_device_id IN (SELECT device_id FROM devices WHERE status = 'offline')
        "#,
        params![now],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
    Ok(())
}

fn parse_phase(value: &str) -> Result<ExecutionPhase, rusqlite::Error> {
    match value {
        "prefill" => Ok(ExecutionPhase::Prefill),
        "decode" => Ok(ExecutionPhase::Decode),
        other => Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid phase {}", other),
            )),
        )),
    }
}

fn failover_config_for_runtime_mode(mode: InferenceRuntimeMode) -> FailoverConfig {
    FailoverConfig {
        prefer_replacement: matches!(
            mode,
            InferenceRuntimeMode::ThroughputFirst | InferenceRuntimeMode::LatencyFirst
        ),
    }
}

fn now_rfc3339() -> ApiResult<String> {
    OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))
}

fn build_group_updates(
    serving_groups: &[&ServingGroupRecord],
    active_participants: &[String],
    target_participants: &[String],
    lost_active: &[String],
    requires_transfer: bool,
) -> Vec<DeviceStatusUpdate> {
    let target_set = target_participants.iter().collect::<HashSet<_>>();
    let active_set = active_participants.iter().collect::<HashSet<_>>();
    let lost_set = lost_active.iter().collect::<HashSet<_>>();

    serving_groups
        .iter()
        .filter_map(|member| {
            if lost_set.contains(&member.device_id) {
                Some(DeviceStatusUpdate {
                    device_id: member.device_id.clone(),
                    status: "failed".to_string(),
                    last_error: Some("participant_lost".to_string()),
                })
            } else if target_set.contains(&member.device_id) {
                Some(DeviceStatusUpdate {
                    device_id: member.device_id.clone(),
                    status: if active_set.contains(&member.device_id) || !requires_transfer {
                        "decode_ready".to_string()
                    } else {
                        "decode_pending_transfer".to_string()
                    },
                    last_error: None,
                })
            } else {
                None
            }
        })
        .collect()
}

fn checkpoint_available(
    session: &InferenceSessionRecord,
    replicas: &[SessionReplicaRecord],
    target_participants: &[String],
) -> bool {
    if session.kv_checkpoint_device_id.is_some() && session.kv_checkpoint_created_at.is_some() {
        return true;
    }

    let target_set = target_participants.iter().collect::<HashSet<_>>();
    replicas.iter().any(|replica| {
        target_set.contains(&replica.device_id)
            && replica.checkpoint_created_at.is_some()
            && replica.kv_sequence_position.is_some()
    })
}

fn select_next_kv_owner(
    target_participants: &[String],
    session: &InferenceSessionRecord,
    replica_map: &HashMap<&str, &SessionReplicaRecord>,
    member_map: &HashMap<&str, &ExecutionGroupMember>,
) -> ApiResult<String> {
    if target_participants.contains(&session.kv_owner_device_id) {
        return Ok(session.kv_owner_device_id.clone());
    }

    target_participants
        .iter()
        .max_by_key(|device_id| {
            let replica = replica_map.get(device_id.as_str()).copied();
            let member = member_map.get(device_id.as_str()).copied();
            (
                replica
                    .and_then(|item| item.kv_sequence_position)
                    .unwrap_or(0),
                u8::from(
                    replica
                        .and_then(|item| item.checkpoint_created_at.as_ref())
                        .is_some(),
                ),
                member.map(|item| item.assigned_capacity_units).unwrap_or(0),
            )
        })
        .cloned()
        .ok_or_else(|| {
            ApiError::Conflict(format!(
                "Session {} has no surviving participant to own KV state",
                session.session_id
            ))
        })
}

fn coverage_target(
    participant_device_ids: &[String],
    member_map: &HashMap<&str, &ExecutionGroupMember>,
) -> ApiResult<(u32, u32)> {
    let mut min_start = None::<u32>;
    let mut max_end = None::<u32>;

    for device_id in participant_device_ids {
        let Some(member) = member_map.get(device_id.as_str()) else {
            return Err(ApiError::Conflict(format!(
                "Execution group is missing participant {}",
                device_id
            )));
        };
        min_start = Some(min_start.map_or(member.shard.column_start, |value| {
            value.min(member.shard.column_start)
        }));
        max_end = Some(max_end.map_or(member.shard.column_end, |value| {
            value.max(member.shard.column_end)
        }));
    }

    Ok((
        min_start
            .ok_or_else(|| ApiError::Conflict("No active participants available".to_string()))?,
        max_end
            .ok_or_else(|| ApiError::Conflict("No active participants available".to_string()))?,
    ))
}

fn find_covering_set(
    members: &[&ExecutionGroupMember],
    required_device_ids: &[String],
    required_span: (u32, u32),
    preferred_size: Option<usize>,
) -> Option<Vec<String>> {
    let required_set = required_device_ids.iter().collect::<HashSet<_>>();
    let candidate_pool = members
        .iter()
        .filter(|member| !required_set.contains(&member.device_id))
        .copied()
        .collect::<Vec<_>>();

    let min_size = preferred_size
        .unwrap_or(required_device_ids.len())
        .max(required_device_ids.len());
    let max_size = preferred_size
        .unwrap_or(members.len())
        .max(required_device_ids.len());
    let max_size = max_size.min(members.len());

    for target_size in min_size..=max_size {
        if target_size < required_device_ids.len() {
            continue;
        }

        let extra_needed = target_size - required_device_ids.len();
        if extra_needed > candidate_pool.len() {
            continue;
        }

        let mut selection = required_device_ids.to_vec();
        if extra_needed == 0 {
            if covers_required_span(members, &selection, required_span) {
                selection.sort();
                return Some(selection);
            }
            continue;
        }

        let mut result = None;
        choose_members(
            &candidate_pool,
            extra_needed,
            0,
            &mut selection,
            members,
            required_span,
            &mut result,
        );
        if result.is_some() {
            return result;
        }
    }

    None
}

fn choose_members(
    candidates: &[&ExecutionGroupMember],
    remaining: usize,
    start: usize,
    current: &mut Vec<String>,
    all_members: &[&ExecutionGroupMember],
    required_span: (u32, u32),
    result: &mut Option<Vec<String>>,
) {
    if result.is_some() {
        return;
    }
    if remaining == 0 {
        if covers_required_span(all_members, current, required_span) {
            let mut selected = current.clone();
            selected.sort();
            *result = Some(selected);
        }
        return;
    }
    if start >= candidates.len() {
        return;
    }

    for index in start..=candidates.len() - remaining {
        current.push(candidates[index].device_id.clone());
        choose_members(
            candidates,
            remaining - 1,
            index + 1,
            current,
            all_members,
            required_span,
            result,
        );
        current.pop();
        if result.is_some() {
            return;
        }
    }
}

fn covers_required_span(
    members: &[&ExecutionGroupMember],
    selected_device_ids: &[String],
    required_span: (u32, u32),
) -> bool {
    let selected = selected_device_ids.iter().collect::<HashSet<_>>();
    let mut ranges = members
        .iter()
        .filter(|member| selected.contains(&member.device_id))
        .map(|member| (member.shard.column_start, member.shard.column_end))
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return false;
    }

    ranges.sort_unstable_by_key(|range| range.0);
    if ranges[0].0 > required_span.0 {
        return false;
    }

    let mut covered_end = ranges[0].1;
    for (start, end) in ranges.into_iter().skip(1) {
        if start > covered_end.saturating_add(1) {
            return false;
        }
        if end > covered_end {
            covered_end = end;
        }
    }

    covered_end >= required_span.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{
        ExecutionGroup, ExecutionGroupMember, ExecutionPhase, ExecutionSegment,
        InferenceExecutionPlan, InferenceRuntimeMode, KvTransferPolicy, PeerPunchPlan, ShardInfo,
        TransportCapabilityTier,
    };
    use crate::connectivity::{
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath,
        InferenceSchedulingPolicy, NetworkConnectivity,
    };
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::provider::{ExecutionProviderInfo, ExecutionProviderKind};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::{device_service::register_device, network_service};
    use rusqlite::params;

    #[test]
    fn leaves_session_healthy_when_losses_do_not_hit_active_participants() {
        let (plan, session, replicas, groups, queue) = decode_fixture(
            KvTransferPolicy::ExportOnHandoff,
            vec![
                member("worker-1", 0, 49),
                member("worker-2", 50, 99),
                member("worker-3", 50, 99),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            Some("2026-04-25T12:00:00Z"),
        );

        let decision = FailoverEngine::evaluate(
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            &["worker-3".to_string()],
            &FailoverConfig::default(),
        )
        .expect("healthy decision");

        assert_eq!(decision.recovery_state, RecoveryState::Healthy);
        assert_eq!(decision.strategy, RegroupStrategy::None);
    }

    #[test]
    fn fails_prefill_when_an_active_participant_is_lost() {
        let (plan, session, replicas, groups) = prefill_fixture();

        let decision = FailoverEngine::evaluate(
            &plan,
            &session,
            &replicas,
            &groups,
            None,
            &["worker-2".to_string()],
            &FailoverConfig::default(),
        )
        .expect("prefill failure decision");

        assert_eq!(decision.recovery_state, RecoveryState::Failed);
        assert!(matches!(decision.strategy, RegroupStrategy::Fail { .. }));
        assert_eq!(decision.pause.session_status.as_deref(), Some("failed"));
    }

    #[test]
    fn pauses_decode_and_requests_checkpoint_transfer_when_replacement_is_needed() {
        let (plan, session, replicas, groups, queue) = decode_fixture(
            KvTransferPolicy::ExportOnHandoff,
            vec![
                member("worker-1", 0, 49),
                member("worker-2", 50, 99),
                member("worker-3", 50, 99),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            Some("2026-04-25T12:00:00Z"),
        );

        let decision = FailoverEngine::evaluate(
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            &["worker-2".to_string()],
            &FailoverConfig::default(),
        )
        .expect("decode replacement decision");

        assert_eq!(decision.recovery_state, RecoveryState::Paused);
        assert_eq!(
            decision.strategy,
            RegroupStrategy::Replace {
                lost_device_ids: vec!["worker-2".to_string()],
                replacement_device_ids: vec!["worker-3".to_string()],
            }
        );
        assert_eq!(
            decision.target_participant_device_ids,
            vec!["worker-1".to_string(), "worker-3".to_string()]
        );
        assert_eq!(
            decision.pause.decode_queue_status.as_deref(),
            Some("blocked_on_transfer")
        );
        assert!(decision
            .replica_updates
            .iter()
            .any(|update| update.device_id == "worker-3"
                && update.status == "decode_pending_transfer"));
    }

    #[test]
    fn shrinks_decode_when_survivors_still_cover_the_span() {
        let (plan, session, replicas, groups, queue) = decode_fixture(
            KvTransferPolicy::CoLocated,
            vec![
                member("worker-1", 0, 99),
                member("worker-2", 50, 99),
                member("worker-3", 0, 49),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            Some("2026-04-25T12:00:00Z"),
        );

        let decision = FailoverEngine::evaluate(
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            &["worker-2".to_string()],
            &FailoverConfig::default(),
        )
        .expect("decode shrink decision");

        assert_eq!(decision.recovery_state, RecoveryState::ResumeReady);
        assert_eq!(
            decision.strategy,
            RegroupStrategy::Shrink {
                removed_device_ids: vec!["worker-2".to_string()],
            }
        );
        assert_eq!(
            decision.target_participant_device_ids,
            vec!["worker-1".to_string()]
        );
        assert_eq!(decision.resume.mode, ResumeMode::Immediate);
        assert_eq!(decision.pause.decode_queue_status.as_deref(), Some("ready"));
    }

    #[test]
    fn fails_decode_when_replacement_needs_a_checkpoint_but_none_exists() {
        let (plan, session, replicas, groups, queue) = decode_fixture(
            KvTransferPolicy::ExportOnHandoff,
            vec![
                member("worker-1", 0, 49),
                member("worker-2", 50, 99),
                member("worker-3", 50, 99),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            None,
        );

        let decision = FailoverEngine::evaluate(
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            &["worker-2".to_string()],
            &FailoverConfig::default(),
        )
        .expect("decode failure decision");

        assert_eq!(decision.recovery_state, RecoveryState::Failed);
        assert!(decision.reason.contains("checkpoint"));
        assert_eq!(decision.pause.session_status.as_deref(), Some("failed"));
    }

    #[test]
    fn reconcile_failover_state_shrinks_decode_session_and_makes_it_resume_ready() {
        let db = create_test_db();
        register_fixture_devices(&db, &["worker-1", "worker-2", "worker-3"]);
        let (plan, session, replicas, groups, queue) = decode_fixture(
            KvTransferPolicy::CoLocated,
            vec![
                member("worker-1", 0, 99),
                member("worker-2", 50, 99),
                member("worker-3", 0, 49),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            Some("2026-04-25T12:00:00Z"),
        );
        persist_session_fixture(
            &db,
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            "running",
        );
        mark_device_status(&db, "worker-2", "offline");

        let reconciled = reconcile_failover_state(&db, &["worker-2".to_string()]).unwrap();
        assert_eq!(reconciled, 1);

        let conn = db.get_conn().unwrap();
        let queue_state: (String, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT status, lease_owner_device_id, lease_expires_at FROM inference_decode_queue WHERE session_id = 'session-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(queue_state.0, "ready");
        assert!(queue_state.1.is_none());
        assert!(queue_state.2.is_none());

        let session_state: (String, String, Option<String>) = conn
            .query_row(
                "SELECT status, kv_owner_device_id, last_error FROM inference_sessions WHERE session_id = 'session-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(session_state.0, "decode_ready");
        assert_eq!(session_state.1, "worker-1");

        let plan_json: String = conn
            .query_row(
                "SELECT execution_plan_json FROM inference_jobs WHERE job_id = 'job-a'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let regrouped_plan: InferenceExecutionPlan = serde_json::from_str(&plan_json).unwrap();
        let decode_segment = regrouped_plan
            .segments
            .iter()
            .find(|segment| segment.segment_id == "segment-decode")
            .unwrap();
        assert_eq!(
            decode_segment.participant_device_ids,
            vec!["worker-1".to_string()]
        );

        let assignment_statuses = load_assignment_statuses(&conn);
        assert_eq!(
            assignment_statuses
                .get("worker-1")
                .map(|entry| entry.0.as_str()),
            Some("pending")
        );
        assert_eq!(
            assignment_statuses
                .get("worker-2")
                .map(|entry| entry.0.as_str()),
            Some("waiting")
        );
    }

    #[test]
    fn reconcile_failover_state_pauses_decode_until_checkpoint_transfer_for_replacement() {
        let db = create_test_db();
        register_fixture_devices(&db, &["worker-1", "worker-2", "worker-3"]);
        let (plan, session, replicas, groups, queue) = decode_fixture(
            KvTransferPolicy::ExportOnHandoff,
            vec![
                member("worker-1", 0, 49),
                member("worker-2", 50, 99),
                member("worker-3", 50, 99),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            Some("2026-04-25T12:00:00Z"),
        );
        persist_session_fixture(
            &db,
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            "running",
        );
        mark_device_status(&db, "worker-2", "offline");

        let reconciled = reconcile_failover_state(&db, &["worker-2".to_string()]).unwrap();
        assert_eq!(reconciled, 1);

        let conn = db.get_conn().unwrap();
        let queue_state: (String, Option<String>) = conn
            .query_row(
                "SELECT status, blocked_reason FROM inference_decode_queue WHERE session_id = 'session-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(queue_state.0, "blocked_on_transfer");
        assert_eq!(queue_state.1.as_deref(), Some("transfer"));

        let session_state: (String, String) = conn
            .query_row(
                "SELECT status, kv_owner_device_id FROM inference_sessions WHERE session_id = 'session-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(session_state.0, "decode_ready");
        assert_eq!(session_state.1, "worker-1");

        let replica_statuses = load_replica_statuses(&conn);
        assert_eq!(
            replica_statuses.get("worker-3").map(String::as_str),
            Some("decode_pending_transfer")
        );
    }

    #[test]
    fn reconcile_failover_state_fails_prefill_job_and_cleans_dead_session_state() {
        let db = create_test_db();
        register_fixture_devices(&db, &["worker-1", "worker-2"]);
        let (plan, session, replicas, groups) = prefill_fixture();
        persist_session_fixture(&db, &plan, &session, &replicas, &groups, None, "running");
        mark_device_status(&db, "worker-2", "offline");

        let reconciled = reconcile_failover_state(&db, &["worker-2".to_string()]).unwrap();
        assert_eq!(reconciled, 1);

        let conn = db.get_conn().unwrap();
        let job_state: (String, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT status, active_segment_id, error FROM inference_jobs WHERE job_id = 'job-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(job_state.0, "failed");
        assert!(job_state.1.is_none());
        assert!(job_state.2.unwrap().contains("Prefill participant loss"));

        let session_state: (String, Option<String>) = conn
            .query_row(
                "SELECT status, active_segment_id FROM inference_sessions WHERE session_id = 'session-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(session_state.0, "failed");
        assert!(session_state.1.is_none());
    }

    #[test]
    fn reconcile_failover_state_cleans_terminal_sessions_and_orphaned_leases() {
        let db = create_test_db();
        register_fixture_devices(&db, &["worker-1", "worker-2", "worker-3"]);
        let (plan, mut session, replicas, mut groups, mut queue) = decode_fixture(
            KvTransferPolicy::CoLocated,
            vec![
                member("worker-1", 0, 49),
                member("worker-2", 50, 99),
                member("worker-3", 50, 99),
            ],
            vec!["worker-1".into(), "worker-2".into()],
            "worker-1",
            Some("2026-04-25T12:00:00Z"),
        );
        session.status = "failed".to_string();
        queue.status = "leased".to_string();
        queue.lease_owner_device_id = Some("worker-2".to_string());
        groups[0].status = "decode_leased".to_string();
        persist_session_fixture(
            &db,
            &plan,
            &session,
            &replicas,
            &groups,
            Some(&queue),
            "failed",
        );
        mark_device_status(&db, "worker-2", "offline");

        let reconciled = reconcile_failover_state(&db, &[]).unwrap();
        assert_eq!(reconciled, 0);

        let conn = db.get_conn().unwrap();
        let session_active_segment: Option<String> = conn
            .query_row(
                "SELECT active_segment_id FROM inference_sessions WHERE session_id = 'session-a'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(session_active_segment.is_none());

        let queue_state: (String, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT status, lease_owner_device_id, lease_expires_at FROM inference_decode_queue WHERE session_id = 'session-a'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(queue_state.0, "failed");
        assert!(queue_state.1.is_none());
        assert!(queue_state.2.is_none());
    }

    fn prefill_fixture() -> (
        InferenceExecutionPlan,
        InferenceSessionRecord,
        Vec<SessionReplicaRecord>,
        Vec<ServingGroupRecord>,
    ) {
        let members = vec![member("worker-1", 0, 49), member("worker-2", 50, 99)];
        let prefill_group = ExecutionGroup {
            group_id: "group-prefill".into(),
            model_id: "model-a".into(),
            phase: ExecutionPhase::Prefill,
            transport_tier: TransportCapabilityTier::DirectPreferred,
            kv_transfer_policy: KvTransferPolicy::CoLocated,
            total_capacity_units: 2,
            members: members.clone(),
            peer_punch_plans: Vec::<PeerPunchPlan>::new(),
        };
        let prefill_segment = ExecutionSegment {
            segment_id: "segment-prefill".into(),
            session_id: "session-a".into(),
            execution_group_id: prefill_group.group_id.clone(),
            phase: ExecutionPhase::Prefill,
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 16,
            temperature: 0.7,
            top_p: 0.9,
            kv_owner_device_id: "worker-1".into(),
            shard_owner_device_ids: vec!["worker-1".into(), "worker-2".into()],
            participant_device_ids: vec!["worker-1".into(), "worker-2".into()],
        };
        let plan = InferenceExecutionPlan {
            plan_id: "plan-a".into(),
            runtime_mode: InferenceRuntimeMode::ThroughputFirst,
            execution_groups: vec![prefill_group],
            segments: vec![prefill_segment.clone()],
            initial_segment_id: prefill_segment.segment_id.clone(),
        };
        let session = InferenceSessionRecord {
            session_id: "session-a".into(),
            job_id: "job-a".into(),
            network_id: "network-a".into(),
            model_id: "model-a".into(),
            status: "prefill_active".into(),
            active_segment_id: Some(prefill_segment.segment_id),
            kv_owner_device_id: "worker-1".into(),
            kv_transfer_policy: "co_located".into(),
            kv_sequence_position: None,
            kv_checkpoint_device_id: None,
            kv_checkpoint_created_at: None,
            last_error: None,
        };
        let replicas = vec![
            replica(
                "session-a",
                "job-a",
                "worker-1",
                "prefill_active",
                None,
                None,
            ),
            replica(
                "session-a",
                "job-a",
                "worker-2",
                "prefill_active",
                None,
                None,
            ),
        ];
        let groups = vec![
            group_record(
                "group-prefill",
                "session-a",
                "job-a",
                "worker-1",
                ExecutionPhase::Prefill,
                0,
                49,
                "prefill_member",
            ),
            group_record(
                "group-prefill",
                "session-a",
                "job-a",
                "worker-2",
                ExecutionPhase::Prefill,
                50,
                99,
                "prefill_member",
            ),
        ];
        (plan, session, replicas, groups)
    }

    fn decode_fixture(
        kv_transfer_policy: KvTransferPolicy,
        members: Vec<ExecutionGroupMember>,
        active_participants: Vec<String>,
        kv_owner_device_id: &str,
        checkpoint_created_at: Option<&str>,
    ) -> (
        InferenceExecutionPlan,
        InferenceSessionRecord,
        Vec<SessionReplicaRecord>,
        Vec<ServingGroupRecord>,
        DecodeQueueRecord,
    ) {
        let decode_group = ExecutionGroup {
            group_id: "group-decode".into(),
            model_id: "model-a".into(),
            phase: ExecutionPhase::Decode,
            transport_tier: TransportCapabilityTier::DirectPreferred,
            kv_transfer_policy,
            total_capacity_units: members.len() as u32,
            members: members.clone(),
            peer_punch_plans: Vec::<PeerPunchPlan>::new(),
        };
        let decode_segment = ExecutionSegment {
            segment_id: "segment-decode".into(),
            session_id: "session-a".into(),
            execution_group_id: decode_group.group_id.clone(),
            phase: ExecutionPhase::Decode,
            prompt_tokens: Vec::new(),
            max_tokens: 16,
            temperature: 0.7,
            top_p: 0.9,
            kv_owner_device_id: kv_owner_device_id.into(),
            shard_owner_device_ids: active_participants.clone(),
            participant_device_ids: active_participants.clone(),
        };
        let plan = InferenceExecutionPlan {
            plan_id: "plan-a".into(),
            runtime_mode: InferenceRuntimeMode::ResilientEdge,
            execution_groups: vec![decode_group],
            segments: vec![decode_segment.clone()],
            initial_segment_id: decode_segment.segment_id.clone(),
        };
        let checkpoint_created_at = checkpoint_created_at.map(str::to_string);
        let session = InferenceSessionRecord {
            session_id: "session-a".into(),
            job_id: "job-a".into(),
            network_id: "network-a".into(),
            model_id: "model-a".into(),
            status: "decode_active".into(),
            active_segment_id: Some(decode_segment.segment_id.clone()),
            kv_owner_device_id: kv_owner_device_id.into(),
            kv_transfer_policy: match kv_transfer_policy {
                KvTransferPolicy::CoLocated => "co_located",
                KvTransferPolicy::ExportOnHandoff => "export_on_handoff",
                KvTransferPolicy::RemoteAccess => "remote_access",
            }
            .into(),
            kv_sequence_position: Some(32),
            kv_checkpoint_device_id: checkpoint_created_at
                .as_ref()
                .map(|_| kv_owner_device_id.to_string()),
            kv_checkpoint_created_at: checkpoint_created_at.clone(),
            last_error: None,
        };
        let replicas = members
            .iter()
            .map(|member| {
                let is_active = active_participants.contains(&member.device_id);
                replica(
                    "session-a",
                    "job-a",
                    &member.device_id,
                    if is_active {
                        "decode_active"
                    } else {
                        "waiting"
                    },
                    Some(32),
                    checkpoint_created_at.as_deref(),
                )
            })
            .collect::<Vec<_>>();
        let groups = members
            .iter()
            .map(|member| {
                group_record(
                    "group-decode",
                    "session-a",
                    "job-a",
                    &member.device_id,
                    ExecutionPhase::Decode,
                    member.shard.column_start,
                    member.shard.column_end,
                    if active_participants.contains(&member.device_id) {
                        "decode_member"
                    } else {
                        "standby"
                    },
                )
            })
            .collect::<Vec<_>>();
        let queue = DecodeQueueRecord {
            session_id: "session-a".into(),
            job_id: "job-a".into(),
            network_id: "network-a".into(),
            segment_id: decode_segment.segment_id,
            group_id: "group-decode".into(),
            status: "active".into(),
            ready_at: Some("2026-04-25T12:00:00Z".into()),
            lease_owner_device_id: Some("worker-1".into()),
            lease_expires_at: Some("2026-04-25T12:05:00Z".into()),
            last_error: None,
        };
        (plan, session, replicas, groups, queue)
    }

    fn member(device_id: &str, start: u32, end: u32) -> ExecutionGroupMember {
        ExecutionGroupMember {
            device_id: device_id.into(),
            peer_id: format!("peer-{}", device_id),
            ring_position: start,
            status: "ready".into(),
            contributed_memory: 1024,
            shard: ShardInfo {
                model_id: "model-a".into(),
                column_start: start,
                column_end: end,
                estimated_memory: 1024,
            },
            left_neighbor: String::new(),
            right_neighbor: String::new(),
            connectivity_state: None,
            listen_addrs: Vec::new(),
            direct_candidates: Vec::new(),
            assigned_capacity_units: 1,
            execution_provider: "cpu".into(),
        }
    }

    fn replica(
        session_id: &str,
        job_id: &str,
        device_id: &str,
        status: &str,
        kv_sequence_position: Option<u32>,
        checkpoint_created_at: Option<&str>,
    ) -> SessionReplicaRecord {
        SessionReplicaRecord {
            session_id: session_id.into(),
            device_id: device_id.into(),
            job_id: job_id.into(),
            status: status.into(),
            active_segment_id: Some("segment-decode".into()),
            kv_sequence_position,
            checkpoint_created_at: checkpoint_created_at.map(str::to_string),
            last_error: None,
        }
    }

    fn group_record(
        group_id: &str,
        session_id: &str,
        job_id: &str,
        device_id: &str,
        phase: ExecutionPhase,
        shard_column_start: u32,
        shard_column_end: u32,
        status: &str,
    ) -> ServingGroupRecord {
        ServingGroupRecord {
            group_id: group_id.into(),
            session_id: session_id.into(),
            job_id: job_id.into(),
            network_id: "network-a".into(),
            model_id: "model-a".into(),
            phase,
            device_id: device_id.into(),
            ring_position: shard_column_start,
            shard_column_start,
            shard_column_end,
            assigned_capacity_units: 1,
            execution_provider: "cpu".into(),
            status: status.into(),
            last_error: None,
        }
    }

    fn persist_session_fixture(
        db: &crate::db::Database,
        plan: &InferenceExecutionPlan,
        session: &InferenceSessionRecord,
        replicas: &[SessionReplicaRecord],
        groups: &[ServingGroupRecord],
        queue: Option<&DecodeQueueRecord>,
        job_status: &str,
    ) {
        let conn = db.get_conn().unwrap();
        let active_segment = session.active_segment_id.as_ref().unwrap();
        conn.execute(
            r#"
            INSERT INTO inference_jobs (
                job_id, network_id, submitted_by_device_id, model_id, prompt, prompt_tokens,
                max_tokens, temperature, top_p, status, ring_worker_count, created_at, updated_at,
                execution_plan_json, active_segment_id
            ) VALUES (?, ?, ?, ?, 'prompt', '[]', 16, 0.7, 0.9, ?, ?, '2026-04-25T12:00:00Z',
                      '2026-04-25T12:00:00Z', ?, ?)
            "#,
            params![
                &session.job_id,
                &session.network_id,
                "submitter-1",
                &session.model_id,
                job_status,
                plan.execution_groups[0].members.len() as i64,
                serde_json::to_string(plan).unwrap(),
                active_segment
            ],
        )
        .unwrap();
        conn.execute(
            r#"
            INSERT INTO inference_sessions (
                session_id, job_id, network_id, model_id, status, active_segment_id,
                kv_owner_device_id, kv_transfer_policy, kv_sequence_position,
                kv_checkpoint_device_id, kv_checkpoint_created_at, last_error,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-04-25T12:00:00Z', '2026-04-25T12:00:00Z')
            "#,
            params![
                &session.session_id,
                &session.job_id,
                &session.network_id,
                &session.model_id,
                &session.status,
                session.active_segment_id.as_deref(),
                &session.kv_owner_device_id,
                &session.kv_transfer_policy,
                session.kv_sequence_position.map(i64::from),
                session.kv_checkpoint_device_id.as_deref(),
                session.kv_checkpoint_created_at.as_deref(),
                session.last_error.as_deref()
            ],
        )
        .unwrap();

        let active_participants = plan
            .segments
            .iter()
            .find(|segment| segment.segment_id == *active_segment)
            .unwrap()
            .participant_device_ids
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        for member in &plan.execution_groups[0].members {
            conn.execute(
                r#"
                INSERT INTO inference_job_assignments (
                    assignment_id, job_id, network_id, device_id, ring_position, status,
                    lease_expires_at, assigned_at, acknowledged_at, completed_at, failure_reason,
                    shard_column_start, shard_column_end, assigned_capacity_units,
                    execution_provider, active_segment_id, reported_completion_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '2026-04-25T12:00:00Z', NULL, NULL, NULL, ?, ?, ?, ?, ?, 0)
                "#,
                params![
                    format!("assignment-{}", member.device_id),
                    &session.job_id,
                    &session.network_id,
                    &member.device_id,
                    i64::from(member.ring_position),
                    if active_participants.contains(&member.device_id) {
                        "leased"
                    } else {
                        "waiting"
                    },
                    if active_participants.contains(&member.device_id) {
                        Some("2026-04-25T12:05:00Z")
                    } else {
                        None
                    },
                    i64::from(member.shard.column_start),
                    i64::from(member.shard.column_end),
                    i64::from(member.assigned_capacity_units),
                    &member.execution_provider,
                    if active_participants.contains(&member.device_id) {
                        Some(active_segment.as_str())
                    } else {
                        None
                    }
                ],
            )
            .unwrap();
        }

        for replica in replicas {
            conn.execute(
                r#"
                INSERT INTO inference_session_replicas (
                    session_id, device_id, job_id, status, active_segment_id,
                    kv_sequence_position, checkpoint_created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '2026-04-25T12:00:00Z', ?)
                "#,
                params![
                    &replica.session_id,
                    &replica.device_id,
                    &replica.job_id,
                    &replica.status,
                    replica.active_segment_id.as_deref(),
                    replica.kv_sequence_position.map(i64::from),
                    replica.checkpoint_created_at.as_deref(),
                    replica.last_error.as_deref()
                ],
            )
            .unwrap();
        }

        for group in groups {
            conn.execute(
                r#"
                INSERT INTO inference_serving_groups (
                    group_id, session_id, job_id, network_id, model_id, phase, device_id,
                    ring_position, shard_column_start, shard_column_end, assigned_capacity_units,
                    execution_provider, status, last_error, lease_owner_device_id, lease_expires_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-04-25T12:00:00Z')
                "#,
                params![
                    &group.group_id,
                    &group.session_id,
                    &group.job_id,
                    &group.network_id,
                    &group.model_id,
                    match group.phase {
                        ExecutionPhase::Prefill => "prefill",
                        ExecutionPhase::Decode => "decode",
                    },
                    &group.device_id,
                    i64::from(group.ring_position),
                    i64::from(group.shard_column_start),
                    i64::from(group.shard_column_end),
                    i64::from(group.assigned_capacity_units),
                    &group.execution_provider,
                    &group.status,
                    group.last_error.as_deref(),
                    if group.status.ends_with("leased") || group.status.ends_with("active") {
                        Some("worker-1")
                    } else {
                        None
                    },
                    if group.status.ends_with("leased") || group.status.ends_with("active") {
                        Some("2026-04-25T12:05:00Z")
                    } else {
                        None
                    }
                ],
            )
            .unwrap();
        }

        if let Some(queue) = queue {
            conn.execute(
                r#"
                INSERT INTO inference_decode_queue (
                    session_id, job_id, network_id, segment_id, group_id, batch_group_key, status,
                    ready_at, blocked_reason, lease_owner_device_id, lease_expires_at,
                    lease_target_session_count, lease_target_batch_size, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 2, 2, ?, '2026-04-25T12:00:00Z')
                "#,
                params![
                    &queue.session_id,
                    &queue.job_id,
                    &queue.network_id,
                    &queue.segment_id,
                    &queue.group_id,
                    "decode:model-a:worker-1,worker-2,worker-3",
                    &queue.status,
                    queue.ready_at.as_deref(),
                    queue.lease_owner_device_id.as_deref(),
                    queue.lease_expires_at.as_deref(),
                    queue.last_error.as_deref()
                ],
            )
            .unwrap();
        }
    }

    fn register_fixture_devices(db: &crate::db::Database, device_ids: &[&str]) {
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            db,
            "network-a".to_string(),
            "network-a".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
        )
        .unwrap();
        register_device(
            db,
            &keypair,
            "submitter-1".to_string(),
            "network-a".to_string(),
            "Submitter".to_string(),
            vec![7u8; 32],
            "12D3KooWSubmitter11111111111111111111111111111".to_string(),
            test_capabilities(),
        )
        .unwrap();
        for (index, device_id) in device_ids.iter().enumerate() {
            register_device(
                db,
                &keypair,
                (*device_id).to_string(),
                "network-a".to_string(),
                (*device_id).to_string(),
                vec![index as u8 + 10; 32],
                format!("12D3KooW{}{:032}", device_id.replace('-', ""), index),
                test_capabilities(),
            )
            .unwrap();
        }
    }

    fn mark_device_status(db: &crate::db::Database, device_id: &str, status: &str) {
        db.get_conn()
            .unwrap()
            .execute(
                "UPDATE devices SET status = ? WHERE device_id = ?",
                params![status, device_id],
            )
            .unwrap();
    }

    fn load_assignment_statuses(
        conn: &rusqlite::Connection,
    ) -> HashMap<String, (String, Option<String>)> {
        let mut stmt = conn
            .prepare(
                "SELECT device_id, status, active_segment_id FROM inference_job_assignments WHERE job_id = 'job-a'",
            )
            .unwrap();
        stmt.query_map([], |row| Ok((row.get(0)?, (row.get(1)?, row.get(2)?))))
            .unwrap()
            .collect::<Result<HashMap<_, _>, _>>()
            .unwrap()
    }

    fn load_replica_statuses(conn: &rusqlite::Connection) -> HashMap<String, String> {
        let mut stmt = conn
            .prepare("SELECT device_id, status FROM inference_session_replicas WHERE session_id = 'session-a'")
            .unwrap();
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<Result<HashMap<_, _>, _>>()
            .unwrap()
    }

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            tier: Tier::Tier1,
            cpu_cores: 4,
            ram_mb: 8192,
            gpu_present: false,
            gpu_vram_mb: None,
            os: "linux".into(),
            arch: "x86_64".into(),
            execution_providers: vec![
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cpu,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Metal,
                    available: false,
                    reason: Some("metal provider is only available on macOS".into()),
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cuda,
                    available: true,
                    reason: None,
                },
            ],
            default_execution_provider: ExecutionProviderKind::Cuda,
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
}
