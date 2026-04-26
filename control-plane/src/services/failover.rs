use std::collections::{BTreeSet, HashMap, HashSet};

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    ExecutionGroup, ExecutionGroupMember, ExecutionPhase, InferenceExecutionPlan,
};

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
}
