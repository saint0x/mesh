use uuid::Uuid;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    ExecutionGroup, ExecutionGroupMember, ExecutionPhase, ExecutionSegment, InferenceExecutionPlan,
    InferenceRuntimeMode, KvTransferPolicy, PunchPathReason, PunchPathStrategy, ShardInfo,
    SubmitInferenceRequest, TransportCapabilityTier,
};
use crate::connectivity::{ConnectivityPath, DeviceConnectivityState, InferenceSchedulingPolicy};
use crate::device::{DeviceCapabilities, Tier};
use crate::provider::ExecutionProviderKind;
use crate::services::ring_manager::{RingTopology, WorkerTopologyInfo};

#[derive(Debug, Clone)]
pub struct PlannerDeviceMetadata {
    pub assigned_capacity_units: u32,
    pub execution_provider: String,
}

pub struct ExecutionPlanner;

impl ExecutionPlanner {
    pub fn plan(
        req: &SubmitInferenceRequest,
        prompt_tokens: &[u32],
        topology: &RingTopology,
        scheduling_policy: &InferenceSchedulingPolicy,
        device_metadata: &[PlannerDeviceMetadata],
    ) -> ApiResult<InferenceExecutionPlan> {
        if topology.workers.len() != device_metadata.len() {
            return Err(ApiError::Internal(
                "planner received mismatched topology and device metadata".to_string(),
            ));
        }

        let members = topology
            .workers
            .iter()
            .zip(device_metadata.iter())
            .map(|(worker, metadata)| build_member(worker, metadata))
            .collect::<Vec<_>>();
        let total_capacity_units = members
            .iter()
            .map(|member| member.assigned_capacity_units)
            .sum();
        let runtime_mode = derive_runtime_mode(scheduling_policy, &members);
        let transport_tier = classify_transport_tier(&members);
        let kv_owner_device_id = select_kv_owner(&members);
        let prefill_participant_device_ids = members
            .iter()
            .map(|member| member.device_id.clone())
            .collect::<Vec<_>>();
        let decode_members = select_decode_members(&members, runtime_mode);
        let decode_transport_tier = classify_transport_tier(&decode_members);
        let decode_kv_owner_device_id = select_kv_owner(&decode_members);
        let decode_participant_device_ids = decode_members
            .iter()
            .map(|member| member.device_id.clone())
            .collect::<Vec<_>>();
        let plan_id = Uuid::new_v4().to_string();
        let session_id = Uuid::new_v4().to_string();
        let prefill_group_id = format!("group-{}-prefill", plan_id);
        let decode_group_id = format!("group-{}-decode", plan_id);
        let prefill_segment_id = format!("segment-{}-prefill", session_id);
        let decode_segment_id = format!("segment-{}-decode", session_id);
        let peer_punch_plans = topology
            .peer_punch_plans
            .iter()
            .map(map_peer_punch_plan)
            .collect::<Vec<_>>();
        let prefill_group = ExecutionGroup {
            group_id: prefill_group_id.clone(),
            model_id: req.model_id.clone(),
            phase: ExecutionPhase::Prefill,
            transport_tier,
            kv_transfer_policy: KvTransferPolicy::CoLocated,
            total_capacity_units,
            members: members.clone(),
            peer_punch_plans: peer_punch_plans.clone(),
        };
        let decode_group = ExecutionGroup {
            group_id: decode_group_id.clone(),
            model_id: req.model_id.clone(),
            phase: ExecutionPhase::Decode,
            transport_tier: decode_transport_tier,
            kv_transfer_policy: if same_participants(&members, &decode_members) {
                KvTransferPolicy::CoLocated
            } else if matches!(runtime_mode, InferenceRuntimeMode::ResilientEdge) {
                KvTransferPolicy::RemoteAccess
            } else {
                KvTransferPolicy::ExportOnHandoff
            },
            total_capacity_units: decode_members
                .iter()
                .map(|member| member.assigned_capacity_units)
                .sum(),
            members: decode_members.clone(),
            peer_punch_plans,
        };
        let prefill_segment = ExecutionSegment {
            segment_id: prefill_segment_id.clone(),
            session_id: session_id.clone(),
            execution_group_id: prefill_group_id,
            phase: ExecutionPhase::Prefill,
            prompt_tokens: prompt_tokens.to_vec(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            kv_owner_device_id: kv_owner_device_id.clone(),
            shard_owner_device_ids: prefill_participant_device_ids.clone(),
            participant_device_ids: prefill_participant_device_ids.clone(),
        };
        let decode_segment = ExecutionSegment {
            segment_id: decode_segment_id,
            session_id,
            execution_group_id: decode_group_id,
            phase: ExecutionPhase::Decode,
            prompt_tokens: Vec::new(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            kv_owner_device_id: decode_kv_owner_device_id,
            shard_owner_device_ids: decode_participant_device_ids.clone(),
            participant_device_ids: decode_participant_device_ids,
        };

        Ok(InferenceExecutionPlan {
            plan_id,
            runtime_mode,
            execution_groups: vec![prefill_group, decode_group],
            segments: vec![prefill_segment, decode_segment],
            initial_segment_id: prefill_segment_id,
        })
    }

    pub fn refresh_decode_plan(
        plan: &InferenceExecutionPlan,
        topology: &RingTopology,
        scheduling_policy: &InferenceSchedulingPolicy,
        device_metadata: &[PlannerDeviceMetadata],
    ) -> ApiResult<InferenceExecutionPlan> {
        if topology.workers.len() != device_metadata.len() {
            return Err(ApiError::Internal(
                "planner received mismatched topology and device metadata".to_string(),
            ));
        }

        let required_coverage = required_decode_coverage(plan).ok_or_else(|| {
            ApiError::Internal(format!(
                "Execution plan {} is missing a valid prefill coverage span",
                plan.plan_id
            ))
        })?;

        let members = topology
            .workers
            .iter()
            .zip(device_metadata.iter())
            .map(|(worker, metadata)| build_member(worker, metadata))
            .collect::<Vec<_>>();
        if !covers_span(&members, required_coverage) {
            return Err(ApiError::Conflict(format!(
                "Live topology no longer covers required shard span {}..{} for decode",
                required_coverage.0, required_coverage.1
            )));
        }

        let runtime_mode = derive_runtime_mode(scheduling_policy, &members);
        let decode_members =
            select_decode_members_for_span(&members, runtime_mode, required_coverage);
        let prefill_members = plan
            .execution_groups
            .iter()
            .find(|group| matches!(group.phase, ExecutionPhase::Prefill))
            .map(|group| group.members.clone())
            .unwrap_or_default();
        let decode_transport_tier = classify_transport_tier(&decode_members);
        let decode_kv_owner_device_id = select_kv_owner(&decode_members);
        let decode_participant_device_ids = decode_members
            .iter()
            .map(|member| member.device_id.clone())
            .collect::<Vec<_>>();

        let mut refreshed = plan.clone();
        refreshed.runtime_mode = runtime_mode;

        let decode_group = refreshed
            .execution_groups
            .iter_mut()
            .find(|group| matches!(group.phase, ExecutionPhase::Decode))
            .ok_or_else(|| {
                ApiError::Internal(format!(
                    "Execution plan {} is missing a decode group",
                    plan.plan_id
                ))
            })?;
        let existing_punch_plans = decode_group.peer_punch_plans.clone();
        decode_group.transport_tier = decode_transport_tier;
        decode_group.kv_transfer_policy = if same_participants(&prefill_members, &decode_members) {
            KvTransferPolicy::CoLocated
        } else if matches!(runtime_mode, InferenceRuntimeMode::ResilientEdge) {
            KvTransferPolicy::RemoteAccess
        } else {
            KvTransferPolicy::ExportOnHandoff
        };
        decode_group.total_capacity_units = decode_members
            .iter()
            .map(|member| member.assigned_capacity_units)
            .sum();
        decode_group.members = decode_members.clone();
        decode_group.peer_punch_plans = existing_punch_plans
            .into_iter()
            .filter(|plan| {
                decode_participant_device_ids
                    .iter()
                    .any(|device_id| device_id == &plan.source_device_id)
                    && decode_participant_device_ids
                        .iter()
                        .any(|device_id| device_id == &plan.target_device_id)
            })
            .collect();

        let decode_segment = refreshed
            .segments
            .iter_mut()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Decode))
            .ok_or_else(|| {
                ApiError::Internal(format!(
                    "Execution plan {} is missing a decode segment",
                    plan.plan_id
                ))
            })?;
        decode_segment.kv_owner_device_id = decode_kv_owner_device_id;
        decode_segment.shard_owner_device_ids = decode_participant_device_ids.clone();
        decode_segment.participant_device_ids = decode_participant_device_ids;

        Ok(refreshed)
    }
}

fn build_member(
    worker: &WorkerTopologyInfo,
    metadata: &PlannerDeviceMetadata,
) -> ExecutionGroupMember {
    ExecutionGroupMember {
        device_id: worker.device_id.clone(),
        peer_id: worker.peer_id.clone(),
        ring_position: worker.position,
        status: worker.status.clone(),
        contributed_memory: worker.contributed_memory,
        shard: ShardInfo {
            model_id: worker.shard.model_id.clone(),
            column_start: worker.shard.column_range.0,
            column_end: worker.shard.column_range.1,
            estimated_memory: worker.shard.estimated_memory,
        },
        left_neighbor: worker.left_neighbor.clone(),
        right_neighbor: worker.right_neighbor.clone(),
        connectivity_state: worker.connectivity_state.clone(),
        listen_addrs: worker.listen_addrs.clone(),
        direct_candidates: worker.direct_candidates.clone(),
        assigned_capacity_units: metadata.assigned_capacity_units,
        execution_provider: metadata.execution_provider.clone(),
    }
}

fn classify_transport_tier(members: &[ExecutionGroupMember]) -> TransportCapabilityTier {
    if members.iter().all(|member| {
        member
            .listen_addrs
            .iter()
            .any(|addr| addr.starts_with("dataplane://"))
            && member
                .connectivity_state
                .as_ref()
                .map(|state| {
                    state.status == crate::connectivity::ConnectivityStatus::Connected
                        && state.active_path == ConnectivityPath::Direct
                })
                .unwrap_or(false)
    }) {
        TransportCapabilityTier::DirectPreferred
    } else if members.iter().all(|member| {
        member
            .listen_addrs
            .iter()
            .any(|addr| addr.starts_with("dataplane://"))
    }) {
        TransportCapabilityTier::DirectTcp
    } else {
        TransportCapabilityTier::RelayFallback
    }
}

fn select_kv_owner(members: &[ExecutionGroupMember]) -> String {
    members
        .iter()
        .max_by_key(|member| {
            (
                member.assigned_capacity_units,
                member.contributed_memory,
                std::cmp::Reverse(member.ring_position),
            )
        })
        .map(|member| member.device_id.clone())
        .unwrap_or_default()
}

fn select_decode_members(
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
) -> Vec<ExecutionGroupMember> {
    let Some(full_coverage) = coverage_span(members) else {
        return members.to_vec();
    };

    select_decode_members_for_span(members, runtime_mode, full_coverage)
}

fn select_decode_members_for_span(
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
    required_coverage: (u32, u32),
) -> Vec<ExecutionGroupMember> {
    if !matches!(
        runtime_mode,
        InferenceRuntimeMode::ThroughputFirst | InferenceRuntimeMode::LatencyFirst
    ) {
        return members.to_vec();
    }

    let mut ranked = members.to_vec();
    ranked.sort_by(|left, right| {
        right
            .assigned_capacity_units
            .cmp(&left.assigned_capacity_units)
            .then_with(|| right.contributed_memory.cmp(&left.contributed_memory))
            .then_with(|| left.ring_position.cmp(&right.ring_position))
    });

    let mut selected = Vec::new();
    for member in ranked {
        selected.push(member);
        if covers_span(&selected, required_coverage) && selected.len() < members.len() {
            selected.sort_by_key(|member| member.ring_position);
            return selected;
        }
    }

    members.to_vec()
}

fn covers_span(members: &[ExecutionGroupMember], required_coverage: (u32, u32)) -> bool {
    if required_coverage.0 >= required_coverage.1 {
        return false;
    }

    let mut intervals = members
        .iter()
        .map(|member| (member.shard.column_start, member.shard.column_end))
        .filter(|(start, end)| *end > required_coverage.0 && *start < required_coverage.1)
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|interval| (interval.0, interval.1));

    let mut cursor = required_coverage.0;
    for (start, end) in intervals {
        if start > cursor {
            return false;
        }
        cursor = cursor.max(end);
        if cursor >= required_coverage.1 {
            return true;
        }
    }

    cursor >= required_coverage.1
}

fn required_decode_coverage(plan: &InferenceExecutionPlan) -> Option<(u32, u32)> {
    let prefill_group = plan
        .execution_groups
        .iter()
        .find(|group| matches!(group.phase, ExecutionPhase::Prefill))?;
    coverage_span(&prefill_group.members)
}

fn coverage_span(members: &[ExecutionGroupMember]) -> Option<(u32, u32)> {
    let min_start = members
        .iter()
        .map(|member| member.shard.column_start)
        .min()?;
    let max_end = members.iter().map(|member| member.shard.column_end).max()?;
    if min_start >= max_end {
        return None;
    }

    let mut intervals = members
        .iter()
        .map(|member| (member.shard.column_start, member.shard.column_end))
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|interval| (interval.0, interval.1));

    let mut cursor = min_start;
    for (start, end) in intervals {
        if start > cursor {
            return None;
        }
        cursor = cursor.max(end);
    }

    if cursor == max_end {
        Some((min_start, max_end))
    } else {
        None
    }
}

fn same_participants(left: &[ExecutionGroupMember], right: &[ExecutionGroupMember]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut left_ids = left
        .iter()
        .map(|member| member.device_id.as_str())
        .collect::<Vec<_>>();
    let mut right_ids = right
        .iter()
        .map(|member| member.device_id.as_str())
        .collect::<Vec<_>>();
    left_ids.sort_unstable();
    right_ids.sort_unstable();
    left_ids == right_ids
}

fn derive_runtime_mode(
    scheduling_policy: &InferenceSchedulingPolicy,
    members: &[ExecutionGroupMember],
) -> InferenceRuntimeMode {
    let has_relay_or_degraded = members.iter().any(|member| {
        member
            .connectivity_state
            .as_ref()
            .map(is_relay_or_degraded)
            .unwrap_or(true)
    });

    if has_relay_or_degraded {
        return InferenceRuntimeMode::ResilientEdge;
    }

    let wide_capacity_spread = scheduling_policy.tier_capacity_units.tier4
        > scheduling_policy
            .tier_capacity_units
            .tier1
            .saturating_mul(4);
    if wide_capacity_spread {
        InferenceRuntimeMode::ThroughputFirst
    } else {
        InferenceRuntimeMode::LatencyFirst
    }
}

fn map_peer_punch_plan(
    plan: &crate::services::ring_manager::PeerPunchPlan,
) -> crate::api::types::PeerPunchPlan {
    crate::api::types::PeerPunchPlan {
        source_device_id: plan.source_device_id.clone(),
        target_device_id: plan.target_device_id.clone(),
        target_peer_id: plan.target_peer_id.clone(),
        strategy: match plan.strategy {
            crate::services::ring_manager::PunchPathStrategy::SimultaneousDial => {
                PunchPathStrategy::SimultaneousDial
            }
        },
        reason: match plan.reason {
            crate::services::ring_manager::PunchPathReason::RelayPath => PunchPathReason::RelayPath,
            crate::services::ring_manager::PunchPathReason::DegradedConnectivity => {
                PunchPathReason::DegradedConnectivity
            }
            crate::services::ring_manager::PunchPathReason::PrivateReachabilityOnly => {
                PunchPathReason::PrivateReachabilityOnly
            }
        },
        relay_rendezvous_required: plan.relay_rendezvous_required,
        attempt_window_ms: plan.attempt_window_ms,
        issued_at_ms: plan.issued_at_ms,
        target_candidates: plan.target_candidates.clone(),
    }
}

fn is_relay_or_degraded(state: &DeviceConnectivityState) -> bool {
    state.active_path == ConnectivityPath::Relayed
        || state.status != crate::connectivity::ConnectivityStatus::Connected
}

pub fn capacity_units_for_tier(policy: &InferenceSchedulingPolicy, tier: Tier) -> u32 {
    match tier {
        Tier::Tier0 => policy.tier_capacity_units.tier0,
        Tier::Tier1 => policy.tier_capacity_units.tier1,
        Tier::Tier2 => policy.tier_capacity_units.tier2,
        Tier::Tier3 => policy.tier_capacity_units.tier3,
        Tier::Tier4 => policy.tier_capacity_units.tier4,
    }
}

pub fn execution_provider_label(provider: ExecutionProviderKind) -> &'static str {
    match provider {
        ExecutionProviderKind::Cpu => "cpu",
        ExecutionProviderKind::Metal => "metal",
        ExecutionProviderKind::Cuda => "cuda",
    }
}

pub fn device_metadata_from_capabilities(
    policy: &InferenceSchedulingPolicy,
    capabilities: &DeviceCapabilities,
) -> PlannerDeviceMetadata {
    PlannerDeviceMetadata {
        assigned_capacity_units: capacity_units_for_tier(policy, capabilities.tier),
        execution_provider: execution_provider_label(capabilities.default_execution_provider)
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::{
        ConnectivityPath, ConnectivityStatus, DeviceConnectivityState, DirectCandidateScope,
        DirectCandidateSource, DirectCandidateTransport, DirectPeerCandidate,
        InferenceSchedulingPolicy,
    };
    use crate::services::ring_manager::{ModelShard, WorkerTopologyInfo};

    fn worker(id: &str, pos: u32) -> WorkerTopologyInfo {
        worker_with_range(id, pos, pos * 10, pos * 10 + 10)
    }

    fn worker_with_range(id: &str, pos: u32, start: u32, end: u32) -> WorkerTopologyInfo {
        WorkerTopologyInfo {
            device_id: id.to_string(),
            peer_id: format!("peer-{}", id),
            position: pos,
            status: "online".to_string(),
            contributed_memory: 1024,
            shard: ModelShard {
                model_id: "model".to_string(),
                column_range: (start, end),
                estimated_memory: 1024,
            },
            left_neighbor: "a".to_string(),
            right_neighbor: "b".to_string(),
            connectivity_state: Some(DeviceConnectivityState {
                active_path: ConnectivityPath::Direct,
                active_endpoint: Some("tcp://x".to_string()),
                status: ConnectivityStatus::Connected,
            }),
            listen_addrs: vec!["dataplane://127.0.0.1:9000".to_string()],
            direct_candidates: vec![DirectPeerCandidate {
                endpoint: "/ip4/127.0.0.1/tcp/9000".to_string(),
                transport: DirectCandidateTransport::Tcp,
                scope: DirectCandidateScope::Loopback,
                source: DirectCandidateSource::LocalListen,
                priority: 0,
                last_updated_ms: 1,
            }],
        }
    }

    #[test]
    fn planner_builds_authoritative_group() {
        let plan = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![worker("a", 0), worker("b", 1)],
                ring_stable: true,
                peer_punch_plans: vec![crate::services::ring_manager::PeerPunchPlan {
                    source_device_id: "a".to_string(),
                    target_device_id: "b".to_string(),
                    target_peer_id: "peer-b".to_string(),
                    strategy: crate::services::ring_manager::PunchPathStrategy::SimultaneousDial,
                    reason: crate::services::ring_manager::PunchPathReason::RelayPath,
                    relay_rendezvous_required: false,
                    attempt_window_ms: 1000,
                    issued_at_ms: 1,
                    target_candidates: vec![],
                }],
            },
            &InferenceSchedulingPolicy::default(),
            &[
                PlannerDeviceMetadata {
                    assigned_capacity_units: 4,
                    execution_provider: "metal".to_string(),
                },
                PlannerDeviceMetadata {
                    assigned_capacity_units: 8,
                    execution_provider: "cuda".to_string(),
                },
            ],
        )
        .unwrap();

        assert_eq!(plan.execution_groups.len(), 2);
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].prompt_tokens, vec![1, 2, 3]);
        assert_eq!(plan.execution_groups[0].peer_punch_plans.len(), 1);
        assert!(matches!(
            plan.execution_groups[1].kv_transfer_policy,
            KvTransferPolicy::CoLocated
        ));
    }

    #[test]
    fn planner_can_shrink_decode_group_when_fast_subset_covers_model() {
        let plan = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_range("a", 0, 0, 10),
                    worker_with_range("b", 1, 10, 20),
                    worker_with_range("c", 2, 0, 20),
                ],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[
                PlannerDeviceMetadata {
                    assigned_capacity_units: 4,
                    execution_provider: "metal".to_string(),
                },
                PlannerDeviceMetadata {
                    assigned_capacity_units: 4,
                    execution_provider: "metal".to_string(),
                },
                PlannerDeviceMetadata {
                    assigned_capacity_units: 16,
                    execution_provider: "cuda".to_string(),
                },
            ],
        )
        .unwrap();

        let decode_group = plan
            .execution_groups
            .iter()
            .find(|group| matches!(group.phase, ExecutionPhase::Decode))
            .expect("expected decode group");
        assert_eq!(decode_group.members.len(), 1);
        assert_eq!(decode_group.members[0].device_id, "c");
        assert!(matches!(
            decode_group.kv_transfer_policy,
            KvTransferPolicy::ExportOnHandoff
        ));

        let decode_segment = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Decode))
            .expect("expected decode segment");
        assert_eq!(decode_segment.participant_device_ids, vec!["c".to_string()]);
        assert_eq!(decode_segment.kv_owner_device_id, "c");
    }
}
