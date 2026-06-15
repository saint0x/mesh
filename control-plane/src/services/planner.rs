use uuid::Uuid;

use crate::api::error::{ApiError, ApiResult};
use crate::api::types::{
    ExecutionGroup, ExecutionGroupMember, ExecutionPhase, ExecutionSegment, InferenceExecutionPlan,
    InferenceRuntimeMode, KvTransferPolicy, PunchPathReason, PunchPathStrategy, RingProtocolClass,
    ShardInfo, SubmitInferenceRequest, SupportGroup, SupportGroupRole, TransportCapabilityTier,
};
use crate::connectivity::{ConnectivityPath, DeviceConnectivityState, InferenceSchedulingPolicy};
use crate::device::{DeviceCapabilities, Tier};
use crate::model_assets::{
    self, ModelManifest, ResolvedServingLayout, ServingLayoutKind, ServingLayoutRule,
};
use crate::provider::{
    BackendContractDescriptor, ExecutionProviderKind, ProviderCompatibilityClass,
};
use crate::services::ring_manager::{RingTopology, WorkerTopologyInfo};

const MIN_RUNTIME_CAPACITY_MULTIPLIER: f64 = 0.5;
const MAX_RUNTIME_CAPACITY_MULTIPLIER: f64 = 2.0;
const SMALL_MODEL_EXPANSION_THRESHOLD_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const EXPANSION_SCORE_MARGIN: i64 = 180;
const SMALL_MODEL_EXPANSION_SCORE_MARGIN: i64 = 320;

#[derive(Debug, Clone)]
pub struct PlannerDeviceMetadata {
    pub assigned_capacity_units: u32,
    pub backend_contract: BackendContractDescriptor,
    pub throughput_multiplier: f64,
    pub observed_tokens_per_second: Option<f64>,
    pub observed_deferred_ratio: Option<f64>,
    pub observed_fill_ratio: Option<f64>,
    pub instability_score: u32,
}

#[derive(Debug, Clone)]
struct ExecutionIsland {
    island_id: String,
    compatibility_class: ProviderCompatibilityClass,
    backend_contract_hash: Option<String>,
    fast_path_eligible: bool,
    protocol_class: RingProtocolClass,
    members: Vec<ExecutionGroupMember>,
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

        let available_members = topology
            .workers
            .iter()
            .zip(device_metadata.iter())
            .map(|(worker, metadata)| build_member(worker, metadata))
            .collect::<Vec<_>>();
        let runtime_mode = derive_runtime_mode(scheduling_policy, &available_members);
        let prefill_members = select_execution_members(
            &req.model_id,
            ExecutionPhase::Prefill,
            &available_members,
            runtime_mode,
        )?;
        let decode_members = select_execution_members(
            &req.model_id,
            ExecutionPhase::Decode,
            &available_members,
            runtime_mode,
        )?;
        let total_capacity_units = prefill_members
            .iter()
            .map(|member| member.assigned_capacity_units)
            .sum();
        let transport_tier = classify_transport_tier(&prefill_members);
        let kv_owner_device_id = select_kv_owner(&prefill_members);
        let prefill_participant_device_ids = prefill_members
            .iter()
            .map(|member| member.device_id.clone())
            .collect::<Vec<_>>();
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
        let prefill_island =
            classify_execution_island(&req.model_id, ExecutionPhase::Prefill, &prefill_members);
        let decode_island =
            classify_execution_island(&req.model_id, ExecutionPhase::Decode, &decode_members);
        let peer_punch_plans = topology
            .peer_punch_plans
            .iter()
            .map(map_peer_punch_plan)
            .collect::<Vec<_>>();
        let prefill_group = ExecutionGroup {
            group_id: prefill_group_id.clone(),
            execution_island_id: prefill_island.island_id.clone(),
            model_id: req.model_id.clone(),
            phase: ExecutionPhase::Prefill,
            compatibility_class: prefill_island.compatibility_class,
            backend_contract_hash: prefill_island.backend_contract_hash.clone(),
            fast_path_eligible: prefill_island.fast_path_eligible,
            protocol_class: prefill_island.protocol_class,
            transport_tier,
            kv_transfer_policy: KvTransferPolicy::CoLocated,
            total_capacity_units,
            members: prefill_members.clone(),
            peer_punch_plans: peer_punch_plans.clone(),
        };
        let decode_group = ExecutionGroup {
            group_id: decode_group_id.clone(),
            execution_island_id: decode_island.island_id.clone(),
            model_id: req.model_id.clone(),
            phase: ExecutionPhase::Decode,
            compatibility_class: decode_island.compatibility_class,
            backend_contract_hash: decode_island.backend_contract_hash.clone(),
            fast_path_eligible: decode_island.fast_path_eligible,
            protocol_class: decode_island.protocol_class,
            transport_tier: decode_transport_tier,
            kv_transfer_policy: if same_participants(&prefill_members, &decode_members) {
                KvTransferPolicy::CoLocated
            } else {
                KvTransferPolicy::CheckpointHandoff
            },
            total_capacity_units: decode_members
                .iter()
                .map(|member| member.assigned_capacity_units)
                .sum(),
            members: decode_members.clone(),
            peer_punch_plans,
        };
        let support_groups = build_support_groups(
            &plan_id,
            &req.model_id,
            &available_members,
            &prefill_members,
            &decode_members,
            &topology.peer_punch_plans,
        );
        let prefill_segment = ExecutionSegment {
            segment_id: prefill_segment_id.clone(),
            session_id: session_id.clone(),
            execution_group_id: prefill_group_id,
            execution_island_id: prefill_island.island_id,
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
            execution_island_id: decode_island.island_id,
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
            support_groups,
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

        let available_members = topology
            .workers
            .iter()
            .zip(device_metadata.iter())
            .map(|(worker, metadata)| build_member(worker, metadata))
            .collect::<Vec<_>>();
        let runtime_mode = derive_runtime_mode(scheduling_policy, &available_members);
        let model_id = phase_model_id(plan, ExecutionPhase::Decode).ok_or_else(|| {
            ApiError::Internal(format!(
                "Execution plan {} is missing a decode group",
                plan.plan_id
            ))
        })?;
        let mut decode_members = select_execution_members(
            &model_id,
            ExecutionPhase::Decode,
            &available_members,
            runtime_mode,
        )
        .map_err(|err| {
            ApiError::Conflict(format!(
                "Live topology can no longer form an execution-valid decode group: {}",
                err
            ))
        })?;
        if let Some(current_decode_members) =
            current_valid_decode_members(plan, &model_id, &available_members)?
        {
            if decode_members.len() > current_decode_members.len()
                && is_full_replica_group(&current_decode_members)
            {
                decode_members = current_decode_members;
            } else {
                let manifest = model_assets::load_model_manifest(&model_id)?;
                let selected_score = scored_candidate_topology(
                    &manifest,
                    ExecutionPhase::Decode,
                    runtime_mode,
                    &available_members,
                    &[decode_members.clone(), current_decode_members.clone()],
                    &decode_members,
                );
                let current_score = scored_candidate_topology(
                    &manifest,
                    ExecutionPhase::Decode,
                    runtime_mode,
                    &available_members,
                    &[decode_members.clone(), current_decode_members.clone()],
                    &current_decode_members,
                );
                let selected_is_expansion = decode_members.len() > current_decode_members.len();
                let required_margin =
                    if manifest.total_model_bytes <= SMALL_MODEL_EXPANSION_THRESHOLD_BYTES {
                        SMALL_MODEL_EXPANSION_SCORE_MARGIN
                    } else {
                        EXPANSION_SCORE_MARGIN
                    };
                let materially_better =
                    selected_score >= current_score.saturating_add(required_margin);
                if (selected_is_expansion
                    || !same_participants(&decode_members, &current_decode_members))
                    && !materially_better
                {
                    decode_members = current_decode_members;
                }
            }
        }
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
        let decode_island =
            classify_execution_island(&model_id, ExecutionPhase::Decode, &decode_members);

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
        decode_group.execution_island_id = decode_island.island_id.clone();
        decode_group.compatibility_class = decode_island.compatibility_class;
        decode_group.backend_contract_hash = decode_island.backend_contract_hash.clone();
        decode_group.fast_path_eligible = decode_island.fast_path_eligible;
        decode_group.protocol_class = decode_island.protocol_class;
        decode_group.transport_tier = decode_transport_tier;
        decode_group.kv_transfer_policy = if same_participants(&prefill_members, &decode_members) {
            KvTransferPolicy::CoLocated
        } else {
            KvTransferPolicy::CheckpointHandoff
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
        decode_segment.execution_island_id = decode_island.island_id;
        decode_segment.shard_owner_device_ids = decode_participant_device_ids.clone();
        decode_segment.participant_device_ids = decode_participant_device_ids;
        refreshed.support_groups = build_support_groups(
            &refreshed.plan_id,
            &model_id,
            &available_members,
            &prefill_members,
            &decode_members,
            &topology.peer_punch_plans,
        );

        Ok(refreshed)
    }
}

fn build_support_groups(
    plan_id: &str,
    model_id: &str,
    all_members: &[ExecutionGroupMember],
    prefill_members: &[ExecutionGroupMember],
    decode_members: &[ExecutionGroupMember],
    peer_punch_plans: &[crate::services::ring_manager::PeerPunchPlan],
) -> Vec<SupportGroup> {
    let mut support_groups = Vec::new();
    let decode_ids = decode_members
        .iter()
        .map(|member| member.device_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let prefill_ids = prefill_members
        .iter()
        .map(|member| member.device_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let non_decode_members = all_members
        .iter()
        .filter(|member| !decode_ids.contains(member.device_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    let kv_members = select_support_members(all_members, &non_decode_members, 2);
    support_groups.push(build_support_group(
        plan_id,
        model_id,
        SupportGroupRole::Kv,
        &kv_members,
        peer_punch_plans,
    ));

    let checkpoint_seed = union_members(&kv_members, &non_decode_members);
    support_groups.push(build_support_group(
        plan_id,
        model_id,
        SupportGroupRole::Checkpoint,
        &checkpoint_seed,
        peer_punch_plans,
    ));

    let recovery_seed = union_members(&kv_members, decode_members);
    support_groups.push(build_support_group(
        plan_id,
        model_id,
        SupportGroupRole::Recovery,
        &recovery_seed,
        peer_punch_plans,
    ));

    let overflow_seed = all_members
        .iter()
        .filter(|member| {
            !decode_ids.contains(member.device_id.as_str())
                || !prefill_ids.contains(member.device_id.as_str())
        })
        .cloned()
        .collect::<Vec<_>>();
    support_groups.push(build_support_group(
        plan_id,
        model_id,
        SupportGroupRole::Overflow,
        if overflow_seed.is_empty() {
            &non_decode_members
        } else {
            &overflow_seed
        },
        peer_punch_plans,
    ));

    support_groups
        .into_iter()
        .filter(|group| !group.members.is_empty())
        .collect()
}

fn select_support_members(
    all_members: &[ExecutionGroupMember],
    preferred_members: &[ExecutionGroupMember],
    max_members: usize,
) -> Vec<ExecutionGroupMember> {
    let mut ranked = if preferred_members.is_empty() {
        all_members.to_vec()
    } else {
        preferred_members.to_vec()
    };
    ranked.sort_by(|left, right| {
        right
            .contributed_memory
            .cmp(&left.contributed_memory)
            .then_with(|| {
                right
                    .assigned_capacity_units
                    .cmp(&left.assigned_capacity_units)
            })
            .then_with(|| left.ring_position.cmp(&right.ring_position))
    });
    ranked.truncate(max_members.max(1).min(ranked.len()));
    ranked
}

fn union_members(
    left: &[ExecutionGroupMember],
    right: &[ExecutionGroupMember],
) -> Vec<ExecutionGroupMember> {
    let mut combined = Vec::new();
    for member in left.iter().chain(right.iter()) {
        if combined
            .iter()
            .all(|existing: &ExecutionGroupMember| existing.device_id != member.device_id)
        {
            combined.push(member.clone());
        }
    }
    combined
}

fn build_support_group(
    plan_id: &str,
    model_id: &str,
    role: SupportGroupRole,
    members: &[ExecutionGroupMember],
    peer_punch_plans: &[crate::services::ring_manager::PeerPunchPlan],
) -> SupportGroup {
    let group_id = format!(
        "support:{}:{}",
        match role {
            SupportGroupRole::Kv => "kv",
            SupportGroupRole::Checkpoint => "checkpoint",
            SupportGroupRole::Recovery => "recovery",
            SupportGroupRole::Overflow => "overflow",
        },
        plan_id
    );
    let island = classify_execution_island(model_id, ExecutionPhase::Decode, members);
    let member_ids = members
        .iter()
        .map(|member| member.device_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    SupportGroup {
        group_id,
        role,
        execution_island_id: island.island_id,
        model_id: model_id.to_string(),
        compatibility_class: island.compatibility_class,
        backend_contract_hash: island.backend_contract_hash,
        fast_path_eligible: island.fast_path_eligible,
        protocol_class: island.protocol_class,
        transport_tier: classify_transport_tier(members),
        total_capacity_units: members
            .iter()
            .map(|member| member.assigned_capacity_units)
            .sum(),
        members: members.to_vec(),
        peer_punch_plans: peer_punch_plans
            .iter()
            .filter(|plan| {
                member_ids.contains(plan.source_device_id.as_str())
                    && member_ids.contains(plan.target_device_id.as_str())
            })
            .map(map_peer_punch_plan)
            .collect(),
    }
}

pub fn validate_serving_group_legality(
    model_id: &str,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
) -> ApiResult<ResolvedServingLayout> {
    model_assets::validate_execution_group(model_id, phase, members)
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
        shard_worker_position: worker.shard_worker_position,
        shard_total_workers: worker.shard_total_workers,
        left_neighbor: worker.left_neighbor.clone(),
        right_neighbor: worker.right_neighbor.clone(),
        connectivity_state: worker.connectivity_state.clone(),
        listen_addrs: worker.listen_addrs.clone(),
        direct_candidates: worker.direct_candidates.clone(),
        assigned_capacity_units: metadata.assigned_capacity_units,
        backend_contract: metadata.backend_contract.clone(),
        throughput_multiplier: Some(metadata.throughput_multiplier),
        observed_tokens_per_second: metadata.observed_tokens_per_second,
        observed_deferred_ratio: metadata.observed_deferred_ratio,
        observed_fill_ratio: metadata.observed_fill_ratio,
        instability_score: Some(metadata.instability_score),
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

fn select_execution_members(
    model_id: &str,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
) -> ApiResult<Vec<ExecutionGroupMember>> {
    let manifest = model_assets::load_model_manifest(model_id)?;
    let candidates = candidate_groups(&manifest, phase, members, runtime_mode);
    let all_members = members.to_vec();
    let mut errors = Vec::new();
    let mut valid_candidates = Vec::new();

    for candidate in candidates {
        match validate_serving_group_legality(model_id, phase, &candidate) {
            Ok(_) => valid_candidates.push(candidate),
            Err(ApiError::Conflict(err)) => errors.push(err),
            Err(err) => return Err(err),
        }
    }

    if let Some(best) = select_best_valid_candidate(
        &manifest,
        phase,
        runtime_mode,
        &all_members,
        &valid_candidates,
    ) {
        return Ok(best);
    }

    Err(ApiError::Conflict(format!(
        "no execution-valid {:?} group could be formed for model {}: {}",
        phase,
        model_id,
        errors.join("; ")
    )))
}

fn select_best_valid_candidate(
    manifest: &ModelManifest,
    phase: ExecutionPhase,
    runtime_mode: InferenceRuntimeMode,
    all_members: &[ExecutionGroupMember],
    valid_candidates: &[Vec<ExecutionGroupMember>],
) -> Option<Vec<ExecutionGroupMember>> {
    valid_candidates
        .iter()
        .max_by_key(|candidate| {
            scored_candidate_topology(
                manifest,
                phase,
                runtime_mode,
                all_members,
                valid_candidates,
                candidate,
            )
        })
        .cloned()
}

fn candidate_groups(
    manifest: &ModelManifest,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
) -> Vec<Vec<ExecutionGroupMember>> {
    let mut candidates = Vec::new();
    for island in build_execution_islands(&manifest.model_id, phase, members) {
        for rule in manifest.layout_rules_for_phase(phase) {
            for candidate in candidates_for_rule(manifest, &island.members, runtime_mode, &rule) {
                push_unique_candidate(&mut candidates, candidate);
            }
        }
    }

    if matches!(phase, ExecutionPhase::Decode)
        && matches!(runtime_mode, InferenceRuntimeMode::ThroughputFirst)
    {
        candidates.sort_by(|left, right| {
            candidate_topology_order_key(phase, runtime_mode, right)
                .cmp(&candidate_topology_order_key(phase, runtime_mode, left))
        });
    }

    candidates
}

fn build_execution_islands(
    model_id: &str,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
) -> Vec<ExecutionIsland> {
    if members.is_empty() {
        return Vec::new();
    }

    let mut islands = Vec::new();
    let mut grouped: std::collections::BTreeMap<
        (ProviderCompatibilityClass, String),
        Vec<ExecutionGroupMember>,
    > = std::collections::BTreeMap::new();
    for member in members {
        if member.backend_contract.fast_path_eligible {
            grouped
                .entry((
                    member.backend_contract.compatibility_class,
                    member.backend_contract.contract_hash.clone(),
                ))
                .or_default()
                .push(member.clone());
        }
    }

    for ((compatibility_class, contract_hash), grouped_members) in grouped {
        islands.push(ExecutionIsland {
            island_id: execution_island_id(
                model_id,
                phase,
                compatibility_class,
                Some(contract_hash.as_str()),
            ),
            compatibility_class,
            backend_contract_hash: Some(contract_hash),
            fast_path_eligible: true,
            protocol_class: RingProtocolClass::ProviderHomogeneousFastRing,
            members: grouped_members,
        });
    }

    islands.push(ExecutionIsland {
        island_id: execution_island_id(
            model_id,
            phase,
            ProviderCompatibilityClass::HeterogeneousPortable,
            None,
        ),
        compatibility_class: ProviderCompatibilityClass::HeterogeneousPortable,
        backend_contract_hash: None,
        fast_path_eligible: false,
        protocol_class: RingProtocolClass::ProviderHeterogeneousPortableRing,
        members: members.to_vec(),
    });

    islands
}

fn execution_island_id(
    model_id: &str,
    phase: ExecutionPhase,
    compatibility_class: ProviderCompatibilityClass,
    contract_hash: Option<&str>,
) -> String {
    let phase = match phase {
        ExecutionPhase::Prefill => "prefill",
        ExecutionPhase::Decode => "decode",
    };
    match contract_hash {
        Some(contract_hash) => format!(
            "island:{}:{}:{:?}:{}",
            model_id, phase, compatibility_class, contract_hash
        )
        .to_ascii_lowercase(),
        None => format!("island:{}:{}:heterogeneous_portable", model_id, phase),
    }
}

fn classify_execution_island(
    model_id: &str,
    phase: ExecutionPhase,
    members: &[ExecutionGroupMember],
) -> ExecutionIsland {
    let unique_hashes = members
        .iter()
        .map(|member| member.backend_contract.contract_hash.clone())
        .collect::<std::collections::BTreeSet<_>>();
    if unique_hashes.len() == 1
        && members
            .iter()
            .all(|member| member.backend_contract.fast_path_eligible)
    {
        let first = &members[0].backend_contract;
        return ExecutionIsland {
            island_id: execution_island_id(
                model_id,
                phase,
                first.compatibility_class,
                Some(first.contract_hash.as_str()),
            ),
            compatibility_class: first.compatibility_class,
            backend_contract_hash: Some(first.contract_hash.clone()),
            fast_path_eligible: true,
            protocol_class: RingProtocolClass::ProviderHomogeneousFastRing,
            members: members.to_vec(),
        };
    }

    ExecutionIsland {
        island_id: execution_island_id(
            model_id,
            phase,
            ProviderCompatibilityClass::HeterogeneousPortable,
            None,
        ),
        compatibility_class: ProviderCompatibilityClass::HeterogeneousPortable,
        backend_contract_hash: None,
        fast_path_eligible: false,
        protocol_class: RingProtocolClass::ProviderHeterogeneousPortableRing,
        members: members.to_vec(),
    }
}

fn candidates_for_rule(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
    rule: &ServingLayoutRule,
) -> Vec<Vec<ExecutionGroupMember>> {
    match rule.layout {
        ServingLayoutKind::TensorParallel => {
            tensor_parallel_candidates(manifest, members, rule.tensor_parallel_degree)
        }
        ServingLayoutKind::FullReplica => {
            full_replica_candidates(manifest, members, runtime_mode, rule)
        }
        ServingLayoutKind::Pipeline => pipeline_candidates(manifest, members, rule),
        ServingLayoutKind::TensorPipelineHybrid => {
            tensor_pipeline_hybrid_candidates(manifest, members, rule)
        }
    }
}

fn tensor_parallel_candidates(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    exact_degree: Option<u32>,
) -> Vec<Vec<ExecutionGroupMember>> {
    let mut candidates = Vec::new();
    if let Some(group) = canonical_tensor_parallel_group(members, manifest.tensor_parallelism_dim) {
        if exact_degree.is_none() || exact_degree == Some(group.len() as u32) {
            candidates.push(group);
        }
    }
    candidates
}

fn full_replica_candidates(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
    rule: &ServingLayoutRule,
) -> Vec<Vec<ExecutionGroupMember>> {
    let replicas = ranked_full_replica_members(members, manifest.tensor_parallelism_dim);
    if replicas.is_empty() {
        return Vec::new();
    }
    let sizes = ordered_candidate_sizes(
        replicas.len(),
        rule.min_members,
        rule.max_members,
        runtime_mode,
    );
    sizes
        .into_iter()
        .filter(|size| *size > 0 && *size <= replicas.len())
        .map(|size| replicas.iter().take(size).cloned().collect::<Vec<_>>())
        .collect()
}

fn pipeline_candidates(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    rule: &ServingLayoutRule,
) -> Vec<Vec<ExecutionGroupMember>> {
    let Some(pipeline_degree) = rule.pipeline_parallel_degree.map(|value| value as usize) else {
        return Vec::new();
    };
    let replicas = sorted_full_replica_members(members, manifest.tensor_parallelism_dim);
    contiguous_windows(&replicas, pipeline_degree)
}

fn tensor_pipeline_hybrid_candidates(
    manifest: &ModelManifest,
    members: &[ExecutionGroupMember],
    rule: &ServingLayoutRule,
) -> Vec<Vec<ExecutionGroupMember>> {
    let Some(pipeline_degree) = rule.pipeline_parallel_degree.map(|value| value as usize) else {
        return Vec::new();
    };
    let Some(tensor_degree) = rule.tensor_parallel_degree.map(|value| value as usize) else {
        return Vec::new();
    };
    let total_members = pipeline_degree.saturating_mul(tensor_degree);
    if total_members == 0 {
        return Vec::new();
    }
    let mut sorted = members.to_vec();
    sorted.sort_by_key(|member| member.ring_position);

    contiguous_windows(&sorted, total_members)
        .into_iter()
        .filter(|candidate| {
            (0..pipeline_degree).all(|stage_index| {
                let start = stage_index * tensor_degree;
                let end = start + tensor_degree;
                stage_chunk_covers_tensor_parallel_span(
                    &candidate[start..end],
                    manifest.tensor_parallelism_dim,
                )
            })
        })
        .collect()
}

fn ordered_candidate_sizes(
    available: usize,
    min_members: Option<u32>,
    max_members: Option<u32>,
    runtime_mode: InferenceRuntimeMode,
) -> Vec<usize> {
    let min = min_members.unwrap_or(1) as usize;
    let max = max_members.unwrap_or(available as u32) as usize;
    let upper = max.min(available);
    if min > upper {
        return Vec::new();
    }
    let mut sizes = (min..=upper).collect::<Vec<_>>();
    match runtime_mode {
        InferenceRuntimeMode::ThroughputFirst => sizes.sort_unstable_by(|a, b| b.cmp(a)),
        InferenceRuntimeMode::FitFirst | InferenceRuntimeMode::ResilientEdge => {
            sizes.sort_unstable()
        }
        InferenceRuntimeMode::LatencyFirst => sizes.sort_unstable(),
    }
    sizes
}

fn sorted_full_replica_members(
    members: &[ExecutionGroupMember],
    tensor_parallelism_dim: u32,
) -> Vec<ExecutionGroupMember> {
    let mut replicas = all_full_replica_members(members, tensor_parallelism_dim);
    replicas.sort_by_key(|member| member.ring_position);
    replicas
}

fn contiguous_windows(
    members: &[ExecutionGroupMember],
    width: usize,
) -> Vec<Vec<ExecutionGroupMember>> {
    if width == 0 || members.len() < width {
        return Vec::new();
    }
    (0..=members.len() - width)
        .map(|start| members[start..start + width].to_vec())
        .collect()
}

fn stage_chunk_covers_tensor_parallel_span(
    members: &[ExecutionGroupMember],
    tensor_parallelism_dim: u32,
) -> bool {
    let mut intervals = members
        .iter()
        .map(|member| (member.shard.column_start, member.shard.column_end))
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|(start, end)| (*start, *end));
    let mut cursor = 0;
    for (start, end) in intervals {
        if start != cursor || end <= start {
            return false;
        }
        cursor = end;
    }
    cursor == tensor_parallelism_dim
}

fn push_unique_candidate(
    candidates: &mut Vec<Vec<ExecutionGroupMember>>,
    mut candidate: Vec<ExecutionGroupMember>,
) {
    if candidate.is_empty() {
        return;
    }
    candidate.sort_by_key(|member| member.ring_position);

    let candidate_ids = candidate
        .iter()
        .map(|member| member.device_id.as_str())
        .collect::<Vec<_>>();
    if candidates.iter().any(|existing| {
        existing
            .iter()
            .map(|member| member.device_id.as_str())
            .collect::<Vec<_>>()
            == candidate_ids
    }) {
        return;
    }

    candidates.push(candidate);
}

fn rank_members(members: &[ExecutionGroupMember]) -> Vec<ExecutionGroupMember> {
    let mut ranked = members.to_vec();
    ranked.sort_by(|left, right| {
        right
            .assigned_capacity_units
            .cmp(&left.assigned_capacity_units)
            .then_with(|| right.contributed_memory.cmp(&left.contributed_memory))
            .then_with(|| left.ring_position.cmp(&right.ring_position))
    });
    ranked
}

fn ranked_full_replica_members(
    members: &[ExecutionGroupMember],
    tensor_parallelism_dim: u32,
) -> Vec<ExecutionGroupMember> {
    rank_members(&all_full_replica_members(members, tensor_parallelism_dim))
}

fn all_full_replica_members(
    members: &[ExecutionGroupMember],
    tensor_parallelism_dim: u32,
) -> Vec<ExecutionGroupMember> {
    members
        .iter()
        .filter(|member| {
            member.shard.column_start == 0 && member.shard.column_end == tensor_parallelism_dim
        })
        .cloned()
        .collect()
}

fn canonical_tensor_parallel_group(
    members: &[ExecutionGroupMember],
    tensor_parallelism_dim: u32,
) -> Option<Vec<ExecutionGroupMember>> {
    let mut ranked = members.to_vec();
    ranked.sort_by(|left, right| {
        left.shard
            .column_start
            .cmp(&right.shard.column_start)
            .then_with(|| left.shard.column_end.cmp(&right.shard.column_end))
            .then_with(|| {
                right
                    .assigned_capacity_units
                    .cmp(&left.assigned_capacity_units)
            })
            .then_with(|| right.contributed_memory.cmp(&left.contributed_memory))
    });

    let mut cursor = 0;
    let mut selected = Vec::new();
    while cursor < tensor_parallelism_dim {
        let next = ranked
            .iter()
            .find(|member| member.shard.column_start == cursor && member.shard.column_end > cursor)?
            .clone();
        cursor = next.shard.column_end;
        selected.push(next);
    }

    Some(selected)
}

fn phase_model_id(plan: &InferenceExecutionPlan, phase: ExecutionPhase) -> Option<String> {
    plan.execution_groups
        .iter()
        .find(|group| group.phase == phase)
        .map(|group| group.model_id.clone())
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

fn is_full_replica_group(members: &[ExecutionGroupMember]) -> bool {
    let Some(max_column_end) = members.iter().map(|member| member.shard.column_end).max() else {
        return false;
    };
    members
        .iter()
        .all(|member| member.shard.column_start == 0 && member.shard.column_end == max_column_end)
}

fn current_valid_decode_members(
    plan: &InferenceExecutionPlan,
    model_id: &str,
    available_members: &[ExecutionGroupMember],
) -> ApiResult<Option<Vec<ExecutionGroupMember>>> {
    let Some(existing_decode_group) = plan
        .execution_groups
        .iter()
        .find(|group| matches!(group.phase, ExecutionPhase::Decode))
    else {
        return Ok(None);
    };
    let current_members = existing_decode_group
        .members
        .iter()
        .map(|member| {
            available_members
                .iter()
                .find(|candidate| candidate.device_id == member.device_id)
                .cloned()
        })
        .collect::<Option<Vec<_>>>();
    let Some(current_members) = current_members else {
        return Ok(None);
    };
    if validate_serving_group_legality(model_id, ExecutionPhase::Decode, &current_members).is_err()
    {
        return Ok(None);
    }
    Ok(Some(current_members))
}

fn derive_runtime_mode(
    _scheduling_policy: &InferenceSchedulingPolicy,
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

    let min_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1))
        .min()
        .unwrap_or(1);
    let max_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1))
        .max()
        .unwrap_or(min_capacity);
    let wide_capacity_spread = max_capacity >= min_capacity.saturating_mul(4).max(4);
    if wide_capacity_spread {
        InferenceRuntimeMode::ThroughputFirst
    } else {
        InferenceRuntimeMode::LatencyFirst
    }
}

pub fn adjusted_capacity_units(base_units: u32, throughput_multiplier: f64) -> u32 {
    let adjusted = (base_units.max(1) as f64)
        * throughput_multiplier.clamp(
            MIN_RUNTIME_CAPACITY_MULTIPLIER,
            MAX_RUNTIME_CAPACITY_MULTIPLIER,
        );
    adjusted.round().max(1.0) as u32
}

pub fn topology_efficiency_score(
    members: &[ExecutionGroupMember],
    runtime_mode: InferenceRuntimeMode,
) -> i64 {
    if members.is_empty() {
        return i64::MIN / 4;
    }

    let width = members.len() as i64;
    let total_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1) as i64)
        .sum::<i64>();
    let min_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1) as i64)
        .min()
        .unwrap_or(1);
    let max_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1) as i64)
        .max()
        .unwrap_or(min_capacity);
    let aligned_capacity = min_capacity * width;
    let excess_capacity = total_capacity.saturating_sub(aligned_capacity);
    let transport_tier = classify_transport_tier(members);
    let risk = collective_scale_risk_score(members, transport_tier);
    let excess_divisor = if width <= 2 {
        2
    } else {
        4 + i64::from(risk / 25)
    };
    let effective_capacity = aligned_capacity + (excess_capacity / excess_divisor.max(1));
    let transport_bonus = match transport_tier {
        TransportCapabilityTier::DirectPreferred => 160,
        TransportCapabilityTier::DirectTcp => 80,
        TransportCapabilityTier::RelayFallback => 0,
    };
    let balance_pct = ((min_capacity * 100) / max_capacity.max(1)).clamp(1, 100);
    let width_penalty = match runtime_mode {
        InferenceRuntimeMode::ThroughputFirst => {
            if width <= 2 {
                0
            } else {
                (width - 2) * 35
            }
        }
        InferenceRuntimeMode::LatencyFirst => (width - 1).max(0) * 70,
        InferenceRuntimeMode::FitFirst => (width - 1).max(0) * 45,
        InferenceRuntimeMode::ResilientEdge => (width - 1).max(0) * 20,
    };

    effective_capacity * 100 + transport_bonus + balance_pct - i64::from(risk) - width_penalty
}

fn scored_candidate_topology(
    manifest: &ModelManifest,
    phase: ExecutionPhase,
    runtime_mode: InferenceRuntimeMode,
    all_members: &[ExecutionGroupMember],
    valid_candidates: &[Vec<ExecutionGroupMember>],
    candidate: &[ExecutionGroupMember],
) -> i64 {
    let mut score = topology_efficiency_score(candidate, runtime_mode);
    if !matches!(phase, ExecutionPhase::Decode) {
        return score;
    }

    let width = candidate.len();
    let narrower_candidates = valid_candidates
        .iter()
        .filter(|other| other.len() < width)
        .collect::<Vec<_>>();
    let best_narrower_score = narrower_candidates
        .iter()
        .map(|other| topology_efficiency_score(other, runtime_mode))
        .max();
    let best_homogeneous_score = valid_candidates
        .iter()
        .filter(|other| is_homogeneous_provider_group(other))
        .map(|other| topology_efficiency_score(other, runtime_mode))
        .max();
    let small_model = manifest.total_model_bytes <= SMALL_MODEL_EXPANSION_THRESHOLD_BYTES;
    let wider_than_needed = narrower_candidates.iter().any(|other| {
        other.len() < width && covers_same_decode_shape(other, candidate, all_members)
    });

    if small_model && width > 2 && wider_than_needed {
        score -= ((width as i64) - 2) * 180;
    }

    if let Some(narrower_score) = best_narrower_score {
        let required_margin = if small_model {
            SMALL_MODEL_EXPANSION_SCORE_MARGIN
        } else {
            EXPANSION_SCORE_MARGIN
        };
        if width > 1 && wider_than_needed && score < narrower_score.saturating_add(required_margin)
        {
            score -= 5_000;
        }
    }

    if !is_homogeneous_provider_group(candidate)
        && wider_than_needed
        && best_homogeneous_score
            .map(|best| best.saturating_add(120) >= score)
            .unwrap_or(false)
    {
        score -= 2_000;
    }

    score
}

fn covers_same_decode_shape(
    candidate: &[ExecutionGroupMember],
    reference: &[ExecutionGroupMember],
    all_members: &[ExecutionGroupMember],
) -> bool {
    let candidate_full_replica = candidate.iter().all(|member| {
        member.shard.column_start == 0
            && member.shard.column_end
                == candidate
                    .iter()
                    .map(|item| item.shard.column_end)
                    .max()
                    .unwrap_or_default()
    });
    let reference_full_replica = reference.iter().all(|member| {
        member.shard.column_start == 0
            && member.shard.column_end
                == reference
                    .iter()
                    .map(|item| item.shard.column_end)
                    .max()
                    .unwrap_or_default()
    });
    if candidate_full_replica && reference_full_replica {
        return true;
    }

    let candidate_total = candidate
        .iter()
        .map(|member| {
            member
                .shard
                .column_end
                .saturating_sub(member.shard.column_start)
        })
        .sum::<u32>();
    let reference_total = reference
        .iter()
        .map(|member| {
            member
                .shard
                .column_end
                .saturating_sub(member.shard.column_start)
        })
        .sum::<u32>();
    let full_span = all_members
        .iter()
        .map(|member| member.shard.column_end)
        .max()
        .unwrap_or_default();

    candidate_total >= full_span && reference_total >= full_span
}

fn candidate_topology_order_key(
    _phase: ExecutionPhase,
    runtime_mode: InferenceRuntimeMode,
    members: &[ExecutionGroupMember],
) -> (i64, u32, i64, i64) {
    let score = topology_efficiency_score(members, runtime_mode);
    let total_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1))
        .sum::<u32>();
    let width_bias = -(members.len() as i64);
    let transport_bias = match classify_transport_tier(members) {
        TransportCapabilityTier::DirectPreferred => 2,
        TransportCapabilityTier::DirectTcp => 1,
        TransportCapabilityTier::RelayFallback => 0,
    };
    (score, transport_bias, total_capacity as i64, width_bias)
}

fn collective_scale_risk_score(
    members: &[ExecutionGroupMember],
    transport_tier: TransportCapabilityTier,
) -> u32 {
    if members.is_empty() {
        return 100;
    }

    let width = members.len() as u32;
    let min_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1))
        .min()
        .unwrap_or(1);
    let max_capacity = members
        .iter()
        .map(|member| member.assigned_capacity_units.max(1))
        .max()
        .unwrap_or(min_capacity);
    let degraded_members = members
        .iter()
        .filter(|member| {
            member
                .connectivity_state
                .as_ref()
                .map(is_relay_or_degraded)
                .unwrap_or(true)
        })
        .count() as u32;
    let mut providers = members
        .iter()
        .map(|member| member.backend_contract.contract_hash.as_str())
        .collect::<Vec<_>>();
    providers.sort_unstable();
    providers.dedup();
    let provider_mix_penalty = providers.len().saturating_sub(1) as u32 * 15;
    let transport_penalty = match transport_tier {
        TransportCapabilityTier::DirectPreferred => 0,
        TransportCapabilityTier::DirectTcp => 15,
        TransportCapabilityTier::RelayFallback => 40,
    };
    let width_penalty = width.saturating_sub(2) * 12;
    let balance_penalty = if max_capacity == 0 {
        0
    } else {
        100u32.saturating_sub((min_capacity.saturating_mul(100) / max_capacity).max(1))
    };
    let instability_penalty = members
        .iter()
        .map(|member| member.instability_score.unwrap_or_default())
        .sum::<u32>();
    let throughput_penalty = throughput_spread_penalty(members);
    let deferral_penalty = average_ratio_penalty(
        members
            .iter()
            .filter_map(|member| member.observed_deferred_ratio)
            .collect::<Vec<_>>()
            .as_slice(),
        120.0,
    );
    let fill_penalty = if width > 2 {
        average_fill_penalty(
            &members
                .iter()
                .filter_map(|member| member.observed_fill_ratio)
                .collect::<Vec<_>>(),
        )
    } else {
        0
    };

    width_penalty
        .saturating_add(balance_penalty)
        .saturating_add(degraded_members.saturating_mul(25))
        .saturating_add(provider_mix_penalty)
        .saturating_add(transport_penalty)
        .saturating_add(instability_penalty)
        .saturating_add(throughput_penalty)
        .saturating_add(deferral_penalty)
        .saturating_add(fill_penalty)
}

fn throughput_spread_penalty(members: &[ExecutionGroupMember]) -> u32 {
    let mut multipliers = members
        .iter()
        .filter_map(|member| member.throughput_multiplier)
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if multipliers.len() < 2 {
        return 0;
    }
    multipliers.sort_by(|left, right| left.total_cmp(right));
    let min = multipliers[0];
    let max = *multipliers.last().unwrap_or(&min);
    if max <= f64::EPSILON {
        return 0;
    }
    (((1.0 - (min / max).clamp(0.0, 1.0)) * 90.0).round() as i64).max(0) as u32
}

fn average_ratio_penalty(values: &[f64], scale: f64) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let average = values.iter().copied().sum::<f64>() / values.len() as f64;
    (average.clamp(0.0, 1.0) * scale).round().max(0.0) as u32
}

fn average_fill_penalty(values: &[f64]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let average = values.iter().copied().sum::<f64>() / values.len() as f64;
    ((1.0 - average.clamp(0.0, 1.0)) * 140.0).round().max(0.0) as u32
}

fn is_homogeneous_provider_group(members: &[ExecutionGroupMember]) -> bool {
    members
        .first()
        .map(|first| {
            members.iter().all(|member| {
                member.backend_contract.contract_hash == first.backend_contract.contract_hash
            })
        })
        .unwrap_or(true)
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
    provider.as_str()
}

pub fn device_metadata_from_capabilities(
    policy: &InferenceSchedulingPolicy,
    capabilities: &DeviceCapabilities,
) -> PlannerDeviceMetadata {
    PlannerDeviceMetadata {
        assigned_capacity_units: capacity_units_for_tier(policy, capabilities.tier),
        backend_contract: capabilities.default_backend_contract(),
        throughput_multiplier: 1.0,
        observed_tokens_per_second: None,
        observed_deferred_ratio: None,
        observed_fill_ratio: None,
        instability_score: 0,
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
    use crate::model_assets::{
        self, ModelExecutionLayout, ModelManifest, PipelineStageLayerRange, ProviderConstraints,
        ServingLayoutKind, ServingLayoutRule,
    };
    use crate::services::ring_manager::{ModelShard, WorkerTopologyInfo};

    fn worker_with_model_range(
        model_id: &str,
        id: &str,
        pos: u32,
        start: u32,
        end: u32,
    ) -> WorkerTopologyInfo {
        WorkerTopologyInfo {
            device_id: id.to_string(),
            peer_id: format!("peer-{}", id),
            position: pos,
            status: "online".to_string(),
            contributed_memory: 1024,
            shard: ModelShard {
                model_id: model_id.to_string(),
                column_range: (start, end),
                estimated_memory: 1024,
            },
            shard_worker_position: pos,
            shard_total_workers: 4,
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

    fn planner_metadata(units: u32, provider: &str) -> PlannerDeviceMetadata {
        let provider = match provider {
            "cpu" => ExecutionProviderKind::Cpu,
            "metal" => ExecutionProviderKind::Metal,
            "cuda" => ExecutionProviderKind::Cuda,
            other => panic!("unknown provider {other}"),
        };
        PlannerDeviceMetadata {
            assigned_capacity_units: units,
            backend_contract: BackendContractDescriptor::for_provider(provider),
            throughput_multiplier: 1.0,
            observed_tokens_per_second: None,
            observed_deferred_ratio: None,
            observed_fill_ratio: None,
            instability_score: 0,
        }
    }

    #[test]
    fn planner_builds_authoritative_group() {
        model_assets::testsupport::ensure_test_model("planner-authoritative", 20);
        model_assets::clear_model_asset_cache();

        let plan = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "planner-authoritative".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_model_range("planner-authoritative", "a", 0, 0, 10),
                    worker_with_model_range("planner-authoritative", "b", 1, 10, 20),
                ],
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
            &[planner_metadata(4, "metal"), planner_metadata(8, "cuda")],
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
        model_assets::testsupport::ensure_test_model("planner-shrink", 20);
        model_assets::clear_model_asset_cache();

        let plan = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "planner-shrink".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_model_range("planner-shrink", "a", 0, 0, 10),
                    worker_with_model_range("planner-shrink", "b", 1, 10, 20),
                    worker_with_model_range("planner-shrink", "c", 2, 0, 20),
                ],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[
                planner_metadata(4, "metal"),
                planner_metadata(4, "metal"),
                planner_metadata(16, "cuda"),
            ],
        )
        .unwrap();

        let decode_group = plan
            .execution_groups
            .iter()
            .find(|group| matches!(group.phase, ExecutionPhase::Decode))
            .expect("expected decode group");
        let prefill_group = plan
            .execution_groups
            .iter()
            .find(|group| matches!(group.phase, ExecutionPhase::Prefill))
            .expect("expected prefill group");
        assert_eq!(prefill_group.members.len(), 1);
        assert_eq!(
            prefill_group
                .members
                .iter()
                .map(|member| member.device_id.as_str())
                .collect::<Vec<_>>(),
            vec!["c"]
        );
        assert_eq!(decode_group.members.len(), 1);
        assert_eq!(decode_group.members[0].device_id, "c");
        assert!(matches!(
            decode_group.kv_transfer_policy,
            KvTransferPolicy::CoLocated
        ));
        assert_eq!(
            decode_group.backend_contract_hash,
            prefill_group.backend_contract_hash
        );
        assert_eq!(
            decode_group.compatibility_class,
            prefill_group.compatibility_class
        );

        let decode_segment = plan
            .segments
            .iter()
            .find(|segment| matches!(segment.phase, ExecutionPhase::Decode))
            .expect("expected decode segment");
        assert_eq!(decode_segment.participant_device_ids, vec!["c".to_string()]);
        assert_eq!(decode_segment.kv_owner_device_id, "c");
    }

    #[test]
    fn planner_refuses_decode_group_when_manifest_provider_constraints_fail() {
        model_assets::testsupport::ensure_test_model_with_manifest(ModelManifest {
            model_id: "planner-provider-guard".to_string(),
            tensor_parallelism_dim: 20,
            total_model_bytes: 1024 * 1024,
            transformer_layer_count: None,
            tokenizer_file: "tokenizer.json".to_string(),
            tokenizer_config_file: "tokenizer_config.json".to_string(),
            execution_layout: ModelExecutionLayout {
                prefill: vec![ServingLayoutRule {
                    variant_name: Some("prefill_tp".to_string()),
                    layout: ServingLayoutKind::TensorParallel,
                    min_members: None,
                    max_members: None,
                    tensor_parallel_degree: Some(2),
                    pipeline_parallel_degree: None,
                    pipeline_stages: Vec::new(),
                    provider_constraints: ProviderConstraints::default(),
                }],
                decode: vec![ServingLayoutRule {
                    variant_name: Some("decode_cuda_replica".to_string()),
                    layout: ServingLayoutKind::FullReplica,
                    min_members: Some(1),
                    max_members: Some(1),
                    tensor_parallel_degree: None,
                    pipeline_parallel_degree: None,
                    pipeline_stages: Vec::new(),
                    provider_constraints: ProviderConstraints {
                        allowed_providers: vec!["cuda".to_string()],
                        requires_homogeneous: false,
                    },
                }],
            },
        });
        model_assets::clear_model_asset_cache();

        let err = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "planner-provider-guard".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_model_range("planner-provider-guard", "a", 0, 0, 10),
                    worker_with_model_range("planner-provider-guard", "b", 1, 10, 20),
                    worker_with_model_range("planner-provider-guard", "c", 2, 0, 20),
                ],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[
                planner_metadata(4, "metal"),
                planner_metadata(4, "metal"),
                planner_metadata(16, "metal"),
            ],
        )
        .unwrap_err();

        match err {
            ApiError::Conflict(message) => {
                assert!(message.contains("allowed providers are cuda"));
            }
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn planner_supports_hybrid_layouts_with_pipeline_partition_metadata() {
        model_assets::testsupport::ensure_test_model_with_manifest(ModelManifest {
            model_id: "planner-hybrid-supported".to_string(),
            tensor_parallelism_dim: 20,
            total_model_bytes: 1024 * 1024,
            transformer_layer_count: Some(24),
            tokenizer_file: "tokenizer.json".to_string(),
            tokenizer_config_file: "tokenizer_config.json".to_string(),
            execution_layout: ModelExecutionLayout {
                prefill: vec![ServingLayoutRule {
                    variant_name: Some("prefill_hybrid".to_string()),
                    layout: ServingLayoutKind::TensorPipelineHybrid,
                    min_members: None,
                    max_members: None,
                    tensor_parallel_degree: Some(2),
                    pipeline_parallel_degree: Some(2),
                    pipeline_stages: vec![
                        PipelineStageLayerRange {
                            stage_index: 0,
                            layer_start: 0,
                            layer_end: 12,
                        },
                        PipelineStageLayerRange {
                            stage_index: 1,
                            layer_start: 12,
                            layer_end: 24,
                        },
                    ],
                    provider_constraints: ProviderConstraints::default(),
                }],
                decode: vec![ServingLayoutRule {
                    variant_name: Some("decode_pipeline".to_string()),
                    layout: ServingLayoutKind::Pipeline,
                    min_members: Some(2),
                    max_members: Some(2),
                    tensor_parallel_degree: None,
                    pipeline_parallel_degree: Some(2),
                    pipeline_stages: vec![
                        PipelineStageLayerRange {
                            stage_index: 0,
                            layer_start: 0,
                            layer_end: 12,
                        },
                        PipelineStageLayerRange {
                            stage_index: 1,
                            layer_start: 12,
                            layer_end: 24,
                        },
                    ],
                    provider_constraints: ProviderConstraints::default(),
                }],
            },
        });
        model_assets::clear_model_asset_cache();

        let plan = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "planner-hybrid-supported".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_model_range("planner-hybrid-supported", "a", 0, 0, 10),
                    worker_with_model_range("planner-hybrid-supported", "b", 1, 10, 20),
                    worker_with_model_range("planner-hybrid-supported", "c", 2, 0, 10),
                    worker_with_model_range("planner-hybrid-supported", "d", 3, 10, 20),
                    worker_with_model_range("planner-hybrid-supported", "e", 4, 0, 20),
                    worker_with_model_range("planner-hybrid-supported", "f", 5, 0, 20),
                ],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[
                planner_metadata(8, "cuda"),
                planner_metadata(8, "cuda"),
                planner_metadata(16, "cuda"),
                planner_metadata(16, "cuda"),
                planner_metadata(24, "cuda"),
                planner_metadata(20, "cuda"),
            ],
        )
        .unwrap();

        assert_eq!(plan.execution_groups.len(), 2);
        assert_eq!(plan.execution_groups[0].members.len(), 4);
        assert_eq!(plan.execution_groups[1].members.len(), 2);
        assert_eq!(
            plan.execution_groups[1]
                .members
                .iter()
                .map(|member| member.device_id.as_str())
                .collect::<Vec<_>>(),
            vec!["e", "f"]
        );
    }

    #[test]
    fn planner_selects_valid_provider_constrained_variant_when_first_variant_is_illegal() {
        model_assets::testsupport::ensure_test_model_with_manifest(ModelManifest {
            model_id: "planner-provider-variant".to_string(),
            tensor_parallelism_dim: 20,
            total_model_bytes: 1024 * 1024,
            transformer_layer_count: None,
            tokenizer_file: "tokenizer.json".to_string(),
            tokenizer_config_file: "tokenizer_config.json".to_string(),
            execution_layout: ModelExecutionLayout {
                prefill: vec![ServingLayoutRule {
                    variant_name: Some("prefill_tp".to_string()),
                    layout: ServingLayoutKind::TensorParallel,
                    min_members: None,
                    max_members: None,
                    tensor_parallel_degree: Some(2),
                    pipeline_parallel_degree: None,
                    pipeline_stages: Vec::new(),
                    provider_constraints: ProviderConstraints::default(),
                }],
                decode: vec![
                    ServingLayoutRule {
                        variant_name: Some("cuda_replica".to_string()),
                        layout: ServingLayoutKind::FullReplica,
                        min_members: Some(1),
                        max_members: Some(1),
                        tensor_parallel_degree: None,
                        pipeline_parallel_degree: None,
                        pipeline_stages: Vec::new(),
                        provider_constraints: ProviderConstraints {
                            allowed_providers: vec!["cuda".to_string()],
                            requires_homogeneous: false,
                        },
                    },
                    ServingLayoutRule {
                        variant_name: Some("metal_tp".to_string()),
                        layout: ServingLayoutKind::TensorParallel,
                        min_members: Some(2),
                        max_members: Some(2),
                        tensor_parallel_degree: Some(2),
                        pipeline_parallel_degree: None,
                        pipeline_stages: Vec::new(),
                        provider_constraints: ProviderConstraints {
                            allowed_providers: vec!["metal".to_string()],
                            requires_homogeneous: true,
                        },
                    },
                ],
            },
        });
        model_assets::clear_model_asset_cache();

        let plan = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "planner-provider-variant".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_model_range("planner-provider-variant", "a", 0, 0, 10),
                    worker_with_model_range("planner-provider-variant", "b", 1, 10, 20),
                    worker_with_model_range("planner-provider-variant", "c", 2, 0, 20),
                ],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[
                planner_metadata(4, "metal"),
                planner_metadata(4, "metal"),
                planner_metadata(16, "metal"),
            ],
        )
        .unwrap();

        let decode_group = plan
            .execution_groups
            .iter()
            .find(|group| matches!(group.phase, ExecutionPhase::Decode))
            .expect("expected decode group");
        assert_eq!(
            decode_group
                .members
                .iter()
                .map(|member| member.device_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn planner_rejects_invalid_pipeline_partition_metadata() {
        model_assets::testsupport::ensure_test_model_with_manifest(ModelManifest {
            model_id: "planner-invalid-pipeline".to_string(),
            tensor_parallelism_dim: 20,
            total_model_bytes: 1024 * 1024,
            transformer_layer_count: Some(24),
            tokenizer_file: "tokenizer.json".to_string(),
            tokenizer_config_file: "tokenizer_config.json".to_string(),
            execution_layout: ModelExecutionLayout {
                prefill: vec![ServingLayoutRule {
                    variant_name: Some("bad_pipeline".to_string()),
                    layout: ServingLayoutKind::TensorPipelineHybrid,
                    min_members: None,
                    max_members: None,
                    tensor_parallel_degree: Some(2),
                    pipeline_parallel_degree: Some(2),
                    pipeline_stages: vec![
                        PipelineStageLayerRange {
                            stage_index: 0,
                            layer_start: 0,
                            layer_end: 8,
                        },
                        PipelineStageLayerRange {
                            stage_index: 1,
                            layer_start: 10,
                            layer_end: 24,
                        },
                    ],
                    provider_constraints: ProviderConstraints::default(),
                }],
                decode: Vec::new(),
            },
        });
        model_assets::clear_model_asset_cache();

        let err = model_assets::load_model_manifest("planner-invalid-pipeline").unwrap_err();
        match err {
            ApiError::BadRequest(message) => {
                assert!(message.contains("starting at 10, expected 8"));
            }
            other => panic!("expected bad request, got {other:?}"),
        }
    }

    #[test]
    fn refresh_decode_plan_refuses_invalid_regroup() {
        model_assets::testsupport::ensure_test_model("planner-refresh", 20);
        model_assets::clear_model_asset_cache();

        let original = ExecutionPlanner::plan(
            &SubmitInferenceRequest {
                device_id: "submitter".to_string(),
                network_id: "net".to_string(),
                model_id: "planner-refresh".to_string(),
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.7,
                top_p: 0.9,
            },
            &[1, 2, 3],
            &RingTopology {
                workers: vec![
                    worker_with_model_range("planner-refresh", "a", 0, 0, 10),
                    worker_with_model_range("planner-refresh", "b", 1, 10, 20),
                ],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[planner_metadata(4, "metal"), planner_metadata(4, "metal")],
        )
        .unwrap();

        let err = ExecutionPlanner::refresh_decode_plan(
            &original,
            &RingTopology {
                workers: vec![worker_with_model_range("planner-refresh", "a", 0, 0, 10)],
                ring_stable: true,
                peer_punch_plans: vec![],
            },
            &InferenceSchedulingPolicy::default(),
            &[planner_metadata(4, "metal")],
        )
        .unwrap_err();

        match err {
            ApiError::Conflict(message) => {
                assert!(message.contains("execution-valid decode group"));
            }
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn topology_efficiency_penalizes_broad_imbalanced_decode_groups() {
        let narrow_members = vec![
            build_member(
                &worker_with_model_range("planner-topology-score", "a", 0, 0, 20),
                &planner_metadata(16, "cuda"),
            ),
            build_member(
                &worker_with_model_range("planner-topology-score", "b", 1, 0, 20),
                &planner_metadata(16, "cuda"),
            ),
        ];
        let broad_members = vec![
            build_member(
                &worker_with_model_range("planner-topology-score", "a", 0, 0, 20),
                &planner_metadata(32, "cuda"),
            ),
            build_member(
                &worker_with_model_range("planner-topology-score", "b", 1, 0, 20),
                &planner_metadata(8, "cuda"),
            ),
            build_member(
                &worker_with_model_range("planner-topology-score", "c", 2, 0, 20),
                &planner_metadata(8, "metal"),
            ),
            build_member(
                &worker_with_model_range("planner-topology-score", "d", 3, 0, 20),
                &planner_metadata(4, "metal"),
            ),
        ];

        let narrow_score =
            topology_efficiency_score(&narrow_members, InferenceRuntimeMode::ThroughputFirst);
        let broad_score =
            topology_efficiency_score(&broad_members, InferenceRuntimeMode::ThroughputFirst);
        assert!(
            narrow_score > broad_score,
            "expected narrow balanced topology score {narrow_score} to beat broad imbalanced score {broad_score}"
        );
    }
}
