# Zip Roadmap

This file tracks the remaining work to turn the current explicit-session,
explicit-segment serving stack into the full production `zip` engine.

## 1. Control-Plane Scheduling

- ✅ Persist authoritative execution plans at submit time
- ✅ Represent explicit prefill and decode segments
- ✅ Re-plan decode placement at prefill handoff against live pool state
- ✅ Move claim-time assignment selection behind a dedicated scheduler service
- ✅ Introduce a first-class scheduler state model for queued, runnable, and blocked jobs
- ✅ Introduce explicit serving-group ownership records rather than inferring from assignments
- ✅ Make decode scheduling honor serving-group lease ownership
- ✅ Prefer filling an already owned decode group before opening a fresh ready group
- ✅ Prefer continuing to fill an owned decode group until its pooled target is satisfied
- ✅ Open fresh ready decode groups once an owned group has reached its pooled target
- ✅ Prefer filling the most under-served owned decode cohort before shallower owned cohorts
- ✅ Prefer opening larger fresh decode cohorts before smaller fresh cohorts
- ✅ Prefer fresh decode cohorts with more immediate ready runway before equally sized but colder cohorts
- ✅ Prefer fresh decode cohorts with less transfer debt when immediate ready runway ties
- ✅ Break final fresh decode cohort ties by cohort age instead of individual session row age
- ✅ Prefer the oldest ready session inside the winning fresh decode cohort
- ✅ Lease sibling ready decode sessions in the same pooled cohort up to the scheduler target when a worker claims decode work
- ✅ Prefer draining already leased decode sessions inside an owned cohort before taking merely ready sibling sessions
- ✅ Prefer denser already owned decode cohorts before thinner owned cohorts when their remaining capacity ties
- ✅ Prefer owned decode cohorts with more immediate ready runway when remaining capacity and density tie
- ✅ Prefer owned decode cohorts with less blocked transfer debt when remaining capacity and density tie
- ✅ Prefer older owned decode cohorts before lease-count noise when cohort-specific latency signals tie
- ✅ In `throughput_first`, prefer draining already leased owned decode work before opening fresh ready decode groups
- Planned: Add admission control across multiple concurrent jobs and models
- Planned: Add topology-aware pool partitioning for prefill-heavy vs decode-heavy work
- In progress: Add scheduler scoring that reasons about queue age, latency target, throughput target, and resiliency target
- In progress: Add policy-specific behavior for `fit_first`, `throughput_first`, `latency_first`, and `resilient_edge`
- ✅ Add scheduler reconciliation loop that repairs stuck or stale ownership
- ✅ Add scheduler observability for queue depth, blocked reasons, and group utilization

## 2. Decode Queue And Continuous Batching

- ✅ Add explicit decode work queue records in the control plane
- ✅ Add decode-step ownership, lease semantics, and batch lease targets
- ✅ Make the agent runtime honor scheduler-issued decode batch targets
- ✅ Pool decode batch targets across sessions that share the same serving participants
- ✅ Persist target-versus-actual decode batch event history
- ✅ Expose pooled decode cohort visibility on lease and session status surfaces
- ✅ Add microbatch formation across multiple active sessions
- ✅ Add fairness policy for per-session decode participation inside a batch
- ✅ Add KV-budget-aware admission for decode steps
- In progress: Add runtime support for merged decode stepping on a serving group
- In progress: Add completion/progress accounting for batched decode steps
- In progress: Add tests for latency-vs-throughput scheduling under mixed workloads
- Planned: Add Fozzy scenarios for multi-session decode batching

## 3. Session, KV, And Residency Management

- ✅ Introduce authoritative serving-session and per-replica control-plane state
- ✅ Support checkpoint-mediated KV handoff through the control plane
- In progress: Add explicit KV residency model beyond single checkpoint ownership
- Planned: Track residency by shard range and replica, not only by session
- In progress: Add live KV transfer metadata and transfer lifecycle
- Planned: Add remote KV access path without requiring full checkpoint download
- Planned: Add eviction policy for KV under memory pressure
- Planned: Add prefix reuse and overlap-aware prompt cache tracking
- Planned: Add compute-near-KV scheduling hints
- Planned: Add durability and cleanup policy for stale checkpoints and stale residency records

## 4. Agent Runtime / Serving Engine

- ✅ Introduce backend/runtime boundary in the agent
- ✅ Preserve explicit session identity in runtime state
- ✅ Preserve checkpoint-backed recovery for lost in-memory state
- In progress: Add long-lived decode worker loop that consumes scheduler-owned decode work
- In progress: Add runtime batching interface at the execution backend boundary
- Planned: Add explicit runtime memory budgeting for active sessions and KV
- Planned: Add session eviction hooks driven by control-plane policy
- In progress: Add runtime pause/resume semantics for regroup and failover
- In progress: Add explicit transport prioritization between decode traffic and bulk transfer
- Planned: Add provider-specialized fast paths behind the backend abstraction

## 5. Failover, Regroup, And Recovery

- In progress: Detect participant loss while a session is active
- In progress: Pause affected sessions without corrupting segment/session state
- In progress: Select replacement participants when topology and manifests allow
- Planned: Resume decode on a regrouped serving group
- In progress: Support shrink-only continuation when full replacement is impossible
- Planned: Add dead-session cleanup and orphaned-lease cleanup
- In progress: Add scheduler policy for failover cost vs restart cost
- ✅ Add control-plane and agent tests for node loss during prefill and during decode
- Planned: Add Fozzy exploration scenarios for regroup/failover behavior

## 6. Model Layout And Planning Constraints

- In progress: Replace shard-span-only decode validity with manifest-driven execution validity
- Planned: Add multiple legal layouts per model
- Planned: Add pipeline partitioning metadata
- Planned: Add mixed tensor/pipeline execution layouts
- Planned: Add provider-constrained layout variants
- In progress: Add planner validation against manifest-declared execution constraints
- Planned: Add tests that prove invalid regroup/replan choices are rejected

## 7. Transport And Collective Evolution

- In progress: Add transport capability model richer than current tier tagging
- Planned: Add multiple concurrent inference streams per serving group
- In progress: Add decode-priority traffic classes
- In progress: Add bulk-transfer throttling and fairness
- Planned: Add persistent channel/session reuse for serving groups
- Planned: Add provider-specific collective optimization hooks
- Planned: Add transport fallback policies that preserve runtime-mode intent

## 8. APIs, Persistence, And Data Model

- ✅ Add scheduler-owned queue tables and serving-group tables
- ✅ Add decode-step and batch accounting tables
- ✅ Add worker-facing decode queue observation and decode lease renew/release APIs
- In progress: Add KV residency and transfer tables
- ✅ Add failover/regroup event history tables
- In progress: Add migration coverage and backward-incompatible cleanup for superseded fields
- ✅ Add API surfaces for scheduler/debug/status inspection
- ✅ Add API surfaces for grouped decode telemetry

## 9. Observability And Operations

- ✅ Add metrics for scheduler backlog, runnable sessions, blocked sessions, and batch size
- Planned: Add metrics for KV residency, transfer volume, and checkpoint fallback rate
- Planned: Add metrics for regroup frequency and recovery latency
- ✅ Add structured event log for scheduler decisions
- ✅ Add operator views in UI/API for session placement and queue state
- Planned: Add production-readiness runbooks for scheduler stalls, KV pressure, and failover

## 10. Testing And Verification

- ✅ Add targeted Rust tests for explicit segment/session behavior
- ✅ Add targeted Rust tests for decode-group shrink and final-segment completion
- ✅ Add targeted Rust tests for checkpoint upload/download and handoff validation
- ✅ Add deterministic Fozzy doctor/test/trace/replay/ci coverage for current dispatch path
- In progress: Add deterministic Fozzy coverage for pooled scheduler behavior
- Planned: Add deterministic Fozzy coverage for continuous batching
- Planned: Add deterministic Fozzy coverage for failover and regroup
- ✅ Add host-backed scenarios that exercise runtime-mode-specific behavior
- Planned: Add longer soak scenarios for queue pressure and KV pressure

## 11. Current Execution Order

- ✅ Land authoritative planning, explicit segments, session authority, and checkpoint handoff
- ✅ Build real scheduler ownership over active decode planning and claim-time routing
- ✅ Add explicit decode queue state and serving-group ownership records
- In progress: Add continuous batching in the agent/runtime and scheduler
- In progress: Expand KV management from checkpoint handoff to live residency management
- In progress: Add regroup/failover
- In progress: Constrain planning with richer manifests
- Planned: Deepen backend/provider optimization after engine control flow is stable
