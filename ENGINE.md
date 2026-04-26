# Zip Roadmap

This file tracks the remaining work to turn the current explicit-session,
explicit-segment serving stack into the full production `zip` engine.

## Status Legend

- ☐ Not started
- ⏳ In progress
- ✅ Done

## 1. Control-Plane Scheduling

- ✅ Persist authoritative execution plans at submit time
- ✅ Represent explicit prefill and decode segments
- ✅ Re-plan decode placement at prefill handoff against live pool state
- ✅ Move claim-time assignment selection behind a dedicated scheduler service
- ✅ Introduce a first-class scheduler state model for queued, runnable, and blocked jobs
- ✅ Introduce explicit serving-group ownership records rather than inferring from assignments
- ☐ Add admission control across multiple concurrent jobs and models
- ☐ Add topology-aware pool partitioning for prefill-heavy vs decode-heavy work
- ⏳ Add scheduler scoring that reasons about queue age, latency target, throughput target, and resiliency target
- ⏳ Add policy-specific behavior for `fit_first`, `throughput_first`, `latency_first`, and `resilient_edge`
- ✅ Add scheduler reconciliation loop that repairs stuck or stale ownership
- ✅ Add scheduler observability for queue depth, blocked reasons, and group utilization

## 2. Decode Queue And Continuous Batching

- ✅ Add explicit decode work queue records in the control plane
- ✅ Add decode-step ownership, lease semantics, and batch lease targets
- ✅ Add microbatch formation across multiple active sessions
- ✅ Add fairness policy for per-session decode participation inside a batch
- ✅ Add KV-budget-aware admission for decode steps
- ⏳ Add runtime support for merged decode stepping on a serving group
- ⏳ Add completion/progress accounting for batched decode steps
- ⏳ Add tests for latency-vs-throughput scheduling under mixed workloads
- ☐ Add Fozzy scenarios for multi-session decode batching

## 3. Session, KV, And Residency Management

- ✅ Introduce authoritative serving-session and per-replica control-plane state
- ✅ Support checkpoint-mediated KV handoff through the control plane
- ⏳ Add explicit KV residency model beyond single checkpoint ownership
- ☐ Track residency by shard range and replica, not only by session
- ⏳ Add live KV transfer metadata and transfer lifecycle
- ☐ Add remote KV access path without requiring full checkpoint download
- ☐ Add eviction policy for KV under memory pressure
- ☐ Add prefix reuse and overlap-aware prompt cache tracking
- ☐ Add compute-near-KV scheduling hints
- ☐ Add durability and cleanup policy for stale checkpoints and stale residency records

## 4. Agent Runtime / Serving Engine

- ✅ Introduce backend/runtime boundary in the agent
- ✅ Preserve explicit session identity in runtime state
- ✅ Preserve checkpoint-backed recovery for lost in-memory state
- ⏳ Add long-lived decode worker loop that consumes scheduler-owned decode work
- ⏳ Add runtime batching interface at the execution backend boundary
- ☐ Add explicit runtime memory budgeting for active sessions and KV
- ☐ Add session eviction hooks driven by control-plane policy
- ⏳ Add runtime pause/resume semantics for regroup and failover
- ⏳ Add explicit transport prioritization between decode traffic and bulk transfer
- ☐ Add provider-specialized fast paths behind the backend abstraction

## 5. Failover, Regroup, And Recovery

- ⏳ Detect participant loss while a session is active
- ⏳ Pause affected sessions without corrupting segment/session state
- ⏳ Select replacement participants when topology and manifests allow
- ☐ Resume decode on a regrouped serving group
- ⏳ Support shrink-only continuation when full replacement is impossible
- ☐ Add dead-session cleanup and orphaned-lease cleanup
- ⏳ Add scheduler policy for failover cost vs restart cost
- ✅ Add control-plane and agent tests for node loss during prefill and during decode
- ☐ Add Fozzy exploration scenarios for regroup/failover behavior

## 6. Model Layout And Planning Constraints

- ⏳ Replace shard-span-only decode validity with manifest-driven execution validity
- ☐ Add multiple legal layouts per model
- ☐ Add pipeline partitioning metadata
- ☐ Add mixed tensor/pipeline execution layouts
- ☐ Add provider-constrained layout variants
- ⏳ Add planner validation against manifest-declared execution constraints
- ☐ Add tests that prove invalid regroup/replan choices are rejected

## 7. Transport And Collective Evolution

- ⏳ Add transport capability model richer than current tier tagging
- ☐ Add multiple concurrent inference streams per serving group
- ⏳ Add decode-priority traffic classes
- ⏳ Add bulk-transfer throttling and fairness
- ☐ Add persistent channel/session reuse for serving groups
- ☐ Add provider-specific collective optimization hooks
- ☐ Add transport fallback policies that preserve runtime-mode intent

## 8. APIs, Persistence, And Data Model

- ✅ Add scheduler-owned queue tables and serving-group tables
- ✅ Add decode-step and batch accounting tables
- ⏳ Add KV residency and transfer tables
- ✅ Add failover/regroup event history tables
- ⏳ Add migration coverage and backward-incompatible cleanup for superseded fields
- ✅ Add API surfaces for scheduler/debug/status inspection
- ✅ Add API surfaces for grouped decode telemetry

## 9. Observability And Operations

- ✅ Add metrics for scheduler backlog, runnable sessions, blocked sessions, and batch size
- ☐ Add metrics for KV residency, transfer volume, and checkpoint fallback rate
- ☐ Add metrics for regroup frequency and recovery latency
- ⏳ Add structured event log for scheduler decisions
- ✅ Add operator views in UI/API for session placement and queue state
- ☐ Add production-readiness runbooks for scheduler stalls, KV pressure, and failover

## 10. Testing And Verification

- ✅ Add targeted Rust tests for explicit segment/session behavior
- ✅ Add targeted Rust tests for decode-group shrink and final-segment completion
- ✅ Add targeted Rust tests for checkpoint upload/download and handoff validation
- ✅ Add deterministic Fozzy doctor/test/trace/replay/ci coverage for current dispatch path
- ⏳ Add deterministic Fozzy coverage for pooled scheduler behavior
- ☐ Add deterministic Fozzy coverage for continuous batching
- ☐ Add deterministic Fozzy coverage for failover and regroup
- ✅ Add host-backed scenarios that exercise runtime-mode-specific behavior
- ☐ Add longer soak scenarios for queue pressure and KV pressure

## 11. Current Execution Order

- ✅ Land authoritative planning, explicit segments, session authority, and checkpoint handoff
- ✅ Build real scheduler ownership over active decode planning and claim-time routing
- ✅ Add explicit decode queue state and serving-group ownership records
- ⏳ Add continuous batching in the agent/runtime and scheduler
- ⏳ Expand KV management from checkpoint handoff to live residency management
- ⏳ Add regroup/failover
- ⏳ Constrain planning with richer manifests
- ☐ Deepen backend/provider optimization after engine control flow is stable
