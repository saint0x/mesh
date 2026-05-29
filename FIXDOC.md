# FIXDOC

## Purpose

This document is the current production fix and completion plan for the bespoke multi-machine zip inference stack. It captures what is already done, what still remains, what should be skipped for now, and how the remaining work is split cleanly across three separate agent-owned workstreams so they do not overlap.

The target state is:

- high performance
- low latency
- strong multi-machine scaling
- lower per-machine memory pressure
- correct failover and regroup behavior
- correct accounting and credit attribution
- benchmarked, observable, production-ready behavior

## Current Status

✅ Failover correctness is substantially stronger than it was at the start of this effort.

✅ Control-plane topology selection and normalization are materially improved.

✅ Runtime decode batching, collective transport, and serving dataplane overhead are much leaner.

✅ The system no longer blindly assumes wider decode groups are always better.

✅ Larger-ring transport identity, slot routing, stream binding, and receive coordination are much healthier.

✅ The remaining biggest production gap is no longer basic correctness or obvious bugs.

✅ The remaining biggest production gap is deeper larger-ring decode scaling architecture, especially the cost of distributed collectives in `n > 2` decode rings.

## Three-Agent Work Split

The remaining work is split across three agents with clean boundaries:

- Agent 1 owns control-plane policy, topology selection, planner/scheduler decisions, ring redistribution, and placement normalization.
- Agent 2 owns the agent-side runtime, decode executor, collectives, transport/dataplane hot path, backend-sensitive execution, and larger-ring runtime scaling behavior.
- Agent 3 owns resilience proof, failover drills, checkpoint and KV recovery validation, scaling benchmarks, observability, release gates, and production readiness verification.

The rules of engagement are:

- Agent 1 does not own executor kernels, serving transport internals, or chaos benchmark implementation details.
- Agent 2 does not own scheduler policy, planner ranking policy, or final failover/readiness gate policy.
- Agent 3 does not own internal scheduler algorithms or executor implementation details, but it does define what signals, tests, and gates those systems must satisfy.

## Global Completed Work

✅ Worker-loss handling now redistributes surviving shard ranges instead of only logging redistribution intent.

✅ Decode regroup validation now rejects one-column shard gaps.

✅ Checkpoint-source validation now rejects regroup plans that depend on an offline checkpoint source.

✅ Capacity normalization now uses measured service-rate information more directly instead of relying only on static tier assumptions.

✅ Decode batching is more selective about KV skew and batch-shape cohesion.

✅ Decode transport reuse is in place and serving-channel rebuild churn is lower.

✅ Decode-local executor waste has been reduced materially through fused projections, better batching, lower transport overhead, and leaner collective handling.

✅ 2-worker distributed decode has a true fast path.

✅ Larger-ring collective identity, stream binding, and slot matching are substantially more correct and deterministic.

✅ Slot-scoped receive ownership and mailbox isolation are in place, reducing contention and cross-slot interference.

✅ Repeated deterministic Fozzy validation, trace verify, replay, CI checks, and host-backed runs have been exercised throughout this work.

## Agent 1: Control Plane, Topology Policy, Scheduling

Agent 1 owns the control-plane decisions that determine whether a multi-machine decode topology should run at all, how workers are weighted, how decode groups are chosen, and how the system responds to worker loss before the runtime starts executing. Agent 1 must not overlap with runtime executor internals owned by Agent 2, and must not overlap with resilience test ownership owned by Agent 3. Agent 1 depends on Agent 2 for accurate runtime metrics and on Agent 3 for proof that the policy behaves correctly under stress.

### Agent 1 Completed Work

✅ Planner now redistributes surviving shard ranges after worker loss instead of only logging intended redistribution.

✅ Decode regroup coverage now rejects one-column shard gaps instead of accepting invalid span continuity.

✅ Checkpoint-transfer failover selection now refuses offline checkpoint sources rather than attempting impossible recovery paths.

✅ Ring order and authoritative decode-group reconstruction were tightened so regrouped execution stays deterministic.

✅ Capacity normalization now incorporates measured runtime service-rate signals more directly instead of relying only on static tier heuristics.

✅ Scheduler and planner now score decode topology quality more realistically instead of implicitly preferring wider groups.

✅ Decode-group ranking now discounts structurally weak groups such as imbalanced, mixed-provider, or weaker-transport topologies.

✅ Throughput-oriented topology rank is now available as a final scheduler tie-break without overriding existing fairness and latency ordering.

✅ Control-plane behavior is now aligned with the principle that extra machines must earn inclusion by expected efficiency, not by width alone.

### Agent 1 Remaining Work

- Add live topology adaptation so decode groups can shrink or avoid expansion when observed collective cost outweighs added capacity.
- Feed runtime-measured collective wait, effective tok/s, and backend/network slowdown back into planner and scheduler decisions as first-class inputs.
- Add explicit topology penalties for unstable or churn-heavy workers so the control plane avoids repeatedly rebuilding around flaky nodes.
- Add stronger mixed-backend policy so CUDA, Metal, and weaker peers are grouped only when measured normalization shows the topology is still worthwhile.
- Add topology admission guardrails that can refuse wider decode groups when predicted throughput is worse than a narrower configuration.
- Add model-size-aware decode-group policy so small models do not over-expand into communication-dominated rings.
- Add placement policy that distinguishes memory-driven expansion from throughput-driven expansion so “fits” and “scales” are not treated as the same outcome.
- Add explicit scheduler support for graceful downshift after worker loss so regrouped decode can prefer the best surviving topology, not just the first valid one.
- Add stronger accounting hooks around topology changes so crediting and resource attribution remain correct across regroup, shrink, and re-expansion events.

## Agent 2: Runtime, Executor, Collectives, Dataplane

Agent 2 owns the agent-side runtime path end to end: decode executor efficiency, collective execution, serving transport/dataplane, batch formation on the runtime side, backend-sensitive hot-path behavior, and the scaling behavior of larger decode rings inside the agent process. Agent 2 must not overlap with scheduler policy ownership owned by Agent 1, and must not overlap with resilience/benchmark gate ownership owned by Agent 3. Agent 2 depends on Agent 1 to choose good topologies and on Agent 3 to prove runtime behavior under production stress.

### Agent 2 Completed Work

✅ Serving transport reuse was introduced so decode hot paths stop rebuilding serving channels repeatedly.

✅ Decode batching gained KV-skew guardrails to avoid harmful mixing of tiny and huge decode contexts.

✅ Decode batching gained fast-path bucket cohesion so incompatible decode cohorts stop contaminating the same microbatch.

✅ Runtime stats now expose guardrail-driven deferrals so runtime shaping is observable.

✅ The decode tail now keeps logits batched longer and performs batched device sampling where cohort parameters match.

✅ Fused QKV projections were implemented to reduce per-layer decode GEMM count.

✅ Fused gate/up projections were implemented to reduce per-layer MLP GEMM count.

✅ The fused projection memory regression was fixed so decode math reductions do not duplicate long-lived device residency unnecessarily.

✅ The fast-path decode loop was simplified to use direct backend-slice attention flow with less per-layer allocation churn.

✅ Local KV-head index tensors for cached device attention are now built once and reused instead of rebuilt every decode step.

✅ Collective chunk-layout computation is cached inside the ring executor.

✅ Collective step choreography for larger rings is cached and reused instead of recomputed each pass.

✅ Cached collective plans were moved to shared ownership so reuse no longer reclones the plan body every time.

✅ A real 2-worker collective fast path was added so the smallest distributed decode ring skips the full two-phase choreography.

✅ Matrix collectives were moved onto the direct wire-byte path instead of materializing temporary float vectors first.

✅ Generic tensor all-reduce was brought onto the same larger-ring direct wire-byte path as the matrix path.

✅ Collective payload encoding now uses explicit little-endian `f32` wire bytes and lower-overhead memcpy-style paths on supported targets.

✅ Frame send overhead was reduced with zero-copy or borrowed payload handling and vectored writes.

✅ Frame receive overhead was reduced with shared frame-body reads, deferred decode, and lazy payload interpretation.

✅ Frame protocol overhead was reduced by removing unused timestamp, shape, and stale padding/header baggage.

✅ Larger-ring collective identity was corrected so send and receive slot ownership is explicit and valid for `n > 2`.

✅ Serving waiters are now keyed by expected sender position so wrong-producer traffic cannot poison the right waiter.

✅ The stale serving-session identity layer was removed, and serving transport now uses lane and neighbor semantics directly.

✅ Logical serving stream IDs now bind to stable underlying TCP channels so planned multi-stream lane behavior is real.

✅ Serving send hot paths now use transport-owned durable bound lane streams instead of shared pool selection on every send.

✅ Receive coordination was moved from global wakeups to slot-scoped waiters and then to slot-owned mailboxes.

✅ The receive path was further split so inbound slot traffic no longer serializes through one shared global inbound lock.

✅ Background checkpoint and bulk transfers no longer get eagerly drained before decode collectives when overlap is safe.

✅ Completed background overlap tasks are reaped more proactively so long decode sessions do not carry stale transport state.

✅ Collective wait telemetry was corrected so send and receive timing are independently meaningful.

✅ Metal decode all-reduce staging now uses reusable pooled scratch instead of repeated per-reduction shared-buffer allocation.

✅ Metal pooled staging was made safer with rotating slots and explicit release on cache or session reset.

✅ Batched decode no longer steals collective scratch ownership from `backends[0]`; it uses dedicated batch-local workspace.

✅ Batched decode collectives now use a deterministic batch-derived collective identity instead of impersonating the first request.

### Agent 2 Remaining Work

- Implement a deeper larger-ring decode strategy that reduces or overlaps the effective two-collective-per-layer cost instead of only trimming setup overhead around it.
- Add collective and computation overlap where safe, especially for `n > 2`, so useful local work can progress while network phases are in flight.
- Evaluate decode-specific collective variants for larger rings instead of assuming the current ring all-reduce structure is the final architecture.
- Push backend-native specialization further, especially for CUDA and Metal collective execution, so fewer steps route through generic host-oriented scaffolding.
- Reduce staging and conversion boundaries between backend tensors and collective buffers even further on device-native paths.
- Introduce backend-aware autotuning for lane counts, stream counts, chunk sizing, and overlap policy based on observed runtime behavior.
- Add stronger runtime-side scaling metrics so the agent can expose collective-share, overlap efficiency, and per-ring degradation signals to Agent 1.
- Add broader larger-ring stress coverage inside the agent runtime, especially for 3-plus worker decode rings with overlapped traffic and mixed stream usage.
- Expand live transport regressions around partial writes, reconnect churn, out-of-order arrival, and high-contention multi-slot receive behavior under larger-ring load.
- Tighten memory-pressure behavior during long decode sessions so pooled scratch and collective buffers stay efficient under repeated topology changes.

## Agent 3: Failover, Recovery, Benchmarks, Observability, Readiness

Agent 3 owns production resilience and proof, not scheduler policy internals or executor and transport internals. This workstream covers failover correctness, regroup behavior after machine loss, checkpoint and KV recovery safety and efficiency, chaos-style validation, scaling benchmark coverage, observability for real production diagnosis, and final readiness gates. Agent 3 depends on Agent 1 for topology policy inputs and on Agent 2 for stable runtime and collective metrics, but does not own their internal implementation.

### Agent 3 Completed Work

✅ Failover now redistributes surviving shard ranges instead of only logging redistribution intent.

✅ Decode regroup validation now rejects one-column shard coverage gaps.

✅ Checkpoint-source validation now rejects regroup and transfer plans that depend on an offline latest checkpoint source.

✅ Decode-group authority and regroup ordering were tightened so failover reconstruction is more deterministic.

✅ Worker-loss handling and failover reconciliation were hardened for faster and safer resume behavior.

✅ Background checkpoint and bulk transfers no longer get eagerly drained at the start of decode collectives, which preserves intended overlap.

✅ Completed overlapped background transfers are now reaped without waiting for a full blocking drain path.

✅ Collective wait telemetry now distinguishes send-side and receive-side wait timing correctly, improving diagnosis of larger-ring bottlenecks.

✅ Larger-ring serving-slot identity was corrected so inbound matching uses true collective identity and expected sender position instead of stale session-style assumptions.

✅ Slot-scoped receive wakeups and mailbox ownership were introduced so overlapped collective traffic is less likely to interfere across unrelated waits.

✅ Real deterministic Fozzy coverage has been exercised repeatedly across failover regroup, runtime serving, integrated serving, trace verify, replay, CI validation, and host-backed execution paths.

### Agent 3 Remaining Work

- Define and land a full chaos matrix for mid-decode worker loss, slow-worker degradation, lane stalls, reconnect churn, and mixed failure-plus-recovery traffic.
- Add deterministic failover drills that explicitly verify instant regroup behavior when a ring shrinks from 4 workers to 3 during active decode.
- Expand checkpoint and KV recovery from correctness-first behavior to efficiency-first behavior, including delta-aware or paged KV movement where feasible.
- Add explicit recovery backpressure rules so checkpoint or KV repair traffic cannot silently starve live decode under pressure.
- Build production scaling benchmark suites across 1, 2, 3, and 4-plus workers with tok/s, per-node memory pressure, regroup latency, checkpoint recovery time, and collective-wait share as first-class outputs.
- Turn scaling benchmarks into release gates so widening a decode topology cannot ship without measured evidence that scaling is neutral-to-positive for the targeted shapes.
- Add mixed-backend benchmark coverage so CUDA, Metal, and heterogeneous rings are evaluated separately instead of relying on one aggregate scaling story.
- Expand observability with production-grade counters and summaries for regroup cause, checkpoint transfer cause, recovery duration, degraded-topology duration, and post-failover throughput impact.
- Define readiness thresholds for acceptable failover recovery time, acceptable degraded-mode throughput loss, and acceptable checkpoint or KV transfer overhead.
- Verify accounting and credit-system correctness across regroup, replacement, degraded execution, checkpoint transfer, and resumed execution flows.

## Work That Can Be Skipped For Now

The following items are not the right next use of time unless later profiling or production incidents prove otherwise:

- more tiny serving-frame header shaving
- more metadata-level protocol trimming that saves a few bytes but adds conceptual complexity
- backward-compatibility layers for older transport semantics
- speculative hot-path compression or encryption work unless wire cost is shown to dominate
- broad rewrites of unrelated scheduler areas before larger-ring decode scaling is fixed
- chasing absolute “every new machine always increases tok/s” guarantees instead of measured, policy-enforced positive scaling in realistic conditions

## Recommended Next Execution Order

1. Agent 2: deeper larger-ring decode collective redesign and overlap
2. Agent 1: adaptive topology shrink and expansion policy using real runtime signals
3. Agent 3: scaling benchmark matrix and release gates
4. Agent 2: backend-native specialization and autotuning
5. Agent 3: chaos drills and failover-readiness gates
6. Agent 1: stronger mixed-backend placement and accounting policy around topology transitions

## Definition Of Done

This body of work is done only when:

- wider decode groups no longer get selected by default and instead prove their value structurally and empirically
- larger-ring decode throughput is measurably stronger for the targeted production topologies
- machine loss during active decode regroups safely and quickly
- checkpoint and KV recovery are both correct and efficient
- accounting stays correct across shrink, regroup, replacement, and resume flows
- scaling, failover, and mixed-backend behavior are covered by deterministic tests, traces, chaos drills, and benchmark release gates
