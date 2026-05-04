# TUNEUP

## Purpose

This document captures a detailed performance and architecture tune-up plan
for Mesh and its `zip` distributed inference engine after a thorough review of:

- the local Mesh/ZIP implementation in this repository
- the sibling `rvllm` repository at `/Users/deepsaint/Desktop/rvllm`

This is not a proposal to turn Mesh into `vLLM`, `rvLLM`, or a generic
single-node inference server.

Mesh and `zip` have a real, distinct purpose:

- distributed inference across multiple machines
- explicit session ownership and execution placement
- direct peer-to-peer tensor transport
- control-plane-backed scheduling, accounting, and orchestration
- resumability, failover, regroup, and KV handoff across a networked system

That purpose remains valid and important.

The goal of this tune-up is therefore not identity drift. The goal is:

- preserve the distributed reason for existence
- preserve the operator and product semantics that make Mesh distinct
- aggressively absorb any executor, kernel, KV, batching, memory, and transport
  implementation improvements that can materially increase performance
- be willing to refactor or replace local execution internals where they are
  acting as hard performance ceilings

The guiding principle is:

`zip` should remain a distributed inference engine, but its local execution core
should be as good as we can possibly make it.

## Executive Summary

The core conclusion from the review is that `rvllm` is ahead of the current
Mesh/ZIP execution stack in the places that most directly determine raw
inference throughput and latency:

- fused per-layer execution
- paged KV cache layout built around the attention kernel contract
- graph-safe preallocation and graph replay
- explicit batch buckets and static metadata layouts
- minimal serving-path overhead during active decode
- aggressively device-native execution rather than high-level tensor orchestration

At the same time, Mesh/ZIP is ahead in the areas that define its distinct
distributed purpose:

- network-aware scheduling
- explicit serving sessions and decode queue state
- cohort scheduling and pooled decode lease semantics
- durable control-plane state
- checkpoint-based and live KV handoff
- failover, regroup, and resume flows
- operator visibility and scheduler observability

This means the right target state is not to imitate another system wholesale.

The right target state is:

- keep Mesh/ZIP’s distributed control-plane and session semantics
- keep its topology-aware scheduling and resilience model where those are truly
  required for the product
- replace or deeply refactor the local execution path so that it approaches
  `rvllm`-class efficiency

In blunt terms:

- keep the distributed shell
- replace the soft local core

## Scope Of Review

The review covered the main execution and scheduling surfaces in both codebases.

### Mesh/ZIP files reviewed

- [README.md](/Users/deepsaint/Desktop/meshnet/README.md)
- [ENGINE.md](/Users/deepsaint/Desktop/meshnet/ENGINE.md)
- [ZIPPERF.md](/Users/deepsaint/Desktop/meshnet/ZIPPERF.md)
- [agent/src/inference/coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs)
- [agent/src/inference/engine.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/engine.rs)
- [agent/src/inference/backend.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/backend.rs)
- [agent/src/inference/forward_pass.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/forward_pass.rs)
- [agent/src/inference/kv_cache.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/kv_cache.rs)
- [agent/src/executor/ring_allreduce.rs](/Users/deepsaint/Desktop/meshnet/agent/src/executor/ring_allreduce.rs)
- [agent/src/network/tensor_plane.rs](/Users/deepsaint/Desktop/meshnet/agent/src/network/tensor_plane.rs)
- [agent/src/main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs)
- [control-plane/src/services/scheduler.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/scheduler.rs)
- [control-plane/src/services/planner.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/planner.rs)
- [control-plane/src/services/failover.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/failover.rs)
- [control-plane/src/db/mod.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/db/mod.rs)

### rvLLM files reviewed

- [README.md](/Users/deepsaint/Desktop/rvllm/README.md)
- [v3/crates/rvllm-runtime/src/engine.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-runtime/src/engine.rs)
- [v3/crates/rvllm-runtime/src/scheduler.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-runtime/src/scheduler.rs)
- [v3/crates/rvllm-runtime/src/layer_exec.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-runtime/src/layer_exec.rs)
- [v3/crates/rvllm-runtime/src/gemma4_bring_up.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-runtime/src/gemma4_bring_up.rs)
- [v3/crates/rvllm-attention/src/decode.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-attention/src/decode.rs)
- [v3/crates/rvllm-attention/src/prefill.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-attention/src/prefill.rs)
- [v3/crates/rvllm-mem/src/kv_layout.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-mem/src/kv_layout.rs)
- [v3/crates/rvllm-mem/src/capture.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-mem/src/capture.rs)
- [v3/crates/rvllm-graph/src/pool.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-graph/src/pool.rs)

## What Must Not Be Lost

Before getting into performance changes, it is important to define the parts of
Mesh/ZIP that should be treated as core product value rather than accidental
implementation detail.

### 1. Distributed inference is the point

Mesh exists to make multiple machines cooperate in inference execution. That
means:

- worker identity matters
- placement matters
- topology matters
- session continuity across devices matters
- network transport quality matters
- ownership and coordination matter

Any performance tune-up that only makes a single node faster but weakens the
distributed system’s reason for existence is incomplete.

### 2. Explicit session semantics are valuable

The explicit session model in `zip` is not fluff. It enables:

- decode continuity
- KV ownership and transfer semantics
- checkpoint-backed recovery
- grouped decode scheduling
- fairness and cohort-aware control
- operator introspection

Those are meaningful features for a distributed engine and should not be thrown
away casually.

### 3. Scheduler and observability work are not wasted

The control-plane scheduling work in Mesh is substantial and valuable. It
already supports:

- explicit serving groups
- decode queue state
- cohort lease behavior
- policy-sensitive scheduling
- blocked/runnable/owned state visibility
- failover and regroup visibility

That work should be preserved where it is serving product goals, even if the
local executor changes dramatically.

### 4. Failover and regroup are still differentiators

A single-machine engine can ignore many failure modes. Mesh cannot. The ability
to:

- pause
- regroup
- shrink
- replace participants
- resume from checkpoint or remote KV state

is part of what makes Mesh meaningfully different from single-node systems.

The tune-up should preserve these capabilities, but may need to relocate them
out of the token hot path.

## High-Level Diagnosis

The central architectural difference between the two systems is this:

- `rvllm` treats model-step execution as sacred and minimizes everything around it
- Mesh/ZIP treats distributed execution semantics as first-class and lets those
  semantics deeply shape the local runtime

That design choice explains most of the performance gap.

In the current Mesh/ZIP implementation:

- the local execution path is still relatively high-level
- the KV representation is optimized for flexibility and recovery, not for
  kernel-native decode
- collectives are implemented as generic framed network operations
- control-plane interactions remain close to active execution
- checkpoint and recovery semantics still visibly influence runtime shape

In `rvllm`:

- execution is organized around preallocated device memory
- kernel launch count is tightly controlled
- graph capture is a first-class assumption
- metadata layouts are static and hash-checked
- KV layout is defined by the attention kernel’s expectations
- the runtime does not let orchestration concerns contaminate the local step

The result is that Mesh/ZIP currently has more scheduling and orchestration
depth, but `rvllm` has a stronger execution substrate.

## Detailed Findings

## Finding 1: The current local executor is too high-level

The single biggest issue is that the current local model execution path in
Mesh/ZIP is still built around a sequence of high-level tensor operations rather
than a purpose-built fused inference executor.

In [agent/src/inference/forward_pass.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/forward_pass.rs:1038),
each layer is effectively executed as:

- RMS norm
- local Q projection matmul
- local K projection matmul
- local V projection matmul
- attention computation
- O projection matmul
- ring all-reduce
- residual add
- MLP norm
- gate projection matmul
- up projection matmul
- activation
- elementwise multiply
- down projection matmul
- ring all-reduce
- residual add

That is a reasonable correctness-first structure, but it is not what a top-end
GPU inference engine should look like.

By contrast, `rvllm`’s layer path in
[v3/crates/rvllm-runtime/src/layer_exec.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-runtime/src/layer_exec.rs:1)
is explicitly designed around:

- fused norm + quant
- fused or packed GEMM paths
- fused RoPE + KV write
- paged attention kernels
- fused quantization
- residual epilogues inside projection paths
- fused activation and MLP pointwise work

This gap matters because kernel launch count, intermediate materialization, and
memory traffic dominate performance once the math itself is reasonably good.

### Implication

If Mesh/ZIP keeps the current high-level forward path for its serious GPU
backend, it will continue to leave a large amount of performance on the table.

### Recommendation

Treat the current Candle-centric path as:

- a correctness path
- a CPU path
- a bring-up path
- a fallback path

Do not treat it as the long-term GPU fast path.

The long-term fast path should be rebuilt around:

- fused kernels
- preallocated workspaces
- static metadata buffers
- graph-capturable launches
- kernel-native KV layout

## Finding 2: KV cache architecture is recovery-friendly but not hot-path-native

Mesh/ZIP’s KV cache is clearly designed to support:

- export/import
- checkpointing
- recovery
- suffix retention
- portability across backends

That is good for distributed semantics, but it is not the ideal live execution
representation for high-performance decode.

The current cache path in:

- [agent/src/inference/kv_cache.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/kv_cache.rs)
- [agent/src/inference/forward_pass.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/forward_pass.rs:210)

uses growable layer caches, append logic, suffix retention, and snapshot export
logic that are much more general than a high-performance paged-attention kernel
actually wants.

`rvllm` instead defines KV layout directly in terms of the kernel contract in
[v3/crates/rvllm-mem/src/kv_layout.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-mem/src/kv_layout.rs:1):

- block-oriented
- paged
- static layout math
- direct mapping to block tables
- no extra conceptual layers between runtime cache and attention kernel

### Implication

Mesh/ZIP is likely paying cost in:

- append/update overhead
- device-side scatter/update complexity
- cache materialization shape
- inability to directly map the live cache to the best attention kernels
- export-driven representation choices leaking into live decode

### Recommendation

Split KV into two representations:

1. **Live execution KV format**
   - paged
   - block-table-driven
   - kernel-native
   - static-stride
   - optimized for decode and prefill

2. **Transfer/checkpoint KV format**
   - portable
   - resumable
   - serialization-friendly
   - may be chunked, compressed, or normalized for transport

Do not require the live execution format to also be the best recovery artifact.

## Finding 3: Graph capture and static execution invariants are missing

One of the most meaningful advantages in `rvllm` is not just fused kernels. It
is the system-level assumption that the execution path is:

- preplanned
- preallocated
- bucketed
- graph-safe
- metadata-stable

This is visible in:

- [v3/crates/rvllm-mem/src/capture.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-mem/src/capture.rs:1)
- [v3/crates/rvllm-graph/src/pool.rs](/Users/deepsaint/Desktop/rvllm/v3/crates/rvllm-graph/src/pool.rs:1)

Important properties there:

- graph capture is explicit
- reallocation during capture is treated as a design error
- graphs are captured up front
- buckets are explicit
- replay validates metadata-layout hashes
- drift is surfaced as typed failure, not silent fallback

Mesh/ZIP currently has no equivalent execution substrate for its hot GPU path.

### Implication

Without graph capture and stable execution buckets, Mesh/ZIP is likely paying:

- repeated launch overhead
- repeated metadata preparation overhead
- extra host-side orchestration cost
- less predictable performance
- weaker batch-shape specialization

### Recommendation

Introduce a dedicated execution layer for GPU backends with:

- fixed decode buckets
- fixed prefill buckets
- preallocated scratch arenas
- stable metadata layouts
- graph-captured forward graphs per bucket where feasible
- hard validation that runtime metadata matches captured expectations

This should be treated as foundational infrastructure, not optional polish.

## Finding 4: The collective layer is functional but not near hardware ceiling

The current ring collective implementation in
[agent/src/executor/ring_allreduce.rs](/Users/deepsaint/Desktop/meshnet/agent/src/executor/ring_allreduce.rs:334)
is doing real distributed work, but it is still fundamentally a generic,
framed, host-coordinated collective path.

Characteristics of the current path:

- chunked reduce-scatter / all-gather
- framed network messages
- timeout wrappers
- explicit send/receive accounting
- buffer slicing and host-managed flow

That is a valid distributed transport implementation, but it is not what the
highest-performance tensor-parallel collective substrate should ultimately look
like for serious GPU serving.

### Implication

Even if the local executor becomes much faster, the collective substrate may
become the new bottleneck unless it evolves too.

Potential costs include:

- too much host involvement
- too many copies
- generic framing overhead
- limited overlap of compute and communication
- no provider-native fast path comparable to NCCL-class behavior

### Recommendation

Evolve the collective layer in phases:

1. keep the existing generic path as the portable baseline
2. add transport- and provider-specialized fast paths
3. make collective buffers device-native where possible
4. aggressively reduce host mediation
5. overlap collective progress with compute where feasible
6. specialize for stable serving groups, not just generic distributed tensor exchange

The current code already hints at this direction with optimization profiles and
persistent serving channels. That work should be deepened substantially.

## Finding 5: The control plane is too close to active decode execution

Mesh’s control plane is a real strength, but it currently sits too close to the
live execution path for a pure performance-first engine.

The agent main loop in [agent/src/main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs:1549)
is actively involved in:

- claiming work
- observing queue state
- acknowledging assignments
- importing checkpoints for decode
- renewing decode leases
- reporting progress
- releasing leases on failure paths

This is well-engineered distributed machinery, but it is still machinery near
the hot path.

Meanwhile the scheduler itself in
[control-plane/src/services/scheduler.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/scheduler.rs:1)
is intentionally rich:

- queue-state classification
- lease ranking
- cohort-aware selection
- stale lease repair
- fairness and policy modes

That sophistication is useful, but it also means there is real system pressure
around frequent state mutation and polling.

This is not theoretical. The local benchmark log in
[ZIPPERF.md](/Users/deepsaint/Desktop/meshnet/ZIPPERF.md:78) already documents:

- `database is locked`
- collective timeout failures
- live inference contention during benchmarking

### Implication

Even if local math gets faster, the overall system can still underperform if:

- lease renewals
- queue visibility
- progress reporting
- polling
- DB write contention

remain close to token cadence.

### Recommendation

Push the control plane outward from the token loop.

Concretely:

- let the control plane decide placement, ownership, and coarse serving intent
- let the worker execute a local decode run with as little external mutation as possible
- batch or coarsen progress/reporting surfaces
- reduce database write amplification during active decode
- decouple scheduler observability from per-token critical execution paths

This does not mean removing the control plane. It means preserving its role
while making the worker execution core more autonomous over short time windows.

## Finding 6: Checkpoint and failover semantics are valuable but too entangled

Mesh/ZIP’s checkpointing, failover, and regroup semantics are real
differentiators. However, they are currently deep enough in the runtime shape
that they influence the architecture of the active executor.

Examples include:

- checkpoint export/import in
  [agent/src/inference/coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs:805)
- checkpoint save/recover in
  [agent/src/inference/coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs:1113)
- failover decision logic in
  [control-plane/src/services/failover.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/failover.rs:1)

### Implication

Recovery semantics are influencing live state representation and runtime control
flow more than they should in a max-performance design.

### Recommendation

Keep these features, but isolate them architecturally:

- the fast path should assume steady-state local execution
- recovery should be a side system
- checkpoint export should be cold-path oriented
- failover decisions should not distort the structure of normal decode
- live KV format should be optimized for execution, with explicit conversion to
  transfer formats when needed

In other words:

- resilience remains a feature
- resilience must stop defining the inner shape of the executor

## Finding 7: Mesh/ZIP may be overpaying for generality in the GPU path

The current backend architecture in
[agent/src/inference/backend.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/backend.rs:1)
is well-structured, but the abstraction still leaves the GPU path looking
relatively general-purpose.

That is good for maintainability and cross-provider portability, but for a
serious fast path it can lead to:

- dynamic dispatch overhead
- too much shared code between fundamentally different provider strategies
- lowest-common-denominator execution structure
- delayed specialization

### Recommendation

Keep the backend boundary, but allow a much sharper split:

- CPU path: correctness-first, broad compatibility, simpler runtime
- Metal path: specialized device-native fast path
- CUDA path: aggressively fused, graph-captured, paged-KV path

The abstraction should preserve behavioral correctness and product semantics,
not force identical execution architecture under every provider.

## Where rvLLM Is Strongest

The review suggests that the highest-value `rvllm` strengths are:

### 1. Fused layer execution

`rvllm` explicitly reduces launch count and intermediate traffic.

This is one of the clearest areas where Mesh/ZIP should learn from or directly
adopt its ideas.

### 2. Paged KV as first principle

The KV layout exists to satisfy the attention path efficiently, not to be
universally flexible.

That is the correct design instinct for live decode.

### 3. Graph safety as architecture, not optimization garnish

The combination of:

- preallocation
- graph-safe capture
- metadata hashing
- bucketed replay

is a major architectural advantage.

### 4. Runtime minimalism

The execution path is not cluttered with serving-stack concerns. Mesh/ZIP should
not copy this blindly, but it should copy the discipline of keeping the local
executor pure.

### 5. Explicit kernel contracts

The `rvllm` codebase tends to define very concrete contracts around:

- shape
- layout
- workspace
- launch path
- supported variants

That is good for speed and debuggability.

## Where Mesh/ZIP Is Strongest

It is equally important to acknowledge where Mesh/ZIP is stronger.

### 1. Distributed control semantics

Mesh/ZIP has much richer semantics around:

- sessions
- ownership
- scheduling
- leases
- cohorts
- regroup
- failover

### 2. Real distributed product shape

Mesh/ZIP is designed around multiple machines as a first-class reality.

### 3. Operator-facing visibility

The scheduler and queue observability work is far deeper than a typical local
executor.

### 4. Recovery and continuity

Mesh/ZIP has a more mature notion of how a live distributed session can recover
through degradation and topology change.

These are genuine strengths and should be carried forward.

## What Should Be Preserved

The following areas should be preserved conceptually, even if their
implementation changes:

- explicit session authority
- decode queue and pooled batching semantics
- topology-aware placement
- KV ownership and handoff semantics
- regroup and failover behavior
- durable job and scheduler state
- operator inspection surfaces

## What Should Be Replaced Or Deeply Refactored

The following areas are likely hard ceilings if left mostly intact:

### 1. The Candle-centric GPU forward path

Keep it for correctness or fallback if needed, but it should not remain the
primary serious-performance GPU implementation.

### 2. The live KV execution representation

It should become paged and kernel-native, with transfer/export separated out.

### 3. The lack of graph-captured bucketed execution

This should be treated as missing infrastructure, not optional enhancement.

### 4. The current collective substrate for high-end GPU serving

The generic ring path is useful, but it should not be assumed to be the final
performance substrate.

### 5. Control-plane write frequency and proximity during decode

The scheduling model may remain, but the rate and intimacy of control-plane
interactions during token generation should be reduced.

## Recommended Target Architecture

The cleanest direction is to split the system into clearer layers.

### Layer 1: `zip-core`

This is the local execution engine.

Responsibilities:

- model execution
- paged KV management
- decode/prefill kernels
- batch buckets
- graph capture/replay
- provider-specialized execution
- device-native sampling tail

Non-responsibilities:

- scheduler DB interaction
- lease ranking
- checkpoint transport orchestration
- failover policy

This layer should feel much closer to `rvllm` than the current local executor.

### Layer 2: `zip-dist`

This is the distributed execution shell.

Responsibilities:

- session placement
- decode ownership
- cohort scheduling
- KV handoff orchestration
- collective setup
- pause/resume/regroup integration
- progress and lifecycle reporting

This layer should preserve the distinct Mesh reason for existence.

### Layer 3: `zip-recovery`

This layer owns:

- checkpoint materialization
- checkpoint transfer
- KV export/import conversion
- recovery policy integration

Its key property should be:

- cold-path semantics do not shape the live hot path more than necessary

### Layer 4: control-plane orchestration

Keep:

- planning
- fairness
- durable state
- observability
- failover decisions

Reduce:

- per-token pressure
- unnecessary synchronous coupling to active decode

## Detailed Recommendations

## Recommendation A: Rebuild the CUDA fast path around fused kernels

For CUDA-backed serving, a new fast path should be introduced that is built
around:

- fused norm + quant
- packed QKV projection
- fused RoPE + KV write
- paged attention decode and prefill
- fused residual epilogues
- fused activation and MLP pointwise sections
- device-side sampling tail

The design should allow:

- stable decode buckets
- stable prefill buckets
- preallocated workspaces
- graph capture

This can either be:

- inspired by `rvllm`
- partially ported from `rvllm`
- or replaced with shared lower-level components where licensing and code shape
  make sense

The important thing is not code provenance. The important thing is getting to an
executor design of that class.

## Recommendation B: Make paged KV the canonical live format

The live decode cache should be redesigned around:

- pages
- block tables
- known strides
- fixed layout math
- kernel-friendly indexing

The current snapshot-oriented cache should move into a translation/export layer.

Benefits:

- better attention kernel compatibility
- lower live cache management overhead
- clearer batching semantics
- cleaner graph capture story
- easier communication of KV ownership in terms of pages or slices

## Recommendation C: Introduce graph-captured execution buckets

Add an execution planning layer that:

- enumerates supported batch buckets
- defines metadata layouts for each bucket
- preallocates scratch and metadata regions
- captures graphs up front where feasible
- verifies replay layout compatibility

This should be done for decode first, then prefill.

Decode benefits are likely immediate because:

- shapes are simpler
- metadata changes are more structured
- token cadence makes launch overhead matter more

## Recommendation D: Separate live decode from control-plane churn

The control plane should continue to own:

- assignment
- ownership
- scheduling policy
- coarse execution control

But once a worker owns a decode run, the worker should be able to execute for a
meaningful interval with minimal external coordination pressure.

Possible directions:

- coarser lease renewal cadence
- batch-level progress reporting instead of finer-grained mutation
- reduced scheduler writes during steady-state decode
- more worker-local execution autonomy within an owned lease window

## Recommendation E: Treat checkpointing as a cold-path export path

Checkpointing should remain a first-class feature, but its representation should
be derived from the live state, not define it.

Desired relationship:

- live KV is execution-optimized
- checkpoint KV is transfer/recovery-optimized
- explicit conversion occurs at export/import boundaries

This will preserve resilience while allowing much tighter live-memory design.

## Recommendation F: Deepen provider specialization

The backend abstraction is good, but specialization should be much sharper.

### CPU

- keep broad compatibility
- prioritize correctness and portability
- do not force CPU constraints onto GPU design

### Metal

- pursue device-native residency and lower-copy paths
- specialize collectives where possible
- use the local provider’s strengths rather than mirroring CUDA architecture

### CUDA

- fused kernels
- static arenas
- graph capture
- paged KV
- specialized collective path

## Recommendation G: Revisit collective strategy by provider class

Not all providers should use the same collective implementation strategy.

The current ring path is a good universal baseline, but the final system may
need:

- different transport behavior for CPU vs GPU
- device-native buffer handling
- serving-group-stable connection reuse
- reduced host copy overhead
- compute/communication overlap

This is especially important if the local executor gets much faster, because the
network/collective share of total step time will rise.

## Recommendation H: Keep the scheduler, but stop making it pay token tax

Mesh’s scheduler work is impressive and valuable. The problem is not that it
exists. The problem is that the execution architecture still seems exposed to
its pressure during active serving.

The right move is:

- keep the scheduler semantics
- keep queue and lease concepts
- keep observability
- keep fairness and topology awareness
- reduce live-step coupling to DB-backed mutation and polling

## Priority Order

If this work is executed in phases, the priority order should be:

### Priority 1: Local executor replacement

Highest expected gain:

- fused CUDA path
- paged KV
- graph capture
- device-native metadata and workspaces

This is the most likely place to unlock major real throughput and latency gains.

### Priority 2: Collective and transport fast paths

Once local execution improves, communication costs will become more obvious.

### Priority 3: Control-plane decoupling during decode

Reduce coordination tax and DB pressure during active serving.

### Priority 4: Recovery/export separation from live state

Important to keep resilience without contaminating the inner execution design.

### Priority 5: Metal-specific and CPU-specific targeted tuning

Still important, but should follow the core architectural cleanup.

## What This Is Not

It is worth explicitly stating what this tune-up is not trying to do.

This is not:

- removing Mesh’s distributed identity
- turning Mesh into a generic local server
- deleting session semantics because a single-node engine does not need them
- removing scheduling intelligence because it is inconvenient for benchmarking
- pretending resilience does not matter

This is also not:

- cargo-culting another engine’s design without regard for Mesh’s actual job
- assuming a single-node benchmark worldview should dominate every decision

The point is not to become someone else’s engine.

The point is to make this engine much harder to beat while still solving the
same distributed problem.

## Bottom Line

The review strongly supports the following conclusion:

Mesh/ZIP should keep its distributed orchestration purpose and product
differentiators, but it should stop treating its current local executor as a
mostly-fixed foundation.

The local executor, live KV design, graph story, and likely the collective fast
path are all fair game for aggressive replacement or deep refactor.

The highest-confidence direction is:

- preserve Mesh’s distributed control and session model
- isolate recovery and checkpoint concerns from the live hot path
- rebuild the high-performance local execution core in the style of `rvllm`’s
  best ideas

If that is done well, Mesh/ZIP does not lose its identity.

It becomes more itself:

- still distributed
- still resumable
- still topology-aware
- still operator-visible
- but far faster and far less burdened by avoidable local execution overhead

That is the target this document recommends.

## Orchestrated Implementation Plan

This section turns the tune-up into an execution document for multiple backend
implementation agents working in parallel.

The intent here is not just to list tasks. The intent is to make the work
parallelizable without collision.

Every agent below is being assigned a distinct area of ownership. Each agent is
expected to:

- stay inside their boundary unless a cross-boundary change is explicitly called
  for in this document
- leave clear interfaces for adjacent workstreams
- avoid re-litigating the product goal of Mesh/ZIP
- preserve the distributed purpose of the engine while making its performance
  ceiling materially higher

As orchestrator guidance, the most important global rule is:

- no agent should independently redesign the entire system

Each agent should improve their assigned layer while preserving compatibility
with the other planned workstreams.

### Global Coordination Rules

- [ ] Preserve Mesh/ZIP’s distinct purpose as a distributed inference engine.
- [ ] Do not collapse the architecture into a single-node-only design.
- [ ] Do not delete explicit session semantics, KV ownership semantics, or
  scheduler visibility semantics just because a local executor would not need
  them.
- [ ] Treat the current CPU/correctness-oriented paths as valuable, even if the
  GPU fast path is replaced.
- [ ] Prefer additive or isolated replacement paths first, then prune obsolete
  machinery once correctness and performance are proven.
- [ ] Keep all hot-path changes benchmarkable and observable.
- [ ] Minimize overlap in edited files wherever possible.
- [ ] Where overlap is unavoidable, align on interface contracts first and let
  one agent own the shared boundary.
- [ ] Keep recovery/cold-path features from dictating live execution layout
  decisions unless absolutely necessary.
- [ ] Escalate cross-agent interface mismatches early instead of patching around
  them locally.

### Recommended Execution Order

The workstreams can run in parallel, but they do not all have equal dependency
weight.

- [ ] Agent 1 should define the target local executor architecture first.
- [ ] Agent 2 should define the target live KV layout in close coordination with
  Agent 1.
- [ ] Agent 3 should define capture, bucket, and workspace invariants once
  Agents 1 and 2 have stabilized the execution shape.
- [ ] Agent 4 should evolve the collective and transport fast path in a way that
  can serve the new executor without waiting for every control-plane change.
- [ ] Agent 5 should reduce control-plane and decode-loop coupling without
  redesigning the executor internals owned by Agents 1 through 3.
- [ ] Agent 6 should isolate checkpoint and failover conversion paths from live
  executor state without changing scheduling policy semantics.
- [ ] Agent 7 should keep benchmarking, measurement, and verification in lockstep
  with all other workstreams so the refactor does not become a faith-based
  rewrite.

### Shared Definitions

For clarity, the following terms are used consistently in the work assignments:

- **live executor**: the code path used during active prefill/decode on a node
- **live KV format**: the in-memory cache representation used by active kernels
- **transfer KV format**: checkpoint/export/import representation used for
  regroup, failover, and persistence
- **control-plane coupling**: any scheduler, DB, polling, lease, or reporting
  behavior that directly increases the cost of active token generation
- **fast path**: the production-intended high-performance execution path for
  serious GPU-backed serving
- **fallback path**: correctness-first or compatibility-first execution that is
  allowed to be slower

## Agent Workstreams

## Agent 1: Local Executor Architecture Owner

You own the redesign of the node-local execution core. Your job is to replace
the current high-level GPU-serving execution shape with a purpose-built local
executor that can realistically approach the execution discipline seen in
`rvllm`, while still fitting inside Mesh/ZIP’s distributed system. You are not
responsible for scheduler policy, control-plane DB semantics, or checkpoint
strategy. You are responsible for defining what the fast path actually is on a
worker once work has already been assigned. You should assume that explicit
sessions, distributed ownership, and remote orchestration remain part of the
product, but you should not let those concerns pollute the inner shape of the
per-layer executor more than necessary. Your output should make it possible for
the system to preserve distributed identity while replacing the current
Candle-centric fast path with a much tighter execution substrate. Coordinate
closely with Agent 2 on KV layout, with Agent 3 on graph/bucket invariants, and
with Agent 4 on collective requirements. Do not redesign checkpoint export,
failover policy, or scheduler ranking logic yourself; those belong to other
agents.

### Agent 1 checklist

- [ ] Define the target fast-path executor boundary for worker-local prefill and
  decode.
- [ ] Decide which current `agent/src/inference/forward_pass.rs` behaviors stay
  as fallback/correctness paths and which are superseded by the new fast path.
- [ ] Produce a concrete layer execution plan for the GPU fast path:
  - fused norm/quant stages
  - packed or fused projection stages
  - fused RoPE + KV write stages
  - paged attention integration
  - fused residual/MLP pointwise stages
  - device-side sampling tail where possible
- [ ] Define how microbatch decode should map into the new executor without
  reintroducing high-level orchestration overhead.
- [ ] Ensure the new executor still supports explicit prefill and decode phases.
- [ ] Preserve compatibility with session-level accounting and progress surfaces.
- [ ] Define the provider split:
  - CPU fallback path
  - Metal fast path expectations
  - CUDA fast path expectations
- [ ] Specify the executor-facing interface contract that other layers must use.
- [ ] Identify current code that should become fallback-only.
- [ ] Identify current code that should be retired once the new path is proven.

### Agent 1 boundaries

- [ ] Do not own the canonical live KV memory layout; that belongs to Agent 2.
- [ ] Do not own graph capture pool design or bucket hashing mechanics; that
  belongs to Agent 3.
- [ ] Do not own transport or collective implementation details; that belongs to
  Agent 4.
- [ ] Do not own control-plane lease, polling, or DB redesign; that belongs to
  Agent 5.
- [ ] Do not own checkpoint/export/import semantics; that belongs to Agent 6.

### Agent 1 border notes

- If you need metadata, workspace, or bucket shapes to make the executor
  plausible, define the required contract and hand that contract to Agent 3.
- If you need specific KV page/block assumptions, specify them clearly and let
  Agent 2 own the implementation.
- If your execution plan assumes specific collective granularity or overlap
  points, write them down for Agent 4 rather than implementing transport-side
  changes ad hoc.

## Agent 2: Live KV Layout And Cache Runtime Owner

You own the redesign of the live KV cache representation used during active
execution. Your mission is to separate “the KV layout the kernels want” from
“the KV blob recovery wants.” The current system is too influenced by
checkpoint/export needs in its live execution structures, and that is likely a
hard performance ceiling. You should define and implement a live KV format that
is paged, block-oriented, static enough to work with top-end attention kernels,
and explicit enough that decode and prefill can operate on it with minimal
translation overhead. At the same time, you must not break the distributed
system’s ability to own, transfer, and resume sessions. That means you are not
deleting exportability; you are moving exportability out of the live format’s
primary design constraints. Coordinate very closely with Agent 1 so the executor
and KV layout fit each other, and with Agent 6 so checkpoint/export/import can
convert into and out of the new live format cleanly. Do not take over graph
capture design or transport logic unless a narrow interface needs to be
specified.

### Agent 2 checklist

- [ ] Define the canonical live KV format for active decode/prefill.
- [ ] Make the live format paged and block-table-oriented unless a better
  kernel-native format is justified with equal clarity and performance.
- [ ] Specify page size, block-table structure, per-layer layout, and required
  stride/offset invariants.
- [ ] Remove export/checkpoint-driven design constraints from the live
  representation.
- [ ] Define the boundary between:
  - live execution KV
  - transfer/checkpoint KV
- [ ] Refactor append/update logic so active decode is optimized around the new
  live format rather than generic growable host-style cache behavior.
- [ ] Ensure multi-session decode batching can read/write live KV without
  repeated format conversion.
- [ ] Define the minimal metadata the executor needs per sequence/session.
- [ ] Ensure the live format can describe ownership and residency in a way the
  distributed system can still reason about.
- [ ] Document the conversion hooks Agent 6 will need for export/import flows.

### Agent 2 boundaries

- [ ] Do not redesign executor kernel sequencing; that belongs to Agent 1.
- [ ] Do not redesign graph bucket capture policy; that belongs to Agent 3.
- [ ] Do not redesign checkpoint policy semantics; that belongs to Agent 6.
- [ ] Do not redesign transport framing or collective reuse; that belongs to
  Agent 4.

### Agent 2 border notes

- If you need the executor to consume specific page/block metadata shapes, lock
  those with Agent 1 before implementation spreads.
- If the live KV format introduces bucket-sensitive metadata, expose that
  contract to Agent 3 so capture/replay assumptions are stable.
- If ownership/residency surfaces need schema-visible changes, define those
  clearly for Agent 5 and Agent 6 rather than making control-plane changes
  directly unless they are narrowly required by your implementation.

## Agent 3: Buckets, Workspaces, And Graph Capture Owner

You own the execution invariants that turn the new fast path into a stable,
preplanned runtime instead of a looser sequence of per-step assembly work. Your
job is to introduce the infrastructure discipline that `rvllm` exhibits around
preallocated arenas, batch buckets, metadata layout stability, and graph-safe
replay. You are not just implementing “CUDA graphs” as a checkbox. You are
defining the execution environment in which graph capture is actually safe and
reliable. That means stable bucket sets, stable workspace contracts, stable
metadata layout hashing, and a refusal to let capture-unsafe dynamic behavior
creep into the hot path. Coordinate with Agent 1 for execution shape and with
Agent 2 for KV metadata layout. You should avoid taking over kernel logic,
control-plane semantics, or checkpoint conversion behavior. Your job is to make
the new fast path structurally repeatable and graph-capable.

### Agent 3 checklist

- [ ] Define supported decode buckets.
- [ ] Define supported prefill buckets or phased prefill bucketing strategy.
- [ ] Define the metadata layout per bucket.
- [ ] Define workspace requirements per bucket.
- [ ] Introduce stable allocation and arena rules for fast-path execution.
- [ ] Ensure graph capture safety rules are explicit and enforceable.
- [ ] Define layout hashing or equivalent replay-validation logic.
- [ ] Ensure capture/replay failure modes are typed and diagnosable.
- [ ] Eliminate or isolate capture-unsafe realloc or metadata drift behavior in
  the hot path.
- [ ] Make bucket assumptions visible to the benchmark and validation workflows.
- [ ] Provide interfaces that let Agent 1 plug executor launches into capture
  cleanly.

### Agent 3 boundaries

- [ ] Do not own the layer kernel sequence itself; that belongs to Agent 1.
- [ ] Do not own the KV memory model itself; that belongs to Agent 2.
- [ ] Do not own collective transport behavior; that belongs to Agent 4.
- [ ] Do not own checkpoint export/import conversion; that belongs to Agent 6.

### Agent 3 border notes

- If Agent 1 needs additional bucket variants for low-batch or edge cases, make
  those tradeoffs explicit rather than silently broadening bucket scope.
- If Agent 2’s KV metadata shape is too unstable for capture, force the issue
  early and drive a contract resolution rather than building around drift.
- If provider-specific behavior diverges sharply, keep the invariant framework
  shared where possible but allow provider-specialized capture strategies rather
  than forcing a fake uniformity.

## Agent 4: Collective And Transport Fast-Path Owner

You own the evolution of the distributed execution substrate beneath the local
executor. Your task is to improve the collective and tensor transport path so it
does not remain the next hard ceiling once the executor gets faster. The
existing ring path is useful and should remain a portable baseline, but the
performance objective requires you to make the serving-group communication path
more efficient, more stable, and more specialized for repeated inference use
instead of generic framed exchange. You should think in terms of device-native
buffers, lower host mediation, persistent serving channels, reduced copy count,
and more principled overlap opportunities. You are not responsible for deciding
how the scheduler chooses work, nor how checkpoint state is represented. You are
responsible for ensuring that once work is assigned, communication does not
waste the gains from the faster executor. Coordinate with Agent 1 on collective
granularity expectations and with Agent 2 on live KV movement assumptions.

### Agent 4 checklist

- [ ] Preserve the existing portable collective path as a baseline/fallback.
- [ ] Define the target serving-group fast path for repeated inference traffic.
- [ ] Reduce host-mediated overhead in collective operations where feasible.
- [ ] Increase reuse of serving-group channels and connections.
- [ ] Align collective buffer behavior with the new executor and live KV design.
- [ ] Identify opportunities for provider-specialized collective behavior:
  - CPU-safe baseline
  - Metal-aware path
  - CUDA-oriented high-throughput path
- [ ] Reduce unnecessary framing/copy overhead in active decode traffic.
- [ ] Evaluate where collective steps can overlap or pipeline better with
  execution.
- [ ] Preserve observability and timeout diagnostics without letting them bloat
  the core fast path.
- [ ] Ensure the fast path remains compatible with multi-session decode cohorts.

### Agent 4 boundaries

- [ ] Do not redesign scheduler ownership or lease logic; that belongs to
  Agent 5.
- [ ] Do not redesign the local executor’s kernel graph; that belongs to
  Agent 1.
- [ ] Do not redefine live KV layout; that belongs to Agent 2.
- [ ] Do not redesign recovery export/import semantics; that belongs to Agent 6.

### Agent 4 border notes

- If the new executor assumes different collective frequency or payload shape,
  require that contract from Agent 1 rather than reverse-engineering it after
  the fact.
- If live KV movement over the network needs page-aware transfer behavior, align
  with Agent 2 and Agent 6 so transport, ownership metadata, and recovery
  semantics do not diverge.
- If you need control-plane-visible capability flags or transport tiers to
  evolve, coordinate with Agent 5 instead of taking over scheduler-side policy.

## Agent 5: Control-Plane Decoupling And Runtime Coupling Owner

You own the problem of active decode being too exposed to scheduler, polling,
lease, and DB pressure. Your job is not to simplify away the scheduler or erase
the explicit control-plane state model. Your job is to preserve that model while
moving it farther from token-by-token execution cost. That means looking hard at
claim, renew, release, progress reporting, scheduler visibility, DB writes, and
queue-state mutation frequency. The benchmark log already shows that SQLite lock
pressure is not hypothetical, so you should treat this as a real performance and
stability issue. The desired result is that Mesh keeps its ownership and
observability semantics while the worker can run a decode interval with much
less external churn. Coordinate with Agent 4 if transport capability surfaces
need to be made visible to the scheduler, and with Agent 6 if failover/recovery
state transitions need cleaner decoupling. Do not redesign the executor or KV
internals yourself.

### Agent 5 checklist

- [ ] Map every control-plane interaction that occurs during active decode.
- [ ] Classify which interactions are essential at token cadence and which are
  not.
- [ ] Reduce DB write amplification during active decode.
- [ ] Reduce polling or observation churn where possible.
- [ ] Revisit lease renewal cadence and lease ownership mutation cost.
- [ ] Preserve queue visibility and operator introspection while coarsening
  hot-path mutation where possible.
- [ ] Preserve fairness and cohort semantics.
- [ ] Ensure assignment, ownership, and resume semantics still work with a more
  autonomous worker-local execution window.
- [ ] Make any schema/API changes needed to support lower hot-path pressure.
- [ ] Preserve scheduler correctness while reducing interference with active
  serving.

### Agent 5 boundaries

- [ ] Do not replace executor internals; that belongs to Agent 1.
- [ ] Do not redesign live KV format; that belongs to Agent 2.
- [ ] Do not redesign graph capture or bucket logic; that belongs to Agent 3.
- [ ] Do not redesign checkpoint serialization itself; that belongs to Agent 6.

### Agent 5 border notes

- If the worker needs a coarser autonomy window, make that explicit in API and
  lease semantics rather than relying on hidden behavior.
- If transport or provider capability should influence scheduling decisions more
  directly, coordinate with Agent 4 to keep capability surfaces coherent.
- If failover state mutations are creating too much hot-path interference,
  isolate the mutation path and hand recovery-specific changes to Agent 6 rather
  than weakening the scheduler model.

## Agent 6: Checkpoint, Failover, And KV Conversion Owner

You own the resilience layer. Your mission is to preserve Mesh/ZIP’s
differentiating recovery and regroup capabilities while preventing those
capabilities from continuing to dictate the live executor’s in-memory design.
That means you should treat checkpointing, export/import, remote session resume,
failover, and regroup as cold-path or side-path responsibilities that integrate
with the live runtime through clear conversion boundaries. You are not here to
delete resilience. You are here to make resilience architecturally cleaner. You
should work very closely with Agent 2 on how live KV converts into transfer KV,
and with Agent 5 on how recovery state mutates control-plane records. Do not
redefine the scheduler’s product semantics, and do not invent your own live KV
model separate from Agent 2. Your work should make the system more recoverable
without forcing the fast path to carry generic checkpoint-first assumptions.

### Agent 6 checklist

- [ ] Define the boundary between live KV and transfer/checkpoint KV.
- [ ] Refactor checkpoint export/import around explicit conversion points.
- [ ] Ensure regroup and failover can still recover sessions under the new live
  KV format.
- [ ] Preserve remote checkpoint download/import flows where still needed.
- [ ] Reduce the amount of live executor state shaped primarily by recovery
  concerns.
- [ ] Preserve session continuity semantics through pause/resume/regroup paths.
- [ ] Validate that ownership, residency, and checkpoint metadata still align
  with the new live executor contracts.
- [ ] Ensure failover state transitions remain observable and debuggable.
- [ ] Keep shrink, replace, and resume-ready semantics intact unless there is an
  explicit system-level decision to alter them.

### Agent 6 boundaries

- [ ] Do not own the live executor layer sequence; that belongs to Agent 1.
- [ ] Do not own the canonical live KV layout; that belongs to Agent 2.
- [ ] Do not own graph capture mechanics; that belongs to Agent 3.
- [ ] Do not own transport fast-path redesign; that belongs to Agent 4.
- [ ] Do not own control-plane scheduling policy; that belongs to Agent 5.

### Agent 6 border notes

- If conversion costs become large enough to matter operationally, document them
  explicitly rather than trying to leak transfer concerns back into the live KV
  model.
- If resume semantics need new metadata, work with Agent 5 so schema and API
  changes land coherently.
- If page-level or slice-level transfer behavior depends on the new live KV
  model, align with Agent 2 and Agent 4 before implementing partial-transfer
  behavior.

## Agent 7: Benchmarking, Validation, And Performance Proof Owner

You own the proof loop. Your job is to ensure the refactor stays grounded in
measured reality rather than architectural optimism. Every major tune-up claim
in this document must become something that can be observed, benchmarked,
regressed, and validated. You should preserve the repo’s Fozzy-first testing
policy, but you also need to push beyond correctness and make sure there are
repeatable measurements for local executor speed, collective cost, control-plane
pressure, and end-to-end distributed serving outcomes. You are not responsible
for implementing the new executor or scheduler changes, but you are responsible
for making it impossible for the system to drift without performance evidence.
Coordinate with every other agent, especially Agents 1 through 5, because their
work must remain visible in metrics and test scenarios. Do not take ownership of
their implementation logic, but do require measurable success criteria.

### Agent 7 checklist

- [ ] Define benchmark suites for:
  - local executor throughput/latency
  - decode batch scaling
  - prefill scaling
  - collective overhead
  - control-plane pressure under active serving
  - checkpoint/export/import overhead
  - end-to-end distributed inference throughput and TTFT
- ✅ Extend Fozzy scenarios to cover new fast-path and coordination behavior.
- ✅ Keep at least one real trace recorded and validated for each major active
  goal area.
- ✅ Add host-backed checks where they are meaningful for runtime delivery.
- [ ] Make regressions attributable by capturing enough telemetry to separate:
  - executor loss
  - collective loss
  - scheduler/control-plane loss
  - recovery-path interference
- [ ] Update or replace performance logs such as `ZIPPERF.md` as the system
  evolves.
- [ ] Ensure old benchmark paths are clearly marked invalid when architectural
  assumptions change.
- [ ] Give each other agent measurable acceptance targets.

### Agent 7 boundaries

- [ ] Do not redesign executor internals; that belongs to Agent 1.
- [ ] Do not redesign KV layout; that belongs to Agent 2.
- [ ] Do not redesign graph capture or transport internals directly unless small
  instrumentation changes are needed.
- [ ] Do not take over scheduler policy or recovery semantics.

### Agent 7 border notes

- Require benchmark hooks and observability from the other agents early so proof
  is built in, not bolted on later.
- If a workstream cannot be meaningfully measured yet, force it to define proxy
  metrics rather than letting the implementation proceed unobserved.

## Cross-Agent Interface Checklist

These are the shared seams that must be aligned explicitly.

- [ ] Agent 1 and Agent 2 agree on live executor to live KV contract.
- [ ] Agent 1 and Agent 3 agree on bucket shape, metadata stability, and
  workspace assumptions.
- [ ] Agent 1 and Agent 4 agree on collective frequency, payload shape, and
  overlap assumptions.
- [ ] Agent 2 and Agent 6 agree on live KV to transfer KV conversion contract.
- [ ] Agent 4 and Agent 5 agree on any transport capability or serving-group
  visibility surfaces exposed to scheduling.
- [ ] Agent 5 and Agent 6 agree on control-plane mutation shape for regroup,
  failover, resume, and ownership transitions.
- [ ] Agent 7 has benchmark hooks and measurement targets for every other
  workstream before the work is considered done.

## Done Criteria

The tune-up is not complete when code has merely been refactored. It is complete
only when the system still serves the same distributed purpose and the new
architecture is measurably better.

## Verified Status

The following items are verified as of the current orchestrator integration
pass.

- ✅ Fozzy deterministic deep validation passed for:
  - `tests/serving_production_integrated.fozzy.json`
  - `tests/transport_collective_evolution.fozzy.json`
  - `tests/incremental_decode_recovery.fozzy.json`
  - `tests/continuous_batching.fozzy.json`
- ✅ Fozzy strict deterministic scenario testing passed for:
  - `tests/serving_production_integrated.fozzy.json`
  - `tests/transport_collective_evolution.fozzy.json`
  - `tests/incremental_decode_recovery.fozzy.json`
  - `tests/continuous_batching.fozzy.json`
- ✅ A real Fozzy trace was recorded for
  `tests/serving_production_integrated.fozzy.json`.
- ✅ The recorded trace passed:
  - `fozzy trace verify --strict`
  - `fozzy replay`
  - `fozzy ci`
- ✅ A host-backed Fozzy run passed for
  `tests/serving_production_integrated.fozzy.json` using:
  - `--proc-backend host`
  - `--fs-backend host`
  - `--http-backend host`
- ✅ `meshnet` workspace Rust tests passed:
  - `cargo test --workspace -- --test-threads=1`
- ✅ The integrated `meshnet` runtime surfaces now have passing Rust coverage
  for the major touched areas:
  - fast-path planning and workspace invariants
  - live paged KV behavior and KV snapshot round-trips
  - decode batching and runtime memory budgeting
  - checkpoint export/import and checkpoint-mediated recovery
  - serving transport lane reuse and collective evolution behavior
  - scheduler, regroup, and failover API/control-plane behavior
- ✅ The OSS `zip` repo was updated to the same validated runtime surface where
  its module structure matches the local `meshnet` engine core.
- ✅ The OSS `zip` repo test suite passed:
  - `cargo test`

The following items are intentionally **not** marked verified yet:

- raw end-to-end performance uplift versus the previous baseline
- quantified TTFT improvement
- quantified tok/s improvement
- quantified collective-overhead reduction under real distributed load
- quantified control-plane contention reduction under live production-like load

Those require benchmark-specific measurement work beyond correctness and
integration validation.

### Product integrity done criteria

- [ ] Mesh/ZIP still clearly operates as a distributed inference engine.
- [ ] Explicit sessions, placement, ownership, and distributed coordination
  semantics still exist.
- [ ] Regroup, failover, and resume semantics still work at the product level.
- [ ] Scheduler and operator visibility remain meaningful.

### Performance done criteria

- [ ] The local GPU fast path is materially more efficient than the current
  Candle-centric path.
- [ ] Live KV behavior is materially better aligned to kernel-native decode.
- [ ] Bucketed, graph-safe execution exists where appropriate.
- [ ] Collective overhead is reduced or better hidden behind improved fast paths.
- [ ] Control-plane interference during active decode is measurably reduced.
- [ ] End-to-end distributed inference is measurably improved, not just
  single-node microbenchmarks.

### Architecture done criteria

- [ ] Live executor concerns are cleanly separated from recovery/export concerns.
- [ ] Control-plane and scheduler concerns are pushed farther from token hot
  paths.
- [ ] Provider specialization is sharper and no longer limited by a
  lowest-common-denominator GPU structure.
- [ ] The old path is either demoted to fallback status or retired intentionally.
