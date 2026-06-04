# Mesh / Zip Target Architecture

## Status

Authoritative architecture plan

## Objective

Build the serving architecture that matches Mesh's actual use case:

- heterogeneous fleets
- mixed compute and memory profiles
- direct preference for peak decode throughput
- large-session and long-context support
- failure recovery without putting recovery traffic on the decode critical path

The success condition is simple:

> Adding machines must increase useful throughput, increase useful capacity, or increase resilience. Adding a machine must not automatically reduce decode tok/s.

---

# North Star

Mesh is not a generic distributed ring runtime.

Mesh is a distributed inference system with explicit role separation.

That means:

- fast compute machines should increase decode throughput
- large-memory machines should increase context and model capacity
- durable or slower machines should improve checkpointing and recovery
- unreliable or relay-only machines should not silently poison the fast path

The architecture must reflect those facts directly.

---

# Core Architectural Rule

The unit of scheduling is:

`ExecutionGroup`

The unit of fault handling is:

`SessionExecutionGraph`

The unit of optimization is:

`Role-specific hot path`

Not every node in a session belongs in the synchronous decode collective.

That is the central rule that everything else follows from.

---

# Problem To Solve

Synchronous tensor-parallel decode is bounded by the slowest participant in the active collective.

That is acceptable only when all active decode participants are close in:

- compute speed
- memory bandwidth
- interconnect quality
- backend contract
- transport stability

It is not acceptable for a serving topology like:

`3090 + 4090 + M-series Mac + CPU node`

if all of them are forced into one decode collective.

In that shape:

- the slowest device can set decode cadence
- the noisiest network path can set collective cadence
- large-memory helper nodes become decode liabilities
- aggregate fleet resources can grow while user-visible tok/s falls

That is the failure mode this architecture exists to eliminate.

---

# Required End State

Every session is executed as a graph of role-specific groups.

```text
SessionExecutionGraph
  prefill_group
  decode_group
  kv_group
  checkpoint_group
  recovery_group
  overflow_group
```

Each group has its own:

- membership
- admission rules
- performance objective
- transport policy
- failure policy

Only the decode group is allowed to sit on the strict synchronous token critical path.

---

# Execution Groups

## 1. Decode Group

Purpose:

`maximum steady-state decode tok/s`

Rules:

- homogeneous or near-homogeneous execution only
- direct transport strongly preferred
- synchronous collectives allowed
- only participants that improve or preserve decode throughput belong here
- width only increases when it increases effective throughput

Required properties:

- stable backend contract
- stable collective shape
- low transport variance
- bounded participant count
- explicit admission based on measured throughput, not only static tiering

Non-goals:

- memory maximization
- heterogeneous inclusiveness
- opportunistic node participation

If a machine slows the decode cadence, it does not belong in the decode group.

## 2. Prefill Group

Purpose:

`maximum prompt ingestion subject to TTFT goals`

Rules:

- may differ from decode membership
- may be broader than decode when prompt throughput or prompt fit improves
- may use different layout or batching policy than decode
- may hand off KV to the decode path after prefill completes

Required properties:

- explicit prefill cost model
- prompt-length-aware planning
- prompt cache reuse
- no requirement that prefill and decode have identical participants

Prefill is throughput-sensitive, but it is not allowed to distort decode architecture.

## 3. KV Group

Purpose:

`persistent live session state ownership`

Rules:

- owns KV residency metadata and live ownership contracts
- may include machines that never decode
- must support large-memory specialization
- must not require decode workers to materialize all remote state unless decode admission requires it

Responsibilities:

- live KV ownership
- residency placement
- slice-level tracking
- eviction policy
- prompt cache residency
- locality hints back to the scheduler

The KV group is a memory service layer for the serving system.

## 4. Checkpoint Group

Purpose:

`durable export and import without decode-path interference`

Rules:

- checkpoint traffic is cold-path traffic
- checkpoint serialization format is optimized for transfer and recovery, not live decode execution
- checkpoint work must be isolated from blocking decode collectives

Responsibilities:

- session snapshots
- durable checkpoint upload
- checkpoint retention and cleanup
- export verification

Checkpointing exists to support recovery and continuity, not to shape the decode fast path.

## 5. Recovery Group

Purpose:

`session resurrection and continuity after disruption`

Rules:

- may use different machines than the original decode group
- may rehydrate from live KV, checkpoint, or mixed state
- must not force normal decode to carry recovery-first complexity

Responsibilities:

- regroup resume
- replacement participant activation
- shrink continuation
- recovery-rate governance

Recovery is a side-path, not the architecture center.

## 6. Overflow Group

Purpose:

`fit-first capacity expansion when model or context footprint exceeds fast-path capacity`

Rules:

- may include slower or memory-oriented nodes
- must not be confused with the throughput path
- must be explicitly chosen by runtime mode or fit requirement

Responsibilities:

- large model fit support
- large KV spill support
- non-throughput-oriented capacity extension

Overflow is allowed to trade speed for fit. Decode is not.

---

# Session Execution Graph

Each session moves through explicit phases with explicit group ownership.

```text
submit
  -> prefill placement
  -> prefill execution
  -> kv ownership confirmation
  -> decode placement
  -> decode execution
  -> background checkpointing
  -> regroup or recovery when needed
  -> completion or eviction
```

The graph must support:

- prefill and decode on different participants
- live KV ownership surviving decode-group changes
- checkpoint export in the background
- regroup without destroying session identity
- decode continuity after participant loss

The graph must not collapse back into:

- one ring
- one topology
- one collective shape
- one ownership model

---

# Performance Invariants

These are non-negotiable.

## Invariant 1

Decode throughput is governed only by the active decode group.

KV nodes, checkpoint nodes, recovery nodes, and overflow nodes are not allowed to become decode participants unless the planner intentionally admits them.

## Invariant 2

Wider decode groups are not automatically better.

The planner must choose decode width only when wider membership produces higher effective tok/s.

## Invariant 3

Memory-rich and compute-poor nodes increase capacity, not decode latency.

Their default role is KV, checkpoint, recovery, or overflow.

## Invariant 4

Checkpoint and recovery traffic are cold-path or side-path operations.

They may overlap decode safely, but they may not block the main decode cadence.

## Invariant 5

Session continuity is preserved through explicit state transfer contracts, not by requiring identical ring membership forever.

## Invariant 6

Transport quality is a first-class performance input.

Relay or degraded paths must heavily constrain decode admission.

## Invariant 7

The planner must optimize for measured behavior, not only declared capability.

Static tiering is insufficient.

---

# Node Capability Model

The planner needs an explicit capability profile per node.

```rust
NodeCapabilityProfile {
    provider,
    backend_contract_hash,
    fast_path_eligible,
    compute_score,
    memory_score,
    network_score,
    transport_stability_score,
    checkpoint_score,
    kv_score,
    failure_risk_score,
    observed_decode_toks_per_sec,
    observed_prefill_toks_per_sec,
    observed_collective_wait_share,
    observed_batch_fill_ratio,
    observed_recovery_load,
}
```

This profile drives role assignment.

The planner must stop treating all useful machines as decode candidates.

Instead:

- compute-heavy machines bias toward decode
- memory-heavy machines bias toward KV and overflow
- stable durable machines bias toward checkpoint and recovery
- degraded transport machines bias away from decode

---

# Scheduler Architecture

The scheduler must become an execution-graph planner, not just a decode lease allocator.

## Planner output

```rust
SessionExecutionPlan {
    runtime_mode,
    prefill_group,
    decode_group,
    kv_group,
    checkpoint_group,
    recovery_group,
    overflow_group,
    transfer_plan,
    regroup_policy,
}
```

## Scheduler responsibilities

- choose group membership by role
- decide whether prefill and decode should share participants
- decide whether KV is co-located, remotely owned, or partially replicated
- choose checkpoint/export policy
- choose regroup options before failures happen
- score decode width against real throughput and collective risk
- avoid opening decode groups that lower effective throughput

## Scheduler anti-goals

- maximizing participant count
- blindly preferring broad topologies
- treating fit-first capacity nodes as throughput nodes
- conflating session ownership with decode ownership

---

# Decode Planner Rules

The decode planner is the architectural center.

It must:

- prefer homogeneous fast-path cohorts
- prefer direct-connected cohorts
- prefer lower collective wait share
- prefer lower variance cohorts
- reject nodes that reduce effective step rate
- cap decode width when communication cost dominates compute gain

It must not:

- include a slower node just because it has free memory
- include a node just because it has the right shard span
- include relay-only nodes in throughput-first decode unless nothing else is legal

The decode objective is:

`maximize sustained delivered tokens per second`

not:

`maximize machines involved`

---

# Prefill Planner Rules

Prefill planning is separate from decode planning.

It must account for:

- prompt length
- prompt cache hit potential
- prefill batch opportunities
- TTFT goals
- handoff cost into decode

Prefill can use a broader topology than decode when that improves total service quality.

But if prefill breadth creates a handoff pattern that lowers overall session throughput, the planner must reject it.

---

# KV Architecture

KV must be treated as a live service layer with explicit contracts.

## Required characteristics

- live residency records
- slice-level ownership
- prompt cache linkage
- remote-reference capability
- explicit owner and replica semantics
- eviction and pinning policy

## Required separation

There are two distinct KV representations:

- live decode KV
- transfer or checkpoint KV

These are not the same thing and must not be forced into the same layout constraints.

## Target behavior

- decode workers use live KV optimized for append-only token stepping
- KV/export paths convert at explicit boundaries
- remote or large-memory nodes can own session memory without joining decode collectives

---

# Checkpoint Architecture

Checkpointing is a durability system, not a decode execution model.

## Requirements

- explicit export boundary from live KV
- explicit import boundary into recovery state
- rate-limited background execution
- operator-visible success and failure metrics
- retention and garbage collection policy

## Hard rule

Checkpoint machinery must not shape the live decode data structures more than necessary.

Live execution wins. Checkpoint conversion adapts around it.

---

# Recovery And Regroup

Recovery and regroup are important product features, but they belong off the hot path.

## Required capabilities

- detect lost decode participants
- pause impacted sessions cleanly
- select replacement or shrink topology
- preserve session identity
- resume decode from live state or checkpoint state

## Architectural rule

Regroup mutates group membership and ownership contracts.

It does not redefine the decode engine into a recovery-first architecture.

## Required outcomes

- replacement when throughput-positive and legal
- shrink continuation when replacement is not worth the cost
- restart only when neither replacement nor shrink is viable

---

# Collective Architecture

Decode collectives remain synchronous.

That is acceptable only if their membership is tightly controlled.

## Collective classes

### Synchronous

Used for:

- decode group execution

Examples:

- all-reduce
- reduce-scatter
- all-gather

### Asynchronous or background

Used for:

- KV transfer
- checkpoint export
- recovery materialization
- overflow movement

These operations are not allowed to define decode cadence.

## Required collective evolution

- group-scoped collectives, not network-wide assumptions
- transport-aware overlap policy
- strict decode-lane priority
- background transfer isolation
- no implicit participation by non-decode roles

---

# Runtime Modes

Runtime modes stay useful, but their meaning must become stricter.

## Throughput First

Goal:

`maximize delivered decode tok/s`

Policy:

- only fastest legal decode cohort
- strong direct-transport preference
- minimal decode width beyond the profitable point
- memory-only nodes pushed to KV and checkpoint roles

## Latency First

Goal:

`minimize response latency`

Policy:

- smallest high-quality decode cohort that satisfies execution legality
- aggressive TTFT-aware prefill planning

## Fit First

Goal:

`serve the largest model or context that the fleet can support`

Policy:

- allow overflow and memory-oriented participation
- accept throughput tradeoffs explicitly

## Resilient Edge

Goal:

`maintain service under instability`

Policy:

- stronger checkpointing
- conservative decode admission under degraded transport
- stronger regroup and shrink preferences

---

# Specialized Fleet Strategy

## GPU workstations and servers

Primary role:

`decode and high-value prefill`

## Apple Silicon large-memory machines

Primary role:

`KV, checkpoint, recovery, overflow`

Secondary role:

`prefill when beneficial`

Default non-role:

`throughput-first decode participant`

## CPU-only nodes

Primary role:

`recovery, checkpoint, auxiliary capacity, fit-first overflow`

Not a default decode participant.

---

# What Must Change In The Codebase

## 1. Promote session plans from two groups to full execution graphs

Today the architecture centers on `prefill` and `decode`.

We need explicit first-class planning and persistence for:

- KV group
- checkpoint group
- recovery group
- overflow group

## 2. Make decode admission strictly role-aware

Decode membership must be based on:

- backend compatibility
- transport quality
- observed throughput
- collective wait share
- width profitability

Not only legality and broad topology efficiency.

## 3. Make remote KV a real serving primitive

Remote KV must stop being only a control-plane concept plus checkpoint resume path.

We need:

- live ownership contracts
- replica readiness contracts
- explicit local materialization policy
- direct support for remote-owned session memory without forcing every resume through full checkpoint import

## 4. Separate live KV from checkpoint KV completely

The runtime must preserve a decode-optimized live layout and convert only at export/import boundaries.

## 5. Convert checkpointing into a true cold-path service

Checkpoint export, upload, and validation must be background isolated and rate governed.

## 6. Strengthen regroup around role continuity

Regroup must preserve:

- decode-group continuity
- KV ownership continuity
- recovery choice clarity

without dragging cold-path complexity into every decode step.

## 7. Make performance attribution scheduler-visible

The planner must reason about:

- batch size realization
- collective wait share
- transfer debt
- post-failover degradation
- decode cohort profitability

## 8. Enforce architecture through tests and benchmarks

We need proof that:

- adding a slow memory node does not lower throughput-first decode tok/s
- adding a fast compatible decode node can raise effective tok/s
- checkpoint traffic does not block decode cadence
- regroup preserves continuity without corrupting throughput behavior
- fit-first and throughput-first produce intentionally different topologies

---

# Acceptance Criteria

We are done when all of the following are true.

## Throughput correctness

- throughput-first decode excludes machines that reduce effective tok/s
- decode width scales only when net tok/s rises
- multi-session batching remains visible and effective on the chosen decode cohort

## Capacity correctness

- large-memory nodes can expand usable context and fit without poisoning decode
- overflow roles are explicit and observable
- KV ownership can live away from decode ownership

## Recovery correctness

- regroup, shrink, and replacement preserve session continuity
- recovery paths remain bounded and operator-visible
- checkpoint fallback is measurable, not silent

## Architectural correctness

- non-decode roles do not implicitly join synchronous decode collectives
- live KV and checkpoint KV are separate representations with explicit conversion points
- scheduling decisions are role-aware and metric-aware

---

# Final Statement

This is the architecture Mesh should implement.

It is optimized for the actual product:

- heterogeneous fleets
- peak decode performance
- large-memory assistance
- background durability
- explicit recovery

The desired behavior is not:

`every machine helps decode`

The desired behavior is:

`every machine helps the system in the role where it adds the most value`

Fast hardware increases throughput.

Large-memory hardware increases capacity.

Durable hardware improves continuity.

The scheduler decides which is which.
