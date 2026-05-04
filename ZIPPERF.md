# ZIPPERF

## Current Status

- Date: 2026-05-04
- Current local repo head when this benchmark framework was refreshed:
  - `39da3a5b`
- Current status:
  - runtime/core integration is validation-green
  - strict deterministic Fozzy scenarios are green
  - workspace Rust tests are green
  - OSS `zip` mirror has been resynced to the validated shared runtime surface
- What this file now is:
  - the benchmark contract for Agent 7
  - the source of truth for which perf claims are still unproven
  - the place where stale benchmark interpretations must be invalidated when architecture changes

## Agent 7 Benchmark Suite

The benchmark program is now split into distinct proof buckets so regressions
can be attributed instead of blended together.

### 1. Local executor throughput and latency

Purpose:

- determine whether the fast path materially improves node-local serving cost
- separate local executor wins from distributed transport effects

Required measurements:

- TTFT
- completion tok/s
- prompt tok/s
- fast-path decode plan rate
- replay-preferred versus layout-validated fast-path mix

Required command surfaces:

- `cargo test --workspace -- --test-threads=1`
- `fozzy test --det --strict tests/serving_production_integrated.fozzy.json --json`
- `mesh inference stats`

### 2. Decode batch scaling

Purpose:

- validate that multi-session decode batching is actually being admitted and used
- identify whether performance loss is due to batching gaps versus executor cost

Required measurements:

- average decode batch size
- peak decode batch size
- multi-session batch rate
- average deferred sessions per microbatch
- batch KV-token footprint

Primary observability surfaces:

- local inference stats JSON and CLI
- scheduler/operator status batch telemetry
- `tests/continuous_batching.fozzy.json`

### 3. Prefill scaling

Purpose:

- verify that prefill bucketing and workspace reuse are working as intended
- prevent regressions where decode improves but prefill silently worsens

Required measurements:

- prefill fast-path plan count
- prefill bucket ceiling usage
- arena reuse rate
- TTFT deltas under prompt-length growth

### 4. Collective and transport overhead

Purpose:

- isolate distributed communication cost from executor cost
- prove whether overlap and background transfer changes are helping

Required measurements:

- average all-reduce latency
- reduce-scatter and all-gather total time
- all-reduce send-wait share
- all-reduce receive-wait share
- collective transport share of total runtime
- checkpoint and bulk transfer byte volume

Primary observability surfaces:

- local inference stats JSON and CLI
- `tests/transport_collective_evolution.fozzy.json`

### 5. Control-plane pressure under active serving

Purpose:

- prove that active decode is less entangled with control-plane churn
- separate scheduler/DB loss from executor and transport loss

Required measurements:

- decode queue depth
- runnable/blocked/leased/active session mix
- deferred decode session count
- batch telemetry coverage
- checkpoint fallback transfer rate
- recent regroup counts
- recent recovery latency metrics

Primary observability surfaces:

- `/api/status/networks/:network_id/scheduler`
- `/api/status/jobs/:job_id/scheduler`
- control-plane scheduler event stream

### 6. Checkpoint, recovery, and KV conversion overhead

Purpose:

- preserve distributed product semantics without letting recovery costs hide in the hot path

Required measurements:

- checkpoints created
- recovery attempts
- recovery success rate
- recovery rejection rate
- recent average and peak recovery latency
- checkpoint fallback transfer count and rate

Primary observability surfaces:

- local inference stats JSON and CLI
- scheduler/operator status recovery metrics
- `tests/incremental_decode_recovery.fozzy.json`

### 7. End-to-end distributed inference

Purpose:

- establish whether the distributed engine itself is better, not just isolated components

Required measurements:

- successful distributed completion
- TTFT
- steady-state tok/s
- failure-free completion under serving topology
- comparison between same-host and cross-host serving topologies

## Attribution Rules

When a run regresses, classify it before calling it “slower.”

- executor loss:
  - low fast-path decode plan rate
  - low arena reuse
  - low multi-session batch rate without corresponding transport pressure
- collective loss:
  - elevated all-reduce latency
  - elevated send/receive wait share
  - elevated collective-runtime share
- scheduler/control-plane loss:
  - growing queue depth
  - higher blocked/leased churn
  - elevated deferred decode sessions with stable local executor counters
- recovery-path interference:
  - checkpoint fallback transfer growth
  - elevated recovery latency
  - increased recovery rejection rate or regroup churn

## Current Acceptance Gates

These are the minimum proof gates for claiming the tune-up is performance-positive.

- Local fast path:
  - fast-path decode plan rate should remain high on accelerated providers
  - multi-session batching should be observable, not theoretical
- Collective path:
  - send/receive wait share must not dominate runtime under healthy serving conditions
  - background checkpoint/bulk transfer behavior must not break blocking collective correctness
- Recovery path:
  - shrink/replace/resume-ready semantics must remain green under failover coverage
  - checkpoint fallback rate should be observable and trendable
- End-to-end:
  - no claim of TTFT or tok/s improvement is valid until a distributed completion baseline exists

## Invalidated Historical Read

The older April 2026 same-host benchmark notes below remain useful as incident
history, but they should no longer be treated as the current performance read.

Reasons they are stale:

- the runtime architecture has materially changed
- fast-path planning, paged KV semantics, collective overlap behavior, and
  recovery conversion boundaries have all moved since that snapshot
- the attribution metrics available now did not exist during that earlier run

Treat the remainder of this document as historical context until replaced by a
fresh successful distributed benchmark pass under the current runtime.

## Purpose

This file is the running performance and benchmark log for the Mesh/zip inference
engine. It is intended to preserve:

- benchmark topology
- model and prompt details
- exact job IDs
- observed throughput/latency numbers
- blocking bugs that invalidate a benchmark
- next optimization targets

## Environment

- Date: 2026-04-26
- Local host: Apple Silicon Mac
- Remote host: `saint@supercomputer.local` (Intel MacBook Air)
- Local repo: `/Users/deepsaint/Desktop/meshnet`
- Remote repo: `/Users/saint/Desktop/mesh-e2e`
- Local control-plane/network used for live benchmarking:
  - control plane home: `/Users/deepsaint/.meshbench/zipperf-cp`
  - network id: `zipperf-direct`
- Model:
  - `tinyllama-1.1b-chat-v1.0`

## Code State

- Local repo head during current live work:
  - `783ef6c9cca1f4f42da480f7039930122c28373f`
- Remote repo pulled forward to the same commit before rebuild.

## Local Fixes Applied During Benchmarking

### Apple Silicon memory-pressure fix

Fixed the runtime memory ownership path in:

- [agent/src/inference/coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs)
- [agent/src/inference/backend.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/backend.rs)
- [agent/src/inference/forward_pass.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/forward_pass.rs)

What changed:

- model residency is now shared across sessions instead of cloning full host
  weights per session
- Candle/device weights are now built once per loaded shard residency instead of
  once per active session
- runtime memory budgeting now includes shared model residency, not only prompt,
  generated-token, and KV bytes
- unused cached model residency is pruned when sessions are torn down

Why this matters:

- the previous runtime shape could duplicate model memory for every active
  session/backend
- on Apple Silicon unified memory, that could drive whole-system pressure even
  for a small model
- earlier same-host benchmark failures were therefore not only transport-related;
  they were also running on a runtime that materially under-accounted memory

Verification for this fix:

- `cargo check -p agent`
- `cargo test -p agent inference::coordinator::tests -- --nocapture`
- `cargo test -p agent inference::forward_pass::tests::test_local_incremental_decode_matches_full_recompute -- --nocapture`
- `cargo test --workspace -- --test-threads=1`
- `fozzy doctor --deep --scenario tests/continuous_batching.fozzy.json --runs 5 --seed 1337 --json`
- `fozzy doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 1337 --json`
- `fozzy test --det --strict tests/production_dispatch.fozzy.json tests/continuous_batching.fozzy.json --json`
- `fozzy run tests/continuous_batching.fozzy.json --det --record /tmp/zip-memory-residency-fix.fozzy --json`
- `fozzy trace verify /tmp/zip-memory-residency-fix.fozzy --strict --json`
- `fozzy replay /tmp/zip-memory-residency-fix.fozzy --json`
- `fozzy ci /tmp/zip-memory-residency-fix.fozzy --json`
- `fozzy run tests/continuous_batching.fozzy.json --det --proc-backend host --fs-backend host --http-backend host --json`

### CLI fix

Fixed `agent ring join` clap flag collision in
[agent/src/main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs):

- `--model-id` keeps `-m`
- `--memory` moved to `-M`

Added a clap sanity test:

- `tests::cli_definitions_pass_clap_debug_asserts`

Result:

- `cargo test -p agent cli_definitions_pass_clap_debug_asserts -- --nocapture`
  passed locally

### Control-plane SQLite pressure mitigation

Local benchmark tree includes the in-progress control-plane DB changes in
[control-plane/src/db/mod.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/db/mod.rs):

- busy timeout
- WAL-related pragmas
- configurable SQLite pool size via `MESHNET_SQLITE_POOL_MAX_SIZE`

These changes helped for some setup paths, but they did not eliminate runtime
`database is locked` failures under active inference polling.

## Benchmark Topologies

### Invalid baseline that was attempted first

Single local worker on a two-shard model.

This is not a valid baseline for the packaged asset because the runtime looked
for:

- `shard-0-of-1.manifest.json`

but the asset is packaged as a real two-shard model.

Observed failed job:

- job id: `086e494d-3462-4b18-92f6-206fef60ea4f`
- terminal error:
  - `Configuration error: Failed to read shard manifest ... shard-0-of-1.manifest.json`

Conclusion:

- the correct baseline for this asset is `2 workers on the same Apple Silicon
  machine`
- the correct comparison run is `1 local worker + 1 Intel worker`

### Correct baseline topology

Two workers on the same Apple Silicon machine:

- worker 1 device id: `fb3bc999-36f3-46c8-950b-61f0d39d5ec1`
- worker 2 device id: `7600d93f-419f-4779-b11c-ccc7b12e270a`
- ring split:
  - worker 1: `0..1024`
  - worker 2: `1024..2048`

Both workers reached:

- `status=online`
- `connectivity_state.status=connected`
- populated `listen_addrs`
- populated direct candidates

## Latest Same-Host 2-Worker Baseline Run

### Submission

- job id: `0b4bb287-a2df-4e1a-809a-a94d498830bb`
- network: `zipperf-direct`
- model: `tinyllama-1.1b-chat-v1.0`
- prompt: `Repeat the word MESH separated by spaces.`
- `max_tokens=96`
- `temperature=0.0`
- `top_p=1.0`

### Outcome

- status: `failed`
- completion tokens: `0`
- execution time: `53010 ms`
- settled credits: `0.0`
- released credits: `124.0`
- terminal error:
  - `Execution error: Ring all-reduce timed out after 30s`

### Assignment state at failure

- worker 1:
  - `status=acknowledged`
  - shard `0..1024`
- worker 2:
  - `status=failed`
  - shard `1024..2048`
  - error: `Execution error: Ring all-reduce timed out after 30s`

### Supporting live signals

Control plane emitted:

- `Database error: database is locked`

Workers emitted:

- `Listen error: Failed to negotiate transport protocol(s)`
- `Identify error ... Timeout error while opening a substream`
- `Tensor plane receive timed out`
- connection close timeouts

## Current Read

We do **not** yet have a trustworthy performance comparison between:

- same-host two-worker sharding
- local + Intel heterogeneous two-worker sharding

Reason:

- the same-host two-worker baseline fails before completion
- the earlier Apple Silicon runtime also had a real shared-model memory bug that
  is now fixed locally, so previous pressure-heavy runs should not be treated as
  representative
- until the baseline completes successfully, any cross-node `tok/s` comparison
  would be misleading

## What Is Verified

- device registration works
- ring join works after the CLI fix
- pool create/join works
- two-worker shard planning works
- control-plane dispatch works
- credit reservation/accounting works
- the engine proceeds far enough to expose real distributed execution failures

## Current Blocking Issues

### 1. Runtime data-plane/collective failure

Primary blocker for performance benchmarking.

Observed symptom:

- ring all-reduce timeout after `30s`

Related live warnings:

- failed transport protocol negotiation
- identify substream timeouts
- tensor plane receive timeouts

Impact:

- no completed decode
- no TTFT
- no steady-state `tok/s`

### 2. Control-plane SQLite lock pressure still exists

Observed during live inference:

- `Database error: database is locked`

Impact:

- status polling and/or assignment progress surfaces still contend under active
  workload
- can obscure or amplify runtime failures during benchmarking

## Remote Intel Status

Remote release agent rebuild completed successfully on:

- `/Users/saint/Desktop/mesh-e2e`

Remote release agent artifact after rebuild:

- timestamp: `2026-04-26 18:28:59`
- size: `23226772`

This means the Intel node is ready for the heterogeneous run once the local
same-host baseline is made to complete.

## Recommended Next Optimization Order

1. Fix the direct data-plane negotiation / identify / tensor-plane timeout path
   so the same-host two-worker baseline completes.
2. Re-run the same-host baseline and capture:
   - TTFT
   - completion tokens
   - total execution time
   - derived steady-state `tok/s`
3. Re-run the identical job with:
   - worker 1 on local Apple Silicon
   - worker 2 on `supercomputer.local`
4. Compare:
   - success/failure
   - TTFT delta
   - steady-state `tok/s` delta
   - transport overhead
5. Only after both runs complete successfully should performance conclusions be
   treated as valid.

## Summary

Current strong read:

- the inference engine is far enough along to perform real distributed setup,
  planning, leasing, and execution attempts
- the next meaningful performance work is blocked by distributed runtime
  transport/collective reliability, not by missing benchmark harness setup
- no valid `tok/s` improvement number exists yet because the correct same-host
  sharded baseline still fails with a ring all-reduce timeout
