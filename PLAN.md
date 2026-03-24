# Production Plan

## Goal

Build the next production phase of `meshnet` around one correct architecture:

- low-latency peer-to-peer tensor movement
- decentralized execution within pools
- explicit, production-only runtime behavior
- no backwards-compatibility branches
- no fake future modes
- no duplicate transport paths left alive after replacement

## Canonical Direction

The next phase is:

1. dedicated tensor data plane
2. stronger NAT traversal and punched-path execution
3. runtime governance and backpressure

That order is intentional.

We are not doing all three architectural shifts at once.
We are first making the tensor data plane the single blessed production path.

## Why This Order

### 1. Dedicated Data Plane First

This is the best fit for the current product goal:

- lower latency
- better throughput
- cleaner hot path
- stronger decentralized pool execution

The current system is already materially better on:

- durable dispatch
- typed connectivity
- authoritative ring metadata
- real direct vs relayed startup behavior
- explicit runtime state reporting

The remaining weak point is the tensor hot path still using generic libp2p request/response framing.

That is good enough for rollout correctness.
It is not the production shape we want for the long-term tensor path.

### 2. NAT Traversal Second

After the hot path is correct, we make more peers succeed directly and more often.

This improves:

- direct peer success rate
- real-world pool usability
- relay avoidance
- decentralization in hostile networks

But it does not by itself make tensor traffic fundamentally faster.

### NAT Traversal Status

- ✅ Direct peer dialing is now deterministic and ranked instead of trusting arbitrary advertised address order.
- ✅ Public direct candidates are preferred first, then DNS direct candidates, then private/LAN candidates, with relay addresses excluded from the direct dial set.
- ✅ Ring neighbor connection setup now treats direct connectivity as the canonical path and only falls back to relay when direct dialing fails.
- ✅ Ad hoc peer job submission now resolves and attempts viable direct addresses before using relay as a degraded fallback.
- ✅ Runtime connectivity state is now persisted separately from static config so heartbeat/status can report the active path the agent most recently established.
- ✅ DCUTR upgrade success and failure are now surfaced in the mesh event stream and reflected in runtime connectivity state.
- ✅ Heartbeat now publishes an explicit ranked `direct_candidates` set instead of requiring peers to infer direct dialing intent from raw `listen_addrs`.
- ✅ Control-plane device state and ring topology now persist and return structured direct-connect candidates as a first-class contract.
- ✅ Ring workers now resolve direct peers from authoritative `direct_candidates` metadata rather than rebuilding dial order from generic listen-address lists.
- ✅ Live swarm-observed external address candidates are now persisted and folded into heartbeat candidate publication alongside local listen addresses.
- ✅ Agent status and pool-status now expose local and remote direct-candidate quality so operators can inspect current reachability and best direct endpoints without digging through raw state files.
- ✅ Agent metrics now persist and display direct-path quality signals including direct vs relayed peer connections, relay fallback usage, direct-upgrade success/failure, and external address discovery counts.
- ✅ Direct-path event handling now has integration-style coverage through the live job-runner metrics path for direct connections, relay fallback, direct upgrades, and external-address discovery.
- ✅ Direct candidate publication now carries explicit source and freshness metadata so observed public reachability hints can outrank stale local-only advertisements deterministically.
- ✅ Control-plane validation and topology persistence now treat candidate recency as part of the production reachability contract instead of an implicit local detail.
- ✅ Ring topology now publishes explicit `peer_punch_plans` for hostile NAT pairs so peers consume a first-class punch coordination contract instead of inferring upgrade behavior from endpoint ranking alone.
- ✅ Ring inference and ad hoc job dialing now apply those explicit punch plans before relay fallback instead of treating every direct attempt as the same kind of path.
- ✅ Agent metrics now distinguish punch-assisted attempts, punch-assisted direct connections, and punch-assisted upgrade outcomes from ordinary direct-path success.
- ✅ NAT coverage now includes a dedicated `punch_path_coordination` Fozzy scenario plus host-backed trace validation for punch-plan topology and agent-side punch-plan consumption.
- ✅ NAT coverage now also exercises multi-peer relay-worker punch-plan generation and swarm neighbor punch-attempt emission, so the gate covers more than a single peer-pair resolution path.
- ✅ NAT coverage now includes a live concurrent relay-attachment runtime test with multiple in-process peers sharing the same relay, not just scripted or single-peer checks.

### NAT Traversal Still Open

- ⬜ Extend from the current host-backed and live relay-attachment coverage into fuller concurrent runtime/NAT execution scenarios that verify successful relay-mediated peer dialing and direct upgrade outcomes.

### 3. Governance Third

After the transport shape is right and connectivity is stronger, add:

- backpressure
- admission control
- concurrency caps
- bandwidth budgeting
- overload protection

This is important for mature production behavior, but it should not define the hot path before the hot path itself is correct.

## Non-Goals

We are not doing any of the following:

- preserving the old tensor transport as a fallback
- keeping compatibility shims for replaced runtime paths
- reintroducing overlay support before it exists as a real backend
- keeping multiple production transport stories alive in parallel
- optimizing for temporary migration convenience over long-term production clarity

## Single Production Rule

At every phase boundary there must be one production way the system works.

That means:

- one tensor transport story
- one active inference transport path
- one connectivity contract
- one ring metadata contract
- one source of truth for reachability and peer identity

If a path is replaced, the old one is deleted.

## Phase 1 Scope: Dedicated Tensor Data Plane

### Target Architecture

Keep libp2p for:

- identity
- discovery
- control coordination
- ring membership coordination
- fallback control messaging where appropriate

Move tensor traffic onto a dedicated peer-to-peer data plane optimized for:

- low framing overhead
- predictable streaming semantics
- explicit flow control
- better timeout behavior
- connection-local metrics
- direct ownership of the hot tensor path

### Production Outcome

After Phase 1:

- libp2p request/response is not the tensor hot path
- tensor exchange uses the dedicated data plane only
- ring execution uses one transport path, not two
- old tensor compatibility code is deleted

## Current Status

### Completed

- ✅ Production direction is locked: dedicated tensor data plane first, NAT traversal second, governance third.
- ✅ The old libp2p tensor request/response protocol file was deleted.
- ✅ Dedicated tensor transport primitives were introduced in `agent/src/network/tensor_plane.rs`.
- ✅ Tensor message types were split into a transport-neutral module in `agent/src/network/tensor_message.rs`.
- ✅ Ring all-reduce now uses the dedicated tensor plane in the active inference path.
- ✅ The inference coordinator now owns the tensor data plane directly instead of depending on tensor request/response callbacks.
- ✅ The active inference path advertises dedicated tensor endpoints through the existing heartbeat/listen-address path.
- ✅ Neighbor tensor endpoints are resolved from authoritative topology metadata rather than placeholder local values.
- ✅ `ring_state.json` was removed from the active runtime/CLI path as a second source of truth.
- ✅ The active tensor path is now single-path by construction in the production inference flow.
- ✅ The production plan and extrapolation docs were updated to reflect shipped vs remaining work.
- ✅ Production work has been consolidated onto `main`.

### Validated

- ✅ `cargo test -p agent --no-run`
- ✅ `cargo test -p control-plane --no-run`
- ✅ `cargo test -p agent select_direct_dial_addrs_prefers_public_quic_then_private_tcp -- --nocapture`
- ✅ `cargo test -p agent current_state_prefers_runtime_state_when_present -- --nocapture`
- ✅ `cargo test -p agent build_direct_peer_candidates_excludes_relay_and_sorts -- --nocapture`
- ✅ `cargo test -p agent direct_candidate_seed_addrs_merge_listen_and_observed -- --nocapture`
- ✅ `cargo test -p agent test_job_stats_connectivity_metrics -- --nocapture`
- ✅ `cargo test -p agent test_handle_event_records_connectivity_path_metrics -- --nocapture`
- ✅ `cargo test -p agent test_handle_event_records_upgrade_and_external_addr_metrics -- --nocapture`
- ✅ `cargo test -p agent test_worker_position -- --nocapture`
- ✅ `cargo test -p control-plane test_update_heartbeat -- --nocapture`
- ✅ `cargo test -p control-plane test_get_topology_handler -- --nocapture`
- ✅ `cargo test -p control-plane test_ring_stability -- --nocapture`
- ✅ `fozzy --cwd . validate tests/production_dispatch.fozzy.json --json`
- ✅ `fozzy --cwd . doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 1 --json`
- ✅ `fozzy --cwd . test --det --strict tests/production_dispatch.fozzy.json --json`
- ✅ `fozzy --cwd . run tests/production_dispatch.fozzy.json --det --record .fozzy/nat-direct.trace.fozzy --json`
- ✅ `fozzy --cwd . trace verify .fozzy/nat-direct.trace.fozzy --strict --json`
- ✅ `fozzy --cwd . replay .fozzy/nat-direct.trace.fozzy --json`
- ✅ `fozzy --cwd . ci .fozzy/nat-direct.trace.fozzy --json`
- ✅ `fozzy --cwd . run tests/production_dispatch.fozzy.json --det --record .fozzy/nat-candidates.trace.fozzy --json`
- ✅ `fozzy --cwd . trace verify .fozzy/nat-candidates.trace.fozzy --strict --json`
- ✅ `fozzy --cwd . replay .fozzy/nat-candidates.trace.fozzy --json`
- ✅ `fozzy --cwd . ci .fozzy/nat-candidates.trace.fozzy --json`
- ✅ `cargo test -p agent build_direct_peer_candidates_prefers_observed_external_hints -- --nocapture`
- ✅ `cargo test -p agent test_handle_event_records_punch_assisted_metrics -- --nocapture`
- ✅ `fozzy --cwd . run tests/production_dispatch.fozzy.json --det --record .fozzy/reachability-hints.trace.fozzy --json`
- ✅ `fozzy --cwd . trace verify .fozzy/reachability-hints.trace.fozzy --strict --json`
- ✅ `fozzy --cwd . replay .fozzy/reachability-hints.trace.fozzy --json`
- ✅ `fozzy --cwd . ci .fozzy/reachability-hints.trace.fozzy --json`
- ✅ `fozzy --cwd . run tests/production_dispatch.fozzy.json --det --record .fozzy/punch-path.trace.fozzy --json`
- ✅ `fozzy --cwd . trace verify .fozzy/punch-path.trace.fozzy --strict --json`
- ✅ `fozzy --cwd . replay .fozzy/punch-path.trace.fozzy --json`
- ✅ `fozzy --cwd . ci .fozzy/punch-path.trace.fozzy --json`
- ✅ `fozzy --cwd . validate tests/punch_path_coordination.fozzy.json --json`
- ✅ `fozzy --cwd . doctor --deep --scenario tests/punch_path_coordination.fozzy.json --runs 5 --seed 1 --json`
- ✅ `fozzy --cwd . test --det --strict tests/punch_path_coordination.fozzy.json --json`
- ✅ `fozzy --cwd . run tests/punch_path_coordination.fozzy.json --det --record .fozzy/punch-path-coordination.trace.fozzy --json`
- ✅ `fozzy --cwd . trace verify .fozzy/punch-path-coordination.trace.fozzy --strict --json`
- ✅ `fozzy --cwd . replay .fozzy/punch-path-coordination.trace.fozzy --json`
- ✅ `fozzy --cwd . ci .fozzy/punch-path-coordination.trace.fozzy --json`
- ✅ `fozzy --cwd . run tests/punch_path_coordination.fozzy.json --det --proc-backend host --fs-backend host --http-backend host --record .fozzy/punch-path-coordination-host.trace.fozzy --json`
- ✅ `fozzy --cwd . trace verify .fozzy/punch-path-coordination-host.trace.fozzy --strict --json`
- ✅ `fozzy --cwd . replay .fozzy/punch-path-coordination-host.trace.fozzy --json`
- ✅ `fozzy --cwd . ci .fozzy/punch-path-coordination-host.trace.fozzy --json`
- ✅ `cargo test -p control-plane test_get_topology_generates_peer_punch_plans_for_relayed_workers -- --nocapture`
- ✅ `cargo test -p agent test_set_ring_neighbors_emits_punch_attempts_for_both_neighbors -- --nocapture`
- ✅ `cargo test -p agent test_multiple_live_peers_attach_to_same_relay_runtime -- --nocapture`
- ✅ `cargo test -p relay-server --no-run`

### Still Open In Phase 1

- ⬜ Extend coverage so the dedicated tensor data plane is exercised by focused runtime/integration tests, not only compile-time and structural tests.
- ⬜ Eliminate any remaining mixed-path assumptions in control-plane and operator surfaces that still treat generic listen addresses as sufficient for tensor execution.
- ⬜ Add stronger transport-level observability for tensor-plane connection health, timeouts, and throughput.
- ⬜ Harden dedicated tensor endpoint selection for multi-interface/NAT-hostile environments before starting the NAT traversal phase.

## Phase 1 Implementation Plan

### Step 1. Define the New Data Plane Boundary

Create a clean internal separation between:

- control-plane and coordination transport
- tensor data-plane transport

Required outcome:

- the inference/ring execution layer depends on a transport interface shaped for tensor streaming
- it does not depend directly on libp2p request/response semantics

### Step 2. Introduce Dedicated Tensor Transport Primitives

Build dedicated transport primitives for:

- peer session establishment
- send/receive of tensor payloads
- acknowledgements and completion semantics
- explicit buffer and timeout handling

Required outcome:

- tensor movement is represented as data-plane operations, not generic RPC request/response messages

### Step 3. Cut Ring Execution Over Completely

Move ring all-reduce and related inference tensor exchange onto the new data plane.

Required outcome:

- ring inference uses the new data plane only
- no dual-write or dual-read period in production code

### Step 4. Delete Old Tensor Transport Path

Delete the old libp2p tensor request/response path after cutover.

Delete in full:

- tensor request/response protocol implementation
- tensor request/response event wiring
- any compatibility helpers that only existed for the old tensor path
- code paths that keep both old and new tensor transports alive

### Step 5. Re-validate Production Path

Run:

- `cargo test -p agent --no-run`
- `cargo test -p control-plane --no-run`
- focused tests covering ring and control-plane behavior
- Fozzy deterministic gates
- recorded trace verification and replay

Required outcome:

- only the new tensor data-plane path is validated as production

## Next

The next production step is:

1. Implement explicit punched-path candidate exchange so peers can attempt direct upgrades with richer candidate sets than passive listen-address advertisement alone.
2. Add focused integration coverage for direct upgrade success, direct dial failure, and relay fallback behavior.
3. Add operator-visible direct-path metrics and candidate-selection reporting.
4. Extend candidate gathering with observed/public reachability hints rather than relying only on current local advertisements.
5. Once direct connectivity is observable, richly exchanged, and covered, move into governance/backpressure.

## Deferred PR Analysis

These PRs are not being merged as branches into the production line, but they are worth analyzing and selectively piping into `main` later:

- ✅ Analyze PR #8: `Enable direct LAN peer connections via beacon discovery`
  Why it matters:
  This is relevant to the NAT/direct-connect phase because it may contain useful LAN peer dialing ideas that can be adapted to the new single-path production transport model.

- ✅ Analyze PR #6: `Add storage reservation system with 24h cooldown enforcement`
  Why it matters:
  This is relevant to the governance/resource phase because it may contain useful storage reservation and resource accounting ideas that fit future production hardening.

- ❌ Do not revive PR #7: `Replace ring all-reduce with pipeline-parallel execution`
  Why not:
  It changes the core distributed inference architecture away from the chosen production direction and should not be piped into the current plan.

## Deletion Policy

This effort explicitly prefers deletion over coexistence.

When a replacement is ready:

- remove the old code
- remove the config knobs that select it
- remove dead tests tied only to the old path
- remove comments/docs describing it as supported

We do not keep deprecated production branches around "just in case."

## Current Legacy Targets To Eliminate In Phase 1

These are the first categories to collapse during the data-plane cutover:

- libp2p tensor request/response as the active tensor transport
- tensor transport code that assumes RPC-like response semantics on the hot path
- any ring-execution logic coupled directly to the old tensor protocol behavior

## Validation Standard

Testing is not complete unless the active goal is validated through the real production path.

For this phase that means:

- deterministic Fozzy first
- recorded traces
- trace verification
- replay
- CI gate checks

Validation must follow the new single-path architecture, not legacy compatibility behavior.

## Decision Record

We are explicitly choosing:

- `data plane` first

We are explicitly not choosing first:

- NAT traversal as the primary next phase
- governance/backpressure as the primary next phase

Those remain next, but they do not define the first production cutover of this phase.

## Working Principle

If there is ever a choice between:

- preserving compatibility with the old tensor path
- making the new production path simpler and more explicit

choose the new production path and delete the old one.
