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
