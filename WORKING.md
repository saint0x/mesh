# Mesh Working Proof

This document records the real end-to-end tests that have already been run against Mesh and what those tests prove.

It is intentionally focused on live validated behavior, not planned behavior.

## What Is Proven

Mesh is working as a distributed local-network and multi-node inference system with:

- real worker registration
- real model-ring membership
- real shard assignment
- real direct tensor transport
- real mixed-provider execution
- real distributed job lifecycle tracking
- real ledger and credit accounting
- real CLI-driven operator workflow

These proofs came from two classes of live tests:

1. public-cloud GPU validation on rented Vast instances
2. local-network validation across two real Macs on the same LAN

## Live Vast GPU Validation

We ran live distributed jobs on two rented Vast GPU instances:

- node A: RTX 3060 12 GB
- node B: RTX 4060 Ti 16 GB

These runs proved:

- the control plane is not fake and is able to coordinate real remote workers
- both workers can register and stay healthy in the same ring
- both workers can load real shard artifacts for the same model
- both workers can participate in ring all-reduce
- the tensor plane works across public internet hosts
- the control plane tracks real job lifecycle transitions
- ledger events and credit accounting are tied to real assignment completion

Important live outcomes from that phase:

- we fixed real runtime bugs in:
  - ring transport identity validation
  - tensor shape preservation
  - out-of-order tensor buffering
  - final RMS norm application
  - grouped-query attention support
  - CPU-to-GPU execution path correctness
- we proved the distributed path was not a single-node false positive
- we proved both workers actually contributed to the same job

Representative live result:

- job `5a1e33bc-5bef-4710-bb5e-94b2bea24b28`
- status: `completed`
- authoritative job wall time recorded
- assignment-level provider/capacity/shard accounting recorded
- ledger showed:
  - `job_started`
  - `credits_earned`
  - `credits_burned`
  - `job_completed`

## Live Local LAN Validation

We then validated the same architecture on a real local network across two physical Macs:

- Apple Silicon Mac running provider `metal`
- Intel Mac running provider `cpu`

This phase proved:

- direct LAN peer discovery works
- pool creation and pool join work
- signed pool membership over LAN works
- model-ring membership works on a mixed-provider ring
- direct local tensor transport works
- mixed `metal + cpu` distributed inference works
- real CLI-managed workers can execute the live path end to end

Representative live LAN result:

- job `0d565a9c-1db4-4b80-bbb2-0fb06267f723`
- status: `completed`
- output: `The MeshNet LAN test has been successfully completed.`

Later refined LAN run:

- job `5a1e33bc-5bef-4710-bb5e-94b2bea24b28`
- status: `completed`
- output: `The MeshNet LAN validation has successfully completed and the results have been submitted to the university for`
- authoritative wall time recorded at the job level
- both assignments completed with provider, shard, and capacity details

## Connectivity Proof

The LAN work also proved the network model itself:

- direct path is the real fast path
- relay is optional, not the primary local-network path
- stale loopback and observed-external candidate publication was a real bug and has now been corrected in the live topology path
- after the candidate-publication fixes, both nodes converged to:
  - `Direct/Connected`
  - 2 viable direct candidates each
  - no noisy degraded state on the Apple node

That means the current topology state is no longer just “job succeeded anyway.” The connectivity reporting is now aligned with the actual working path.

## Provider Proof

The currently validated providers are:

- `metal`
  - live on Apple Silicon
- `cpu`
  - live on Intel Mac
- `cuda`
  - validated in the production runtime path during the GPU-node phase

What is proven about providers:

- provider choice is explicit
- provider inventory is reported by the node
- provider selection is part of configuration
- the runtime does not rely on a synthetic compatibility path
- mixed-provider execution across the same ring is possible

## Accounting Proof

The following accounting behavior is live and validated:

- jobs create durable lifecycle records
- assignments create durable participant-level records
- credits are earned per worker based on actual participation
- credits are burned against the submitter side
- job-level wall time is recorded from authoritative lifecycle timing, not the last worker report
- shard range and assigned capacity are persisted per assignment

Validated examples from live runs showed:

- balanced earned/burned totals in ledger summary
- participant-specific `credits_earned`
- job-scoped `credits_burned`
- `job_completed` carrying authoritative wall-clock timing metadata

## CLI Proof

The grouped CLI surface has been QA’d on both Macs through installed binaries:

- `mesh device`
- `mesh resource`
- `mesh ring`
- `mesh job`
- `mesh ledger`
- `mesh pool`
- `mesh doctor`

Verified through the installed `mesh` command:

- `mesh doctor`
- `mesh device status`
- `mesh ring status`
- `mesh ring topology`
- `mesh ring shard`
- `mesh pool list`
- `mesh pool status`
- `mesh job stats`
- `mesh ledger summary`
- `mesh ledger events`

Installer proof:

- `install.sh` now installs:
  - `mesh`
  - `mesh-control-plane`
  - `mesh-relay`
- installer updates common shell init files so `~/.local/bin` is actually usable
- installer output now reflects the grouped CLI commands instead of the removed legacy flat command names

## Current Reality

What is already true:

- Mesh is not a mock
- Mesh is not a single-machine illusion
- Mesh can distribute real inference across multiple workers
- Mesh can run across a public internet GPU pair
- Mesh can run across a local mixed-provider LAN pair
- Mesh can account for real participation and job completion durably

What this does not claim:

- that every future model artifact will be correct automatically
- that every possible network environment is already hardened
- that every UI surface is finished

This document is only asserting what has been directly proven live.

## Working Definition

Mesh is working in the sense that matters most:

- a distributed worker ring can be formed
- real shards can be loaded
- real jobs can be executed across multiple nodes
- direct tensor exchange can complete
- results and credits can be recorded durably
- the installed CLI can operate the system on real machines

That core system is working end to end.
