# Mesh Architecture Notes

This document is an architecture note, not a phase tracker.

## Core Thesis

Mesh is trying to solve a specific problem: let a trusted group of consumer devices cooperate on inference without routing the tensor hot path through a centralized coordinator.

The repo is built around three distinct responsibilities:

- `control-plane`: durable orchestration and job state
- `agent`: device runtime, ring participation, and inference execution
- `relay-server`: fallback connectivity for peers that cannot connect directly

## Control Plane

The control plane is responsible for:

- network and device registration
- ring topology state
- durable inference dispatch
- assignment leasing and acknowledgement
- result and status persistence

It should not be on the hot tensor path.

Relevant files:

- [control-plane/src/api/mod.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/api/mod.rs)
- [control-plane/src/api/inference.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/api/inference.rs)
- [control-plane/src/services/ring_manager.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/ring_manager.rs)

## Agent

The agent is where the runtime lives:

- device configuration and identity
- pool and beacon participation
- ring membership
- assignment polling
- inference coordination
- checkpointing and recovery
- tensor-plane transport

Relevant files:

- [agent/src/main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs)
- [agent/src/inference/coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs)
- [agent/src/network/tensor_plane.rs](/Users/deepsaint/Desktop/meshnet/agent/src/network/tensor_plane.rs)
- [agent/src/connectivity.rs](/Users/deepsaint/Desktop/meshnet/agent/src/connectivity.rs)

## Tensor Path

The current design intent is:

- use the control plane for durable orchestration
- use peer-to-peer paths for inference traffic
- keep backpressure explicit
- keep recovery bounded

This repo now includes a dedicated tensor plane instead of relying only on generic request/response messaging for hot-path traffic.

## Connectivity Model

Connectivity is not a single binary mode. The current runtime supports:

- LAN discovery for pool-local visibility
- direct peer connectivity where available
- relay-backed fallback paths
- relay-first to direct-upgrade flows covered by runtime scenarios

The current documentation should describe those as active runtime behaviors, not future plans.

## Validation Model

The best current picture of runtime confidence comes from two layers:

- Fozzy deterministic and trace-based scenario validation
- Rust unit and integration tests

Right now, the scenario validation is stronger and cleaner than the whole-workspace Rust test stability.

## Important Honesty Rule

Do not describe the repo as a fully polished end-user product.

More accurate language:

- native distributed inference runtime
- durable control-plane plus dedicated tensor-plane architecture
- active productionization and hardening work still underway
