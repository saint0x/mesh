# Mesh

> Cooperative distributed inference across consumer devices.

Mesh is a Rust workspace for running cooperative inference over a mix of direct and relayed peer connectivity. The repo currently includes:

- an `agent` daemon and CLI for device setup, pool management, ring participation, and inference execution
- a `control-plane` service for device registration, topology management, durable inference dispatch, and job status
- a `relay-server` for NAT traversal and fallback connectivity

The architecture is centered on a dedicated tensor data plane between workers and a separate control plane for orchestration.

## Current State

What is implemented today:

- durable inference submit, claim, acknowledge, result, and status flows
- worker ring topology management with persisted worker metadata
- direct and relayed connectivity modes
- LAN beacon discovery and pool membership flows
- dedicated tensor-plane transport with bounded backpressure and bandwidth governance
- checkpointing and bounded recovery inside the inference coordinator
- Fozzy scenarios for production dispatch and relay/direct-upgrade runtime coverage

Important caveats:

- the current inference path is still primarily validated with mock/Xavier-initialized weight flows
- the repo has strong deterministic scenario coverage, but the full Rust test suite is not perfectly clean yet on every run
- this project is a native distributed inference runtime, not a polished end-user mesh product

## Install

```bash
git clone https://github.com/saint0x/mesh.git
cd mesh
./install.sh
```

The installer builds the workspace and installs:

- `mesh`
- `mesh-control-plane`
- `mesh-relay`

into `~/.local/bin`.

## Local Bring-Up

Terminal 1:

```bash
mesh-relay
```

Terminal 2:

```bash
mesh-control-plane
```

Terminal 3:

```bash
mesh init --network-id demo --name "Worker 1"
mesh join-ring --model-id llama-70b
mesh start
```

Terminal 4:

```bash
export MESHNET_HOME=~/.meshnet-worker2
mesh init --network-id demo --name "Worker 2"
mesh join-ring --model-id llama-70b
mesh start
```

Terminal 5:

```bash
mesh inference --prompt "Hello, world!" --max-tokens 10 --model-id llama-70b
```

## Architecture

### Control Plane

The control plane is a centralized coordinator for:

- network and device registration
- ring topology management
- durable inference dispatch
- assignment leasing and acknowledgement
- job result persistence and status lookup

See [control-plane/src/api/mod.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/api/mod.rs) and [control-plane/src/api/inference.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/api/inference.rs).

### Agent

The agent provides:

- device initialization and runtime state
- pool creation and membership flows
- worker ring participation
- inference coordinator startup and assignment polling
- dedicated tensor-plane transport for hot-path inference traffic

See [agent/src/main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs) and [agent/src/inference/coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs).

### Relay

The relay server provides Circuit Relay v2 support for:

- reservation-backed connectivity
- relayed fallback paths
- rendezvous for later direct upgrades

See [relay-server/README.md](/Users/deepsaint/Desktop/meshnet/relay-server/README.md).

## Testing

This repo uses Fozzy first for system validation.

Start with:

```bash
fozzy map suites --root . --scenario-root tests --profile pedantic --json
fozzy doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 424242 --json
fozzy test --det --strict tests/production_dispatch.fozzy.json tests/live_relay_runtime.fozzy.json --json
fozzy run tests/production_dispatch.fozzy.json --det --record /tmp/production_dispatch_trace.fozzy --json
fozzy trace verify /tmp/production_dispatch_trace.fozzy --strict --json
fozzy replay /tmp/production_dispatch_trace.fozzy --json
fozzy ci /tmp/production_dispatch_trace.fozzy --json
```

You should also run Rust tests:

```bash
cargo test --workspace
```

At the time of this documentation update, deterministic Fozzy checks were passing for the main production-dispatch scenario, while the full Rust workspace suite showed one intermittent connectivity failure when run as a whole. Treat that as an active hardening item, not a resolved issue.

## Documentation

- [QUICKSTART.md](/Users/deepsaint/Desktop/meshnet/QUICKSTART.md): local bring-up and multi-device walkthrough
- [LAN_DISCOVERY_TEST.md](/Users/deepsaint/Desktop/meshnet/LAN_DISCOVERY_TEST.md): LAN pool and beacon validation
- [INSIGHT.md](/Users/deepsaint/Desktop/meshnet/INSIGHT.md): architecture notes
- [FINAL_STATUS.md](/Users/deepsaint/Desktop/meshnet/FINAL_STATUS.md): current repo status snapshot
- [CONTRIBUTING.md](/Users/deepsaint/Desktop/meshnet/CONTRIBUTING.md): contributor workflow and testing expectations

## Non-Goals

- anonymous public marketplace compute
- tokenized or blockchain-based coordination
- bidding and spot-pricing dynamics
- pretending current mock-weight validation is the same thing as production model quality

## License

MIT
