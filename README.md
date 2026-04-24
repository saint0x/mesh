# Mesh

Mesh is a distributed network for sharing model execution across machines on a local network, with a control plane coordinating device registration, ring membership, job dispatch, status, and accounting.

The core idea is simple:
- workers on the same LAN contribute compute
- workers join a model ring for the model they serve
- jobs are dispatched through the control plane
- tensors move directly between workers on the dataplane
- results and credits are recorded durably by the control plane

Mesh has one production execution path. There is no mock or synthetic executor in this repo.

## How It Works

Mesh is split into two layers:

- local worker mesh:
  - agents run on each device
  - devices discover peers, join pools, and participate in a model ring
  - workers load real shard artifacts from disk
  - workers exchange tensor data directly over the dataplane
- control plane:
  - registers devices
  - stores network, ring, job, and ledger state
  - assigns distributed jobs to the active ring
  - exposes topology, status, and accounting APIs

For constrained networks, Mesh can also use a relay for peer connectivity, but the intended fast path is direct local-network connectivity.

## Functionality

- local-network compute sharing across multiple workers
- explicit model-ring membership and shard ownership
- distributed inference job submission and tracking
- direct tensor transport between workers
- durable control-plane state for jobs, topology, and ledger events
- explicit execution providers:
  - `cpu`
  - `metal`
  - `cuda`
- pool creation and LAN peer discovery
- credit accounting tied to real worker participation

## CLI Surface

Mesh ships one grouped CLI:

- `mesh device`
  - initialize device identity
  - start the agent
  - inspect local device status
- `mesh resource`
  - lock, unlock, and inspect committed resources
- `mesh ring`
  - join a model ring
  - leave a ring
  - inspect ring status, topology, and shard assignment
- `mesh job`
  - submit a distributed inference job
  - fetch job status
  - watch a job
  - inspect local runtime stats
- `mesh ledger`
  - inspect summary and event history for the current network
- `mesh pool`
  - create pools
  - join pools
  - list pools and peers
  - inspect LAN discovery state
- `mesh doctor`
  - verify local setup and control-plane reachability
- `mesh ui`
  - launch the local UI

## Execution Providers

Mesh now exposes one execution architecture with explicit provider selection underneath it:

- `cpu`: baseline runtime for broad compatibility, including Intel Macs and CPU-only Linux machines
- `metal`: native Apple path for Apple Silicon workers
- `cuda`: native Linux/NVIDIA path for datacenter and workstation GPUs

Provider choice is part of node configuration and capability reporting. Nodes advertise the providers they can actually run, the control plane stores that inventory, and the agent binds the tensor backend to the selected provider at startup. There is no silent provider fallback path.

Default provider selection is simple:

- prefer `metal` when available
- otherwise prefer `cuda` when available
- otherwise use `cpu`

To pin a node to a provider, set it in `~/.meshnet/device.toml`:

```toml
[execution]
preferred_provider = "cpu"
```

This is useful for:

- running Intel Macs as CPU workers on a LAN mesh
- forcing CPU parity checks on an Apple Silicon machine
- forcing a known GPU backend during bring-up and debugging

## Install

```bash
git clone https://github.com/saint0x/mesh.git
cd mesh
./install.sh
```

This installs:
- `mesh`
- `mesh-control-plane`
- `mesh-relay`

## Quick Start

Start infrastructure:

```bash
mesh-relay
mesh-control-plane
```

Start worker 1:

```bash
mesh device init --network-id demo --name "Worker 1"
mesh ring join --model-id tinyllama-1.1b
mesh device start
```

Start worker 2:

```bash
export MESHNET_HOME=~/.meshnet-worker2
mesh device init --network-id demo --name "Worker 2"
mesh ring join --model-id tinyllama-1.1b
mesh device start
```

Submit inference:

```bash
mesh job run --prompt "hello from mesh" --max-tokens 16 --model-id tinyllama-1.1b
```

Useful checks:

```bash
mesh doctor
mesh ring status
mesh ring topology
mesh ring shard
mesh pool list
mesh pool peers --pool-id <POOL_ID>
mesh ledger summary
mesh ledger events
mesh ui
```

For local UI development, `mesh-ui`'s existing `dev` command now boots the real local Mesh UI API first and then starts Vite, so the dashboard talks to live Mesh state instead of a frontend-only server.

## Model Assets

Every worker needs real model assets under `~/.meshnet/models/<model_id>/`:

- `model.json`
- `tokenizer.json`
- `shard-<worker>-of-<total>.manifest.json`
- `shard-<worker>-of-<total>.safetensors`

`model.json` defines the real tensor-parallel dimension and total model size. The control plane uses it for shard assignment, and the workers use the tokenizer for output decoding. The shard loader validates safetensors payloads against their manifests in [artifact_loader.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/artifact_loader.rs).

The same canonical artifacts are used across providers. Provider choice changes execution, not model semantics.

## Core Components

- `agent`: worker runtime and CLI for device bring-up, pool participation, ring membership, shard loading, inference execution, and dataplane transport. See [main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs) and [coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs).
- `control-plane`: durable coordinator for registration, topology, distributed job dispatch, status polling, and ledger events. See [inference.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/api/inference.rs) and [ring_manager.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/ring_manager.rs).
- `relay-server`: optional connectivity layer for environments that cannot keep workers directly connected. See [relay-server/README.md](/Users/deepsaint/Desktop/meshnet/relay-server/README.md).

## Verification

Mesh uses Fozzy first for system validation.

```bash
fozzy doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 424242 --json
fozzy test --det --strict tests/production_dispatch.fozzy.json tests/live_relay_runtime.fozzy.json --json
fozzy run tests/production_dispatch.fozzy.json --det --record /tmp/production_dispatch_trace.fozzy --json
fozzy trace verify /tmp/production_dispatch_trace.fozzy --strict --json
fozzy replay /tmp/production_dispatch_trace.fozzy --json
fozzy ci /tmp/production_dispatch_trace.fozzy --json
cargo test --workspace
```

For provider work, validate both the runtime and the provider contract:

```bash
fozzy doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 424242 --json
fozzy test --det --strict tests/production_dispatch.fozzy.json tests/live_relay_runtime.fozzy.json --json
fozzy run tests/production_dispatch.fozzy.json --det --record /tmp/production_dispatch_trace.fozzy --json
fozzy trace verify /tmp/production_dispatch_trace.fozzy --strict --json
fozzy replay /tmp/production_dispatch_trace.fozzy --json
fozzy ci /tmp/production_dispatch_trace.fozzy --json
```

## More Docs

- [QUICKSTART.md](/Users/deepsaint/Desktop/meshnet/QUICKSTART.md)
- [CONTRIBUTING.md](/Users/deepsaint/Desktop/meshnet/CONTRIBUTING.md)
- [INSIGHT.md](/Users/deepsaint/Desktop/meshnet/INSIGHT.md)
- [LAN_DISCOVERY_TEST.md](/Users/deepsaint/Desktop/meshnet/LAN_DISCOVERY_TEST.md)

## License

MIT
