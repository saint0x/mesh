# Mesh

Mesh is an open source Rust runtime for distributed model execution across multiple machines. It is built for running one real inference job through a coordinated worker ring, with explicit model membership, durable control-plane state, and a dedicated tensor transport between workers.

## What It Does

- runs a control plane for device registration, ring topology, job dispatch, status, and ledger events
- runs an agent on each worker for model shard loading, assignment polling, execution, and dataplane transport
- runs a relay for environments where peers cannot connect directly
- executes real shard artifacts from disk with safetensors manifests and model-specific tokenizers
- records durable distributed job lifecycle state instead of treating submit as success

## Good Fits

- small operator-managed inference meshes across a few trusted machines
- native apps or protocols that want a low-latency inference substrate rather than a hosted SaaS API
- experiments with explicit worker rings, credit accounting, and topology-aware dispatch
- environments where model shards live on the worker and control traffic stays separate from tensor traffic

## Runtime Model

Mesh has one production execution path:
- devices register with the control plane
- devices explicitly join a model ring
- each worker serves a real shard for that model
- inference is submitted against that model
- workers execute through the dedicated tensor plane and report durable results back to the control plane

There is no parallel mock or synthetic executor in this repo anymore.

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

## Model Assets

Every worker needs real model assets under `~/.meshnet/models/<model_id>/`:

- `model.json`
- `tokenizer.json`
- `shard-<worker>-of-<total>.manifest.json`
- `shard-<worker>-of-<total>.safetensors`

`model.json` defines the real tensor-parallel dimension and total model size. The control plane uses it for shard assignment, and the workers use the tokenizer for output decoding. The shard loader validates safetensors payloads against their manifests in [artifact_loader.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/artifact_loader.rs).

The same canonical artifacts are used across providers. Provider choice changes execution, not model semantics.

## Core Components

- `agent`: worker runtime and CLI for initialization, ring membership, shard loading, inference execution, and dataplane transport. See [main.rs](/Users/deepsaint/Desktop/meshnet/agent/src/main.rs) and [coordinator.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/coordinator.rs).
- `control-plane`: durable coordinator for registration, ring topology, inference dispatch, status polling, and ledger events. See [inference.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/api/inference.rs) and [ring_manager.rs](/Users/deepsaint/Desktop/meshnet/control-plane/src/services/ring_manager.rs).
- `relay-server`: relay path for constrained networks that cannot keep the worker mesh fully direct. See [relay-server/README.md](/Users/deepsaint/Desktop/meshnet/relay-server/README.md).

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
