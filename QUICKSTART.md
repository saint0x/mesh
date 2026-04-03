# Mesh Quickstart

This guide covers the current production runtime bring-up flow.

## Prerequisites

- Rust installed on each machine
- this repository checked out on each machine
- devices on the same LAN if you want beacon-based discovery

## Install

Run on each machine:

```bash
./install.sh
```

That builds the workspace and installs `mesh`, `mesh-control-plane`, and `mesh-relay` into `~/.local/bin`.

## Fast Local Bring-Up

### 1. Start infrastructure

Terminal 1:

```bash
mesh-relay
```

Terminal 2:

```bash
mesh-control-plane
```

### 2. Initialize a worker

Terminal 3:

```bash
mesh init --network-id demo --name "Worker 1"
mesh join-ring --model-id llama-70b
mesh start
```

### 3. Add another worker

Terminal 4:

```bash
export MESHNET_HOME=~/.meshnet-worker2
mesh init --network-id demo --name "Worker 2"
mesh join-ring --model-id llama-70b
mesh start
```

### 4. Submit an inference job

Terminal 5:

```bash
mesh inference --prompt "Hello world" --max-tokens 10 --model-id llama-70b
```

Before starting workers, place real shard artifacts for the selected `model_id` on each node:

- `~/.meshnet/models/<model_id>/shard-<worker>-of-<total>.manifest.json`
- `~/.meshnet/models/<model_id>/shard-<worker>-of-<total>.safetensors`

The safetensors file must satisfy the loader contract in [agent/src/inference/artifact_loader.rs](/Users/deepsaint/Desktop/meshnet/agent/src/inference/artifact_loader.rs).

## Multi-Device LAN Pool Flow

### Device 1

```bash
mesh init --network-id test-network --name "Device 1"
mesh pool-create --name "LAN Test Pool"
mesh start
```

Save the printed:

- pool ID
- pool root public key

### Device 2+

```bash
mesh init --network-id test-network --name "Device 2"
mesh pool-join --pool-id <POOL_ID> --pool-root-pubkey <POOL_ROOT_PUBKEY> --name "LAN Test Pool"
mesh start
```

## Useful Checks

List pools:

```bash
mesh pool-list
```

Show discovered peers in a pool:

```bash
mesh pool-peers --pool-id <POOL_ID>
```

Show ring topology:

```bash
mesh ring-status
mesh pool-status
```

Show shard assignment and stats:

```bash
mesh shard-status
mesh inference-stats
```

## Testing The Current Runtime

System validation:

```bash
fozzy doctor --deep --scenario tests/production_dispatch.fozzy.json --runs 5 --seed 424242 --json
fozzy test --det --strict tests/production_dispatch.fozzy.json tests/live_relay_runtime.fozzy.json --json
```

Trace-based validation:

```bash
fozzy run tests/production_dispatch.fozzy.json --det --record /tmp/production_dispatch_trace.fozzy --json
fozzy trace verify /tmp/production_dispatch_trace.fozzy --strict --json
fozzy replay /tmp/production_dispatch_trace.fozzy --json
fozzy ci /tmp/production_dispatch_trace.fozzy --json
```

Rust tests:

```bash
cargo test --workspace
```

## Current Caveats

- Durable inference dispatch is implemented.
- The agent does poll and execute inference assignments from the control plane.
- The hot path uses the dedicated tensor plane, not just generic control-plane messaging.
- The runtime no longer includes the older generic job executor or synthetic shard loader.
- Real model quality still depends on shipping correct shard manifests, safetensors weights, and tokenizer assets.

## Troubleshooting

If peers are not discovered:

- verify all devices are on the same LAN
- verify UDP multicast is allowed on port `42424`
- verify each device joined the same pool

If control-plane calls fail:

- confirm `mesh-control-plane` is running
- confirm workers are using the right `--control-plane` URL

If relay connectivity fails:

- confirm `mesh-relay` is running
- confirm relay `advertised_addrs` point at the node's real reachable addresses
- inspect relay logs
- run the Fozzy relay runtime scenarios in `tests/`
