# Mesh

> Cooperative tensor-parallel distributed inference across consumer devices.

**Enable groups to pool compute resources and collectively run AI models too large for any individual machine.**

## Quick Install

```bash
# Clone and install
git clone <repo-url>
cd meshnet
./install.sh

# Restart your shell or run:
source ~/.zshrc  # or ~/.bashrc

# Verify installation
mesh --version
```

This installs the `mesh` command to `~/.local/bin` and updates your PATH.

### Quick Start

```bash
# Initialize your device
mesh init --network-id test --name "My Laptop"

# LAN Discovery (Phase 1)
mesh pool-create --name "My Pool"              # Create a pool
mesh pool-join --pool-id <id> --pool-root-pubkey <key>  # Join pool
mesh pool-list                                   # List pools
mesh start                                       # Start agent with LAN discovery

# Check discovered peers on your LAN
mesh pool-peers --pool-id <id>
```

See [LAN_DISCOVERY_TEST.md](LAN_DISCOVERY_TEST.md) for multi-device testing.

## The Core Problem

You can't run a 64GB Llama-70B model on your 8GB laptop. But 10 friends with 8GB laptops **can** run it together.

## The Solution

Mesh uses **tensor parallelism** with **ring all-reduce** to split model weights across devices, allowing cooperative execution of distributed inference workloads.

### Key Concepts

**Cooperative Pooling** - Workers contribute locked resources and share benefits fairly
- Contribute 10% of pool capacity → Use 10% of inference throughput
- Resource locking with 24-hour cooldown ensures stable pools
- Credit-based allocation aligns incentives

**Tensor Parallelism** - All workers participate in every layer (not pipeline)
- Each worker holds different columns of weight matrices (10% for 10 workers)
- All workers compute partial matrix multiplications in parallel
- Ring all-reduce combines results after each layer
- Same technique as NCCL/Horovod, optimized for consumer internet

**Ring Topology** - Workers form P2P ring for optimal bandwidth utilization
- Direct connections between left/right neighbors via libp2p
- Ring all-reduce: 2×(N-1) steps instead of N×(N-1) all-to-all transfers
- Full bisection bandwidth utilized (all links active simultaneously)
- NAT traversal with DCUTR, fallback to relay server

## Architecture

### Three-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL PLANE                             │
│  • Job queue and distribution                                │
│  • Worker ring topology management                           │
│  • Health monitoring (heartbeats)                            │
│  • Checkpoint coordination                                   │
│  • Credit accounting                                         │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ (topology management)
┌─────────────────────────────┐
│    WORKER RING (P2P)        │
│                             │
│  Device 1 ←→ Device 2       │
│      ↑            ↓         │
│  Device 10 ←→ Device 3      │
│      ↑            ↓         │
│  Device 9  ←→ Device 4      │
│      ↑            ↓         │
│  Device 8  ←→ ... ←→ D5     │
│                             │
│  Each device:               │
│  - 7GB locked memory        │
│  - Model shard (10% cols)   │
│  - Ring all-reduce          │
└─────────────────────────────┘
```

### Data Plane vs Control Plane Separation

**Data Plane (Ring P2P Network):**
- Direct peer-to-peer connections between adjacent workers
- Tensor passing for all-reduce operations
- Model weight storage and activation buffers
- **NOT routed through control plane** (direct for lowest latency)

**Control Plane (Centralized Coordinator):**
- Job queue and distribution to worker ring
- Worker registration and health monitoring
- Ring topology updates (device join/leave)
- Checkpoint coordination
- Credit balance tracking
- **Only control messages, NOT tensor data**

## What Makes This Different

### vs. Petals (Collaborative Inference)
- ✅ **Tensor parallelism** (parallel, lower latency) vs. pipeline parallelism (sequential, high latency)
- ✅ **Private cooperative pools** (trusted workers) vs. public swarm (untrusted participants)
- ✅ **Resource locking** with cooldown (stable pool) vs. no locking (unstable pool)
- ✅ **Fair allocation** based on contribution vs. free-for-all

### vs. DeepSpeed / Megatron (Training Frameworks)
- ✅ **Same tensor parallelism strategy** (ring all-reduce)
- ✅ **Optimized for consumer devices** over WAN vs. enterprise GPU clusters
- ✅ **Works with consumer internet** (50 Mbps+) vs. high-bandwidth interconnect required
- ✅ **Fault tolerance via checkpointing** for inference

### vs. Ray (Distributed Compute Framework)
- ✅ **Tensor parallelism** (distribute single model inference) vs. task parallelism (independent tasks)
- ✅ **Decentralized worker ring** (P2P) vs. centralized scheduler
- ✅ **Cooperative pooling** (shared resources) vs. task queue

## Core Design Principles

### 1. Cooperative, Not Competitive
Workers pool resources and share benefits. No marketplace dynamics, no bidding, no price discovery. Just fair allocation based on contribution.

### 2. Tensor Parallelism for Low Latency
Each layer executes in parallel across all workers (90ms) instead of sequentially (500ms). Ring all-reduce combines partial results efficiently.

### 3. Resource Locking for Stability
Workers lock memory with 24-hour cooldown. Pool needs predictable resource availability to function reliably.

### 4. Private Trusted Networks
Like Tailscale - create private networks with trusted participants. Not anonymous public compute.

### 5. Fault Tolerance Through Checkpointing
Checkpoint every 50 tokens. Device failures only lose ~30 seconds of work instead of entire job.

## Technical Highlights

### Ring All-Reduce Algorithm
The same algorithm NCCL uses for multi-GPU training, adapted for WAN:

```rust
// Phase 1: Reduce-Scatter (N-1 steps)
// Each device accumulates one chunk
for step in 0..(n-1) {
    let received = send_to_right_recv_from_left(chunks[send_idx]);
    chunks[recv_idx] = chunks[recv_idx].add(&received);
}

// Phase 2: All-Gather (N-1 steps)
// Distribute all chunks to all devices
for step in 0..(n-1) {
    chunks[recv_idx] = send_to_right_recv_from_left(chunks[send_idx]);
}

// Result: All devices have identical complete tensor
```

**Bandwidth Efficiency:**
- Each device transfers ~1.8× tensor size (optimal)
- Compare to naive all-to-all: 90 transfers for 10 workers
- Ring: 18 total steps
- Full bisection bandwidth utilized

### Tensor-Parallel Forward Pass

```rust
// Each worker computes partial matmul with their columns
let partial_output = input.matmul(&my_shard.weights)?;

// Ring all-reduce combines results
let full_output = ring_all_reduce(workers, partial_output).await?;

// Apply activation (each worker does this identically)
let activated = full_output.gelu()?;
```

Repeat for all 70 transformer layers. Each layer takes ~90ms (50ms compute + 40ms all-reduce).

### NAT Traversal with libp2p
- DCUTR (Direct Connection Upgrade through Relay) for hole-punching
- Automatic fallback to relay if direct connection fails
- Works behind NATs, firewalls, CGNATs
- WebRTC-style connectivity without WebRTC complexity

### Checkpointing for Fault Tolerance
- Workers checkpoint KV cache + generated tokens every 50 tokens
- Control plane tracks checkpoint metadata
- Device failures trigger recovery from latest checkpoint
- Only lose ~30 seconds of work instead of entire job

## Tech Stack

**Control Plane:** Rust + Axum + SQLite
**Workers:** Rust + libp2p + tokio
**Networking:** libp2p (Circuit Relay v2, DCUTR, Request-Response)
**Serialization:** CBOR (tensors), JSON (API)
**Cryptography:** Ed25519 (device identity), Noise (transport)

## Getting Started

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/saint0x/mesh.git
cd mesh
```

### Quick Start

```bash
# Terminal 1: Start relay server
cargo run --release --bin relay-server

# Terminal 2: Start control plane
cargo run --release --bin control-plane

# Terminal 3: Initialize and start first worker
cargo run --release --bin agent -- init --network-id demo --name "Worker 1"
cargo run --release --bin agent -- join-ring --model-id llama-70b
cargo run --release --bin agent -- start

# Terminal 4: Initialize and start second worker
export MESHNET_HOME=~/.meshnet-worker2
cargo run --release --bin agent -- init --network-id demo --name "Worker 2"
cargo run --release --bin agent -- join-ring --model-id llama-70b
cargo run --release --bin agent -- start

# Terminal 5: Submit inference job
cargo run --release --bin agent -- inference --prompt "Hello, world!" --max-tokens 10
```

Workers will poll for jobs, execute distributed inference via ring all-reduce, and log completion.

### Run Tests

```bash
# Run full test suite (298 tests)
cargo test --release

# Run specific component tests
cargo test --release -p control-plane
cargo test --release -p agent
cargo test --release -p relay-server
```

## Project Structure

```
mesh/
├── agent/                  # Worker daemon + CLI
│   ├── src/inference/      # Tensor operations, forward pass, coordinator
│   ├── src/network/        # libp2p swarm, ring topology
│   ├── src/model/          # Shard registry, assignments
│   └── src/checkpoint/     # Fault tolerance
│
├── control-plane/          # Control plane API
│   ├── src/api/            # REST endpoints (ring, inference)
│   ├── src/services/       # Ring manager, topology notifier
│   └── src/db/             # SQLite database
│
└── relay-server/           # NAT traversal relay
    └── src/relay.rs        # Circuit Relay v2 server
```

## Documentation

- **Architecture:** [`INSIGHT.md`](INSIGHT.md) - Detailed architectural vision and implementation
- **API Reference:** `cargo doc --open --no-deps`
- **Relay Deployment:** [`relay-server/README.md`](relay-server/README.md)

## Current Status

**Phase 1: ✅ COMPLETE** - Full tensor-parallel distributed inference operational
- Control plane job distribution and polling
- Worker ring topology with P2P connections
- Inference coordinator with ring all-reduce
- Complete tensor operations and forward pass
- Mock weight validation framework (298 tests passing)

**Next: Phase 2** - Executor containers and OpenAI-compatible API

## Target Use Cases

**Small Teams & Friend Groups**
- Pool devices within a trusted network (like Tailscale)
- Share access to large models cooperatively
- Fair allocation based on contribution

**Local Development**
- Distribute inference across your own devices
- Test large model behavior without cloud costs
- Iterate quickly with local compute

**Research & Education**
- Study distributed inference algorithms
- Experiment with tensor parallelism
- Learn production distributed systems techniques

## Non-Goals

- ❌ Public anonymous marketplace compute
- ❌ Blockchain or cryptocurrency
- ❌ Competitive pricing or bidding
- ❌ Untrusted workers (assume cooperative pools)

## License

MIT

---

**Like DeepSpeed, but for consumer devices. Like Tailscale, but for AI inference.**

This is not a marketplace. This is not a blockchain. This is a cooperative compute pool using state-of-the-art distributed inference techniques.
