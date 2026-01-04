# Mesh

> Tailscale-style private networks that pool local + trusted compute to run AI workloads.

Create a network, add devices and people, and route AI jobs across pooled compute with simple credits.

## Project Status

**Current Phase:** Phase 1.5 - Network Layer Implementation (In Progress)
**Completed Modules:** 5/12 Phase 1 modules
**Test Coverage:** 41 tests passing (34 unit + 7 doc tests)
**Next Milestone:** Module 2.3 - Control Plane Registration API

### Completed Modules

✅ **Module 1.1: Project Foundation** (Cargo workspace, dependencies)
✅ **Module 1.2: Device Identity** (Ed25519 keypairs, multibase serialization)
✅ **Module 1.3: Database Schema** (PostgreSQL + SQLite migrations)
✅ **Module 1.4: Relay Server** (Circuit Relay v2 + token auth)
✅ **Module 1.5: Network Swarm** (libp2p with Identify, RelayClient, DCUTR)
✅ **Module 2.2: Job Protocol** (Request-response job distribution with CBOR)

### Currently Building

🚧 **Module 2.3:** Control Plane Registration API
🚧 **Module 2.4:** Job execution engine

## What Is This?

Mesh enables:
- **Private compute networks** (like Tailscale) for trusted groups
- **Opportunistic AI workload pooling** across your devices (desktop, mobile)
- **Simple credit system** for fair resource allocation
- **NAT-friendly networking** with minimal setup

### Target Users
- Builders running local AI across their personal devices
- Small teams that want private pooled compute (friends/team tailnet)
- Communities who opt-in to share compute within a private network

### Non-Goals (v1)
- ❌ Public anonymous marketplace compute
- ❌ Precise FLOPs metering (basic credit system is sufficient for v1)
- ❌ Model-parallel distributed inference across many unreliable mobile nodes
- ❌ On-chain tokenomics

### Core Requirements
- ✅ Cryptographically secure verification of compute results
- ✅ Trust score system with fraud detection
- ✅ VRF-based spot-checking for job validation

## Project Structure

```
mesh/
├── PLAN.md                # Full product specification (JSON)
├── IMPLEMENTATION.md      # Phase-by-phase implementation roadmap
├── README.md              # This file
├── Cargo.toml             # Rust workspace configuration
│
├── agent/                 # ✅ Desktop agent implementation (Rust)
│   ├── src/
│   │   ├── device/        # ✅ Ed25519 identity, capabilities detection
│   │   ├── network/       # ✅ libp2p mesh swarm + job protocol
│   │   │   ├── mesh_swarm.rs    # ✅ Relay + DCUTR + Job Protocol
│   │   │   ├── job_protocol.rs  # ✅ Request-response job distribution
│   │   │   └── events.rs        # ✅ Network event types
│   │   └── errors.rs      # ✅ Error handling
│   └── examples/
│       └── relay_connectivity.rs # ✅ Integration test example
│
├── relay-server/          # ✅ Relay server implementation (Rust)
│   ├── src/
│   │   ├── relay.rs       # ✅ Circuit Relay v2 server
│   │   ├── config.rs      # ✅ Configuration + auth tokens
│   │   └── auth.rs        # ✅ Token-based authentication
│   └── README.md          # ✅ Deployment guide
│
├── control-plane/         # 🚧 Control plane (stub, TypeScript planned)
│   └── src/
│       └── db/            # ✅ PostgreSQL schema + migrations
│
└── reference/             # Reference VPN code study (DO NOT USE DIRECTLY)
    ├── README.md          # Study guide for reference implementation
    ├── behaviour.rs       # libp2p swarm composition ⭐
    ├── ip.rs              # Protocol handler + ring buffers ⭐
    └── config.rs          # Ed25519 key management ⭐
```

## Reference Implementation

The `reference/` directory contains a VPN mesh networking module (see `.env` for details, MPL-2.0 license) as architectural inspiration.

**Key files to study:**
1. **reference/behaviour.rs** (231 lines) - libp2p swarm with Relay + DCUTR for NAT traversal
2. **reference/ip.rs** (354 lines) - Async protocol handler with lock-free ring buffer queues
3. **reference/config.rs** (128 lines) - Ed25519 keypair generation + multibase serialization

See `reference/README.md` for detailed study guide.

### What We Learn from Reference Implementation
- ✅ Relay + DCUTR pattern for NAT traversal (battle-tested)
- ✅ Lock-free ring buffer queuing (256 jobs/device)
- ✅ Async state machine for protocol handlers
- ✅ Ed25519 identity + mTLS transport
- ✅ libp2p protocol composition

### What We DON'T Copy
- ❌ TUN device / IP routing (Mesh routes job streams, not IP packets)
- ❌ IP address generation (has collision risk, unnecessary)
- ❌ Kademlia DHT (Mesh uses centralized control plane)

## Implementation Roadmap

See `IMPLEMENTATION.md` for the complete phase-by-phase checklist.

### Phase 1: Foundation & Infrastructure (Current)

**Status:** 5/12 modules complete

**✅ Completed:**
- Device identity system (Ed25519 keypairs, multibase serialization)
- Database schemas (PostgreSQL + SQLite migrations)
- Relay server (Circuit Relay v2 with token auth)
- Network swarm (libp2p: Identify + RelayClient + DCUTR)
- Job protocol (request-response with CBOR serialization)

**🚧 In Progress:**
- Control plane registration API
- Job execution engine
- Desktop agent CLI
- Ledger & credit system

**Deliverables:**
- ✅ Relay server deployable
- ✅ Agent can connect to relay and establish circuits
- ✅ Job protocol can send/receive job requests
- 🚧 Desktop agent can register with control plane
- 🚧 Desktop agent can execute jobs locally
- 🚧 Credit tracking functional

### Phase 2: Desktop MVP (Planned)

**Goal:** Working desktop-only proof of concept

**Major components:**
- Desktop agent CLI
- Control plane API (device registration, network management)
- Embeddings workload (first workload type)
- Credit ledger system
- Job routing and execution

**Success criteria:**
- 2+ desktop devices can join a network
- Jobs route between devices via relay
- Credits properly tracked
- End-to-end job execution verified

### Phase 3: Mobile & Production (Future)

**Goal:** Mobile agents + production hardening

**Major components:**
- iOS + Android mobile agents
- Multiple workload types (OCR, chat, etc.)
- Web dashboard
- P2P fast-path optimization
- Security audit & monitoring

## Tech Stack

### Control Plane
- **Language:** TypeScript (Node.js) or Rust (Axum)
- **Database:** PostgreSQL 15+
- **Cache:** Redis 7+
- **Auth:** NextAuth.js or OAuth libraries

### Relay Gateway
- **Language:** Rust
- **Framework:** tokio + libp2p
- **Protocol:** QUIC (quinn) + WebSocket fallback

### Agents
- **Desktop:** Rust (tokio, libp2p)
- **iOS:** Swift + Core ML
- **Android:** Kotlin + TFLite

### Web Dashboard
- **Framework:** Next.js 14
- **Language:** TypeScript
- **Styling:** Tailwind CSS

## Getting Started

### Prerequisites
- Rust 1.75+ (https://rustup.rs)
- PostgreSQL 15+ (for control plane)
- Git

### Build & Test

```bash
# Clone repository
git clone <repo-url>
cd meshnet

# Build all components
cargo build --workspace

# Run tests (41 tests)
cargo test --workspace

# Run clippy
cargo clippy --workspace -- -D warnings
```

### Run Relay Server

```bash
# Start relay server (default: localhost:4001)
cargo run --bin relay-server

# Or with custom config
cargo run --bin relay-server -- --config relay.toml
```

See `relay-server/README.md` for deployment guide.

### Test Agent Connectivity

```bash
# Terminal 1: Start relay server
cargo run --bin relay-server

# Terminal 2: Run integration test (two agents via relay)
RUST_LOG=debug cargo run --example relay_connectivity

# Expected: Both agents connect to relay and establish circuit
```

### Study the Reference Implementation
```bash
cd reference
cat README.md  # Comprehensive study guide
```

## Documentation

- **Product Spec:** `PLAN.md` - Full product specification (JSON format)
- **Implementation Guide:** `IMPLEMENTATION.md` - Phase-by-phase implementation checklist
- **Reference Code Study:** `reference/README.md` - Study guide for reference VPN implementation
- **Relay Server:** `relay-server/README.md` - Deployment and configuration guide

## Key Technologies

- **Networking:** libp2p 0.56 (Circuit Relay v2, DCUTR, Request-Response)
- **Serialization:** CBOR (job envelopes), TOML (config), JSON (API)
- **Cryptography:** Ed25519 (device identity), Noise (transport encryption)
- **Database:** PostgreSQL (control plane), SQLite (local agent storage)
- **Async Runtime:** Tokio 1.47
- **Logging:** tracing + tracing-subscriber

## Resources

- **libp2p Documentation:** https://docs.libp2p.io/
- **Circuit Relay v2 Spec:** https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md
- **CBOR Spec:** https://cbor.io/

## License

**Mesh code:** MIT
**Reference code (reference/):** MPL-2.0 (from Carcass project)

See `reference/README.md` for license compliance details.

## Attribution

Network architecture patterns inspired by a reference mesh VPN implementation (see `.env` for details), licensed under MPL-2.0. We study libp2p implementation patterns and adapt them for job execution (not IP routing).

---

**Build Status:** ✅ All tests passing (41 tests)
**Current Phase:** Phase 1 - Foundation & Infrastructure (5/12 modules complete)
**Next Milestone:** Module 2.3 - Control Plane Registration API
