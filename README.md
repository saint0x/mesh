# Mesh

> Tailscale-style private networks that pool local + trusted compute to run AI workloads.

Create a network, add devices and people, and route AI jobs across pooled compute with simple credits.

## Project Status

**Phase:** Initial Planning & Reference Study
**Timeline:** 8-11 months to production (realistic)
**Team Size Needed:** 5-7 people, ~7 FTE

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
- ❌ Precise FLOPs metering / cryptographic verifiable compute
- ❌ Model-parallel distributed inference across many unreliable mobile nodes
- ❌ On-chain tokenomics

## Project Structure

```
mesh/
├── plan.md                 # Full product specification (JSON)
├── IMPLEMENTATION.md       # Phase-by-phase implementation roadmap (200+ tasks)
├── README.md              # This file
├── Cargo.toml             # Rust workspace configuration
├── reference/             # Carcass VPN reference code (DO NOT USE DIRECTLY)
│   ├── README.md          # Study guide for reference implementation
│   ├── behaviour.rs       # libp2p swarm composition ⭐
│   ├── ip.rs              # Protocol handler + ring buffers ⭐
│   ├── config.rs          # Ed25519 key management ⭐
│   └── ...
└── (future directories)
    ├── agent/             # Desktop and mobile agent (Rust)
    ├── relay/             # Relay server (Rust)
    └── control-plane/     # Control plane API (TypeScript or Rust)
```

## Reference Implementation

The `reference/` directory contains the **con** (VPN mesh networking) module from the Carcass project (https://github.com/cull-os/carcass, MPL-2.0 license) as architectural inspiration.

**Key files to study:**
1. **reference/behaviour.rs** (231 lines) - libp2p swarm with Relay + DCUTR for NAT traversal
2. **reference/ip.rs** (354 lines) - Async protocol handler with lock-free ring buffer queues
3. **reference/config.rs** (128 lines) - Ed25519 keypair generation + multibase serialization

See `reference/README.md` for detailed study guide.

### What We Learn from Carcass
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

### Phase 0: Prototype (6-8 weeks)
**Goal:** Desktop-only network with relay-based embeddings job execution

**Critical validations (Week 1-2):**
- [ ] iOS background compute test (SHOWSTOPPER RISK - 40% failure probability)
- [ ] Relay throughput benchmark
- [ ] mTLS on mobile compatibility

**Core networking (Week 3-4):**
- [ ] libp2p swarm setup (inspired by reference/behaviour.rs)
- [ ] Job protocol handler (inspired by reference/ip.rs)
- [ ] Deploy relay server
- [ ] Test NAT traversal

**Deliverables:**
- 2 desktop devices can join network via relay
- Embeddings job executes successfully
- Ledger tracks credits burned

### Phase 1: MVP (12-16 weeks after Phase 0)
**Goal:** Production-ready alpha with mobile agents, 2 workloads, web dashboard

**Major components:**
- iOS + Android mobile agents (8-10 weeks - HIGHEST RISK)
- OCR + Small Chat workloads
- Web dashboard (ledger UI, device management)
- Roles & policies (admin/member/guest)
- Job retries & streaming

**Success metrics:**
- 25-50 alpha users
- 50% weekly active users run ≥1 job
- 70% job success rate

### Phase 2: Production Hardening (16-20 weeks after Phase 1)
**Goal:** Public beta ready

**Major components:**
- P2P fast-path (reduce relay bandwidth by 50%+)
- Advanced scheduling (eligibility + weighted scoring)
- Security audit + GDPR compliance
- Monitoring & alerting

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
- Git

### Study the Reference Implementation
```bash
cd meshnet/reference
cat README.md  # Comprehensive study guide
```

### Start Phase 0 Implementation

**Week 1-2: Critical Validation**
1. Build iOS background compute test
2. Benchmark relay server throughput
3. Test mTLS on mobile

**Decision Point:** Proceed based on iOS validation results.

## Documentation

- **Product Spec:** `plan.md` (668 lines JSON)
- **Implementation Guide:** `IMPLEMENTATION.md` (phase-by-phase checklist)
- **Reference Code Study:** `reference/README.md`

## License

**Mesh code:** MIT
**Reference code (reference/):** MPL-2.0 (from Carcass project)

See `reference/README.md` for license compliance details.

## Attribution

Network architecture patterns inspired by the Carcass mesh VPN project (https://github.com/cull-os/carcass), licensed under MPL-2.0. We study their libp2p implementation patterns and adapt them for job execution (not IP routing).

## Critical Risks

1. **iOS Background Compute (40% failure probability)**
   - **Impact:** Mobile value prop collapses
   - **Mitigation:** Validate in Week 1-2, pivot to foreground-only if needed

2. **Relay Bandwidth Costs (30% probability)**
   - **Impact:** Unit economics break
   - **Mitigation:** Use Cloudflare for cheap egress, accelerate P2P fast-path

3. **Credit System Gaming (50% probability at scale)**
   - **Impact:** Economic collapse
   - **Mitigation:** Spot-check validation, rate limits, social trust

## Team Requirements

| Role | FTE | Skills |
|------|-----|--------|
| Backend Engineer | 2.0 | Rust/Go, distributed systems, databases |
| iOS Engineer | 1.0 | Swift, Core ML, **background execution expertise** |
| Android Engineer | 1.0 | Kotlin, NNAPI/TFLite |
| Desktop Engineer | 0.5 | Rust, cross-platform |
| ML Engineer | 0.75 | Model optimization, quantization |
| DevOps | 0.5 | Kubernetes, PostgreSQL, Redis |
| Product/Design | 0.5 | Developer tools UX |

**Total: 5-7 people, ~7 FTE**

## Resources

- **Carcass Project:** https://github.com/cull-os/carcass
- **libp2p Documentation:** https://docs.libp2p.io/
- **Relay Specification:** https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md

## Contact

TBD

---

**Status:** Initial planning complete. Ready to begin Phase 0 (Week 1-2: Critical Validation Prototypes).
