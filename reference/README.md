# Carcass VPN Reference Implementation

This directory contains the **con** (VPN mesh networking) module from the Carcass project as a reference implementation. This code is **NOT used directly** in MeshNet - it serves as architectural inspiration for our own implementation.

## Source

- **Project:** Carcass - https://github.com/cull-os/carcass
- **License:** MPL-2.0 (Mozilla Public License 2.0)
- **Authors:** Cull OS contributors (RGBCube <git@rgbcu.be>)
- **Module:** `con` - VPN mesh networking using libp2p

## What This Code Does

The `con` module is a production mesh VPN implementation that:
- Creates virtual network interfaces (TUN devices) on each peer
- Routes IP packets between peers using libp2p for NAT traversal
- Uses relay servers for NAT-challenged devices (DCUTR protocol)
- Generates deterministic IPv4/IPv6 addresses from peer IDs
- Implements custom IP packet forwarding protocol

**Key Difference from MeshNet:** This code routes IP packets for a VPN. MeshNet routes **job payloads** for distributed AI workloads. We adapt the patterns, not the code.

## Files in This Reference

### Core Networking Files

#### `behaviour.rs` (231 lines) ⭐ **CRITICAL REFERENCE**
**What it does:**
- Composes libp2p behaviors (Identify, Relay, DCUTR, Kademlia, custom IP protocol)
- Sets up SwarmBuilder with QUIC + TCP transports
- Configures Noise encryption + Yamux multiplexing
- Implements main event loop

**What we learn:**
- How to compose multiple libp2p protocols
- Relay + DCUTR pattern for NAT traversal
- Transport stack configuration (QUIC, TCP, Noise, Yamux)
- Event-driven architecture with tokio::select!

**MeshNet adaptation:**
- Remove Kademlia DHT (we use centralized control plane)
- Replace IP protocol with JobProtocol
- Keep relay + DCUTR setup (proven NAT solution)

---

#### `ip.rs` (354 lines) ⭐ **CRITICAL REFERENCE**
**What it does:**
- Implements custom libp2p NetworkBehaviour for IP packet forwarding
- Uses ring buffer queues (256 packets per peer, lock-free)
- Async state machine: Reading → Writing → Idle
- Length-prefixed packet framing (u16 LE bytes)

**What we learn:**
- Async state machine pattern for protocol handlers
- Lock-free ring buffer queuing (ringbuf crate)
- Producer/Consumer pattern for async work
- Connection lifecycle management

**MeshNet adaptation:**
- Replace `Packet(Vec<u8>)` with `JobEnvelope { job_id, payload, ... }`
- Change serialization from raw bytes to CBOR
- Add `Executing` state for running jobs
- Reduce buffer size 256 → 64 (jobs are heavier)

**Code to study:**
- Lines 100-103: Ring buffer setup
- Lines 88-241: Handler state machine
- Lines 243-354: NetworkBehaviour implementation

---

#### `config.rs` (128 lines)
**What it does:**
- Generates Ed25519 keypairs for device identity
- Serializes keypairs with multibase encoding (human-readable)
- TOML configuration format
- Bootstrap peer management

**What we learn:**
- Custom serde for cryptographic keys
- TOML config pattern
- Keypair generation and persistence

**MeshNet adaptation:**
- Keep keypair generation + multibase serialization
- Add `network_id`, `control_plane_url` fields
- Add `DeviceCapabilities` (CPU, GPU, RAM, tier)
- Replace bootstrap peers with relay addresses from control plane

---

#### `address.rs` (140 lines)
**What it does:**
- Generates IPv4/IPv6 addresses from peer IDs using XOR
- Bidirectional mapping (peer → IP, IP → peer)
- Deterministic address allocation

**What we learn:**
- Deterministic ID generation pattern
- Bidirectional hash maps

**MeshNet adaptation:**
- ⚠️ **DO NOT USE** - We don't need IP address generation
- MeshNet uses device IDs (UUIDs) directly
- Note: Has collision risk (TODO on line 79)

---

#### `interface.rs` (42 lines)
**What it does:**
- Wraps TUN device for virtual network interface
- Platform-specific handling (macOS, Linux)
- MTU configuration (1420 bytes)

**MeshNet adaptation:**
- ⚠️ **DO NOT USE** - MeshNet doesn't route IP packets
- We route job streams at application layer

---

### Supporting Files

#### `main.rs` (3799 lines)
CLI application entry point. Commands like `inbox`, `peers`, `ping` are marked `todo!()` (incomplete).

#### `mod.rs` (240 lines)
Module exports and public API surface.

#### `Cargo.toml` (1023 lines)
Dependencies for the VPN module.

---

## Key Patterns to Adopt

### 1. Relay + DCUTR NAT Traversal
```rust
// From behaviour.rs lines 40-47
#[derive(NetworkBehaviour)]
pub struct MeshSwarm {
    identify: libp2p::identify::Behaviour,
    relay_client: libp2p::relay::client::Behaviour,
    dcutr: libp2p::dcutr::Behaviour,
    job_protocol: JobProtocol,  // Our custom protocol
}
```

**Why this works:**
- All devices connect to relay server (solves NAT)
- DCUTR upgrades relay connections to direct P2P when possible
- Mobile-friendly (works behind any NAT)

---

### 2. Ring Buffer Async Queuing
```rust
// From ip.rs lines 100-103
const JOB_BUFFER_SIZE: usize = 64;  // Reduced from 256

type JobProducer = ringbuf::CachingProd<
    Arc<ringbuf::StaticRb<JobEnvelope, JOB_BUFFER_SIZE>>
>;
type JobConsumer = ringbuf::CachingCons<
    Arc<ringbuf::StaticRb<JobEnvelope, JOB_BUFFER_SIZE>>
>;
```

**Why this works:**
- Lock-free, bounded queue (no unbounded memory growth)
- Prevents scheduler from overwhelming device
- Graceful overflow handling (drop oldest job)

---

### 3. Async State Machine for Protocol Handler
```rust
// From ip.rs lines 88-92
enum ConnectionState {
    Reading(BoxFuture<'static, io::Result<(Stream, JobEnvelope)>>),
    Writing(BoxFuture<'static, io::Result<Stream>>),
    Executing(Uuid, BoxFuture<'static, io::Result<JobResult>>),  // Added for MeshNet
    Idle(Option<Stream>),
}
```

**Why this works:**
- Clean separation of async operations
- No blocking in event loop
- Proper futures lifecycle management

---

### 4. Ed25519 Identity + Multibase Encoding
```rust
// From config.rs lines 12-37
mod keypair_serde {
    pub fn serialize(keypair: &ed25519::Keypair, s: S) -> Result<S::Ok, S::Error> {
        let encoded = multibase::encode(multibase::Base::Base58Btc, keypair.to_bytes());
        s.serialize_str(&encoded)
    }
}
```

**Why this works:**
- Human-readable keys in config files
- Compatible with libp2p peer IDs
- No passwords needed (certificate-based trust)

---

## What NOT to Copy

❌ **TUN device code** (interface.rs) - We don't route IP packets
❌ **IP address generation** (address.rs) - Has collision risk, unnecessary for job routing
❌ **Kademlia DHT** (behaviour.rs) - We use centralized control plane
❌ **Incomplete CLI commands** (main.rs) - Build MeshNet-specific CLI

---

## Implementation Checklist

When implementing MeshNet networking based on this reference:

### Week 3-4: Core Networking (from behaviour.rs + ip.rs)
- [ ] Create `src/network/mesh_swarm.rs` (inspired by behaviour.rs)
  - [ ] Set up libp2p SwarmBuilder with QUIC + TCP + Noise + Yamux
  - [ ] Add Relay client behavior
  - [ ] Add DCUTR behavior
  - [ ] Remove Kademlia (replace with control plane discovery)
- [ ] Create `src/network/job_protocol.rs` (inspired by ip.rs)
  - [ ] Define `JobEnvelope` type (replace `Packet`)
  - [ ] Implement CBOR serialization (replace raw bytes)
  - [ ] Port async state machine pattern
  - [ ] Port ring buffer queuing pattern
  - [ ] Add `Executing` state for job execution
- [ ] Deploy relay server (libp2p relay daemon)
- [ ] Test NAT traversal with mobile device

### Week 5-6: Device Identity (from config.rs)
- [ ] Create `src/device/device_config.rs` (inspired by config.rs)
  - [ ] Port Ed25519 keypair generation
  - [ ] Port multibase serde implementation
  - [ ] Add MeshNet-specific fields (network_id, capabilities, control_plane_url)
  - [ ] TOML config format
- [ ] Implement device registration with control plane
- [ ] Test config save/load

---

## Code Study Guide

### If you're new to libp2p:
1. Start with `behaviour.rs` - understand how protocols compose
2. Read libp2p docs: https://docs.libp2p.io/
3. Understand Relay protocol: https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md

### If you're implementing job protocol:
1. Study `ip.rs` lines 88-241 (Handler state machine)
2. Study `ip.rs` lines 243-354 (NetworkBehaviour impl)
3. Read ringbuf docs: https://docs.rs/ringbuf/latest/ringbuf/

### If you're implementing device config:
1. Study `config.rs` lines 12-37 (keypair serde)
2. Study `config.rs` lines 65-128 (config generation)
3. Read multibase docs: https://docs.rs/multibase/latest/multibase/

---

## License Compliance

This reference code is licensed under **MPL-2.0** (Mozilla Public License 2.0).

### What This Means:
- ✅ We can study and learn from this code
- ✅ We can implement similar patterns in our own code
- ✅ MeshNet code can be under any license we choose
- ⚠️ If we copy code verbatim, those files must remain MPL-2.0
- ✅ Architectural patterns and algorithms are not copyrightable

### Our Approach:
1. **Study** this code to understand libp2p patterns
2. **Reimplement** patterns in MeshNet with our own names and structure
3. **Adapt** for job execution use case (not IP routing)
4. **Document** which patterns we learned from Carcass

**Full License:** See https://www.mozilla.org/en-US/MPL/2.0/

---

## File Size Reference

| File | Lines | Complexity | Study Priority |
|------|-------|------------|----------------|
| ip.rs | 354 | High | ⭐⭐⭐ Critical |
| behaviour.rs | 231 | Medium | ⭐⭐⭐ Critical |
| config.rs | 128 | Low | ⭐⭐ Important |
| address.rs | 140 | Low | ⚠️ Skip (not applicable) |
| interface.rs | 42 | Low | ⚠️ Skip (not applicable) |
| main.rs | 3799 | Medium | ⚠️ Skip (incomplete) |

**Total useful reference code: ~700 lines** (behaviour.rs + ip.rs + config.rs)

---

## Additional Resources

- **Carcass Project:** https://github.com/cull-os/carcass
- **libp2p Documentation:** https://docs.libp2p.io/
- **Relay Specification:** https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md
- **Ringbuf Crate:** https://docs.rs/ringbuf/latest/ringbuf/
- **Multibase Spec:** https://github.com/multiformats/multibase

---

**Remember:** This is **reference code**, not production code for MeshNet. We learn the patterns and reimplement them for our job execution use case.
