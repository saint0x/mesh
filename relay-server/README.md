# Mesh Relay Server

Production-ready libp2p Circuit Relay v2 server for NAT traversal and device connectivity in the Mesh network.

## Overview

The relay server enables Mesh agents to connect to each other even when behind NATs or firewalls. It implements [Circuit Relay v2](https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md) using the libp2p Rust stack.

**Key Features:**
- Dual transport: TCP (port 4001) + QUIC (UDP 4001)
- Resource limits: Configurable reservations, circuits, duration, and bytes
- Structured logging with tracing (JSON or pretty format)
- Token-based authentication (placeholder for MVP)
- Graceful shutdown handling
- Persistent relay identity (keypair stored at `~/.meshnet/relay_keypair.bin`)

## Quick Start

### 1. Generate Default Configuration

```bash
cargo run --bin relay-server -- --generate-config
```

This creates `~/.meshnet/relay.toml` with default settings.

### 2. Configure the Relay

Edit `~/.meshnet/relay.toml`:

```toml
[relay]
max_reservations = 100              # Global reservation limit
max_reservations_per_peer = 5       # Per-peer limit
max_circuits_per_peer = 16          # Concurrent circuits per peer
max_circuit_duration_secs = 120     # Circuit lifetime (2 minutes)
max_circuit_bytes = 10485760        # Max data per circuit (10MB)

[network]
tcp_listen_addr = "/ip4/0.0.0.0/tcp/4001"
quic_listen_addr = "/ip4/0.0.0.0/udp/4001/quic-v1"

[auth]
auth_token = "CHANGE_ME_IN_PRODUCTION"
auth_enabled = false                # Enable for production

[logging]
level = "info"                      # trace, debug, info, warn, error
log_format = "pretty"               # or "json"
```

**⚠️ IMPORTANT:** Change `auth_token` before enabling authentication in production!

### 3. Start the Relay Server

```bash
cargo run --bin relay-server
```

Or with a custom config path:

```bash
cargo run --bin relay-server -- --config /path/to/relay.toml
```

You should see:

```
Mesh Relay Server is running!
Press Ctrl+C to stop

Listening on: /ip4/127.0.0.1/tcp/4001
Listening on: /ip4/127.0.0.1/udp/4001/quic-v1
```

## Configuration Reference

### Relay Settings

| Field | Default | Description |
|-------|---------|-------------|
| `max_reservations` | 100 | Global limit for concurrent reservations |
| `max_reservations_per_peer` | 5 | Prevents peers from monopolizing resources |
| `max_circuits_per_peer` | 16 | Maximum concurrent relay circuits per peer |
| `max_circuit_duration_secs` | 120 | Circuit lifetime before automatic closure |
| `max_circuit_bytes` | 10485760 | Maximum data transfer per circuit (10MB) |

### Network Settings

| Field | Default | Description |
|-------|---------|-------------|
| `tcp_listen_addr` | `/ip4/0.0.0.0/tcp/4001` | TCP listen multiaddr |
| `quic_listen_addr` | `/ip4/0.0.0.0/udp/4001/quic-v1` | QUIC listen multiaddr |

### Authentication Settings (Placeholder)

| Field | Default | Description |
|-------|---------|-------------|
| `auth_token` | `CHANGE_ME_IN_PRODUCTION` | Shared secret for token validation |
| `auth_enabled` | `false` | Enable token authentication |

**Note:** Token auth is a placeholder for MVP. Production deployments (Phase 1) will use mTLS with control plane integration.

### Logging Settings

| Field | Default | Description |
|-------|---------|-------------|
| `level` | `info` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `log_format` | `pretty` | Output format: `pretty` (human-readable) or `json` (structured) |

## Testing

### Run Unit Tests

```bash
cargo test -p relay-server
```

This runs 16 tests covering:
- Configuration validation
- Error handling
- Keypair persistence
- Swarm initialization
- Token authentication

### Test with Example Client

The relay server includes a test client example for integration testing.

**Terminal 1: Start the relay server**
```bash
cargo run --bin relay-server
```

**Terminal 2: Start client A**
```bash
cargo run --example test_client -- --name clientA
```

**Terminal 3: Start client B**
```bash
cargo run --example test_client -- --name clientB
```

The clients will:
1. Connect to the relay at `127.0.0.1:4001`
2. Make reservations
3. Listen for incoming connections via relay
4. Log all relay events

## CLI Options

```
Mesh Relay Server for NAT traversal and device connectivity

Usage: relay-server [OPTIONS]

Options:
  -c, --config <CONFIG>
          Configuration file path [default: ~/.meshnet/relay.toml]

  -l, --log-level <LOG_LEVEL>
          Override log level (trace, debug, info, warn, error)

      --generate-config
          Generate default config and exit

  -h, --help
          Print help

  -V, --version
          Print version
```

## Architecture

### Relay Identity

The relay generates an Ed25519 keypair on first run and stores it at `~/.meshnet/relay_keypair.bin`. This ensures:
- Consistent PeerId across restarts
- Clients can reliably reference the relay
- Control plane can register the relay's address

### Resource Management

The relay enforces multiple layers of limits:

1. **Global Reservation Limit** (`max_reservations`): Total concurrent reservations
2. **Per-Peer Reservation Limit** (`max_reservations_per_peer`): Prevents monopolization
3. **Circuit Limits** (`max_circuits_per_peer`): Controls relay load
4. **Circuit Duration** (`max_circuit_duration_secs`): Automatic cleanup
5. **Circuit Data** (`max_circuit_bytes`): Bandwidth management

### Event Handling

The relay logs structured events for monitoring:

**Connection Events:**
- `ConnectionEstablished` - Peer connected (peer_id, endpoint, num_established)
- `ConnectionClosed` - Peer disconnected (peer_id, cause, remaining)

**Reservation Events:**
- `ReservationReqAccepted` - Reservation granted (peer_id, renewed flag)
- `ReservationReqDenied` - Reservation rejected (likely hit max limit)
- `ReservationTimedOut` - Reservation expired (peer_id)

**Circuit Events:**
- `CircuitReqAccepted` - Circuit established (src_peer_id → dst_peer_id)
- `CircuitReqDenied` - Circuit rejected (src_peer_id → dst_peer_id)
- `CircuitClosed` - Circuit terminated (src_peer_id → dst_peer_id, error)

## Integration with Mesh Network

### Agent Connection Flow

1. **Agent Registration**: Agent registers with control plane
2. **Relay Address**: Control plane returns relay multiaddr
3. **Dial Relay**: Agent dials relay: `/ip4/127.0.0.1/tcp/4001/p2p/<relay_peer_id>`
4. **Make Reservation**: Agent makes reservation to listen for incoming connections
5. **Circuit Communication**: Other agents connect via relay circuits

### Control Plane Integration

The control plane's registration endpoint returns relay addresses:

```json
{
  "relay_addresses": [
    "/ip4/127.0.0.1/tcp/4001/p2p/12D3KooW...",
    "/ip4/127.0.0.1/udp/4001/quic-v1/p2p/12D3KooW..."
  ]
}
```

Agents use these addresses for NAT traversal and fallback connectivity.

## Troubleshooting

### Relay Won't Start

**Error:** `Configuration file not found`
```bash
relay-server --generate-config
```

**Error:** `Address already in use`
- Another process is using port 4001
- Change `tcp_listen_addr` and `quic_listen_addr` in config

### Clients Can't Connect

**Check relay is listening:**
```bash
# TCP
netstat -an | grep 4001

# QUIC (UDP)
netstat -an | grep -i udp | grep 4001
```

**Check firewall:**
```bash
# macOS
sudo pfctl -s rules | grep 4001

# Linux
sudo iptables -L -n | grep 4001
```

**Enable debug logging:**
```bash
relay-server --log-level debug
```

### Reservations Denied

**Cause:** Hit global or per-peer reservation limit

**Solution:** Increase limits in config:
```toml
[relay]
max_reservations = 200              # Increase global limit
max_reservations_per_peer = 10      # Increase per-peer limit
```

**Monitor:** Watch logs for `ReservationReqDenied` events

### High Resource Usage

**Check circuit limits:**
```toml
[relay]
max_circuits_per_peer = 8           # Reduce concurrent circuits
max_circuit_duration_secs = 60      # Reduce circuit lifetime
max_circuit_bytes = 5242880         # Reduce max data (5MB)
```

**Enable JSON logging for metrics:**
```toml
[logging]
log_format = "json"
```

Then parse logs to track:
- Active reservations per peer
- Circuit duration distribution
- Data transfer per circuit

## Production Deployment

### Current Status (MVP)

The relay server is production-ready for **local development** with:
- ✅ Resource limits and validation
- ✅ Structured logging and error handling
- ✅ Graceful shutdown
- ✅ Persistent identity
- ✅ Comprehensive tests

### Deferred to Phase 1

**Authentication:**
- Current: Basic token auth (placeholder)
- Phase 1: mTLS with control plane integration

**Deployment:**
- Current: Local development only
- Phase 1 (Module 7.1): Docker + cloud deployment (Fly.io/DigitalOcean)

**Monitoring:**
- Current: Structured logging
- Phase 1: Metrics + Prometheus integration

**Security:**
- Rate limiting per-IP
- Ban lists / abuse prevention
- Health check endpoints

## File Structure

```
relay-server/
├── src/
│   ├── main.rs      # CLI entry point, event loop
│   ├── config.rs    # TOML configuration
│   ├── relay.rs     # libp2p swarm setup
│   ├── events.rs    # Event handling
│   ├── auth.rs      # Token auth (placeholder)
│   └── errors.rs    # Error types
├── examples/
│   └── test_client.rs  # Integration test client
├── Cargo.toml
└── README.md
```

## Next Steps

After completing the relay server implementation:

1. **Module 2.1**: Network Swarm Setup (agent connects to relay)
2. **Module 2.2**: Job Protocol Handler (custom protocol over relay)
3. **Module 2.3**: Control Plane API (registration returns relay addresses)
4. **Module 7.1**: Cloud Deployment (Dockerfile, Fly.io/DigitalOcean)

## Resources

- [libp2p Circuit Relay v2 Spec](https://github.com/libp2p/specs/blob/master/relay/circuit-v2.md)
- [libp2p Rust Documentation](https://docs.rs/libp2p/latest/libp2p/)
- [libp2p Relay Rust Docs](https://docs.rs/libp2p/latest/libp2p/relay/index.html)
- [Mesh Implementation Guide](../IMPLEMENTATION.md)

## License

MIT
