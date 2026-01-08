# LAN Beacon Discovery Testing Guide

## Phase 1 Implementation Complete!

We've implemented offline-first LAN discovery using UDP multicast beacons. Here's how to test it:

## Prerequisites

- 2+ devices on the same WiFi/Ethernet network
- Meshnet installed on each device (run `./install.sh` from repo root)

## Testing Steps

### 1. Initialize Devices

On each device, initialize the agent (if not already done):

```bash
mesh init --network-id test-network --name "Device 1"
```

### 2. Create a Pool (Device 1 only)

On the first device, create a new pool:

```bash
mesh pool-create --name "LAN Test Pool"
```

This will output:
- Pool ID (hex)
- Pool Root Public Key (hex)

**Copy these values** - you'll need them for other devices.

### 3. Join the Pool (Other Devices)

On each additional device, join the pool using the values from Device 1:

```bash
mesh pool-join \
  --pool-id <POOL_ID_FROM_DEVICE_1> \
  --pool-root-pubkey <POOL_ROOT_PUBKEY_FROM_DEVICE_1> \
  --name "LAN Test Pool"
```

### 4. List Pools

On any device, verify the pool configuration:

```bash
mesh pool-list
```

You should see:
- Pool name
- Your node ID
- Role (admin or member)
- Expiry information

### 5. Start the Agent Daemon

On **each device**, start the agent:

```bash
mesh start
```

You should see output like:
```
🚀 Starting Mesh AI agent daemon...

📋 Device: Device 1 (...)
   Network: test-network
   Tier: ...

🔍 Starting LAN beacon discovery...
   Pools: 1
   Node ID: abc123def456...
   ✓ Beacon listener started
   ✓ Beacon broadcaster started
```

### 6. Monitor Discovery

Watch the logs on each device. Within 5-10 seconds, you should see:

```
[INFO] LAN peer discovered: <node_id> at <lan_addr> (pool: <pool_id>)
```

This means devices are discovering each other via LAN beacons!

### 7. Check Discovered Peers

On any device, view discovered peers:

```bash
mesh pool-peers --pool-id <POOL_ID>
```

You should see:
```
Pool: LAN Test Pool
  Pool ID: abc123...

Discovered peers: 2

Peer: def456...
  LAN Address:     192.168.1.100:4001
  Discovery:       LAN
  Status:          ONLINE
  Last Seen:       3s ago

Peer: ghi789...
  LAN Address:     192.168.1.101:4001
  Discovery:       LAN
  Status:          ONLINE
  Last Seen:       5s ago
```

## Success Criteria

✅ Multiple devices on same LAN can create/join pools
✅ Beacon broadcaster sends beacons every 5 seconds
✅ Beacon listener receives beacons from other devices
✅ Peers appear in pool-peers output
✅ Discovery works without internet (offline-first)

## Troubleshooting

**No peers discovered:**
- Ensure all devices are on the **same WiFi/Ethernet network**
- Check firewall settings (UDP port 42424)
- Verify all devices have the same pool ID
- Look for errors in agent logs

**Peers show as STALE/OFFLINE:**
- Check if beacons are being sent (should be every 5s)
- Verify network connectivity between devices
- Restart the agent daemon

**Installation errors:**
- Ensure you've run `./install.sh` from the repo root
- Check that `mesh` is in your PATH: `which mesh`
- Try restarting your shell or running: `source ~/.zshrc`

## What's Next (Phase 2)

Phase 1 gives us LAN discovery - devices can find each other automatically on the same network.

Phase 2 will add:
- QUIC connections (actual connectivity, not just discovery)
- Mutual authentication using pool membership certificates
- HTTP rendezvous for cross-location discovery
- Integration with distributed inference ring

## Architecture Notes

**LAN vs HTTP:**
- LAN beacons work offline, no internet required
- HTTP rendezvous will connect nodes across different locations
- LAN is prioritized inside a pool (the "universe")
- HTTP is for coordination and cross-location bridging

**Security:**
- Phase 1: Beacons are signed but not verified (trust LAN)
- Phase 2: Full signature verification + QUIC mutual auth
- Pool membership certs expire in 7 days (self-signed in Phase 1)

## Files Created

- `agent/src/pki/` - PKI types (pools, certs, node IDs)
- `agent/src/discovery/` - Beacon broadcaster and listener
- `~/.meshnet/pools/{pool_id}/` - Pool storage per device
