# Mesh LAN Pool - Quick Start Guide

This guide shows you how to quickly set up a 2-device LAN pool for testing distributed inference.

## Prerequisites

- Two devices on the same WiFi/LAN network
- Rust installed on both devices
- This repository cloned on both devices

## Installation

On both devices, run:

```bash
./install.sh
```

This will:
- Build all binaries in release mode
- Install `mesh`, `mesh-relay`, and `mesh-control-plane` to `~/.local/bin`
- Add `~/.local/bin` to your PATH (in `.zshrc` or `.bashrc`)

After installation, restart your shell or run:
```bash
source ~/.zshrc  # or ~/.bashrc
```

## Device 1 (Admin) Setup

On the first device (pool creator/admin):

```bash
./device1.sh
```

This script will:
1. Build the project
2. Start relay server in background (port 4001)
3. Start control plane in background (port 8080)
4. Initialize the device
5. Create a new pool
6. Display pool credentials to share with Device 2
7. Start the agent daemon

**Important:** Copy the Pool ID and Pool Root Public Key that are displayed. You'll need them for Device 2.

The script creates a `.pool-info` file containing the pool credentials.

## Device 2 (Member) Setup

On the second device (pool member):

### Option 1: Copy .pool-info file

If both devices can access a shared location:

1. Copy `.pool-info` from Device 1 to Device 2's meshnet directory
2. Run:
```bash
./device2.sh
```

### Option 2: Manual entry

If you can't copy the file:

1. Run:
```bash
./device2.sh
```
2. The script will prompt you for:
   - Pool ID
   - Pool Root Public Key
   - Pool Name (default: test-pool)
   - Network ID (default: test-network)

This script will:
1. Build the project
2. Initialize the device
3. Request certificate from Device 1 via LAN beacon
4. Wait up to 60 seconds for admin to sign the certificate
5. Start the agent daemon

**Note:** Make sure Device 1 is running before starting Device 2, as the certificate signing happens over LAN.

## Verification

After both devices are running, verify the setup:

### Check discovered peers

On either device:
```bash
mesh pool-peers --pool-id <POOL_ID>
```

Should show:
- The other peer's node ID
- LAN address
- Status: ONLINE
- Last seen timestamp

### Check ring topology

```bash
mesh pool-list
```

Should show your pool with both members.

### Check ring convergence

```bash
cat ~/.meshnet/ring_topology_<pool_id>.json
```

Should show:
- Both members with node IDs
- Ring positions (0 and 1)
- Shard column ranges (8192 / 2 = 4096 each)

## Expected Timeline

```
T+0s:  Device 1 starts (relay + control plane + agent)
T+5s:  Device 1 broadcasts LAN beacons
T+10s: Device 2 starts
T+10s: Device 2 requests certificate via LAN beacon
T+15s: Device 1 auto-signs certificate, responds via beacon
T+15s: Device 2 receives cert, joins pool
T+20s: Both devices discover each other via beacons
T+30s: Ring gossip converges
T+40s: Ring topology saved, ready for inference
```

## Stopping Services

Press `Ctrl+C` in the terminal running the device script.

For Device 1, this will stop:
- Agent daemon
- Control plane
- Relay server

For Device 2, this will stop:
- Agent daemon

## Environment Variables

You can customize the setup with environment variables:

### Device 1 (device1.sh)
```bash
NETWORK_ID=my-network \
DEVICE_NAME="MacBook-Admin" \
POOL_NAME="my-pool" \
RELAY_PORT=4002 \
CONTROL_PLANE_PORT=8081 \
./device1.sh
```

### Device 2 (device2.sh)
```bash
DEVICE_NAME="MacBook-Member" \
./device2.sh
```

## Troubleshooting

### Port conflicts

If port 8080 or 4001 is already in use:

- Control plane will automatically find an available port (8080-8179)
- Relay server: use `RELAY_PORT=4002 ./device1.sh`

### Certificate timeout

If Device 2 times out waiting for certificate:

1. Verify both devices are on same LAN
2. Check firewall allows UDP multicast (port 42424)
3. Verify Device 1 is running
4. Try:
```bash
sudo tcpdump -i any udp port 42424
```

Should see beacon packets every 5 seconds.

### Peers not discovered

Check LAN connectivity:
```bash
# Verify same subnet
ip addr show  # Linux
ifconfig      # macOS

# Check if beacons are being sent
sudo tcpdump -i any udp port 42424
```

### Check logs

Logs are saved to `~/.meshnet/logs/`:
- `relay.log` - Relay server logs
- `control-plane.log` - Control plane logs
- `agent.log` - Agent daemon logs (if run in background)

View logs:
```bash
tail -f ~/.meshnet/logs/agent.log
tail -f ~/.meshnet/logs/control-plane.log
tail -f ~/.meshnet/logs/relay.log
```

## Manual Commands (Using mesh CLI)

If you prefer manual control:

### Device 1
```bash
# Start services
mesh-relay --port 4001 &
mesh-control-plane --port 8080 &

# Initialize and create pool
mesh init --network-id test --name Device1
mesh pool-create --name test-pool
# Copy Pool ID and Root Pubkey from output

# Start agent
mesh start
```

### Device 2
```bash
# Initialize
mesh init --network-id test --name Device2

# Join pool (replace with actual values)
mesh pool-join \
  --pool-id <POOL_ID> \
  --pool-root-pubkey <ROOT_PUBKEY> \
  --name test-pool

# Start agent
mesh start
```

## Next Steps

Once the ring has converged (both peers online, topology saved):

1. Submit inference job (from either device):
```bash
mesh inference --prompt "Hello world" --max-tokens 10
```

2. Check inference stats:
```bash
mesh inference-stats
```

3. Monitor ring status:
```bash
mesh ring-status
mesh pool-status
```

## Architecture Notes

- **P2P-first design**: Pool creation and membership are fully P2P (no HTTP required)
- **LAN beacons**: Discovery via UDP multicast (239.192.0.1:42424)
- **Certificate signing**: Admin signs member certs via LAN beacons
- **Ring gossip**: Topology convergence via P2P gossip protocol
- **Offline-capable**: Works without internet once pool is created

The control plane is used for legacy HTTP-based ring join (backup method) and job distribution. The P2P layer handles pool membership and ring formation.
