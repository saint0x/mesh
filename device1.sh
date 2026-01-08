#!/bin/bash
# Device 1 (Admin) - Pool Creator
# This script sets up the admin device for a LAN pool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MESH - Device 1 (Pool Admin)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Configuration
NETWORK_ID="${NETWORK_ID:-test-network}"
DEVICE_NAME="${DEVICE_NAME:-Device1-Admin}"
POOL_NAME="${POOL_NAME:-test-pool}"
RELAY_PORT="${RELAY_PORT:-4001}"
CONTROL_PLANE_PORT="${CONTROL_PLANE_PORT:-8080}"

# Build project
echo "📦 Building Mesh binaries (release mode)..."
cargo build --release
echo "✓ Build complete"
echo ""

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "relay-server" 2>/dev/null || true
pkill -f "control-plane" 2>/dev/null || true
pkill -f "agent.*start" 2>/dev/null || true
sleep 2
echo "✓ Cleanup complete"
echo ""

# Start relay server in background
echo "🌐 Starting relay server (port $RELAY_PORT)..."
RUST_LOG=info ./target/release/relay-server --port "$RELAY_PORT" > ~/.meshnet/logs/relay.log 2>&1 &
RELAY_PID=$!
echo "✓ Relay server started (PID: $RELAY_PID)"
sleep 2
echo ""

# Start control plane in background
echo "⚙️  Starting control plane (port $CONTROL_PLANE_PORT)..."
./target/release/control-plane --port "$CONTROL_PLANE_PORT" > ~/.meshnet/logs/control-plane.log 2>&1 &
CONTROL_PLANE_PID=$!
echo "✓ Control plane started (PID: $CONTROL_PLANE_PID)"
sleep 3
echo ""

# Initialize device if not already done
if [ ! -f ~/.meshnet/device.toml ]; then
    echo "🔧 Initializing device..."
    ./target/release/agent init \
        --network-id "$NETWORK_ID" \
        --name "$DEVICE_NAME" \
        --control-plane "http://localhost:$CONTROL_PLANE_PORT"
    echo ""
else
    echo "✓ Device already initialized (found ~/.meshnet/device.toml)"
    echo ""
fi

# Create pool if not already done
POOL_INFO_FILE="$SCRIPT_DIR/.pool-info"
if [ ! -f "$POOL_INFO_FILE" ]; then
    echo "🎯 Creating pool '$POOL_NAME'..."

    # Capture pool creation output
    POOL_OUTPUT=$(./target/release/agent pool-create --name "$POOL_NAME" 2>&1)
    echo "$POOL_OUTPUT"
    echo ""

    # Extract pool ID and root pubkey from output
    POOL_ID=$(echo "$POOL_OUTPUT" | grep "Pool ID:" | awk '{print $NF}' | tr -d '\n')
    POOL_ROOT_PUBKEY=$(echo "$POOL_OUTPUT" | grep -A1 "Pool Root Public Key" | tail -1 | tr -d ' \n')

    # Save to file for device2.sh to read
    cat > "$POOL_INFO_FILE" <<EOF
POOL_ID=$POOL_ID
POOL_ROOT_PUBKEY=$POOL_ROOT_PUBKEY
POOL_NAME=$POOL_NAME
NETWORK_ID=$NETWORK_ID
RELAY_PORT=$RELAY_PORT
CONTROL_PLANE_PORT=$CONTROL_PLANE_PORT
EOF

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📋 SHARE THESE VALUES WITH DEVICE 2:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Pool ID:            $POOL_ID"
    echo "  Pool Root Pubkey:   $POOL_ROOT_PUBKEY"
    echo ""
    echo "  These values saved to: $POOL_INFO_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
else
    echo "✓ Pool already created (found $POOL_INFO_FILE)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📋 Pool Info (share with Device 2):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat "$POOL_INFO_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# Create log directory
mkdir -p ~/.meshnet/logs

# Start agent daemon
echo "🚀 Starting agent daemon..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agent daemon starting (logs: ~/.meshnet/logs/agent.log)"
echo "  Press Ctrl+C to stop all services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $RELAY_PID 2>/dev/null || true
    kill $CONTROL_PLANE_PID 2>/dev/null || true
    echo "✓ Services stopped"
    exit 0
}

trap cleanup INT TERM

# Start agent in foreground
./target/release/agent start \
    --relay "/ip4/127.0.0.1/tcp/$RELAY_PORT" \
    --control-plane "http://localhost:$CONTROL_PLANE_PORT" \
    --log-level info

cleanup
