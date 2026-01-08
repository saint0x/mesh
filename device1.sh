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

# Detect binary path (macOS uses aarch64-apple-darwin)
if [ -f "./target/aarch64-apple-darwin/release/relay-server" ]; then
    BINARY_PATH="./target/aarch64-apple-darwin/release"
else
    BINARY_PATH="./target/release"
fi

# Start relay server in background
echo "🌐 Starting relay server (port $RELAY_PORT)..."
RUST_LOG=info "$BINARY_PATH/relay-server" --port "$RELAY_PORT" > ~/.meshnet/logs/relay.log 2>&1 &
RELAY_PID=$!
sleep 2

# Verify relay server started (it doesn't auto-change ports like control plane)
if ! ps -p $RELAY_PID > /dev/null 2>&1; then
    echo "❌ Relay server failed to start - check ~/.meshnet/logs/relay.log"
    exit 1
fi
echo "✓ Relay server started (PID: $RELAY_PID)"
echo ""

# Start control plane in background
echo "⚙️  Starting control plane (port $CONTROL_PLANE_PORT)..."
cd control-plane && "$SCRIPT_DIR/$BINARY_PATH/control-plane" --port "$CONTROL_PLANE_PORT" > ~/.meshnet/logs/control-plane.log 2>&1 &
CONTROL_PLANE_PID=$!
cd "$SCRIPT_DIR"

# Wait for control plane to start (give migrations time to complete)
sleep 2

# Check for migration errors early
if grep -q "Error:" ~/.meshnet/logs/control-plane.log 2>/dev/null; then
    echo "⚠️  Migration error detected - cleaning database and restarting..."
    kill $CONTROL_PLANE_PID 2>/dev/null || true
    rm -f ~/.meshnet/control-plane.db*
    sleep 1
    cd control-plane && "$SCRIPT_DIR/$BINARY_PATH/control-plane" --port "$CONTROL_PLANE_PORT" > ~/.meshnet/logs/control-plane.log 2>&1 &
    CONTROL_PLANE_PID=$!
    cd "$SCRIPT_DIR"
fi

# Wait for startup to complete
sleep 3

# Detect actual port control plane is using (it may have changed if port was in use)
ACTUAL_PORT=$(grep "Using port" ~/.meshnet/logs/control-plane.log 2>/dev/null | grep -o "[0-9]\+" | tail -1)
if [ -z "$ACTUAL_PORT" ]; then
    # Try parsing from "address=0.0.0.0:PORT" format
    ACTUAL_PORT=$(grep "listening on.*address" ~/.meshnet/logs/control-plane.log 2>/dev/null | grep -o ":[0-9]\+" | tr -d ':' | tail -1)
fi
if [ -z "$ACTUAL_PORT" ]; then
    # Default to requested port if we can't detect
    ACTUAL_PORT=$CONTROL_PLANE_PORT
fi

if [ "$ACTUAL_PORT" != "$CONTROL_PLANE_PORT" ]; then
    echo "⚠️  Control plane using port $ACTUAL_PORT (requested $CONTROL_PLANE_PORT was in use)"
    CONTROL_PLANE_PORT=$ACTUAL_PORT
else
    echo "✓ Control plane started on port $CONTROL_PLANE_PORT (PID: $CONTROL_PLANE_PID)"
fi

# Verify control plane started successfully
if ! ps -p $CONTROL_PLANE_PID > /dev/null 2>&1; then
    echo "❌ Control plane failed to start - check ~/.meshnet/logs/control-plane.log"
    tail -10 ~/.meshnet/logs/control-plane.log
    kill $RELAY_PID 2>/dev/null || true
    exit 1
fi

# Final check for any errors in log
if grep -q "Error:" ~/.meshnet/logs/control-plane.log; then
    echo "❌ Control plane has errors - check ~/.meshnet/logs/control-plane.log"
    tail -10 ~/.meshnet/logs/control-plane.log
    kill $RELAY_PID 2>/dev/null || true
    kill $CONTROL_PLANE_PID 2>/dev/null || true
    exit 1
fi
echo ""

# Initialize device if not already done
if [ ! -f ~/.meshnet/device.toml ]; then
    echo "🔧 Initializing device..."
    "$BINARY_PATH/agent" init \
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
    POOL_OUTPUT=$("$BINARY_PATH/agent" pool-create --name "$POOL_NAME" 2>&1)
    echo "$POOL_OUTPUT"
    echo ""

    # Extract pool ID and root pubkey from output
    POOL_ID=$(echo "$POOL_OUTPUT" | grep "Pool ID:" | awk '{print $NF}' | tr -d '\n')
    POOL_ROOT_PUBKEY=$(echo "$POOL_OUTPUT" | grep -A1 "Pool Root Public Key" | tail -1 | tr -d ' \n')

    # Save to file for device2.sh to read (use actual detected port)
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
"$BINARY_PATH/agent" start \
    --relay "/ip4/127.0.0.1/tcp/$RELAY_PORT" \
    --control-plane "http://localhost:$CONTROL_PLANE_PORT" \
    --log-level info

cleanup
