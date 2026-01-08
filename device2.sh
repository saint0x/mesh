#!/bin/bash
# Device 2 (Member) - Pool Joiner
# This script sets up a member device for a LAN pool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MESH - Device 2 (Pool Member)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Hardcoded pool credentials (from Device 1 admin)
POOL_ID="18df84ceb61ea385ca3692d2c56de53dfe03bf31ff6414f1fc9cc2c7ec08ef3a"
POOL_ROOT_PUBKEY="d71df381393fd1da8c3ae1e613d44a63e04239834ae8ac2b347c982b03f79cab"

# Configuration
DEVICE_NAME="${DEVICE_NAME:-Device2-Member}"
POOL_NAME="${POOL_NAME:-test-pool}"
NETWORK_ID="${NETWORK_ID:-test-network}"
RELAY_PORT="${RELAY_PORT:-4001}"
CONTROL_PLANE_PORT="${CONTROL_PLANE_PORT:-8080}"

echo "  Pool ID:         $POOL_ID"
echo "  Pool Root Pubkey: ${POOL_ROOT_PUBKEY:0:16}..."
echo "  Pool Name:       $POOL_NAME"
echo ""

# Build project
echo "📦 Building Mesh binaries (release mode)..."
cargo build --release
echo "✓ Build complete"
echo ""

# Detect binary path (macOS uses aarch64-apple-darwin)
if [ -f "./target/aarch64-apple-darwin/release/agent" ]; then
    BINARY_PATH="./target/aarch64-apple-darwin/release"
else
    BINARY_PATH="./target/release"
fi

# Kill any existing agent processes
echo "🧹 Cleaning up old agent processes..."
pkill -f "agent.*start" 2>/dev/null || true
sleep 2
echo "✓ Cleanup complete"
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

# Check if already joined pool
POOL_DIR="$HOME/.meshnet/pools/$POOL_ID"
if [ ! -d "$POOL_DIR" ]; then
    echo "🎯 Joining pool '$POOL_NAME'..."
    echo ""
    echo "⏳ Waiting for certificate from pool admin (60s timeout)..."
    echo "   Make sure Device 1 (admin) is running on the same LAN!"
    echo ""

    "$BINARY_PATH/agent" pool-join \
        --pool-id "$POOL_ID" \
        --pool-root-pubkey "$POOL_ROOT_PUBKEY" \
        --name "$POOL_NAME"

    echo ""
    echo "✓ Successfully joined pool!"
    echo ""
else
    echo "✓ Already member of pool (found $POOL_DIR)"
    echo ""
fi

# Detect admin's relay and control plane addresses from saved info
POOL_DIR="$HOME/.meshnet/pools/$POOL_ID"
ADMIN_RELAY_INFO="$POOL_DIR/admin_relay.toml"

if [ -f "$ADMIN_RELAY_INFO" ]; then
    ADMIN_IP=$(grep "ip_address" "$ADMIN_RELAY_INFO" | cut -d'"' -f2)
    if [ -n "$ADMIN_IP" ]; then
        RELAY_ADDR="/ip4/$ADMIN_IP/tcp/$RELAY_PORT"
        CONTROL_PLANE_ADDR="http://$ADMIN_IP:$CONTROL_PLANE_PORT"
        echo "✓ Using admin's relay at $ADMIN_IP:$RELAY_PORT"
        echo "✓ Using admin's control plane at $ADMIN_IP:$CONTROL_PLANE_PORT"
    else
        RELAY_ADDR="/ip4/127.0.0.1/tcp/$RELAY_PORT"
        CONTROL_PLANE_ADDR="http://localhost:$CONTROL_PLANE_PORT"
        echo "⚠️  Could not parse admin IP, using localhost"
    fi
else
    # Fallback to localhost (single-machine setup or old pool)
    RELAY_ADDR="/ip4/127.0.0.1/tcp/$RELAY_PORT"
    CONTROL_PLANE_ADDR="http://localhost:$CONTROL_PLANE_PORT"
    echo "ℹ️  No admin relay info found, using localhost"
fi
echo ""

# Create log directory
mkdir -p ~/.meshnet/logs

# Start agent daemon
echo "🚀 Starting agent daemon..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agent daemon starting (logs: ~/.meshnet/logs/agent.log)"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down agent..."
    echo "✓ Agent stopped"
    exit 0
}

trap cleanup INT TERM

# Start agent in foreground
"$BINARY_PATH/agent" start \
    --relay "$RELAY_ADDR" \
    --control-plane "$CONTROL_PLANE_ADDR" \
    --log-level info

cleanup
