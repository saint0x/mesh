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

# Configuration
DEVICE_NAME="${DEVICE_NAME:-Device2-Member}"
POOL_INFO_FILE="${POOL_INFO_FILE:-$SCRIPT_DIR/.pool-info}"

# Check if pool info exists
if [ ! -f "$POOL_INFO_FILE" ]; then
    echo "❌ Pool info file not found: $POOL_INFO_FILE"
    echo ""
    echo "Please provide pool details:"
    echo ""
    read -p "Pool ID: " POOL_ID
    read -p "Pool Root Pubkey: " POOL_ROOT_PUBKEY
    read -p "Pool Name [test-pool]: " POOL_NAME
    read -p "Network ID [test-network]: " NETWORK_ID
    read -p "Relay Port [4001]: " RELAY_PORT
    read -p "Control Plane Port [8080]: " CONTROL_PLANE_PORT

    POOL_NAME="${POOL_NAME:-test-pool}"
    NETWORK_ID="${NETWORK_ID:-test-network}"
    RELAY_PORT="${RELAY_PORT:-4001}"
    CONTROL_PLANE_PORT="${CONTROL_PLANE_PORT:-8080}"
else
    echo "✓ Found pool info file: $POOL_INFO_FILE"
    source "$POOL_INFO_FILE"
    echo ""
    echo "  Pool ID:         $POOL_ID"
    echo "  Pool Name:       $POOL_NAME"
    echo "  Network ID:      $NETWORK_ID"
    echo ""
fi

# Build project
echo "📦 Building Mesh binaries (release mode)..."
cargo build --release
echo "✓ Build complete"
echo ""

# Kill any existing agent processes
echo "🧹 Cleaning up old agent processes..."
pkill -f "agent.*start" 2>/dev/null || true
sleep 2
echo "✓ Cleanup complete"
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

# Check if already joined pool
POOL_DIR="$HOME/.meshnet/pools/$POOL_ID"
if [ ! -d "$POOL_DIR" ]; then
    echo "🎯 Joining pool '$POOL_NAME'..."
    echo ""
    echo "⏳ Waiting for certificate from pool admin (60s timeout)..."
    echo "   Make sure Device 1 (admin) is running on the same LAN!"
    echo ""

    ./target/release/agent pool-join \
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
./target/release/agent start \
    --relay "/ip4/127.0.0.1/tcp/$RELAY_PORT" \
    --control-plane "http://localhost:$CONTROL_PLANE_PORT" \
    --log-level info

cleanup
