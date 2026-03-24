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
POOL_ID="602e587a5e549df0335ddddde1d7a73e163116015dfd047d071640f36e74e00a"
POOL_ROOT_PUBKEY="08b75d02170a95193f595961eec988e11e103fbea60ba6c607570550dc859150"

# Configuration
DEVICE_NAME="${DEVICE_NAME:-Device2-Member}"
POOL_NAME="${POOL_NAME:-test-pool}"
NETWORK_ID="${NETWORK_ID:-test-network}"
RELAY_PORT="${RELAY_PORT:-4001}"
CONTROL_PLANE_PORT="${CONTROL_PLANE_PORT:-8080}"
CONTROL_PLANE_HOST="${CONTROL_PLANE_HOST:-localhost}"

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
        --control-plane "http://$CONTROL_PLANE_HOST:$CONTROL_PLANE_PORT"
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
    --log-level info

cleanup
