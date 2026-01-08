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
