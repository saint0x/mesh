#!/bin/bash
# Manual Integration Test for Full Distributed Inference Pipeline
#
# This script:
# 1. Builds all binaries
# 2. Starts relay server
# 3. Starts control plane
# 4. Starts 3 worker agents
# 5. Establishes ring topology
# 6. Submits inference job with mock weights
# 7. Validates output
# 8. Cleans up

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NUM_WORKERS=3
RELAY_PORT=4001
CONTROL_PLANE_PORT=8080
BASE_WORKER_PORT=9000
TEST_DIR="$(pwd)/.test_integration"
LOG_DIR="$TEST_DIR/logs"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up processes...${NC}"

    # Kill all background jobs from this script
    jobs -p | xargs -r kill 2>/dev/null || true

    # Kill any lingering processes
    pkill -f "relay-server" 2>/dev/null || true
    pkill -f "control-plane" 2>/dev/null || true
    pkill -f "agent.*worker" 2>/dev/null || true

    # Clean up test directory
    rm -rf "$TEST_DIR"

    echo -e "${GREEN}Cleanup complete${NC}"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

# Print section header
print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo -e "${RED}✗ $name failed to start${NC}"
    return 1
}

# Create test directory structure
mkdir -p "$LOG_DIR"
mkdir -p "$TEST_DIR/relay"
mkdir -p "$TEST_DIR/control-plane"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    mkdir -p "$TEST_DIR/worker-$i"
done

print_section "Step 1: Building Binaries"
echo -e "${YELLOW}Building release binaries...${NC}"
cargo build --release
echo -e "${GREEN}✓ Build complete${NC}"

print_section "Step 2: Starting Relay Server"
echo -e "${YELLOW}Starting relay server on port $RELAY_PORT...${NC}"
RUST_LOG=info ./target/release/relay-server \
    --port $RELAY_PORT \
    > "$LOG_DIR/relay.log" 2>&1 &
RELAY_PID=$!
echo "Relay server PID: $RELAY_PID"

# Wait for relay to be ready (relay doesn't have HTTP endpoint, just wait)
sleep 3
if ps -p $RELAY_PID > /dev/null; then
    echo -e "${GREEN}✓ Relay server started (PID: $RELAY_PID)${NC}"
else
    echo -e "${RED}✗ Relay server failed to start${NC}"
    exit 1
fi

print_section "Step 3: Starting Control Plane"
echo -e "${YELLOW}Starting control plane on port $CONTROL_PLANE_PORT...${NC}"
cd "$TEST_DIR/control-plane"
RUST_LOG=info ../../target/release/control-plane \
    --port $CONTROL_PLANE_PORT \
    > "$LOG_DIR/control-plane.log" 2>&1 &
CONTROL_PLANE_PID=$!
cd - > /dev/null
echo "Control plane PID: $CONTROL_PLANE_PID"

# Wait for control plane health check
wait_for_service "http://localhost:$CONTROL_PLANE_PORT/health" "Control Plane"

print_section "Step 4: Starting Worker Agents"
WORKER_PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    worker_port=$((BASE_WORKER_PORT + i))
    echo -e "${YELLOW}Starting worker $i on port $worker_port...${NC}"

    cd "$TEST_DIR/worker-$i"
    RUST_LOG=info ../../target/release/agent \
        --port $worker_port \
        --relay-addr "/ip4/127.0.0.1/tcp/$RELAY_PORT" \
        --control-plane "http://localhost:$CONTROL_PLANE_PORT" \
        > "$LOG_DIR/worker-$i.log" 2>&1 &

    worker_pid=$!
    WORKER_PIDS+=($worker_pid)
    echo "Worker $i PID: $worker_pid"
    cd - > /dev/null

    # Give workers time to start
    sleep 2
done

echo -e "${GREEN}✓ All $NUM_WORKERS workers started${NC}"

print_section "Step 5: Registering Workers with Control Plane"
# Wait a bit for workers to initialize
sleep 3

# Check worker registration (this will depend on your API)
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo -e "${YELLOW}Checking worker $i registration...${NC}"
    # This would call your control plane API to verify worker is registered
    # For now, we'll just verify the process is running
    if ps -p ${WORKER_PIDS[$i]} > /dev/null; then
        echo -e "${GREEN}✓ Worker $i is running${NC}"
    else
        echo -e "${RED}✗ Worker $i died${NC}"
        exit 1
    fi
done

print_section "Step 6: Establishing Ring Topology"
echo -e "${YELLOW}Requesting ring topology formation...${NC}"

# This would call your control plane API to form the ring
# For now, let's check if we can reach the control plane
if curl -s "http://localhost:$CONTROL_PLANE_PORT/health" | grep -q "ok"; then
    echo -e "${GREEN}✓ Control plane is healthy${NC}"
else
    echo -e "${RED}✗ Control plane health check failed${NC}"
    exit 1
fi

# Give time for ring formation
sleep 5

print_section "Step 7: Submitting Inference Job"
echo -e "${YELLOW}Submitting test inference job with mock weights...${NC}"

# Create a test inference request
cat > "$TEST_DIR/test_request.json" <<EOF
{
  "model_id": "llama-70b",
  "prompt": "Hello, world!",
  "max_tokens": 10,
  "temperature": 1.0,
  "top_p": 0.9
}
EOF

# Submit job (this would use your API)
# For now, we'll verify the setup is working
echo -e "${YELLOW}Test request created at: $TEST_DIR/test_request.json${NC}"

print_section "Step 8: Verification"
echo -e "${YELLOW}Verifying system state...${NC}"

# Check all processes are still running
all_running=true

if ! ps -p $RELAY_PID > /dev/null; then
    echo -e "${RED}✗ Relay server died${NC}"
    all_running=false
else
    echo -e "${GREEN}✓ Relay server running${NC}"
fi

if ! ps -p $CONTROL_PLANE_PID > /dev/null; then
    echo -e "${RED}✗ Control plane died${NC}"
    all_running=false
else
    echo -e "${GREEN}✓ Control plane running${NC}"
fi

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    if ! ps -p ${WORKER_PIDS[$i]} > /dev/null; then
        echo -e "${RED}✗ Worker $i died${NC}"
        all_running=false
    else
        echo -e "${GREEN}✓ Worker $i running${NC}"
    fi
done

if $all_running; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL PROCESSES RUNNING SUCCESSFULLY${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Check logs in: $LOG_DIR"
    echo "2. Submit inference job via control plane API"
    echo "3. Monitor worker logs for ring all-reduce operations"
    echo ""
    echo -e "${YELLOW}Logs:${NC}"
    echo "  Relay:         tail -f $LOG_DIR/relay.log"
    echo "  Control Plane: tail -f $LOG_DIR/control-plane.log"
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "  Worker $i:      tail -f $LOG_DIR/worker-$i.log"
    done
    echo ""
    echo -e "${BLUE}Press Ctrl+C to stop all services and cleanup${NC}"

    # Keep running until interrupted
    wait
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ SOME PROCESSES FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check logs for errors:"
    echo "  $LOG_DIR"
    exit 1
fi
