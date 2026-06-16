#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mesh-real-cluster.XXXXXX")"
KEEP_TEST_ROOT="${MESHNET_KEEP_TEST_ROOT:-0}"
CONTROL_PORT="${CONTROL_PORT:-43180}"
MODEL_ID="${MESHNET_REAL_CLUSTER_MODEL_ID:-smollm2-135m-instruct}"
ORIGINAL_HOME="${HOME:-}"
MODEL_STORE="${MESHNET_MODEL_STORE:-${ORIGINAL_HOME}/.meshnet/models}"
TEST_RUST_LOG="${MESHNET_TEST_RUST_LOG:-info}"
PREFERRED_PROVIDER="${MESHNET_REAL_CLUSTER_PREFERRED_PROVIDER:-}"
NETWORK_ID="real-cluster-e2e"
CONTROL_HOME="$TEST_ROOT/control-plane"
WORKER1_HOME="$TEST_ROOT/worker1"
WORKER2_HOME="$TEST_ROOT/worker2"
CONTROL_LOG="$TEST_ROOT/control-plane.log"
WORKER1_LOG="$TEST_ROOT/worker1.log"
WORKER2_LOG="$TEST_ROOT/worker2.log"
TOPOLOGY_LOG="$TEST_ROOT/topology-before-job.json"
JOB1_LOG="$TEST_ROOT/job-single.log"
JOB2_LOG="$TEST_ROOT/job-concurrent-1.log"
JOB3_LOG="$TEST_ROOT/job-concurrent-2.log"
JOB4_LOG="$TEST_ROOT/job-concurrent-3.log"

cleanup() {
    for pid_var in JOB4_PID JOB3_PID JOB2_PID WORKER2_PID WORKER1_PID CONTROL_PID; do
        local pid="${!pid_var:-}"
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ "$KEEP_TEST_ROOT" == "1" ]]; then
        echo "preserved test root: $TEST_ROOT" >&2
    else
        rm -rf "$TEST_ROOT"
    fi
}
trap cleanup EXIT INT TERM

run_with_timeout() {
    local duration="$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$duration" "$@"
        return
    fi
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$duration" "$@"
        return
    fi
    python3 - "$duration" "$@" <<'PY'
import subprocess
import sys

duration = sys.argv[1]
cmd = sys.argv[2:]
subprocess.run(cmd, timeout=float(duration.rstrip("s")), check=True)
PY
}

wait_for_http() {
    local url="$1"
    local attempts="${2:-80}"
    for _ in $(seq 1 "$attempts"); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

wait_for_topology() {
    local attempts="${1:-120}"
    local url="http://127.0.0.1:${CONTROL_PORT}/api/ring/topology?network_id=${NETWORK_ID}"
    for _ in $(seq 1 "$attempts"); do
        if curl -fsS "$url" | python3 -c '
import json
import sys

data = json.load(sys.stdin)
workers = data.get("workers", [])
if len(workers) != 2:
    raise SystemExit(1)
if not data.get("ring_stable", False):
    raise SystemExit(1)
if any(worker.get("status") != "online" for worker in workers):
    raise SystemExit(1)
if any(not any(addr.startswith("dataplane://") for addr in worker.get("listen_addrs", [])) for worker in workers):
    raise SystemExit(1)
' 
        then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_local_tensor_endpoint() {
    local home_dir="$1"
    local attempts="${2:-60}"
    local path="$home_dir/.meshnet/listen_addrs.json"
    for _ in $(seq 1 "$attempts"); do
        if [[ -f "$path" ]] && grep -q 'dataplane://' "$path"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

run_agent_cli() {
    local home_dir="$1"
    shift
    (
        cd "$home_dir"
        env \
            HOME="$home_dir" \
            MESHNET_HOME="$home_dir" \
            MESHNET_MODEL_STORE="$MODEL_STORE" \
            CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}" \
            RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}" \
            "$AGENT_BIN" "$@"
    )
}

require_preflight_runtime_readiness() {
    local home_dir="$1"
    (
        cd "$home_dir"
        run_with_timeout 240s env \
            HOME="$home_dir" \
            MESHNET_HOME="$home_dir" \
            MESHNET_MODEL_STORE="$MODEL_STORE" \
            CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}" \
            RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}" \
            "$AGENT_BIN" doctor --stage preflight
    )
}

require_production_readiness() {
    local home_dir="$1"
    (
        cd "$home_dir"
        run_with_timeout 240s env \
            HOME="$home_dir" \
            MESHNET_HOME="$home_dir" \
            MESHNET_MODEL_STORE="$MODEL_STORE" \
            CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}" \
            RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}" \
            "$AGENT_BIN" doctor --stage production
    )
}

device_id_from_home() {
    local home_dir="$1"
    awk -F'"' '/^device_id = / { print $2; exit }' "$home_dir/.meshnet/device.toml"
}

if [[ ! -d "$MODEL_STORE/$MODEL_ID" ]]; then
    echo "required real model artifacts not found at $MODEL_STORE/$MODEL_ID" >&2
    exit 1
fi

if [[ "$PREFERRED_PROVIDER" == "cpu" ]]; then
    echo "cpu is not a production-serving provider; refusing to run real cluster production validation" >&2
    exit 1
fi

MESHNET_ENABLE_REAL_ARTIFACT_TEST=1 \
MESHNET_REAL_ARTIFACT_MODEL_ID="$MODEL_ID" \
MESHNET_REAL_ARTIFACT_PROVIDER="${PREFERRED_PROVIDER:-}" \
bash "$ROOT_DIR/scripts/test_real_artifact_loading.sh"

mkdir -p "$CONTROL_HOME" "$WORKER1_HOME" "$WORKER2_HOME"

cd "$ROOT_DIR"
cargo build -p agent --bin agent -p control-plane --bin control-plane >/dev/null

AGENT_BIN="$(find "$ROOT_DIR/target" -type f \( -path '*/debug/agent' -o -path '*/debug/agent.exe' \) | head -n 1)"
CONTROL_BIN="$(find "$ROOT_DIR/target" -type f \( -path '*/debug/control-plane' -o -path '*/debug/control-plane.exe' \) | head -n 1)"
if [[ -z "$AGENT_BIN" || -z "$CONTROL_BIN" ]]; then
    echo "failed to locate built agent/control-plane binaries" >&2
    exit 1
fi

(
    cd "$CONTROL_HOME"
    env \
        HOME="$CONTROL_HOME" \
        MESHNET_MODEL_STORE="$MODEL_STORE" \
        RUST_LOG="$TEST_RUST_LOG" \
        CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}" \
        RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}" \
        "$CONTROL_BIN" --port "$CONTROL_PORT" >"$CONTROL_LOG" 2>&1
) &
CONTROL_PID=$!

if ! wait_for_http "http://127.0.0.1:${CONTROL_PORT}/health" 120; then
    echo "control-plane failed to become healthy" >&2
    cat "$CONTROL_LOG" >&2 || true
    exit 1
fi

CONTROL_URL="http://127.0.0.1:${CONTROL_PORT}"

curl -fsS -X POST "${CONTROL_URL}/api/networks" \
    -H "Content-Type: application/json" \
    -d "{
        \"network_id\": \"${NETWORK_ID}\",
        \"name\": \"Real Cluster E2E\",
        \"owner_user_id\": \"local-e2e\",
        \"connectivity\": {
            \"preferred_path\": \"direct\",
            \"attachments\": []
        }
    }" >/dev/null

WORKER1_INIT_ARGS=(device init --network-id "$NETWORK_ID" --name "Worker 1" --control-plane "$CONTROL_URL")
WORKER2_INIT_ARGS=(device init --network-id "$NETWORK_ID" --name "Worker 2" --control-plane "$CONTROL_URL")
if [[ -n "$PREFERRED_PROVIDER" ]]; then
    WORKER1_INIT_ARGS+=(--preferred-provider "$PREFERRED_PROVIDER")
    WORKER2_INIT_ARGS+=(--preferred-provider "$PREFERRED_PROVIDER")
fi
run_agent_cli "$WORKER1_HOME" "${WORKER1_INIT_ARGS[@]}" >/dev/null
run_agent_cli "$WORKER2_HOME" "${WORKER2_INIT_ARGS[@]}" >/dev/null

WORKER1_DEVICE_ID="$(device_id_from_home "$WORKER1_HOME")"
WORKER2_DEVICE_ID="$(device_id_from_home "$WORKER2_HOME")"

for device_id in "$WORKER1_DEVICE_ID" "$WORKER2_DEVICE_ID"; do
    curl -fsS -X POST "${CONTROL_URL}/api/ledger/events" \
        -H "Content-Type: application/json" \
        -d "{
            \"network_id\": \"${NETWORK_ID}\",
            \"event_type\": \"credits_earned\",
            \"job_id\": null,
            \"device_id\": \"${device_id}\",
            \"credits_amount\": 10000.0,
            \"metadata\": {
                \"credit_model\": \"bootstrap_e2e_funds\",
                \"reason\": \"seed real cluster submitter credits\"
            }
        }" >/dev/null
done

run_agent_cli "$WORKER1_HOME" ring join --model-id "$MODEL_ID" --memory 1GB >/dev/null
run_agent_cli "$WORKER2_HOME" ring join --model-id "$MODEL_ID" --memory 1GB >/dev/null

(
    cd "$WORKER1_HOME"
    env \
        HOME="$WORKER1_HOME" \
        MESHNET_HOME="$WORKER1_HOME" \
        MESHNET_MODEL_STORE="$MODEL_STORE" \
        RUST_LOG="$TEST_RUST_LOG" \
        CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}" \
        RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}" \
        "$AGENT_BIN" device runtime --log-level info >"$WORKER1_LOG" 2>&1
) &
WORKER1_PID=$!

(
    cd "$WORKER2_HOME"
    env \
        HOME="$WORKER2_HOME" \
        MESHNET_HOME="$WORKER2_HOME" \
        MESHNET_MODEL_STORE="$MODEL_STORE" \
        RUST_LOG="$TEST_RUST_LOG" \
        CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}" \
        RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}" \
        "$AGENT_BIN" device runtime --log-level info >"$WORKER2_LOG" 2>&1
) &
WORKER2_PID=$!

if ! wait_for_topology 180; then
    echo "workers failed to become online and ring-stable" >&2
    cat "$CONTROL_LOG" >&2 || true
    cat "$WORKER1_LOG" >&2 || true
    cat "$WORKER2_LOG" >&2 || true
    exit 1
fi

require_preflight_runtime_readiness "$WORKER1_HOME"
require_preflight_runtime_readiness "$WORKER2_HOME"

if ! wait_for_topology 180; then
    echo "workers did not remain ring-stable after runtime readiness validation" >&2
    cat "$CONTROL_LOG" >&2 || true
    cat "$WORKER1_LOG" >&2 || true
    cat "$WORKER2_LOG" >&2 || true
    exit 1
fi

curl -fsS "http://127.0.0.1:${CONTROL_PORT}/api/ring/topology?network_id=${NETWORK_ID}" \
    >"$TOPOLOGY_LOG"

run_with_timeout 240s bash -lc \
    "cd '$WORKER1_HOME' && HOME='$WORKER1_HOME' MESHNET_HOME='$WORKER1_HOME' MESHNET_MODEL_STORE='$MODEL_STORE' CARGO_HOME='${CARGO_HOME:-$ORIGINAL_HOME/.cargo}' RUSTUP_HOME='${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}' '$AGENT_BIN' job run --model-id '$MODEL_ID' --prompt 'Say hello from mesh in five words.' --max-tokens 16 --temperature 0.0 --top-p 1.0" \
    >"$JOB1_LOG" 2>&1

if ! grep -q "Status:          completed" "$JOB1_LOG"; then
    echo "single-job inference did not complete successfully" >&2
    cat "$JOB1_LOG" >&2 || true
    exit 1
fi

if ! python3 - "$JOB1_LOG" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
token_match = re.search(r"Tokens:\s+(\d+)", text)
if not token_match or int(token_match.group(1)) <= 0:
    raise SystemExit(1)
if "Completion:" not in text:
    raise SystemExit(1)
completion = text.split("Completion:", 1)[1].strip()
if not completion:
    raise SystemExit(1)
PY
then
    echo "single-job inference did not emit completion text" >&2
    cat "$JOB1_LOG" >&2 || true
    exit 1
fi

run_with_timeout 300s bash -lc \
    "cd '$WORKER1_HOME' && HOME='$WORKER1_HOME' MESHNET_HOME='$WORKER1_HOME' MESHNET_MODEL_STORE='$MODEL_STORE' CARGO_HOME='${CARGO_HOME:-$ORIGINAL_HOME/.cargo}' RUSTUP_HOME='${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}' '$AGENT_BIN' job run --model-id '$MODEL_ID' --prompt 'Count upward in short words.' --max-tokens 32 --temperature 0.0 --top-p 1.0" \
    >"$JOB2_LOG" 2>&1 &
JOB2_PID=$!

run_with_timeout 300s bash -lc \
    "cd '$WORKER1_HOME' && HOME='$WORKER1_HOME' MESHNET_HOME='$WORKER1_HOME' MESHNET_MODEL_STORE='$MODEL_STORE' CARGO_HOME='${CARGO_HOME:-$ORIGINAL_HOME/.cargo}' RUSTUP_HOME='${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}' '$AGENT_BIN' job run --model-id '$MODEL_ID' --prompt 'List three tiny animals.' --max-tokens 32 --temperature 0.0 --top-p 1.0" \
    >"$JOB3_LOG" 2>&1 &
JOB3_PID=$!

run_with_timeout 300s bash -lc \
    "cd '$WORKER1_HOME' && HOME='$WORKER1_HOME' MESHNET_HOME='$WORKER1_HOME' MESHNET_MODEL_STORE='$MODEL_STORE' CARGO_HOME='${CARGO_HOME:-$ORIGINAL_HOME/.cargo}' RUSTUP_HOME='${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}' '$AGENT_BIN' job run --model-id '$MODEL_ID' --prompt 'Name a few bright colors.' --max-tokens 32 --temperature 0.0 --top-p 1.0" \
    >"$JOB4_LOG" 2>&1 &
JOB4_PID=$!

wait "$JOB2_PID"
wait "$JOB3_PID"
wait "$JOB4_PID"

for job_log in "$JOB2_LOG" "$JOB3_LOG" "$JOB4_LOG"; do
    if ! grep -q "Status:          completed" "$job_log"; then
        echo "concurrent inference job did not complete successfully" >&2
        cat "$job_log" >&2 || true
        exit 1
    fi
done

SCHEDULER_STATUS_JSON="$(
    curl -fsS "http://127.0.0.1:${CONTROL_PORT}/api/status/networks/${NETWORK_ID}/scheduler"
)"

python3 - "$SCHEDULER_STATUS_JSON" <<'PY'
import json
import sys

status = json.loads(sys.argv[1])
readiness = status.get("readiness") or {}
metrics = status.get("metrics") or {}

if not readiness.get("ready", False):
    blockers = readiness.get("blockers") or []
    raise SystemExit(
        "scheduler readiness was not green after real concurrent serving: "
        + ", ".join(str(blocker) for blocker in blockers)
    )

peak_batch_size = metrics.get("peak_batch_size") or 0
if peak_batch_size < 2:
    raise SystemExit(
        f"scheduler peak_batch_size did not prove pooled decode serving: {peak_batch_size}"
    )
PY

STATS_SUMMARY="$(python3 - "$WORKER1_HOME/.meshnet/inference_stats.json" "$WORKER2_HOME/.meshnet/inference_stats.json" <<'PY'
import json
import sys

paths = sys.argv[1:]
stats = []
for path in paths:
    with open(path) as handle:
        stats.append(json.load(handle))

total_tokens = sum(int(item.get("total_tokens_generated", 0)) for item in stats)
avg_tps = sum(float(item.get("avg_tokens_per_second", 0.0)) for item in stats)
allreduce_ops = sum(int(item.get("allreduce_operations", 0)) for item in stats)
tensor_bytes_sent = sum(int(item.get("tensor_bytes_sent", 0)) for item in stats)
max_multi_session_rate = max(float(item.get("multi_session_batch_rate", 0.0)) for item in stats)
max_decode_batch = max(float(item.get("avg_decode_batch_size", 0.0)) for item in stats)
max_peak_decode_batch = max(int(item.get("decode_batch_size_peak", 0)) for item in stats)
ttft_values = [item.get("time_to_first_token_ms") for item in stats if item.get("time_to_first_token_ms") is not None]

if total_tokens <= 0:
    raise SystemExit("total_tokens_generated was not positive")
if avg_tps <= 0.0:
    raise SystemExit("avg_tokens_per_second was not positive")
if allreduce_ops <= 0:
    raise SystemExit("allreduce_operations was not positive")
if tensor_bytes_sent <= 0:
    raise SystemExit("tensor_bytes_sent was not positive")
if max_decode_batch < 1.0:
    raise SystemExit("avg_decode_batch_size was invalid")
if max_peak_decode_batch < 2:
    raise SystemExit(
        f"concurrent real serving never produced a pooled decode peak batch; peak_batch_size={max_peak_decode_batch}"
    )
if max_multi_session_rate <= 0.0:
    raise SystemExit(
        "concurrent real serving never produced a pooled decode microbatch; "
        f"multi_session_batch_rate={max_multi_session_rate:.3f} avg_decode_batch_size={max_decode_batch:.2f} peak_batch_size={max_peak_decode_batch}"
    )

print(
    f"tokens={total_tokens} avg_tps={avg_tps:.2f} allreduce_ops={allreduce_ops} "
    f"tensor_bytes_sent={tensor_bytes_sent} multi_session_batch_rate={max_multi_session_rate:.3f} "
    f"avg_decode_batch_size={max_decode_batch:.2f} peak_batch_size={max_peak_decode_batch} "
    f"production_decode_pooling_ready=yes"
)
PY
)"

require_production_readiness "$WORKER1_HOME"
require_production_readiness "$WORKER2_HOME"

echo "real end-to-end inference test passed"
echo "real serving dynamics test passed"
echo "$STATS_SUMMARY"
echo "real cluster inference test passed"
