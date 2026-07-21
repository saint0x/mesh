#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MESHNET_LAN_MODEL_ID:-smollm2-135m-instruct}"
NETWORK_ID="${MESHNET_LAN_NETWORK_ID:-mac-gmk-lan}"
CONTROL_PORT="${MESHNET_LAN_CONTROL_PORT:-43180}"
MAC_TENSOR_PORT="${MESHNET_MAC_TENSOR_PORT:-43210}"
GMK_TENSOR_PORT="${MESHNET_GMK_TENSOR_PORT:-43211}"
MAC_PROVIDER="${MESHNET_MAC_PROVIDER:-metal}"
GMK_PROVIDER="${MESHNET_GMK_PROVIDER:-rocm}"
MAC_MEMORY="${MESHNET_MAC_MEMORY:-8GB}"
GMK_MEMORY="${MESHNET_GMK_MEMORY:-1GB}"
MAC_HOST_IP="${MESHNET_MAC_HOST_IP:-}"
GMK_WINDOWS_HOST="${MESHNET_GMK_WINDOWS_HOST:-192.168.1.139}"
GMK_WINDOWS_USER="${MESHNET_GMK_WINDOWS_USER:-deepsaint}"
GMK_LAN_IP="${MESHNET_GMK_LAN_IP:-$GMK_WINDOWS_HOST}"
GMK_TENSOR_ADVERTISED_ADDR="${MESHNET_GMK_TENSOR_ADVERTISED_ADDR:-127.0.0.1:${GMK_TENSOR_PORT}}"
GMK_REPO="${MESHNET_GMK_REPO:-/home/deepsaint/work/meshnet}"
GMK_MODEL_STORE="${MESHNET_GMK_MODEL_STORE:-/home/deepsaint/.meshnet/models}"
LOCAL_MODEL_STORE="${MESHNET_MODEL_STORE:-$HOME/.meshnet/models}"
RUN_ROOT="${MESHNET_LAN_RUN_ROOT:-/tmp/mesh-mac-gmk}"
GMK_RUN_ROOT="${MESHNET_GMK_RUN_ROOT:-/home/deepsaint/.meshnet/runs/mac-gmk}"
KEEP_RUN_ROOT="${MESHNET_KEEP_RUN_ROOT:-0}"
RUST_LOG_LEVEL="${MESHNET_LAN_RUST_LOG:-info}"
SKIP_MODEL_SYNC="${MESHNET_SKIP_MODEL_SYNC:-0}"

MAC_HOME="$RUN_ROOT/mac-worker"
CONTROL_HOME="$RUN_ROOT/control-plane"
LOG_DIR="$RUN_ROOT/logs"
CONTROL_LOG="$LOG_DIR/control-plane.log"
MAC_LOG="$LOG_DIR/mac-agent.log"
JOB_LOG="$LOG_DIR/job.log"

CONTROL_PID=""
MAC_AGENT_PID=""
GMK_TUNNEL_PID=""
LOCAL_AGENT_BIN=""
LOCAL_CONTROL_BIN=""

usage() {
    cat <<EOF
usage: $(basename "$0") [command]

Commands:
  run          Build, sync artifacts, start Mac+GMK Mesh, run one smoke inference.
  sync-model   Copy $MODEL_ID artifacts from Mac to GMK.
  stop         Stop processes from the current run root on Mac and GMK.

Environment:
  MESHNET_LAN_MODEL_ID          Model ID to serve (default: $MODEL_ID)
  MESHNET_MAC_HOST_IP           Mac LAN IP override
  MESHNET_GMK_WINDOWS_HOST      Windows host SSH/LAN IP (default: $GMK_WINDOWS_HOST)
  MESHNET_GMK_LAN_IP            GMK advertised LAN IP (default: Windows host)
  MESHNET_MAC_TENSOR_PORT       Fixed Mac tensor port (default: $MAC_TENSOR_PORT)
  MESHNET_GMK_TENSOR_PORT       Fixed GMK tensor port (default: $GMK_TENSOR_PORT)
  MESHNET_GMK_TENSOR_ADVERTISED_ADDR  GMK tensor endpoint visible from Mac (default: $GMK_TENSOR_ADVERTISED_ADDR)
  MESHNET_SKIP_MODEL_SYNC=1     Skip model artifact copy
  MESHNET_KEEP_RUN_ROOT=1       Preserve local temp run root on exit
  MESHNET_GMK_RUN_ROOT          Persistent GMK run root (default: $GMK_RUN_ROOT)
EOF
}

log() {
    printf '[mesh-mac-gmk] %s\n' "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

detect_mac_ip() {
    if [[ -n "$MAC_HOST_IP" ]]; then
        printf '%s\n' "$MAC_HOST_IP"
        return
    fi
    ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || {
        ifconfig | awk '/inet / && $2 !~ /^127\./ {print $2; exit}'
    }
}

gmk_bash() {
    local script="$1"
    printf '%s\n' "$script" | ssh "${GMK_WINDOWS_USER}@${GMK_WINDOWS_HOST}" 'wsl.exe -e bash -l -s'
}

gmk_wsl_ip() {
    gmk_bash 'ip -4 addr show eth0 | sed -n "s/.*inet \([0-9.]*\).*/\1/p" | head -n 1'
}

start_gmk_tensor_bridge() {
    local _wsl_ip="$1"
    local advertised_host="${GMK_TENSOR_ADVERTISED_ADDR%:*}"
    local advertised_port="${GMK_TENSOR_ADVERTISED_ADDR##*:}"
    local bridge_log="$LOG_DIR/gmk-tensor-bridge.log"
    log "starting GMK tensor stdio bridge ${GMK_TENSOR_ADVERTISED_ADDR} -> WSL 127.0.0.1:${GMK_TENSOR_PORT}"
    python3 - "$advertised_host" "$advertised_port" "$GMK_WINDOWS_USER" "$GMK_WINDOWS_HOST" "$GMK_TENSOR_PORT" >"$bridge_log" 2>&1 <<'PY' &
import socket
import os
import subprocess
import sys
import threading

listen_host = sys.argv[1]
listen_port = int(sys.argv[2])
windows_user = sys.argv[3]
windows_host = sys.argv[4]
gmk_tensor_port = sys.argv[5]


def log(message):
    print(message, flush=True)


def pump(client):
    peer = client.getpeername()
    log(f"accepted local tensor stream from {peer}")
    proc = subprocess.Popen(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            f"{windows_user}@{windows_host}",
            "wsl.exe",
            "-e",
            "nc",
            "127.0.0.1",
            gmk_tensor_port,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    log(f"spawned WSL tensor connector pid={proc.pid} for {peer}")

    def client_to_wsl():
        try:
            while True:
                data = client.recv(65536)
                if not data:
                    break
                os.write(proc.stdin.fileno(), data)
        except OSError as exc:
            log(f"client->WSL pump stopped peer={peer} error={exc!r}")
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def wsl_to_client():
        try:
            while True:
                data = os.read(proc.stdout.fileno(), 65536)
                if not data:
                    break
                client.sendall(data)
        except OSError as exc:
            log(f"WSL->client pump stopped peer={peer} error={exc!r}")
        finally:
            try:
                client.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    upstream = threading.Thread(target=client_to_wsl, daemon=True)
    downstream = threading.Thread(target=wsl_to_client, daemon=True)
    upstream.start()
    downstream.start()
    try:
        proc.wait()
    finally:
        if proc.poll() is not None:
            stderr = os.read(proc.stderr.fileno(), 65536).decode("utf-8", "replace")
            log(f"WSL tensor connector exited rc={proc.returncode} peer={peer} stderr={stderr!r}")
        else:
            log(f"closing WSL tensor connector pid={proc.pid} peer={peer}")
        try:
            client.close()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((listen_host, listen_port))
    listener.listen()
    while True:
        client, _ = listener.accept()
        thread = threading.Thread(target=pump, args=(client,), daemon=True)
        thread.start()
PY
    GMK_TUNNEL_PID=$!
    sleep 1
    if ! kill -0 "$GMK_TUNNEL_PID" 2>/dev/null; then
        wait "$GMK_TUNNEL_PID" 2>/dev/null || true
        cat "$bridge_log" >&2 || true
        die "failed to start GMK tensor stdio bridge"
    fi
}

sync_model() {
    local source_dir="$LOCAL_MODEL_STORE/$MODEL_ID"
    [[ -d "$source_dir" ]] || die "model artifacts not found at $source_dir"
    log "syncing model artifacts $MODEL_ID to GMK:$GMK_MODEL_STORE"
    COPYFILE_DISABLE=1 tar -C "$LOCAL_MODEL_STORE" -cf - "$MODEL_ID" \
        | ssh "${GMK_WINDOWS_USER}@${GMK_WINDOWS_HOST}" \
            "wsl.exe -e bash -lc \"mkdir -p '$GMK_MODEL_STORE' && tar -xf - -C '$GMK_MODEL_STORE' && find '$GMK_MODEL_STORE/$MODEL_ID' -name '._*' -delete\""
}

wait_for_http() {
    local url="$1"
    local attempts="${2:-120}"
    for _ in $(seq 1 "$attempts"); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

wait_for_topology() {
    local control_url="$1"
    local attempts="${2:-180}"
    for _ in $(seq 1 "$attempts"); do
        if curl -fsS "$control_url/api/ring/topology?network_id=$NETWORK_ID" | python3 -c '
import json
import sys

data = json.load(sys.stdin)
workers = data.get("workers") or []
if len(workers) != 2:
    raise SystemExit(1)
if not data.get("ring_stable", False):
    raise SystemExit(1)
if any(worker.get("status") != "online" for worker in workers):
    raise SystemExit(1)
if any(not any(str(addr).startswith("dataplane://") for addr in worker.get("listen_addrs", [])) for worker in workers):
    raise SystemExit(1)
'
        then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_runtime_readiness() {
    local remote_home="$1"
    local attempts="${2:-300}"
    for _ in $(seq 1 "$attempts"); do
        local mac_ready=1
        local gmk_ready=1
        if grep -q "Loaded verified safetensors shard.*model_id.*${MODEL_ID}" "$MAC_LOG"; then
            mac_ready=0
        fi
        if gmk_bash "grep -q 'Loaded verified safetensors shard.*model_id.*${MODEL_ID}' '$remote_home/.meshnet/logs/agent.log'"; then
            gmk_ready=0
        fi
        if [[ "$mac_ready" -eq 0 && "$gmk_ready" -eq 0 ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_tcp_endpoint() {
    local host="$1"
    local port="$2"
    local attempts="${3:-60}"
    for _ in $(seq 1 "$attempts"); do
        if python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=2.0):
    pass
PY
        then
            return 0
        fi
        sleep 1
    done
    return 1
}

build_local() {
    log "building local Mac Mesh binaries"
    cargo build -p agent --bin agent -p control-plane --bin control-plane >/dev/null
    LOCAL_AGENT_BIN="$(find "$ROOT_DIR/target" -type f -path '*/debug/agent' | head -n 1)"
    LOCAL_CONTROL_BIN="$(find "$ROOT_DIR/target" -type f -path '*/debug/control-plane' | head -n 1)"
    [[ -n "$LOCAL_AGENT_BIN" ]] || die "failed to locate built local agent binary"
    [[ -n "$LOCAL_CONTROL_BIN" ]] || die "failed to locate built local control-plane binary"
}

build_gmk() {
    log "building GMK ROCm Mesh agent"
    gmk_bash "set -e
cd '$GMK_REPO'
. ~/.cargo/env
ROCM_PATH=/opt/rocm cargo build -p agent --features rocm >/dev/null
"
}

create_network() {
    local control_url="$1"
    curl -fsS -X POST "$control_url/api/networks" \
        -H "Content-Type: application/json" \
        -d "{
            \"network_id\": \"${NETWORK_ID}\",
            \"name\": \"Mac GMK LAN\",
            \"owner_user_id\": \"local-lan\",
            \"connectivity\": {
                \"preferred_path\": \"direct\",
                \"attachments\": []
            }
        }" >/dev/null
}

seed_credits() {
    local control_url="$1"
    local home_dir="$2"
    local device_id
    device_id="$(awk -F'"' '/^device_id = / { print $2; exit }' "$home_dir/.meshnet/device.toml")"
    [[ -n "$device_id" ]] || die "missing device_id in $home_dir"
    curl -fsS -X POST "$control_url/api/ledger/events" \
        -H "Content-Type: application/json" \
        -d "{
            \"network_id\": \"${NETWORK_ID}\",
            \"event_type\": \"credits_earned\",
            \"job_id\": null,
            \"device_id\": \"${device_id}\",
            \"credits_amount\": 10000.0,
            \"metadata\": {
                \"credit_model\": \"bootstrap_lan_funds\",
                \"reason\": \"seed Mac+GMK LAN smoke credits\"
            }
        }" >/dev/null
}

seed_gmk_credits() {
    local control_url="$1"
    local remote_home="$2"
    local device_id
    device_id="$(gmk_bash "awk -F'\\\"' '/^device_id = / { print \$2; exit }' '$remote_home/.meshnet/device.toml'")"
    [[ -n "$device_id" ]] || die "missing GMK device_id in $remote_home"
    curl -fsS -X POST "$control_url/api/ledger/events" \
        -H "Content-Type: application/json" \
        -d "{
            \"network_id\": \"${NETWORK_ID}\",
            \"event_type\": \"credits_earned\",
            \"job_id\": null,
            \"device_id\": \"${device_id}\",
            \"credits_amount\": 10000.0,
            \"metadata\": {
                \"credit_model\": \"bootstrap_lan_funds\",
                \"reason\": \"seed GMK LAN smoke credits\"
            }
        }" >/dev/null
}

start_control_plane() {
    mkdir -p "$CONTROL_HOME" "$LOG_DIR"
    (
        cd "$CONTROL_HOME"
        env HOME="$CONTROL_HOME" MESHNET_MODEL_STORE="$LOCAL_MODEL_STORE" RUST_LOG="$RUST_LOG_LEVEL" \
            "$LOCAL_CONTROL_BIN" --port "$CONTROL_PORT" >"$CONTROL_LOG" 2>&1
    ) &
    CONTROL_PID=$!
    log "control-plane pid=$CONTROL_PID log=$CONTROL_LOG"
}

start_mac_agent() {
    local mac_ip="$1"
    (
        cd "$MAC_HOME"
        env \
            HOME="$MAC_HOME" \
            MESHNET_HOME="$MAC_HOME" \
            MESHNET_MODEL_STORE="$LOCAL_MODEL_STORE" \
            RUST_LOG="$RUST_LOG_LEVEL" \
            RUST_BACKTRACE=1 \
            MESHNET_TENSOR_BIND_ADDR="0.0.0.0:${MAC_TENSOR_PORT}" \
            MESHNET_TENSOR_ADVERTISED_ADDR="${mac_ip}:${MAC_TENSOR_PORT}" \
            "$LOCAL_AGENT_BIN" device runtime --log-level info >"$MAC_LOG" 2>&1
    ) &
    MAC_AGENT_PID=$!
    log "mac agent pid=$MAC_AGENT_PID log=$MAC_LOG"
}

start_gmk_agent() {
    local control_url="$1"
    local remote_home="$2"
    gmk_bash "set -e
cd '$GMK_REPO'
mkdir -p '$remote_home/.meshnet/logs'
. ~/.cargo/env
AGENT_BIN=\"\$(find '$GMK_REPO/target' -type f -path '*/debug/agent' | head -n 1)\"
test -n \"\$AGENT_BIN\"
cat > '$remote_home/start-agent.sh' <<'REMOTE_START'
#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOME='$remote_home'
GMK_REPO='$GMK_REPO'
GMK_MODEL_STORE='$GMK_MODEL_STORE'
RUST_LOG_LEVEL='$RUST_LOG_LEVEL'
GMK_TENSOR_PORT='$GMK_TENSOR_PORT'
GMK_TENSOR_ADVERTISED_ADDR='$GMK_TENSOR_ADVERTISED_ADDR'

mkdir -p \"\$REMOTE_HOME/.meshnet/logs\"
launch_log=\"\$REMOTE_HOME/.meshnet/logs/agent-launch.log\"
agent_log=\"\$REMOTE_HOME/.meshnet/logs/agent.log\"
pid_file=\"\$REMOTE_HOME/.meshnet/agent.pid\"

{
  echo \"launch_time=\$(date -Is)\"
  echo \"remote_home=\$REMOTE_HOME\"
  echo \"repo=\$GMK_REPO\"
  echo \"model_store=\$GMK_MODEL_STORE\"
  echo \"tensor_bind=0.0.0.0:\$GMK_TENSOR_PORT\"
  echo \"tensor_advertised=\$GMK_TENSOR_ADVERTISED_ADDR\"
} >\"\$launch_log\"

cd \"\$GMK_REPO\"
. ~/.cargo/env
AGENT_BIN=\"\$(find \"\$GMK_REPO/target\" -type f -path '*/debug/agent' | head -n 1)\"
test -n \"\$AGENT_BIN\"
echo \"agent_bin=\$AGENT_BIN\" >>\"\$launch_log\"

rm -f \"\$pid_file\"
nohup env \
	  HOME=\"\$REMOTE_HOME\" \
	  MESHNET_HOME=\"\$REMOTE_HOME\" \
	  MESHNET_MODEL_STORE=\"\$GMK_MODEL_STORE\" \
	  RUST_LOG=\"\$RUST_LOG_LEVEL\" \
	  RUST_BACKTRACE=1 \
	  ROCM_PATH=/opt/rocm \
	  MESHNET_TENSOR_BIND_ADDR=\"0.0.0.0:\$GMK_TENSOR_PORT\" \
	  MESHNET_TENSOR_ADVERTISED_ADDR=\"\$GMK_TENSOR_ADVERTISED_ADDR\" \
	  \"\$AGENT_BIN\" device runtime --log-level info >>\"\$agent_log\" 2>&1 &
echo \$! >\"\$pid_file\"

sleep 2
if ! kill -0 \"\$(cat \"\$pid_file\")\" 2>/dev/null; then
  echo \"worker exited during launch\" >>\"\$launch_log\"
  tail -n 200 \"\$agent_log\" >&2 || true
  exit 1
fi
echo \"pid=\$(cat \"\$pid_file\")\" >>\"\$launch_log\"
REMOTE_START
chmod +x '$remote_home/start-agent.sh'
'$remote_home/start-agent.sh'
"
    log "gmk agent remote_home=$remote_home log=$remote_home/.meshnet/logs/agent.log control=$control_url"
}

stop_processes() {
    set +e
    if [[ -n "${MAC_AGENT_PID:-}" ]]; then
        kill "$MAC_AGENT_PID" 2>/dev/null
        wait "$MAC_AGENT_PID" 2>/dev/null
    fi
    if [[ -n "${CONTROL_PID:-}" ]]; then
        kill "$CONTROL_PID" 2>/dev/null
        wait "$CONTROL_PID" 2>/dev/null
    fi
    if [[ -n "${GMK_TUNNEL_PID:-}" ]]; then
        kill "$GMK_TUNNEL_PID" 2>/dev/null
        wait "$GMK_TUNNEL_PID" 2>/dev/null
    fi
    gmk_bash "set +e
	for pid_file in '$GMK_RUN_ROOT/gmk-worker/.meshnet/agent.pid'; do
  if [[ -f \"\$pid_file\" ]]; then
    kill \"\$(cat \"\$pid_file\")\" 2>/dev/null || true
  fi
done
" >/dev/null 2>&1 || true
    set -e
}

fetch_gmk_run_logs() {
    if ! ssh "${GMK_WINDOWS_USER}@${GMK_WINDOWS_HOST}" "wsl.exe -e bash -lc \"test -d '$GMK_RUN_ROOT/gmk-worker'\"" >/dev/null 2>&1; then
        return 0
    fi
    mkdir -p "$LOG_DIR/gmk-worker"
    ssh "${GMK_WINDOWS_USER}@${GMK_WINDOWS_HOST}" "wsl.exe -e bash -lc \"tar -C '$GMK_RUN_ROOT/gmk-worker' -cf - .meshnet/logs .meshnet/agent.pid start-agent.sh 2>/dev/null\"" \
        | tar -xf - -C "$LOG_DIR/gmk-worker" 2>/dev/null || true
}

cleanup() {
    stop_processes
    if [[ "$KEEP_RUN_ROOT" != "1" ]]; then
        rm -rf "$RUN_ROOT"
        gmk_bash "rm -rf '$GMK_RUN_ROOT/gmk-worker'" >/dev/null 2>&1 || true
    else
        fetch_gmk_run_logs
        log "preserved run root: $RUN_ROOT"
    fi
}

run_cluster() {
    require_cmd cargo
    require_cmd curl
    require_cmd ssh
    require_cmd tar
    require_cmd python3

    local mac_ip
    mac_ip="$(detect_mac_ip)"
    [[ -n "$mac_ip" ]] || die "could not determine Mac LAN IP; set MESHNET_MAC_HOST_IP"

    [[ -d "$LOCAL_MODEL_STORE/$MODEL_ID" ]] || die "model artifacts missing: $LOCAL_MODEL_STORE/$MODEL_ID"

    rm -rf "$RUN_ROOT"
    mkdir -p "$MAC_HOME" "$CONTROL_HOME" "$LOG_DIR"
    trap cleanup EXIT INT TERM

    log "mac_ip=$mac_ip gmk_ip=$GMK_LAN_IP model=$MODEL_ID network=$NETWORK_ID"
    local gmk_wsl_ip_value
    gmk_wsl_ip_value="$(gmk_wsl_ip)"
    [[ -n "$gmk_wsl_ip_value" ]] || die "could not resolve GMK WSL IP"
    start_gmk_tensor_bridge "$gmk_wsl_ip_value"
    if [[ "$SKIP_MODEL_SYNC" != "1" ]]; then
        sync_model
    fi
    build_local
    build_gmk

    local control_url_local="http://127.0.0.1:${CONTROL_PORT}"
    local control_url_lan="http://${mac_ip}:${CONTROL_PORT}"
    local remote_home="$GMK_RUN_ROOT/gmk-worker"
    gmk_bash "rm -rf '$remote_home' && mkdir -p '$remote_home'"

    start_control_plane
    wait_for_http "$control_url_local/health" 120 || {
        cat "$CONTROL_LOG" >&2 || true
        die "control-plane did not become healthy"
    }
    create_network "$control_url_local"

    log "initializing Mac worker provider=$MAC_PROVIDER"
    env HOME="$MAC_HOME" MESHNET_HOME="$MAC_HOME" MESHNET_MODEL_STORE="$LOCAL_MODEL_STORE" \
        "$LOCAL_AGENT_BIN" device init \
        --network-id "$NETWORK_ID" \
        --name "Mac Metal" \
        --control-plane "$control_url_lan" \
        --preferred-provider "$MAC_PROVIDER" >/dev/null

    log "initializing GMK worker provider=$GMK_PROVIDER"
    gmk_bash "set -e
mkdir -p '$remote_home'
cd '$GMK_REPO'
. ~/.cargo/env
AGENT_BIN=\"\$(find '$GMK_REPO/target' -type f -path '*/debug/agent' | head -n 1)\"
test -n \"\$AGENT_BIN\"
env HOME='$remote_home' MESHNET_HOME='$remote_home' MESHNET_MODEL_STORE='$GMK_MODEL_STORE' ROCM_PATH=/opt/rocm \
  \"\$AGENT_BIN\" device init \
    --network-id '$NETWORK_ID' \
    --name 'GMK ROCm' \
    --control-plane '$control_url_lan' \
    --preferred-provider '$GMK_PROVIDER' >/dev/null
"

    seed_credits "$control_url_local" "$MAC_HOME"
    seed_gmk_credits "$control_url_local" "$remote_home"

    log "joining model ring model=$MODEL_ID"
    env HOME="$MAC_HOME" MESHNET_HOME="$MAC_HOME" MESHNET_MODEL_STORE="$LOCAL_MODEL_STORE" \
        "$LOCAL_AGENT_BIN" ring join --model-id "$MODEL_ID" --memory "$MAC_MEMORY" >/dev/null
    gmk_bash "set -e
cd '$GMK_REPO'
. ~/.cargo/env
AGENT_BIN=\"\$(find '$GMK_REPO/target' -type f -path '*/debug/agent' | head -n 1)\"
test -n \"\$AGENT_BIN\"
env HOME='$remote_home' MESHNET_HOME='$remote_home' MESHNET_MODEL_STORE='$GMK_MODEL_STORE' ROCM_PATH=/opt/rocm \
  \"\$AGENT_BIN\" ring join --model-id '$MODEL_ID' --memory '$GMK_MEMORY' >/dev/null
"

    start_mac_agent "$mac_ip"
    start_gmk_agent "$control_url_lan" "$remote_home"

    if ! wait_for_topology "$control_url_local" 240; then
        log "topology failed; dumping logs"
        curl -fsS "$control_url_local/api/ring/topology?network_id=$NETWORK_ID" >&2 || true
        cat "$CONTROL_LOG" >&2 || true
        cat "$MAC_LOG" >&2 || true
        gmk_bash "cat '$remote_home/.meshnet/logs/agent.log' 2>/dev/null || true" >&2 || true
        die "workers did not become ring-stable"
    fi

    log "topology stable; waiting for production shard readiness"
    if ! wait_for_runtime_readiness "$remote_home" 300; then
        log "runtime readiness failed; dumping logs"
        cat "$MAC_LOG" >&2 || true
        gmk_bash "cat '$remote_home/.meshnet/logs/agent.log' 2>/dev/null || true" >&2 || true
        die "workers did not load production shard artifacts"
    fi

    log "validating advertised tensor endpoints"
    wait_for_tcp_endpoint "$mac_ip" "$MAC_TENSOR_PORT" 30 || die "Mac tensor endpoint ${mac_ip}:${MAC_TENSOR_PORT} is not reachable"
    wait_for_tcp_endpoint "${GMK_TENSOR_ADVERTISED_ADDR%:*}" "${GMK_TENSOR_ADVERTISED_ADDR##*:}" 30 || {
        gmk_bash "ss -ltnp | grep '${GMK_TENSOR_PORT}' || true" >&2 || true
        die "GMK tensor endpoint ${GMK_TENSOR_ADVERTISED_ADDR} is not reachable"
    }
    sleep 3

    log "topology stable; running smoke inference"
    env HOME="$MAC_HOME" MESHNET_HOME="$MAC_HOME" MESHNET_MODEL_STORE="$LOCAL_MODEL_STORE" \
        "$LOCAL_AGENT_BIN" job run \
        --model-id "$MODEL_ID" \
        --prompt "Say hello from the Mac and GMK mesh in five words." \
        --max-tokens 16 \
        --temperature 0.0 \
        --top-p 1.0 >"$JOB_LOG" 2>&1

    if ! grep -q "Status:          completed" "$JOB_LOG"; then
        cat "$JOB_LOG" >&2 || true
        die "smoke inference did not complete"
    fi

    log "smoke inference completed"
    cat "$JOB_LOG"
    log "logs: $LOG_DIR"
}

case "${1:-run}" in
    run)
        run_cluster
        ;;
    sync-model)
        sync_model
        ;;
    stop)
        stop_processes
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
