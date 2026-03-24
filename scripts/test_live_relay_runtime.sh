#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mesh-live-relay.XXXXXX")"
RELAY_PORT="${RELAY_PORT:-43113}"
RELAY_LOG="$TEST_ROOT/relay.log"
EXAMPLE_LOG="$TEST_ROOT/relay-connectivity.log"
RELAY_CONFIG="$TEST_ROOT/relay.toml"
ORIGINAL_HOME="${HOME:-}"

cleanup() {
    if [[ -n "${RELAY_PID:-}" ]]; then
        kill "$RELAY_PID" 2>/dev/null || true
        wait "$RELAY_PID" 2>/dev/null || true
    fi
    rm -rf "$TEST_ROOT"
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

export HOME="$TEST_ROOT/home"
if [[ -n "$ORIGINAL_HOME" ]]; then
    export CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}"
    export RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}"
fi
mkdir -p "$HOME"

cd "$ROOT_DIR"

cargo build -p relay-server --bin relay-server >/dev/null

RELAY_BIN="$(find "$ROOT_DIR/target" -type f \( -path '*/debug/relay-server' -o -path '*/debug/relay-server.exe' \) | head -n 1)"
if [[ -z "$RELAY_BIN" ]]; then
    echo "failed to locate built relay-server binary" >&2
    exit 1
fi

cat >"$RELAY_CONFIG" <<EOF
[relay]
max_reservations = 100
max_reservations_per_peer = 5
max_circuits_per_peer = 16
max_circuit_duration_secs = 120
max_circuit_bytes = 10485760

[network]
tcp_listen_addr = "/ip4/127.0.0.1/tcp/${RELAY_PORT}"
quic_listen_addr = "/ip4/127.0.0.1/udp/${RELAY_PORT}/quic-v1"
advertised_addrs = ["/ip4/127.0.0.1/tcp/${RELAY_PORT}", "/ip4/127.0.0.1/udp/${RELAY_PORT}/quic-v1"]

[auth]
auth_token = "CHANGE_ME_IN_PRODUCTION"
auth_enabled = false

[logging]
level = "info"
log_to_file = false
log_file_path = "~/.meshnet/logs/relay-server.log"
log_format = "pretty"
EOF

MESHNET_RELAY_ADDR="/ip4/127.0.0.1/tcp/${RELAY_PORT}" \
    "$RELAY_BIN" --config "$RELAY_CONFIG" \
    >"$RELAY_LOG" 2>&1 &
RELAY_PID=$!

for _ in $(seq 1 20); do
    if ps -p "$RELAY_PID" >/dev/null 2>&1; then
        sleep 0.25
        break
    fi
    sleep 0.25
done

if ! ps -p "$RELAY_PID" >/dev/null 2>&1; then
    echo "relay-server failed to stay up" >&2
    cat "$RELAY_LOG" >&2 || true
    exit 1
fi

if ! run_with_timeout 30s env MESHNET_RELAY_ADDR="/ip4/127.0.0.1/tcp/${RELAY_PORT}" \
    cargo run -p agent --example relay_connectivity >"$EXAMPLE_LOG" 2>&1; then
    echo "relay connectivity example failed" >&2
    cat "$EXAMPLE_LOG" >&2 || true
    exit 1
fi

if ! grep -Eq "SUCCESS!|DCUTR SUCCESS!" "$EXAMPLE_LOG"; then
    echo "relay connectivity example did not report successful live connection" >&2
    cat "$EXAMPLE_LOG" >&2 || true
    exit 1
fi

echo "live relay runtime test passed"
