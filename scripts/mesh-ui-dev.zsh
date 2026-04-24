#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_PID=""

cleanup() {
  if [[ -n "${API_PID}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

if command -v mesh >/dev/null 2>&1 && mesh ui --help 2>/dev/null | grep -q -- '--api-only'; then
  mesh ui --api-only --api-port 43111 &
elif [[ -x "${HOME}/.local/bin/mesh" ]] && "${HOME}/.local/bin/mesh" ui --help 2>/dev/null | grep -q -- '--api-only'; then
  "${HOME}/.local/bin/mesh" ui --api-only --api-port 43111 &
else
  cargo run -p agent -- ui --api-only --api-port 43111 &
fi
API_PID=$!

for _ in {1..240}; do
  if curl -fsS "http://127.0.0.1:43111/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

if ! curl -fsS "http://127.0.0.1:43111/health" >/dev/null 2>&1; then
  echo "Mesh UI local API failed to start on 127.0.0.1:43111" >&2
  exit 1
fi

exec corepack pnpm --dir "${ROOT_DIR}/mesh-ui" exec vite --host 127.0.0.1 --port 5178 --strictPort
