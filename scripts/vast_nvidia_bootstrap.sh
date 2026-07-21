#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/vast_nvidia_bootstrap.sh status [INSTANCE_ID]
  scripts/vast_nvidia_bootstrap.sh wait [INSTANCE_ID]
  scripts/vast_nvidia_bootstrap.sh verify [INSTANCE_ID]
  scripts/vast_nvidia_bootstrap.sh sync [INSTANCE_ID]
  scripts/vast_nvidia_bootstrap.sh bootstrap [INSTANCE_ID]

Environment:
  VASTAI_BIN       VastAI CLI path. Defaults to `vastai`, then /tmp/mesh-vastai-venv/bin/vastai.
  REMOTE_WORKDIR  Remote parent directory. Defaults to /workspace/work.
  FOZZYLANG_DIR   Local Fozzylang source. Defaults to /Users/deepsaint/Desktop/fozzylang.
  SSH_OPTS        Extra ssh options.
EOF
}

resolve_vastai_bin() {
  if [[ -n "${VASTAI_BIN:-}" ]]; then
    printf '%s\n' "$VASTAI_BIN"
  elif command -v vastai >/dev/null 2>&1; then
    command -v vastai
  elif [[ -x /tmp/mesh-vastai-venv/bin/vastai ]]; then
    printf '%s\n' /tmp/mesh-vastai-venv/bin/vastai
  else
    printf 'vastai CLI not found. Install it or set VASTAI_BIN.\n' >&2
    exit 2
  fi
}

vastai_json() {
  local vastai_bin
  vastai_bin="$(resolve_vastai_bin)"
  "$vastai_bin" --raw show instances 2>/tmp/meshnet-vastai.err
}

newest_instance_id() {
  vastai_json | python3 -c '
import json, sys
instances = json.load(sys.stdin)
nvidia = [x for x in instances if x.get("gpu_arch") == "nvidia"]
if not nvidia:
    raise SystemExit("no NVIDIA VastAI instances are present")
newest = max(nvidia, key=lambda x: (x.get("id") or 0, x.get("start_date") or 0))
print(newest["id"])
'
}

instance_summary() {
  local instance_id="$1"
  vastai_json | INSTANCE_ID="$instance_id" python3 -c '
import json, os, sys
instance_id = int(os.environ["INSTANCE_ID"])
keys = ["id", "label", "gpu_name", "actual_status", "cur_state", "intended_status", "next_state", "ssh_host", "ssh_port"]
for item in json.load(sys.stdin):
    if item.get("id") == instance_id:
        print({key: item.get(key) for key in keys})
        raise SystemExit(0)
raise SystemExit(f"instance {instance_id} not found")
'
}

ssh_url() {
  local instance_id="$1"
  local vastai_bin
  vastai_bin="$(resolve_vastai_bin)"
  "$vastai_bin" ssh-url "$instance_id" | tail -n 1
}

ssh_target() {
  local url="$1"
  python3 -c '
from urllib.parse import urlparse
import sys
url = urlparse(sys.argv[1])
user = url.username or "root"
host = url.hostname
port = url.port
if not host or not port:
    raise SystemExit(f"invalid ssh url: {sys.argv[1]}")
print(f"{user} {host} {port}")
' "$url"
}

remote_ssh() {
  local instance_id="$1"
  shift
  local user host port
  read -r user host port < <(ssh_target "$(ssh_url "$instance_id")")
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ${SSH_OPTS:-} -p "$port" "$user@$host" "$@"
}

wait_for_ssh() {
  local instance_id="$1"
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    instance_summary "$instance_id" || true
    if remote_ssh "$instance_id" 'hostname >/dev/null' >/dev/null 2>&1; then
      remote_ssh "$instance_id" 'hostname; nvidia-smi -L'
      return 0
    fi
    sleep 10
  done
  printf 'Timed out waiting for SSH on VastAI instance %s\n' "$instance_id" >&2
  return 1
}

sync_tree() {
  local instance_id="$1"
  local local_dir="$2"
  local remote_name="$3"
  local remote_workdir="${REMOTE_WORKDIR:-/workspace/work}"
  [[ -d "$local_dir" ]] || {
    printf 'Local source directory does not exist: %s\n' "$local_dir" >&2
    exit 2
  }
  remote_ssh "$instance_id" "mkdir -p '$remote_workdir'"
  COPYFILE_DISABLE=1 tar \
    --exclude target \
    --exclude .fozzy \
    --exclude .DS_Store \
    -C "$local_dir" \
    -cf - . \
    | remote_ssh "$instance_id" "rm -rf '$remote_workdir/$remote_name' && mkdir -p '$remote_workdir/$remote_name' && tar -C '$remote_workdir/$remote_name' -xf -"
}

sync_sources() {
  local instance_id="$1"
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  sync_tree "$instance_id" "$repo_root" meshnet
  sync_tree "$instance_id" "${FOZZYLANG_DIR:-/Users/deepsaint/Desktop/fozzylang}" fozzylang
}

verify_cuda() {
  local instance_id="$1"
  remote_ssh "$instance_id" 'set -euo pipefail; hostname; uname -a; nvidia-smi; command -v nvcc || true; command -v cargo || true; command -v rustc || true'
}

main() {
  local command="${1:-}"
  [[ -n "$command" ]] || {
    usage
    exit 2
  }
  shift || true
  local instance_id="${1:-}"
  if [[ -z "$instance_id" ]]; then
    instance_id="$(newest_instance_id)"
  fi

  case "$command" in
    status) instance_summary "$instance_id" ;;
    wait) wait_for_ssh "$instance_id" ;;
    verify) verify_cuda "$instance_id" ;;
    sync) sync_sources "$instance_id" ;;
    bootstrap)
      wait_for_ssh "$instance_id"
      verify_cuda "$instance_id"
      sync_sources "$instance_id"
      ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
