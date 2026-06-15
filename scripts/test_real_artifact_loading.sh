#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_STORE="${MESHNET_MODEL_STORE:-${HOME}/.meshnet/models}"

if [[ ! -d "$MODEL_STORE" ]]; then
    echo "real artifact model store not found at $MODEL_STORE" >&2
    exit 1
fi

MODEL_ID="${MESHNET_REAL_ARTIFACT_MODEL_ID:-}"
if [[ -z "$MODEL_ID" ]]; then
    DISCOVERED_MANIFEST="$(find "$MODEL_STORE" -mindepth 2 -maxdepth 2 -name 'shard-*-of-*.manifest.json' | head -n 1)"
    if [[ -z "$DISCOVERED_MANIFEST" ]]; then
        echo "no real artifact shard manifests found under $MODEL_STORE" >&2
        exit 1
    fi
    MODEL_ID="$(basename "$(dirname "$DISCOVERED_MANIFEST")")"
fi

cd "$ROOT_DIR"
MESHNET_ENABLE_REAL_ARTIFACT_TEST=1 \
MESHNET_REAL_ARTIFACT_MODEL_ID="$MODEL_ID" \
cargo test -p agent --test real_artifact_loading -- --nocapture
