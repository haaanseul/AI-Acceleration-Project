#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VIDEO="${VIDEO:-number.mp4}"
OUTPUT="${OUTPUT:-preview_yolo.mp4}"
WEIGHTS="${YOLO_WEIGHTS:-}"

args=(
    yolo/preview_detection.py
    --video "$VIDEO"
    --output "$OUTPUT"
)

if [[ -n "$WEIGHTS" ]]; then
    args+=(--weights "$WEIGHTS")
fi

exec "$PYTHON_BIN" "${args[@]}"
