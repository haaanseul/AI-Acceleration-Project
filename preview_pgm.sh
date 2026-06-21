#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PGM_DIR="${PGM_DIR:-$SCRIPT_DIR/mnistCUDNN/pgm_output}"
OUTPUT="${OUTPUT:-$PGM_DIR/pgm_preview.png}"

"$PYTHON_BIN" "$SCRIPT_DIR/yolo/preview_pgm_grid.py" \
    --input "$PGM_DIR" \
    --output "$OUTPUT" \
    "$@"

echo "PGM preview: $OUTPUT"
