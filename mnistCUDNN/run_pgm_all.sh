#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PGM_DIR="${PGM_DIR:-$SCRIPT_DIR/pgm_output}"
EXE="${EXE:-$SCRIPT_DIR/mnistCUDNN}"
VIDEO="${VIDEO:-$ROOT_DIR/number.mp4}"
YOLO_SCRIPT="${YOLO_SCRIPT:-$ROOT_DIR/yolo/extract_pgm.py}"

if [[ -z "${YOLO_WEIGHTS:-}" ]]; then
    weight_candidates=(
        "$ROOT_DIR/weights/digit_mixed_best.pt"
        "$ROOT_DIR/weights/best.pt"
        "$ROOT_DIR/yolo/runs/detect/digit_mixed_big/weights/best.pt"
        "$ROOT_DIR/yolo/weights/digit_mixed_best.pt"
    )
    for candidate in "${weight_candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            YOLO_WEIGHTS="$candidate"
            break
        fi
    done
    YOLO_WEIGHTS="${YOLO_WEIGHTS:-${weight_candidates[0]}}"
fi

YOLO_CONF="${YOLO_CONF:-0.35}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_FRAME_STRIDE="${YOLO_FRAME_STRIDE:-1}"
YOLO_HALF="${YOLO_HALF:-0}"
RUN_YOLO="${RUN_YOLO:-1}"
RUN_MNIST="${RUN_MNIST:-1}"
REUSE_PGM="${REUSE_PGM:-0}"

EVENT_MISSING_FRAMES="${EVENT_MISSING_FRAMES:-8}"
EVENT_NEW_IOU="${EVENT_NEW_IOU:-0.15}"
EVENT_NEW_CENTER="${EVENT_NEW_CENTER:-0.08}"
EVENT_CHANGE_FRAMES="${EVENT_CHANGE_FRAMES:-2}"
EVENT_COOLDOWN_FRAMES="${EVENT_COOLDOWN_FRAMES:-20}"
EVENT_PRESENT_FRAMES="${EVENT_PRESENT_FRAMES:-3}"
EVENT_CONFIRM_CONF="${EVENT_CONFIRM_CONF:-0.8}"
EVENT_CONFIRM_FRAMES="${EVENT_CONFIRM_FRAMES:-3}"
EVENT_SAVE_DELAY_FRAMES="${EVENT_SAVE_DELAY_FRAMES:-12}"
MIN_BOX_AREA_RATIO="${MIN_BOX_AREA_RATIO:-0.02}"
MAX_BOX_AREA_RATIO="${MAX_BOX_AREA_RATIO:-0.75}"
EDGE_MARGIN_RATIO="${EDGE_MARGIN_RATIO:-0.06}"

# Demo answer sequence. Override without editing the file, e.g.:
# ANSWERS="6 6 1 5" bash mnistCUDNN/run_pgm_all.sh
DEFAULT_ANSWERS=(6 6 6 6 1 5 6 8 6 3 8 6)
if [[ -n "${ANSWERS:-}" ]]; then
    read -r -a answers <<< "$ANSWERS"
else
    answers=("${DEFAULT_ANSWERS[@]}")
fi
EVAL_COUNT="${EVAL_COUNT:-${#answers[@]}}"

require_file() {
    local path="$1"
    local message="$2"
    if [[ ! -f "$path" ]]; then
        echo "$message: $path" >&2
        exit 1
    fi
}

if [[ "$REUSE_PGM" == "1" ]] && find "$PGM_DIR" -type f -name "*.pgm" -print -quit 2>/dev/null | grep -q .; then
    echo "Reusing existing PGM files in: $PGM_DIR"
    RUN_YOLO=0
fi

if [[ "$RUN_YOLO" == "1" ]]; then
    require_file "$YOLO_SCRIPT" "YOLO script not found"
    require_file "$YOLO_WEIGHTS" "YOLO weights not found. Copy digit_mixed_best.pt into weights/"
    require_file "$VIDEO" "Input video not found"

    yolo_args=(
        "$YOLO_SCRIPT"
        --video "$VIDEO"
        --weights "$YOLO_WEIGHTS"
        --output "$PGM_DIR"
        --conf "$YOLO_CONF"
        --device "$YOLO_DEVICE"
        --imgsz "$YOLO_IMGSZ"
        --frame-stride "$YOLO_FRAME_STRIDE"
        --target-count "$EVAL_COUNT"
        --clear-output
        --event-mode box
        --event-missing-frames "$EVENT_MISSING_FRAMES"
        --event-new-iou "$EVENT_NEW_IOU"
        --event-new-center "$EVENT_NEW_CENTER"
        --event-change-frames "$EVENT_CHANGE_FRAMES"
        --event-cooldown-frames "$EVENT_COOLDOWN_FRAMES"
        --event-present-frames "$EVENT_PRESENT_FRAMES"
        --event-confirm-conf "$EVENT_CONFIRM_CONF"
        --event-confirm-frames "$EVENT_CONFIRM_FRAMES"
        --event-save-delay-frames "$EVENT_SAVE_DELAY_FRAMES"
        --min-box-area-ratio "$MIN_BOX_AREA_RATIO"
        --max-box-area-ratio "$MAX_BOX_AREA_RATIO"
        --edge-margin-ratio "$EDGE_MARGIN_RATIO"
    )
    if [[ "$YOLO_HALF" == "1" ]]; then
        yolo_args+=(--half)
    fi
    "$PYTHON_BIN" "${yolo_args[@]}"
fi

if [[ "$RUN_MNIST" != "1" ]]; then
    exit 0
fi

if [[ ! -x "$EXE" ]]; then
    make -C "$SCRIPT_DIR"
fi

mapfile -t images < <(find "$PGM_DIR" -type f -name "*.pgm" | sort -V | head -n "$EVAL_COUNT")

if [[ "${#images[@]}" -eq 0 ]]; then
    echo "No PGM files found in: $PGM_DIR" >&2
    echo "YOLO did not save any PGM files, or PGM_DIR points to the wrong directory." >&2
    exit 1
fi

if [[ "${#images[@]}" -lt "$EVAL_COUNT" ]]; then
    echo "Warning: expected $EVAL_COUNT PGM files, found ${#images[@]} in $PGM_DIR" >&2
fi

total=0
correct=0
total_infer_ms=0

for img in "${images[@]}"; do
    echo "================================"
    if [[ "$img" == "$SCRIPT_DIR/"* ]]; then
        display_img="${img#$SCRIPT_DIR/}"
    else
        display_img="${img#$ROOT_DIR/}"
    fi
    echo "INPUT: $display_img"

    output=$(cd "$SCRIPT_DIR" && "$EXE" image="$img")
    result=$(printf '%s\n' "$output" | awk '/Result of classification/ {print $4; exit}')
    infer_ms=$(printf '%s\n' "$output" | awk '/Inference time:/ {print $3; exit}')

    gt="${answers[$total]:-NA}"

    echo "정답: $gt , 추론: $result"
    echo "Inference time: $infer_ms ms"

    if [[ "$result" == "$gt" ]]; then
        echo "결과: O"
        correct=$((correct+1))
    else
        echo "결과: X"
    fi

    if [[ "$infer_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_infer_ms=$(awk -v a="$total_infer_ms" -v b="$infer_ms" 'BEGIN { printf "%.6f", a + b }')
    fi

    total=$((total+1))
done

echo
echo "============= SUMMARY ============="
echo "Total Images              : $total"
echo "Correct Predictions       : $correct"
echo "Total Inference Time      : $total_infer_ms ms"
echo "==================================="
