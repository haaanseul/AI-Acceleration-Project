#!/bin/bash
set -euo pipefail

PGM_DIR="${PGM_DIR:-pgm_output}"
EXE="${EXE:-./mnistCUDNN}"
VIDEO="${VIDEO:-../number.mp4}"
YOLO_SCRIPT="${YOLO_SCRIPT:-../yolo/extract_pgm.py}"
YOLO_ENGINE_DEFAULT="../yolo/runs/detect/digit/weights/best.engine"
YOLO_PT_DEFAULT="../yolo/runs/detect/digit/weights/best.pt"
if [[ -z "${YOLO_WEIGHTS:-}" ]]; then
    if [[ -f "$YOLO_ENGINE_DEFAULT" ]]; then
        YOLO_WEIGHTS="$YOLO_ENGINE_DEFAULT"
    else
        YOLO_WEIGHTS="$YOLO_PT_DEFAULT"
    fi
fi
YOLO_CONF="${YOLO_CONF:-0.35}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_FRAME_STRIDE="${YOLO_FRAME_STRIDE:-2}"
YOLO_HALF="${YOLO_HALF:-0}"
RUN_YOLO="${RUN_YOLO:-1}"
PIPELINE_START_MS=$(date +%s%3N)

# Replace this array with the answer labels announced for the demo video.
answers=(6 6 6 6 1 5 6 8 6 3 8 6)
EVAL_COUNT="${EVAL_COUNT:-${#answers[@]}}"

if [[ "$RUN_YOLO" == "1" ]]; then
    if [[ ! -f "$YOLO_SCRIPT" ]]; then
        echo "YOLO script not found: $YOLO_SCRIPT" >&2
        exit 1
    fi
    if [[ ! -f "$YOLO_WEIGHTS" ]]; then
        echo "YOLO weights not found: $YOLO_WEIGHTS" >&2
        echo "Train on the shared server first, then copy best.pt to this path." >&2
        exit 1
    fi

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
    )
    if [[ "$YOLO_HALF" == "1" ]]; then
        yolo_args+=(--half)
    fi
    python3 "${yolo_args[@]}"
fi

if [[ ! -x "$EXE" ]]; then
    make
fi

mapfile -t images < <(find "$PGM_DIR" -type f -name "*.pgm" | sort -V | head -n "$EVAL_COUNT")

total=0
correct=0
total_infer_ms=0

for img in "${images[@]}"; do
    echo "================================"
    echo "INPUT: $img"

    output=$("$EXE" image="$img")
    result=$(printf '%s\n' "$output" | awk '/Result of classification/ {print $4; exit}')
    infer_ms=$(printf '%s\n' "$output" | awk '/Inference time:/ {print $3; exit}')

    gt="${answers[$total]:-NA}"

    echo "GT: $gt , Prediction: $result"
    echo "Inference time: $infer_ms ms"

    if [[ "$result" == "$gt" ]]; then
        echo "Result: O"
        correct=$((correct+1))
    else
        echo "Result: X"
    fi

    if [[ "$infer_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        total_infer_ms=$(awk -v a="$total_infer_ms" -v b="$infer_ms" 'BEGIN { printf "%.6f", a + b }')
    fi

    total=$((total+1))
done

if [[ "$total" -gt 0 ]]; then
    total_infer_sec=$(awk -v a="$total_infer_ms" 'BEGIN { printf "%.6f", a / 1000.0 }')
    avg_infer_ms=$(awk -v a="$total_infer_ms" -v n="$total" 'BEGIN { printf "%.6f", a / n }')
else
    total_infer_sec=0
    avg_infer_ms=0
fi

echo
echo "============= SUMMARY ============="
echo "Total Images              : $total"
echo "Correct Predictions       : $correct"
echo "Total Inference Time      : $total_infer_ms ms"
echo "Total Inference Time      : $total_infer_sec sec"
echo "Average Inference / Image : $avg_infer_ms ms"
PIPELINE_END_MS=$(date +%s%3N)
PIPELINE_TOTAL_MS=$((PIPELINE_END_MS - PIPELINE_START_MS))
PIPELINE_TOTAL_SEC=$(awk -v a="$PIPELINE_TOTAL_MS" 'BEGIN { printf "%.6f", a / 1000.0 }')
echo "Pipeline Total Time       : $PIPELINE_TOTAL_MS ms"
echo "Pipeline Total Time       : $PIPELINE_TOTAL_SEC sec"
echo "==================================="
