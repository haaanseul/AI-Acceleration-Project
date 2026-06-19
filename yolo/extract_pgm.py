#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Candidate:
    frame_index: int
    cls_id: int
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]
    pgm: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect handwritten digits with YOLO and save one 28x28 PGM per appearance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default="number.mp4", help="Input video path.")
    parser.add_argument("--weights", default="", help="YOLO weights path, .pt.")
    parser.add_argument("--output", default="pgm_output", help="Output directory for PGM files.")
    parser.add_argument("--manifest", default="", help="Optional CSV manifest path.")
    parser.add_argument("--device", default="0", help="Ultralytics device value: 0, 1, cpu, etc.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--half", action="store_true", help="Use FP16 PyTorch inference where supported.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run YOLO every N frames.")
    parser.add_argument("--target-count", type=int, default=0, help="Stop after this many PGM files.")
    parser.add_argument("--clear-output", action="store_true", help="Remove old .pgm files first.")
    parser.add_argument("--event-mode", default="box", choices=("box",), help="Kept for command compatibility; only box mode is supported.")
    parser.add_argument("--min-event-frames", type=int, default=2, help="Minimum detections per saved event.")
    parser.add_argument("--end-missing-frames", type=int, default=8, help="Missing detections to end an event.")
    parser.add_argument("--event-missing-frames", type=int, help="Alias for --end-missing-frames.")
    parser.add_argument("--new-event-iou", "--event-new-iou", dest="new_event_iou", type=float, default=0.15)
    parser.add_argument("--new-event-center", "--event-new-center", dest="new_event_center", type=float, default=0.08)
    parser.add_argument("--event-change-frames", type=int, default=2)
    parser.add_argument("--event-cooldown-frames", type=int, default=20)
    parser.add_argument("--event-present-frames", type=int, default=3)
    parser.add_argument("--event-confirm-conf", type=float, default=0.8)
    parser.add_argument("--event-confirm-frames", type=int, default=3)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.02)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.75)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.06)
    parser.add_argument("--pad", type=float, default=0.05, help="Box padding ratio before preprocessing.")
    return parser.parse_args()


def sanitize_label(label: str, cls_id: int) -> str:
    label = str(label)
    if re.fullmatch(r"[0-9]", label):
        return label
    return str(cls_id)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def center_distance_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    frame_shape: tuple[int, int, int],
) -> float:
    height, width = frame_shape[:2]
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    diag = max(1.0, float((width * width + height * height) ** 0.5))
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / diag)


def preprocess_digit(frame: np.ndarray, xyxy: tuple[float, float, float, float], pad_ratio: float) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    pad = pad_ratio * max(x2 - x1, y2 - y1)
    x1 = max(0, int(round(x1 - pad)))
    y1 = max(0, int(round(y1 - pad)))
    x2 = min(width, int(round(x2 + pad)))
    y2 = min(height, int(round(y2 + pad)))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((28, 28), dtype=np.uint8), 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    quality = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = 255 - binary

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28, 28), dtype=np.uint8), quality

    digit = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = max(digit.shape[:2])
    square = np.zeros((side, side), dtype=np.uint8)
    oy = (side - digit.shape[0]) // 2
    ox = (side - digit.shape[1]) // 2
    square[oy : oy + digit.shape[0], ox : ox + digit.shape[1]] = digit

    target_side = 20
    resized = cv2.resize(square, (target_side, target_side), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    offset = (28 - target_side) // 2
    canvas[offset : offset + target_side, offset : offset + target_side] = resized
    return canvas, quality


def write_pgm(path: Path, image: np.ndarray) -> None:
    if image.shape != (28, 28):
        raise ValueError(f"Expected 28x28 image, got {image.shape}")
    path.write_bytes(b"P5\n28 28\n255\n" + image.astype(np.uint8).tobytes())


def best_detection(model, frame: np.ndarray, args: argparse.Namespace) -> Candidate | None:
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
        "half": args.half,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    result = model.predict(frame, **predict_kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    height, width = frame.shape[:2]
    frame_area = max(1.0, float(width * height))
    scores: list[float] = []

    for conf, box in zip(confs, xyxy):
        x1, y1, x2, y2 = box
        area_ratio = max(0.0, float((x2 - x1) * (y2 - y1)) / frame_area)
        cx = (x1 + x2) * 0.5 / max(1.0, width)
        cy = (y1 + y2) * 0.5 / max(1.0, height)
        near_edge = (
            cx < args.edge_margin_ratio
            or cx > 1.0 - args.edge_margin_ratio
            or cy < args.edge_margin_ratio
            or cy > 1.0 - args.edge_margin_ratio
        )
        if area_ratio < args.min_box_area_ratio or area_ratio > args.max_box_area_ratio or near_edge:
            scores.append(-1.0)
        else:
            scores.append(float(conf) * (area_ratio ** 0.35))

    index = int(np.argmax(scores))
    if scores[index] < 0:
        return None

    cls_id = int(cls_ids[index])
    names = getattr(result, "names", {}) or {}
    label = sanitize_label(names.get(cls_id, str(cls_id)), cls_id)
    box = tuple(float(v) for v in xyxy[index])
    pgm, quality = preprocess_digit(frame, box, args.pad)
    x1, y1, x2, y2 = box
    area_ratio = max(0.0, float((x2 - x1) * (y2 - y1)) / frame_area)
    cx = (x1 + x2) * 0.5 / max(1.0, width)
    cy = (y1 + y2) * 0.5 / max(1.0, height)
    center_bias = 1.0 - min(1.0, abs(cx - 0.5) + abs(cy - 0.5))
    score = float(confs[index]) + 0.15 * center_bias + 0.05 * min(area_ratio / 0.25, 1.0)
    return Candidate(
        frame_index=0,
        cls_id=cls_id,
        label=label,
        conf=float(confs[index]),
        xyxy=box,
        pgm=pgm,
        score=score,
    )


def find_weights(requested: str) -> Path:
    if requested:
        return Path(requested)

    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "weights" / "digit_mixed_best.pt",
        root / "weights" / "best.pt",
        root / "yolo" / "runs" / "detect" / "digit_mixed_big" / "weights" / "best.pt",
        root / "yolo" / "weights" / "digit_mixed_best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def flush_event(
    best: Candidate | None,
    output_dir: Path,
    event_index: int,
    rows: list[dict[str, str]],
) -> int:
    if best is None:
        return event_index
    filename = (
        f"frame_{best.frame_index:06d}_digit_{best.label}_"
        f"conf_{best.conf:.2f}_{event_index:04d}.pgm"
    )
    path = output_dir / filename
    write_pgm(path, best.pgm)
    rows.append(
        {
            "event": str(event_index),
            "frame": str(best.frame_index),
            "class_id": str(best.cls_id),
            "label": best.label,
            "confidence": f"{best.conf:.4f}",
            "file": str(path),
        }
    )
    print(f"saved: {path}")
    return event_index + 1


def main() -> int:
    args = parse_args()
    global cv2, np
    import cv2
    import numpy as np

    if args.event_missing_frames is not None:
        args.end_missing_frames = args.event_missing_frames

    video_path = Path(args.video)
    weights_path = find_weights(args.weights)
    output_dir = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.csv"

    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output:
        for old_pgm in output_dir.glob("*.pgm"):
            old_pgm.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it on the shared server or Jetson "
            "with `pip install ultralytics`."
        ) from exc

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}", file=sys.stderr)
        return 2

    active = False
    missing_frames = 0
    event_index = 0
    track: Candidate | None = None
    changed_frames = 0
    present_frames = 0
    confirm_frames = 0
    rows: list[dict[str, str]] = []
    frame_index = 0
    last_event_frame = -10**9

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if args.frame_stride > 1 and (frame_index - 1) % args.frame_stride != 0:
            continue

        current = best_detection(model, frame, args)
        if current is None:
            present_frames = 0
            confirm_frames = 0
            changed_frames = 0
            if active:
                missing_frames += 1
                if missing_frames >= args.end_missing_frames:
                    active = False
                    missing_frames = 0
                    track = None
            continue

        current.frame_index = frame_index
        missing_frames = 0
        present_frames += 1
        if current.conf >= args.event_confirm_conf:
            confirm_frames += 1
        else:
            confirm_frames = 0

        if not active:
            if (
                present_frames >= args.event_present_frames
                and confirm_frames >= args.event_confirm_frames
                and frame_index - last_event_frame >= args.event_cooldown_frames
            ):
                event_index = flush_event(current, output_dir, event_index, rows)
                last_event_frame = frame_index
                if args.target_count and event_index >= args.target_count:
                    break
                active = True
                track = current
                changed_frames = 0
            else:
                continue
        elif track is not None:
            center_shift = center_distance_ratio(track.xyxy, current.xyxy, frame.shape)
            low_iou = box_iou(track.xyxy, current.xyxy) < args.new_event_iou
            far_center = center_shift > args.new_event_center
            if far_center or (low_iou and center_shift > args.new_event_center * 0.5):
                changed_frames += 1
            else:
                changed_frames = 0
                track = current

            if changed_frames >= args.event_change_frames and confirm_frames >= args.event_confirm_frames:
                event_index = flush_event(current, output_dir, event_index, rows)
                last_event_frame = frame_index
                if args.target_count and event_index >= args.target_count:
                    break
                active = True
                track = current
                changed_frames = 0

    cap.release()

    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Total saved PGM files: {event_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
