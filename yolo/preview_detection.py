#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated preview video for YOLO digit detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default="number.mp4", help="Input video path.")
    parser.add_argument("--weights", default="", help="YOLO weights path, .pt.")
    parser.add_argument("--output", default="preview_yolo.mp4", help="Annotated output video path.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--event-mode", default="box", choices=("box",), help="Kept for command compatibility; only box mode is supported.")
    parser.add_argument("--event-missing-frames", type=int, default=8)
    parser.add_argument("--event-new-iou", type=float, default=0.15)
    parser.add_argument("--event-new-center", type=float, default=0.08)
    parser.add_argument("--event-change-frames", type=int, default=2)
    parser.add_argument("--event-cooldown-frames", type=int, default=20)
    parser.add_argument("--event-present-frames", type=int, default=3)
    parser.add_argument("--event-confirm-conf", type=float, default=0.8)
    parser.add_argument("--event-confirm-frames", type=int, default=3)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.02)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.75)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.06)
    return parser.parse_args()


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
    width: int,
    height: int,
) -> float:
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    diag = max(1.0, float((width * width + height * height) ** 0.5))
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / diag)


def best_box(
    result,
    width: int,
    height: int,
    min_area_ratio: float,
    max_area_ratio: float,
    edge_margin_ratio: float,
) -> tuple[tuple[float, float, float, float] | None, int, float]:
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0, 0.0

    confs = result.boxes.conf.detach().cpu().numpy()
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    frame_area = max(1.0, float(width * height))
    scores: list[float] = []
    valid_count = 0

    for conf, box in zip(confs, xyxy):
        x1, y1, x2, y2 = box
        area_ratio = max(0.0, float((x2 - x1) * (y2 - y1)) / frame_area)
        cx = (x1 + x2) * 0.5 / max(1.0, width)
        cy = (y1 + y2) * 0.5 / max(1.0, height)
        near_edge = (
            cx < edge_margin_ratio
            or cx > 1.0 - edge_margin_ratio
            or cy < edge_margin_ratio
            or cy > 1.0 - edge_margin_ratio
        )
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio or near_edge:
            scores.append(-1.0)
        else:
            valid_count += 1
            scores.append(float(conf) * (area_ratio ** 0.35))

    index = int(np.argmax(scores))
    if scores[index] < 0:
        return None, valid_count, 0.0
    return tuple(float(v) for v in xyxy[index]), valid_count, float(confs[index])


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


def main() -> int:
    args = parse_args()
    global cv2, np
    import cv2
    import numpy as np

    video_path = Path(args.video)
    weights_path = find_weights(args.weights)
    output_path = Path(args.output)

    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 2
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: ultralytics. Install it with `pip install ultralytics`.") from exc

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.frame_stride),
        (width, height),
    )

    total_frames = 0
    processed_frames = 0
    detected_frames = 0
    total_boxes = 0
    event_count = 0
    active_event = False
    missing_frames = 0
    track_box: tuple[float, float, float, float] | None = None
    changed_frames = 0
    present_frames = 0
    confirm_frames = 0
    last_event_frame = -10**9

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        if args.frame_stride > 1 and (total_frames - 1) % args.frame_stride != 0:
            continue
        if args.max_frames and processed_frames >= args.max_frames:
            break

        result = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        annotated = result.plot()
        raw_boxes = 0 if result.boxes is None else len(result.boxes)
        current_box, valid_boxes, current_conf = best_box(
            result,
            width,
            height,
            args.min_box_area_ratio,
            args.max_box_area_ratio,
            args.edge_margin_ratio,
        )
        if raw_boxes:
            detected_frames += 1
            total_boxes += raw_boxes

        debug_iou = 0.0
        debug_center = 0.0
        debug_reason = "-"

        if current_box is None:
            present_frames = 0
            confirm_frames = 0
            changed_frames = 0
            if active_event:
                missing_frames += 1
                if missing_frames >= args.event_missing_frames:
                    active_event = False
                    track_box = None
                    missing_frames = 0
        else:
            missing_frames = 0
            present_frames += 1
            if current_conf >= args.event_confirm_conf:
                confirm_frames += 1
            else:
                confirm_frames = 0

            if not active_event:
                if (
                    present_frames >= args.event_present_frames
                    and confirm_frames >= args.event_confirm_frames
                    and total_frames - last_event_frame >= args.event_cooldown_frames
                ):
                    event_count += 1
                    last_event_frame = total_frames
                    active_event = True
                    track_box = current_box
                    changed_frames = 0
                    debug_reason = "start"
            elif track_box is not None:
                center_shift = center_distance_ratio(track_box, current_box, width, height)
                iou = box_iou(track_box, current_box)
                low_iou = iou < args.event_new_iou
                far_center = center_shift > args.event_new_center
                debug_iou = iou
                debug_center = center_shift

                if far_center or (low_iou and center_shift > args.event_new_center * 0.5):
                    changed_frames += 1
                else:
                    changed_frames = 0
                    track_box = current_box

                if changed_frames >= args.event_change_frames and confirm_frames >= args.event_confirm_frames:
                    event_count += 1
                    last_event_frame = total_frames
                    track_box = current_box
                    changed_frames = 0
                    debug_reason = "jump"

        debug_lines = [
            f"frame {total_frames}  events {event_count}  reason {debug_reason}",
            f"raw {raw_boxes}  valid {valid_boxes}  conf {current_conf:.2f}  ok {confirm_frames}",
            f"present {present_frames}  missing {missing_frames}  changed {changed_frames}",
            f"iou {debug_iou:.2f}  center {debug_center:.3f}",
        ]
        panel_width = min(width - 16, 520)
        panel_height = 22 + 28 * len(debug_lines)
        cv2.rectangle(annotated, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
        for line_index, line in enumerate(debug_lines):
            cv2.putText(
                annotated,
                line,
                (18, 34 + 28 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(annotated)
        processed_frames += 1

    cap.release()
    writer.release()

    print(f"preview video: {output_path}")
    print(f"processed frames: {processed_frames}")
    print(f"detected frames: {detected_frames}")
    print(f"total boxes: {total_boxes}")
    print(f"estimated digit events: {event_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
