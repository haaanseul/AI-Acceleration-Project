#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Missing dependency: ultralytics. Install it with `pip install ultralytics`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe YOLO false positives on blank/line/noise images.")
    parser.add_argument("--weights", required=True, help="YOLO weights path, .pt or .engine.")
    parser.add_argument("--output", default="false_positive_probe", help="Output directory.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def paper_background(height: int, width: int) -> np.ndarray:
    base = random.randint(210, 242)
    image = np.full((height, width, 3), base, dtype=np.uint8)
    noise = np.random.normal(0, random.uniform(2.0, 7.0), image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def make_negative_image(height: int, width: int, index: int) -> tuple[str, np.ndarray]:
    image = paper_background(height, width)
    kind = index % 4

    if kind == 0:
        name = "blank"
    elif kind == 1:
        name = "single_line"
        x1 = random.randint(width // 5, width * 4 // 5)
        y1 = random.randint(height // 5, height * 4 // 5)
        x2 = min(width - 1, max(0, x1 + random.randint(-width // 4, width // 4)))
        y2 = min(height - 1, max(0, y1 + random.randint(-height // 4, height // 4)))
        cv2.line(image, (x1, y1), (x2, y2), (random.randint(5, 45),) * 3, random.randint(3, 9))
    elif kind == 2:
        name = "short_marks"
        for _ in range(random.randint(2, 5)):
            x = random.randint(width // 6, width * 5 // 6)
            y = random.randint(height // 6, height * 5 // 6)
            cv2.line(
                image,
                (x, y),
                (x + random.randint(-40, 40), y + random.randint(-40, 40)),
                (random.randint(10, 70),) * 3,
                random.randint(2, 6),
            )
    else:
        name = "shadow"
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(width // 2, width)
        y2 = random.randint(height // 2, height)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (random.randint(150, 205),) * 3, -1)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

    return name, image


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}", file=sys.stderr)
        return 2

    output_dir = Path(args.output)
    clean_dir = output_dir / "clean"
    annotated_dir = output_dir / "annotated"
    clean_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    total_boxes = 0
    detected_images = 0

    for i in range(args.count):
        kind, image = make_negative_image(args.height, args.width, i)
        result = model.predict(
            image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        boxes = 0 if result.boxes is None else len(result.boxes)
        total_boxes += boxes
        if boxes:
            detected_images += 1

        filename = f"{i:04d}_{kind}_boxes_{boxes}.jpg"
        cv2.imwrite(str(clean_dir / filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(annotated_dir / filename), result.plot(), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"negative images: {args.count}")
    print(f"images with false positives: {detected_images}")
    print(f"total false-positive boxes: {total_boxes}")
    print(f"annotated output: {annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
