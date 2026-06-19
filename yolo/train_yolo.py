#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Missing dependency: ultralytics. Install it with `pip install ultralytics`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared-server YOLO training entry point.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset yaml path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or previous weights.")
    parser.add_argument("--project", default="", help="Training output directory.")
    parser.add_argument("--name", default="digit", help="Run name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml not found: {data_path}")

    project = args.project or str(Path(__file__).resolve().parent / "runs" / "detect")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        project=project,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
    )
    print(f"Best weights should be under: {Path(project) / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
