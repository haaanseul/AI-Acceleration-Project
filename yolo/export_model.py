#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Missing dependency: ultralytics. Install it with `pip install ultralytics`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO weights for faster inference.")
    parser.add_argument(
        "--weights",
        default=str(Path(__file__).resolve().parent / "runs" / "detect" / "digit" / "weights" / "best.pt"),
        help="Trained .pt weights path.",
    )
    parser.add_argument(
        "--format",
        default="engine",
        choices=("engine", "onnx"),
        help="Export format. Use engine on Jetson for TensorRT.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="Export/use FP16 where supported.")
    parser.add_argument("--int8", action="store_true", help="Export INT8. Requires calibration data.")
    parser.add_argument("--data", default="", help="Dataset yaml for INT8 calibration.")
    parser.add_argument("--workspace", type=float, default=2.0, help="TensorRT workspace in GiB.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    if args.int8 and not args.data:
        raise SystemExit("INT8 export needs --data for calibration. Use --half first unless INT8 is tested.")

    model = YOLO(str(weights))
    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "workspace": args.workspace,
    }
    if args.data:
        export_args["data"] = args.data

    try:
        exported = model.export(**export_args)
    except TypeError:
        export_args.pop("workspace", None)
        exported = model.export(**export_args)

    print(f"Exported model: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
