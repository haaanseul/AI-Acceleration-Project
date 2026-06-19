#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an enlarged PNG contact sheet from generated 28x28 PGM files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="", help="Directory containing .pgm files.")
    parser.add_argument("--output", default="", help="Output PNG path.")
    parser.add_argument("--cols", type=int, default=6, help="Grid columns.")
    parser.add_argument("--scale", type=int, default=12, help="Pixel scale factor.")
    parser.add_argument("--gap", type=int, default=10, help="Gap between images in output pixels.")
    parser.add_argument("--invert", action="store_true", help="Invert black/white for display.")
    return parser.parse_args()


def default_input_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    preferred = root / "mnistCUDNN" / "pgm_output"
    if preferred.exists():
        return preferred
    return root / "pgm_output"


def read_token(data: bytes, offset: int) -> tuple[bytes, int]:
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    if offset < len(data) and data[offset] == ord("#"):
        while offset < len(data) and data[offset] not in b"\r\n":
            offset += 1
        return read_token(data, offset)
    start = offset
    while offset < len(data) and data[offset] not in b" \t\r\n":
        offset += 1
    return data[start:offset], offset


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    magic, offset = read_token(data, 0)
    if magic != b"P5":
        raise ValueError(f"{path} is not a binary PGM(P5) file")
    width_token, offset = read_token(data, offset)
    height_token, offset = read_token(data, offset)
    maxval_token, offset = read_token(data, offset)
    width = int(width_token)
    height = int(height_token)
    maxval = int(maxval_token)
    if maxval != 255:
        raise ValueError(f"{path} uses unsupported max value {maxval}")
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    pixels = data[offset : offset + width * height]
    if len(pixels) != width * height:
        raise ValueError(f"{path} has invalid pixel data size")
    return width, height, pixels


def write_png_gray(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        start = y * width
        raw.extend(pixels[start : start + width])

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(png)


def paste_scaled(
    canvas: bytearray,
    canvas_width: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    pixels: bytes,
    scale: int,
    invert: bool,
) -> None:
    for y in range(height):
        for x in range(width):
            value = pixels[y * width + x]
            if invert:
                value = 255 - value
            for sy in range(scale):
                row = (y0 + y * scale + sy) * canvas_width
                start = row + x0 + x * scale
                canvas[start : start + scale] = bytes([value]) * scale


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input) if args.input else default_input_dir()
    output_path = Path(args.output) if args.output else input_dir / "pgm_preview.png"
    pgm_paths = sorted(input_dir.glob("*.pgm"))
    if not pgm_paths:
        raise SystemExit(f"No PGM files found in: {input_dir}")

    images = [(path, *read_pgm(path)) for path in pgm_paths]
    cell_width = max(width for _, width, _, _ in images) * args.scale
    cell_height = max(height for _, _, height, _ in images) * args.scale
    cols = max(1, args.cols)
    rows = math.ceil(len(images) / cols)
    canvas_width = cols * cell_width + (cols + 1) * args.gap
    canvas_height = rows * cell_height + (rows + 1) * args.gap
    canvas = bytearray([235] * (canvas_width * canvas_height))

    for index, (_path, width, height, pixels) in enumerate(images):
        col = index % cols
        row = index // cols
        x0 = args.gap + col * (cell_width + args.gap)
        y0 = args.gap + row * (cell_height + args.gap)
        paste_scaled(canvas, canvas_width, x0, y0, width, height, pixels, args.scale, args.invert)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png_gray(output_path, canvas_width, canvas_height, canvas)
    print(f"PGM preview image: {output_path}")
    print(f"PGM files: {len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
