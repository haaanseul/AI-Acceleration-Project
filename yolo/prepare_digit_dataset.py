#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-class YOLO dataset from handwritten digit zip files."
    )
    parser.add_argument("--zip", action="append", dest="zips", required=True, help="Input zip file.")
    parser.add_argument("--output", default="digit_dataset", help="Output dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-size", type=int, default=1280, help="Resize long side for training images.")
    parser.add_argument("--pad-ratio", type=float, default=0.08, help="Extra padding around detected foreground.")
    parser.add_argument("--min-area-ratio", type=float, default=0.0002)
    parser.add_argument("--clear-output", action="store_true", help="Remove existing output dataset first.")
    parser.add_argument("--mnist-cache", default="mnist.npz", help="Local cache path for downloaded MNIST npz.")
    parser.add_argument("--mnist-digits", default="", help="Comma-separated MNIST digits to synthesize, e.g. 1,3,5.")
    parser.add_argument("--mnist-count-per-digit", type=int, default=60)
    parser.add_argument("--mnist-size", default="960x1280", help="Synthetic image size as HEIGHTxWIDTH.")
    parser.add_argument("--mnist-height-ratio-min", type=float, default=0.45, help="Minimum synthetic MNIST digit height relative to image height.")
    parser.add_argument("--mnist-height-ratio-max", type=float, default=0.70, help="Maximum synthetic MNIST digit height relative to image height.")
    parser.add_argument("--negative-count", type=int, default=0, help="Add blank/line images with empty labels.")
    parser.add_argument(
        "--full-image-box",
        action="store_true",
        help="Use the full image as the label box if foreground detection is unreliable.",
    )
    return parser.parse_args()


def read_zip_images(zip_paths: list[str]) -> list[tuple[str, np.ndarray]]:
    images: list[tuple[str, np.ndarray]] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(IMAGE_EXTS):
                    continue
                data = np.frombuffer(zf.read(name), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"skip unreadable image: {zip_path}:{name}")
                    continue
                stem = Path(zip_path).stem + "_" + Path(name).stem
                images.append((stem, image))
    return images


def resize_long_side(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_size:
        return image
    scale = max_size / float(long_side)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def foreground_box(image: np.ndarray, pad_ratio: float, min_area_ratio: float) -> tuple[float, float, float, float] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect dark ink only. Including the bright paper mask makes YOLO learn full-frame boxes.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    useful_rects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * min_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area > image_area * 0.85:
            continue
        useful_rects.append((x, y, x + w, y + h))

    if not useful_rects:
        return None

    x1 = min(rect[0] for rect in useful_rects)
    y1 = min(rect[1] for rect in useful_rects)
    x2 = max(rect[2] for rect in useful_rects)
    y2 = max(rect[3] for rect in useful_rects)
    height, width = image.shape[:2]
    pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    return x1, y1, x2, y2


def yolo_label(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        height_text, width_text = size_text.lower().split("x", 1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise SystemExit("--mnist-size must look like HEIGHTxWIDTH, for example 960x1280") from exc
    if height < 128 or width < 128:
        raise SystemExit("--mnist-size is too small for YOLO synthetic images")
    return height, width


def parse_digits(digits_text: str) -> list[int]:
    if not digits_text.strip():
        return []
    digits = []
    for item in digits_text.split(","):
        item = item.strip()
        if not item:
            continue
        digit = int(item)
        if digit < 0 or digit > 9:
            raise SystemExit(f"Invalid MNIST digit: {digit}")
        digits.append(digit)
    return digits


def download_mnist(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    print(f"downloading MNIST: {url}")
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def load_mnist_samples(cache_path: Path, digits: list[int]) -> dict[int, np.ndarray]:
    path = download_mnist(cache_path)
    data = np.load(path)
    images = data["x_train"]
    labels = data["y_train"]
    samples: dict[int, np.ndarray] = {}
    for digit in digits:
        digit_images = images[labels == digit]
        filtered = [image for image in digit_images if valid_mnist_sample(image)]
        samples[digit] = np.asarray(filtered if filtered else digit_images)
        if len(samples[digit]) == 0:
            raise SystemExit(f"No MNIST samples found for digit {digit}")
    return samples


def valid_mnist_sample(image: np.ndarray) -> bool:
    mask = image > 20
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return False
    if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= 26 or ys.max() >= 26:
        return False
    area = len(xs)
    return 12 <= area <= 260


def synthetic_background(height: int, width: int) -> np.ndarray:
    base = random.randint(210, 238)
    background = np.full((height, width, 3), base, dtype=np.uint8)
    noise = np.random.normal(0, random.uniform(2.0, 7.0), background.shape).astype(np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(background, (3, 3), 0)


def synthesize_negative_image(height: int, width: int, index: int) -> np.ndarray:
    image = synthetic_background(height, width)
    kind = index % 4
    if kind == 1:
        x1 = random.randint(width // 5, width * 4 // 5)
        y1 = random.randint(height // 5, height * 4 // 5)
        x2 = min(width - 1, max(0, x1 + random.randint(-width // 4, width // 4)))
        y2 = min(height - 1, max(0, y1 + random.randint(-height // 4, height // 4)))
        cv2.line(image, (x1, y1), (x2, y2), (random.randint(5, 45),) * 3, random.randint(3, 9))
    elif kind == 2:
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
    elif kind == 3:
        overlay = image.copy()
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(width // 2, width)
        y2 = random.randint(height // 2, height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (random.randint(150, 205),) * 3, -1)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    return image


def synthesize_mnist_image(
    digit_image: np.ndarray,
    height: int,
    width: int,
    height_ratio_min: float,
    height_ratio_max: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    mask = digit_image > 20
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return synthetic_background(height, width), (0.0, 0.0, float(width), float(height))

    crop = digit_image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    crop = np.pad(crop, ((5, 5), (5, 5)), mode="constant", constant_values=0)
    ratio_min = max(0.08, min(height_ratio_min, height_ratio_max))
    ratio_max = min(0.90, max(height_ratio_min, height_ratio_max))
    target_h = random.randint(int(height * ratio_min), int(height * ratio_max))
    scale = target_h / max(1, crop.shape[0])
    target_w = max(8, int(round(crop.shape[1] * scale)))
    target_h = max(8, int(round(crop.shape[0] * scale)))

    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    alpha = resized.astype(np.float32) / 255.0

    angle = random.uniform(-10.0, 10.0)
    center = (target_w / 2.0, target_h / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rotation[0, 0])
    sin = abs(rotation[0, 1])
    rot_w = int(target_h * sin + target_w * cos)
    rot_h = int(target_h * cos + target_w * sin)
    rotation[0, 2] += rot_w / 2.0 - center[0]
    rotation[1, 2] += rot_h / 2.0 - center[1]
    alpha = cv2.warpAffine(alpha, rotation, (rot_w, rot_h), flags=cv2.INTER_LINEAR, borderValue=0)

    background = synthetic_background(height, width)
    margin = max(32, int(min(height, width) * 0.04))
    max_x = max(margin, width - rot_w - margin)
    max_y = max(margin, height - rot_h - margin)
    x = random.randint(margin, max_x)
    y = random.randint(margin, max_y)

    ink = random.randint(10, 45)
    roi = background[y : y + rot_h, x : x + rot_w].astype(np.float32)
    alpha_3 = alpha[:, :, None]
    roi = roi * (1.0 - alpha_3) + ink * alpha_3
    background[y : y + rot_h, x : x + rot_w] = np.clip(roi, 0, 255).astype(np.uint8)

    placed_mask = alpha > 0.08
    ys, xs = np.where(placed_mask)
    if len(xs) == 0:
        box = (float(x), float(y), float(x + rot_w), float(y + rot_h))
    else:
        x1 = max(0, x + int(xs.min()) - 8)
        y1 = max(0, y + int(ys.min()) - 8)
        x2 = min(width, x + int(xs.max()) + 9)
        y2 = min(height, y + int(ys.max()) + 9)
        box = (float(x1), float(y1), float(x2), float(y2))

    if random.random() < 0.25:
        background = cv2.GaussianBlur(background, (3, 3), 0)
    return background, box


def add_mnist_synthetic(
    dataset_dir: Path,
    cache_path: Path,
    digits: list[int],
    count_per_digit: int,
    val_ratio: float,
    size_text: str,
    height_ratio_min: float,
    height_ratio_max: float,
    start_index: int,
) -> int:
    if not digits:
        return 0

    height, width = parse_size(size_text)
    samples = load_mnist_samples(cache_path, digits)
    total = 0
    for digit in digits:
        digit_samples = samples[digit]
        for i in range(count_per_digit):
            sample = digit_samples[random.randrange(len(digit_samples))]
            image, box = synthesize_mnist_image(sample, height, width, height_ratio_min, height_ratio_max)
            split = "val" if random.random() < val_ratio else "train"
            out_name = f"mnist_{digit}_{start_index + total:04d}.jpg"
            image_path = dataset_dir / "images" / split / out_name
            label_path = dataset_dir / "labels" / split / out_name.replace(".jpg", ".txt")
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            label_path.write_text(yolo_label(box, width, height), encoding="utf-8")
            total += 1
    return total


def add_negative_images(
    dataset_dir: Path,
    count: int,
    val_ratio: float,
    size_text: str,
    start_index: int,
) -> int:
    if count <= 0:
        return 0

    height, width = parse_size(size_text)
    for i in range(count):
        image = synthesize_negative_image(height, width, i)
        split = "val" if random.random() < val_ratio else "train"
        out_name = f"negative_{start_index + i:04d}.jpg"
        image_path = dataset_dir / "images" / split / out_name
        label_path = dataset_dir / "labels" / split / out_name.replace(".jpg", ".txt")
        cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        label_path.write_text("", encoding="utf-8")
    return count


def write_yaml(dataset_dir: Path) -> None:
    try:
        dataset_path = dataset_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        dataset_path = dataset_dir.name
    yaml_text = (
        f"path: {dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: digit\n"
    )
    (dataset_dir / "digit_dataset.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.output)
    if args.clear_output and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = read_zip_images(args.zips)
    if not images:
        raise SystemExit("No images found in input zip files.")
    random.shuffle(images)

    val_count = max(1, int(round(len(images) * args.val_ratio)))
    val_names = {name for name, _ in images[:val_count]}

    written = 0
    fallback = 0
    for index, (name, image) in enumerate(images):
        image = resize_long_side(image, args.max_size)
        height, width = image.shape[:2]

        box = foreground_box(image, args.pad_ratio, args.min_area_ratio)
        if box is None:
            if not args.full_image_box:
                print(f"skip no foreground: {name}")
                continue
            box = (0.0, 0.0, float(width), float(height))
            fallback += 1

        split = "val" if name in val_names else "train"
        out_name = f"digit_{index:04d}.jpg"
        image_path = dataset_dir / "images" / split / out_name
        label_path = dataset_dir / "labels" / split / out_name.replace(".jpg", ".txt")
        cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        label_path.write_text(yolo_label(box, width, height), encoding="utf-8")
        written += 1

    write_yaml(dataset_dir)
    mnist_written = add_mnist_synthetic(
        dataset_dir=dataset_dir,
        cache_path=Path(args.mnist_cache),
        digits=parse_digits(args.mnist_digits),
        count_per_digit=args.mnist_count_per_digit,
        val_ratio=args.val_ratio,
        size_text=args.mnist_size,
        height_ratio_min=args.mnist_height_ratio_min,
        height_ratio_max=args.mnist_height_ratio_max,
        start_index=written,
    )
    negative_written = add_negative_images(
        dataset_dir=dataset_dir,
        count=args.negative_count,
        val_ratio=args.val_ratio,
        size_text=args.mnist_size,
        start_index=written + mnist_written,
    )
    print(f"images written: {written}")
    print(f"mnist synthetic images written: {mnist_written}")
    print(f"negative images written: {negative_written}")
    print(f"full-image fallback labels: {fallback}")
    print(f"dataset yaml: {dataset_dir / 'digit_dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
