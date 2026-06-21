from pathlib import Path
import cv2
import numpy as np

INPUT_ROOT = Path("custom_digits")
OUTPUT_ROOT = Path("custom_digits_pgm")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.count_nonzero(binary) > binary.size // 2:
        binary = 255 - binary

    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    digit = binary[ys.min():ys.max()+1, xs.min():xs.max()+1]

    side = max(digit.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    yoff = (side - digit.shape[0]) // 2
    xoff = (side - digit.shape[1]) // 2
    square[yoff:yoff+digit.shape[0], xoff:xoff+digit.shape[1]] = digit

    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = resized

    return canvas

def write_pgm(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P5\n28 28\n255\n" + img.astype(np.uint8).tobytes())

count = 0

for label in ["6", "8"]:
    in_dir = INPUT_ROOT / label
    out_dir = OUTPUT_ROOT / label

    for img_path in sorted(in_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print("skip unreadable:", img_path)
            continue

        pgm = preprocess(img)
        if pgm is None:
            print("skip no digit:", img_path)
            continue

        out_path = out_dir / f"{label}_{count:04d}.pgm"
        write_pgm(out_path, pgm)
        count += 1

print("done, saved:", count)
print("output:", OUTPUT_ROOT)