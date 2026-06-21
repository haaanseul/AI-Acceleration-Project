from pathlib import Path
import numpy as np
import cv2
import urllib.request
import shutil

OUT = Path("train_digits")
MNIST_CACHE = Path("mnist.npz")
DIGITS = [1, 3, 5, 6, 8]
COUNT_PER_DIGIT = 500

def download_mnist():
    if MNIST_CACHE.exists():
        return
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    urllib.request.urlretrieve(url, MNIST_CACHE)

def write_pgm(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P5\n28 28\n255\n" + img.astype(np.uint8).tobytes())

download_mnist()
data = np.load(MNIST_CACHE)
x = data["x_train"]
y = data["y_train"]

for d in DIGITS:
    out_dir = OUT / str(d)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = x[y == d][:COUNT_PER_DIGIT]

    for i, img in enumerate(imgs):
        # MNIST는 원래 검은 배경 + 흰 숫자라 그대로 사용
        write_pgm(out_dir / f"mnist_{d}_{i:04d}.pgm", img)

# 네 실제 6/8 PGM 추가
for d in [6, 8]:
    src_dir = Path("custom_digits_pgm") / str(d)
    dst_dir = OUT / str(d)

    if src_dir.exists():
        for i, p in enumerate(sorted(src_dir.glob("*.pgm"))):
            shutil.copy(p, dst_dir / f"real_{d}_{i:04d}.pgm")

print("done:", OUT)