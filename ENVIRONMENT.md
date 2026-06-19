# Environment Setup

This project keeps source code in git, but does not commit local Python
environments, datasets, videos, trained weights, or generated PGM files.

## What To Commit

- Source code and shell scripts
- `mnistCUDNN/Makefile`
- `weights/.gitkeep`
- Environment notes and requirements files

## What Not To Commit

- `.venv/`, `venv/`
- `digit_dataset*/`
- `weights/*.pt`
- `weights/*.engine`
- `*.mp4`
- `pgm_output/`
- `torchvision/` source build folder

These are already covered by `.gitignore`.

## Shared Server: YOLO Training

Use a virtual environment on the training server.

```bash
cd ~/AI-Acceleration-Project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-train.txt
```

Install a CUDA-compatible PyTorch build separately if the server does not
already have a working one. Check it before training:

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Train:

```bash
python3 yolo/train_yolo.py \
  --data digit_dataset_mixed_big/digit_dataset.yaml \
  --model yolov8n.pt \
  --name digit_mixed_big \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Copy the trained weight to the standard runtime path:

```bash
mkdir -p weights
cp yolo/runs/detect/digit_mixed_big/weights/best.pt weights/digit_mixed_best.pt
```

## Jetson: Demo Runtime

Do not install normal desktop PyTorch from PyPI on Jetson. Use the NVIDIA
Jetson PyTorch wheel that matches JetPack, then install the other packages
without replacing torch.

Check the board:

```bash
cat /etc/nv_tegra_release
python3 --version
python3 - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.is_available())
PY
```

If using a virtual environment on Jetson, create it with system packages so it
can see the NVIDIA torch install:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install runtime Python packages that are safe to manage with pip:

```bash
python3 -m pip install --user -r requirements-jetson.txt
```

Install `ultralytics` without dependencies so pip does not replace the NVIDIA
torch/torchvision packages:

```bash
python3 -m pip install --user --no-deps ultralytics==8.4.61
```

If `cv2` is missing on Jetson, prefer the Ubuntu package:

```bash
sudo apt update
sudo apt install -y python3-opencv
```

Put files in the standard locations:

```bash
cp /path/to/digit_mixed_best.pt weights/digit_mixed_best.pt
cp /path/to/number.mp4 ./number.mp4
```

Run preview:

```bash
./preview_yolo.sh
```

Run full demo:

```bash
./run_pgm_all.sh
```

Override answer labels when the demo sequence changes:

```bash
ANSWERS="6 6 1 5 8" ./run_pgm_all.sh
```
