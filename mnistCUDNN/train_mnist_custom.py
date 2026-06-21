from pathlib import Path
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

TRAIN_DIR = Path("train_digits")
MNIST_CACHE = Path("mnist.npz")

OLD_WEIGHT_DIR = Path("data_orig")
NEW_WEIGHT_DIR = Path("data")

DIGITS = [1, 3, 5, 6, 8]
MNIST_COUNT_PER_DIGIT = 3000

EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEW_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)


def read_pgm(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32) / 255.0


def download_mnist():
    if MNIST_CACHE.exists():
        return

    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    print("Downloading MNIST...")
    urllib.request.urlretrieve(url, MNIST_CACHE)


class CustomMNISTDataset(Dataset):
    def __init__(self):
        self.samples = []

        for digit in DIGITS:
            digit_dir = TRAIN_DIR / str(digit)
            if digit_dir.exists():
                for p in sorted(digit_dir.glob("*.pgm")):
                    self.samples.append(("pgm", p, digit))

        download_mnist()
        data = np.load(MNIST_CACHE)
        x_train = data["x_train"]
        y_train = data["y_train"]

        for digit in DIGITS:
            imgs = x_train[y_train == digit][:MNIST_COUNT_PER_DIGIT]
            for img in imgs:
                self.samples.append(("mnist", img, digit))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def augment(self, img: np.ndarray) -> np.ndarray:
        tx = np.random.randint(-2, 3)
        ty = np.random.randint(-2, 3)
        mat = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, mat, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)

        angle = np.random.uniform(-4, 4)
        rot = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
        img = cv2.warpAffine(img, rot, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)

        if np.random.rand() < 0.05:
            kernel = np.ones((2, 2), np.uint8)
            tmp = (img * 255).astype(np.uint8)
            if np.random.rand() < 0.5:
                tmp = cv2.dilate(tmp, kernel, iterations=1)
            else:
                tmp = cv2.erode(tmp, kernel, iterations=1)
            img = tmp.astype(np.float32) / 255.0

        return np.clip(img, 0.0, 1.0)

    def __getitem__(self, idx):
        kind, data, label = self.samples[idx]

        if kind == "pgm":
            img = read_pgm(data)
        else:
            img = data.astype(np.float32) / 255.0

        img = self.augment(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return img, label


class MnistCUDNNLike(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.relu = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(size=5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.lrn(x)
        x = x.squeeze(-1).squeeze(-1)

        x = self.fc2(x)
        return x


def load_bin(path: Path, shape):
    if not path.exists():
        return None

    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))

    if arr.size != expected:
        print(f"skip {path}: {arr.size} != {expected}")
        return None

    return torch.tensor(arr.reshape(shape), dtype=torch.float32)


def load_old_weights(model: MnistCUDNNLike):
    mapping = [
        ("conv1.weight", OLD_WEIGHT_DIR / "conv1.bin", (20, 1, 5, 5)),
        ("conv1.bias", OLD_WEIGHT_DIR / "conv1.bias.bin", (20,)),
        ("conv2.weight", OLD_WEIGHT_DIR / "conv2.bin", (50, 20, 5, 5)),
        ("conv2.bias", OLD_WEIGHT_DIR / "conv2.bias.bin", (50,)),
        ("fc1.weight", OLD_WEIGHT_DIR / "ip1.bin", (500, 800)),
        ("fc1.bias", OLD_WEIGHT_DIR / "ip1.bias.bin", (500,)),
        ("fc2.weight", OLD_WEIGHT_DIR / "ip2.bin", (10, 500)),
        ("fc2.bias", OLD_WEIGHT_DIR / "ip2.bias.bin", (10,)),
    ]

    state = model.state_dict()
    loaded = 0

    for key, path, shape in mapping:
        tensor = load_bin(path, shape)
        if tensor is not None:
            state[key] = tensor
            loaded += 1
            print(f"loaded {path}")

    model.load_state_dict(state)
    print(f"Loaded weights: {loaded}/8")


def save_bin(path: Path, tensor: torch.Tensor):
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(path)


def save_new_weights(model: MnistCUDNNLike):
    save_bin(NEW_WEIGHT_DIR / "conv1.bin", model.conv1.weight)
    save_bin(NEW_WEIGHT_DIR / "conv1.bias.bin", model.conv1.bias)

    save_bin(NEW_WEIGHT_DIR / "conv2.bin", model.conv2.weight)
    save_bin(NEW_WEIGHT_DIR / "conv2.bias.bin", model.conv2.bias)

    save_bin(NEW_WEIGHT_DIR / "ip1.bin", model.fc1.weight)
    save_bin(NEW_WEIGHT_DIR / "ip1.bias.bin", model.fc1.bias)

    save_bin(NEW_WEIGHT_DIR / "ip2.bin", model.fc2.weight)
    save_bin(NEW_WEIGHT_DIR / "ip2.bias.bin", model.fc2.bias)

    print(f"Saved new weights to {NEW_WEIGHT_DIR}")


def main():
    print("Device:", DEVICE)

    dataset = CustomMNISTDataset()

    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MnistCUDNNLike().to(DEVICE)
    load_old_weights(model)

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc2.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=1e-4)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = {d: 0 for d in DIGITS}
        class_total = {d: 0 for d in DIGITS}

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs)
                pred = outputs.argmax(dim=1)

                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

                for p, y in zip(pred.cpu().numpy(), labels.cpu().numpy()):
                    y = int(y)
                    p = int(p)
                    if y in class_total:
                        class_total[y] += 1
                        if p == y:
                            class_correct[y] += 1

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        for d in DIGITS:
            if class_total[d] > 0:
                acc = class_correct[d] / class_total[d]
                print(f"digit {d}: {acc:.4f} ({class_correct[d]}/{class_total[d]})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    save_new_weights(model)
    print(f"Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()