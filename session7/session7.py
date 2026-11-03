"""Session 7 — Convolutional Neural Networks.

Converted from `class_show copy.ipynb` into functions.

Plan (as in class):
1. Convolution, ReLU and pooling on small tensors we can check by hand.
2. A small CNN from scratch, tracing the shape after every layer.
3. Pretrained ResNet-18: frozen head, then partial fine-tuning.
4. Same lesion-grouped split idea as earlier sessions (dermoscopic views).

Data: MILK10K_DIR or ../milk10k. No dependency on a `common` package.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# Paths, device, constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT / "milk10k"))
IMG_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.csv"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output"
CACHE_PATH = OUTPUT_DIR / "images_224.pt"

CLASSES = ["Benign", "Indeterminate", "Malignant"]
TO_INT = {name: i for i, name in enumerate(CLASSES)}
SEED = 0
IMAGE_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def count_params(module: nn.Module) -> int:
    """Number of learnable parameters in a layer or model."""
    return sum(p.numel() for p in module.parameters())


def out_size(n: int, kernel: int, stride: int = 1, pad: int = 0) -> int:
    """Session 4 formula: floor((n + 2*pad - kernel) / stride) + 1."""
    return (n + 2 * pad - kernel) // stride + 1


def setup(seed: int = SEED) -> str:
    """Print device info and fix the torch seed."""
    torch.manual_seed(seed)
    print(device, torch.__version__)
    return device


# ---------------------------------------------------------------------------
# 1. Convolution by hand → nn.Conv2d
# ---------------------------------------------------------------------------

def make_toy_lesion() -> torch.Tensor:
    """5×5 Session-4 toy: 100 = skin, 10 = lesion."""
    return torch.tensor(
        [
            [100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100],
            [100, 100, 10, 10, 10],
            [100, 100, 10, 10, 10],
            [100, 100, 10, 10, 10],
        ],
        dtype=torch.float32,
    )


def make_sobel_x() -> torch.Tensor:
    """Sobel-x kernel from Session 4."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def demo_conv2d_matches_hand(toy: torch.Tensor | None = None, sobel_x: torch.Tensor | None = None):
    """F.conv2d on the toy lesion should reproduce Session 4 / 5 Sobel-x numbers."""
    toy = make_toy_lesion() if toy is None else toy
    sobel_x = make_sobel_x() if sobel_x is None else sobel_x
    out = F.conv2d(toy.view(1, 1, 5, 5), sobel_x.view(1, 1, 3, 3))
    print(out.shape)
    print(out[0, 0])
    return out[0, 0]


def demo_conv_layer_with_sobel(toy: torch.Tensor | None = None, sobel_x: torch.Tensor | None = None):
    """nn.Conv2d starts random; overwrite weights with Sobel to show it is the same op."""
    toy = make_toy_lesion() if toy is None else toy
    sobel_x = make_sobel_x() if sobel_x is None else sobel_x
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
    print(conv.weight.shape, conv.bias.shape)
    print(conv.weight.data[0, 0])
    with torch.no_grad():
        conv.weight[:] = sobel_x.view(1, 1, 3, 3)
        conv.bias.zero_()
    result = conv(toy.view(1, 1, 5, 5))[0, 0]
    print(result)
    return result


def demo_dense_vs_conv_params():
    """Compare parameter counts: dense vs conv on a 224×224 RGB input."""
    dense = nn.Linear(224 * 224 * 3, 128)
    conv32 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    print(
        f"dense: {count_params(dense):,}   "
        f"conv: {count_params(conv32):,}   "
        f"ratio: {count_params(dense) / count_params(conv32):,.0f}x"
    )
    x = torch.randn(1, 3, 224, 224)
    print("conv out", tuple(conv32(x).shape))
    print("dense out", tuple(dense(x.flatten(1)).shape))
    return dense, conv32


# ---------------------------------------------------------------------------
# 2. ReLU and max pooling
# ---------------------------------------------------------------------------

def demo_relu_and_pool():
    """Reproduce the by-hand ReLU + 2×2 max-pool slide."""
    fmap = torch.tensor(
        [
            [-3.0, 5.0, 2.0, 0.0],
            [1.0, 8.0, -1.0, 4.0],
            [0.0, -2.0, 6.0, 7.0],
            [3.0, 1.0, -4.0, 2.0],
        ]
    )
    relu_map = F.relu(fmap)
    pooled = F.max_pool2d(relu_map.view(1, 1, 4, 4), kernel_size=2)
    print(relu_map)
    print(pooled[0, 0])

    shifted = relu_map.clone()
    shifted[1, 1], shifted[1, 0] = 0.0, 8.0
    print(shifted)
    print(F.max_pool2d(shifted.view(1, 1, 4, 4), 2)[0, 0])
    return relu_map, pooled[0, 0]


def demo_out_size_formula():
    print(out_size(224, kernel=3, pad=1))  # 224: same padding
    print(out_size(224, kernel=2, stride=2))  # 112: pooling halves


# ---------------------------------------------------------------------------
# 3. Small CNN from scratch
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Two Conv→ReLU→Pool blocks, then a linear classifier (class notebook)."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def trace_simple_cnn_shapes(model: SimpleCNN | None = None):
    """Push one empty 224×224 image through each layer and print shapes/params."""
    model = SimpleCNN() if model is None else model
    x = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    for layer in [*model.features, *model.classifier]:
        x = layer(x)
        print(
            f"{layer.__class__.__name__:10s} {str(tuple(x.shape)):20s} "
            f"params: {count_params(layer):>8,}"
        )
    print(f"total parameters: {count_params(model):,}")
    return model


# ---------------------------------------------------------------------------
# 4. MILK10k data: lesion-grouped split + uint8 cache
# ---------------------------------------------------------------------------

class MilkDermDataset(Dataset):
    """Dermoscopic images with integer diagnosis_1 labels."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = Image.open(IMG_DIR / f"{row['isic_id']}.jpg").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = TO_INT[row["diagnosis_1"]]
        return img, label


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(META_PATH)


def make_dermoscopic_splits(seed: int = SEED, n_splits: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lesion-grouped stratified split, dermoscopic view only (class notebook style).

    Uses 10 folds: fold 0 = test (~10%), folds 1–2 = val (~20%), rest = train (~70%).
    """
    meta = load_metadata()
    derm = meta.loc[meta["image_type"] == "dermoscopic"].copy()
    y = derm["diagnosis_1"].to_numpy()
    groups = derm["lesion_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.empty(len(derm), dtype=int)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(derm)), y, groups)):
        fold_of[held] = fold
    derm = derm.assign(fold=fold_of)
    train_df = derm.loc[~derm["fold"].isin([0, 1, 2])].reset_index(drop=True)
    val_df = derm.loc[derm["fold"].isin([1, 2])].reset_index(drop=True)
    test_df = derm.loc[derm["fold"] == 0].reset_index(drop=True)
    return train_df, val_df, test_df


def peek_dataloader(train_df: pd.DataFrame, batch_size: int = 8):
    """Show one batch shape from a live DataLoader (slow path)."""
    transform = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])
    ds = MilkDermDataset(train_df, transform=transform)
    images, labels = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=True)))
    print(images.shape, labels)
    return images, labels


def build_or_load_image_cache(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_path: Path = CACHE_PATH,
) -> dict[str, torch.Tensor]:
    """Decode/resize every image once to a uint8 tensor cache on disk."""
    if cache_path.is_file():
        data = torch.load(cache_path, weights_only=False)
        print(f"loaded cache {cache_path}")
        return data

    t0 = time.perf_counter()
    data: dict[str, torch.Tensor] = {}
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X = torch.empty((len(split), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
        for i, isic_id in enumerate(split["isic_id"]):
            img = Image.open(IMG_DIR / f"{isic_id}.jpg").convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            X[i] = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
        data[f"X_{name}"] = X
        data[f"y_{name}"] = torch.tensor(split["diagnosis_1"].map(TO_INT).to_numpy())
        print(f"  cached {name}: {X.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    print(f"cache built in {time.perf_counter() - t0:.0f} s -> {cache_path}")
    return data


def show_train_samples(X_train: torch.Tensor, y_train: torch.Tensor, n: int = 6):
    """Quick visual check that images and labels still match."""
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for ax, i in zip(axes, range(n)):
        ax.imshow(X_train[i].permute(1, 2, 0))
        ax.set_title(CLASSES[int(y_train[i])])
        ax.axis("off")
    fig.tight_layout()
    out = OUTPUT_DIR / "train_samples.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# 5. Training toolkit: prep, augment, predict, fit
# ---------------------------------------------------------------------------

def prep(xb: torch.Tensor) -> torch.Tensor:
    """uint8 0–255 batch → ImageNet-normalized float on `device`."""
    return (xb.to(device).float() / 255 - MEAN) / STD


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Independent random horizontal and vertical flips (lesions have no 'up')."""
    n = xb.size(0)
    flip_h = torch.rand(n, 1, 1, 1, device=xb.device) < 0.5
    xb = torch.where(flip_h, xb.flip(3), xb)
    flip_v = torch.rand(n, 1, 1, 1, device=xb.device) < 0.5
    return torch.where(flip_v, xb.flip(2), xb)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Argmax class predictions over a cached uint8 tensor."""
    model.eval()
    chunks = [
        model(prep(X[i : i + batch_size])).argmax(1).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(chunks)


def class_weights_from_labels(y_train: torch.Tensor) -> torch.Tensor:
    """Balanced weights: n / (n_classes * n_c), as in Session 6."""
    return (len(y_train) / (len(CLASSES) * torch.bincount(y_train).float())).to(device)


def fit(
    model: nn.Module,
    params,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int = 64,
    aug: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Session-6 training loop with class-weighted CE and best-val-F1 checkpoint."""
    model.to(device)
    optimizer = torch.optim.Adam(params, lr=lr)
    weights = class_weights_from_labels(y_train)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    history: list[dict] = []
    best = {"f1": -1.0, "epoch": -1, "state": None}

    for epoch in range(epochs):
        model.train()
        t0, total = time.perf_counter(), 0.0
        order = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = order[i : i + batch_size]
            xb, yb = prep(X_train[idx]), y_train[idx].to(device)
            if aug:
                xb = augment(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        epoch_s = time.perf_counter() - t0
        f1 = f1_score(y_val.numpy(), predict(model, X_val).numpy(), average="macro")
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(X_train),
                "val_f1": f1,
                "seconds": epoch_s,
            }
        )
        print(
            f"epoch {epoch}  loss {total / len(X_train):.3f}  "
            f"val macro-F1 {f1:.3f}  ({epoch_s:.0f} s)"
        )
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }

    model.load_state_dict(best["state"])
    return pd.DataFrame(history).set_index("epoch"), best


def show_conv_filters(weight: torch.Tensor, rows: int = 1, cols: int | None = None, title: str = ""):
    """Display first-layer filters as tiny RGB images (rescale each filter to [0, 1])."""
    w = weight.detach().cpu()
    cols = cols or w.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(max(cols * 0.9, 4), 1.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, w):
        f = f.permute(1, 2, 0)
        ax.imshow((f - f.min()) / (f.max() - f.min() + 1e-8))
        ax.axis("off")
    for ax in axes[len(w) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6–9. Train scratch CNN, frozen ResNet, fine-tune layer4
# ---------------------------------------------------------------------------

def train_simple_cnn(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
) -> tuple[SimpleCNN, pd.DataFrame, dict]:
    torch.manual_seed(SEED)
    model = SimpleCNN()
    hist, best = fit(model, model.parameters(), X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)
    print("best epoch", best["epoch"], "val macro-F1", round(best["f1"], 3))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = show_conv_filters(model.features[0].weight, cols=16, title="SimpleCNN first layer")
    fig.savefig(OUTPUT_DIR / "scratch_filters.png", dpi=120)
    plt.close(fig)
    return model, hist, best


def load_resnet18_imagenet() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print(model.fc)
    print(f"{count_params(model):,} parameters")
    for name, child in model.named_children():
        print(f"{name:8s} {count_params(child):>10,}")
    return model


def show_resnet_filters(resnet: nn.Module):
    fig = show_conv_filters(resnet.conv1.weight, rows=4, cols=16, title="ResNet-18 conv1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "resnet_filters.png", dpi=120)
    plt.close(fig)


def show_resnet_feature_hierarchy(resnet: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor):
    """One feature map after each ResNet stage on the first validation image."""
    x = prep(X_val[:1])
    resnet.to(device).eval()
    with torch.no_grad():
        h = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(x))))
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
        axes[0].imshow(X_val[0].permute(1, 2, 0))
        axes[0].set_title(CLASSES[int(y_val[0])])
        axes[0].axis("off")
        for ax, name in zip(axes[1:], ["layer1", "layer2", "layer3", "layer4"]):
            h = getattr(resnet, name)(h)
            k = int(h[0].flatten(1).std(1).argmax())
            ax.imshow(h[0, k].cpu(), cmap="magma")
            ax.set_title(f"{name} {tuple(h.shape[1:])}")
            ax.axis("off")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "resnet_hierarchy.png", dpi=120)
    plt.close(fig)


@torch.no_grad()
def embed_resnet(resnet: nn.Module, X: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """512-D backbone features (fc replaced by Identity)."""
    resnet.eval()
    chunks = [
        resnet(prep(X[i : i + batch_size])).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(chunks)


def train_frozen_resnet_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
) -> tuple[nn.Module, nn.Module, float]:
    """Strategy 1: freeze ResNet-18, train only a new Linear(512, 3) on embeddings."""
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.to(device).eval()

    t0 = time.perf_counter()
    F_train = embed_resnet(resnet, X_train)
    F_val = embed_resnet(resnet, X_val)
    print(F_train.shape, f"{time.perf_counter() - t0:.0f} s")

    torch.manual_seed(SEED)
    head = nn.Linear(512, 3).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_from_labels(y_train))
    loader = DataLoader(TensorDataset(F_train, y_train), batch_size=64, shuffle=True)

    for _ in range(epochs):
        for fb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(fb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred = head(F_val.to(device)).argmax(1).cpu()
    val_f1 = float(f1_score(y_val.numpy(), pred.numpy(), average="macro"))
    print(f"frozen ResNet-18 + linear head: val macro-F1 = {val_f1:.3f}")

    ConfusionMatrixDisplay.from_predictions(
        y_val.numpy(), pred.numpy(), display_labels=CLASSES, cmap="Oranges"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_frozen.png", dpi=120)
    plt.close()
    return resnet, head, val_f1


def make_partially_unfrozen_resnet(
    train_layers: tuple[str, ...] | None = ("layer4", "fc"),
) -> tuple[nn.Module, list]:
    """Fresh ImageNet ResNet-18; only named blocks keep requires_grad=True.

    Pass ``train_layers=None`` for strategy 3 (full fine-tuning: every parameter).
    """
    torch.manual_seed(SEED)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 3)
    if train_layers is None:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for name, p in model.named_parameters():
            p.requires_grad = any(name.startswith(prefix) for prefix in train_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"training {sum(p.numel() for p in trainable):,} of {count_params(model):,} parameters")
    return model, trainable


def fine_tune_resnet(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    train_layers: tuple[str, ...] | None = ("layer4", "fc"),
    epochs: int = 6,
    lr: float = 1e-4,
    confusion_tag: str | None = None,
) -> tuple[nn.Module, pd.DataFrame, dict]:
    """Fine-tune ResNet-18. ``train_layers=None`` = strategy 3 (whole network)."""
    model, trainable = make_partially_unfrozen_resnet(train_layers)
    n_trainable = sum(p.numel() for p in trainable)
    hist, best = fit(
        model, trainable, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, aug=True
    )
    print("best epoch", best["epoch"], "val macro-F1", round(best["f1"], 3))
    best["n_trainable"] = int(n_trainable)
    best["mean_seconds_per_epoch"] = float(hist["seconds"].mean()) if "seconds" in hist else float("nan")

    ConfusionMatrixDisplay.from_predictions(
        y_val.numpy(), predict(model, X_val).numpy(), display_labels=CLASSES, cmap="Oranges"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = confusion_tag or ("full" if train_layers is None else "_".join(train_layers))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_finetune_{tag}.png", dpi=120)
    plt.close()
    return model, hist, best


def run_mini_challenge(epochs: int = 6, lr: float = 1e-4) -> dict:
    """Slide 29 mini-challenge: layer4+fc vs layer3+layer4+fc at lr=1e-4.

    Measures trainable params, seconds/epoch, and best validation macro-F1.
    The test set stays closed.
    """
    import json

    setup()
    if not META_PATH.is_file() or not IMG_DIR.is_dir():
        raise FileNotFoundError(f"MILK10k not found at {DATA_DIR}")

    train_df, val_df, test_df = make_dermoscopic_splits(seed=SEED)
    data = build_or_load_image_cache(train_df, val_df, test_df)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"splits train={len(train_df)} val={len(val_df)} test={len(test_df)} (test unused)")

    configs = [
        ("layer4_only", ("layer4", "fc")),
        ("layer3_and_4", ("layer3", "layer4", "fc")),
    ]
    summary = {"epochs": epochs, "lr": lr, "seed": SEED, "runs": {}}

    for tag, layers in configs:
        print(f"\n=== Mini-challenge: unfreeze {layers} ===")
        _, hist, best = fine_tune_resnet(
            X_train, y_train, X_val, y_val, train_layers=layers, epochs=epochs, lr=lr
        )
        summary["runs"][tag] = {
            "layers": list(layers),
            "n_trainable": best["n_trainable"],
            "mean_seconds_per_epoch": best["mean_seconds_per_epoch"],
            "best_epoch": best["epoch"],
            "best_val_macro_f1": best["f1"],
            "epoch_history": hist.reset_index().to_dict(orient="records"),
        }

    a = summary["runs"]["layer4_only"]
    b = summary["runs"]["layer3_and_4"]
    delta_f1 = b["best_val_macro_f1"] - a["best_val_macro_f1"]
    delta_s = b["mean_seconds_per_epoch"] - a["mean_seconds_per_epoch"]
    worth = bool(delta_f1 > 0.01)  # clear gain beyond noise on ~600 val lesions
    summary["comparison"] = {
        "delta_val_macro_f1": float(delta_f1),
        "delta_seconds_per_epoch": float(delta_s),
        "worth_unfreezing_layer3": worth,
        "decision": (
            "Worth it: layer3+layer4 improves validation macro-F1 enough to justify the cost."
            if worth
            else "Not clearly worth it: the F1 gain is small/none relative to extra compute and "
            "overfitting risk on this validation set size."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "mini_challenge_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + summary["comparison"]["decision"])
    print("wrote", out)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_toy_demos():
    """Sections 1–3 of the class notebook (no MILK10k needed)."""
    setup()
    print("\n--- 1.1 F.conv2d ---")
    demo_conv2d_matches_hand()
    print("\n--- 1.2 nn.Conv2d + Sobel ---")
    demo_conv_layer_with_sobel()
    print("\n--- 1.3 / 1.4 dense vs conv ---")
    demo_dense_vs_conv_params()
    print("\n--- 2. ReLU + pool ---")
    demo_relu_and_pool()
    print("\n--- 2.3 out_size ---")
    demo_out_size_formula()
    print("\n--- 3. SimpleCNN shapes ---")
    trace_simple_cnn_shapes()


def run_training(
    train_scratch: bool = True,
    train_frozen: bool = True,
    train_finetune: bool = True,
    scratch_epochs: int = 10,
    finetune_epochs: int = 6,
):
    """Sections 4–9: cache images and train the CNN / ResNet variants."""
    setup()
    if not META_PATH.is_file() or not IMG_DIR.is_dir():
        raise FileNotFoundError(
            f"MILK10k not found at {DATA_DIR}. Set MILK10K_DIR or place milk10k next to the repo."
        )

    train_df, val_df, test_df = make_dermoscopic_splits(seed=SEED)
    print(len(train_df), len(val_df), len(test_df))
    peek_dataloader(train_df)

    data = build_or_load_image_cache(train_df, val_df, test_df)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(X_train.shape, X_train.dtype, torch.bincount(y_train))
    show_train_samples(X_train, y_train)

    print("\nprep check:", ((200 / 255) - 0.485) / 0.229)

    results = {}
    if train_scratch:
        print("\n=== 6. SimpleCNN from scratch ===")
        _, hist_s, best_s = train_simple_cnn(
            X_train, y_train, X_val, y_val, epochs=scratch_epochs
        )
        results["scratch"] = {"best": best_s, "history": hist_s}

    print("\n=== 7. ResNet-18 ImageNet ===")
    resnet = load_resnet18_imagenet()
    show_resnet_filters(resnet)
    show_resnet_feature_hierarchy(resnet, X_val, y_val)

    if train_frozen:
        print("\n=== 8. Frozen ResNet + linear head ===")
        _, _, val_f1 = train_frozen_resnet_head(X_train, y_train, X_val, y_val)
        results["frozen_head_f1"] = val_f1

    if train_finetune:
        print("\n=== 9. Fine-tune layer4 + fc ===")
        _, hist_ft, best_ft = fine_tune_resnet(
            X_train, y_train, X_val, y_val, train_layers=("layer4", "fc"), epochs=finetune_epochs
        )
        results["finetune_layer4"] = {"best": best_ft, "history": hist_ft}

    return results


def main():
    """Run toy demos always; run MILK10k training when the dataset is present."""
    run_toy_demos()
    if META_PATH.is_file() and IMG_DIR.is_dir():
        run_training()
    else:
        print(f"\nSkip training: MILK10k not found at {DATA_DIR}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mini-challenge":
        run_mini_challenge()
    else:
        main()
