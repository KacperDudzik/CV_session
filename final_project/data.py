"""Data: the one frozen split, the labels for either target, and the image cache.

load_image() is the ONLY place where an image file becomes model input. Training,
validation, predict() and the submission all go through it (Session 10: no train/serve skew).
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
import session7 as s7  # make_dermoscopic_splits, prep (ImageNet normalisation)


def load_image(path) -> torch.Tensor:
    """One image file -> uint8 tensor (3, 224, 224): RGB, PIL resize, no random step."""
    img = Image.open(path).convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def load_splits() -> dict[str, pd.DataFrame]:
    """Train / val / test tables (one dermoscopic image = one lesion). Created once, then frozen.

    The first run calls the Session 7 split and saves the lesion lists to splits/*.csv;
    every later run reads those files, so every model sees exactly the same lesions.
    """
    names = ("train", "val", "test")
    if not all((config.SPLIT_DIR / f"{n}.csv").is_file() for n in names):
        parts = s7.make_dermoscopic_splits(seed=config.SPLIT_SEED)
        gt = pd.read_csv(config.GT_PATH)
        gt["dx11"] = gt[config.CLASSES["dx11"]].idxmax(axis=1)          # Session 10, slide 22
        tone = pd.read_csv(config.INPUT_PATH)[["isic_id", "skin_tone_class"]]
        config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in zip(names, parts):
            df = df.merge(gt[["lesion_id", "dx11"]], on="lesion_id").merge(tone, on="isic_id", how="left")
            cols = ["lesion_id", "isic_id", "diagnosis_1", "dx11", "sex", "age_approx", "skin_tone_class"]
            df[cols].to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
    splits = {n: pd.read_csv(config.SPLIT_DIR / f"{n}.csv") for n in names}

    # Session 3 rule, checked every time: no lesion in two splits, one row per lesion.
    ids = [set(df["lesion_id"]) for df in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "lesion leakage between splits"
    assert all(df["lesion_id"].is_unique for df in splits.values())
    return splits


def labels(df: pd.DataFrame, target: str) -> torch.Tensor:
    """Integer labels 0..n-1 in the order of config.CLASSES[target]."""
    return torch.tensor(df[target].map(config.CLASSES[target].index).to_numpy())


def load_images(df: pd.DataFrame, name: str) -> torch.Tensor:
    """All images of one split as uint8 (n, 3, 224, 224); decoded once, then read from disk."""
    path = config.CACHE_DIR / f"{name}_{len(df)}_{config.IMAGE_SIZE}.pt"
    if not path.is_file():
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        X = torch.stack([load_image(config.IMG_DIR / f"{i}.jpg") for i in df["isic_id"]])
        torch.save(X, path)
    return torch.load(path)


def train_norm_stats() -> tuple[list[float], list[float]]:
    """Per-channel mean/std of the TRAINING images only (S3: never the full dataset)."""
    X = load_images(load_splits()["train"], "train").float() / 255
    return X.mean(dim=(0, 2, 3)).tolist(), X.std(dim=(0, 2, 3)).tolist()


def augment_m1(xb: torch.Tensor) -> torch.Tensor:
    """M1 / Session 3 augmentation on a normalised GPU batch, a fresh draw per image.

    Flips (session7.augment), rotation up to 15 degrees, brightness / contrast / saturation
    x[0.9, 1.1], hue untouched (s1_s3_homework/outputs/augmentation_table.csv).
    """
    xb = s7.augment(xb)
    n, dev = xb.size(0), xb.device
    x = (xb * s7.STD + s7.MEAN).clamp(0, 1)                       # back to [0, 1] RGB
    angle = (torch.rand(n, device=dev) * 2 - 1) * np.pi * 15 / 180
    cos, sin, zero = torch.cos(angle), torch.sin(angle), torch.zeros(n, device=dev)
    theta = torch.stack([torch.stack([cos, -sin, zero], 1), torch.stack([sin, cos, zero], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, align_corners=False)   # black corners, as PIL
    def factor():
        return 0.9 + 0.2 * torch.rand(n, 1, 1, 1, device=dev)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    x = x * factor()                                                      # brightness
    x = (x - gray.mean(dim=(2, 3), keepdim=True)) * factor() + gray.mean(dim=(2, 3), keepdim=True)  # contrast
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    x = gray + (x - gray) * factor()                                      # saturation
    return (x.clamp(0, 1) - s7.MEAN) / s7.STD


AUGMENTATIONS = {"flips": None, "m1": augment_m1}  # None = session7.augment (the S7 default)


def load_data(target: str) -> dict:
    """Everything a training or evaluation script needs, for one target."""
    splits = load_splits()
    out = {"splits": splits, "classes": config.CLASSES[target]}
    for name, df in splits.items():
        out[f"X_{name}"] = load_images(df, name)
        out[f"y_{name}"] = labels(df, target)
    return out
