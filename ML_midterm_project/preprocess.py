"""Image resize, color spaces, and normalization used at train and inference."""

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab

# Luminosity weights from Session 2 / OpenCV.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def resize_rgb(rgb, size):
    """Resize with bilinear interpolation. `size` is (width, height)."""
    if size is None:
        return rgb
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return np.asarray(image.resize(size, Image.BILINEAR), dtype=np.float64)


def to_gray_mean(rgb):
    """Session 5 simple gray used by the original texture block."""
    return rgb.mean(axis=2)


def to_gray_luma(rgb):
    """Weighted grayscale Gray = 0.299R + 0.587G + 0.114B."""
    return rgb @ LUMA


def to_hsv(rgb):
    """RGB 0–255 -> HSV with H in [0, 1], S/V in [0, 1] (skimage convention)."""
    return rgb2hsv(np.clip(rgb, 0, 255) / 255.0)


def to_lab(rgb):
    """RGB 0–255 -> CIE Lab."""
    return rgb2lab(np.clip(rgb, 0, 255) / 255.0)


def normalize(rgb, mode="minmax"):
    """Return float image ready for filter banks. Stats are not fit here."""
    x = rgb.astype(np.float64)
    if mode == "none":
        return x
    if mode == "minmax":
        return x / 255.0
    if mode == "zscore":
        if x.ndim == 3:
            mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
            std = x.reshape(-1, x.shape[-1]).std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            return (x - mean) / std
        mean = x.mean()
        std = x.std() or 1.0
        return (x - mean) / std
    raise ValueError(f"Unknown normalization mode: {mode}")


def channel_stats_and_hist(channels, names, n_bins, value_range):
    """Mean/std and share histogram for each channel of an HxWxC array."""
    values = {}
    for i, name in enumerate(names):
        plane = channels[..., i]
        values[f"{name}_mean"] = float(plane.mean())
        values[f"{name}_std"] = float(plane.std())
        counts, _ = np.histogram(plane, bins=n_bins, range=value_range)
        shares = counts / max(plane.size, 1)
        for b, share in enumerate(shares, start=1):
            values[f"hist_bin{b}_{name}"] = float(share)
    return values
