"""Color, histogram, and convolution features from Sessions 4 and 5.

`extract_features` is the original Session 5 + extended texture block and must
stay bit-compatible with the working baseline. New color spaces, filter banks,
GLCM/shape/optional descriptors are added by `extract_all_features`.
"""

import numpy as np
from scipy import ndimage

from config import FILTER_IMAGE_SIZE, INCLUDE_OPTIONAL_DESCRIPTORS, N_BINS
from descriptors import optional_descriptor_features
from kernels import filter_bank_features
from preprocess import (
    channel_stats_and_hist,
    normalize,
    resize_rgb,
    to_gray_luma,
    to_gray_mean,
    to_hsv,
    to_lab,
)
from shape_texture import color_cluster_features, glcm_features, shape_features

COLOR_NAMES = [
    "RED_mean", "GREEN_mean", "BLUE_mean",
    "RED_std", "GREEN_std", "BLUE_std",
]
HIST_NAMES = [
    f"hist_bin{i}_{channel}"
    for channel in ("R", "G", "B")
    for i in range(1, N_BINS + 1)
]
TEXTURE_NAMES = ["edge_density", "roughness", "laplacian_variance"]
EXTRA_NAMES = ["sobel_std", "sobel_p95", "asymmetry"]
CLASS_33 = COLOR_NAMES + HIST_NAMES + TEXTURE_NAMES
ALL_NAMES = CLASS_33 + EXTRA_NAMES


def _histograms(rgb):
    shares = []
    for channel in range(3):
        counts, _ = np.histogram(rgb[:, :, channel], bins=N_BINS, range=(0, 256))
        shares.append(counts / rgb[:, :, channel].size)
    return np.concatenate(shares)


def _texture(gray):
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    magnitude = np.hypot(gx, gy)
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_sq_mean = ndimage.uniform_filter(gray ** 2, size=9)
    roughness = (local_sq_mean - local_mean ** 2).mean()
    laplacian_var = ndimage.laplace(gray).var()
    return {
        "edge_density": float(magnitude.mean()),
        "roughness": float(roughness),
        "laplacian_variance": float(laplacian_var),
        "sobel_std": float(magnitude.std()),
        "sobel_p95": float(np.percentile(magnitude, 95)),
    }


def _asymmetry(rgb):
    """Mean absolute gap between left-half and right-half channel means."""
    half = rgb.shape[1] // 2
    left = rgb[:, :half].mean(axis=(0, 1))
    right = rgb[:, -half:].mean(axis=(0, 1))
    return float(np.abs(left - right).mean())


def extract_features(rgb):
    """One image -> original named features. `rgb` is float H×W×3 in 0–255."""
    means = rgb.mean(axis=(0, 1))
    stds = rgb.std(axis=(0, 1))
    values = {
        "RED_mean": float(means[0]),
        "GREEN_mean": float(means[1]),
        "BLUE_mean": float(means[2]),
        "RED_std": float(stds[0]),
        "GREEN_std": float(stds[1]),
        "BLUE_std": float(stds[2]),
    }
    for name, share in zip(HIST_NAMES, _histograms(rgb)):
        values[name] = float(share)
    values.update(_texture(rgb.mean(axis=2)))
    values["asymmetry"] = _asymmetry(rgb)
    return values


def _colorspace_features(rgb):
    values = {}
    luma = to_gray_luma(rgb)
    values["luma_mean"] = float(luma.mean())
    values["luma_std"] = float(luma.std())
    counts, _ = np.histogram(luma, bins=N_BINS, range=(0, 256))
    for i, share in enumerate(counts / max(luma.size, 1), start=1):
        values[f"hist_bin{i}_luma"] = float(share)

    hsv = to_hsv(rgb)
    values.update(channel_stats_and_hist(hsv, ["HSV_H", "HSV_S", "HSV_V"], N_BINS, (0.0, 1.0)))
    lab = to_lab(rgb)
    values.update(
        channel_stats_and_hist(lab, ["Lab_L", "Lab_a", "Lab_b"], N_BINS, (-128.0, 128.0))
    )
    return values


def extract_all_features(rgb):
    """Original features plus colorspace, filters, GLCM/shape, optional descriptors."""
    values = extract_features(rgb)
    values.update(_colorspace_features(rgb))

    small = resize_rgb(rgb, FILTER_IMAGE_SIZE)
    gray_mean = to_gray_mean(small)
    gray_luma = to_gray_luma(small)
    values.update(filter_bank_features(gray_mean))
    values.update(glcm_features(gray_luma))
    values.update(color_cluster_features(small))
    values.update(shape_features(small))

    if INCLUDE_OPTIONAL_DESCRIPTORS:
        gray_u8 = np.clip(gray_luma, 0, 255).astype(np.uint8)
        values.update(optional_descriptor_features(gray_u8))

    minmax = normalize(small, "minmax")
    zscore = normalize(small, "zscore")
    values["norm_minmax_mean"] = float(minmax.mean())
    values["norm_zscore_std"] = float(zscore.std())
    return values


def _discover_blocks():
    rng = np.random.default_rng(0)
    toy = rng.integers(0, 256, size=(48, 48, 3)).astype(np.float64)
    full = extract_all_features(toy)
    original = set(ALL_NAMES)
    colorspace = set(_colorspace_features(toy))
    filters = set(filter_bank_features(to_gray_mean(resize_rgb(toy, FILTER_IMAGE_SIZE))))
    advanced = set(glcm_features(to_gray_luma(toy))) | set(color_cluster_features(toy)) | set(shape_features(toy))
    optional = set(optional_descriptor_features(np.clip(to_gray_luma(toy), 0, 255).astype(np.uint8))) if INCLUDE_OPTIONAL_DESCRIPTORS else set()
    norms = {"norm_minmax_mean", "norm_zscore_std"}
    return {
        "full": sorted(full),
        "colorspace": sorted(colorspace),
        "filters": sorted(filters),
        "advanced": sorted(advanced),
        "optional": sorted(optional),
        "norms": sorted(norms),
        "original": sorted(original),
    }


_BLOCKS = _discover_blocks()
COLORSPACE_NAMES = _BLOCKS["colorspace"]
FILTER_NAMES = _BLOCKS["filters"]
ADVANCED_NAMES = _BLOCKS["advanced"]
OPTIONAL_NAMES = _BLOCKS["optional"]
NORM_NAMES = _BLOCKS["norms"]
FULL_NAMES = sorted(set(_BLOCKS["full"]) | {"is_dermoscopic"})

FEATURE_SETS = {
    "color_6": COLOR_NAMES,
    "hist_24": HIST_NAMES,
    "texture_3": TEXTURE_NAMES,
    "color_hist_30": COLOR_NAMES + HIST_NAMES,
    "class_33": CLASS_33,
    "class_33_view": CLASS_33 + ["is_dermoscopic"],
    "extended": ALL_NAMES + ["is_dermoscopic"],
    "colorspace": COLORSPACE_NAMES + ["is_dermoscopic"],
    "filters": FILTER_NAMES + ["is_dermoscopic"],
    "glcm_shape": ADVANCED_NAMES + ["is_dermoscopic"],
    "optional_descriptors": (OPTIONAL_NAMES + ["is_dermoscopic"]) if OPTIONAL_NAMES else ["is_dermoscopic"],
    "original_plus_filters": ALL_NAMES + FILTER_NAMES + ["is_dermoscopic"],
    "original_plus_advanced": ALL_NAMES + ADVANCED_NAMES + ["is_dermoscopic"],
    "all_features": FULL_NAMES,
}
