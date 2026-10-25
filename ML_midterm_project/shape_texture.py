"""GLCM, color clusters, and mask-based lesion shape features."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import SEED


def glcm_features(gray):
    """Haralick-style contrast, homogeneity, energy, correlation over 4 directions."""
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        return {name: 0.0 for name in _glcm_names()}

    # Quantize to 8 gray levels for a small, stable matrix.
    g = np.clip(gray, 0, 255).astype(np.float64)
    q = np.floor(g / 32.0).astype(np.uint8)
    matrix = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )
    values = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        props = graycoprops(matrix, prop).ravel()
        values[f"glcm_{prop}_mean"] = float(props.mean())
        values[f"glcm_{prop}_std"] = float(props.std())
    return values


def _glcm_names():
    names = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        names.extend([f"glcm_{prop}_mean", f"glcm_{prop}_std"])
    return names


def color_cluster_features(rgb, n_clusters=3):
    """k-means on RGB pixels: cluster-center stats and occupancy shares."""
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    if len(pixels) < n_clusters:
        return {name: 0.0 for name in _cluster_names(n_clusters)}
    sample = pixels
    if len(pixels) > 5000:
        rng = np.random.default_rng(SEED)
        sample = pixels[rng.choice(len(pixels), 5000, replace=False)]
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, n_init=3, batch_size=1024)
    model.fit(sample)
    labels = model.predict(pixels)
    centers = model.cluster_centers_
    values = {}
    for i in range(n_clusters):
        share = float((labels == i).mean())
        values[f"kmeans_share_{i}"] = share
        values[f"kmeans_r_{i}"] = float(centers[i, 0])
        values[f"kmeans_g_{i}"] = float(centers[i, 1])
        values[f"kmeans_b_{i}"] = float(centers[i, 2])
    values["kmeans_n_effective"] = float(np.sum([values[f"kmeans_share_{i}"] > 0.05 for i in range(n_clusters)]))
    spreads = [float(np.linalg.norm(centers[i] - centers[j])) for i in range(n_clusters) for j in range(i + 1, n_clusters)]
    values["kmeans_center_spread"] = float(np.mean(spreads)) if spreads else 0.0
    return values


def _cluster_names(n_clusters=3):
    names = ["kmeans_n_effective", "kmeans_center_spread"]
    for i in range(n_clusters):
        names.extend([f"kmeans_share_{i}", f"kmeans_r_{i}", f"kmeans_g_{i}", f"kmeans_b_{i}"])
    return names


def lesion_mask(rgb):
    """Heuristic lesion mask: pixels darker than the image median on luma.

    Not a clinical segmentation. Used only when a rough mask is possible.
    """
    from preprocess import to_gray_luma

    gray = to_gray_luma(rgb)
    thr = float(np.median(gray) - 0.15 * np.std(gray))
    mask = gray < thr
    # Require a plausible lesion area; otherwise declare mask unavailable.
    frac = float(mask.mean())
    if frac < 0.02 or frac > 0.85:
        return None
    return mask


def shape_features(rgb):
    """Asymmetry, border irregularity, compactness, area fraction from a mask."""
    mask = lesion_mask(rgb)
    base = {
        "mask_available": 0.0,
        "mask_area_frac": 0.0,
        "mask_asymmetry": 0.0,
        "mask_perimeter": 0.0,
        "mask_compactness": 0.0,
        "mask_border_irregularity": 0.0,
    }
    if mask is None:
        return base
    area = float(mask.sum())
    if area < 1:
        return base
    # Perimeter via binary erosion difference.
    from scipy import ndimage

    eroded = ndimage.binary_erosion(mask)
    perimeter = float(mask.sum() - eroded.sum())
    # Left/right and top/bottom area asymmetry of the mask.
    h, w = mask.shape
    left = mask[:, : w // 2].sum()
    right = mask[:, w // 2 :].sum()
    top = mask[: h // 2, :].sum()
    bottom = mask[h // 2 :, :].sum()
    asymmetry = 0.5 * (
        abs(left - right) / max(left + right, 1.0) + abs(top - bottom) / max(top + bottom, 1.0)
    )
    compactness = (perimeter ** 2) / (4.0 * np.pi * area) if area > 0 else 0.0
    return {
        "mask_available": 1.0,
        "mask_area_frac": float(area / mask.size),
        "mask_asymmetry": float(asymmetry),
        "mask_perimeter": perimeter,
        "mask_compactness": float(compactness),
        "mask_border_irregularity": float(compactness),
    }


def advanced_feature_names():
    toy_rgb = np.zeros((32, 32, 3), dtype=np.float64)
    toy_gray = toy_rgb.mean(axis=2)
    names = list(glcm_features(toy_gray)) + list(color_cluster_features(toy_rgb)) + list(shape_features(toy_rgb))
    return sorted(set(names))
