"""Handcrafted convolution filters and feature-map summaries (Sessions 4 / 4.1)."""

import numpy as np
from scipy import ndimage

from config import EDGE_THRESHOLD

# Session 4 / live-lab kernels.
BOX_3 = np.ones((3, 3), dtype=np.float64) / 9.0
BOX_5 = np.ones((5, 5), dtype=np.float64) / 25.0
BOX_9 = np.ones((9, 9), dtype=np.float64) / 81.0
GAUSSIAN_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
EMBOSS = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
LAPLACIAN_3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
LAPLACIAN_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.float64,
)


def correlate(image, kernel):
    return ndimage.correlate(image, kernel, mode="reflect")


def summarize_map(response, prefix, edge_threshold=EDGE_THRESHOLD):
    """Mean, std, var, 95th pct, edge fraction, and four quadrant means."""
    flat = response.astype(np.float64)
    values = {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_var": float(flat.var()),
        f"{prefix}_p95": float(np.percentile(flat, 95)),
        f"{prefix}_frac_edge": float((np.abs(flat) > edge_threshold).mean()),
    }
    h, w = flat.shape
    hm, wm = h // 2, w // 2
    quads = {
        "q1": flat[:hm, :wm],
        "q2": flat[:hm, wm:],
        "q3": flat[hm:, :wm],
        "q4": flat[hm:, wm:],
    }
    for name, patch in quads.items():
        values[f"{prefix}_{name}_mean"] = float(patch.mean()) if patch.size else 0.0
    return values


def gabor_kernel(size, theta, frequency, sigma=None):
    """Real Gabor kernel. `theta` radians, `frequency` cycles/pixel."""
    if size % 2 == 0:
        raise ValueError("Gabor size must be odd.")
    if sigma is None:
        sigma = 0.56 / max(frequency, 1e-6)
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-0.5 * (x_theta**2 + y_theta**2) / (sigma**2))
    wave = np.cos(2 * np.pi * frequency * x_theta)
    kernel = gaussian * wave
    return kernel - kernel.mean()


def filter_bank_features(gray):
    """Apply the Session 4 kernels, multi-sigma blur, Laplacian scales, Gabor bank."""
    values = {}
    gray = gray.astype(np.float64)

    for name, kernel in (
        ("box3", BOX_3),
        ("box5", BOX_5),
        ("box9", BOX_9),
        ("gauss3", GAUSSIAN_3),
        ("sharpen", SHARPEN),
        ("emboss", EMBOSS),
        ("sobel_x", SOBEL_X),
        ("sobel_y", SOBEL_Y),
        ("lap3", LAPLACIAN_3),
        ("lap5", LAPLACIAN_5),
    ):
        values.update(summarize_map(correlate(gray, kernel), name))

    for sigma in (0.5, 1.0, 2.0, 5.0):
        blurred = ndimage.gaussian_filter(gray, sigma=sigma)
        values.update(summarize_map(blurred, f"gauss_s{sigma:g}"))

    gx = correlate(gray, SOBEL_X)
    gy = correlate(gray, SOBEL_Y)
    values.update(summarize_map(np.hypot(gx, gy), "sobel_mag"))

    # Compact Gabor bank: 4 orientations × 2 frequencies, 9×9 support.
    for i, theta in enumerate((0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
        for j, freq in enumerate((0.1, 0.25)):
            kernel = gabor_kernel(9, theta=theta, frequency=freq)
            values.update(summarize_map(correlate(gray, kernel), f"gabor_o{i}_f{j}"))
    return values


def filter_feature_names():
    """Stable name list for cache validation (one dummy gray pass)."""
    toy = np.zeros((16, 16), dtype=np.float64)
    return sorted(filter_bank_features(toy).keys())
