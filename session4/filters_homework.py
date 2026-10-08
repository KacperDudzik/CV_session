"""Session 4: filters on MILK10k images.

Kernels: box blur, sharpen, Laplacian, Sobel x/y.
Gaussian blur is scipy.ndimage.gaussian_filter.
Canny is implemented here: Gaussian blur, Sobel, non-maximum suppression, hysteresis.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, convolve, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT / "milk10k"))
IMG_DIR = DATA_DIR / "images"

BOX = np.ones((3, 3), dtype=np.float64) / 9.0
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
SIGMAS = (0.5, 1.0, 2.0, 5.0)


def load_gray(isic_id: str) -> np.ndarray:
    rgb = np.array(Image.open(IMG_DIR / f"{isic_id}.jpg").convert("RGB"), dtype=np.float64)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    return np.hypot(convolve(image, SOBEL_X), convolve(image, SOBEL_Y))


def edge_strength(image: np.ndarray) -> float:
    return float(sobel_magnitude(image).mean())


def texture_strength(image: np.ndarray) -> float:
    return float(np.abs(convolve(image, LAPLACIAN)).mean())


def non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Keep a pixel only if it is a local maximum along the gradient direction."""
    degrees = np.rad2deg(angle) % 180
    padded = np.pad(magnitude, 1, mode="edge")
    east, west = padded[1:-1, 2:], padded[1:-1, :-2]
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    north_east, south_west = padded[:-2, 2:], padded[2:, :-2]
    north_west, south_east = padded[:-2, :-2], padded[2:, 2:]
    horizontal = (degrees < 22.5) | (degrees >= 157.5)
    diagonal_up = (degrees >= 22.5) & (degrees < 67.5)
    vertical = (degrees >= 67.5) & (degrees < 112.5)
    diagonal_down = (degrees >= 112.5) & (degrees < 157.5)
    keep = np.zeros(magnitude.shape, dtype=bool)
    keep[horizontal] = (magnitude[horizontal] >= east[horizontal]) & (magnitude[horizontal] >= west[horizontal])
    keep[diagonal_up] = (magnitude[diagonal_up] >= north_east[diagonal_up]) & (magnitude[diagonal_up] >= south_west[diagonal_up])
    keep[vertical] = (magnitude[vertical] >= north[vertical]) & (magnitude[vertical] >= south[vertical])
    keep[diagonal_down] = (magnitude[diagonal_down] >= north_west[diagonal_down]) & (magnitude[diagonal_down] >= south_east[diagonal_down])
    return magnitude * keep


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep weak edge pixels only when they touch a strong edge."""
    edges = strong.copy()
    kernel = np.ones((3, 3), dtype=bool)
    for _ in range(64):
        grown = binary_dilation(edges, structure=kernel) & (strong | weak)
        if np.array_equal(grown, edges):
            break
        edges = grown
    return edges


def canny(image: np.ndarray, sigma: float = 1.0, low_ratio: float = 0.1, high_ratio: float = 0.2) -> np.ndarray:
    """Canny edges: blur, Sobel, thin the ridges, then keep strong edges and their neighbours."""
    smooth = gaussian_filter(image, sigma=sigma)
    gx = convolve(smooth, SOBEL_X)
    gy = convolve(smooth, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    thin = non_max_suppression(magnitude, np.arctan2(gy, gx))
    high = high_ratio * thin.max()
    low = low_ratio * thin.max()
    return hysteresis(thin >= high, (thin >= low) & (thin < high))


def pick_images() -> dict[str, str]:
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    return {
        label: meta.loc[meta["diagnosis_1"] == label, "isic_id"].iloc[0]
        for label in ("Benign", "Malignant")
    }


def class_filter_stats(images: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    stats = {}
    for label, gray in images.items():
        stats[label] = {
            "sobel": edge_strength(gray),
            "laplacian": texture_strength(gray),
            "sharpen_std": float(convolve(gray, SHARPEN).std()),
        }
    return stats


def sigma_experiment(image: np.ndarray, sigmas: tuple[float, ...] = SIGMAS) -> list[tuple[float, float]]:
    return [(sigma, edge_strength(gaussian_filter(image, sigma=sigma))) for sigma in sigmas]


def canny_sobel_stats(image: np.ndarray) -> dict[str, float]:
    sobel = sobel_magnitude(image)
    edges = canny(image, sigma=1.0)
    return {
        "sobel_top10_pct": float((sobel > np.percentile(sobel, 90)).mean() * 100),
        "canny_edge_pct": float(edges.mean() * 100),
    }


def self_check(image: np.ndarray) -> None:
    assert abs(BOX.sum() - 1) < 1e-9
    assert SHARPEN.sum() == 1
    assert LAPLACIAN.sum() == 0
    edges = canny(image)
    assert edges.shape == image.shape
    assert set(np.unique(edges)).issubset({False, True, 0, 1})
    small = edge_strength(gaussian_filter(image, sigma=1))
    large = edge_strength(gaussian_filter(image, sigma=5))
    assert large < small
    print("self check passed")


def main() -> None:
    ids = pick_images()
    images = {label: load_gray(isic_id) for label, isic_id in ids.items()}
    stats = class_filter_stats(images)
    sigmas = sigma_experiment(images["Malignant"])
    edges = canny_sobel_stats(images["Malignant"])

    print("Benign", ids["Benign"], stats["Benign"])
    print("Malignant", ids["Malignant"], stats["Malignant"])
    print("sigma experiment (malignant):")
    for sigma, strength in sigmas:
        print(f"  sigma {sigma:g}: mean Sobel {strength:.1f}")
    print(
        f"Sobel brightest 10% covers {edges['sobel_top10_pct']:.1f}% of pixels; "
        f"Canny marks {edges['canny_edge_pct']:.1f}%"
    )
    self_check(images["Malignant"])


if __name__ == "__main__":
    main()
