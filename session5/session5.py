"""33 features for one benign and one malignant MILK10k image.

Colour: mean and std of R, G, B.
Histogram: 8 bins per channel, as a share of pixels (counts / n), range 0-255.
Texture: edge density, roughness, Laplacian variance, as in session5_code.ipynb.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT / "milk10k"))
IMG_DIR = DATA_DIR / "images"
N_BINS = 8


def load_rgb(isic_id: str) -> np.ndarray:
    return np.array(Image.open(IMG_DIR / f"{isic_id}.jpg").convert("RGB"), dtype=np.float64)


def texture_features(rgb: np.ndarray) -> np.ndarray:
    """Edge density, roughness, Laplacian variance. Same definitions as the notebook."""
    gray = rgb.mean(axis=2)
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    edge_density = np.hypot(gx, gy).mean()
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_sq_mean = ndimage.uniform_filter(gray ** 2, size=9)
    roughness = (local_sq_mean - local_mean ** 2).mean()
    laplacian_var = ndimage.laplace(gray).var()
    return np.array([edge_density, roughness, laplacian_var])


def feature_names() -> list[str]:
    names = [
        "RED_mean", "GREEN_mean", "BLUE_mean",
        "RED_std", "GREEN_std", "BLUE_std",
    ]
    for channel in ("R", "G", "B"):
        names.extend(f"hist_bin{i}_{channel}" for i in range(1, N_BINS + 1))
    names.extend(["edge_density", "roughness", "laplacian_variance"])
    return names


def feature_vector(rgb: np.ndarray) -> np.ndarray:
    means = rgb.mean(axis=(0, 1))
    stds = rgb.std(axis=(0, 1))
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(rgb[:, :, channel], bins=N_BINS, range=(0, 256))
        histograms.append(counts / rgb[:, :, channel].size)
    return np.concatenate([means, stds, *histograms, texture_features(rgb)])


def pick_images() -> dict[str, str]:
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    return {
        label: meta.loc[meta["diagnosis_1"] == label, "isic_id"].iloc[0]
        for label in ("Benign", "Malignant")
    }


def main() -> None:
    names = feature_names()
    assert len(names) == 33
    ids = pick_images()
    for label in ("Benign", "Malignant"):
        vector = feature_vector(load_rgb(ids[label]))
        assert vector.shape == (33,)
        print(f"\n{label}  {ids[label]}")
        for name, value in zip(names, vector):
            print(f"{name}: {value:.6f}")
        print("vector:", vector)


if __name__ == "__main__":
    main()


# OUTPUT I GOT AFTER RUNNING THE SCRIPT:

# Benign  ISIC_0073863
# RED_mean: 142.334159
# GREEN_mean: 121.896715
# BLUE_mean: 119.439533
# RED_std: 41.868156
# GREEN_std: 42.592416
# BLUE_std: 45.564842
# hist_bin1_R: 0.034578
# hist_bin2_R: 0.061948
# hist_bin3_R: 0.064800
# hist_bin4_R: 0.064304
# hist_bin5_R: 0.183663
# hist_bin6_R: 0.590611
# hist_bin7_R: 0.000096
# hist_bin8_R: 0.000000
# hist_bin1_G: 0.068944
# hist_bin2_G: 0.091926
# hist_bin3_G: 0.068478
# hist_bin4_G: 0.097304
# hist_bin5_G: 0.655544
# hist_bin6_G: 0.017796
# hist_bin7_G: 0.000007
# hist_bin8_G: 0.000000
# hist_bin1_B: 0.091326
# hist_bin2_B: 0.094811
# hist_bin3_B: 0.067885
# hist_bin4_B: 0.101041
# hist_bin5_B: 0.591419
# hist_bin6_B: 0.053489
# hist_bin7_B: 0.000030
# hist_bin8_B: 0.000000
# edge_density: 16.638007
# roughness: 24.883319
# laplacian_variance: 34.038108
# vector: [1.42334159e+02 1.21896715e+02 1.19439533e+02 4.18681563e+01
#  4.25924161e+01 4.55648419e+01 3.45777778e-02 6.19481481e-02   
#  6.48000000e-02 6.43037037e-02 1.83662963e-01 5.90611111e-01   
#  9.62962963e-05 0.00000000e+00 6.89444444e-02 9.19259259e-02   
#  6.84777778e-02 9.73037037e-02 6.55544444e-01 1.77962963e-02   
#  7.40740741e-06 0.00000000e+00 9.13259259e-02 9.48111111e-02   
#  6.78851852e-02 1.01040741e-01 5.91418519e-01 5.34888889e-02   
#  2.96296296e-05 0.00000000e+00 1.66380068e+01 2.48833190e+01   
#  3.40381078e+01]

# Malignant  ISIC_0051817
# RED_mean: 133.988396
# GREEN_mean: 94.374126
# BLUE_mean: 109.632333
# RED_std: 28.961875
# GREEN_std: 30.564123
# BLUE_std: 36.678870
# hist_bin1_R: 0.000611
# hist_bin2_R: 0.006519
# hist_bin3_R: 0.091430
# hist_bin4_R: 0.323137
# hist_bin5_R: 0.353800
# hist_bin6_R: 0.222448
# hist_bin7_R: 0.002056
# hist_bin8_R: 0.000000
# hist_bin1_G: 0.020956
# hist_bin2_G: 0.142222
# hist_bin3_G: 0.350070
# hist_bin4_G: 0.341248
# hist_bin5_G: 0.136967
# hist_bin6_G: 0.008470
# hist_bin7_G: 0.000067
# hist_bin8_G: 0.000000
# hist_bin1_B: 0.019385
# hist_bin2_B: 0.089337
# hist_bin3_B: 0.245793
# hist_bin4_B: 0.319048
# hist_bin5_B: 0.231478
# hist_bin6_B: 0.092959
# hist_bin7_B: 0.001993
# hist_bin8_B: 0.000007
# edge_density: 42.584883
# roughness: 137.516670
# laplacian_variance: 135.250832
# vector: [1.33988396e+02 9.43741259e+01 1.09632333e+02 2.89618747e+01
#  3.05641235e+01 3.66788705e+01 6.11111111e-04 6.51851852e-03   
#  9.14296296e-02 3.23137037e-01 3.53800000e-01 2.22448148e-01   
#  2.05555556e-03 0.00000000e+00 2.09555556e-02 1.42222222e-01   
#  3.50070370e-01 3.41248148e-01 1.36966667e-01 8.47037037e-03   
#  6.66666667e-05 0.00000000e+00 1.93851852e-02 8.93370370e-02   
#  2.45792593e-01 3.19048148e-01 2.31477778e-01 9.29592593e-02   
#  1.99259259e-03 7.40740741e-06 4.25848826e+01 1.37516670e+02   
#  1.35250832e+02]