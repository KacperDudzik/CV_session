"""Session 2 — pixels, color, histograms, normalization (Q1–Q6)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "milk10k"
IMG_DIR = DATA_DIR / "images"


def load_image(isic_id: str) -> np.ndarray:
    return np.array(Image.open(IMG_DIR / f"{isic_id}.jpg"))


def to_gray(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def quantize(arr: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits
    q = np.floor(arr.astype(np.float64) / 256 * levels)
    q = np.clip(q, 0, levels - 1)
    return (q * (255 / (levels - 1))).astype(np.uint8)


def q1_pixels_and_resolution(arr: np.ndarray, isic_id: str):
    print("\n=== Q1  Pixels & Resolution ===")
    h, w, c = arr.shape
    print(f"{isic_id}: NumPy shape {arr.shape}, dtype {arr.dtype}")
    print(f"spatial resolution: {h} x {w}  (height x width)")
    print(
        "This is the pixel grid: each (row, col) is one sample of the scene. "
        "More pixels over the same physical area means finer spatial detail "
        "(smaller structures, sharper lesion edges)."
    )
    print(
        "Three numbers, not two: (H, W, C). C=3 is the color channels (R, G, B). "
        "A grayscale image would be (H, W) only."
    )
    print(
        f"dtype {arr.dtype} is 8-bit unsigned int, so each channel is an integer "
        "in [0, 255] (256 intensity levels). PIL size is (W, H); array shape is (H, W, C)."
    )

    cy, cx = h // 2, w // 2
    crop = arr[cy - 50 : cy + 50, cx - 50 : cx + 50]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(arr)
    axes[0].set_title(f"Original  {h}x{w}")
    axes[1].imshow(crop)
    axes[1].set_title("Center crop 100x100")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(
        "The crop keeps local texture but drops the rest of the lesion, surrounding "
        "skin, and scale. That can hide size, asymmetry, and border context that "
        "matter for diagnosis, so a 100x100 window is often not enough on its own."
    )


# ### Pixels & resolution
# This picture is 450 pixels tall and 600 pixels wide. That just means the photo
# is made of a 450-by-600 grid of tiny squares. More squares in the same photo
# means you can see smaller details (a sharp edge, a tiny spot). It does *not*
# by itself tell you how many millimetres the lesion is on the skin.
#
# In NumPy the shape is (450, 600, 3). The extra 3 is color: red, green, and blue.
# A black-and-white image would only have two numbers: height and width.
# Also: NumPy lists height first, then width. PIL lists width first, then height.
#
# The values are whole numbers from 0 (black in that color) to 255 (as bright as
# it can get). This photo only goes up to 230, so it is not using the full range.
#
# The 100x100 crop from the middle keeps a close-up of texture, but you lose the
# rest of the lesion and the skin around it. That matters for diagnosis: you cannot
# really judge size, shape, or the border if you only see the center.


def q2_color_depth(arr: np.ndarray):
    print("\n=== Q2  Color Depth ===")
    variants = [
        ("8-bit (256)", arr),
        ("4-bit (16)", quantize(arr, 4)),
        ("2-bit (4)", quantize(arr, 2)),
        ("1-bit (2)", quantize(arr, 1)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, im) in zip(axes, variants):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(
        "The lesion/skin boundary is still readable at 4-bit, weaker at 2-bit, "
        "and mostly gone at 1-bit (only a silhouette). Banding also hides subtle "
        "color and shade changes. A diagnostic tool needs those gradients; too few "
        "levels would miss faint borders and pigment variation."
    )


# ### Color depth
# 8-bit is the original photo (256 brightness steps). At 4-bit you can still see
# the lesion, but it looks more "blocky". At 2-bit the edge is already hard to
# follow. At 1-bit you mostly get a black/white blob.
# That matters because a doctor (or a model) needs the small shade changes to
# tell where the lesion stops and normal skin starts. If you throw those away,
# you miss the faint border.


def q3_rgb_channels(arr: np.ndarray):
    print("\n=== Q3  RGB Channels ===")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, i, name in zip(axes, range(3), ["R", "G", "B"]):
        ax.imshow(arr[:, :, i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name} intensity")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    means = arr.mean(axis=(0, 1))
    print(f"mean intensity R, G, B: {means}")
    print(
        "Shown as grayscale: bright = high value in that channel. On dermoscopy, "
        "the lesion is often darkest (most contrast vs skin) in G or R, because "
        "melanin absorbs more in those bands; B is typically noisier / lower "
        "contrast. Use the plot: the channel with the largest lesion–skin gap "
        "is the most useful for this image."
    )


# ### RGB channels
# These plots are not colored red/green/blue. They are gray on purpose: brighter
# means "more of that color" in the pixel. On this photo the average is about
# red 134, green 94, blue 110 — so green is the darkest overall, red the brightest.
#
# Looking at the lesion vs the skin around it, blue (and then red) show the
# difference a bit more clearly here. That can change from photo to photo.
# There is no single "correct" channel. Use whichever one makes the lesion
# stand out from the skin the most.


def q4_grayscale_conversion(arr: np.ndarray):
    print("\n=== Q4  Grayscale Conversion ===")
    naive = arr.mean(axis=2)
    luma = to_gray(arr)
    diff = np.abs(naive - luma)
    print(f"max |naive - luminosity| = {diff.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(naive, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Naive (R+G+B)/3")
    axes[1].imshow(luma, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Luminosity 0.299R+0.587G+0.114B")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(
        "They disagree because luminosity weights channels by human sensitivity "
        "(green most, then red, blue least). Equal averaging treats B as important "
        "as G, so blues look too bright and greens too dark. The weighted formula "
        "better matches perception (Rec. 601 luma)."
    )


# ### Grayscale conversion
# One way: just average red, green, and blue. Other way: give green the most
# weight (0.587), then red (0.299), then blue (0.114), because our eyes work
# that way.
#
# On this image the two gray photos differ by up to about 11 brightness steps.
# They disagree because the simple average treats blue as important as green,
# but we barely see blue compared to green. The weighted version looks closer
# to how a person would see the photo.


def q5_histograms(arr: np.ndarray, meta: pd.DataFrame):
    print("\n=== Q5  Histograms ===")
    gray = to_gray(arr)

    plt.figure(figsize=(6, 4))
    plt.hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
    plt.title("Grayscale histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    for i, name, color in zip(range(3), ["R", "G", "B"], ["r", "g", "b"]):
        plt.hist(arr[:, :, i].ravel(), bins=256, range=(0, 256), color=color, alpha=0.45, label=name)
    plt.title("RGB histograms (overlaid)")
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.legend()
    plt.tight_layout()
    plt.show()

    a = meta.loc[meta["diagnosis_1"] == "Benign", "isic_id"].iloc[0]
    b = meta.loc[meta["diagnosis_1"] == "Malignant", "isic_id"].iloc[0]
    img_a, img_b = load_image(a), load_image(b)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(img_a)
    axes[0, 0].set_title(f"Benign  {a}")
    axes[0, 1].imshow(img_b)
    axes[0, 1].set_title(f"Malignant  {b}")
    axes[1, 0].hist(to_gray(img_a).ravel(), bins=256, range=(0, 256), color="gray")
    axes[1, 0].set_title("Benign grayscale hist")
    axes[1, 1].hist(to_gray(img_b).ravel(), bins=256, range=(0, 256), color="gray")
    axes[1, 1].set_title("Malignant grayscale hist")
    for ax in axes[0]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    print(
        "Histogram shape can differ by class, but lighting, lesion size (how much "
        "dark pigment vs skin), and skin tone also shift it. A darker/larger lesion "
        "pushes mass toward low intensities; bright skin or gel glare piles up at "
        "the high end. Do not treat one pair of histograms as a class signature."
    )


# ### Histograms
# A histogram just counts how many pixels have each brightness. For this photo
# most pixels sit in the middle-dark range (around 108), not at full black or
# full white. On the RGB plot, red sits further to the right (brighter) than
# green and blue.
#
# The benign photo is a bit brighter and more spread out than the malignant one
# (about 128 vs 108 on average). You can see a difference, but that does not
# mean "darker histogram = cancer". Lighting, how much of the picture is lesion
# vs skin, and skin tone can all change the shape.


def q6_normalization(arr: np.ndarray):
    print("\n=== Q6  Normalization ===")
    raw = arr.astype(np.float64)
    minmax = (raw - raw.min(axis=(0, 1))) / (raw.max(axis=(0, 1)) - raw.min(axis=(0, 1)))
    zscore = (raw - raw.mean(axis=(0, 1))) / raw.std(axis=(0, 1))

    print("raw     min/max/mean:", raw.min(), raw.max(), raw.mean())
    print("min-max min/max/mean:", minmax.min(), minmax.max(), minmax.mean())
    print("z-score min/max/mean:", zscore.min(), zscore.max(), zscore.mean())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(raw.ravel(), bins=64, color="steelblue")
    axes[0].set_title("Raw [0, 255]")
    axes[1].hist(minmax.ravel(), bins=64, color="steelblue")
    axes[1].set_title("Min-max [0, 1]")
    axes[2].hist(zscore.ravel(), bins=64, color="steelblue")
    axes[2].set_title("Z-score (mean 0, std 1)")
    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Pixels")
    plt.tight_layout()
    plt.show()
    print(
        "For a neural net, use min-max to [0, 1] (or dataset z-score / ImageNet "
        "stats), not raw 0–255. Normalization puts features on a shared scale so "
        "gradients stay well-behaved and training is more stable and faster."
    )


# ### Normalization
# Min-max stretches the photo so the darkest pixel becomes 0 and the brightest
# becomes 1 (everything else sits in between). Z-score instead asks "how far is
# this pixel from the average?", so the new average is 0.
#
# The three histograms look like the same hill, just drawn on different number
# lines: 0-255, then 0-1, then centered on 0.
#
# For a neural network I would use min-max (0 to 1), not the raw 0-255 values.
# Networks train more calmly when numbers are small and on a similar scale.
# Huge 0-255 values can make learning jumpy and slow.


def main():
    meta = pd.read_csv(DATA_DIR / "metadata.csv")
    isic_id = meta.iloc[0]["isic_id"]
    arr = load_image(isic_id)
    print(f"Loaded {isic_id}  diagnosis_1={meta.iloc[0]['diagnosis_1']}")

    q1_pixels_and_resolution(arr, isic_id)
    q2_color_depth(arr)
    q3_rgb_channels(arr)
    q4_grayscale_conversion(arr)
    q5_histograms(arr, meta)
    q6_normalization(arr)


if __name__ == "__main__":
    main()
