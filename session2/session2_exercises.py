from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
img = Image.open(sorted((ROOT / "milk10k" / "images").glob("*.jpg"))[0])
plt.imshow(img)
plt.axis("off")
plt.show()
arr = np.array(img)

w, h = img.size  # PIL: (W, H)
H, W, C = arr.shape  # NumPy: (H, W, C)

# print("PIL  size :", img.size, "  (W, H)")
# print("NumPy shape:", arr.shape, "(H, W, C)")
# print("same order (H, W)?", img.size == (H, W))
# print("swapped (W, H)?   ", (w, h) == (W, H))

print("mean intensity R, G, B:", arr.mean(axis=(0, 1)))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, i, name in zip(axes, range(3), ["Red", "Green", "Blue"]):
    ch = np.zeros_like(arr)
    ch[:, :, i] = arr[:, :, i]
    ax.imshow(ch)
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, i, name, color in zip(axes, range(3), ["Red", "Green", "Blue"], ["r", "g", "b"]):
    ax.hist(arr[:, :, i].ravel(), bins=256, range=(0, 256), color=color)
    ax.set_title(name)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
plt.tight_layout()
plt.show()

print("pixels with intensity 45  R, G, B:", (arr == 45).sum(axis=(0, 1)))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, i, name, color in zip(axes, [1, 2], ["Green", "Blue"], ["g", "b"]):
    ax.hist(arr[:, :, i].ravel(), bins=256, range=(0, 256), color=color)
    ax.set_title(name)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
plt.tight_layout()
plt.show()

# Gray = 0.299*R + 0.587*G + 0.114*B
gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
gray_cv2 = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(arr)
axes[0].set_title("RGB")
axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Grayscale")
axes[2].imshow(gray_cv2, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Grayscale (cv2)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

arr_f = arr.astype(np.float64)
minmax = (arr_f - arr_f.min(axis=(0, 1))) / (arr_f.max(axis=(0, 1)) - arr_f.min(axis=(0, 1)))
standard = (arr_f - arr_f.mean(axis=(0, 1))) / arr_f.std(axis=(0, 1))
print("original  min/max/mean/std:", arr_f.min(axis=(0, 1)), arr_f.max(axis=(0, 1)), arr_f.mean(axis=(0, 1)), arr_f.std(axis=(0, 1)))
print("min-max   min/max/mean/std:", minmax.min(axis=(0, 1)), minmax.max(axis=(0, 1)), minmax.mean(axis=(0, 1)), minmax.std(axis=(0, 1)))
print("standard  min/max/mean/std:", standard.min(axis=(0, 1)), standard.max(axis=(0, 1)), standard.mean(axis=(0, 1)), standard.std(axis=(0, 1)))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(arr)
axes[0].set_title("RGB")
axes[1].imshow(minmax)
axes[1].set_title("Min-max [0, 1]")
axes[2].imshow(np.clip((standard + 3) / 6, 0, 1))
axes[2].set_title("Standardized (z-score)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

colors = [
    ("white", [255, 255, 255]),
    ("black", [0, 0, 0]),
    ("red", [255, 0, 0]),
    ("green", [0, 255, 0]),
    ("blue", [0, 0, 255]),
]
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, (name, rgb) in zip(axes, colors):
    ax.imshow(np.full((5, 5, 3), rgb, dtype=np.uint8))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.show()
