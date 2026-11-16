"""Session 8 — Vision Tasks: Classification, Detection & Segmentation.

Converted from `class_student.ipynb` into reusable functions you can run one by one.

Plan (as in class):
1. Boxes: formats, IoU, NMS.
2. Classical segmentation on MILK10k (darkness → blur → threshold → morphology).
3. Mask → box, Dice / mask IoU, threshold sensitivity.
4. When rules fail: Otsu, saturation, hard cases, asymmetry.
5. Pretrained Faster R-CNN and DeepLabV3 (COCO / VOC classes — wrong for lesions).

Data: MILK10K_DIR or ../milk10k. Figures go to session8/_output/.

Run examples:
  python session8.py setup
  python session8.py boxes
  python session8.py nms
  python session8.py segment
  python session8.py exercises
  python session8.py pretrained
  python session8.py all
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Paths, constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT / "milk10k"))
IMG_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output"

SEED = 0
DEMO_LESION = "ISIC_0073863"  # nevus used throughout the notebook
PINK_BCC = "ISIC_0081055"

# Notebook candidate boxes for NMS demos / checks
CAND_BOXES = [
    (72, 44, 233, 171),
    (64, 50, 222, 166),
    (80, 34, 246, 180),
    (245, 178, 285, 215),
    (52, 62, 210, 196),
]
CAND_SCORES = [0.95, 0.90, 0.80, 0.75, 0.60]


def setup() -> pd.DataFrame:
    """0.1 — find MILK10k and load metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not META_PATH.is_file() or not IMG_DIR.is_dir():
        raise FileNotFoundError(
            f"MILK10k not found at {DATA_DIR}. Set MILK10K_DIR or place milk10k next to the repo."
        )
    meta = pd.read_csv(META_PATH)
    print(len(meta), "images in the metadata")
    print("IMG_DIR =", IMG_DIR)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    return meta


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save a figure under _output/ and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ---------------------------------------------------------------------------
# 0.2 Helpers
# ---------------------------------------------------------------------------

def load_rgb(isic_id: str, size: tuple[int, int] = (300, 225)) -> np.ndarray:
    """One lesion as a float array of shape (H, W, 3), values 0–255.

    Pillow resize takes (width, height); the returned array is (height, width, 3).
    """
    img = Image.open(IMG_DIR / f"{isic_id}.jpg").convert("RGB").resize(size)
    return np.array(img, dtype=float)


def check(name: str, got, expected, tol: float = 1e-3) -> bool:
    """Print OK/FAIL for an exercise answer (notebook check helper)."""
    try:
        ok = bool(np.allclose(got, expected, atol=tol)) if np.ndim(expected) or isinstance(expected, float) else got == expected
    except Exception:
        ok = False
    print(("OK  " if ok else "FAIL") + f" {name}: got {got}, expected {expected}")
    return ok


def load_demo_lesion() -> np.ndarray:
    """Usual nevus used in the notebook demos."""
    lesion = load_rgb(DEMO_LESION)
    print(lesion.shape)  # (225, 300, 3)
    return lesion


# ---------------------------------------------------------------------------
# 1. Boxes and IoU
# ---------------------------------------------------------------------------

def xyxy_to_cxcywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Corner box (x1, y1, x2, y2) → center format (cx, cy, w, h)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def iou(box_a, box_b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes. Slide 9 / Exercise 1."""
    # Overlap box: left/top = max of lefts/tops, right/bottom = min of rights/bottoms.
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)  # no overlap → 0, never negative
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def demo_box_formats(lesion: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """1.1 — convert formats and draw a box on the lesion."""
    lesion = load_demo_lesion() if lesion is None else lesion
    box = (72, 44, 233, 171)
    print(xyxy_to_cxcywh(box))  # (152.5, 107.5, 161, 127)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(lesion.astype(np.uint8))
    ax.add_patch(
        patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, color="lime", lw=2,
        )
    )
    ax.set_title("(x1, y1) = top-left, (x2, y2) = bottom-right")
    ax.axis("off")
    savefig(fig, "01_box_xyxy.png")
    return box


def check_exercise1_iou() -> None:
    """CHECK Exercise 1 — slide example and edge cases."""
    check("slide example", iou((0, 0, 10, 10), (2, 0, 14, 10)), 80 / 140)
    check("same box", iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    check("boxes that don't touch", iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
    check("one box inside the other", iou((0, 0, 10, 10), (2, 2, 7, 7)), 25 / 100)


# ---------------------------------------------------------------------------
# 2. Non-max suppression
# ---------------------------------------------------------------------------

def nms(boxes, scores, threshold: float = 0.5) -> list[int]:
    """Keep highest-score boxes; drop duplicates with IoU > threshold. Slide 12 / Exercise 2."""
    order = list(np.argsort(scores)[::-1])  # highest score first
    keep = []
    while order:
        i = int(order.pop(0))
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= threshold]
    return keep


def demo_candidate_boxes(lesion: np.ndarray | None = None) -> None:
    """2.1 — five overlapping detector candidates on the lesion."""
    lesion = load_demo_lesion() if lesion is None else lesion
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(lesion.astype(np.uint8))
    colors = ["lime", "orange", "yellow", "cyan", "magenta"]
    for (x1, y1, x2, y2), s, c in zip(CAND_BOXES, CAND_SCORES, colors):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=c, lw=1.5))
        ax.text(x1, y1 - 3, f"{s:.2f}", color=c, fontsize=8, fontweight="bold")
    ax.set_title("5 candidate boxes")
    ax.axis("off")
    savefig(fig, "02_nms_candidates.png")


def check_exercise2_nms() -> None:
    """CHECK Exercise 2 — notebook candidates + torchvision on 50 random boxes."""
    import torch
    from torchvision.ops import nms as torch_nms

    check("our 5 candidates", nms(CAND_BOXES, CAND_SCORES), [0, 3])

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, (50, 2))
    wh = rng.uniform(20, 80, (50, 2))
    rand_boxes = np.hstack([xy, xy + wh])
    rand_scores = rng.uniform(0, 1, 50)
    expected = torch_nms(
        torch.tensor(rand_boxes, dtype=torch.float32),
        torch.tensor(rand_scores, dtype=torch.float32),
        0.5,
    ).tolist()
    check(
        "50 random boxes vs torchvision",
        nms([tuple(b) for b in rand_boxes], rand_scores),
        expected,
    )


def demo_nms_result(lesion: np.ndarray | None = None) -> list[int]:
    """2.2 — IoUs with the top box, then draw boxes that survive NMS."""
    lesion = load_demo_lesion() if lesion is None else lesion
    for i, b in enumerate(CAND_BOXES):
        print(f"box {i}  score {CAND_SCORES[i]:.2f}  IoU with box 0 = {iou(CAND_BOXES[0], b):.2f}")

    kept = nms(CAND_BOXES, CAND_SCORES)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(lesion.astype(np.uint8))
    for i in kept:
        x1, y1, x2, y2 = CAND_BOXES[i]
        ax.add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=2)
        )
    ax.set_title(f"after NMS: boxes {kept}")
    ax.axis("off")
    savefig(fig, "02_nms_kept.png")
    return kept


# ---------------------------------------------------------------------------
# 3. Segmentation from scratch
# ---------------------------------------------------------------------------

def darkness_of(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel darkness = 255 − luminance (Session 2 weights)."""
    return 255 - (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def segment(score: np.ndarray, threshold_fn=lambda s: np.percentile(s, 78)) -> np.ndarray:
    """Largest blob where blurred score > threshold (notebook pipeline)."""
    smooth = ndimage.gaussian_filter(score, sigma=3)
    m = smooth > threshold_fn(smooth)
    m = ndimage.binary_closing(m, structure=np.ones((9, 9)))
    m = ndimage.binary_opening(m, structure=np.ones((5, 5)))
    m = ndimage.binary_fill_holes(m)
    labeled, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def demo_segmentation_steps(lesion: np.ndarray | None = None) -> np.ndarray:
    """3.1 — build a lesion mask one visible step at a time."""
    lesion = load_demo_lesion() if lesion is None else lesion
    gray = 0.299 * lesion[..., 0] + 0.587 * lesion[..., 1] + 0.114 * lesion[..., 2]
    darkness = 255 - gray
    smooth = ndimage.gaussian_filter(darkness, sigma=3)
    threshold = np.percentile(smooth, 78)
    raw_mask = smooth > threshold

    clean = ndimage.binary_closing(raw_mask, structure=np.ones((9, 9)))
    clean = ndimage.binary_opening(clean, structure=np.ones((5, 5)))
    clean = ndimage.binary_fill_holes(clean)

    labeled, n_regions = ndimage.label(clean)
    sizes = ndimage.sum(clean, labeled, range(1, n_regions + 1))
    mask = labeled == (np.argmax(sizes) + 1)
    print(f"{n_regions} region(s) after cleaning; the lesion covers {mask.mean():.0%} of the image")

    steps = [lesion.astype(np.uint8), darkness, smooth, raw_mask, mask]
    titles = ["image", "darkness", "blurred", f"threshold ({threshold:.0f})", "cleaned, largest region"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for ax, im, t in zip(axes, steps, titles):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t)
        ax.axis("off")
    savefig(fig, "03_segmentation_steps.png")
    return mask


def demo_segment_function(lesion: np.ndarray | None = None) -> np.ndarray:
    """3.2 — same pipeline packaged as segment()."""
    lesion = load_demo_lesion() if lesion is None else lesion
    mask = segment(darkness_of(lesion))
    print(mask.shape, mask.dtype, f"{mask.mean():.0%} of the pixels")
    return mask


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight (x1, y1, x2, y2) around True pixels; x2/y2 are exclusive. Exercise 3."""
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def check_exercise3_mask_to_bbox() -> None:
    """CHECK Exercise 3 — toy mask bbox and width."""
    toy = np.zeros((8, 8), dtype=bool)
    toy[2:5, 1:6] = True
    check("toy mask", mask_to_bbox(toy), (1, 2, 6, 5))
    check("box width = 5 pixels", mask_to_bbox(toy)[2] - mask_to_bbox(toy)[0], 5)


def demo_three_tasks(lesion: np.ndarray | None = None, mask: np.ndarray | None = None) -> tuple:
    """3.3 — same lesion, three outputs: label / box / mask overlay."""
    lesion = load_demo_lesion() if lesion is None else lesion
    mask = segment(darkness_of(lesion)) if mask is None else mask
    box = mask_to_bbox(mask)

    overlay = lesion.copy()
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([230.0, 30.0, 30.0])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(lesion.astype(np.uint8))
    axes[0].set_title("classification: one label")
    axes[0].text(5, 15, "Benign (example)", color="white", backgroundcolor="black", fontsize=9)
    axes[1].imshow(lesion.astype(np.uint8))
    axes[1].set_title(f"detection: box {box}")
    axes[1].add_patch(
        patches.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, color="lime", lw=2)
    )
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("segmentation: one label per pixel")
    savefig(fig, "03_three_tasks.png")
    return box, mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """|A ∩ B| / |A ∪ B|. Slides 22–23 / Exercise 4."""
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """2·|A ∩ B| / (|A| + |B|). Slides 22–23 / Exercise 4."""
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


def check_exercise4_mask_metrics() -> None:
    """CHECK Exercise 4 — slide example (800 overlap of 900 vs 1000)."""
    true_mask = np.zeros(2000, dtype=bool)
    true_mask[0:1000] = True
    pred_mask = np.zeros(2000, dtype=bool)
    pred_mask[200:1100] = True
    check("IoU  = 800 / 1100", mask_iou(pred_mask, true_mask), 800 / 1100)
    check("Dice = 1600 / 1900", dice(pred_mask, true_mask), 1600 / 1900)


def demo_threshold_sensitivity(lesion: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """3.4 — segment at percentiles 70 / 78 / 85; compare agreement."""
    lesion = load_demo_lesion() if lesion is None else lesion
    masks = {
        p: segment(darkness_of(lesion), lambda s, p=p: np.percentile(s, p))
        for p in (70, 78, 85)
    }
    for p, m in masks.items():
        print(
            f"percentile {p}: area {m.mean():.0%}   "
            f"Dice vs 78 = {dice(m, masks[78]):.2f}   "
            f"IoU vs 78 = {mask_iou(m, masks[78]):.2f}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (p, m) in zip(axes, masks.items()):
        ax.imshow(lesion.astype(np.uint8))
        ax.contour(m, colors="lime", linewidths=1.5)
        ax.set_title(f"percentile {p}")
        ax.axis("off")
    savefig(fig, "03_threshold_sensitivity.png")
    return masks


# ---------------------------------------------------------------------------
# 4. When the simple rule fails
# ---------------------------------------------------------------------------

def otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold: maximize w0·w1·(μ0 − μ1)² over histogram cuts."""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    return float(centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def demo_otsu(lesion: np.ndarray | None = None, ref_mask: np.ndarray | None = None) -> np.ndarray:
    """4.1 — automatic threshold from the darkness histogram."""
    lesion = load_demo_lesion() if lesion is None else lesion
    blurred = ndimage.gaussian_filter(darkness_of(lesion), 3)
    t = otsu(blurred)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.hist(blurred.ravel(), bins=100, color="#BBBBBB")
    ax.axvline(t, color="#F08200", lw=2, label=f"Otsu threshold = {t:.0f}")
    ax.legend()
    ax.set_xlabel("blurred darkness")
    savefig(fig, "04_otsu_histogram.png")

    mask_otsu = segment(darkness_of(lesion), otsu)
    if ref_mask is None:
        ref_mask = segment(darkness_of(lesion), lambda s: np.percentile(s, 78))
    print(f"Otsu vs 78th percentile: Dice = {dice(mask_otsu, ref_mask):.2f}")
    return mask_otsu


def demo_pink_darkness_fails() -> tuple[np.ndarray, np.ndarray]:
    """4.2 — darkness rule fails on a pink BCC."""
    pink = load_rgb(PINK_BCC)
    mask_dark = segment(darkness_of(pink), otsu)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(pink.astype(np.uint8))
    axes[0].set_title("pink BCC")
    axes[1].imshow(pink.astype(np.uint8))
    axes[1].contour(mask_dark, colors="lime", linewidths=1.5)
    axes[1].set_title("darkness mask")
    for ax in axes:
        ax.axis("off")
    savefig(fig, "04_pink_darkness_fail.png")
    return pink, mask_dark


def saturation_of(rgb: np.ndarray) -> np.ndarray:
    """(max − min) / max per pixel. Slide 30 / Exercise 5."""
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    return (mx - mn) / np.maximum(mx, 1e-8)  # avoid / 0 on black pixels


def check_exercise5_saturation(
    pink: np.ndarray | None = None, mask_dark: np.ndarray | None = None
) -> np.ndarray:
    """CHECK Exercise 5 — formula on gray/red/pink pixels + compare masks."""
    px = np.array([[[128.0, 128.0, 128.0], [255.0, 0.0, 0.0], [240.0, 160.0, 200.0]]])
    check(
        "saturation of gray, red, pink",
        np.round(saturation_of(px)[0], 3),
        np.array([0.0, 1.0, 0.333]),
    )

    if pink is None or mask_dark is None:
        pink, mask_dark = demo_pink_darkness_fails()
    mask_sat = segment(saturation_of(pink), otsu)
    print(f"Dice between the darkness and saturation masks: {dice(mask_dark, mask_sat):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, m, t in zip(axes, [mask_dark, mask_sat], ["darkness", "saturation"]):
        ax.imshow(pink.astype(np.uint8))
        ax.contour(m, colors="lime", linewidths=1.5)
        ax.set_title(t)
        ax.axis("off")
    savefig(fig, "04_saturation_vs_darkness.png")
    return mask_sat


def demo_hard_cases() -> None:
    """4.3 — darkness vs saturation on four difficult lesions."""
    hard = {
        "ISIC_0077054": "BCC with hairs",
        "ISIC_0076255": "faint BCC",
        "ISIC_0124902": "two lesions in one photo",
        "ISIC_0156507": "melanoma",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (i, title) in zip(axes, hard.items()):
        rgb = load_rgb(i)
        m_dark = segment(darkness_of(rgb), otsu)
        m_sat = segment(saturation_of(rgb), otsu)
        ax.imshow(rgb.astype(np.uint8))
        ax.contour(m_dark, colors="lime", linewidths=1.5)
        ax.contour(m_sat, colors="cyan", linewidths=1.5)
        ax.set_title(f"{title}\nDice dark vs sat = {dice(m_dark, m_sat):.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("lime = darkness, cyan = saturation")
    savefig(fig, "04_hard_cases.png")


def demo_area_by_diagnosis(meta: pd.DataFrame | None = None, n: int = 300) -> pd.DataFrame:
    """4.4 — lesion area fraction over n dermoscopic images, by diagnosis_1."""
    if meta is None:
        meta = setup()
    der = meta[meta["image_type"] == "dermoscopic"].sample(n, random_state=SEED).copy()
    t0 = time.perf_counter()
    der["area"] = [segment(darkness_of(load_rgb(i)), otsu).mean() for i in der["isic_id"]]
    print(f"{time.perf_counter() - t0:.0f} s")
    summary = der.groupby("diagnosis_1")["area"].describe()[["count", "mean", "50%"]].round(2)
    print(summary)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    order = summary.sort_values("mean").index.tolist()
    means = summary.loc[order, "mean"]
    ax.barh(order, means, color="#888888")
    ax.set_xlabel("mean lesion area fraction")
    ax.set_title(f"darkness+Otsu area on {n} dermoscopic images")
    savefig(fig, "04_area_by_diagnosis.png")

    out = OUTPUT_DIR / "area_by_diagnosis.csv"
    summary.to_csv(out)
    print("wrote", out)
    return summary


def asymmetry(m: np.ndarray) -> float:
    """1 − Dice(crop, left-right flip). Slide 32 / Exercise 6."""
    x1, y1, x2, y2 = mask_to_bbox(m)
    crop = m[y1:y2, x1:x2]
    return 1 - dice(crop, np.fliplr(crop))


def check_exercise6_asymmetry(mask: np.ndarray | None = None) -> None:
    """CHECK Exercise 6 — rectangle vs L-shape, then our lesion."""
    rect = np.zeros((10, 10), dtype=bool)
    rect[2:8, 3:7] = True
    ell = np.zeros((10, 10), dtype=bool)
    ell[2:8, 2:4] = True
    ell[6:8, 2:8] = True
    check("symmetric rectangle", asymmetry(rect), 0.0)
    check("L shape", round(asymmetry(ell), 3), 0.4)
    if mask is None:
        mask = segment(darkness_of(load_rgb(DEMO_LESION)))
    print(f"our lesion: asymmetry = {asymmetry(mask):.2f}")


# ---------------------------------------------------------------------------
# 5. Pretrained detection / segmentation
# ---------------------------------------------------------------------------

def demo_faster_rcnn(isic_id: str = DEMO_LESION) -> dict:
    """5.1 — Faster R-CNN (COCO) on a lesion; classes will be wrong for skin."""
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    detector = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval()

    big = load_rgb(isic_id, size=(600, 450))
    x = torch.from_numpy(big).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        pred = detector([x])[0]

    names = weights.meta["categories"]
    for b, l, s in zip(pred["boxes"][:3], pred["labels"][:3], pred["scores"][:3]):
        print(f"{names[l]:10s} confidence {s:.2f}  box {b.round().tolist()}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(big.astype(np.uint8))
    for b, l, s in zip(pred["boxes"][:5], pred["labels"][:5], pred["scores"][:5]):
        if float(s) < 0.3:
            continue
        x1, y1, x2, y2 = b.tolist()
        ax.add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=1.5)
        )
        ax.text(x1, max(y1 - 4, 8), f"{names[int(l)]} {float(s):.2f}", color="lime", fontsize=8)
    ax.set_title("Faster R-CNN (COCO classes)")
    ax.axis("off")
    savefig(fig, "05_faster_rcnn.png")
    return {"boxes": pred["boxes"], "labels": pred["labels"], "scores": pred["scores"]}


def demo_deeplabv3(isic_id: str = DEMO_LESION) -> np.ndarray:
    """5.2 — DeepLabV3 (VOC-style classes) on a lesion."""
    import torch
    from torchvision.models.segmentation import (
        DeepLabV3_MobileNet_V3_Large_Weights,
        deeplabv3_mobilenet_v3_large,
    )

    sweights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    segmenter = deeplabv3_mobilenet_v3_large(weights=sweights).eval()

    big = load_rgb(isic_id, size=(600, 450))
    x = torch.from_numpy(big).permute(2, 0, 1).float() / 255
    # DeepLab's preprocess resizes; map predictions back to the display size.
    with torch.no_grad():
        out = segmenter(sweights.transforms()(x).unsqueeze(0))["out"]
    pred_mask = out.argmax(dim=1)[0]

    classes = sweights.meta["categories"]
    print(tuple(out.shape))
    print(
        {
            classes[k]: f"{(pred_mask == k).float().mean():.0%}"
            for k in pred_mask.unique().tolist()
        }
    )

    # Resize class map back to the image we displayed (preprocess changes size).
    pm = pred_mask.cpu().numpy().astype(np.uint8)
    pm = np.array(Image.fromarray(pm).resize((big.shape[1], big.shape[0]), Image.NEAREST))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(big.astype(np.uint8))
    axes[0].set_title("input")
    axes[1].imshow(pm, cmap="tab20", interpolation="nearest")
    axes[1].set_title("DeepLabV3 class map (COCO/VOC — wrong for lesions)")
    for ax in axes:
        ax.axis("off")
    savefig(fig, "05_deeplabv3.png")
    return pm


# ---------------------------------------------------------------------------
# Section runners
# ---------------------------------------------------------------------------

def run_boxes() -> None:
    """Section 1: box formats + IoU exercise checks."""
    setup()
    demo_box_formats()
    check_exercise1_iou()


def run_nms() -> None:
    """Section 2: candidates, NMS checks, kept boxes."""
    setup()
    lesion = load_demo_lesion()
    demo_candidate_boxes(lesion)
    check_exercise2_nms()
    demo_nms_result(lesion)


def run_segment() -> None:
    """Section 3: classical pipeline, three tasks, threshold sensitivity."""
    setup()
    lesion = load_demo_lesion()
    mask = demo_segmentation_steps(lesion)
    demo_segment_function(lesion)
    check_exercise3_mask_to_bbox()
    demo_three_tasks(lesion, mask)
    check_exercise4_mask_metrics()
    demo_threshold_sensitivity(lesion)


def run_failures() -> None:
    """Section 4: Otsu, pink BCC, saturation, hard cases, asymmetry."""
    setup()
    lesion = load_demo_lesion()
    mask = segment(darkness_of(lesion))
    demo_otsu(lesion, mask)
    pink, mask_dark = demo_pink_darkness_fails()
    check_exercise5_saturation(pink, mask_dark)
    demo_hard_cases()
    check_exercise6_asymmetry(mask)


def run_area(n: int = 300) -> None:
    """Section 4.4 — area-by-diagnosis (slower; samples n images)."""
    meta = setup()
    demo_area_by_diagnosis(meta, n=n)


def run_pretrained() -> None:
    """Section 5: Faster R-CNN + DeepLabV3 (downloads weights first time)."""
    setup()
    demo_faster_rcnn()
    demo_deeplabv3()


def run_exercises() -> None:
    """All exercise CHECK cells only (needs MILK10k for lesion asymmetry print)."""
    setup()
    check_exercise1_iou()
    check_exercise2_nms()
    check_exercise3_mask_to_bbox()
    check_exercise4_mask_metrics()
    check_exercise5_saturation()
    check_exercise6_asymmetry()


def run_all(include_area: bool = False, include_pretrained: bool = True) -> None:
    """Full notebook path in order."""
    meta = setup()
    lesion = load_demo_lesion()

    print("\n=== 1. Boxes and IoU ===")
    demo_box_formats(lesion)
    check_exercise1_iou()

    print("\n=== 2. NMS ===")
    demo_candidate_boxes(lesion)
    check_exercise2_nms()
    demo_nms_result(lesion)

    print("\n=== 3. Segmentation ===")
    mask = demo_segmentation_steps(lesion)
    demo_segment_function(lesion)
    check_exercise3_mask_to_bbox()
    demo_three_tasks(lesion, mask)
    check_exercise4_mask_metrics()
    demo_threshold_sensitivity(lesion)

    print("\n=== 4. When rules fail ===")
    demo_otsu(lesion, mask)
    pink, mask_dark = demo_pink_darkness_fails()
    check_exercise5_saturation(pink, mask_dark)
    demo_hard_cases()
    check_exercise6_asymmetry(mask)
    if include_area:
        demo_area_by_diagnosis(meta)

    if include_pretrained:
        print("\n=== 5. Pretrained models ===")
        demo_faster_rcnn()
        demo_deeplabv3()


COMMANDS = {
    "setup": setup,
    "boxes": run_boxes,
    "nms": run_nms,
    "segment": run_segment,
    "failures": run_failures,
    "area": run_area,
    "pretrained": run_pretrained,
    "exercises": run_exercises,
    "all": run_all,
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print("Usage: python session8.py <command>")
        print("Commands:", ", ".join(COMMANDS))
        return
    cmd = argv[0]
    if cmd not in COMMANDS:
        raise SystemExit(f"Unknown command {cmd!r}. Choose from: {', '.join(COMMANDS)}")
    COMMANDS[cmd]()


if __name__ == "__main__":
    main()
