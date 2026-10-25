"""Train-only geometric and photometric augmentations."""

import numpy as np
from PIL import Image, ImageEnhance


def augment_rgb(rgb, rng, mode="random"):
    """Return a label-preserving transform of an RGB float image.

    Modes: flip_h, flip_v, rotate, brightness, contrast, crop, random.
    """
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if mode == "random":
        mode = rng.choice(["flip_h", "flip_v", "rotate", "brightness", "contrast", "crop"])
    if mode == "flip_h":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == "flip_v":
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif mode == "rotate":
        angle = float(rng.uniform(-15, 15))
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    elif mode == "brightness":
        factor = float(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Brightness(image).enhance(factor)
    elif mode == "contrast":
        factor = float(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(factor)
    elif mode == "crop":
        image = _safe_center_zoom(image, rng)
    else:
        raise ValueError(f"Unknown augment mode: {mode}")
    return np.asarray(image, dtype=np.float64)


def _safe_center_zoom(image, rng):
    """Zoom crop that keeps at least 80% of the center region (lesion-safe)."""
    w, h = image.size
    scale = float(rng.uniform(0.8, 0.95))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    left = (w - nw) // 2
    top = (h - nh) // 2
    cropped = image.crop((left, top, left + nw, top + nh))
    return cropped.resize((w, h), Image.BILINEAR)
