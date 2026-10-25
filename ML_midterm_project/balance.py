"""Train-only resampling and targeted augmentation of minority classes."""

import numpy as np
import pandas as pd

from augment import augment_rgb
from config import CLASSES, SEED
from preprocess import load_rgb


def report_distribution(frame, label="before"):
    counts = frame["diagnosis_1"].value_counts().reindex(CLASSES, fill_value=0)
    return pd.DataFrame(
        {
            "stage": label,
            "class": CLASSES,
            "count": [int(counts[c]) for c in CLASSES],
            "share": [float(counts[c] / max(counts.sum(), 1)) for c in CLASSES],
        }
    )


def oversample_rows(train, rng=None):
    """Repeat minority-class image rows until each class matches the majority."""
    rng = np.random.default_rng(SEED if rng is None else None)
    counts = train["diagnosis_1"].value_counts()
    target = int(counts.max())
    pieces = [train]
    for label in CLASSES:
        part = train.loc[train["diagnosis_1"] == label]
        need = target - len(part)
        if need <= 0 or part.empty:
            continue
        extra = part.sample(need, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        pieces.append(extra)
    return pd.concat(pieces, ignore_index=True)


def undersample_rows(train, rng=None):
    """Downsample every class to the size of the smallest class."""
    rng = np.random.default_rng(SEED if rng is None else None)
    counts = train["diagnosis_1"].value_counts()
    target = int(counts.min())
    pieces = []
    for label in CLASSES:
        part = train.loc[train["diagnosis_1"] == label]
        pieces.append(part.sample(target, random_state=int(rng.integers(0, 1_000_000))))
    return pd.concat(pieces, ignore_index=True)


def targeted_augment_rows(train, extract_fn, copies_per_image=1, rng=None):
    """Add augmented feature rows for classes below the majority count.

    `extract_fn(rgb) -> dict` must match the feature extractor used in training.
    Only train rows are passed in. Paths must be present on the frame.
    """
    rng = np.random.default_rng(SEED if rng is None else None)
    counts = train["diagnosis_1"].value_counts()
    majority = int(counts.max())
    extras = []
    for label in CLASSES:
        part = train.loc[train["diagnosis_1"] == label]
        need = majority - len(part)
        if need <= 0 or part.empty or "path" not in part.columns:
            continue
        made = 0
        while made < need:
            row = part.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            rgb = load_rgb(row["path"])
            aug = augment_rgb(rgb, rng, mode="random")
            values = extract_fn(aug)
            values.update(
                {
                    "isic_id": f"{row['isic_id']}_aug{made}",
                    "lesion_id": row["lesion_id"],
                    "image_type": row.get("image_type", "augmented"),
                    "diagnosis_1": label,
                    "split": "train",
                    "is_dermoscopic": float(row["is_dermoscopic"]),
                    "path": row["path"],
                }
            )
            extras.append(values)
            made += copies_per_image
            if made >= need:
                break
    if not extras:
        return train.copy()
    return pd.concat([train, pd.DataFrame(extras)], ignore_index=True)


def apply_balance(train, strategy, extract_fn=None):
    before = report_distribution(train, "before")
    if strategy == "none":
        after_frame = train
    elif strategy == "oversample":
        after_frame = oversample_rows(train)
    elif strategy == "undersample":
        after_frame = undersample_rows(train)
    elif strategy == "targeted_aug":
        if extract_fn is None:
            raise ValueError("targeted_aug needs extract_fn")
        after_frame = targeted_augment_rows(train, extract_fn)
    else:
        raise ValueError(f"Unknown balance strategy: {strategy}")
    after = report_distribution(after_frame, f"after_{strategy}")
    return after_frame, pd.concat([before, after], ignore_index=True)
