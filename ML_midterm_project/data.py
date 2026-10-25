"""Load MILK10k and assign a lesion-level 70/20/10 split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    CLASSES,
    DX_COLS,
    GT_PATH,
    IMG_DIR,
    META_PATH,
    N_SPLITS,
    OUTPUT_DIR,
    SEED,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VAL_FOLDS,
)


def load_tables():
    meta = pd.read_csv(META_PATH)
    gt = pd.read_csv(GT_PATH)
    return meta, gt


def _with_dx(gt):
    out = gt.copy()
    out["dx"] = out[DX_COLS].idxmax(axis=1)
    return out


def build_lesions(meta, gt):
    """One row per lesion, with both image ids and the 3-class label."""
    ids = meta.pivot(index="lesion_id", columns="image_type", values="isic_id")
    ids = ids.rename(columns={"dermoscopic": "derm_id", "clinical: close-up": "clinical_id"})
    labels = meta.groupby("lesion_id", as_index=False)["diagnosis_1"].first()
    lesions = labels.merge(ids, on="lesion_id").merge(
        _with_dx(gt)[["lesion_id", "dx"]], on="lesion_id"
    )
    if lesions["derm_id"].isna().any() or lesions["clinical_id"].isna().any():
        raise RuntimeError("Every lesion needs one dermoscopic and one clinical image.")
    if not lesions["lesion_id"].is_unique:
        raise RuntimeError("lesion_id is not unique.")
    return lesions


def assign_splits(lesions):
    """10 stratified group folds: 1 test, 2 val, 7 train (10/20/70).

    Stratify on diagnosis_1. The 11-class label has a class with only 9
    lesions, which cannot fill 10 folds.
    """
    y = lesions["diagnosis_1"].to_numpy()
    groups = lesions["lesion_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = np.empty(len(lesions), dtype=int)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(lesions)), y, groups)):
        fold_of[held] = fold
    out = lesions.copy()
    out["fold"] = fold_of
    out["split"] = "train"
    out.loc[out["fold"].isin(VAL_FOLDS), "split"] = "val"
    out.loc[out["fold"].isin(TEST_FOLDS), "split"] = "test"
    unexpected = set(out["fold"]).symmetric_difference(set(TRAIN_FOLDS + VAL_FOLDS + TEST_FOLDS))
    if unexpected:
        raise RuntimeError(f"Unexpected folds: {sorted(unexpected)}")
    return out


def build_image_table(lesions):
    meta, _ = load_tables()
    split_of = lesions.set_index("lesion_id")["split"]
    keep = ["isic_id", "lesion_id", "image_type", "diagnosis_1"]
    images = meta[keep].copy()
    images["split"] = images["lesion_id"].map(split_of)
    images["is_dermoscopic"] = images["image_type"].eq("dermoscopic").astype(np.float64)
    images["path"] = images["isic_id"].map(lambda name: str(IMG_DIR / f"{name}.jpg"))
    if images["split"].isna().any():
        raise RuntimeError("An image has no lesion split.")
    if set(images["diagnosis_1"]) != set(CLASSES):
        raise RuntimeError(f"Unexpected labels: {sorted(images['diagnosis_1'].unique())}")
    # Both views of a lesion must share the split assignment.
    n_splits = images.groupby("lesion_id")["split"].nunique()
    if (n_splits != 1).any():
        raise RuntimeError("A lesion has images in more than one split.")
    return images


def assert_split(lesions):
    """No shared lesions, all three labels in each split, sizes near 70/20/10."""
    parts = {name: set(lesions.loc[lesions["split"] == name, "lesion_id"]) for name in ("train", "val", "test")}
    if parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"]:
        raise RuntimeError("A lesion appears in more than one split.")
    n = len(lesions)
    fractions = {name: len(ids) / n for name, ids in parts.items()}
    expected = {"train": 0.70, "val": 0.20, "test": 0.10}
    for name, target in expected.items():
        if abs(fractions[name] - target) > 0.02:
            raise RuntimeError(f"{name} fraction {fractions[name]:.3f} is far from {target:.2f}.")
    for name in ("train", "val", "test"):
        present = set(lesions.loc[lesions["split"] == name, "diagnosis_1"])
        if present != set(CLASSES):
            raise RuntimeError(f"{name} is missing a class: {sorted(present)}")
    return fractions


def save_split(lesions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        part = lesions.loc[lesions["split"] == name, ["lesion_id", "diagnosis_1", "dx", "fold"]]
        part.to_csv(OUTPUT_DIR / f"{name}_lesions.csv", index=False)
