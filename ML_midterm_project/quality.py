"""Missing-value, near-duplicate, and leakage audits before training."""

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from config import CLASSES, OUTPUT_DIR


def audit_missingness(meta):
    """Summarize NaN rates for fields used in subgroup analysis."""
    columns = [
        "age_approx",
        "sex",
        "anatom_site_general",
        "fitzpatrick_skin_type",
        "diagnosis_1",
        "lesion_id",
        "isic_id",
    ]
    present = [c for c in columns if c in meta.columns]
    rates = meta[present].isna().mean().rename("missing_rate").reset_index()
    rates.columns = ["column", "missing_rate"]
    return rates


def audit_near_duplicates(images, sample_limit=2000, seed=42):
    """Flag identical tiny-image hashes across different lesion_ids.

    Same-lesion dermoscopic/clinical pairs are expected and ignored.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    subset = images
    if len(images) > sample_limit:
        subset = images.sample(sample_limit, random_state=seed)
    buckets = defaultdict(list)
    for row in subset.itertuples(index=False):
        try:
            with Image.open(row.path) as img:
                tiny = img.convert("RGB").resize((8, 8))
                digest = hashlib.md5(tiny.tobytes()).hexdigest()
        except OSError:
            continue
        buckets[digest].append((row.lesion_id, row.isic_id))
    cross = []
    for digest, items in buckets.items():
        lesion_ids = {lesion for lesion, _ in items}
        if len(lesion_ids) > 1:
            cross.append({"hash": digest, "n_lesions": len(lesion_ids), "n_images": len(items)})
    return pd.DataFrame(cross)


def audit_leakage(lesions, meta):
    """Confirm lesion-grouped splits and warn about target-like columns."""
    parts = {
        name: set(lesions.loc[lesions["split"] == name, "lesion_id"])
        for name in ("train", "val", "test")
    }
    overlaps = {
        "train_val": len(parts["train"] & parts["val"]),
        "train_test": len(parts["train"] & parts["test"]),
        "val_test": len(parts["val"] & parts["test"]),
    }
    risky = []
    for column in ("diagnosis_confirm_type", "diagnosis_2", "diagnosis_3", "diagnosis_4"):
        if column in meta.columns:
            risky.append(column)
    pair_counts = meta.groupby("lesion_id").size()
    return {
        "overlaps": overlaps,
        "risky_columns_present": risky,
        "lesions_without_two_images": int((pair_counts != 2).sum()),
        "both_views_same_split": _both_views_same_split(meta, lesions),
    }


def _both_views_same_split(meta, lesions):
    split_of = lesions.set_index("lesion_id")["split"]
    mapped = meta[["lesion_id", "isic_id"]].copy()
    mapped["split"] = mapped["lesion_id"].map(split_of)
    n_unique = mapped.groupby("lesion_id")["split"].nunique()
    return bool((n_unique == 1).all())


def class_distribution(frame, label_col="diagnosis_1", by="split"):
    """Counts and shares before/after balancing."""
    if by is None:
        counts = frame[label_col].value_counts().reindex(CLASSES, fill_value=0)
        shares = counts / max(counts.sum(), 1)
        return pd.DataFrame({"count": counts, "share": shares})
    rows = []
    for part, group in frame.groupby(by):
        counts = group[label_col].value_counts().reindex(CLASSES, fill_value=0)
        for name in CLASSES:
            rows.append(
                {
                    by: part,
                    "class": name,
                    "count": int(counts[name]),
                    "share": float(counts[name] / max(counts.sum(), 1)),
                }
            )
    return pd.DataFrame(rows)


def save_quality_reports(meta, lesions, images):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = audit_missingness(meta)
    missing.to_csv(OUTPUT_DIR / "missingness.csv", index=False)
    dups = audit_near_duplicates(images)
    dups.to_csv(OUTPUT_DIR / "near_duplicates.csv", index=False)
    leakage = audit_leakage(lesions, meta)
    pd.Series(leakage["overlaps"]).to_csv(OUTPUT_DIR / "split_overlaps.csv", header=["n_shared"])
    dist = class_distribution(lesions)
    dist.to_csv(OUTPUT_DIR / "class_distribution_splits.csv", index=False)
    return {"missing": missing, "near_duplicates": dups, "leakage": leakage, "distribution": dist}
