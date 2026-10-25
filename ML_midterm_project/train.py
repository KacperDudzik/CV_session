"""Cache image features, fit model/feature pairs, keep the validation winner."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from balance import apply_balance
from config import (
    CV_FOLDS,
    FEATURE_CACHE_PATH,
    FEATURE_CACHE_VERSION,
    MODEL_RANK,
    OUTPUT_DIR,
    PRIMARY_METRIC,
    SEED,
)
from features import ALL_NAMES, FEATURE_SETS, FULL_NAMES, extract_all_features, extract_features
from models import candidates
from preprocess import load_rgb

# Re-export for older imports; cache path is versioned in config.
CACHE_PATH = FEATURE_CACHE_PATH
META_COLUMNS = ["isic_id", "lesion_id", "image_type", "diagnosis_1", "split", "is_dermoscopic"]


def _row_from_image(row):
    rgb = load_rgb(row.path)
    values = extract_all_features(rgb)
    values["isic_id"] = row.isic_id
    values["is_dermoscopic"] = float(row.is_dermoscopic)
    return values


def load_or_extract(images):
    """Feature table for every image. Reuses the versioned cache when complete."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    needed = set(images["isic_id"])
    stored = _read_cache()
    if stored is not None:
        have = set(stored["isic_id"])
        missing = images.loc[~images["isic_id"].isin(have)]
    else:
        stored = pd.DataFrame()
        missing = images
    if len(missing):
        print(f"extracting features for {len(missing)} images ({len(needed) - len(missing)} cached)")
        stored = _extract_rows(missing, stored)
    stored["is_dermoscopic"] = stored["is_dermoscopic"].astype(np.float64)
    stored = stored.drop(columns=[c for c in ("split",) if c in stored.columns])
    merged = images[META_COLUMNS + (["path"] if "path" in images.columns else [])].merge(
        stored, on=["isic_id", "is_dermoscopic"], how="left"
    )
    required = list(ALL_NAMES) + list(FULL_NAMES)
    required = sorted(set(required))
    absent = [c for c in required if c not in merged.columns]
    if absent:
        raise RuntimeError(f"Feature cache is missing columns: {absent[:8]}...")
    if merged[ALL_NAMES].isna().any().any():
        raise RuntimeError("Feature cache is missing original feature values.")
    return merged


def _read_cache():
    if not CACHE_PATH.is_file():
        return None
    stored = pd.read_csv(CACHE_PATH)
    expected = {"isic_id", "is_dermoscopic", *ALL_NAMES, *FULL_NAMES}
    if not expected.issubset(stored.columns):
        print(f"cache {CACHE_PATH.name} is stale for version {FEATURE_CACHE_VERSION}; regenerating")
        return None
    return stored


def _extract_rows(images, stored):
    rows = []
    started = time.perf_counter()
    total = len(images)
    for i, row in enumerate(images.itertuples(index=False), start=1):
        rows.append(_row_from_image(row))
        if i % 200 == 0 or i == total:
            stored = _write_cache(stored, rows)
            rows = []
            rate = i / max(time.perf_counter() - started, 1e-6)
            print(f"  {i}/{total}  {rate:.1f} img/s")
    return stored


def _write_cache(stored, rows):
    fresh = pd.DataFrame(rows)
    stored = pd.concat([stored, fresh], ignore_index=True)
    stored = stored.drop_duplicates("isic_id", keep="last")
    stored.to_csv(CACHE_PATH, index=False)
    return stored


def _xy(frame, columns):
    return frame[columns].to_numpy(dtype=np.float64), frame["diagnosis_1"].to_numpy()


def search(frame, model_specs=None, feature_sets=None, balance_strategies=None):
    """Fit candidates on train; rank by lesion-level macro-F1 on validation.

    Satisfaction remains reported. Tie-breakers: malignant recall, fewer
    features, then simpler model.
    """
    from evaluate import lesion_scores

    specs = candidates(SEED) if model_specs is None else model_specs
    sets = FEATURE_SETS if feature_sets is None else feature_sets
    strategies = ("none",) if balance_strategies is None else balance_strategies
    train_base = frame.loc[frame["split"] == "train"].copy()
    val = frame.loc[frame["split"] == "val"]
    records = []
    best = None
    balance_reports = []

    for strategy in strategies:
        train, dist = apply_balance(
            train_base,
            strategy,
            extract_fn=extract_all_features if strategy == "targeted_aug" else None,
        )
        balance_reports.append(dist.assign(balance=strategy))
        for model_name, estimator, grid in specs:
            for params in ParameterGrid(grid):
                for set_name, columns in sets.items():
                    missing = [c for c in columns if c not in train.columns]
                    if missing:
                        continue
                    model = clone(estimator).set_params(**params)
                    x_train, y_train = _xy(train, columns)
                    started = time.perf_counter()
                    model.fit(x_train, y_train)
                    scores = lesion_scores(model, val, columns)
                    train_scores = lesion_scores(model, train_base, columns)
                    row = {
                        "model": model_name,
                        "feature_set": set_name,
                        "balance": strategy,
                        "n_features": len(columns),
                        "params": str(params),
                        "accuracy": scores.get("accuracy", float("nan")),
                        "precision": scores["precision"],
                        "recall": scores["recall"],
                        "f1": scores["f1"],
                        "satisfaction": scores["satisfaction"],
                        "malignant_recall": scores["per_class"]["Malignant"]["recall"],
                        "train_f1": train_scores["f1"],
                        "overfit_gap_f1": train_scores["f1"] - scores["f1"],
                        "seconds": time.perf_counter() - started,
                    }
                    records.append(row)
                    print(
                        f"{model_name:10} {set_name:22} {strategy:12} "
                        f"F1={row['f1']:.3f} sat={row['satisfaction']:.3f} "
                        f"P={row['precision']:.3f} R={row['recall']:.3f}"
                    )
                    if _better(row, best):
                        best = {
                            **row,
                            "estimator": model,
                            "columns": columns,
                            "params_dict": params,
                        }

    board = pd.DataFrame(records)
    board["model_rank"] = board["model"].map(MODEL_RANK).fillna(99)
    board = board.sort_values(
        [PRIMARY_METRIC, "malignant_recall", "n_features", "model_rank"],
        ascending=[False, False, True, True],
    ).drop(columns="model_rank")
    board = board.reset_index(drop=True)
    balance_table = pd.concat(balance_reports, ignore_index=True) if balance_reports else pd.DataFrame()
    return board, best, balance_table


def grouped_cv(frame, best, n_splits=CV_FOLDS):
    """Stratified lesion-grouped CV on train+val only (test never touched)."""
    from evaluate import lesion_scores

    usable = frame.loc[frame["split"].isin(["train", "val"])].copy()
    lesions = usable.drop_duplicates("lesion_id")
    y = lesions["diagnosis_1"].to_numpy()
    groups = lesions["lesion_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    columns = best["columns"]
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(lesions)), y, groups)):
        train_ids = set(lesions.iloc[tr]["lesion_id"])
        val_ids = set(lesions.iloc[va]["lesion_id"])
        train = usable.loc[usable["lesion_id"].isin(train_ids)]
        val = usable.loc[usable["lesion_id"].isin(val_ids)]
        model = clone(best["estimator"])
        model.fit(*_xy(train, columns))
        scores = lesion_scores(model, val, columns)
        rows.append(
            {
                "fold": fold,
                "f1": scores["f1"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "satisfaction": scores["satisfaction"],
                "malignant_recall": scores["per_class"]["Malignant"]["recall"],
            }
        )
        print(f"cv fold {fold}: F1={scores['f1']:.3f}")
    cv = pd.DataFrame(rows)
    summary = {
        "f1_mean": float(cv["f1"].mean()),
        "f1_std": float(cv["f1"].std(ddof=0)),
        "precision_mean": float(cv["precision"].mean()),
        "recall_mean": float(cv["recall"].mean()),
        "satisfaction_mean": float(cv["satisfaction"].mean()),
    }
    return cv, summary


def _better(row, best):
    if best is None:
        return True
    key = (
        row[PRIMARY_METRIC],
        row["malignant_recall"],
        -row["n_features"],
        -MODEL_RANK.get(row["model"], 99),
    )
    other = (
        best[PRIMARY_METRIC],
        best["malignant_recall"],
        -best["n_features"],
        -MODEL_RANK.get(best["model"], 99),
    )
    return key > other
