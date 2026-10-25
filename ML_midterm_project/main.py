"""Train the extended classical MILK10k baseline and score the winner once on test.

Run from this folder:

    python main.py
"""

import json

import joblib
import pandas as pd

from config import BALANCE_STRATEGIES, OUTPUT_DIR, SEARCH_FEATURE_SETS, SEED
from data import assert_split, assign_splits, build_image_table, build_lesions, load_tables, save_split
from evaluate import feature_importance, lesion_scores, save_test_report, subgroup_scores
from features import FEATURE_SETS
from models.majority import build_majority
from quality import save_quality_reports
from train import grouped_cv, load_or_extract, search


def prepare():
    meta, gt = load_tables()
    lesions = assign_splits(build_lesions(meta, gt))
    fractions = assert_split(lesions)
    save_split(lesions)
    images = build_image_table(lesions)
    quality = save_quality_reports(meta, lesions, images)
    print(
        "lesions "
        + ", ".join(f"{name}={frac:.1%}" for name, frac in fractions.items())
        + f"  images={len(images)}  seed={SEED}"
    )
    print(
        "quality: "
        f"overlaps={quality['leakage']['overlaps']} "
        f"near_dup_groups={len(quality['near_duplicates'])} "
        f"both_views_same_split={quality['leakage']['both_views_same_split']}"
    )
    return images, fractions, meta


def _baseline_compare(board):
    """Best validation F1 on original extended set vs all_features."""
    rows = []
    for name in ("extended", "class_33", "all_features", "original_plus_filters"):
        part = board.loc[board["feature_set"] == name]
        if part.empty:
            continue
        best = part.iloc[0]
        rows.append(
            {
                "feature_set": name,
                "model": best["model"],
                "f1": best["f1"],
                "precision": best["precision"],
                "recall": best["recall"],
                "balance": best.get("balance", "none"),
            }
        )
    return rows


def main():
    images, fractions, meta = prepare()
    frame = load_or_extract(images)

    search_sets = {name: FEATURE_SETS[name] for name in SEARCH_FEATURE_SETS if name in FEATURE_SETS}
    board, best, balance_table = search(
        frame,
        feature_sets=search_sets,
        balance_strategies=BALANCE_STRATEGIES,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_table, cv_summary = grouped_cv(frame, best)
    cv_table.to_csv(OUTPUT_DIR / "cv_folds.csv", index=False)

    test = frame.loc[frame["split"] == "test"]
    scores = lesion_scores(best["estimator"], test, best["columns"])
    train = frame.loc[frame["split"] == "train"]
    majority, _ = build_majority()
    majority.fit(train[best["columns"]], train["diagnosis_1"])
    majority_scores = lesion_scores(majority, test, best["columns"])

    importance = feature_importance(best["estimator"], train, best["columns"], seed=SEED)
    subgroups = subgroup_scores(best["estimator"], test, best["columns"], meta)
    compare = [
        {k: (float(v) if hasattr(v, "item") else v) for k, v in row.items()}
        for row in _baseline_compare(board)
    ]

    joblib.dump(
        {
            "model": best["estimator"],
            "columns": best["columns"],
            "feature_set": best["feature_set"],
            "balance": best.get("balance", "none"),
            "params": best["params_dict"],
        },
        OUTPUT_DIR / "model.joblib",
    )
    report = save_test_report(
        best,
        scores,
        majority_scores,
        fractions,
        board=board,
        cv_summary=cv_summary,
        importance=importance,
        subgroups=subgroups,
        balance_table=balance_table,
        baseline_compare=compare,
    )
    print("\n" + report)
    summary = json.loads((OUTPUT_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    if summary["test"]["f1"] < summary["majority_test"]["f1"]:
        print("Winner does not beat the majority baseline on test macro-F1.")
    return summary


if __name__ == "__main__":
    main()
