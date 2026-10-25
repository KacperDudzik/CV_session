"""Lesion-level metrics, AUCs, importance, subgroups, and report artifacts."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import CLASSES, OUTPUT_DIR


def lesion_probabilities(model, frame, columns):
    """Mean class probability of the images that belong to each lesion."""
    proba = model.predict_proba(frame[columns].to_numpy(dtype=np.float64))
    order = [list(model.classes_).index(name) for name in CLASSES]
    aligned = pd.DataFrame(proba[:, order], columns=CLASSES)
    aligned["lesion_id"] = frame["lesion_id"].to_numpy()
    return aligned.groupby("lesion_id", sort=False)[CLASSES].mean()


def lesion_scores(model, frame, columns):
    probs = lesion_probabilities(model, frame, columns)
    truth = (
        frame.drop_duplicates("lesion_id")
        .set_index("lesion_id")
        .loc[probs.index, "diagnosis_1"]
    )
    pred = probs[CLASSES].idxmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, labels=CLASSES, average="macro", zero_division=0
    )
    per_p, per_r, per_f, support = precision_recall_fscore_support(
        truth, pred, labels=CLASSES, average=None, zero_division=0
    )
    matrix = confusion_matrix(truth, pred, labels=CLASSES)
    per_class = {
        name: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(CLASSES)
    }
    y_bin = np.vstack([(truth == name).astype(int) for name in CLASSES]).T
    try:
        roc = float(roc_auc_score(y_bin, probs.to_numpy(), average="macro", multi_class="ovr"))
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = float(
            average_precision_score(y_bin, probs.to_numpy(), average="macro")
        )
    except ValueError:
        pr_auc = float("nan")
    malignant_fn = int(((truth == "Malignant") & (pred != "Malignant")).sum())
    malignant_n = int((truth == "Malignant").sum())
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "satisfaction": float((precision + recall + f1) / 3),
        "roc_auc_macro": roc,
        "pr_auc_macro": pr_auc,
        "per_class": per_class,
        "confusion": matrix,
        "y_true": truth.to_numpy(),
        "y_pred": pred.to_numpy(),
        "malignant_false_negatives": malignant_fn,
        "malignant_false_negative_rate": float(malignant_fn / max(malignant_n, 1)),
        "probabilities": probs,
    }


def feature_importance(model, frame, columns, seed=42):
    """Logistic coefficients and/or tree importances; limited permutation fallback."""
    rows = []
    clf = model.named_steps.get("clf", model)
    # Unwrap calibrated wrappers (SVM).
    if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
        inner = getattr(clf.calibrated_classifiers_[0], "estimator", None)
        if inner is not None:
            clf = inner
    if hasattr(clf, "coef_"):
        coef = np.abs(np.asarray(clf.coef_, dtype=np.float64))
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        for name, value in zip(columns, coef):
            rows.append({"feature": name, "importance": float(value), "source": "coef"})
    if hasattr(clf, "feature_importances_"):
        for name, value in zip(columns, clf.feature_importances_):
            rows.append({"feature": name, "importance": float(value), "source": "impurity"})
    if not rows:
        # Models without native importances (e.g. calibrated SVM): proxy forest.
        from sklearn.ensemble import RandomForestClassifier

        sample = frame.sample(min(len(frame), 2000), random_state=seed)
        proxy = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1, class_weight="balanced_subsample"
        )
        proxy.fit(sample[columns].to_numpy(dtype=np.float64), sample["diagnosis_1"].to_numpy())
        for name, value in zip(columns, proxy.feature_importances_):
            rows.append({"feature": name, "importance": float(value), "source": "proxy_forest"})
    table = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return table


def subgroup_scores(model, frame, columns, meta):
    """Age / sex / Fitzpatrick subgroups when metadata is reliable."""
    lesions = frame.drop_duplicates("lesion_id")[["lesion_id", "diagnosis_1"]].copy()
    fields = [c for c in ("age_approx", "sex", "fitzpatrick_skin_type") if c in meta.columns]
    if not fields:
        return pd.DataFrame()
    attrs = meta.groupby("lesion_id", as_index=False)[fields].first()
    lesions = lesions.merge(attrs, on="lesion_id", how="left")
    probs = lesion_probabilities(model, frame, columns)
    pred = probs[CLASSES].idxmax(axis=1)
    lesions = lesions.set_index("lesion_id")
    lesions["y_pred"] = pred
    lesions["y_true"] = lesions["diagnosis_1"]

    rows = []
    if "sex" in lesions.columns:
        for sex, group in lesions.dropna(subset=["sex"]).groupby("sex"):
            rows.append(_subgroup_row("sex", sex, group))
    if "fitzpatrick_skin_type" in lesions.columns:
        for tone, group in lesions.dropna(subset=["fitzpatrick_skin_type"]).groupby("fitzpatrick_skin_type"):
            rows.append(_subgroup_row("fitzpatrick_skin_type", tone, group))
    if "age_approx" in lesions.columns:
        ages = lesions.dropna(subset=["age_approx"])
        bins = pd.cut(ages["age_approx"], bins=[0, 40, 60, 80, 120], labels=["<=40", "41-60", "61-80", "81+"])
        for label, group in ages.groupby(bins):
            rows.append(_subgroup_row("age_bin", str(label), group))
    return pd.DataFrame(rows)


def _subgroup_row(kind, value, group):
    p, r, f, _ = precision_recall_fscore_support(
        group["y_true"], group["y_pred"], labels=CLASSES, average="macro", zero_division=0
    )
    return {
        "subgroup": kind,
        "value": value,
        "n": int(len(group)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
    }


def save_test_report(
    best,
    scores,
    majority_scores,
    fractions,
    board=None,
    cv_summary=None,
    importance=None,
    subgroups=None,
    balance_table=None,
    baseline_compare=None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(scores["confusion"], index=CLASSES, columns=CLASSES)
    matrix.to_csv(OUTPUT_DIR / "confusion_test.csv")
    _save_confusion_figure(matrix)
    if importance is not None and len(importance):
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        _save_importance_figure(importance)
    if subgroups is not None and len(subgroups):
        subgroups.to_csv(OUTPUT_DIR / "subgroup_metrics.csv", index=False)
    if balance_table is not None and len(balance_table):
        balance_table.to_csv(OUTPUT_DIR / "class_balance_report.csv", index=False)
        _save_balance_figure(balance_table)
    if board is not None:
        board.to_csv(OUTPUT_DIR / "val_leaderboard.csv", index=False)
        _save_ablation(board)

    payload = {
        "model": best["model"],
        "feature_set": best["feature_set"],
        "balance": best.get("balance", "none"),
        "params": best["params_dict"],
        "validation": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "satisfaction": best["satisfaction"],
            "train_f1": best.get("train_f1"),
            "overfit_gap_f1": best.get("overfit_gap_f1"),
        },
        "test": {
            "accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "satisfaction": scores["satisfaction"],
            "roc_auc_macro": scores["roc_auc_macro"],
            "pr_auc_macro": scores["pr_auc_macro"],
            "malignant_false_negatives": scores["malignant_false_negatives"],
            "malignant_false_negative_rate": scores["malignant_false_negative_rate"],
            "per_class": scores["per_class"],
        },
        "majority_test": {
            "precision": majority_scores["precision"],
            "recall": majority_scores["recall"],
            "f1": majority_scores["f1"],
            "satisfaction": majority_scores["satisfaction"],
        },
        "cv": cv_summary,
        "baseline_compare": baseline_compare,
        "split_fractions": fractions,
    }
    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "best_config.json").write_text(
        json.dumps(
            {
                "model": best["model"],
                "feature_set": best["feature_set"],
                "balance": best.get("balance", "none"),
                "params": best["params_dict"],
                "columns": best["columns"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = _report_text(payload, matrix, importance)
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    return report


def _save_confusion_figure(matrix):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix.iloc[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_test.png", dpi=140)
    plt.close(fig)


def _save_importance_figure(importance, top=20):
    top_rows = importance.groupby("feature", as_index=False)["importance"].max()
    top_rows = top_rows.sort_values("importance", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_rows["feature"], top_rows["importance"], color="#2a6f97")
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=140)
    plt.close(fig)


def _save_balance_figure(balance_table):
    pivot = balance_table.copy()
    if "balance" in pivot.columns:
        pivot = pivot.loc[pivot["balance"] == pivot["balance"].iloc[0]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    stages = pivot["stage"].unique()
    x = np.arange(len(CLASSES))
    width = 0.35 if len(stages) > 1 else 0.5
    for i, stage in enumerate(stages[:2]):
        part = pivot.loc[pivot["stage"] == stage].set_index("class").reindex(CLASSES)
        ax.bar(x + i * width, part["count"].to_numpy(), width=width, label=stage)
    ax.set_xticks(x + width / 2, CLASSES)
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=140)
    plt.close(fig)


def _save_ablation(board):
    if board.empty:
        return
    best_per_set = (
        board.sort_values("f1", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()[["feature_set", "model", "f1", "precision", "recall", "n_features"]]
    )
    best_per_set.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)


def _report_text(payload, matrix, importance):
    test = payload["test"]
    val = payload["validation"]
    base = payload["majority_test"]
    lines = [
        "# Classical MILK10k baseline (extended)",
        "",
        "## Methods",
        "",
        "- Lesion-level 70/20/10 split, seed 42, stratified on `diagnosis_1`.",
        "- Both images of a lesion share a split; probabilities are averaged before argmax.",
        "- Original Session 5 features are preserved; new colorspace, filter-bank, GLCM/shape,",
        "  and optional Canny/Harris/LBP/HOG/SIFT descriptors are selectable feature sets.",
        "- Train-only balancing: none / oversample / undersample / targeted augmentation.",
        "- Models: majority, logistic, decision tree, kNN, SVM (linear/RBF), random forest.",
        "- Selection metric: lesion-level **macro-F1** on validation. Tie-breakers: malignant",
        "  recall, fewer features, then simpler model. Test is scored once after freezing.",
        "",
        f"Winner: `{payload['model']}` on `{payload['feature_set']}` "
        f"(balance=`{payload.get('balance')}`) with `{payload['params']}`.",
        "",
        "## Satisfaction and F1",
        "",
        "| Split | Accuracy | Precision | Recall | F1 | Satisfaction |",
        "|-------|----------|-----------|--------|----|--------------|",
        f"| Validation | — | {val['precision']:.3f} | {val['recall']:.3f} | {val['f1']:.3f} | {val['satisfaction']:.3f} |",
        f"| Test | {test['accuracy']:.3f} | {test['precision']:.3f} | {test['recall']:.3f} | {test['f1']:.3f} | {test['satisfaction']:.3f} |",
        f"| Majority on test | — | {base['precision']:.3f} | {base['recall']:.3f} | {base['f1']:.3f} | {base['satisfaction']:.3f} |",
        "",
        f"Test ROC-AUC (macro OVR): {test.get('roc_auc_macro')}",
        f"Test PR-AUC (macro): {test.get('pr_auc_macro')}",
        f"Malignant false negatives: {test.get('malignant_false_negatives')} "
        f"(rate {test.get('malignant_false_negative_rate')})",
        "",
    ]
    if payload.get("cv"):
        cv = payload["cv"]
        lines.extend(
            [
                "## Grouped cross-validation (train+val only)",
                "",
                f"- macro-F1 mean ± std: {cv['f1_mean']:.3f} ± {cv['f1_std']:.3f}",
                f"- precision mean: {cv['precision_mean']:.3f}",
                f"- recall mean: {cv['recall_mean']:.3f}",
                "",
            ]
        )
    if payload.get("baseline_compare"):
        lines.extend(["## Original vs extended", "", "```", str(payload["baseline_compare"]), "```", ""])
    lines.extend(
        [
            "## Test, per class",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|----|---------|",
        ]
    )
    for name, row in test["per_class"].items():
        lines.append(
            f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |"
        )
    lines.extend(["", "## Test confusion (rows are the truth)", "", "```", matrix.to_string(), "```", ""])
    if importance is not None and len(importance):
        top = importance.groupby("feature", as_index=False)["importance"].max().sort_values(
            "importance", ascending=False
        ).head(15)
        lines.extend(["## Top features", "", "```", top.to_string(index=False), "```", ""])
    lines.extend(
        [
            "## Limitations",
            "",
            "- Heuristic lesion masks are not clinical segmentations; no mm-calibrated diameter.",
            "- No longitudinal images, so ABCDE evolution is not computed.",
            "- Optional descriptors can be noisy (hair as edges; SIFT keypoint counts).",
            "- Indeterminate remains rare; its per-class scores are unstable.",
            "",
        ]
    )
    return "\n".join(lines)
