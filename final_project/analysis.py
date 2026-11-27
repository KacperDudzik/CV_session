"""M4 error analysis and fairness check of the final model, on the test predictions.

  python analysis.py            # needs outputs/diagnosis_1/test_predictions_best.csv (evaluate.py --test)

Writes to outputs/<target>/analysis/:
  confusion_matrix.png      rows = truth (Session 5)
  per_class.csv             precision, recall, F1, support per class
  errors_by_dx11.csv        error rate per fine diagnosis (11 classes) inside the target
  misclassified.png         8 wrong test lesions with their Otsu lesion mask (Session 8)
  mask_area.csv             Otsu mask area: correct vs wrong predictions
  fairness.csv              macro-F1 per skin_tone_class and per sex, with group sizes
  learning_curves.png       training loss and validation macro-F1 per epoch, all seeds (S6)
Nothing here changes the model: the test split is only read.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import config
import metrics
import session8 as s8  # load_rgb, darkness_of, segment, otsu

SMALL_GROUP = 30  # fewer test lesions than this -> the F1 is too unstable to compare


def otsu_mask(isic_id: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = s8.load_rgb(isic_id)                          # 300 x 225, as in Session 8
    return rgb, s8.segment(s8.darkness_of(rgb), s8.otsu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=config.ANALYSIS_TARGET, choices=list(config.CLASSES))
    args = parser.parse_args()
    classes = config.CLASSES[args.target]
    out = config.OUT_DIR / args.target / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(config.OUT_DIR / args.target / "test_predictions_best.csv")
    test = pd.read_csv(config.SPLIT_DIR / "test.csv")
    df = pred.merge(test[["lesion_id", "dx11", "sex", "age_approx", "skin_tone_class"]], on="lesion_id")
    df["correct"] = df["true"] == df["pred"]
    # which fine diagnoses (11 classes) are behind the errors of the coarse task
    by_dx = df.groupby("dx11").agg(n=("correct", "size"), wrong=("correct", lambda c: int((~c).sum())))
    by_dx["error_rate"] = (by_dx["wrong"] / by_dx["n"]).round(3)
    by_dx.sort_values("error_rate", ascending=False).to_csv(out / "errors_by_dx11.csv")
    y = df["true"].map(classes.index).to_numpy()
    p = df["pred"].map(classes.index).to_numpy()

    # 1. confusion matrix + per-class numbers
    fig, ax = plt.subplots(figsize=(4 + 0.5 * len(classes), 3.5 + 0.5 * len(classes)))
    ConfusionMatrixDisplay.from_predictions(df["true"], df["pred"], labels=classes, cmap="Oranges",
                                            ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"test, macro-F1 {metrics.macro_f1(y, p, len(classes)):.3f}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    report = classification_report(df["true"], df["pred"], labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(3).to_csv(out / "per_class.csv")

    # 2. open misclassified images, with the Session 8 lesion mask
    wrong = df[~df["correct"]].sample(min(8, (~df["correct"]).sum()), random_state=0)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        rgb, mask = otsu_mask(r["isic_id"])
        ax.imshow(rgb.astype(np.uint8))
        ax.contour(mask, colors="lime", linewidths=1)
        ax.set_title(f"true {r['true']} / pred {r['pred']}\n"
                     f"p={r[r['pred']]:.2f}  mask {mask.mean():.0%}  tone {r['skin_tone_class']}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(wrong):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "misclassified.png", dpi=120)
    plt.close(fig)

    # 3. does lesion size (Otsu mask area) differ between right and wrong answers?
    df["mask_area"] = [otsu_mask(i)[1].mean() for i in df["isic_id"]]
    area = df.groupby("correct")["mask_area"].describe()[["count", "mean", "50%"]]
    area.round(3).to_csv(out / "mask_area.csv")

    # 4. fairness: macro-F1 per skin tone and per sex (brief, M4 point 3)
    rows = []
    for col in ("skin_tone_class", "sex"):
        for value, g in df.groupby(col, dropna=False):
            gy, gp = g["true"].map(classes.index), g["pred"].map(classes.index)
            rows.append({"attribute": col, "group": value, "n_lesions": len(g),
                         "macro_f1": round(metrics.macro_f1(gy, gp, len(classes)), 3),
                         "accuracy": round(float((gy == gp).mean()), 3),
                         "classes_present": int(gy.nunique()),
                         "note": "small group: unstable" if len(g) < SMALL_GROUP else ""})
    fairness = pd.DataFrame(rows)
    fairness.to_csv(out / "fairness.csv", index=False)

    # 5. learning curves of the winning experiment: where early stopping picked the epoch (S6)
    best = json.loads((config.OUT_DIR / args.target / "best_model.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for seed_dir in sorted((config.HERE / best["run_dir"]).parent.glob("seed*")):
        h = pd.read_csv(seed_dir / "history.csv", index_col="epoch")
        ax1.plot(h.index, h["train_loss"], marker=".", label=seed_dir.name)
        line, = ax2.plot(h.index, h["val_f1"], marker=".", label=seed_dir.name)
        ax2.axvline(h["val_f1"].idxmax(), color=line.get_color(), ls=":", lw=1)
    ax1.set(title="training loss (cross-entropy)", xlabel="epoch")
    ax2.set(title="validation macro-F1 (dotted: kept epoch)", xlabel="epoch")
    ax1.legend()
    fig.suptitle(best["label"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=130)
    plt.close(fig)

    print(pd.DataFrame(report).T.round(3).to_string())
    print("\nmask area by correctness\n", area.round(3).to_string())
    print("\nfairness\n", fairness.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
