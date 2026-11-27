"""Scoreboard: validation chooses, the test split is scored once (M4); 11-class estimate (M5).

  python evaluate.py                  # diagnosis_1: validation scoreboard, winner -> models/model_m4.pt
  python evaluate.py --test           # + test macro-F1 of every model, written ONCE (refuses to redo)
  python evaluate.py --target dx11    # 11-class: best seed -> models/model_11class.pt + local estimate

Selection rule: the deep experiment with the highest MEAN validation macro-F1 over its seeds
wins; inside it, the seed with the highest validation macro-F1 becomes the final model.
The random forest is the M2 baseline row; it is scored but not a candidate (the final model
must be a network that can be retrained with 11 outputs).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import data
import metrics
import baselines
from models import load_checkpoint, predict_proba


def collect(target: str) -> pd.DataFrame:
    files = sorted((config.RUNS_DIR / target).glob("*/seed*/result.json"))
    if not files:
        raise SystemExit(f"no finished runs in {config.RUNS_DIR / target}: run train.py first")
    runs = pd.DataFrame([json.loads(f.read_text()) | {"dir": str(f.parent)} for f in files])
    return runs


def stage_of(exp: str) -> str:
    if exp in baselines.LABELS:
        return "M2 baseline (features)"
    return next((s for s, exps in config.STAGES.items() if exp in exps), "other")


def summarise(runs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = runs.groupby("experiment")
    table = g[["label"]].first().assign(seeds=g.size())
    table.insert(0, "stage", [stage_of(e) for e in table.index])
    for col in cols:
        table[f"{col}_mean"] = g[col].mean()
        table[f"{col}_std"] = g[col].std().fillna(0.0)
    return table.sort_values(f"{cols[0]}_mean", ascending=False)


def pick_winner(runs: pd.DataFrame) -> pd.Series:
    """Image networks only (config.EXPERIMENTS); the feature baselines are reference rows."""
    deep = runs[runs["experiment"].isin(config.EXPERIMENTS)]
    best_exp = deep.groupby("experiment")["val_macro_f1"].mean().idxmax()
    return deep[deep["experiment"] == best_exp].sort_values("val_macro_f1").iloc[-1]


def test_probs(row: pd.Series, d: dict) -> np.ndarray:
    if row["experiment"] in baselines.LABELS:
        F_test = baselines.features(d["splits"]["test"], "test")
        return baselines.predict_proba(Path(row["dir"]), F_test, len(d["classes"]))
    model, _ = load_checkpoint(f"{row['dir']}/model.pt")
    return predict_proba(model, d["X_test"]).numpy()


def leaderboard_thresholds(winner: pd.Series, final_path, best: dict) -> np.ndarray:
    """Tune one threshold per class on VALIDATION, store it in the final checkpoint."""
    classes = config.CLASSES[config.LEADERBOARD_TARGET]
    val = pd.read_csv(f"{winner['dir']}/val_probs.csv")
    split_val = data.load_splits()["val"]
    assert list(val["lesion_id"]) == list(split_val["lesion_id"])
    y_val = data.labels(split_val, config.LEADERBOARD_TARGET).numpy()
    probs = val[classes].to_numpy()
    thr = metrics.tune_thresholds(y_val, probs, config.THRESHOLD_GRID, config.MIN_VAL_POSITIVES)
    best["thresholds"] = dict(zip(classes, thr.tolist()))
    best["val_leaderboard_f1_thresholded"] = metrics.leaderboard_f1(y_val, metrics.rescale(probs, thr))
    ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
    ckpt["thresholds"] = thr.tolist()
    torch.save(ckpt, final_path)
    print(f"thresholds (validation): {best['thresholds']}")
    print(f"val leaderboard F1: raw {best['val_leaderboard_f1']:.3f}, "
          f"thresholded {best['val_leaderboard_f1_thresholded']:.3f} (optimistic: tuned on val)")
    return thr


def plot_scoreboard(table: pd.DataFrame, winner: str, path) -> None:
    """Validation macro-F1 per experiment: mean dot, +-1 std bar over seeds; winner marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("val_macro_f1_mean")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 0.27 * len(t) + 1.2))
    ax.errorbar(t["val_macro_f1_mean"], y, xerr=t["val_macro_f1_std"], fmt="o", ms=5,
                color="#257de0", ecolor="#257de0", elinewidth=1.5, capsize=0)
    w = list(t.index).index(winner)
    ax.plot(t["val_macro_f1_mean"].iloc[w], w, "o", ms=8, color="#f47230", zorder=3)
    ax.annotate("selected", (t["val_macro_f1_mean"].iloc[w], w), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=8, color="#191919")
    if "random_forest" in t.index:
        ax.axvline(t.loc["random_forest", "val_macro_f1_mean"], color="#9c9c9c", ls="--", lw=1)
        ax.text(t.loc["random_forest", "val_macro_f1_mean"], len(t) - 0.4, " M2 random forest",
                fontsize=7, color="#666666", va="bottom")
    ax.set_yticks(y, [f"{e}  [{s}]" for e, s in zip(t.index, t["stage"])], fontsize=7, color="#191919")
    ax.set_xlabel("validation macro-F1 (mean ± std over seeds)", color="#666666")
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fmt(mean: float, std: float, n: int) -> str:
    return f"{mean:.3f}" if n == 1 else f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=config.ANALYSIS_TARGET, choices=list(config.CLASSES))
    parser.add_argument("--test", action="store_true", help="score the test split (once)")
    parser.add_argument("--force", action="store_true", help="re-score the test split (say why in the report)")
    args = parser.parse_args()
    out = config.OUT_DIR / args.target
    out.mkdir(parents=True, exist_ok=True)

    runs = collect(args.target)
    winner = pick_winner(runs)
    val_table = summarise(runs, ["val_macro_f1", "val_leaderboard_f1"])
    val_table.to_csv(out / "scoreboard_val.csv")
    runs.drop(columns="dir").to_csv(out / "runs_val.csv", index=False)
    md = ["| Stage | Model | Seeds | Val macro-F1 | Val leaderboard F1 |", "|---|---|---|---|---|"]
    for _, r in val_table.iterrows():
        md.append(f"| {r['stage']} | {r['label']} | {r['seeds']} | "
                  f"{fmt(r['val_macro_f1_mean'], r['val_macro_f1_std'], r['seeds'])} | "
                  f"{fmt(r['val_leaderboard_f1_mean'], r['val_leaderboard_f1_std'], r['seeds'])} |")
    (out / "scoreboard_val.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_scoreboard(val_table, winner["experiment"], out / "scoreboard_val.png")
    print(val_table.drop(columns="label").round(3).to_string())

    final_path = config.MODEL_M4 if args.target == config.ANALYSIS_TARGET else config.MODEL_11
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{winner['dir']}/model.pt", final_path)
    best = {"target": args.target, "experiment": winner["experiment"], "label": winner["label"],
            "seed": int(winner["seed"]), "val_macro_f1": float(winner["val_macro_f1"]),
            "val_leaderboard_f1": float(winner["val_leaderboard_f1"]),
            "run_dir": Path(winner["dir"]).relative_to(config.HERE).as_posix(),
            "model_file": final_path.relative_to(config.HERE).as_posix()}  # relative: portable
    thresholds = None
    if args.target == config.LEADERBOARD_TARGET:
        thresholds = leaderboard_thresholds(winner, final_path, best)
    (out / "best_model.json").write_text(json.dumps(best, indent=2))
    if args.target == config.ANALYSIS_TARGET:
        config.BEST_JSON.write_text(json.dumps(best, indent=2))
    print(f"\nwinner: {best['experiment']} seed {best['seed']}  val macro-F1 {best['val_macro_f1']:.3f}"
          f"  -> {final_path.name}")

    if not args.test:
        return
    test_csv = out / "runs_test.csv"
    if test_csv.is_file() and not args.force:
        raise SystemExit(f"{test_csv} exists: the test split was already scored once. "
                         "Use --force only for a documented reason.")
    d = data.load_data(args.target)
    y_test, n = d["y_test"].numpy(), len(d["classes"])
    rows = []
    for _, row in runs.iterrows():
        probs = test_probs(row, d)
        rows.append(row.to_dict() | {"test_macro_f1": metrics.macro_f1(y_test, probs.argmax(1), n),
                                     "test_leaderboard_f1": metrics.leaderboard_f1(y_test, probs)})
        if row["dir"] == winner["dir"]:
            best["test_macro_f1"] = rows[-1]["test_macro_f1"]
            best["test_leaderboard_f1"] = rows[-1]["test_leaderboard_f1"]
            if thresholds is not None:  # unbiased check: thresholds were tuned on validation
                best["test_leaderboard_f1_thresholded"] = metrics.leaderboard_f1(
                    y_test, metrics.rescale(probs, thresholds))
            (out / "best_model.json").write_text(json.dumps(best, indent=2))
            if args.target == config.ANALYSIS_TARGET:
                config.BEST_JSON.write_text(json.dumps(best, indent=2))
            test_df = d["splits"]["test"][["lesion_id", "isic_id"]].assign(
                true=[d["classes"][i] for i in y_test], pred=[d["classes"][i] for i in probs.argmax(1)])
            test_df[d["classes"]] = probs
            test_df.to_csv(out / "test_predictions_best.csv", index=False)
    runs_test = pd.DataFrame(rows).drop(columns="dir")
    runs_test.to_csv(test_csv, index=False)

    table = summarise(runs_test, ["val_macro_f1", "test_macro_f1"])
    table.to_csv(out / "scoreboard.csv")
    win_test = runs_test[(runs_test["experiment"] == winner["experiment"])
                         & (runs_test["seed"] == winner["seed"])].iloc[0]
    lines = [f"Macro-F1 on {args.target} ({n} classes). Same split for every model; "
             "mean ± std over seeds; validation chooses, test scored once.", "",
             "| Stage | Model | Seeds | Val macro-F1 | Test macro-F1 |", "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['stage']} | {r['label']} | {r['seeds']} | "
                     f"{fmt(r['val_macro_f1_mean'], r['val_macro_f1_std'], r['seeds'])}"
                     f" | {fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'], r['seeds'])} |")
    lines.append(f"| final | **Best → final: {winner['label']}, seed {winner['seed']}** | 1 | "
                 f"**{win_test['val_macro_f1']:.3f}** | **{win_test['test_macro_f1']:.3f}** |")
    (out / "scoreboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
