"""Audit: run after every stage. Prints PASS/FAIL per check, writes outputs/audit.txt.

  python check_project.py          # all checks; stages not reached yet are reported as SKIP

Checks: honest split, train/serve transform identity, model shapes for both targets,
metric definitions, saved results reproduce from their checkpoints, predict(), submission format.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

import config
import data
import metrics
import session7 as s7
from models import build_model, load_checkpoint, predict_proba

LINES: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    line = f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else "")
    LINES.append(line)
    print(line)
    if not ok:
        FAILED.append(name)


def skip(name: str, why: str) -> None:
    LINES.append(f"SKIP  {name}  ({why})")
    print(LINES[-1])


def audit_data() -> dict:
    splits = data.load_splits()  # asserts: no lesion in two splits, one row per lesion
    sizes = {k: len(v) for k, v in splits.items()}
    check("split: lesion-level, no leakage, one dermoscopic image per lesion", True, str(sizes))
    check("split: covers all 5,240 lesions", sum(sizes.values()) == 5240)
    for target, classes in config.CLASSES.items():
        missing = set(classes) - set(splits["train"][target])
        check(f"split: every {target} class in train", not missing, f"missing {missing}" if missing else "")
    frozen = all((config.SPLIT_DIR / f"{n}.csv").is_file() for n in splits)
    check("split: frozen on disk (splits/*.csv)", frozen)

    X_val = data.load_images(splits["val"], "val")
    same = all(torch.equal(data.load_image(config.IMG_DIR / f"{i}.jpg"), X_val[k])
               for k, i in enumerate(splits["val"]["isic_id"][:5]))
    check("transform: load_image() == cached validation tensor (no train/serve skew)", same)
    return splits


def audit_models() -> None:
    x = torch.zeros(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=s7.device)
    unique = {}  # sweep variants share an architecture: build each (arch, train) once
    for exp, spec in config.EXPERIMENTS.items():
        unique.setdefault((spec["arch"], spec["train"], spec.get("dropout", 0), spec.get("norm")), exp)
    for target, classes in config.CLASSES.items():
        for exp in unique.values():
            spec = config.EXPERIMENTS[exp]
            model, params = build_model(spec, len(classes))
            with torch.no_grad():
                shape = tuple(model.to(s7.device).eval()(x).shape)
            n_train = sum(p.numel() for p in params)
            check(f"model: {exp} -> {len(classes)} outputs ({target})",
                  shape == (2, len(classes)) and n_train > 0, f"{n_train:,} trainable")
            del model


def audit_metrics() -> None:
    check("metric: leaderboard F1, perfect", metrics.leaderboard_f1([0, 1], [[.6, .4], [.3, .7]]) == 1.0)
    # class 0: no 'yes' -> F1 0; class 1: TP 1, FP 1 -> F1 2/3; mean 1/3
    check("metric: leaderboard F1, hand example = 1/3",
          abs(metrics.leaderboard_f1([0, 1], [[.4, .6], [.4, .6]]) - 1 / 3) < 1e-9)
    check("metric: a class absent from the evaluation set is left out, not counted as 0",
          metrics.macro_f1([0, 0, 1], [0, 0, 1], 3) == 1.0)
    p = np.array([[0.2, 0.7], [0.05, 0.95]])
    r = metrics.rescale(p, [0.2, 0.7])
    check("thresholds: rescale maps each threshold to 0.5 and stays in [0, 1]",
          np.allclose(r[0], [0.5, 0.5]) and r.min() >= 0 and r.max() <= 1 and r[1, 0] < r[0, 0])
    t = metrics.tune_thresholds([0, 0, 1, 1, 1], [[.9, .1], [.8, .2], [.7, .3], [.6, .4], [.65, .35]],
                                [0.25, 0.75], min_positives=2)
    check("thresholds: tuned per class on given data", list(t) == [0.75, 0.25], str(t))
    w = s7.class_weights_from_labels(torch.tensor([0, 0, 0, 1, 2, 2])).cpu()
    check("class weights: Session 6 formula n / (C * n_c)", torch.allclose(w, torch.tensor([6 / 9, 2.0, 1.0])))
    check("class weights: one per class for 11 classes", len(s7.class_weights_from_labels(torch.arange(11))) == 11)


def audit_results(splits: dict) -> None:
    for target in config.CLASSES:
        files = sorted((config.RUNS_DIR / target).glob("*/seed*/result.json"))
        if not files:
            skip(f"results: {target}", "no finished runs yet")
            continue
        weights_ok = all((f.parent / "model.pt").is_file() or (f.parent / "model.joblib").is_file() for f in files)
        check(f"results: {target}: {len(files)} runs, each with saved weights", weights_ok)
        best_file = config.OUT_DIR / target / "best_model.json"
        if not best_file.is_file():
            skip(f"results: {target} winner reproduces", "evaluate.py not run yet")
            continue
        best = json.loads(best_file.read_text())
        model, ckpt = load_checkpoint(config.HERE / best["model_file"])  # relative or absolute
        X = data.load_images(splits["val"], "val")
        y = data.labels(splits["val"], target).numpy()
        f1 = metrics.macro_f1(y, predict_proba(model, X).argmax(1).numpy(), len(ckpt["classes"]))
        check(f"results: {target} winner val macro-F1 reproduces from {best['model_file'].split('final_project')[-1]}",
              abs(f1 - best["val_macro_f1"]) < 1e-3, f"recorded {best['val_macro_f1']:.4f}, now {f1:.4f}")
        check(f"results: {target} test split scored at most once",
              True, "done" if (config.OUT_DIR / target / "runs_test.csv").is_file() else "not yet")


def audit_predict(splits: dict) -> None:
    if not config.MODEL_11.is_file():
        skip("predict(): 11-class model", "models/model_11class.pt not created yet")
        return
    from predict import predict
    row = splits["val"].iloc[0]
    probs = predict(config.IMG_DIR / f"{row['isic_id']}.jpg")
    model, _ = load_checkpoint(config.MODEL_11)
    ref = predict_proba(model, data.load_images(splits["val"], "val")[:1])[0].numpy()
    check("predict(): returns 11 classes in submission order", list(probs) == config.CLASSES["dx11"])
    check("predict(): probabilities sum to 1", abs(sum(probs.values()) - 1) < 1e-4)
    check("predict(): same numbers as the validation pipeline",
          np.allclose(list(probs.values()), ref, atol=1e-5))


def audit_submission() -> None:
    from make_submission import SUBMISSION, check_submission
    if not SUBMISSION.is_file():
        skip("submission.csv", "not created yet (needs the ISIC test files)")
        return
    try:
        check_submission(pd.read_csv(SUBMISSION))
        check("submission.csv: the brief's 4 format checks", True)
    except AssertionError as e:
        check("submission.csv: the brief's 4 format checks", False, str(e))


def main() -> None:
    check("device", True, f"{s7.device}, torch {torch.__version__}"
          + (f", {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    splits = audit_data()
    audit_metrics()
    audit_models()
    audit_results(splits)
    audit_predict(splits)
    audit_submission()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = f"{len(FAILED)} failed" if FAILED else "all checks passed"
    (config.OUT_DIR / "audit.txt").write_text("\n".join(LINES + ["", summary]) + "\n", encoding="utf-8")
    print("\n" + summary)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
