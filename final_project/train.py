"""Train experiments from config.EXPERIMENTS on the frozen split (M3/M4, and the 11-class M5 model).

  python train.py                                   # every experiment, every seed, diagnosis_1
  python train.py --exp resnet18_layer3_4 --seeds 0 # one run
  python train.py --target dx11 --exp best          # the validation winner, retrained with 11 outputs

Each run writes runs/<target>/<experiment>/seed<k>/: model.pt, history.csv, val_probs.csv,
result.json. A finished run is skipped, so the script can be stopped and restarted.
The test split is never touched here.
"""

import argparse
import json
import time

import pandas as pd
import torch

import config
import data
import metrics
import session7 as s7  # fit(): Session 6/7 loop, class-weighted loss, flips, best-val checkpoint
from models import build_model, predict_proba


def run_dir(target: str, exp: str, seed: int):
    return config.RUNS_DIR / target / exp / f"seed{seed}"


def train_one(d: dict, target: str, exp: str, seed: int) -> None:
    out = run_dir(target, exp, seed)
    if (out / "result.json").is_file():
        print(f"skip {target}/{exp}/seed{seed} (done)")
        return
    spec = config.EXPERIMENTS[exp]
    classes = d["classes"]
    print(f"\n=== {target} / {exp} / seed {seed} ===")

    norm_stats = data.train_norm_stats() if spec.get("norm") == "train" else None
    model, params = build_model(spec, len(classes), seed, norm_stats)
    epochs = spec.get("epochs", config.EPOCHS)
    torch.manual_seed(seed)  # data order and flips depend on the seed
    t0 = time.perf_counter()
    hist, best = s7.fit(model, params, d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                        epochs=epochs, lr=spec["lr"],
                        batch_size=spec.get("batch_size", config.BATCH_SIZE),
                        weight_decay=spec.get("weight_decay", 0.0),
                        augment_fn=data.AUGMENTATIONS[spec.get("augment", "flips")],
                        class_weighted=spec.get("class_weights", True),
                        oversample=spec.get("oversample", False))
    minutes = (time.perf_counter() - t0) / 60

    probs = predict_proba(model, d["X_val"])
    y_val = d["y_val"].numpy()
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "arch": spec["arch"], "train": spec["train"], "spec": spec, "classes": classes,
                "target": target, "experiment": exp, "seed": seed,
                "image_size": config.IMAGE_SIZE}, out / "model.pt")
    hist.to_csv(out / "history.csv")
    val_ids = d["splits"]["val"]["lesion_id"]
    pd.DataFrame(probs.numpy(), columns=classes).assign(lesion_id=val_ids).to_csv(out / "val_probs.csv", index=False)
    result = {
        "experiment": exp, "target": target, "seed": seed, **spec,
        "epochs": epochs, "best_epoch": int(best["epoch"]),
        "n_trainable": int(sum(p.numel() for p in params)),
        "val_macro_f1": metrics.macro_f1(y_val, probs.argmax(1).numpy(), len(classes)),
        "val_leaderboard_f1": metrics.leaderboard_f1(y_val, probs.numpy()),
        "minutes": round(minutes, 2),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"val macro-F1 {result['val_macro_f1']:.3f}  (best epoch {best['epoch']}, {minutes:.1f} min)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=config.ANALYSIS_TARGET, choices=list(config.CLASSES))
    parser.add_argument("--exp", nargs="+", default=["all"],
                        help="experiment names, 'all', a stage (stage1, sweep, architectures, "
                             "regularization), or 'best' (the M4 winner in outputs/best_model.json)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    args = parser.parse_args()

    exps = args.exp
    if exps == ["all"]:
        exps = list(config.EXPERIMENTS)
    elif len(exps) == 1 and exps[0] in config.STAGES:
        exps = config.STAGES[exps[0]]
    elif exps == ["best"]:
        exps = [json.loads(config.BEST_JSON.read_text())["experiment"]]
    s7.setup()
    torch.backends.cudnn.benchmark = True  # fastest convolution algorithms for fixed 224x224 input
    d = data.load_data(args.target)
    for name in ("X_train", "X_val"):      # uint8 images stay on the GPU: no copy per batch
        d[name] = d[name].to(s7.device)
    print({k: tuple(v.shape) for k, v in d.items() if k.startswith("X_")})
    for exp in exps:
        for seed in args.seeds:
            train_one(d, args.target, exp, seed)


if __name__ == "__main__":
    main()
