"""M2 baselines on the frozen split: Session 5's 33 features + classical models and Session 6 MLPs.

  python baselines.py [--target diagnosis_1]

Features: session5.feature_vector on the full-size dermoscopic image, cached in cache/.
Scaling: mean/std of the TRAINING features only (S3 leakage rule), inside each model.
  random_forest        S5 / midterm: 300 trees, class_weight="balanced_subsample"
  logistic_regression  S5: one neuron = one straight line, class_weight="balanced"
  knn                  S5: k = 15 nearest training lesions
  svm_rbf              midterm winner: RBF SVM, C = 1, class_weight="balanced"
  mlp_64               S6: 33-64-3, Adam 1e-3, class-weighted loss, 50 epochs
  mlp_256x2_reg        S6 slide 32: 33-256-256-3 + dropout 0.3 + weight decay 1e-4, 100 epochs
Every model keeps its best validation epoch where it has epochs (early stopping, S6).
"""

import argparse
import copy
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import data
import metrics
import session5 as s5  # load_rgb, feature_vector, feature_names

LABELS = {
    "random_forest": config.RF_LABEL,
    "logistic_regression": "Logistic regression, 33 S5 features (S5)",
    "knn": "kNN (k=15), 33 S5 features (S5)",
    "svm_rbf": "SVM RBF (C=1), 33 S5 features (midterm)",
    "mlp_64": "MLP 33-64-3, 33 S5 features (S6)",
    "mlp_256x2_reg": "MLP 33-256-256-3 + dropout 0.3 + weight decay 1e-4 (S6)",
}
MLP = {"mlp_64": {"hidden": (64,), "dropout": 0.0, "weight_decay": 0.0, "epochs": 50},
       "mlp_256x2_reg": {"hidden": (256, 256), "dropout": 0.3, "weight_decay": 1e-4, "epochs": 100}}


def features(df: pd.DataFrame, name: str) -> np.ndarray:
    """33 Session 5 features per image, computed once per split."""
    path = config.CACHE_DIR / f"features_s5_{name}_{len(df)}.csv"
    if not path.is_file():
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        F = np.stack([s5.feature_vector(s5.load_rgb(i)) for i in df["isic_id"]])
        pd.DataFrame(F, columns=s5.feature_names(), index=df["isic_id"]).to_csv(path)
    return pd.read_csv(path, index_col=0).to_numpy()


def sklearn_model(name: str, seed: int):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      min_samples_leaf=5, random_state=seed, n_jobs=-1)
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced",
                                                   probability=True, random_state=seed))
    raise ValueError(name)


def make_mlp(hidden, dropout, n_classes, seed):
    """Session 6 make_mlp: Linear-ReLU(-Dropout) blocks, then the output layer."""
    torch.manual_seed(seed)
    layers, d_in = [], 33
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        d_in = h
    return nn.Sequential(*layers, nn.Linear(d_in, n_classes))


def train_mlp(name, F_train, y_train, F_val, y_val, n_classes, seed):
    """Session 6 loop: Adam, class-weighted CE, batches of 64; keep the best validation epoch."""
    cfg = MLP[name]
    mean, std = F_train.mean(0), F_train.std(0) + 1e-8               # training statistics only
    Xtr = torch.tensor((F_train - mean) / std, dtype=torch.float32)
    Xva = torch.tensor((F_val - mean) / std, dtype=torch.float32)
    ytr = torch.tensor(y_train)
    model = make_mlp(cfg["hidden"], cfg["dropout"], n_classes, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    counts = torch.bincount(ytr, minlength=n_classes).float()
    loss_fn = nn.CrossEntropyLoss(weight=len(ytr) / (n_classes * counts))
    best = (-1.0, None, -1)
    for epoch in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f1 = metrics.macro_f1(y_val, model(Xva).argmax(1).numpy(), n_classes)
        if f1 > best[0]:
            best = (f1, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    ckpt = {"state_dict": model.state_dict(), "mean": mean, "std": std, "n_classes": n_classes, **cfg}
    return ckpt, best[2]


def predict_proba(run_dir, F: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probabilities of a saved baseline (sklearn .joblib or Session 6 MLP .pt)."""
    if (run_dir / "model.joblib").is_file():
        clf = joblib.load(run_dir / "model.joblib")
        probs = np.zeros((len(F), n_classes))
        probs[:, clf.classes_] = clf.predict_proba(F)
        return probs
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    model = make_mlp(ckpt["hidden"], ckpt["dropout"], ckpt["n_classes"], 0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        x = torch.tensor((F - ckpt["mean"]) / ckpt["std"], dtype=torch.float32)
        return torch.softmax(model(x), dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=config.ANALYSIS_TARGET, choices=list(config.CLASSES))
    args = parser.parse_args()
    classes = config.CLASSES[args.target]
    splits = data.load_splits()
    F_train, F_val = features(splits["train"], "train"), features(splits["val"], "val")
    y_train = data.labels(splits["train"], args.target).numpy()
    y_val = data.labels(splits["val"], args.target).numpy()

    for name in LABELS:
        for seed in config.SEEDS:
            out = config.RUNS_DIR / args.target / name / f"seed{seed}"
            if (out / "result.json").is_file():
                print(f"skip {out} (done)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            best_epoch = None
            if name in MLP:
                ckpt, best_epoch = train_mlp(name, F_train, y_train, F_val, y_val, len(classes), seed)
                torch.save(ckpt, out / "model.pt")
            else:
                clf = sklearn_model(name, seed).fit(F_train, y_train)
                joblib.dump(clf, out / "model.joblib")
            probs = predict_proba(out, F_val, len(classes))
            pd.DataFrame(probs, columns=classes).assign(lesion_id=splits["val"]["lesion_id"]).to_csv(
                out / "val_probs.csv", index=False)
            result = {"experiment": name, "label": LABELS[name], "target": args.target, "seed": seed,
                      "arch": name, "features": "session5_33", "best_epoch": best_epoch,
                      "val_macro_f1": metrics.macro_f1(y_val, probs.argmax(1), len(classes)),
                      "val_leaderboard_f1": metrics.leaderboard_f1(y_val, probs)}
            (out / "result.json").write_text(json.dumps(result, indent=2))
            print(f"{name} seed {seed}: val macro-F1 {result['val_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
