"""Session 7 homework (slide 30).

1. Finalise Sessions 5–7 scoreboard (val + test macro-F1).
2. One sentence: DenseNet vs ResNet.
3. Optional strategy 3: fine-tune the whole ResNet-18.

Run:
  python homework.py              # train, score test once, write PDF
  python homework.py pdf-only     # rebuild PDF from saved JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

import session7 as s7

ROOT = Path(__file__).resolve().parents[1]
OUT = s7.OUTPUT_DIR
RESULTS_PATH = OUT / "homework_results.json"
PDF_PATH = Path(__file__).resolve().parent / "HOMEWORK.pdf"
REPO_URL = "https://github.com/KacperDudzik/CV_session"
MIDTERM_METRICS = ROOT / "ML_midterm_project" / "outputs" / "test_metrics.json"

# Course notebook baselines (Session 5 forest / Session 6 MLP on the same 33 features).
COURSE_S5_FOREST = {"val_macro_f1": 0.534, "test_macro_f1": 0.495}
COURSE_S6_MLP = {"val_macro_f1": 0.493, "test_macro_f1": 0.472}

DENSENET_VS_RESNET = (
    "DenseNet differs from ResNet by concatenating each layer's feature maps to all "
    "later layers (dense connectivity) instead of adding residual skip connections, "
    "which encourages feature reuse and typically needs fewer parameters."
)


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(f1_score(y_true.numpy(), y_pred.numpy(), average="macro"))


def eval_split(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    return macro_f1(y, s7.predict(model, X))


def load_midterm_svm() -> dict:
    raw = json.loads(MIDTERM_METRICS.read_text(encoding="utf-8"))
    return {
        "model": f"Session 5+ midterm SVM ({raw.get('feature_set')}, {raw.get('balance')})",
        "val_macro_f1": float(raw["validation"]["f1"]),
        "test_macro_f1": float(raw["test"]["f1"]),
        "source": "ML_midterm_project/outputs/test_metrics.json",
    }


def train_frozen_with_test(
    X_train, y_train, X_val, y_val, X_test, y_test, epochs: int = 25, lr: float = 1e-3
) -> dict:
    """Strategy 1 with a single test evaluation after choosing the head on val."""
    resnet = s7.resnet18(weights=s7.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.to(s7.device).eval()
    F_train = s7.embed_resnet(resnet, X_train)
    F_val = s7.embed_resnet(resnet, X_val)
    F_test = s7.embed_resnet(resnet, X_test)

    torch.manual_seed(s7.SEED)
    head = nn.Linear(512, 3).to(s7.device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=s7.class_weights_from_labels(y_train))
    loader = DataLoader(TensorDataset(F_train, y_train), batch_size=64, shuffle=True)

    best = {"f1": -1.0, "state": None, "epoch": -1}
    for epoch in range(epochs):
        head.train()
        for fb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(head(fb.to(s7.device)), yb.to(s7.device))
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = head(F_val.to(s7.device)).argmax(1).cpu()
        f1 = macro_f1(y_val, pred)
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "epoch": epoch}

    head.load_state_dict(best["state"])
    with torch.no_grad():
        test_pred = head(F_test.to(s7.device)).argmax(1).cpu()
    test_f1 = macro_f1(y_test, test_pred)
    print(f"frozen head: best val={best['f1']:.3f} (ep {best['epoch']})  test={test_f1:.3f}")
    return {
        "n_trainable": 512 * 3 + 3,
        "best_epoch": best["epoch"],
        "val_macro_f1": float(best["f1"]),
        "test_macro_f1": float(test_f1),
    }


def run_finetune_row(
    name: str,
    layers: tuple[str, ...] | None,
    X_train, y_train, X_val, y_val, X_test, y_test,
    epochs: int,
    lr: float,
) -> dict:
    print(f"\n=== {name} ===")
    model, hist, best = s7.fine_tune_resnet(
        X_train, y_train, X_val, y_val, train_layers=layers, epochs=epochs, lr=lr
    )
    test_f1 = eval_split(model, X_test, y_test)
    print(f"{name}: val={best['f1']:.3f}  test={test_f1:.3f}")
    return {
        "layers": None if layers is None else list(layers),
        "n_trainable": best["n_trainable"],
        "mean_seconds_per_epoch": best["mean_seconds_per_epoch"],
        "best_epoch": best["epoch"],
        "val_macro_f1": float(best["f1"]),
        "test_macro_f1": float(test_f1),
        "epoch_history": hist.reset_index().to_dict(orient="records"),
    }


def run_homework(
    scratch_epochs: int = 8,
    finetune_epochs: int = 6,
    frozen_epochs: int = 25,
    lr_ft: float = 1e-4,
) -> dict:
    s7.setup()
    if not s7.META_PATH.is_file() or not s7.IMG_DIR.is_dir():
        raise FileNotFoundError(f"MILK10k not found at {s7.DATA_DIR}")

    train_df, val_df, test_df = s7.make_dermoscopic_splits(seed=s7.SEED)
    data = s7.build_or_load_image_cache(train_df, val_df, test_df)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"splits train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    rows: dict[str, dict] = {}

    print("\n=== SimpleCNN from scratch ===")
    model_s, hist_s, best_s = s7.train_simple_cnn(
        X_train, y_train, X_val, y_val, epochs=scratch_epochs
    )
    rows["s7_simple_cnn"] = {
        "model": "Session 7 - SimpleCNN (scratch)",
        "val_macro_f1": float(best_s["f1"]),
        "test_macro_f1": eval_split(model_s, X_test, y_test),
        "best_epoch": best_s["epoch"],
        "epoch_history": hist_s.reset_index().to_dict(orient="records"),
    }
    print(f"SimpleCNN test macro-F1 = {rows['s7_simple_cnn']['test_macro_f1']:.3f}")

    frozen = train_frozen_with_test(
        X_train, y_train, X_val, y_val, X_test, y_test, epochs=frozen_epochs
    )
    rows["s7_frozen_resnet"] = {
        "model": "Session 7 - ResNet-18 frozen + linear head (strategy 1)",
        **frozen,
    }

    rows["s7_ft_layer4"] = {
        "model": "Session 7 - ResNet-18 fine-tune layer4+fc (strategy 2)",
        **run_finetune_row(
            "layer4+fc", ("layer4", "fc"),
            X_train, y_train, X_val, y_val, X_test, y_test, finetune_epochs, lr_ft,
        ),
    }
    rows["s7_ft_layer3_4"] = {
        "model": "Session 7 - ResNet-18 fine-tune layer3+layer4+fc",
        **run_finetune_row(
            "layer3+layer4+fc", ("layer3", "layer4", "fc"),
            X_train, y_train, X_val, y_val, X_test, y_test, finetune_epochs, lr_ft,
        ),
    }
    rows["s7_ft_full"] = {
        "model": "Session 7 - ResNet-18 full fine-tune (strategy 3)",
        **run_finetune_row(
            "full network", None,
            X_train, y_train, X_val, y_val, X_test, y_test, finetune_epochs, lr_ft,
        ),
    }

    midterm = load_midterm_svm()
    scoreboard = [
        {
            "model": "Session 5 - random forest (33 features, course)",
            "val_macro_f1": COURSE_S5_FOREST["val_macro_f1"],
            "test_macro_f1": COURSE_S5_FOREST["test_macro_f1"],
            "source": "session6 notebook / course baseline_scores",
        },
        midterm,
        {
            "model": "Session 6 - MLP 33-64-3 (33 features)",
            "val_macro_f1": COURSE_S6_MLP["val_macro_f1"],
            "test_macro_f1": COURSE_S6_MLP["test_macro_f1"],
            "source": "session6/class.ipynb",
        },
        {
            "model": rows["s7_simple_cnn"]["model"],
            "val_macro_f1": rows["s7_simple_cnn"]["val_macro_f1"],
            "test_macro_f1": rows["s7_simple_cnn"]["test_macro_f1"],
            "source": "homework run",
        },
        {
            "model": rows["s7_frozen_resnet"]["model"],
            "val_macro_f1": rows["s7_frozen_resnet"]["val_macro_f1"],
            "test_macro_f1": rows["s7_frozen_resnet"]["test_macro_f1"],
            "source": "homework run",
        },
        {
            "model": rows["s7_ft_layer4"]["model"],
            "val_macro_f1": rows["s7_ft_layer4"]["val_macro_f1"],
            "test_macro_f1": rows["s7_ft_layer4"]["test_macro_f1"],
            "source": "homework run",
        },
        {
            "model": rows["s7_ft_layer3_4"]["model"],
            "val_macro_f1": rows["s7_ft_layer3_4"]["val_macro_f1"],
            "test_macro_f1": rows["s7_ft_layer3_4"]["test_macro_f1"],
            "source": "homework run",
        },
        {
            "model": rows["s7_ft_full"]["model"],
            "val_macro_f1": rows["s7_ft_full"]["val_macro_f1"],
            "test_macro_f1": rows["s7_ft_full"]["test_macro_f1"],
            "source": "homework run",
        },
    ]

    best = max(scoreboard, key=lambda r: r["test_macro_f1"])
    summary = {
        "seed": s7.SEED,
        "device": s7.device,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "densenet_vs_resnet": DENSENET_VS_RESNET,
        "scoreboard": scoreboard,
        "best_model": {
            "name": best["model"],
            "val_macro_f1": best["val_macro_f1"],
            "test_macro_f1": best["test_macro_f1"],
        },
        "strategy3_full_finetune": rows["s7_ft_full"],
        "s7_runs": rows,
        "notes": [
            "Session 7 models use lesion-grouped dermoscopic StratifiedGroupKFold (seed=0).",
            "Midterm SVM uses its own seed=42 lesion split; reported for completeness as best classical.",
            "Test macro-F1 is computed once per model after selecting the best validation checkpoint.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nBest model:", best["model"], f"test macro-F1={best['test_macro_f1']:.3f}")
    print("wrote", RESULTS_PATH)
    return summary


def write_pdf(summary: dict | None = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    if summary is None:
        summary = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Heading1"], fontSize=16, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=6)
    small = ParagraphStyle("S", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
    link = ParagraphStyle("L", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#0B57D0"))

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story = []
    best = summary["best_model"]

    story.append(Paragraph("Session 7 Homework — Scoreboard & Strategy 3", title))
    story.append(Paragraph("Kacper Dudzik · MILK10k dermoscopic classification (macro-F1)", body))
    story.append(Paragraph(f'<link href="{REPO_URL}">{REPO_URL}</link>', link))
    story.append(
        Paragraph(
            f'Code: <link href="{REPO_URL}/blob/main/session7/homework.py">'
            f"session7/homework.py</link> · "
            f'<link href="{REPO_URL}/blob/main/session7/session7.py">session7/session7.py</link> · '
            f'<link href="{REPO_URL}/blob/main/session7/_output/homework_results.json">'
            f"homework_results.json</link>",
            link,
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("1. Scoreboard (Sessions 5–7)", h2))
    story.append(
        Paragraph(
            "Metric: lesion-level <b>macro-F1</b>. Session 7 rows were trained in this homework run; "
            "test is scored once after the best validation epoch.",
            body,
        )
    )

    table_data = [["Model", "Val macro-F1", "Test macro-F1"]]
    for row in summary["scoreboard"]:
        table_data.append(
            [
                Paragraph(row["model"], small),
                f"{row['val_macro_f1']:.3f}",
                f"{row['test_macro_f1']:.3f}",
            ]
        )
    tbl = Table(table_data, colWidths=[11.2 * cm, 2.6 * cm, 2.8 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            f"<b>Best model:</b> {best['name']} — "
            f"val {best['val_macro_f1']:.3f}, <b>test macro-F1 = {best['test_macro_f1']:.3f}</b>.",
            body,
        )
    )

    story.append(Paragraph("2. DenseNet vs ResNet (one sentence)", h2))
    story.append(Paragraph(summary["densenet_vs_resnet"], body))

    story.append(Paragraph("3. Optional: full fine-tuning (strategy 3)", h2))
    ft = summary["strategy3_full_finetune"]
    s2 = summary["s7_runs"]["s7_ft_layer4"]
    story.append(
        Paragraph(
            "Strategy 3 unfreezes every ResNet-18 parameter (≈11.2M trainable) and trains at lr=1e-4 "
            "with class-weighted CE and flips, same protocol as partial fine-tuning.",
            body,
        )
    )
    cmp = Table(
        [
            ["Setting", "Trainable params", "s/epoch", "Val F1", "Test F1"],
            [
                "Strategy 2 (layer4+fc)",
                f"{s2['n_trainable']:,}",
                f"{s2['mean_seconds_per_epoch']:.0f}",
                f"{s2['val_macro_f1']:.3f}",
                f"{s2['test_macro_f1']:.3f}",
            ],
            [
                "Strategy 3 (full network)",
                f"{ft['n_trainable']:,}",
                f"{ft['mean_seconds_per_epoch']:.0f}",
                f"{ft['val_macro_f1']:.3f}",
                f"{ft['test_macro_f1']:.3f}",
            ],
        ],
        colWidths=[4.5 * cm, 3.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm],
    )
    cmp.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(cmp)
    story.append(Spacer(1, 0.2 * cm))

    delta = ft["test_macro_f1"] - s2["test_macro_f1"]
    if delta > 0.01:
        verdict = "Full fine-tuning improved test macro-F1 enough to be useful here."
    elif delta > 0:
        verdict = "Full fine-tuning helped only slightly on test; cost is higher."
    else:
        verdict = "Full fine-tuning did not beat partial layer4 fine-tuning on test; keep strategy 2 as default."
    story.append(Paragraph(f"<b>Finding:</b> {verdict}", body))

    story.append(Paragraph("Notes", h2))
    for note in summary.get("notes", []):
        story.append(Paragraph(f"- {note}", small))
    story.append(
        Paragraph(
            f"Device: {summary.get('device')} · seed {summary.get('seed')} · "
            f"splits {summary.get('split')}",
            small,
        )
    )

    doc.build(story)
    print("wrote", PDF_PATH)
    return PDF_PATH


def smoke_test() -> None:
    """Lightweight checks that do not need MILK10k or GPU."""
    assert "concatenat" in DENSENET_VS_RESNET.lower() or "dense" in DENSENET_VS_RESNET.lower()
    mid = load_midterm_svm()
    assert 0.0 < mid["test_macro_f1"] < 1.0
    model, _ = s7.make_partially_unfrozen_resnet(None)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == s7.count_params(model)
    print("smoke_test OK", {"midterm_test_f1": round(mid["test_macro_f1"], 3), "full_trainable": n})


def main(argv: list[str]) -> None:
    if argv and argv[0] == "smoke":
        smoke_test()
        return
    if argv and argv[0] == "pdf-only":
        write_pdf()
        return
    smoke_test()
    summary = run_homework()
    write_pdf(summary)
    # Drop rebuilt cache so it cannot re-enter git.
    if s7.CACHE_PATH.is_file():
        s7.CACHE_PATH.unlink()
        print("removed cache", s7.CACHE_PATH)


if __name__ == "__main__":
    main(sys.argv[1:])
