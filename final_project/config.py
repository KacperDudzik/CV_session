"""Every setting of the final project in one place.

Change a setting here, never inside the other scripts. See README.md, "Rules for experiments".
"""

import os
import sys
from pathlib import Path

# Windows consoles (cp1250) cannot print "→" or "±": never crash on printing, files stay UTF-8.
sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")

# --- paths ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = Path(os.environ.get("MILK10K_DIR", REPO / "milk10k"))
IMG_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.csv"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"
INPUT_PATH = DATA_DIR / "supplements" / "training_input.csv"  # skin_tone_class per image

TEST_DIR = HERE / "isic_test"  # put the downloaded ISIC test files here (see README)
TEST_IMG_DIR = TEST_DIR / "MILK10k_Test_Input"
TEST_META_PATH = TEST_DIR / "MILK10k_Test_Metadata.csv"

SPLIT_DIR = HERE / "splits"   # frozen lesion lists: the one split every model uses
CACHE_DIR = HERE / "cache"    # decoded 224x224 images (not committed)
RUNS_DIR = HERE / "runs"      # one folder per target / experiment / seed
OUT_DIR = HERE / "outputs"    # scoreboard, figures, tables for the report
MODEL_DIR = HERE / "models"   # model_m4.pt and model_11class.pt

# The class code lives in the session folders; we import it instead of copying it.
for session in ("session5", "session7", "session8"):
    sys.path.insert(0, str(REPO / session))

# --- targets (Session 10, slide 22) -----------------------------------------
CLASSES = {
    "diagnosis_1": ["Benign", "Indeterminate", "Malignant"],
    "dx11": ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
             "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"],  # = training_gt.csv = submission order
}
ANALYSIS_TARGET = "diagnosis_1"   # M2-M4 comparison (Path A: the target used so far)
LEADERBOARD_TARGET = "dx11"       # M5: always 11 classes

# --- split (Session 3 / Session 7: lesion-grouped, dermoscopic view) -------------
SPLIT_SEED = 0          # session7.make_dermoscopic_splits: 10 folds, 0 = test, 1-2 = val
IMAGE_SIZE = 224        # PIL resize, the same for training, validation and inference

# --- experiments ---------------------------------------------------------------
# One row = one model. "train" = parameter-name prefixes that are trained (Session 7 rule);
# None = every parameter. Same epochs for every deep model; lr 1e-3 when only a new head
# is trained, 1e-4 when backbone layers are unfrozen (Session 7). fit() keeps the epoch
# with the best validation macro-F1 (early stopping, Session 10 recap).
EPOCHS = 10
SEEDS = (0, 1, 2)

EXPERIMENTS = {
    "simple_cnn":        {"label": "CNN from scratch (S7)",
                          "arch": "simple_cnn", "train": None, "lr": 1e-3},
    "resnet18_frozen":   {"label": "ResNet-18 frozen + new head (S7 strategy 1, S9)",
                          "arch": "resnet18", "train": ("fc",), "lr": 1e-3},
    "vit_b16_frozen":    {"label": "ViT-B/16 frozen + new head (S9)",
                          "arch": "vit_b_16", "train": ("heads",), "lr": 1e-3},
    "resnet18_layer4":   {"label": "ResNet-18 fine-tune layer4 (S7 strategy 2)",
                          "arch": "resnet18", "train": ("layer4", "fc"), "lr": 1e-4},
    "resnet18_layer3_4": {"label": "ResNet-18 fine-tune layer3+4 (S7 mini-challenge)",
                          "arch": "resnet18", "train": ("layer3", "layer4", "fc"), "lr": 1e-4},
    "vit_b16_last2":     {"label": "ViT-B/16 fine-tune last 2 blocks (S7 strategy 2 on S9 ViT)",
                          "arch": "vit_b_16", "lr": 1e-4,
                          "train": ("encoder.layers.encoder_layer_10",
                                    "encoder.layers.encoder_layer_11",
                                    "encoder.ln", "heads")},
    "resnet18_full":     {"label": "ResNet-18 full fine-tune (S7 strategy 3)",
                          "arch": "resnet18", "train": None, "lr": 1e-4},
    # Same question as the S7 mini-challenge (layer4 vs layer3+4), asked of the ViT.
    "vit_b16_last4":     {"label": "ViT-B/16 fine-tune last 4 blocks (S7 mini-challenge on S9 ViT)",
                          "arch": "vit_b_16", "lr": 1e-4,
                          "train": ("encoder.layers.encoder_layer_8",
                                    "encoder.layers.encoder_layer_9",
                                    "encoder.layers.encoder_layer_10",
                                    "encoder.layers.encoder_layer_11",
                                    "encoder.ln", "heads")},
}
RF_LABEL = "Random forest, 33 Session 5 features (M2)"

# --- stage 2: hyperparameter sweep (README rule 10) -----------------------------------
# Only the two knobs taught in class change: the learning rate (S6) and the number of
# epochs (fit() keeps the best epoch = early stopping, S10). Each variant is a normal
# experiment named "<base>__lr<lr>_ep<epochs>"; selection stays on validation.
SWEEP = [
    ("resnet18_layer3_4", 1e-4, 20),
    ("resnet18_layer3_4", 3e-5, 20),
    ("resnet18_layer3_4", 3e-4, 20),
    ("resnet18_full", 3e-5, 20),
    ("vit_b16_last2", 3e-5, 10),
    ("vit_b16_last4", 3e-5, 10),
]


def sweep_name(base: str, lr: float, epochs: int) -> str:
    return f"{base}__lr{lr:g}_ep{epochs}"


def variant(base: str, suffix: str, note: str, **changes) -> str:
    """Register EXPERIMENTS['<base>__<suffix>'] = base spec with `changes`; return its name."""
    name = f"{base}__{suffix}"
    EXPERIMENTS[name] = {**EXPERIMENTS[base], **changes, "label": f"{EXPERIMENTS[base]['label']}; {note}"}
    return name


STAGE1 = list(EXPERIMENTS)
STAGE2 = [variant(b, sweep_name(b, lr, ep).split("__")[1], f"lr {lr:g}, {ep} epochs", lr=lr, epochs=ep)
          for b, lr, ep in SWEEP]

# --- stage 3a: more architectures from class (S7 slide 18 and homework) -----------------
# Same two strategies as ResNet-18: frozen + new head (lr 1e-3), last stage unfrozen (lr 1e-4).
EXPERIMENTS.update({
    "vgg16_frozen":           {"label": "VGG-16 frozen + new head (S7 slide 18)",
                               "arch": "vgg16", "train": ("classifier.6",), "lr": 1e-3},
    "vgg16_classifier":       {"label": "VGG-16 fine-tune 3 FC layers, dropout 0.5 (S7 slide 18)",
                               "arch": "vgg16", "train": ("classifier",), "lr": 1e-4},
    "densenet121_frozen":     {"label": "DenseNet-121 frozen + new head (S7 homework)",
                               "arch": "densenet121", "train": ("classifier",), "lr": 1e-3},
    "densenet121_block4":     {"label": "DenseNet-121 fine-tune last dense block (S7 homework)",
                               "arch": "densenet121", "lr": 1e-4,
                               "train": ("features.denseblock4", "features.norm5", "classifier")},
    "efficientnet_b0_frozen": {"label": "EfficientNet-B0 frozen + new head (S7 homework)",
                               "arch": "efficientnet_b0", "train": ("classifier",), "lr": 1e-3},
    "efficientnet_b0_last2":  {"label": "EfficientNet-B0 fine-tune last 2 stages (S7 homework)",
                               "arch": "efficientnet_b0", "lr": 1e-4,
                               "train": ("features.7", "features.8", "classifier")},
})
STAGE3_ARCH = ["densenet121_frozen", "densenet121_block4", "efficientnet_b0_frozen",
               "efficientnet_b0_last2", "vgg16_frozen", "vgg16_classifier"]

# --- stage 3b: regularisation, imbalance, normalisation (S6 slide 32, S3 slides 34/45/55) --
# One factor at a time against the stage-1 recipe, so each effect is measured on its own.
_R = "resnet18_layer3_4"
STAGE3_REG = [
    variant(_R, "dropout0.3", "dropout 0.3 before the head", dropout=0.3),
    variant(_R, "wd1e-4", "weight decay 1e-4", weight_decay=1e-4),
    variant(_R, "dropout0.3_wd1e-4", "dropout 0.3 + weight decay 1e-4", dropout=0.3, weight_decay=1e-4),
    variant(_R, "aug_m1", "M1 augmentation: flips, rotation 15, colour 0.1", augment="m1"),
    variant(_R, "oversample", "oversampling instead of class weights", oversample=True, class_weights=False),
    variant(_R, "no_class_weights", "no class weights (control)", class_weights=False),
    variant("simple_cnn", "train_norm", "training-set mean/std normalisation", norm="train"),
    variant("simple_cnn", "dropout0.3_wd1e-4", "dropout 0.3 + weight decay 1e-4", dropout=0.3, weight_decay=1e-4),
    variant("simple_cnn", "aug_m1", "M1 augmentation", augment="m1"),
]

# --- stage 4: final candidates = only the changes that helped on validation in stages 2-3 ----
# Evidence (ResNet-18 layer3+4, mean of 3 seeds): 20 epochs +0.027; oversampling instead of
# class weights +0.021 with 7x smaller spread; dropout 0.3: same mean, 5x smaller spread.
_V = "vit_b16_last4"
STAGE4 = [
    variant(_R, "oversample_ep20", "oversampling, 20 epochs",
            oversample=True, class_weights=False, epochs=20),
    variant(_R, "oversample_dropout0.3_ep20", "oversampling, dropout 0.3, 20 epochs",
            oversample=True, class_weights=False, dropout=0.3, epochs=20),
    variant("resnet18_full", "lr3e-05_oversample_ep20", "lr 3e-05, oversampling, 20 epochs",
            lr=3e-5, oversample=True, class_weights=False, epochs=20),
    variant(_V, "oversample", "oversampling", oversample=True, class_weights=False),
    variant(_V, "ep20", "20 epochs", epochs=20),
]

STAGES = {"stage1": STAGE1, "sweep": STAGE2, "architectures": STAGE3_ARCH,
          "regularization": STAGE3_REG, "final_candidates": STAGE4}

# --- leaderboard thresholds (brief slide 15, optional) ---------------------------------
# One threshold per class, tuned on validation only; probabilities are then rescaled so
# that threshold sits at 0.5. Classes with fewer validation lesions keep 0.5 (too noisy).
THRESHOLD_GRID = [round(0.05 * k, 2) for k in range(1, 20)]
MIN_VAL_POSITIVES = 5
BATCH_SIZE = 64

# --- final model ----------------------------------------------------------------
# Written by evaluate.py (the winner on validation); read by the M5 scripts.
BEST_JSON = OUT_DIR / "best_model.json"
MODEL_M4 = MODEL_DIR / "model_m4.pt"
MODEL_11 = MODEL_DIR / "model_11class.pt"
