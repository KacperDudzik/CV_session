"""Paths, split, labels, feature cache version, and experiment knobs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT.parent / "milk10k"))
IMG_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.csv"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
N_SPLITS = 10
TEST_FOLDS = (0,)
VAL_FOLDS = (1, 2)
TRAIN_FOLDS = (3, 4, 5, 6, 7, 8, 9)

CLASSES = ["Benign", "Indeterminate", "Malignant"]
DX_COLS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
]
N_BINS = 8

# Bump when feature definitions change so stale caches are ignored.
FEATURE_CACHE_VERSION = "v2"
FEATURE_CACHE_PATH = OUTPUT_DIR / f"features_{FEATURE_CACHE_VERSION}.csv"

# Resize used for filter banks and optional descriptors. Original RGB stats
# in extract_features still use the native image so Session 5 numbers stay.
FILTER_IMAGE_SIZE = (128, 128)
EDGE_THRESHOLD = 50.0
INCLUDE_OPTIONAL_DESCRIPTORS = True
CV_FOLDS = 5

# Primary selection metric is lesion-level macro-F1. Tie-breakers follow.
PRIMARY_METRIC = "f1"
MODEL_RANK = {
    "majority": 0,
    "logistic": 1,
    "tree": 2,
    "knn": 3,
    "svm": 4,
    "forest": 5,
}

# targeted_aug is available but off by default (re-extracts minority images).
BALANCE_STRATEGIES = ("none", "oversample", "undersample")
BALANCE_STRATEGIES_FULL = ("none", "oversample", "undersample", "targeted_aug")

# Feature sets used in the model grid. Every set remains independently selectable
# via features.FEATURE_SETS; this list only controls search runtime.
SEARCH_FEATURE_SETS = (
    "color_6",
    "hist_24",
    "texture_3",
    "class_33",
    "extended",
    "colorspace",
    "filters",
    "glcm_shape",
    "optional_descriptors",
    "original_plus_filters",
    "original_plus_advanced",
    "all_features",
)
