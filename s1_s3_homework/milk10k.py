"""MILK10k helpers: paths, checks, splits, models, datasets, and transforms.

Data folder: environment variable MILK10K_DIR, or ../milk10k next to this file.
"""

import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

HOMEWORK_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MILK10K_DIR", HOMEWORK_DIR.parent / "milk10k"))
IMG_DIR = DATA_DIR / "images"
META_PATH = DATA_DIR / "metadata.csv"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"
SEED = 42
IMAGE_SIZE = 224
VAL_SIZE = 0.2
TEST_SIZE = 0.2
OUTPUT_DIR = HOMEWORK_DIR / "outputs"
FIGURE_DIR = HOMEWORK_DIR / "figures"
DX_COLS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
]
RARE_DX = ["MAL_OTH", "DF", "INF", "VASC", "BEN_OTH"]
PAIR_FIELDS = [
    "age_approx", "sex", "anatom_site_general",
    "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4",
]

FEATURES_NUM = ["age_approx"]
FEATURES_CAT = ["sex", "anatom_site_general", "image_type"]
CLASSES = ["Benign", "Indeterminate", "Malignant"]
COST = np.array(
    [
        [0, 1, 1],
        [5, 0, 5],
        [50, 50, 0],
    ]
)


def load_tables(meta_path=META_PATH, gt_path=GT_PATH):
    return pd.read_csv(meta_path), pd.read_csv(gt_path)


def with_dx(gt):
    out = gt.copy()
    out["dx"] = out[DX_COLS].idxmax(axis=1)
    return out


def class_to_diagnosis(meta, gt):
    lesions = meta.drop_duplicates("lesion_id")[["lesion_id", "diagnosis_1"]]
    labeled = with_dx(gt)[["lesion_id", "dx"]].merge(lesions, on="lesion_id")
    return pd.crosstab(labeled["dx"], labeled["diagnosis_1"])


def check_labels(meta, gt):
    violations = {}
    meta_ids, gt_ids = set(meta["lesion_id"]), set(gt["lesion_id"])
    if meta_ids != gt_ids:
        violations["lesion_id_set"] = {
            "only_metadata": sorted(meta_ids - gt_ids),
            "only_gt": sorted(gt_ids - meta_ids),
        }
    positive = gt[DX_COLS].sum(axis=1)
    bad_rows = gt.loc[positive != 1, "lesion_id"].tolist()
    if bad_rows:
        violations["one_hot"] = bad_rows
    mapping = class_to_diagnosis(meta, gt)
    multi = mapping.index[(mapping > 0).sum(axis=1) != 1].tolist()
    if multi:
        violations["class_to_diagnosis_1"] = mapping.loc[multi]
    lesions = meta.drop_duplicates("lesion_id")
    d2 = lesions.groupby("diagnosis_2")["diagnosis_1"].nunique()
    if (d2 > 1).any():
        violations["diagnosis_2_to_1"] = d2[d2 > 1]
    known_d3 = lesions.dropna(subset=["diagnosis_3"])
    d3 = known_d3.groupby("diagnosis_3")["diagnosis_2"].nunique()
    if (d3 > 1).any():
        violations["diagnosis_3_to_2"] = d3[d3 > 1]
    pair = meta.groupby("lesion_id")[PAIR_FIELDS].nunique(dropna=False)
    if (pair > 1).any().any():
        violations["image_pair"] = pair.loc[(pair > 1).any(axis=1)]
    return violations, mapping


def assert_labels(meta, gt):
    violations, mapping = check_labels(meta, gt)
    print("A1.1 label checks")
    if not violations:
        print("  no violations")
    for name, detail in violations.items():
        print(f"  violation: {name}")
        print(detail)
    for key in ("lesion_id_set", "one_hot", "diagnosis_2_to_1", "diagnosis_3_to_2", "image_pair"):
        assert key not in violations
    return violations, mapping


def build_lesions(meta, gt):
    ids = meta.pivot(index="lesion_id", columns="image_type", values="isic_id")
    ids = ids.rename(columns={"dermoscopic": "derm_id", "clinical: close-up": "clinical_id"})
    attrs = meta.groupby("lesion_id", as_index=False).agg(
        diagnosis_1=("diagnosis_1", "first"),
        age=("age_approx", "first"),
        sex=("sex", "first"),
        site=("anatom_site_general", "first"),
    )
    lesions = attrs.merge(ids, on="lesion_id").merge(with_dx(gt)[["lesion_id", "dx"]], on="lesion_id")
    lesions = lesions[
        ["lesion_id", "derm_id", "clinical_id", "diagnosis_1", "dx", "age", "sex", "site"]
    ]
    assert len(lesions) == 5240
    assert lesions["derm_id"].notna().all() and lesions["clinical_id"].notna().all()
    assert lesions["lesion_id"].is_unique
    return lesions


def answer_five_questions(meta, gt):
    lesions = build_lesions(meta, gt)
    bcc = gt.loc[gt["BCC"] == 1, ["lesion_id"]].merge(lesions, on="lesion_id")
    q1 = int(((bcc["age"] >= 70) & (bcc["site"] == "head/neck")).sum())
    images = meta.merge(with_dx(gt)[["lesion_id", "dx"]], on="lesion_id")
    rates = images.assign(is_altered=images["image_manipulation"].eq("altered")).groupby("dx")["is_altered"].mean()
    rates = rates.sort_values(ascending=False)
    mel = lesions.loc[lesions["dx"].eq("MEL")].dropna(subset=["age"])
    has_site = lesions["site"].notna()
    change = (
        lesions.loc[has_site, "dx"].value_counts(normalize=True)
        - lesions.loc[~has_site, "dx"].value_counts(normalize=True)
    ).dropna() * 100
    confirm = meta.groupby("diagnosis_confirm_type")["diagnosis_1"].apply(
        lambda s: (s == "Malignant").mean() * 100
    ).round(2)
    return {
        "q1_bcc_age70_head_neck": q1,
        "q2_class": rates.index[0],
        "q2_altered_pct": float(rates.iloc[0] * 100),
        "q3_youngest": float(mel["age"].min()),
        "q3_oldest": float(mel["age"].max()),
        "q3_gap": float(mel["age"].max() - mel["age"].min()),
        "q4_class": change.abs().idxmax(),
        "q4_points": float(change.abs().max()),
        "q5_malignant_pct": confirm.to_dict(),
    }


def numpy_geometry(arr):
    return {
        "left_right": arr[:, ::-1],
        "up_down": arr[::-1],
        "rot90": np.rot90(arr, 1),
        "transpose": arr.transpose(1, 0, 2),
    }


def pillow_geometry(arr):
    image = Image.fromarray(arr)
    ops = {
        "left_right": Image.Transpose.FLIP_LEFT_RIGHT,
        "up_down": Image.Transpose.FLIP_TOP_BOTTOM,
        "rot90": Image.Transpose.ROTATE_90,
        "transpose": Image.Transpose.TRANSPOSE,
    }
    return {name: np.array(image.transpose(op)) for name, op in ops.items()}


def psnr(original, other):
    mse = np.mean((original.astype(np.float64) - other.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255 ** 2) / mse))


def jpeg_roundtrip(arr, quality):
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=quality)
    size = buffer.tell()
    buffer.seek(0)
    decoded = np.array(Image.open(buffer).convert("RGB"))
    return size, decoded, psnr(arr, decoded)


def memory_bytes(height, width, n_images, dtype_bytes):
    return int(height) * int(width) * 3 * int(n_images) * int(dtype_bytes)


def _take_fold(frame, y_col, n_splits, seed):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(
        splitter.split(np.zeros(len(frame)), frame[y_col].to_numpy(), frame["lesion_id"].to_numpy())
    )
    return rest, held


def split_lesions(lesions, val_size, test_size, seed):
    """Three lesion_id lists. Sizes are fractions of all lesions."""
    n_test = int(round(1 / test_size))
    rest, test_idx = _take_fold(lesions, "dx", n_test, seed)
    remaining = lesions.iloc[rest]
    n_val = int(round(1 / (val_size / (1 - test_size))))
    train_rel, val_rel = _take_fold(remaining, "dx", n_val, seed)
    return (
        remaining.iloc[train_rel]["lesion_id"].tolist(),
        remaining.iloc[val_rel]["lesion_id"].tolist(),
        lesions.iloc[test_idx]["lesion_id"].tolist(),
    )


def assert_split_properties(lesions, val_size, test_size, seeds):
    rare_in_both = {name: 0 for name in RARE_DX}
    dx = lesions.set_index("lesion_id")["dx"]
    n = len(lesions)
    for seed in seeds:
        first = split_lesions(lesions, val_size, test_size, seed)
        assert first == split_lesions(lesions, val_size, test_size, seed)
        train, val, test = (set(part) for part in first)
        assert not (train & val or train & test or val & test)
        assert abs(len(test) / n - test_size) <= 0.01
        assert abs(len(val) / n - val_size) <= 0.01
        for name in RARE_DX:
            rare_in_both[name] += int((dx.loc[list(val)] == name).any() and (dx.loc[list(test)] == name).any())
    return rare_in_both


def _fold_report(name, lesion_ids, y, folds):
    leaked = 0
    proportions = []
    classes = pd.unique(y)
    mal_oth_missing = 0
    for train_idx, val_idx in folds:
        leaked += len(set(lesion_ids[train_idx]) & set(lesion_ids[val_idx]))
        counts = pd.Series(y[val_idx]).value_counts(normalize=True)
        proportions.append(counts.reindex(classes, fill_value=0))
        if counts.get("MAL_OTH", 0) == 0:
            mal_oth_missing += 1
    spread = ((pd.DataFrame(proportions).max() - pd.DataFrame(proportions).min()) * 100).max()
    return {
        "strategy": name,
        "leaked_lesions": leaked,
        "largest_class_spread_pp": float(spread),
        "folds_without_MAL_OTH": mal_oth_missing,
    }


def compare_split_strategies(images, seed=42):
    y = images["dx"].to_numpy()
    groups = images["lesion_id"].to_numpy()
    x = np.zeros(len(images))
    return pd.DataFrame(
        [
            _fold_report("GroupKFold", groups, y, GroupKFold(n_splits=5).split(x, y, groups)),
            _fold_report(
                "StratifiedKFold",
                groups,
                y,
                StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y),
            ),
            _fold_report(
                "StratifiedGroupKFold",
                groups,
                y,
                StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, groups),
            ),
        ]
    )


def metadata_pipeline(seed):
    preprocess = ColumnTransformer(
        [
            ("age", SimpleImputer(strategy="median"), FEATURES_NUM),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                FEATURES_CAT,
            ),
        ]
    )
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, random_state=seed, n_jobs=-1)
    return Pipeline([("prep", preprocess), ("model", model)])


def _score_split(frame, train_idx, test_idx, seed):
    model = metadata_pipeline(seed)
    cols = FEATURES_NUM + FEATURES_CAT
    model.fit(frame.iloc[train_idx][cols], frame.iloc[train_idx]["diagnosis_1"])
    pred = model.predict(frame.iloc[test_idx][cols])
    return balanced_accuracy_score(frame.iloc[test_idx]["diagnosis_1"], pred)


def naive_indices(frame, seed, test_size=0.2):
    index = np.arange(len(frame))
    return train_test_split(index, test_size=test_size, random_state=seed, stratify=frame["diagnosis_1"])


def grouped_indices(frame, seed, n_splits=5):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(frame)), frame["diagnosis_1"], frame["lesion_id"]))


def sibling_oracle(frame, train_idx, test_idx):
    train = set(np.asarray(train_idx).tolist())
    sibling = {}
    for rows in frame.groupby("lesion_id").indices.values():
        rows = list(rows)
        if len(rows) == 2:
            sibling[rows[0]], sibling[rows[1]] = rows[1], rows[0]
    majority = frame.iloc[train_idx]["diagnosis_1"].mode().iloc[0]
    preds, helped = [], 0
    for idx in test_idx:
        other = sibling.get(int(idx))
        if other is not None and other in train:
            preds.append(frame.iloc[other]["diagnosis_1"])
            helped += 1
        else:
            preds.append(majority)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(frame.iloc[test_idx]["diagnosis_1"], preds)),
        "pct_sibling_in_train": 100 * helped / len(test_idx),
    }


def leakage_experiment(frame, seeds):
    frame = frame.reset_index(drop=True)
    packed = {}
    for kind, index_fn in (("naive", naive_indices), ("grouped", grouped_indices)):
        scores, oracles = [], []
        for seed in seeds:
            train_idx, test_idx = index_fn(frame, seed)
            scores.append(_score_split(frame, train_idx, test_idx, seed))
            oracles.append(sibling_oracle(frame, train_idx, test_idx))
        packed[kind] = {
            "rf_mean": float(np.mean(scores)),
            "rf_std": float(np.std(scores)),
            "oracle_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in oracles])),
            "pct_sibling_in_train": float(np.mean([row["pct_sibling_in_train"] for row in oracles])),
        }
    return packed


def expected_cost(y_true, y_pred):
    true_i = pd.Categorical(y_true, CLASSES).codes
    pred_i = pd.Categorical(y_pred, CLASSES).codes
    return float(COST[true_i, pred_i].mean())


def dummy_costs(frame, train_idx, test_idx, seed):
    y_train = frame.iloc[train_idx]["diagnosis_1"]
    y_test = frame.iloc[test_idx]["diagnosis_1"]
    specs = {
        "always_malignant": DummyClassifier(strategy="constant", constant="Malignant"),
        "stratified_random": DummyClassifier(strategy="stratified", random_state=seed),
        "uniform_random": DummyClassifier(strategy="uniform", random_state=seed),
        "always_benign": DummyClassifier(strategy="constant", constant="Benign"),
    }
    rows, matrix = [], None
    x_train, x_test = np.zeros((len(train_idx), 1)), np.zeros((len(test_idx), 1))
    for name, model in specs.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "predictor": name,
                "accuracy": float((pred == y_test.to_numpy()).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro", labels=CLASSES, zero_division=0)),
                "expected_cost": expected_cost(y_test, pred),
            }
        )
        if name == "stratified_random":
            matrix = confusion_matrix(y_test, pred, labels=CLASSES)
    return pd.DataFrame(rows), matrix


def circular_mean_deg(degrees):
    radians = np.deg2rad(degrees)
    angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(angle) % 360)


def circular_distance(a, b):
    return float(abs((a - b + 180) % 360 - 180))


def center_hue_value(path):
    image = np.array(Image.open(path).convert("RGB"))
    height, width = image.shape[:2]
    crop = image[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return circular_mean_deg(hsv[:, :, 0].astype(np.float64) * 2), float(hsv[:, :, 2].mean())


def group_hue_value(paths):
    hues, values = zip(*(center_hue_value(path) for path in paths))
    return circular_mean_deg(np.array(hues)), float(np.mean(values))


def jitter_changes(paths, hue_levels, brightness_levels):
    hue_rows, bright_rows = [], []
    for level in hue_levels:
        shifts = []
        for path in paths:
            before, _ = center_hue_value(path)
            shifted = np.array(TF.adjust_hue(Image.open(path).convert("RGB"), level))
            hsv = cv2.cvtColor(shifted, cv2.COLOR_RGB2HSV)
            height, width = shifted.shape[:2]
            crop = hsv[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
            after = circular_mean_deg(crop[:, :, 0].astype(np.float64) * 2)
            shifts.append(circular_distance(before, after))
        hue_rows.append({"parameter": level, "mean_change_deg": float(np.mean(shifts))})
    for level in brightness_levels:
        deltas = []
        for path in paths:
            _, before = center_hue_value(path)
            shifted = np.array(TF.adjust_brightness(Image.open(path).convert("RGB"), 1 + level))
            hsv = cv2.cvtColor(shifted, cv2.COLOR_RGB2HSV)
            height, width = shifted.shape[:2]
            crop = hsv[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
            deltas.append(abs(float(crop[:, :, 2].mean()) - before))
        bright_rows.append({"parameter": level, "mean_change_v": float(np.mean(deltas))})
    return hue_rows, bright_rows


def _open_rgb(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


class LesionDataset(Dataset):
    """One item is one lesion, with one tensor per requested view."""

    def __init__(self, lesion_table, img_dir, label_col, transform, views=("derm", "clinical"), label_map=None):
        self.table = lesion_table.reset_index(drop=True)
        self.img_dir = path_of(img_dir)
        self.label_col = label_col
        self.transform = transform
        self.views = views
        labels = self.table[label_col].astype(str)
        self.label_map = label_map or {name: i for i, name in enumerate(sorted(labels.unique()))}
        self._id_col = {"derm": "derm_id", "clinical": "clinical_id"}

    def __len__(self):
        return len(self.table)

    def _load(self, isic_id):
        image = _open_rgb(self.img_dir / f"{isic_id}.jpg")
        return image if self.transform is None else self.transform(image)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        item = {view: self._load(row[self._id_col[view]]) for view in self.views}
        item["label"] = self.label_map[str(row[self.label_col])]
        item["lesion_id"] = row["lesion_id"]
        return item


class MilkImageDataset(Dataset):
    """One image per item. A missing file raises unless allow_missing is set."""

    def __init__(self, split_frame, img_dir, label_map, transform, allow_missing=False):
        self.frame = split_frame.reset_index(drop=True)
        self.img_dir = path_of(img_dir)
        self.label_map = label_map
        self.transform = transform
        exists = pd.Series((self.img_dir / f"{isic}.jpg").is_file() for isic in self.frame["isic_id"])
        missing = self.frame.loc[~exists.to_numpy(), "isic_id"]
        if len(missing) and not allow_missing:
            raise FileNotFoundError(f"{len(missing)} images missing, first: {missing.iloc[0]}")
        if allow_missing:
            self.frame = self.frame.loc[exists.to_numpy()].reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        tensor = self.transform(_open_rgb(self.img_dir / f"{row['isic_id']}.jpg"))
        return tensor, self.label_map[row["diagnosis_1"]], row["isic_id"]


def path_of(img_dir):
    from pathlib import Path
    return Path(img_dir)


def aggregate_predictions(image_probs, image_table):
    """Average the two views of a lesion, then take the highest probability."""
    probs = torch.tensor(image_probs, dtype=torch.float32)
    table = image_table.reset_index(drop=True).copy()
    table["_row"] = range(len(table))
    rows = []
    for lesion_id, group in table.groupby("lesion_id", sort=False):
        index = torch.tensor(group["_row"].tolist())
        mean = probs[index].mean(dim=0)
        rows.append({"lesion_id": lesion_id, "prediction": int(mean.argmax())})
    return pd.DataFrame(rows)


def show_labeled_grid(images, labels, names, path):
    """Save a grid of transformed images with the class name as the title."""
    import matplotlib.pyplot as plt

    n = len(images)
    columns = 8 if n > 8 else n
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 2.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, tensor, label in zip(axes, images, labels):
        shown = tensor.permute(1, 2, 0).numpy()
        shown = (shown - shown.min()) / (shown.max() - shown.min() + 1e-6)
        axis.imshow(shown)
        axis.set_title(names[int(label)], fontsize=8)
        axis.axis("off")
    for axis in axes[n:]:
        axis.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_transforms(mean, std):
    """Train transform is random. Eval transform is fixed."""
    shared_end = [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ]
    eval_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *shared_end])
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
            *shared_end,
        ]
    )
    return train_transform, eval_transform


AUGMENTATION_TABLE = [
    ("RandomHorizontalFlip p=0.5", "The camera may face the lesion from either side."),
    ("RandomRotation 15 degrees", "A small tilt does not change the diagnosis."),
    ("ColorJitter brightness/contrast/saturation 0.1, hue 0", "Brightness 0.1 stays under the measured class gap. Hue is off: a factor of 0.02 already exceeds it."),
    ("Resize 224 and /255 then train mean/std", "Same geometry and scale at train and test time."),
]
