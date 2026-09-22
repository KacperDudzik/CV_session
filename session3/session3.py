from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "milk10k" / "images"
LABEL = "diagnosis_1"
SEED = 42


def _section(title):
    print(f"\n=== {title} ===")


def load_metadata():
    df = pd.read_csv(ROOT / "milk10k" / "metadata.csv")
    print("columns:", df.columns.tolist())
    print("rows:", len(df), "| unique lesion_id:", df["lesion_id"].nunique())
    print("nulls per column:\n", df.isna().sum().to_string())
    return df


def show_class_imbalance(df, title="class imbalance"):
    _section(title)
    counts = df[LABEL].value_counts()
    share = df[LABEL].value_counts(normalize=True)
    inv = df[LABEL].map(1 / counts)
    print(counts.to_string())
    print("share:\n", share.round(3).to_string())
    print("1 / count per class (one row's share of its class):\n", (1 / counts).to_string())
    print("one row as % of its class:\n", ((1 / counts) * 100).round(4).to_string())
    print("mapped onto each row (.map(1 / count)), head:\n", inv.head(8).to_string())
    print(
        "Malignant dominates; Indeterminate is rare. "
        "A model that always predicts Malignant would already look fairly accurate."
    )
    return counts


def _leakage(train, val, test):
    t, v, s = set(train["lesion_id"]), set(val["lesion_id"]), set(test["lesion_id"])
    return {
        "train & val": len(t & v),
        "train & test": len(t & s),
        "val & test": len(v & s),
    }


def show_data_leakage(df, test_size=0.2, val_size=0.25):
    """Naive row split. Each lesion_id has 2 images, so the same lesion can leak."""
    _section("data leakage (naive split by image)")
    print("images per lesion:\n", df.groupby("lesion_id").size().value_counts().sort_index().to_string())

    tv, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df[LABEL]
    )
    train, val = train_test_split(
        tv, test_size=val_size, random_state=SEED, stratify=tv[LABEL]
    )

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s} {len(part):5d} images | {part['lesion_id'].nunique():4d} lesions")

    leak = _leakage(train, val, test)
    print("lesion_id overlap:", leak)
    print(
        "Every lesion has 2 images. A naive split can put one photo in train and "
        "the other in test. The model has already seen that lesion, so the test "
        "score looks better than it would on a new patient."
    )
    return train, val, test


def split_by_lesion(df, test_size=0.2, val_size=0.25):
    """Correct split: whole lesion stays in one set. 60/20/20."""
    _section("correct split (by lesion_id)")
    lesions = df.drop_duplicates("lesion_id")
    ids, y = lesions["lesion_id"], lesions[LABEL]

    ids_tv, ids_test, y_tv, _ = train_test_split(
        ids, y, test_size=test_size, random_state=SEED, stratify=y
    )
    ids_train, ids_val, _, _ = train_test_split(
        ids_tv, y_tv, test_size=val_size, random_state=SEED, stratify=y_tv
    )

    train = df[df["lesion_id"].isin(ids_train)].copy()
    val = df[df["lesion_id"].isin(ids_val)].copy()
    test = df[df["lesion_id"].isin(ids_test)].copy()

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s} {len(part):5d} images | {part['lesion_id'].nunique():4d} lesions")
        print(part[LABEL].value_counts().to_string(), "\n")
    print("lesion_id overlap:", _leakage(train, val, test))
    return train, val, test


def oversample(df):
    n = df[LABEL].value_counts().max()
    parts = [
        resample(g, replace=True, n_samples=n, random_state=SEED)
        for _, g in df.groupby(LABEL)
    ]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def undersample(df):
    n = df[LABEL].value_counts().min()
    parts = [
        resample(g, replace=False, n_samples=n, random_state=SEED)
        for _, g in df.groupby(LABEL)
    ]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def balanced_class_weights(df):
    classes = np.unique(df[LABEL])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=df[LABEL])
    mapping = dict(zip(classes, weights))
    print("class_weight='balanced':", mapping)
    print("formula: n_samples / (n_classes * count(class))")
    return mapping


def fix_imbalance_on_train(train):
    """Resample / reweight train only. Val and test stay as they are."""
    _section("fix class imbalance (train only)")
    train_over = oversample(train)
    train_under = undersample(train)
    print("oversampled:\n", train_over[LABEL].value_counts().to_string(), "\n")
    print("undersampled:\n", train_under[LABEL].value_counts().to_string(), "\n")
    balanced_class_weights(train)
    return train_over, train_under


def plot_augmentations(isic_id):
    _section("augmentation (train-time only)")
    img = Image.open(IMG_DIR / f"{isic_id}.jpg")
    variants = {
        "original": img,
        "h-flip": TF.hflip(img),
        "rotate +15": TF.rotate(img, 15),
        "rotate -15": TF.rotate(img, -15),
        "v-flip": TF.vflip(img),
        "brighter": TF.adjust_brightness(img, 1.3),
    }
    fig, axes = plt.subplots(1, len(variants), figsize=(16, 4))
    for ax, (name, im) in zip(axes, variants.items()):
        ax.imshow(im)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(isic_id)
    plt.tight_layout()
    plt.show()


def main():
    df = load_metadata()
    show_class_imbalance(df)
    show_data_leakage(df)
    train, val, test = split_by_lesion(df)
    show_class_imbalance(train, title="class imbalance still in train")
    fix_imbalance_on_train(train)
    plot_augmentations(train.iloc[0]["isic_id"])


if __name__ == "__main__":
    main()
