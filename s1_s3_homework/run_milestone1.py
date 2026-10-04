"""Milestone 1: integrity, figures, labels, splits, transforms, loaders."""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from milk10k import (
    AUGMENTATION_TABLE,
    DATA_DIR,
    FIGURE_DIR,
    IMAGE_SIZE,
    IMG_DIR,
    OUTPUT_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
    MilkImageDataset,
    build_lesions,
    build_transforms,
    class_to_diagnosis,
    load_tables,
    show_labeled_grid,
    split_lesions,
)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scan_images(meta):
    missing, bad = [], []
    widths, heights = [], []
    for isic_id in meta["isic_id"]:
        path = IMG_DIR / f"{isic_id}.jpg"
        if not path.is_file():
            missing.append(isic_id)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            widths.append(width)
            heights.append(height)
        except Exception as exc:
            bad.append({"isic_id": isic_id, "error": str(exc)})
    summary = pd.DataFrame(
        {
            "stat": ["min", "median", "max", "fraction_both_sides_above_224"],
            "width": [
                float(np.min(widths)),
                float(np.median(widths)),
                float(np.max(widths)),
                float(np.mean((np.array(widths) >= IMAGE_SIZE) & (np.array(heights) >= IMAGE_SIZE))),
            ],
            "height": [
                float(np.min(heights)),
                float(np.median(heights)),
                float(np.max(heights)),
                np.nan,
            ],
        }
    )
    return missing, bad, summary


def channel_mean_std(isic_ids):
    total = np.zeros(3)
    squared = np.zeros(3)
    count = 0
    for isic_id in isic_ids:
        image = Image.open(IMG_DIR / f"{isic_id}.jpg").convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float64) / 255.0
        total += array.sum(axis=(0, 1))
        squared += np.square(array).sum(axis=(0, 1))
        count += array.shape[0] * array.shape[1]
    mean = total / count
    variance = squared / count - np.square(mean)
    return mean, np.sqrt(np.maximum(variance, 0))


def assign_split(meta, lesions, train_ids, val_ids, test_ids):
    split_of = {}
    for name, ids in ("train", train_ids), ("val", val_ids), ("test", test_ids):
        for lesion_id in ids:
            split_of[lesion_id] = name
    frame = meta.merge(lesions[["lesion_id", "dx"]], on="lesion_id")
    frame["split"] = frame["lesion_id"].map(split_of)
    return frame


def verify_splits(frame, lesions):
    assert frame["split"].notna().all()
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frame.loc[frame["split"].eq(left), "lesion_id"]) & set(
            frame.loc[frame["split"].eq(right), "lesion_id"]
        )
        assert not overlap, overlap
    sizes = frame.groupby(["split", "lesion_id"]).size()
    assert (sizes == 2).all()
    pair_types = frame.groupby("lesion_id")["image_type"].agg(lambda values: set(values))
    assert pair_types.map(lambda values: values == {"dermoscopic", "clinical: close-up"}).all()
    global_dx = lesions["dx"].value_counts(normalize=True)
    rows = []
    max_deviation = 0.0
    for split, group in lesions.assign(
        split=lesions["lesion_id"].map(
            frame.drop_duplicates("lesion_id").set_index("lesion_id")["split"]
        )
    ).groupby("split"):
        share = group["dx"].value_counts(normalize=True)
        deviation = (share - global_dx).abs().max()
        max_deviation = max(max_deviation, float(deviation))
        rows.append(share.rename(split))
    dx_table = pd.DataFrame(rows).T.fillna(0)
    d1_table = pd.crosstab(frame["split"], frame["diagnosis_1"], normalize="index")
    return dx_table, d1_table, max_deviation


def save_figures(meta, lesions, mapping):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    counts = meta["diagnosis_1"].value_counts()
    fig, axis = plt.subplots(figsize=(5, 3))
    axis.bar(counts.index, counts.values)
    axis.set_title("diagnosis_1")
    axis.set_ylabel("Images")
    fig.savefig(FIGURE_DIR / "diagnosis_1_counts.png", bbox_inches="tight")
    plt.close()

    dx_counts = lesions["dx"].value_counts()
    fig, axis = plt.subplots(figsize=(8, 3))
    axis.bar(dx_counts.index, dx_counts.values)
    axis.set_yscale("log")
    axis.set_title("11-class lesion counts")
    fig.savefig(FIGURE_DIR / "dx_counts_log.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for axis, dx in zip(axes.ravel(), list(dx_counts.index) + [None]):
        axis.axis("off")
        if dx is None:
            continue
        isic_id = lesions.loc[lesions["dx"].eq(dx), "derm_id"].iloc[0]
        axis.imshow(Image.open(IMG_DIR / f"{isic_id}.jpg"))
        axis.set_title(dx, fontsize=8)
    fig.savefig(FIGURE_DIR / "class_gallery.png", bbox_inches="tight")
    plt.close()
    mapping.to_csv(OUTPUT_DIR / "class_to_diagnosis_1.csv")


def augmentation_figure(train_frame, train_transform):
    picked = []
    for dx in ("NV", "MEL", "MAL_OTH"):
        rows = train_frame[train_frame["dx"] == dx]
        picked.append(rows.iloc[0])
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for row_i, row in enumerate(picked):
        image = Image.open(IMG_DIR / f"{row['isic_id']}.jpg").convert("RGB")
        axes[row_i, 0].imshow(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[row_i, 0].set_title(row["dx"], fontsize=8)
        for col in range(1, 8):
            tensor = train_transform(image)
            shown = tensor.permute(1, 2, 0).numpy()
            shown = (shown - shown.min()) / (shown.max() - shown.min() + 1e-6)
            axes[row_i, col].imshow(shown)
        for axis in axes[row_i]:
            axis.axis("off")
    fig.savefig(FIGURE_DIR / "augmentations.png", bbox_inches="tight")
    plt.close()


def main():
    set_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    meta, gt = load_tables()
    print("MILK10K_DIR =", DATA_DIR)

    missing, bad, summary = scan_images(meta)
    print("missing files:", len(missing))
    print("unreadable files:", len(bad))
    if missing:
        print(missing[:10])
    summary.to_csv(OUTPUT_DIR / "image_size_summary.csv", index=False)
    print(summary)
    assert not missing

    lesions = build_lesions(meta, gt)
    lesions.to_csv(OUTPUT_DIR / "lesions.csv", index=False)
    mapping = class_to_diagnosis(meta, gt)
    save_figures(meta, lesions, mapping)

    findings = pd.DataFrame(
        [
            ["Benign lesions are younger than malignant ones", "yes", "Age may be used, but it is not a pixel feature"],
            ["diagnosis_2, diagnosis_3 and diagnosis_4 repeat diagnosis_1", "yes", "Do not use them as inputs"],
            ["Histopathology confirmation tracks the malignant class", "yes", "Do not use diagnosis_confirm_type or biopsy"],
            ["Colour histograms of the classes overlap", "yes", "Do not treat mean colour as the diagnosis"],
            ["Every lesion has two image types", "yes", "Split and score by lesion_id"],
        ],
        columns=["session_2_finding", "still_true", "pipeline_consequence"],
    )
    findings.to_csv(OUTPUT_DIR / "session2_findings.csv", index=False)

    label_map = {
        "primary_target": "diagnosis_1",
        "indeterminate_policy": "keep as a third class",
        "rare_class_policy": "keep every 11-class label and balance with train weights",
        "diagnosis_1": {"Benign": 0, "Indeterminate": 1, "Malignant": 2},
        "dx": {name: i for i, name in enumerate(sorted(lesions["dx"].unique()))},
    }
    (OUTPUT_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    quality = """# Data-quality decisions

Missing values:
- age_approx: impute the train median inside the model pipeline only.
- anatom_site_general: keep a category "unknown". Do not drop the row.
- anatom_site_special: do not use. Almost every value is empty.
- diagnosis_3 and diagnosis_4: leave blank. They are not model inputs.
- melanocytic: do not use. Blank is not a random hole; it mostly means "not marked melanocytic".

Label consistency:
- 5,240 lesions, exactly two images each.
- One image is dermoscopic and the other is clinical: close-up.
- Age, sex, site, and diagnosis fields agree across the pair.

Shortcuts:
- image_type is half and half inside every diagnosis_1 class. It names the view, not the disease.
- image_manipulation is usually "instrument only". Do not use it as an input; it describes the file, not the lesion.
- diagnosis_confirm_type and concomitant_biopsy are the same biopsy decision. Malignant lesions were almost all confirmed by histopathology.

Columns that must not be model inputs, because they leak the label or the archive process:
- diagnosis_2, diagnosis_3, diagnosis_4
- diagnosis_confirm_type, concomitant_biopsy, melanocytic
- isic_id and lesion_id are identifiers, not features
"""
    (OUTPUT_DIR / "quality_report.md").write_text(quality, encoding="utf-8")

    train_ids, val_ids, test_ids = split_lesions(lesions, VAL_SIZE, TEST_SIZE, SEED)
    frame = assign_split(meta, lesions, train_ids, val_ids, test_ids)
    dx_table, d1_table, max_deviation = verify_splits(frame, lesions)
    print("max dx deviation from global:", round(max_deviation, 4))
    print(d1_table.round(3))
    dx_table.to_csv(OUTPUT_DIR / "split_dx_proportions.csv")
    keep = ["isic_id", "lesion_id", "diagnosis_1", "dx", "image_type", "age_approx", "sex", "anatom_site_general"]
    for name in ("train", "val", "test"):
        frame.loc[frame["split"].eq(name), keep].to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    train = frame[frame["split"].eq("train")]
    mean, std = channel_mean_std(train["isic_id"])
    (OUTPUT_DIR / "norm_stats.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8",
    )
    train_transform, eval_transform = build_transforms(mean, std)
    sample = Image.open(IMG_DIR / f"{train.iloc[0]['isic_id']}.jpg").convert("RGB")
    first = eval_transform(sample)
    second = eval_transform(sample)
    assert torch.equal(first, second)

    y = train["diagnosis_1"].map(label_map["diagnosis_1"]).to_numpy()
    classes = np.array(sorted(label_map["diagnosis_1"].values()))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    names = {index: name for name, index in label_map["diagnosis_1"].items()}
    class_weights = {names[int(index)]: float(weight) for index, weight in zip(classes, weights)}
    (OUTPUT_DIR / "class_weights.json").write_text(
        json.dumps(class_weights, indent=2), encoding="utf-8"
    )
    print("train class weights:", class_weights)
    print("train diagnosis_1 counts:")
    print(train["diagnosis_1"].value_counts())

    augmentation_figure(train, train_transform)
    pd.DataFrame(AUGMENTATION_TABLE, columns=["augmentation", "why_the_label_is_unchanged"]).to_csv(
        OUTPUT_DIR / "augmentation_table.csv", index=False
    )

    def loader_for(name, transform, shuffle, sampler=None):
        part = pd.read_csv(OUTPUT_DIR / f"{name}.csv")
        data = MilkImageDataset(part, IMG_DIR, label_map["diagnosis_1"], transform)
        return DataLoader(
            data,
            batch_size=32 if sampler is None else 32,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,
        )

    sample_weights = np.array(
        train["diagnosis_1"].map(class_weights), dtype=np.float64
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = loader_for("train", train_transform, shuffle=False, sampler=sampler)
    val_loader = loader_for("val", eval_transform, shuffle=False)
    test_loader = loader_for("test", eval_transform, shuffle=False)
    images, labels, ids = next(iter(train_loader))
    print("batch", tuple(images.shape), images.dtype, float(images.min()), float(images.max()))
    print("ids", len(ids))

    seen = []
    for i, (_, batch_labels, _) in enumerate(train_loader):
        seen.extend(batch_labels.tolist())
        if i == 19:
            break
    print("label counts in 20 train batches:", pd.Series(seen).value_counts().sort_index().to_dict())

    start = time.perf_counter()
    for _ in train_loader:
        pass
    epoch_seconds = time.perf_counter() - start
    print(f"one train epoch: {epoch_seconds:.1f} s")

    visual = DataLoader(
        MilkImageDataset(train.iloc[:16], IMG_DIR, label_map["diagnosis_1"], eval_transform),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    grid, grid_labels, _ = next(iter(visual))
    names = {index: name for name, index in label_map["diagnosis_1"].items()}
    show_labeled_grid(grid, grid_labels, names, FIGURE_DIR / "train_batch.png")
    print("val batches", len(val_loader), "test batches", len(test_loader))
    ratio = train["diagnosis_1"].value_counts(normalize=True)
    report = f"""# Milestone 1

## Label strategy

The primary target is diagnosis_1 with three classes. Indeterminate is kept. There are 123 indeterminate lesions, all of them AKIEC, and they are not safe to call benign or malignant. Dropping them would train the model as if every lesion were decidable. Merging them into malignant would create false cancer calls.

The 11-class stretch goal keeps every class, including MAL_OTH (9 lesions), BEN_OTH (44), VASC (47), INF (50), and DF (52). Merging these into "other" would hide the diagnosis. Class weights from the training split are stored in `outputs/class_weights.json`. A score on MAL_OTH will be unstable because a test set holds only one or two of those lesions.

## Splits

Lesions are split with StratifiedGroupKFold on the 11-class label, seed 42, on 4 October 2026. About 60% train, 20% validation, 20% test. Images are assigned after the lesion split. Checked in code: no lesion appears in two splits, and each lesion has exactly two images in its split. The largest deviation of an 11-class share from the global share is {max_deviation:.4f}. diagnosis_1 shares by split:

{d1_table.round(3).to_string()}

## Imbalance on the training images

{train["diagnosis_1"].value_counts().to_string()}

Malignant is {ratio.get("Malignant", 0):.0%} of the training images. Accuracy will look strong for a model that always says malignant. Training uses class weights and a weighted sampler. The number to watch is malignant recall, or the cost of a miss, not accuracy.

## Quality

Every metadata row has an image file. Unreadable files: {len(bad)}. Width ranges from {summary.loc[0, "width"]:.0f} to {summary.loc[2, "width"]:.0f} pixels (median {summary.loc[1, "width"]:.0f}). Height ranges from {summary.loc[0, "height"]:.0f} to {summary.loc[2, "height"]:.0f} (median {summary.loc[1, "height"]:.0f}). The fraction of images at least 224 pixels on both sides is {summary.loc[3, "width"]:.2f}.

Missing age is imputed with the training median inside the model only. Missing body site becomes "unknown". `anatom_site_special` is not used. `diagnosis_2`, `diagnosis_3`, `diagnosis_4`, `diagnosis_confirm_type`, `concomitant_biopsy`, and `melanocytic` are not inputs: they repeat the label or the decision to biopsy.

## Preprocessing

Input size is 224x224. Almost every image is larger than 224, so this only downsamples. Pixels are divided by 255, then shifted by the training-set mean and standard deviation (`outputs/norm_stats.json`). The same statistics are used at evaluation.

Each training item is one image with the lesion label. The other two options are: use only the dermoscopic photo, or feed both views as one item. Evaluation must average the two views of a lesion. Scoring each image separately double-counts patients and can test a photo whose sibling was in the training set.

Augmentations, and why they do not change the label, are in `outputs/augmentation_table.csv`. Hue jitter is off: a hue factor of 0.02 already moves colour by more than the benign-malignant gap. The evaluation transform has no random step; applying it twice gives the same tensor. One training epoch took {epoch_seconds:.0f} seconds with `num_workers=0`.

## Why this set is not a clinic

MILK10k does not show how often skin cancer appears among people who ask about a mark. Of 10,480 images, 10,032 were confirmed by histopathology and 448 by one clinical look. In the histopathology group, 72% of images are malignant. In the clinical group, 2% are malignant. A lesion was kept because someone already chose a biopsy, or because it was collected for this study. People with ordinary benign spots that are never removed are mostly absent. Malignant is therefore the majority class here, about 69% of images, while it is the minority in screening. Accuracy on MILK10k answers how often a prediction matches this archive. It does not answer how often the same rule would be right for the next new patient. A model can look accurate by saying malignant most of the time, because that is what the files contain. The number that matches the clinical cost is whether a malignant lesion is missed. The 11-class task has the same problem in a sharper form: MAL_OTH has nine lesions because the archive was not sampled to match how rare that disease is. A high score on the common classes can hide a useless score on the rare ones. Read a held-out accuracy with that selection in mind.

## What can still go wrong

A grouped split removes the direct leak of one view into training and the other into test. It does not remove near-duplicate lesions of different people, or a camera style that tracks a clinic. The set is enriched for biopsied cancers, so accuracy on MILK10k is not the accuracy in a screening clinic. A clinician should treat the score as a study-set estimate, not as a rate of cancer in the next patient who walks in.
"""
    Path(__file__).resolve().parent.joinpath("milestone1_report.md").write_text(report, encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
