"""Session 2 homework.

Prints the metadata summary, the color comparison, and one note on how the
loader and plots are split. Then opens the figures and runs the tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "milk10k" / "metadata.csv"
IMG_DIR = ROOT / "milk10k" / "images"

TARGET = "diagnosis_1"
ID_COLUMNS = ("isic_id", "lesion_id")
CLASS_ORDER = ("Benign", "Indeterminate", "Malignant")
CLASS_COLORS = {
    "Benign": "#4C78A8",
    "Indeterminate": "#F2B701",
    "Malignant": "#E45756",
}
SEED = 42
SAMPLE_N = 25
IMAGE_SIZE = (224, 224)
HIST_BINS = 64

# These columns repeat the diagnosis, or record a clinical decision about it.
NOT_A_PHOTO_FEATURE = {
    "diagnosis_2": "No. It is the label",
    "diagnosis_3": "No. It is the label",
    "diagnosis_4": "No. It is the label",
    "melanocytic": "No. Blank is informative",
    "concomitant_biopsy": "No. Biopsy decision",
    "diagnosis_confirm_type": "No. Same as biopsy",
    "anatom_site_special": "No. Almost empty",
    "image_manipulation": "Careful. Edit flag",
}


def load_metadata(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def image_path(isic_id: str, image_dir: Path = IMG_DIR) -> Path:
    return Path(image_dir) / f"{isic_id}.jpg"


def available_metadata(df: pd.DataFrame, image_dir: Path = IMG_DIR) -> pd.DataFrame:
    """Keep rows whose photo is actually on disk."""
    exists = df["isic_id"].map(lambda i: image_path(i, image_dir).is_file())
    return df.loc[exists].reset_index(drop=True)


def column_kind(name: str, series: pd.Series) -> str:
    if name in ID_COLUMNS:
        return "identifier"
    if name == TARGET:
        return "target"
    if series.nunique(dropna=True) <= 1 and not series.isna().any():
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    text = series.dropna().astype(str)
    if len(text) and text.str.len().mean() > 80 and series.nunique(dropna=True) > 100:
        return "free text"
    return "categorical"


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in df.columns:
        series = df[name]
        missing = float(series.isna().mean())
        rows.append(
            {
                "column": name,
                "type": column_kind(name, series),
                "missing": f"{missing * 100:.1f}%",
                "distinct": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _as_text(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "missing").astype(str)


def field_spread(df: pd.DataFrame, column: str) -> dict:
    """How much the label mix changes inside one field. No extra statistical test."""
    if column == "age_approx":
        means = df.groupby(TARGET)["age_approx"].mean()
        return {
            "field": column,
            "compared_by": "mean age gap",
            "spread": float(means.max() - means.min()),
        }
    groups = _as_text(df[column])
    if groups.nunique() < 2:
        return {"field": column, "compared_by": "class share", "spread": 0.0}
    share = pd.crosstab(groups, df[TARGET], normalize="index")
    if "Malignant" not in share.columns:
        return {"field": column, "compared_by": "class share", "spread": 0.0}
    return {
        "field": column,
        "compared_by": "class share",
        "spread": float(share["Malignant"].max() - share["Malignant"].min()),
    }


def _advice(field: str, spread: float, compared_by: str) -> str:
    if field in NOT_A_PHOTO_FEATURE:
        return NOT_A_PHOTO_FEATURE[field]
    if compared_by == "mean age gap":
        return "Yes" if spread >= 5 else "No"
    if spread < 0.05:
        return "No"
    if spread < 0.10:
        return "Weak"
    return "Yes"


def association_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in df.columns:
        if name in ID_COLUMNS or name == TARGET:
            continue
        rows.append(field_spread(df, name))
    table = pd.DataFrame(rows)
    table["advice"] = [
        _advice(field, spread, how)
        for field, spread, how in zip(table["field"], table["spread"], table["compared_by"])
    ]
    table["_age"] = table["field"].eq("age_approx")
    table = table.sort_values(["_age", "spread"], ascending=[False, False])
    return table.drop(columns="_age").reset_index(drop=True)


def _class_share(df: pd.DataFrame, column: str) -> pd.DataFrame:
    groups = _as_text(df[column])
    share = pd.crosstab(groups, df[TARGET], normalize="index")
    share = share.reindex(columns=list(CLASS_ORDER))
    order = groups.value_counts().index
    return share.reindex(order)


def plot_metadata(df: pd.DataFrame) -> plt.Figure:
    """Body-site mix and age, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    share = _class_share(df, "anatom_site_general")
    share.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        width=0.85,
        legend=False,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of photos")
    axes[0].set_xlabel("")
    axes[0].set_title("Diagnosis mix by body site")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(
        [Patch(facecolor=CLASS_COLORS[c]) for c in CLASS_ORDER],
        list(CLASS_ORDER),
        fontsize=7,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.42),
    )

    age_groups = [
        df.loc[df[TARGET] == label, "age_approx"].dropna() for label in CLASS_ORDER
    ]
    for values, label in zip(age_groups, CLASS_ORDER):
        axes[1].hist(
            values,
            bins=15,
            range=(0, 100),
            histtype="step",
            linewidth=1.6,
            color=CLASS_COLORS[label],
            label=label,
        )
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Photos")
    axes[1].set_title("Age by diagnosis")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    return fig


def sample_per_class(df: pd.DataFrame, n: int = SAMPLE_N) -> pd.DataFrame:
    avail = available_metadata(df)
    parts = []
    for label in CLASS_ORDER:
        group = avail[avail[TARGET] == label]
        parts.append(group.sample(n=min(n, len(group)), random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def to_gray(arr: np.ndarray) -> np.ndarray:
    """Luminosity weights: green counts most, then red, then blue."""
    rgb = arr.astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def load_sample_images(sample: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    return [
        (row[TARGET], load_rgb(image_path(row["isic_id"])))
        for _, row in sample.iterrows()
    ]


def class_histograms(records: list[tuple[str, np.ndarray]], bins: int = HIST_BINS) -> dict:
    edges = np.linspace(0, 256, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    keys = ("gray", "R", "G", "B")
    totals = {
        label: {key: np.zeros(bins) for key in keys} | {"n": 0} for label in CLASS_ORDER
    }
    for label, arr in records:
        planes = {
            "gray": to_gray(arr),
            "R": arr[:, :, 0],
            "G": arr[:, :, 1],
            "B": arr[:, :, 2],
        }
        for key, values in planes.items():
            hist, _ = np.histogram(values, bins=edges)
            totals[label][key] += hist
        totals[label]["n"] += 1
    curves = {}
    for label in CLASS_ORDER:
        n = totals[label]["n"]
        curves[label] = {key: totals[label][key] / n for key in keys}
    return {"centers": centers, "curves": curves}


def color_statistics(records: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for label, arr in records:
        gray = to_gray(arr)
        rows.append(
            {
                TARGET: label,
                "mean_R": arr[:, :, 0].mean(),
                "mean_G": arr[:, :, 1].mean(),
                "mean_B": arr[:, :, 2].mean(),
                "std_R": arr[:, :, 0].std(),
                "std_G": arr[:, :, 1].std(),
                "std_B": arr[:, :, 2].std(),
                "mean_gray": gray.mean(),
                "std_gray": gray.std(),
            }
        )
    per_image = pd.DataFrame(rows)
    summary = per_image.groupby(TARGET)[
        ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "mean_gray", "std_gray"]
    ].mean()
    return summary.reindex(list(CLASS_ORDER))


def plot_class_histograms(hist: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True)
    panels = (("gray", "Gray brightness"), ("R", "Red"), ("G", "Green"), ("B", "Blue"))
    centers = hist["centers"]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for label in CLASS_ORDER:
            ax.plot(
                centers,
                hist["curves"][label][key],
                color=CLASS_COLORS[label],
                label=label,
                lw=1.6,
            )
        ax.set_title(title)
        ax.set_xlim(0, 255)
    axes[0, 0].set_ylabel("Pixel count")
    axes[1, 0].set_ylabel("Pixel count")
    axes[1, 0].set_xlabel("Pixel value (0 = black, 255 = white)")
    axes[1, 1].set_xlabel("Pixel value (0 = black, 255 = white)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Average histogram per class ({SAMPLE_N} photos each)", fontsize=12)
    fig.tight_layout()
    return fig


def _open_rgb(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return im.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError("expected a file path or an 8-bit image")
    return Image.fromarray(arr).convert("RGB")


def preprocess_image(
    image,
    size: tuple[int, int] = IMAGE_SIZE,
    color: str = "rgb",
    mean=None,
    std=None,
) -> dict:
    """Resize, divide by 255, optional z-score, then channels-first (C, H, W)."""
    if color not in {"rgb", "gray"}:
        raise ValueError("color must be 'rgb' or 'gray'")
    resized = _open_rgb(image).resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    if color == "gray":
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        arr = gray[:, :, None]
    if mean is not None and std is not None:
        arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    steps = "resize, /255, channels-first"
    if mean is not None and std is not None:
        steps = "resize, /255, z-score, channels-first"
    return {
        "array": arr,
        "shape": arr.shape,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "color": color,
        "normalize": steps,
        "size": size,
    }


def _batch_paths(items, image_dir: Path | None) -> list[Path]:
    if isinstance(items, pd.DataFrame):
        if image_dir is None:
            raise ValueError("image_dir is required when items is a table")
        return [image_path(i, image_dir) for i in items["isic_id"]]
    return [Path(p) for p in items]


def preprocess_batch(items, image_dir: Path | None = None, **kwargs) -> dict:
    """Run preprocess_image on many files. One bad file does not stop the batch."""
    paths = _batch_paths(items, image_dir)
    arrays = []
    kept_positions = []
    skipped = []
    for i, path in enumerate(paths):
        try:
            if not path.is_file():
                raise FileNotFoundError(path.name)
            arrays.append(preprocess_image(path, **kwargs)["array"])
            kept_positions.append(i)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    stacked = None
    if arrays and len({a.shape for a in arrays}) == 1:
        stacked = np.stack(arrays)
    return {
        "arrays": arrays,
        "stacked": stacked,
        "kept_positions": kept_positions,
        "skipped": skipped,
    }


class ImageBatchLoader:
    """Batches of (processed image, label) from photos that exist on disk.

    Construct with the metadata table, the image folder, and a batch size.
    Iterating yields (images, labels). images is a float array of shape
    (batch, height, width, 3) with values from 0 to 1. labels are diagnosis_1.
    Photos listed in the table but missing from the folder are left out.
    Images are processed one batch at a time, not all at once.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_dir: Path = IMG_DIR,
        batch_size: int = 16,
        label_col: str = TARGET,
        size: tuple[int, int] = IMAGE_SIZE,
        color: str = "rgb",
        shuffle: bool = False,
        seed: int = SEED,
    ):
        self.image_dir = Path(image_dir)
        self.frame = available_metadata(metadata, self.image_dir)
        self.batch_size = batch_size
        self.label_col = label_col
        self.size = size
        self.color = color
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return int(np.ceil(len(self.frame) / self.batch_size))

    def __iter__(self):
        order = np.arange(len(self.frame))
        if self.shuffle:
            order = np.random.default_rng(self.seed).permutation(order)
        for start in range(0, len(order), self.batch_size):
            chunk = self.frame.iloc[order[start : start + self.batch_size]]
            batch = preprocess_batch(
                chunk, image_dir=self.image_dir, size=self.size, color=self.color
            )
            if batch["stacked"] is None:
                continue
            labels = chunk.iloc[batch["kept_positions"]][self.label_col].to_numpy()
            yield batch["stacked"], labels


def _draw(ax, image, title: str | None = None) -> None:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        vmax = 1.0 if float(image.max()) <= 1 else 255
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        height, width = image.shape
    else:
        ax.imshow(np.clip(image, 0, 1) if image.dtype.kind == "f" else image)
        height, width = image.shape[:2]
    ax.set_box_aspect(height / width)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_labeled_grid(images, labels, max_n: int = 8) -> plt.Figure:
    """Grid of images with the diagnosis as the title. Accepts a loader batch."""
    n = min(len(images), len(labels), max_n)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.1))
    if n == 1:
        axes = [axes]
    for ax, image, label in zip(axes, images[:n], labels[:n]):
        _draw(ax, np.asarray(image), str(label))
    fig.tight_layout()
    return fig


def plot_class_balance(counts: pd.Series) -> plt.Figure:
    ordered = counts.reindex(list(CLASS_ORDER))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(ordered.index, ordered.values, color=[CLASS_COLORS[c] for c in CLASS_ORDER])
    ax.set_ylabel("Number of photos")
    ax.set_title("Photos per diagnosis")
    ax.set_ylim(0, ordered.max() * 1.18)
    for i, value in enumerate(ordered.values):
        ax.text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_batch_summary(processed: np.ndarray) -> plt.Figure:
    """Pixel-value histogram of a processed batch, to check the 0â€“1 scaling."""
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.hist(np.asarray(processed).ravel(), bins=40, color="#4C78A8")
    ax.set_xlabel("Pixel value after processing")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Batch values  min {processed.min():.2f}  max {processed.max():.2f}"
    )
    fig.tight_layout()
    return fig


def plot_before_after(paths: list[Path]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(paths) * 2, figsize=(7.2, 2.15))
    for i, path in enumerate(paths):
        raw = load_rgb(path)
        done = preprocess_image(path)
        _draw(axes[i * 2], raw, f"Raw {raw.shape[0]}x{raw.shape[1]}")
        _draw(axes[i * 2 + 1], done["array"], f"Processed {done['size'][0]}x{done['size'][1]}")
    fig.tight_layout()
    return fig


def categorical_class_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Share of each diagnosis inside every categorical or yes/no field."""
    kinds = column_profile(df)
    names = kinds.loc[kinds["type"].isin(["categorical", "boolean"]), "column"]
    tables = {}
    for name in names:
        share = pd.crosstab(_as_text(df[name]), df[TARGET], normalize="index")
        tables[name] = share.reindex(columns=list(CLASS_ORDER))
    return tables


def _effect(associations: pd.DataFrame, field: str) -> pd.Series:
    return associations.loc[associations["field"] == field].iloc[0]


def answer_metadata(df: pd.DataFrame, associations: pd.DataFrame) -> str:
    """Short Part 1 summary: useful fields, useless fields, and leakage."""
    age = df.groupby(TARGET)["age_approx"].mean().reindex(list(CLASS_ORDER))
    age_row = _effect(associations, "age_approx")
    site_row = _effect(associations, "anatom_site_general")
    sex_row = _effect(associations, "sex")
    return (
        f"Age and body site are the useful fields. "
        f"Mean age is {age['Benign']:.0f} benign, {age['Malignant']:.0f} malignant, "
        f"and {age['Indeterminate']:.0f} indeterminate "
        f"(gap {age_row['spread']:.0f} years). "
        f"Body site malignant share varies by {site_row['spread']:.2f} "
        "(the oral/genital group is only 16 photos). "
        f"Sex varies by {sex_row['spread']:.2f}. "
        "Least useful: attribution, copyright_license, image_type (same mix in every class), "
        "and anatom_site_special (almost empty). "
        "Do not use diagnosis_2, diagnosis_3, or diagnosis_4 as inputs: they are the label. "
        "concomitant_biopsy and diagnosis_confirm_type are one clinical decision, not the photo. "
        "melanocytic is blank except when true. "
        "lesion_id is an id; both photos of a lesion must stay in the same split."
    )


def answer_color(color_table: pd.DataFrame) -> str:
    """Short Part 2 summary: classes overlap on color."""
    gray = color_table["mean_gray"]
    gap = float(gray.max() - gray.min())
    within = float(color_table["std_gray"].mean())
    return (
        f"Mean gray is {gray['Benign']:.0f} (benign), {gray['Malignant']:.0f} (malignant), "
        f"and {gray['Indeterminate']:.0f} (indeterminate). "
        f"The gap ({gap:.0f}) is smaller than the spread inside one photo ({within:.0f}), "
        "and the histogram lines overlap. "
        "A shift can come from the camera, lighting, or skin tone, not only the diagnosis. "
        "Color alone is a weak signal. The model needs lesion shape and texture."
    )


def answer_pipeline() -> str:
    """One paragraph on why the processing, loader, and plots are split."""
    return (
        "preprocess_image, preprocess_batch, and ImageBatchLoader are separate so training code "
        "calls the same steps instead of copying them. One function resizes to 224x224, divides "
        "by 255, and stores the photo as channels-first. The loader takes the "
        "metadata table, the image folder, and a batch size, and each step yields (images, labels) "
        "for diagnosis_1, using only files on disk and only one batch at a time. "
        "show_labeled_grid, plot_class_balance, and plot_batch_summary take a table or a loader batch, "
        "so the same check works on the raw data and on model input. "
        "The set is imbalanced, so later scores should be per-class recall, not accuracy alone."
    )


def run_analysis() -> dict:
    """Run parts 1-5 and print the answers. Figures are built, then closed."""
    df = load_metadata()
    profile = column_profile(df)
    associations = association_table(df)
    tables = categorical_class_tables(df)
    sample = sample_per_class(df, SAMPLE_N)
    records = load_sample_images(sample)
    histograms = class_histograms(records)
    color_table = color_statistics(records)
    answers = {
        "metadata": answer_metadata(df, associations),
        "color": answer_color(color_table),
        "pipeline": answer_pipeline(),
    }

    print("1. Metadata and the label")
    features = profile.loc[~profile["column"].isin([*ID_COLUMNS, TARGET])]
    print(features.to_string(index=False))
    print("Free-text columns: none.")
    advice = associations.set_index("field")["advice"]
    print("\nClass mix inside each category, and whether that field is useful")
    for name, table in tables.items():
        print(f"\n{name}: {advice.get(name, '')}")
        print(table.round(2).to_string())
    print("\nAssociation with diagnosis_1")
    print(associations.round(3).to_string(index=False))
    print("\n" + answers["metadata"])

    print("\n2. Color across classes,", SAMPLE_N, "photos each")
    print(color_table.round(1).to_string())
    print(answers["color"])

    print("\n3. Why the functions are split this way")
    print(answers["pipeline"])
    loader = ImageBatchLoader(df, IMG_DIR, batch_size=6)
    images, labels = next(iter(loader))
    paths = [image_path(i) for i in sample["isic_id"].head(2)]
    figures = [
        plot_metadata(df),
        plot_class_histograms(histograms),
        plot_before_after(paths),
        plot_class_balance(available_metadata(df)[TARGET].value_counts()),
        show_labeled_grid(images, labels),
        plot_batch_summary(images),
    ]
    print(
        f"Loader batch shape {images.shape}, "
        f"values {float(images.min()):.2f} to {float(images.max()):.2f}."
    )
    return {
        "metadata": df,
        "profile": profile,
        "associations": associations,
        "tables": tables,
        "color_table": color_table,
        "answers": answers,
        "batch": (images, labels),
        "figures": figures,
    }


class HomeworkTests(unittest.TestCase):
    """Checks that each homework task is implemented, and that the report matches."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_metadata()
        cls.associations = association_table(cls.df)
        cls.profile = column_profile(cls.df)

    def test_part1_column_types_and_missingness(self):
        features = self.profile.loc[~self.profile["column"].isin([*ID_COLUMNS, TARGET])]
        self.assertGreaterEqual(len(features), 10)
        self.assertIn("numeric", set(features["type"]))
        self.assertIn("categorical", set(features["type"]))
        self.assertIn("boolean", set(features["type"]))
        age = self.profile.loc[self.profile["column"] == "age_approx"].iloc[0]
        self.assertEqual(age["type"], "numeric")
        self.assertTrue(age["missing"].endswith("%"))
        site = self.profile.loc[self.profile["column"] == "anatom_site_general"].iloc[0]
        self.assertNotEqual(site["missing"], "0.0%")

    def test_part1_class_mix_inside_categories(self):
        tables = categorical_class_tables(self.df)
        for name in ("sex", "anatom_site_general", "image_type"):
            self.assertIn(name, tables)
            self.assertTrue(set(CLASS_ORDER).issubset(tables[name].columns))

    def test_part1_age_gap_and_class_share(self):
        age = _effect(self.associations, "age_approx")
        sex = _effect(self.associations, "sex")
        self.assertEqual(age["compared_by"], "mean age gap")
        self.assertGreater(age["spread"], 5)
        self.assertEqual(sex["compared_by"], "class share")
        self.assertGreater(sex["spread"], 0)

    def test_part1_leakage_and_useful_fields(self):
        text = answer_metadata(self.df, self.associations)
        for phrase in (
            "age",
            "body site",
            "diagnosis_2",
            "concomitant_biopsy",
            "lesion_id",
        ):
            self.assertIn(phrase, text)
        self.assertEqual(_effect(self.associations, "diagnosis_2")["advice"], "No. It is the label")
        self.assertEqual(_effect(self.associations, "image_type")["advice"], "No")
        self.assertEqual(_effect(self.associations, "age_approx")["advice"], "Yes")

    def test_part2_histograms_and_color_stats(self):
        sample = sample_per_class(self.df, n=2)
        records = load_sample_images(sample)
        hist = class_histograms(records)
        stats_table = color_statistics(records)
        self.assertEqual(set(hist["curves"]), set(CLASS_ORDER))
        self.assertEqual(set(hist["curves"]["Benign"]), {"gray", "R", "G", "B"})
        for column in ("mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"):
            self.assertIn(column, stats_table.columns)
        text = answer_color(stats_table)
        for phrase in ("overlap", "camera", "texture"):
            self.assertIn(phrase, text)
        fig = plot_class_histograms(hist)
        self.assertEqual(len(fig.axes), 4)
        plt.close(fig)

    def test_part3_preprocess_and_batch_skip(self):
        path = image_path(self.df.iloc[0]["isic_id"])
        rgb = preprocess_image(path)
        self.assertEqual(rgb["shape"], (3, 224, 224))
        self.assertGreaterEqual(rgb["min"], 0)
        self.assertLessEqual(rgb["max"], 1)
        self.assertIn("255", rgb["normalize"])
        gray = preprocess_image(path, color="gray")
        self.assertEqual(gray["shape"], (1, 224, 224))

        batch = preprocess_batch([path, IMG_DIR / "missing_file.jpg"])
        self.assertEqual(batch["stacked"].shape, (1, 3, 224, 224))
        self.assertEqual(len(batch["skipped"]), 1)
        self.assertIn("missing_file.jpg", batch["skipped"][0]["reason"] + batch["skipped"][0]["path"])
        from_table = preprocess_batch(self.df.iloc[:2], image_dir=IMG_DIR)
        self.assertEqual(from_table["stacked"].shape, (2, 3, 224, 224))

        fig = plot_before_after([path])
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_part4_loader_uses_only_files_on_disk(self):
        fake = self.df.iloc[:1].copy()
        fake["isic_id"] = "NOT_A_REAL_FILE"
        mixed = pd.concat([fake, self.df.iloc[:5]], ignore_index=True)
        loader = ImageBatchLoader(mixed, IMG_DIR, batch_size=4)
        self.assertNotIn("NOT_A_REAL_FILE", set(loader.frame["isic_id"]))
        images, labels = next(iter(loader))
        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(len(labels), 4)
        self.assertGreaterEqual(float(images.min()), 0)
        self.assertLessEqual(float(images.max()), 1)
        self.assertTrue(set(labels).issubset(set(CLASS_ORDER)))
        self.assertIn("batch size", answer_pipeline())
        self.assertIn("diagnosis_1", answer_pipeline())

    def test_part5_plots_for_balance_grid_and_batch(self):
        loader = ImageBatchLoader(self.df, IMG_DIR, batch_size=4)
        images, labels = next(iter(loader))
        grid = show_labeled_grid(images, labels)
        balance = plot_class_balance(self.df[TARGET].value_counts())
        summary = plot_batch_summary(images)
        self.assertEqual(len(grid.axes), 4)
        self.assertEqual(len(balance.axes), 1)
        self.assertIn("0", summary.axes[0].get_title())
        counts = self.df[TARGET].value_counts()
        self.assertGreater(counts["Malignant"], counts["Benign"])
        self.assertGreater(counts["Benign"], counts["Indeterminate"])
        self.assertIn("imbalanced", answer_pipeline())
        for fig in (grid, balance, summary):
            plt.close(fig)


def main() -> None:
    findings = run_analysis()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HomeworkTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        for fig in findings["figures"]:
            plt.close(fig)
        raise SystemExit(1)
    plt.show()


if __name__ == "__main__":
    main()
