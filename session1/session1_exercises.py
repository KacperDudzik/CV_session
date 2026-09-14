from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("milk10k")
IMG_DIR = DATA_DIR / "images"
DIAGNOSIS_COLS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
]


def load_data(data_dir: Path = DATA_DIR):
    metadata = pd.read_csv(data_dir / "metadata.csv")
    gt = pd.read_csv(data_dir / "supplements" / "training_gt.csv")
    return metadata, gt


def summarize_dataset(df: pd.DataFrame, gt: pd.DataFrame, img_dir: Path = IMG_DIR):
    n_images = len(list(img_dir.glob("*.jpg")))
    print("Number of images:", n_images)
    print("Metadata shape:", df.shape)
    print("GT shape:", gt.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("diagnosis_1 unique:", df["diagnosis_1"].unique())
    print(df["diagnosis_1"].value_counts())


def plot_class_counts(df: pd.DataFrame):
    counts = df["diagnosis_1"].value_counts()
    print(counts)
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Number of Samples per Class")
    plt.xlabel("Diagnosis Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()


def plot_subclass_counts(gt: pd.DataFrame):
    counts = (
        gt[DIAGNOSIS_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .sum()
        .sort_values(ascending=False)
    )
    print(counts)
    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Number of Samples per Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_age_distribution(df: pd.DataFrame):
    age = df["age_approx"].dropna()
    print("Minimum age:", age.min())
    print("Maximum age:", age.max())
    print("Age range:", age.max() - age.min())
    print(age.describe())
    plt.figure(figsize=(8, 5))
    plt.hist(age, bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Approximate Age")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()


def plot_sex_by_class(df: pd.DataFrame):
    sex_distribution = pd.crosstab(df["diagnosis_1"], df["sex"])
    print(sex_distribution)
    sex_distribution.plot(kind="bar", figsize=(8, 5))
    plt.title("Sex Distribution per Diagnosis Class")
    plt.xlabel("Diagnosis Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.legend(title="Sex")
    plt.tight_layout()
    plt.show()


def plot_site_by_class(df: pd.DataFrame):
    site_distribution = pd.crosstab(df["diagnosis_1"], df["anatom_site_general"])
    print(site_distribution)
    site_distribution.plot(kind="bar", figsize=(10, 5))
    plt.title("Anatomical Site per Diagnosis Class")
    plt.xlabel("Diagnosis Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.legend(title="Site")
    plt.tight_layout()
    plt.show()


def run_eda(df: pd.DataFrame, gt: pd.DataFrame):
    summarize_dataset(df, gt)
    plot_class_counts(df)
    plot_subclass_counts(gt)
    plot_age_distribution(df)
    plot_sex_by_class(df)
    plot_site_by_class(df)


def main():
    df, gt = load_data()
    run_eda(df, gt)


if __name__ == "__main__":
    main()
