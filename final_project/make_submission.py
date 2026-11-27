"""submission.csv for the official MILK10k leaderboard (M5; brief slides 14-19).

  python make_submission.py            # rescaled with the validation thresholds (if the model has them)
  python make_submission.py --raw      # plain softmax probabilities

Needs the ISIC test files in isic_test/ (see README) and models/model_11class.pt.
One row per lesion (dermoscopic image only, as in class): 479 rows x 12 columns.
"""

import argparse

import pandas as pd
import torch

import config
import metrics
from predict import predict

SUBMISSION = config.HERE / "submission" / "submission.csv"
CLASSES = config.CLASSES["dx11"]
# The brief (slide 14) names the first column "lesion", but the ISIC validator rejects that:
# 'Missing column in CSV: "image" or "lesion_id"' (27/11/2026). The official name is lesion_id.
ID_COL = "lesion_id"


def check_submission(sub: pd.DataFrame) -> None:
    """The four checks from the brief (slide 19), with the column name ISIC accepts."""
    assert list(sub.columns) == [ID_COL] + CLASSES
    assert len(sub) == 479 and sub[ID_COL].is_unique
    assert sub[CLASSES].isna().sum().sum() == 0
    assert ((sub[CLASSES] >= 0) & (sub[CLASSES] <= 1)).all().all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="do not apply the validation thresholds")
    args = parser.parse_args()
    if not config.TEST_META_PATH.is_file():
        raise SystemExit(f"missing {config.TEST_META_PATH}: download the ISIC test files first (README)")
    meta = pd.read_csv(config.TEST_META_PATH)
    derm = meta[meta.image_type == "dermoscopic"]
    rows = []
    for lesion, isic in zip(derm.lesion_id, derm.isic_id):
        probs = predict(config.TEST_IMG_DIR / lesion / f"{isic}.jpg", config.MODEL_11)
        rows.append([lesion] + [probs[c] for c in CLASSES])
    sub = pd.DataFrame(rows, columns=[ID_COL] + CLASSES)

    thresholds = torch.load(config.MODEL_11, map_location="cpu", weights_only=False).get("thresholds")
    if thresholds is not None and not args.raw:
        sub[CLASSES] = metrics.rescale(sub[CLASSES].to_numpy(), thresholds)
        print("applied validation thresholds:", dict(zip(CLASSES, thresholds)))

    check_submission(sub)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION, index=False)
    print(f"wrote {SUBMISSION}: {sub.shape[0]} rows x {sub.shape[1]} columns, all checks passed")


if __name__ == "__main__":
    main()
