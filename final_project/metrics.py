"""The two numbers we report.

macro_f1        our metric since Session 5: argmax prediction, F1 per class, plain mean.
leaderboard_f1  the official MILK10k metric (brief, slide 15): every probability >= 0.5
                counts as "yes, this class", F1 per class, plain mean of the classes.

Both average over the classes PRESENT in y_true: a class with no lesion in an evaluation set
(MAL_OTH in our test split) has no defined F1 and is left out instead of counted as 0.
"""

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    present = sorted(set(np.asarray(y_true).tolist()))
    assert max(present) < n_classes
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def leaderboard_f1(y_true, probs) -> float:
    probs = np.asarray(probs)
    onehot = np.eye(probs.shape[1], dtype=int)[np.asarray(y_true)]
    present = onehot.sum(axis=0) > 0
    return float(f1_score(onehot[:, present], (probs[:, present] >= 0.5).astype(int),
                          average="macro", zero_division=0))


def tune_thresholds(y_true, probs, grid, min_positives: int) -> np.ndarray:
    """Brief slide 15 (optional): one threshold per class, chosen on VALIDATION only.

    The leaderboard F1 is a mean of independent per-class F1s, so each class is tuned on
    its own (one-vs-rest). Classes with fewer than min_positives lesions keep 0.5.
    """
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    thresholds = np.full(probs.shape[1], 0.5)
    for c in range(probs.shape[1]):
        truth = y_true == c
        if truth.sum() < min_positives:
            continue
        scores = [f1_score(truth, probs[:, c] >= t, zero_division=0) for t in grid]
        thresholds[c] = grid[int(np.argmax(scores))]
    return thresholds


def rescale(probs, thresholds) -> np.ndarray:
    """Monotone map per class: threshold t -> 0.5, 0 -> 0, 1 -> 1. Values stay in [0, 1]."""
    p, t = np.asarray(probs, dtype=float), np.asarray(thresholds, dtype=float)
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))
