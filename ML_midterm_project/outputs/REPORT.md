# Classical MILK10k baseline (extended)

## Methods

- Lesion-level 70/20/10 split, seed 42, stratified on `diagnosis_1`.
- Both images of a lesion share a split; probabilities are averaged before argmax.
- Original Session 5 features are preserved; new colorspace, filter-bank, GLCM/shape, and optional Canny/Harris/LBP/HOG/SIFT descriptors are selectable feature sets.
- Train-only balancing: none / oversample / undersample / targeted augmentation.
- Models: majority, logistic, decision tree, kNN, SVM (linear/RBF), random forest.
- Selection metric: lesion-level **macro-F1** on validation. Tie-breakers: malignant recall, fewer features, then simpler model. Test is scored once after freezing.

Winner: `svm` on `all_features` (balance=`oversample`) with `{'clf__estimator__C': 1.0, 'clf__estimator__kernel': 'rbf'}`.

## Satisfaction and F1

| Split | Accuracy | Precision | Recall | F1 | Satisfaction |
|-------|----------|-----------|--------|----|--------------|
| Validation | — | 0.552 | 0.593 | 0.567 | 0.571 |
| Test | 0.740 | 0.577 | 0.634 | 0.597 | 0.603 |
| Majority on test | — | 0.232 | 0.333 | 0.273 | 0.279 |

- Test ROC-AUC (macro OVR): 0.805
- Test PR-AUC (macro): 0.615
- Malignant false negatives: 89 (rate 0.245)

## Grouped cross-validation (train+val only)

- macro-F1 mean ± std: 0.484 ± 0.011
- precision mean: 0.496
- recall mean: 0.480

## Original vs extended (best validation F1 per set)

| Feature set | Model | Balance | Precision | Recall | F1 |
|-------------|-------|---------|-----------|--------|-----|
| extended | forest | oversample | 0.597 | 0.494 | 0.506 |
| class_33 | forest | oversample | 0.646 | 0.503 | 0.524 |
| original_plus_filters | svm | oversample | 0.539 | 0.597 | 0.558 |
| all_features | svm | oversample | 0.552 | 0.593 | 0.567 |

## Test, per class

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Benign | 0.571 | 0.730 | 0.641 | 148 |
| Indeterminate | 0.294 | 0.417 | 0.345 | 12 |
| Malignant | 0.865 | 0.755 | 0.806 | 364 |

## Test confusion (rows are the truth)

```
               Benign  Indeterminate  Malignant
Benign            108              3         37
Indeterminate       1              5          6
Malignant          80              9        275
```

## Limitations

- Heuristic lesion masks are not clinical segmentations; no mm-calibrated diameter.
- No longitudinal images, so ABCDE evolution is not computed.
- Optional descriptors can be noisy (hair as edges; SIFT keypoint counts).
- Indeterminate remains rare; its per-class scores are unstable.
