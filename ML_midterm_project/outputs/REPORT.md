# Classical MILK10k baseline (extended)

## Methods

- Lesion-level 70/20/10 split, seed 42, stratified on `diagnosis_1`.
- Both images of a lesion share a split; probabilities are averaged before argmax.
- Original Session 5 features are preserved; new colorspace, filter-bank, GLCM/shape,
  and optional Canny/Harris/LBP/HOG/SIFT descriptors are selectable feature sets.
- Train-only balancing: none / oversample / undersample / targeted augmentation.
- Models: majority, logistic, decision tree, kNN, SVM (linear/RBF), random forest.
- Selection metric: lesion-level **macro-F1** on validation. Tie-breakers: malignant
  recall, fewer features, then simpler model. Test is scored once after freezing.

Winner: `svm` on `all_features` (balance=`oversample`) with `{'clf__estimator__C': 1.0, 'clf__estimator__kernel': 'rbf'}`.

## Satisfaction and F1

| Split | Accuracy | Precision | Recall | F1 | Satisfaction |
|-------|----------|-----------|--------|----|--------------|
| Validation | — | 0.552 | 0.593 | 0.567 | 0.571 |
| Test | 0.740 | 0.577 | 0.634 | 0.597 | 0.603 |
| Majority on test | — | 0.232 | 0.333 | 0.273 | 0.279 |

Test ROC-AUC (macro OVR): 0.8050339112091107
Test PR-AUC (macro): 0.6153221647710264
Malignant false negatives: 89 (rate 0.2445054945054945)

## Grouped cross-validation (train+val only)

- macro-F1 mean ± std: 0.484 ± 0.011
- precision mean: 0.495
- recall mean: 0.480

## Original vs extended

```
[{'feature_set': 'extended', 'model': 'forest', 'f1': np.float64(0.5060788469540897), 'precision': np.float64(0.5971434904457481), 'recall': np.float64(0.4936397831285497), 'balance': 'oversample'}, {'feature_set': 'class_33', 'model': 'forest', 'f1': np.float64(0.5238884678539851), 'precision': np.float64(0.6459585952168826), 'recall': np.float64(0.5031968006520965), 'balance': 'oversample'}, {'feature_set': 'all_features', 'model': 'svm', 'f1': np.float64(0.5671059083975215), 'precision': np.float64(0.55173880160395), 'recall': np.float64(0.5929328050488069), 'balance': 'oversample'}, {'feature_set': 'original_plus_filters', 'model': 'svm', 'f1': np.float64(0.557975357975358), 'precision': np.float64(0.5391150474546277), 'recall': np.float64(0.5969399018459082), 'balance': 'oversample'}]
```

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

## Top features

```
              feature  importance
gabor_o0_f0_frac_edge    0.010632
            Lab_a_std    0.010197
         lap3_q1_mean    0.009088
      sobel_y_q3_mean    0.008938
 glcm_correlation_std    0.008504
      sobel_x_q1_mean    0.008240
gabor_o1_f0_frac_edge    0.008115
    sobel_mag_q2_mean    0.004721
      hist_bin2_HSV_S    0.004407
gabor_o2_f0_frac_edge    0.004355
       kmeans_share_1    0.003116
         sobel_y_mean    0.003078
      hist_bin5_HSV_V    0.002981
      glcm_energy_std    0.002801
         lap5_q1_mean    0.002639
```

## Limitations

- Heuristic lesion masks are not clinical segmentations; no mm-calibrated diameter.
- No longitudinal images, so ABCDE evolution is not computed.
- Optional descriptors can be noisy (hair as edges; SIFT keypoint counts).
- Indeterminate remains rare; its per-class scores are unstable.
