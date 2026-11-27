# Model card: MILK10k lesion classifier (`models/model_11class.pt`)

## Model

- **Architecture:** ResNet-18 pretrained on ImageNet (torchvision `IMAGENET1K_V1`); `layer3`,
  `layer4` and a new `fc = nn.Linear(512, 11)` trained, earlier layers frozen
  (10.5 M of 11.2 M weights trained).
- **Training:** Adam, lr 1e-4, batch 64, 20 epochs, the epoch with the best validation macro-F1
  kept (here epoch 9); cross-entropy with oversampling of rare classes; random horizontal and
  vertical flips. Seed 0 of 3.
- **Output:** 11 softmax probabilities in the order AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH,
  MEL, NV, SCCKA, VASC. `predict()` returns these. For the leaderboard file only, each class is
  rescaled with a validation-tuned threshold so that the threshold sits at 0.5.
- **Selected how:** the recipe with the highest mean validation macro-F1 among 40 experiments
  on the 3-class task (Path A of the brief), then retrained once with 11 outputs.
- **Companion:** `models/model_m4.pt`, the same recipe with 3 outputs (Benign / Indeterminate /
  Malignant).

## Intended use

A course project (EADA, Computer Vision) for learning how to build, evaluate and report a
computer-vision pipeline honestly, and for the official MILK10k leaderboard.

## Not for

- Any clinical decision: whether to biopsy, treat, reassure or discharge a patient.
- Screening the general population: the training archive is biopsy-enriched (69% malignant),
  so the model's priors are wrong for a screening clinic.
- Clinical close-up photographs, phone photos, or any image that is not a dermoscopic image
  like those in MILK10k.
- People with very dark skin (skin tones 0–1 are 2% of the data; 9 test lesions).
- Commercial use (data licence CC-BY-NC).

## Data

MILK10k (ISIC Archive, doi:10.34970/262082), dermoscopic image of each of 5,240 lesions; lesion-level
split 3,668 / 1,048 / 524 (train / validation / test), frozen in `splits/`. 96% of diagnoses
confirmed by histopathology. Labels: `training_gt.csv` (11 classes).

## Input

One dermoscopic JPEG, RGB. Preprocessing (identical in training, validation and `predict()`):
PIL resize to 224×224, scale to [0, 1], ImageNet mean (0.485, 0.456, 0.406) and std
(0.229, 0.224, 0.225). No random step at inference.

## Performance

| Metric | Validation (1,048 lesions) | Held-out test (524 lesions) |
|---|---|---|
| Macro-F1, 11 classes (argmax) | 0.467 | 0.405 |
| Leaderboard F1 (threshold 0.5), raw | 0.447 | 0.385 |
| Leaderboard F1, validation thresholds | 0.495 (optimistic) | 0.402 |
| **Official ISIC score** (479 blind lesions, 11 classes incl. MAL_OTH; approach EADA-CV26-KD) | — | **0.360** (= local 0.402 x 10/11 with MAL_OTH at 0) |

3-class companion (`model_m4.pt`), test split: macro-F1 0.618, accuracy 0.79; malignant recall
0.84 (57 of 363 malignant lesions called benign); Indeterminate recall 0.25 (3 of 12).

Per class (11 classes, validation): BCC F1 0.84 and NV 0.70 are reliable; MEL 0.61, SCCKA 0.56,
AKIEC 0.44, BKL 0.42 are moderate; BEN_OTH 0.18, INF 0.31, DF 0.50 rest on 6–10 lesions;
**MAL_OTH 0.00** (2 validation lesions, never predicted correctly).

## Limitations and risks

- **Missed cancers.** On our test split the 3-class model calls 16% of malignant lesions benign,
  and gets 21 of 41 melanomas wrong. Errors are often made with high confidence (p ≥ 0.95).
- **Rare classes** (MAL_OTH, BEN_OTH, INF, DF, VASC: 9–52 lesions each in the whole archive)
  are close to unlearnable at this data size; their scores move by large amounts between seeds.
- **Non-pigmented lesions** (pale, pink) are the most common failure; the classical lesion mask
  also fails on them, so they are hard in general, not just for this network.
- **Fairness.** No difference by sex (macro-F1 0.61 vs 0.62). Accuracy is similar across skin
  tones 2–5 (0.74–0.88); dark skin (tones 0–1) is too rare to evaluate.
- **Optimistic validation.** The recipe was chosen as the best of 40 on one validation set;
  validation overstates test macro-F1 by about 0.09. Use the test and official numbers.

## Reproduce

`README.md`, section "Run everything". Requires `requirements.txt` (PyTorch 2.14, CUDA optional).
