# MILK10k lesion classification: final pipeline (M2–M5)

One pipeline, one split, one metric. Every model of the course (Sessions 5–9) is trained and
scored the same way; the winner on validation is retrained with 11 outputs for the official
MILK10k leaderboard (Session 10). The code reuses the class modules (`session5/`, `session7/`,
`session8/`) instead of copying them.

## Results in one table

| | Val macro-F1 | Test macro-F1 | Leaderboard F1 (held-out test) |
|---|---|---|---|
| M2 random forest, 33 features | 0.537 ± 0.001 | 0.511 ± 0.023 | — |
| Final 3-class model (`models/model_m4.pt`) | 0.710 | 0.618 | — |
| Final 11-class model (`models/model_11class.pt`, submitted) | 0.467 | 0.405 | 0.402 local; **0.360 official** (ISIC, EADA-CV26-KD) |

**Where the 11-class score comes from** (held-out test, leaderboard F1, same predictions):

| | Leaderboard F1 |
|---|---|
| Uniform probabilities (1/11 each) | 0.000 |
| Always "BCC" (the majority class) | 0.065 |
| Random guess in proportion to class frequency | 0.099 |
| **Final 11-class model** | **0.402** (4× random guessing) |
| — on the 6 common classes (AKIEC, BCC, BKL, MEL, NV, SCCKA; 207–1,760 training lesions each) | 0.568 |
| — on the 4 rare classes present in test (BEN_OTH, DF, INF, VASC; 31–38 training lesions each) | 0.153 |

The macro mean weights every class equally, so the five rare classes (3.9% of all lesions,
3–8 test lesions each) pull the score from 0.57 to 0.40. MAL_OTH (7 training lesions) is absent
from our test split but present in the official one. This is a data limit, not a weighting
problem: oversampling already shows the network every class equally often.

Final recipe: ResNet-18, layer3 + layer4 + new head fine-tuned, lr 1e-4, 20 epochs, oversampling.
Chosen on validation among 40 experiments x 3 seeds. Report: [report.md](report.md);
model card: [model_card.md](model_card.md); full scoreboard:
[outputs/diagnosis_1/scoreboard.md](outputs/diagnosis_1/scoreboard.md);
leaderboard file: [submission/submission.csv](submission/submission.csv).

## Layout

| File | Role | Class method it reuses |
|---|---|---|
| `config.py` | every setting: paths, targets, split, experiments, seeds | — |
| `data.py` | frozen lesion-level split, labels (3 or 11 classes), image cache, `load_image()` | S3 split by lesion, S7 `make_dermoscopic_splits`, S10 resize rule |
| `models.py` | `build_model(arch, train, n_classes)`, checkpoint loading, probabilities | S7 `SimpleCNN`, `make_partially_unfrozen_resnet`, `prep`; S9 ViT-B/16 |
| `metrics.py` | macro-F1 (argmax) and the leaderboard macro-F1 (threshold 0.5) | S5 macro-F1; brief slide 15 |
| `baselines.py` | M2 baselines on 33 Session 5 features: random forest, logistic regression, kNN, SVM, two MLPs | S5 `feature_vector` and models, midterm SVM, S6 `make_mlp` + regularisation |
| `train.py` | trains experiments × seeds, saves weights, history, validation probabilities | S6/S7 `fit` (class weights, flips, best-val checkpoint) |
| `evaluate.py` | scoreboard; validation picks the winner; test scored once | S5/S9 fair-comparison rules |
| `analysis.py` | confusion matrix, misclassified images + Otsu masks, fairness | S5 confusion, S8 `segment`/`otsu` |
| `predict.py` | `predict(image_path)` → class probabilities | S10 inference transform |
| `make_submission.py` | `submission.csv` + the brief's 4 checks | brief slides 14–19 |
| `check_project.py` | audit of the whole pipeline (writes `outputs/audit.txt`) | — |
| `report.md`, `model_card.md` | final report (the brief's 7 sections) and model card | — |
| `make_report_pdf.py` | `report.md` → `report.pdf` (Moodle report and ISIC manuscript), via headless Edge/Chrome | — |
| `submission/` | `submission.csv`, `approach_description.txt` (ISIC form text), `leaderboard.png` (screenshot) | — |

Generated folders: `splits/` (frozen lesion lists, committed), `runs/<target>/<experiment>/seed<k>/`
(result.json, history.csv, val_probs.csv committed; weights not committed), `outputs/` (scoreboards,
figures, tables), `models/` (`model_m4.pt`, `model_11class.pt`), `logs/` (console log of every step),
`cache/` (decoded images, not committed).

## Run everything

Python 3.14, packages as in the course plus a CUDA build of PyTorch (any build works; CUDA is faster):

```text
pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Data in `../milk10k` (or set `MILK10K_DIR`). From this folder:

```text
python check_project.py                     # audit before training
python baselines.py                         # M2 on the same split (6 feature-based models)
python train.py                             # M3/M4: every experiment x 3 seeds (diagnosis_1)
                                            #   = stage1 + sweep + architectures + regularization;
                                            #   one stage only: python train.py --exp sweep
python evaluate.py --test                   # M4 scoreboard; winner -> models/model_m4.pt; test once
python analysis.py                          # M4 error analysis + fairness
python train.py --target dx11 --exp best    # M5: same architecture, 11 outputs
python evaluate.py --target dx11            # best seed -> models/model_11class.pt, local estimate
python make_submission.py                   # needs isic_test/ (below) -> submission/submission.csv
python check_project.py                     # audit after
```

Every script can be stopped and restarted: finished runs are skipped.

ISIC test data: download `MILK10k_Test_Input.zip` and `MILK10k_Test_Metadata.csv` from
challenge.isic-archive.com/data and place them as `isic_test/MILK10k_Test_Input/<lesion>/<isic>.jpg`
and `isic_test/MILK10k_Test_Metadata.csv`. Upload `submission/submission.csv` as
`EADA-CV26-KD` and save the screenshot as `submission/leaderboard.png`.

## Rules for experiments

These keep every number comparable. Breaking one makes the scoreboard meaningless.

1. **One split.** `splits/*.csv` is created once and never edited. All models train on `train`,
   choose epochs and models on `val`, and touch `test` only in `evaluate.py --test`.
2. **One place for settings.** A new model is one new entry in `config.EXPERIMENTS`
   (`arch`, `train` = layer-name prefixes to train, `lr`, optional `batch_size`). Nothing else changes.
3. **Same training for all deep models.** `session7.fit`: class-weighted cross-entropy (S6),
   random flips (S3/S7), Adam, `EPOCHS` epochs, the epoch with the best validation macro-F1 is kept.
   lr 1e-3 when only a new head trains, 1e-4 when backbone layers are unfrozen (S7).
4. **Seeds.** Every experiment runs with every seed in `config.SEEDS`; report mean ± std.
   A difference smaller than the std is not a difference.
5. **Selection.** The deep experiment with the highest mean validation macro-F1 wins; its best
   seed becomes `model_m4.pt`. The test split is scored once, for every model, after selection.
   `evaluate.py --test` refuses to run twice (`--force` only with a written reason).
6. **Inference = validation.** `data.load_image` (RGB, PIL resize 224) + `session7.prep`
   (ImageNet mean/std), no random step. Training, `predict()` and the submission all use it.
7. **11 classes for the leaderboard.** Retrain the winner with `--target dx11`; same split,
   same recipe, only the last layer changes (S10 slide 22). Report its validation macro-F1
   and the leaderboard-style F1 next to the official score.
8. **Changing the split or the image size** means a new cache and rerunning everything; never
   mix runs from two splits in one table.
9. **Audit.** Run `check_project.py` before and after any change; it must end with
   "all checks passed".
10. **Hyperparameter sweep (stage 2).** Only the knobs taught in class change: learning rate
    and number of epochs (`config.SWEEP`, run with `python train.py --exp sweep`). Each variant
    is an ordinary experiment with all seeds; it competes on validation like any other row,
    and the sweep must be finished before the test split is scored.
11. **Stages.** `config.STAGES` groups the experiments in the order they were run:
    `stage1` (course models), `sweep` (learning rate, epochs), `architectures` (VGG-16,
    DenseNet-121, EfficientNet-B0), `regularization` (dropout, weight decay, M1 augmentation,
    oversampling vs class weights, train-set normalisation; one factor at a time), and
    `final_candidates` (only the changes that helped, combined on the two best architectures).
    Every stage ends before the test split is scored; all rows compete on validation.
12. **Leaderboard thresholds (optional, brief slide 15).** `evaluate.py --target dx11` tunes one
    threshold per class on validation (classes with fewer than 5 validation lesions keep 0.5),
    stores them in `model_11class.pt`, and reports the validation (optimistic) and held-out test
    (honest) effect. `make_submission.py` rescales the probabilities so each threshold sits at
    0.5; `--raw` writes plain softmax probabilities instead. `predict()` always returns the
    plain probabilities.
