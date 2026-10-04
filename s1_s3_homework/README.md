# MILK10k diagnosis pipeline (Milestone 1)

The task is to predict `diagnosis_1` (Benign, Malignant, or Indeterminate) from a skin-lesion photograph. Age, sex, and body site may be used later. They are not a substitute for the image. The stretch goal is the 11-class label in `training_gt.csv`.

MILK10k comes from the ISIC Archive (DOI 10.34970/648456). This copy has 10,480 images and 5,240 lesions, two images per lesion. The licence is CC-BY-NC. Attribution: MILK study team. Milestone 1 builds a leak-free split and a loader. It does not train a diagnostic model.

## Setup

Python 3.14. From this folder:

```text
pip install -r requirements.txt
```

The data folder is the `MILK10K_DIR` environment variable. If that variable is unset, the code uses `../milk10k` (the folder next to this one). That folder must contain `metadata.csv`, `supplements/training_gt.csv`, and `images/`.

```text
python homework_part_a.py
python run_milestone1.py
```

The first command reruns Part A. The submitted write-up of that part is `homework_part_a.pdf`. The second command checks every image, writes the splits, the label map, the class weights, the train-set normalisation statistics, and the figures.

## Repository structure

```text
s1_s3_homework/
  README.md                 this file
  SUBMISSION.md             links to the deliverables
  requirements.txt          the packages this code imports
  homework_part_a.py        part A source, runnable from top to bottom
  homework_part_a.pdf       part A export, with the answers and plots
  run_milestone1.py         writes the milestone 1 artefacts
  milk10k.py                paths, checks, splits, models, datasets, transforms
  milestone1_report.md      two-page milestone report
  outputs/                  splits, JSON, CSV, quality report
  figures/                  plots
```

All reusable code is in `milk10k.py`, including the train and eval transforms, so it is not copied into the notebook. Paths and the seed are set once at the top of that file. `outputs/` and `figures/` are generated. The raw images stay outside this folder and are not committed.

## Data handling

Raw images are not committed (`milk10k/` is listed in the repository `.gitignore`). Generated tables and figures are committed. The split seed is 42. The splits were created on 4 October 2026.

## Decisions so far

- Primary label: keep Indeterminate as a third class. See `milestone1_report.md`.
- Split: lesion-level `StratifiedGroupKFold` on the 11-class label, seed 42, 60/20/20.
- Preprocessing: resize to 224, scale by 255, then the mean and standard deviation of the training images.
- Imbalance: class weights for the loss, and a weighted sampler for the training loader.
- Do not use `diagnosis_2`, `diagnosis_3`, `diagnosis_4`, `diagnosis_confirm_type`, `concomitant_biopsy`, or `melanocytic` as inputs.
