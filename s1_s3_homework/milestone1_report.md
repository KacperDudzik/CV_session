# Milestone 1

## Label strategy

The primary target is diagnosis_1 with three classes. Indeterminate is kept. There are 123 indeterminate lesions, all of them AKIEC, and they are not safe to call benign or malignant. Dropping them would train the model as if every lesion were decidable. Merging them into malignant would create false cancer calls.

The 11-class stretch goal keeps every class, including MAL_OTH (9 lesions), BEN_OTH (44), VASC (47), INF (50), and DF (52). Merging these into "other" would hide the diagnosis. Class weights from the training split are stored in `outputs/class_weights.json`. A score on MAL_OTH will be unstable because a test set holds only one or two of those lesions.

## Splits

Lesions are split with StratifiedGroupKFold on the 11-class label, seed 42, on 4 October 2026. About 60% train, 20% validation, 20% test. Images are assigned after the lesion split. Checked in code: no lesion appears in two splits, and each lesion has exactly two images in its split. The largest deviation of an 11-class share from the global share is 0.0008. diagnosis_1 shares by split:

diagnosis_1  Benign  Indeterminate  Malignant
split                                        
test          0.284          0.019      0.697
train         0.283          0.024      0.693
val           0.282          0.027      0.691

## Imbalance on the training images

diagnosis_1
Malignant        4360
Benign           1778
Indeterminate     150

Malignant is 69% of the training images. Accuracy will look strong for a model that always says malignant. Training uses class weights and a weighted sampler. The number to watch is malignant recall, or the cost of a miss, not accuracy.

## Quality

Every metadata row has an image file. Unreadable files: 0. Width ranges from 600 to 600 pixels (median 600). Height ranges from 450 to 450 (median 450). The fraction of images at least 224 pixels on both sides is 1.00.

Missing age is imputed with the training median inside the model only. Missing body site becomes "unknown". `anatom_site_special` is not used. `diagnosis_2`, `diagnosis_3`, `diagnosis_4`, `diagnosis_confirm_type`, `concomitant_biopsy`, and `melanocytic` are not inputs: they repeat the label or the decision to biopsy.

## Preprocessing

Input size is 224x224. Almost every image is larger than 224, so this only downsamples. Pixels are divided by 255, then shifted by the training-set mean and standard deviation (`outputs/norm_stats.json`). The same statistics are used at evaluation.

Each training item is one image with the lesion label. The other two options are: use only the dermoscopic photo, or feed both views as one item. Evaluation must average the two views of a lesion. Scoring each image separately double-counts patients and can test a photo whose sibling was in the training set.

Augmentations, and why they do not change the label, are in `outputs/augmentation_table.csv`. Hue jitter is off: a hue factor of 0.02 already moves colour by more than the benign-malignant gap. The evaluation transform has no random step; applying it twice gives the same tensor. One training epoch took 56 seconds with `num_workers=0`.

## Why this set is not a clinic

MILK10k does not show how often skin cancer appears among people who ask about a mark. Of 10,480 images, 10,032 were confirmed by histopathology and 448 by one clinical look. In the histopathology group, 72% of images are malignant. In the clinical group, 2% are malignant. A lesion was kept because someone already chose a biopsy, or because it was collected for this study. People with ordinary benign spots that are never removed are mostly absent. Malignant is therefore the majority class here, about 69% of images, while it is the minority in screening. Accuracy on MILK10k answers how often a prediction matches this archive. It does not answer how often the same rule would be right for the next new patient. A model can look accurate by saying malignant most of the time, because that is what the files contain. The number that matches the clinical cost is whether a malignant lesion is missed. The 11-class task has the same problem in a sharper form: MAL_OTH has nine lesions because the archive was not sampled to match how rare that disease is. A high score on the common classes can hide a useless score on the rare ones. Read a held-out accuracy with that selection in mind.

## What can still go wrong

A grouped split removes the direct leak of one view into training and the other into test. It does not remove near-duplicate lesions of different people, or a camera style that tracks a clinic. The set is enriched for biopsied cancers, so accuracy on MILK10k is not the accuracy in a screening clinic. A clinician should treat the score as a study-set estimate, not as a rate of cancer in the next patient who walks in.
