# Course Notes — Sessions 1–5

Structured reference for solving MILK10k exercises the way class defined them.
Use class kernels, formulas, and evaluation rules before inventing alternatives.

Course: Computer Vision and Speech Recognition (EADA). Dataset: MILK10k (ISIC).

---

## Course map

| Session | Theme | What you produce |
|--------|--------|------------------|
| S1 | What CV is; MILK10k EDA | Load metadata/images; class imbalance; 2 images/lesion |
| S2 | Pixels → numbers | RGB/gray, histograms, normalization |
| S3 | Data prep | Leak-free split, imbalance, augmentation, Dataset |
| S4 / 4.1 | Filters & convolution | Hand kernels; filter → map → feature |
| S5 | Classical ML baseline | 33 features → sklearn models → macro-F1 |
| Midterm | After S4 | — |
| S6–S10 | ANN → CNN → tasks → ViT → deploy | Compare to S5 baseline |

---

## Golden rules (always)

1. **Meaning ≠ raw pixels.** Infer scene properties; a single pixel is useless without context.
2. **Split by `lesion_id`**, never by image row. Prefer `StratifiedGroupKFold`.
3. **Test set once.** Tune on validation only.
4. **Norm stats from train only.** Apply the same transform at inference.
5. **Accuracy lies under imbalance.** Prefer macro-F1, per-class recall, confusion matrix.
6. **Medical augmentations are claims** that the label is unchanged — hue jitter is risky; flips/small rotation/mild brightness usually OK.
7. **Baseline before DL.** Classical 33-feature pipeline is the number later CNNs must beat.
8. **Match class code.** Use kernels and feature definitions from the slides/notebooks, not Session-4 homework variants, unless the exercise asks for them.

---

## Session 1 — Introduction & MILK10k

### Core ideas
- Image = spatial measurement of a physical phenomenon; digitally `F(X, Y)` sampled to a pixel grid.
- Human vision is hierarchical (edges → shapes → objects) — inspiration for CNNs.
- Perception recovers scene properties (objects, depth, materials), not just light intensities (illusions).
- CV tasks by output shape:
  - **Classification** `f(I) → c`
  - **Detection** `f(I) → {(box, class)}`
  - **Segmentation** `f(I) →` mask H×W
- Project task: **classification** on MILK10k.

### MILK10k facts to hard-wire
- **10,480 images**, **5,240 lesions**, exactly **2 images/lesion** (dermoscopic + clinical close-up).
- `metadata.csv` = one row per **image**; `supplements/training_gt.csv` = one row per **lesion** (11-class one-hot).
- `diagnosis_1`: Benign / Malignant / Indeterminate.
- 11 classes: BCC, NV, BKL, SCCKA, MEL, AKIEC, DF, INF, VASC, BEN_OTH, MAL_OTH (BCC dominates ~48%).
- Link images with `lesion_id`; image files `images/{isic_id}.jpg`.
- Sampling is **biopsy-enriched** (~69% malignant) — not clinic prevalence.

### Practice habits
- `Image.open(...).convert("RGB")` then `np.array` → shape `(H, W, 3)`, `uint8`, 0–255.
- Always plot class counts before modeling.
- Reusable grid viewer by diagnosis.
- Confirm `(groupby lesion_id).size() == 2` and both image types per lesion.

---

## Session 2 — Image representation

### Digitization
- **Sampling** → spatial resolution (H×W).
- **Quantization** → bit depth (usually 8-bit → 0–255).

### Indexing (print shapes always)
| API | Order |
|-----|--------|
| PIL `.size` | `(W, H)` |
| NumPy `.shape` | `(H, W, C)` |
| Indexing | `[row, col]` = `[y, x]` |

### Color
- RGB trichromacy; array `H × W × 3`.
- Other spaces: **HSV** (separate hue from brightness), **Lab** (perceptual distance).
- Grayscale luminosity (class / OpenCV):
  ```text
  Gray = 0.299·R + 0.587·G + 0.114·B
  ```
- For MILK10k, color often carries diagnostic signal — do not convert to gray too early for the final model; gray is fine for classical filters/texture.

### Histograms
- Counts (or fractions) of pixels per intensity bin; one hist per channel for RGB.
- Left mass → dark; right → bright; narrow → low contrast.

### Normalization (same at train and inference)
| Method | Formula | When |
|--------|---------|------|
| Min–max | `x / 255` | Default to [0, 1] |
| Z-score | `(x − mean) / std` | Zero-centered models |
| Dataset stats | mean/std on **train** | Domain-specific (dermatoscopy) |
| ImageNet stats | ≈[0.485, 0.456, 0.406] / stds | Transfer learning |

Pipeline order class uses: load → resize → `/255` → (optional) standardize → **HWC → CHW**.

---

## Session 3 — Data preparation

### Split roles
| Split | Job | How often |
|-------|-----|-----------|
| Train | Fit weights | Every step |
| Val | Tune / early stop / compare | Every epoch |
| Test | Honest final score | **Once** |

### Leakage types (MILK10k)
| Type | Example |
|------|---------|
| Group | Same `lesion_id` in train and test |
| Preprocessing | mean/std computed on full data including test |
| Duplicate | Near-duplicate photos of same lesion on both sides |
| Target | Features that encode the label (e.g. `diagnosis_confirm_type`) |

**Rule:** split on grouping unit + stratify → `StratifiedGroupKFold`.

### Imbalance
- Accuracy meaningless (always-BCC / always-Malignant look “good”).
- Prefer: precision/recall per class, **macro-F1**, PR-AUC under imbalance.
- Error cost asymmetric: FN on melanoma ≫ FP on nevus.
- Tools: class weights, oversampling, undersampling (last resort), targeted aug, stratified splits.
- Rare classes (e.g. MAL_OTH ~9): CV estimates unstable — report with caution.

### Augmentation (train only)
- Safe-ish: horizontal flip, small rotation (~±15°), mild brightness/contrast, careful crop.
- Risky: strong **hue** jitter (pigmentation/erythema are diagnostic); crops that remove the lesion.
- Vertical flip OK for skin (no fixed “up”); wrong for anatomy with orientation (X-ray).

### Pipeline order
Load + validate → group+stratify split → freeze test → augment train only → Dataset/DataLoader.

### Missingness (class guidance)
- `age_approx`: median impute (fit on train) or drop.
- `anatom_site_general`: keep explicit `unknown`.
- `diagnosis_4` often empty by design; `melanocytic` NaN can mean N/A.

---

## Session 4 — Filters & convolution

### Definition
At each position: multiply kernel weights by the patch under the window, sum → one output pixel.
Repeated over the image → **feature map**.

Formal (class): `(I * K)(x,y) = Σᵢ Σⱼ I(x+i,y+j)·K(i,j)`.

DL frameworks usually implement **cross-correlation** (no 180° flip) and still call it convolution.

### Output size
```text
out = floor((n + 2·padding − k) / stride) + 1
```
- **Padding:** `valid` (shrink) vs `same` (preserve size). Mirror/reflect avoids fake dark borders from zero-pad.
- **Stride:** 1 dense; 2+ downsamples (CNN style).

### Class kernels (memorize)

```text
BOX 3×3:     ones(3,3)/9                         sum=1
GAUSSIAN:    [[1,2,1],[2,4,2],[1,2,1]]/16       sum=1
SHARPEN:     [[0,-1,0],[-1,5,-1],[0,-1,0]]       sum=1
OUTLINE:     [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]   sum=0
EMBOSS:      [[-2,-1,0],[-1,1,1],[0,1,2]]        sum=1
SOBEL_X:     [[-1,0,1],[-2,0,2],[-1,0,1]]        sum=0
SOBEL_Y:     [[-1,-2,-1],[0,0,0],[1,2,1]]        sum=0
IDENTITY:    center 1                              sum=1
```

- **sum ≈ 1** → brightness-preserving (flat region stays flat).
- **sum ≈ 0** → edge detectors (flat → 0).
- Edge magnitude: `√(Gx² + Gy²)` = `np.hypot(Gx, Gy)`.
- Display tip: for sum≠1 maps, percentile stretch (1st–99th), not raw min–max.

### Live-lab conventions (`class_live_session4`)
- Toy 5×5 lesion border: skin=100, lesion=10.
- Manual `conv_step` / `conv2d`; fast path `ndimage.correlate(..., mode="reflect")`.
- Real demo image often `ISIC_0073863`, grayscale `"L"`, sometimes resized.

### Homework themes (S4)
- Compare filtered benign vs malignant.
- `gaussian_filter` **sigma** = std in pixels.
- Implement Canny; compare to Sobel.

---

## Session 4.1 — Filters as features (bridge to S5 / CNN)

1. **Filter = question** asked at every pixel → feature map (highlight, don’t decide).
2. **Map → few numbers:** mean, std, percentiles, fraction above threshold, per-quadrant means = **features**. CNNs call the analogous step **pooling**.
3. **Feature vector** = several filters × several summaries.
4. How many filters? Start small (3–12), different questions/scales; keep what helps **validation**.
5. Classical limit: we only find what we thought to look for; CNNs **learn** kernel weights and stack layers.

Classical vs CNN (same convolution op):

| | Classical S4–5 | CNN S7+ |
|--|----------------|---------|
| Filters | Hand-chosen | Learned |
| Summarize | mean/std/hist | Pooling |
| Decide | Separate classifier | End layers jointly trained |
| Data | Small OK | Needs more data/GPU |

Optional names (not required in class): Canny, Harris, Gabor, LBP, HOG, SIFT.

---

## Session 5 — Classical ML for vision

### Pipeline
```text
Image (810k values)
  → handcrafted features (33)
  → classical model
  → Benign / Malignant / Indeterminate
```

### Canonical 33-feature vector (class order)

| Block | Count | Names / definition |
|-------|------:|--------------------|
| Color stats | 6 | `RED_mean`, `GREEN_mean`, `BLUE_mean`, `RED_std`, `GREEN_std`, `BLUE_std` |
| Histograms | 24 | 8 bins × R,G,B → fractions `counts / n_pixels`, `range=(0,256)` |
| Texture | 3 | `edge_density`, `roughness`, `laplacian_variance` |

**Texture on simple gray** `gray = rgb.mean(axis=2)` (not luminosity unless asked):

```python
gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
edge_density = np.hypot(gx, gy).mean()

local_mean = ndimage.uniform_filter(gray, size=9)
local_sq_mean = ndimage.uniform_filter(gray**2, size=9)
roughness = (local_sq_mean - local_mean**2).mean()   # mean(x²)−mean(x)² in 9×9

laplacian_variance = ndimage.laplace(gray).var()
```

Notes:
- Color/hist ignore **where** pixels are → texture needed.
- Hairs are strong edges → edge density partly measures hair.
- Optional later: GLCM (Haralick), ABCDE shape features (need mask / Session 8).
- Standardize features with **train** μ,σ for kNN / logistic; trees/forests don’t need scaling.

### Models to compare
| Model | Idea | Scaling? |
|-------|------|----------|
| Majority class | Always train-mode class | Floor baseline |
| kNN | Vote of k nearest | Yes |
| Logistic regression | Linear weights → sigmoid; `class_weight="balanced"` | Yes |
| Decision tree / Random forest | Axis splits; forest = many trees | No |
| SVM | Max-margin (+ kernels) | Usually yes |

Same features can disagree across models — that is expected.

### Evaluation recipe (class)
1. Lesion-level stratified group split.
2. Majority baseline → must beat its **macro-F1**, not its accuracy (~69% always-malignant ≈ 0.27 macro-F1 in the slide example).
3. Logistic on standardized features.
4. Stronger models (kNN, RF) — compare on **validation**.
5. Report final **macro-F1** (and confusion / per-class recall) on **test once**.

Confusion matrix: rows = truth, columns = prediction; diagonal = correct; row → recall, column → precision; F1 = `2PR/(P+R)`; macro = unweighted mean over classes.

---

## Exercise playbook

When a new homework lands:

1. **Locate the session** and open the matching slide PDF / live notebook.
2. **Reuse class definitions** (kernels, gray method, hist bins, texture formulas, metrics).
3. **Respect S3 data rules** (lesion groups, train-only stats, no test peeking).
4. **Keep scripts computational**; PDF/report separate unless asked.
5. **Relative paths / `MILK10K_DIR`** — no hardcoded machine paths.
6. **Print named features / metrics** so results match the brief’s naming.
7. **Compare to majority + previous baseline** before claiming progress.

### Known class demo IDs
- Benign often: `ISIC_0073863`
- Malignant often: `ISIC_0051817`
- Toy 5×5: top-left bright 100s, bottom-right dark 10s (lesion corner).

### What not to mix up
| Topic | Class (S4/S5) | Elsewhere in this repo |
|-------|---------------|------------------------|
| Gray for texture | `rgb.mean(axis=2)` | S4 homework luminosity Sobel |
| Edge density | full-map `.mean()` of hypot | Valid 3×3 interior only in toy printout |
| Hist bins for 33-vec | **8** per channel | Toy slide used **4** bins |
| Convolution impl | `correlate` / slide-window | True math conv flips kernel |

---

## Coming later (orientation only)

- **S6** ANNs · **S7** CNNs (learned filters; beat S5 baseline) · **S8** detection/segmentation · **S9** ViTs · **S10** end-to-end + multimodal.
- Official MILK10k leaderboard submission is part of the course project.

---

## Source files in this repo

- `session1/Session 1 - Introduction to Computer Vision  Real-World AI Challenges.pptx.pdf`
- `session2/Session 2 - Image Representation  Digital Image Fundamentals.pptx.pdf`
- `session3/Session 3 - Data Preparation for Machine Learning Projects.pptx.pdf`
- `session4/Session 4 - Filters  Convolution.pptx (2).pdf`
- `session4/Session 4.1 - Why Convolutions - Filters as Features.pptx.pdf`
- `session4/class_live_session4 (1).pdf`
- `session5/Session 5 - Classical Machine Learning for Vision.pptx.pdf`
- `session5/session5_code.ipynb` (feature definitions in code)
