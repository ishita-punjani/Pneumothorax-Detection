# Pneumothorax Detection — Classical (CNN-Free) Pipeline

Detecting pneumothorax (lung collapse) from chest X-rays using traditional image
processing and classical machine learning — no CNNs or deep learning at any stage.

Built on a balanced 500-image subset of the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/achmadbauravindah/nih-chest-xrays-original)
(250 Pneumothorax, 250 No Finding), using LBP and Sobel handcrafted features with
SVM and Random Forest classifiers.

> **Pipeline versions:** This repository contains two versions of the classical pneumothorax detection pipeline.
>
> - **Pipeline V1** uses a coarse bilateral elliptical lung ROI followed by constrained Otsu thresholding and morphological refinement.
> - **Pipeline V2** introduces a classical lung segmentation pipeline using intensity-based thresholding, morphological processing, connected-component analysis, anatomical filtering, left/right lung-pair selection, and constrained region growing.
>
> Both versions use the same 36 handcrafted LBP, Sobel, and left/right asymmetry features with SVM and Random Forest classifiers. This allows the effect of the improved lung segmentation approach to be evaluated while keeping the feature-extraction methodology consistent.

## Results

### Pipeline V1 — Elliptical ROI

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM (RBF) | 65.19% | 0.6526 | 0.7381 | 0.6927 | 0.6613 |
| Random Forest | 71.52% | 0.7349 | 0.7262 | 0.7305 | 0.7449 |

### Pipeline V2 — Classical Lung Segmentation

Lung segmentation succeeded on **461/500 images (92.2%)**. The 39 failed segmentations are recorded in the feature CSV and excluded from model training and evaluation.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM (RBF) | 72.34% | 0.8000 | 0.6667 | 0.7273 | 0.7582 |
| Random Forest | 69.50% | 0.7465 | 0.6795 | 0.7114 | 0.7786 |

For a fair V1–V2 comparison, both pipelines were also evaluated on the same 141 successfully segmented V2 test images. V2 improved SVM accuracy from **69.50% to 72.34%** and ROC-AUC from **0.7126 to 0.7582**. For Random Forest, ROC-AUC increased from **0.7682 to 0.7786**, while accuracy decreased from **73.76% to 69.50%**. Therefore, V2 improved some classification metrics but did not universally outperform V1.

Literature baseline ([Chan et al., 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5903299/)): 85.8% accuracy,
using a dedicated lung segmentation approach; this result is included only as a literature reference and is not directly comparable to the present experiment.

## Repository structure

```
Pneumothorax-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── Pneumothorax_Detection_Pipeline_v1.ipynb
│   └── Pneumothorax_Detection_Pipeline_v2.ipynb
├── features/
│   ├── pneumothorax_extracted_features.csv
│   └── pneumothorax_extracted_features_v2.csv
├── models/
│   ├── feature_cols.joblib
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   ├── svm_model.joblib
│   └── v2/
│       ├── feature_cols.joblib
│       ├── rf_model.joblib
│       ├── scaler.joblib
│       └── svm_model.joblib
└── data/
    ├── images/
    └── pneumothorax_500_metadata.csv
```

`data/` is intentionally excluded from version control via `.gitignore` — it's
regenerated locally by running `01_data_cleaning.ipynb` (see Setup below). Do not
commit raw images or `kaggle.json` to this repository.

## Setup and reproduction

Everything in this pipeline runs locally, end to end.

1. **Get a Kaggle API token**: [kaggle.com](https://www.kaggle.com) → Account →
   *Create New API Token* — downloads `kaggle.json`. Place it at
   `~/.kaggle/kaggle.json` (never commit this file).
2. `pip install -r requirements.txt`
3. Run `notebooks/01_data_cleaning.ipynb`. This downloads the metadata CSV,
   selects a balanced 500-image subset (250 Pneumothorax, 250 No Finding), and
   downloads the corresponding images into `data/images/`.

   > **Note:** this notebook filters on an exact match (`Finding Labels == "Pneumothorax"`),
   > which finds 2,194 candidate images rather than the ~5,302 mentioned in the
   > implementation plan — the larger figure includes images with multiple
   > co-occurring findings (e.g. `"Pneumothorax|Effusion"`). Relevant if scaling up
   > the dataset later — use a `str.contains("Pneumothorax")` filter instead of an
   > exact match.
4. Run `notebooks/Pneumothorax_Detection_Pipeline_v1.ipynb` to reproduce Pipeline V1 using the elliptical ROI approach.
5. Run `notebooks/Pneumothorax_Detection_Pipeline_v2.ipynb` to reproduce Pipeline V2. This notebook performs classical lung segmentation, LBP and Sobel feature extraction, patient-wise splitting, SVM and Random Forest training, evaluation, and direct V1–V2 comparison.
6. Extracted features are saved as:
   - V1: `features/pneumothorax_extracted_features.csv`
   - V2: `features/pneumothorax_extracted_features_v2.csv`
7. Trained models are saved as:
   - V1: `models/`
   - V2: `models/v2/`

### Using the pre-trained models directly

The `models/` folder already contains trained models from the last run, so you can
skip retraining:

```python
import joblib

scaler = joblib.load("models/scaler.joblib")
rf_model = joblib.load("models/rf_model.joblib")
feature_cols = joblib.load("models/feature_cols.joblib")

# new_feature_df must have the same columns as feature_cols, in the same order
X_new = scaler.transform(new_feature_df[feature_cols])
predictions = rf_model.predict(X_new)
```

## Methodology summary

- **Preprocessing:** Grayscale image loading followed by gamma correction (γ=2.0).
- **Pipeline V1 ROI:** Fixed bilateral elliptical lung approximation followed by constrained Otsu thresholding and morphological cleanup.
- **Pipeline V2 lung segmentation:** Classical segmentation using outside-air removal, Gaussian smoothing, recursive/second Otsu thresholding, morphological closing and hole filling, 8-connected component analysis, anatomical candidate filtering, left/right lung-pair selection, constrained growing, and final mask cleanup.
- **Features:** Both pipelines use the same 36 handcrafted features based on Local Binary Pattern (LBP), Sobel edge magnitude, and left/right lung asymmetry.
- **Classification:** SVM with RBF kernel and Random Forest classifiers are trained using patient-wise train/test splitting. StandardScaler is applied before classification.
- **Segmentation handling:** V2 successfully segmented 461 of 500 images (92.2%). Failed segmentations remain recorded in the feature CSV but are excluded from model training and evaluation.
- **Visualizations:** The notebooks include before-and-after processing visualizations, lung segmentation stages, LBP texture representations, Sobel edge maps, feature distributions, confusion matrices, ROC curves, and V1–V2 performance comparisons.


## Future work

- Scale to the full balanced subset (~5,302 Pneumothorax images + matched negatives)
- Web frontend for image upload and prediction