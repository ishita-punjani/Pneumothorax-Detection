# Pneumothorax Detection — Classical (CNN-Free) Pipeline

Detecting pneumothorax (lung collapse) from chest X-rays using traditional image
processing and classical machine learning — no CNNs or deep learning at any stage.

Built on a balanced 500-image subset of the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/achmadbauravindah/nih-chest-xrays-original)
(250 Pneumothorax, 250 No Finding), using LBP and Sobel handcrafted features with
SVM and Random Forest classifiers.

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| SVM (RBF) | 65.2% | 0.653 | 0.738 | 0.693 | 0.661 |
| Random Forest | 69.0% | 0.711 | 0.702 | 0.707 | 0.753 |

Literature baseline ([Chan et al., 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5903299/)): 85.8% accuracy,
using true lung segmentation rather than the ROI approximation used here.

## Repository structure

```
Pneumothorax-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_cleaning.ipynb              # Dataset acquisition & subset selection
│   └── Pneumothorax_Detection_Pipeline.ipynb  # Preprocessing, features, training, evaluation
├── models/                                 # Pre-trained SVM, Random Forest, scaler (joblib)
│   ├── feature_cols.joblib
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   └── svm_model.joblib
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

4. Run `notebooks/Pneumothorax_Detection_Pipeline.ipynb` — covers ROI extraction,
   LBP/Sobel feature extraction, patient-wise train/test split, feature scaling,
   SVM and Random Forest training, and evaluation (confusion matrices, ROC curves).
5. Trained models save automatically to `models/`.

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

- **Preprocessing:** grayscale loading, gamma correction (γ=2.0). No resizing,
  denoising, or CLAHE (deliberately excluded — see implementation plan).
- **ROI:** a fixed bilateral elliptical approximation, refined with Otsu thresholding
  constrained to the ellipse (to avoid background bleed from unconstrained Otsu),
  followed by morphological cleanup.
- **Features:** Local Binary Pattern (LBP, uniform, P=8, R=1) and Sobel edge
  magnitude, both split left/right with asymmetry features (36 total).
- **Split:** patient-wise 70/30, stratified, with an "any-positive" grouping rule for
  the 5 patients with mixed-label images across visits — prevents patient leakage.
- **Classification:** SVM (RBF) and Random Forest, both grid-searched via 5-fold
  cross-validation.
- **Tested and rejected:** GLCM features and RFECV feature selection were both
  implemented and evaluated but did not improve performance on this dataset —
  documented in the literature review as a negative result rather than omitted.


## Future work

- Scale to the full balanced subset (~5,302 Pneumothorax images + matched negatives)
- Web frontend for image upload and prediction
- Modularize the pipeline into reusable `src/` modules
