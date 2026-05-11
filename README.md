# Breast Cancer Tumor Classification

**CS 4824 Final Project**

Anika Joshi-Awasthi, Brianna Wang, Fetra Ramiandrisoa

---

## Overview

This project compares seven machine learning classifiers on three breast cancer datasets with different measurement modalities. The goal is to classify tumors as malignant or benign, focusing on recall (sensitivity) as the primary metric, since a missed malignancy is clinically more harmful than a false positive.

### Models Implemented
- Logistic Regression (LR) (untuned baseline)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Neural Network (MLP)
- Gradient Boosting (GB)
- XGBoost
- Stacking Ensemble (LR + RF + KNN + SVM + MLP with LR meta-learner)

### Datasets

| Dataset | Source | Samples | Features | Feature Type |
|---|---|---|---|---|
| Wisconsin Diagnostic | Kaggle / UCI | 569 | 30 | Cell nucleus geometry (FNA imaging) |
| Coimbra Breast Cancer | Kaggle / UCI | 4000 | 9 | Blood biomarkers |
| Mammographic Mass | UCI (id=161) | ~830 | 5 | Clinical + radiological |

---

## Repository Structure

```
.
├── FinalProject.ipynb        # Main notebook — all code, analysis, and visualizations
├── Coimbra_breast_cancer_dataset.csv   # Dataset 2 (must be uploaded manually to Colab)
├── breast-cancer.csv                   # Dataset 1 (must be uploaded manually to Colab)
└── README.md
```

> Dataset 3 (Mammographic Mass) is fetched automatically via `ucimlrepo`, so no manual download is needed.

---

## Requirements

All code runs in Google Colab. The following packages are used:

| Package | Notes |
|---|---|
| `pandas`, `numpy`, `matplotlib`, `seaborn` | Pre-installed in Colab |
| `scikit-learn` | Pre-installed in Colab |
| `xgboost` | Pre-installed in Colab |
| `mlxtend` | Install once: `!pip install mlxtend` (cell included in notebook) |
| `ucimlrepo` | Install once: `!pip install ucimlrepo` (cell included in notebook) |

---

## How to Run

1. Open the notebook in Google Colab
   - Upload `FinalProject.ipynb` to [colab.research.google.com](https://colab.research.google.com) or open directly from GitHub via `File > Open notebook > GitHub`

2. Upload the datasets
   - In the Colab left sidebar, click the folder icon
   - Click the upload icon or drag the files to upload `breast-cancer.csv` and `Coimbra_breast_cancer_dataset.csv`
   - These files must be re-uploaded each time the Colab session times out

3. Run all cells in order
   - `Runtime > Run all` (or `Ctrl+F9`)
   - The `mlxtend` and `ucimlrepo` install cells only need to run once per session

> Cells must be run top to bottom. Variables like trained models, scalers, and split data are reused across cells — running out of order will cause errors.

---

## Design Decisions

Feature selection:

Wisconsin has 30 features; the top 10 by ANOVA F-score are used for distance-based models (LR, KNN, SVM, MLP) to reduce noise. RF, GB, and XGBoost use all 30 features since tree-based models handle high dimensionality well. Coimbra and Mammographic use all features (too few to warrant selection).

Custom recall scorer:

All `GridSearchCV` calls use a custom scorer that pins `pos_label=1` (malignant/patient) to prevent the optimizer from gaming recall by predicting the majority class.

Decision threshold:

RF on Wisconsin uses a lowered threshold of 0.3 (vs. default 0.5) to further prioritize recall. The threshold sensitivity plot in the notebook justifies this choice.

Stacking ensemble:

Uses the top-10 scaled feature set so all base learners share a consistent input matrix. GB and XGBoost are excluded as base learners since they were trained on a different feature matrix (all 30 features).

Class imbalance:

RF and SVM use `class_weight="balanced"`. XGBoost uses `scale_pos_weight` set to the ratio of negative to positive training samples.
