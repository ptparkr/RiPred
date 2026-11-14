# RiPred v1.0 — Comprehensive Model Evaluation & Architecture Report

## 1. Executive Summary
RiPred v1 is fully trained, benchmarked, and serialized. All transformations, cluster assignments, and regression/classification models strictly observe chronological separation (Train: 2000–2011, Validation: 2012–2013, Test: 2014–2015).

## 2. Supervised Yield Regression Leaderboard

| Model Architecture | Train MAE | Val MAE (kg/ha) | Val RMSE (kg/ha) | Val R² | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² |
|---|---|---|---|---|---|---|---|
| **baseline_mean** | 800.5 | 798.73 | 1010.42 | -0.1335 | 823.71 | 1051.54 | -0.1416 |
| **ridge_regression** | 349.05 | 469.68 | 634.07 | 0.5536 | 474.49 | 643.18 | 0.5729 |
| **decision_tree** | 290.35 | 494.95 | 681.69 | 0.4841 | 526.15 | 701.29 | 0.4923 |
| **random_forest** | 195.64 | 429.94 | 587.07 | 0.6174 | 458.15 | 616.47 | 0.6077 |
| **gradient_boosting** | 210.58 | 420.79 | 570.85 | 0.6382 | 452.36 | 603.38 | 0.6241 |
| **xgboost** | 215.14 | 421.48 | 569.34 | 0.6401 | 456.79 | 607.72 | 0.6187 |

**Selected Best Regressor:** `gradient_boosting`

## 3. Unsupervised Condition Discovery (k=4)
- **K-Means Silhouette:** 0.1719
- **Hierarchical Silhouette:** 0.1493
- **DBSCAN Anomaly Count:** 208 (8.31%)

### Condition Profiles:
- **Cluster 0: Moderate Rainfall / Balanced Agro-Climatic Zone** (42.69% of active area): Mean Rain = 919.31 mm, Mean Stress = 144.48, Top States: madhya pradesh, maharashtra, rajasthan
- **Cluster 1: Moderate Rainfall / Balanced Agro-Climatic Zone** (5.96% of active area): Mean Rain = 1208.97 mm, Mean Stress = 50.65, Top States: himachal pradesh, uttarakhand
- **Cluster 2: Semi-Arid / High Moisture Stress Zone** (30.74% of active area): Mean Rain = 641.77 mm, Mean Stress = 152.9, Top States: uttar pradesh, punjab, bihar
- **Cluster 3: Humid / High Monsoon Rainfall Zone** (20.62% of active area): Mean Rain = 1917.95 mm, Mean Stress = 47.61, Top States: kerala, west bengal, assam

## 4. Hybrid Architecture Evaluation (Model A vs Model B)
- **Model A (Base gradient_boosting):** Test MAE = 452.36 kg/ha, Test R² = 0.6241
- **Model B (Hybrid gradient_boosting with Cluster):** Test MAE = 453.76 kg/ha, Test R² = 0.6236
- **Δ MAE:** 1.4 kg/ha | **Δ R²:** -0.0006
- **Decision:** Retain Clustering as Independent Analytical Component (Do not force into prediction model)

## 5. Yield-Risk Classification Leaderboard

| Classifier Architecture | Val Precision (w) | Val Recall (w) | Val F1 (w) | Val ROC-AUC | Test F1 (w) |
|---|---|---|---|---|---|
| **logistic_regression** | 0.6543 | 0.5983 | 0.6094 | 0.8117 | 0.7015 |
| **random_forest** | 0.7187 | 0.6638 | 0.6702 | 0.8436 | 0.6846 |
| **gradient_boosting** | 0.729 | 0.6747 | 0.6838 | 0.8511 | 0.7322 |

**Selected Best Classifier:** `gradient_boosting`  
**Thresholds (derived from training percentiles):**
- Low: < 1386.0 kg/ha
- Normal: 1386.0 - 2300.6 kg/ha
- High: > 2300.6 kg/ha

## 6. Conformal Reliability & Uncertainty Intervals
- **Nominal Target Confidence:** 90%
- **Actual Empirical Coverage on Unseen Test (2014-2015):** 87.3%
- **Calibrated Error Margin:** ± 863.9 kg/ha
