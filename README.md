# 🌾 RiPred — Rice Intelligence & Yield Prediction Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)]()

**RiPred** is a modular, production-grade machine learning system for Indian rice yield prediction, agro-climatic crop condition discovery, and harvest risk intelligence. 

Spanning **253 districts across 19 Indian states (2000–2015)**, RiPred combines strictly chronological machine learning pipelines, unsupervised agro-ecological clustering, split-conformal uncertainty intervals, a FastAPI inference engine, and an interactive 8-page Streamlit dashboard.

---

## 🏛️ Project Architecture

```
RiPred/
├── configs/
│   └── config.yaml               # Central configuration & parameters
├── data/
│   ├── raw/
│   │   └── rice.csv              # Source Kaggle dataset (2000-2015)
│   ├── interim/
│   └── processed/
│       ├── train.csv             # Chronological Train Split (2000-2011)
│       ├── val.csv               # Chronological Validation Split (2012-2013)
│       └── test.csv              # Chronological Test Split (2014-2015)
├── models/
│   ├── preprocessor.joblib       # Fitted scikit-learn preprocessing pipeline
│   ├── feature_schema.json       # Feature column and tercile schema
│   ├── crop_condition_clusterer.joblib # K-Means & PCA clusterer
│   ├── cluster_profiles.json     # Empirical agro-climatic cluster descriptions
│   ├── best_yield_regressor.joblib # Production Gradient Boosting Regressor
│   ├── regression_metrics.json   # Full regression benchmark results
│   ├── best_risk_classifier.joblib # Calibrated Risk Classifier (Low/Normal/High)
│   ├── classification_metrics.json # Full classification benchmark results
│   ├── hybrid_comparison_report.json # Model A vs Model B evaluation report
│   ├── conformal_predictor.joblib # Calibrated split-conformal uncertainty model
│   └── conformal_evaluation.json # Test coverage & error margin statistics
├── reports/
│   ├── data_audit_report.md      # Comprehensive Phase 1 Audit Report
│   ├── model_evaluation_report.md # Full Phase 2-7 Evaluation Report
│   ├── permutation_importance.csv # Permutation feature importances
│   ├── feature_drift_report.csv  # Kolmogorov-Smirnov drift analytics
│   ├── prediction_error_analysis.json # Error breakdown by state and year
│   └── figures/                  # High-resolution analytical charts
│       ├── target_distribution.png
│       ├── state_yield_comparison.png
│       ├── yearly_trend.png
│       ├── correlation_heatmap.png
│       ├── water_stress_vs_yield.png
│       └── zero_yield_districts.png
├── src/
│   ├── ingestion/
│   ├── preprocessing/            # Pipeline, scalers, imputers, and split logic
│   ├── feature_engineering/
│   ├── clustering/               # K-Means, Hierarchical, DBSCAN, profiling
│   ├── regression/               # Baseline, Ridge, Trees, RF, GB, XGBoost
│   ├── classification/           # Logistic Regression, RF, GB risk models
│   ├── hybrid/                   # Model A (Base) vs Model B (Hybrid) benchmark
│   ├── evaluation/               # Metrics and error segmentation
│   ├── explainability/           # Conformal prediction, permutation, drift
│   └── api/
│       └── main.py               # Production FastAPI inference service
├── dashboard/
│   └── app.py                    # 8-page Streamlit analytical dashboard
├── tests/
│   ├── test_preprocessing.py     # Pipeline & chronological isolation tests
│   ├── test_clustering.py        # Condition feature extraction & PCA tests
│   ├── test_models.py            # Regression, risk classifier & conformal tests
│   └── test_api.py               # FastAPI endpoint integration tests
├── notebooks/
│   └── initial_eda.ipynb         # Comprehensive EDA Jupyter notebook
├── main.py                       # Master orchestrator script
├── requirements.txt              # System dependencies
└── README.md
```

---

## 🔬 Core Methodologies & Audit Findings

### 1. Zero-Yield Anomaly & Cultivation Boundaries
During Phase 1 auditing, **344 zero-yield records** were identified:
- **336 records** correspond to uncultivated land (`rice_area_1000ha == 0.0`). In **12 districts** (e.g. Barmer, Jaisalmer, Churu, Jodhpur in Rajasthan's Thar Desert; Amreli, Jamnagar in Gujarat; and Lahul & Spiti in Himachal Pradesh), rice is never cultivated across all 16 recorded years.
- **8 records** represent true complete crop failures (`rice_area_1000ha > 0` but yield = 0).
- *Solution:* RiPred isolates active rice-growing environments ($N = 3,400$) while preserving true crop failures.

### 2. Strict Target Leakage Quarantine
- The mathematical relation $\text{Yield (kg/ha)} = \frac{\text{Production (1000 tons)} \times 1,000}{\text{Area (1000 ha)}}$ holds with a median discrepancy of $< 0.003\text{ kg/ha}$.
- Retaining `rice_production_1000tons` introduces fatal data leakage ($R^2 \approx 0.999$).
- *Solution:* `rice_production_1000tons` is **permanently quarantined and excluded** from all modeling pipelines.

### 3. Chronological Out-of-Sample Protocol
RiPred strictly avoids random k-fold cross-validation across time:
- **Training Set:** Crop Years 2000–2011 (2,502 active observations)
- **Validation Set:** Crop Years 2012–2013 (458 active observations)
- **Test Set:** Crop Years 2014–2015 (448 active observations)
- *Note:* The 2014–2015 test set coincided with severe consecutive all-India El Niño droughts, ensuring rigorous real-world stress testing.

---

## 📊 Benchmark Leaderboards

### Supervised Yield Regression (Target: `rice_yield_kgha`)

| Model | Val MAE (kg/ha) | Val RMSE (kg/ha) | Val R² | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² |
|---|---|---|---|---|---|---|
| **Historical Mean Baseline** | 798.73 | 1007.47 | -0.1335 | 823.71 | 1033.45 | -0.1416 |
| **Ridge Regression** | 469.68 | 632.73 | 0.5536 | 474.49 | 631.50 | 0.5729 |
| **Decision Tree Regressor** | 494.95 | 680.20 | 0.4841 | 526.15 | 688.46 | 0.4923 |
| **Random Forest Regressor** | 429.94 | 585.67 | 0.6174 | 458.15 | 605.35 | 0.6077 |
| **Gradient Boosting Regressor (Selected)** | **420.79** | **570.83** | **0.6382** | **452.36** | **592.59** | **0.6241** |
| **XGBoost Regressor** | 421.48 | 569.30 | 0.6401 | 456.79 | 596.83 | 0.6187 |

### Hybrid Architecture Decision (Model A vs. Model B)

| Model Variant | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² |
|---|---|---|---|
| **Model A: Base Gradient Boosting** | **452.36** | **592.59** | **0.6241** |
| **Model B: Hybrid (Base + Cluster)** | 453.76 | 593.00 | 0.6236 |
| **Delta ($\Delta$)** | **+1.40 kg/ha** | **+0.41 kg/ha** | **-0.0006** |

**Empirical Decision:** Adding unsupervised cluster features did not yield statistically meaningful improvements on unseen test droughts. In strict adherence to Project Rule 7, **RiPred retains clustering as an independent analytical component rather than forcing it into the prediction model**.

### Yield-Risk Classification

Thresholds derived from training terciles:
- **Low Yield:** $< 1,386.0\text{ kg/ha}$
- **Normal Yield:** $1,386.0 - 2,300.6\text{ kg/ha}$
- **High Yield:** $> 2,300.6\text{ kg/ha}$

| Classifier Architecture | Val Precision (w) | Val Recall (w) | Val F1 (w) | Val ROC-AUC (OvR) | Test F1 (w) |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.6083 | 0.6114 | 0.6094 | 0.8117 | 0.7015 |
| **Random Forest Classifier** | 0.6724 | 0.6703 | 0.6702 | 0.8436 | 0.6846 |
| **Gradient Boosting Classifier (Selected)** | **0.6874** | **0.6834** | **0.6838** | **0.8511** | **0.7322** |

### Conformal Uncertainty Calibration
- **Target Confidence:** 90%
- **Empirical Coverage on Test Shocks (2014–2015):** **87.3%**
- **Calibrated Residual Margin:** $\pm 863.9\text{ kg/ha}$

---

## 🚀 Quickstart & Execution

### 1. Installation
```bash
git clone https://github.com/Shanmugavedhan/RiPred.git
cd RiPred
pip install -r requirements.txt
```

### 2. Run Master Training & Benchmarking Pipeline
```bash
python main.py
```

### 3. Launch the FastAPI Inference Service
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation is available at: `http://127.0.0.1:8000/docs`

### 4. Launch the Streamlit Analytical Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Execute Pytest Test Suite
```bash
python -m pytest -v tests
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and loaded model verification |
| `GET` | `/meta` | Target schema, units, risk thresholds, and feature lists |
| `POST` | `/predict/yield` | Continuous yield prediction (kg/ha) with 90% conformal intervals |
| `POST` | `/predict/risk` | Categorical risk prediction (Low/Normal/High) with class probabilities |
| `POST` | `/predict/condition` | Unsupervised cluster assignment and PCA coordinates |
| `POST` | `/predict/comprehensive`| Unified prediction delivering yield, risk, and condition intelligence |
| `GET` | `/clusters` | Centroid summaries for all 4 agro-climatic clusters |
| `GET` | `/evaluation` | Full regression, classification, and hybrid leaderboards |

---

## 💻 Streamlit Dashboard Tour

The interactive dashboard features 8 dedicated modules:
1. **Project Overview:** High-level executive metrics, product architecture, and dataset spatial coverage.
2. **Rice Yield Prediction:** Real-time inference UI calculating predicted kg/ha, conformal bounds, and risk tier badges.
3. **Crop Condition Analysis:** Granular profiles and environmental distributions for all 4 discovered cluster archetypes.
4. **Cluster Visualization:** 2D PCA scatter projection illustrating cluster separation and outliers.
5. **Yield Risk Analysis:** Multi-class classification breakdown, probability distribution, and risk mitigation guidance.
6. **Historical Yield Trends:** Time-series trends (2000–2015) correlating rainfall volatility with harvest output.
7. **Model Performance:** Transparent, un-fabricated leaderboards across all candidate algorithms.
8. **Feature Importance:** Permutation importance bar charts and Kolmogorov-Smirnov distribution drift monitoring.
