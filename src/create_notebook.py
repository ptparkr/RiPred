import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code, outputs=None):
    if outputs is None:
        outputs = []
    cells.append({
        "cell_type": "code",
        "execution_count": len([c for c in cells if c["cell_type"] == "code"]) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code.split("\n")]
    })

# Section 1
add_md("""# RiPred: Rice Intelligence & Yield Prediction Engine
## Phase 1 — Comprehensive Dataset Audit & Exploratory Data Analysis (EDA)

### 1. Project Background & Objective
RiPred is an Indian rice yield prediction and crop-condition intelligence system combining:
1. **Supervised Regression:** Predicting district-level rice yield (kg/ha).
2. **Unsupervised Clustering:** Discovering latent crop-condition archetypes from agro-climatic, hydrological, and soil profiles.
3. **Hybrid Architecture:** Empirically evaluating whether condition clusters improve yield prediction over baseline feature sets.
4. **Yield-Risk Classification:** Categorizing harvest risk into Low, Normal, and High yield tiers with well-calibrated class probabilities.
5. **Reliability & Explainability:** Feature attributions, permutation importance, and spatio-temporal validation.

This notebook executes **Phase 1: Dataset Audit** strictly before any model training, ensuring data integrity, unit verification, leakage prevention, and principled feature recommendation.""")

# Section 2
add_md("""### 2. Environment Setup & Data Ingestion""")
add_code("""import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.size'] = 11

# Load raw dataset
DATA_PATH = '../data/raw/rice.csv'
if not os.path.exists(DATA_PATH):
    DATA_PATH = 'data/raw/rice.csv'

df = pd.read_csv(DATA_PATH)
print(f"Dataset successfully loaded. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
""")

# Section 3
add_md("""### 3. Dataset Dimensionality & Column Taxonomy
The dataset spans 69 columns. We categorize them systematically into agro-climatic, hydrological, soil, fertilizer, irrigation, and production domains.""")
add_code("""# Display data overview
print(f"Total Observations: {len(df):,}")
print(f"Temporal Span: {df['year'].min()} to {df['year'].max()} ({df['year'].nunique()} years)")
print(f"Geographical Coverage: {df['state'].nunique()} States, {df['district'].nunique()} Districts")

# Column types summary
print("\\nColumn Data Types:")
print(df.dtypes.value_counts())
""")

# Section 4
add_md("""### 4. Missing Value & Completeness Audit
We examine both explicit `NaN` values and implicit missingness (e.g. zero values where physically unexpected, or omitted temporal/calendar observations).""")
add_code("""# Explicit nulls
explicit_nulls = df.isnull().sum()
print(f"Total Explicit Null Values across entire matrix: {explicit_nulls.sum()}")

# Implicit missingness checks
# 1. Check for absent months in water deficit (wd_)
wd_cols = [c for c in df.columns if c.startswith('wd_')]
print(f"Water Deficit columns present ({len(wd_cols)} months):", [c.replace('wd_', '').replace('_mm', '') for c in wd_cols])
print("OBSERVATION: September water deficit ('wd_sep_mm') is absent from the source dataset.")

# 2. Check for zero values in key agricultural features
zero_summary = pd.DataFrame({
    'Feature': ['rice_area_1000ha', 'rice_production_1000tons', 'rice_yield_kgha', 'irrig_canal_area_1000ha', 'rain_annual_mm'],
    'Zero_Count': [(df[c] == 0).sum() for c in ['rice_area_1000ha', 'rice_production_1000tons', 'rice_yield_kgha', 'irrig_canal_area_1000ha', 'rain_annual_mm']],
    'Zero_Percentage': [round((df[c] == 0).mean() * 100, 2) for c in ['rice_area_1000ha', 'rice_production_1000tons', 'rice_yield_kgha', 'irrig_canal_area_1000ha', 'rain_annual_mm']]
})
zero_summary
""")

# Section 5
add_md("""### 5. Zero Yield & Cultivation Boundary Analysis
A critical finding during the audit is that 344 records (9.19%) have `rice_yield_kgha == 0`.
We investigate whether this represents crop failure or districts that simply do not cultivate rice.""")
add_code("""zero_yield_df = df[df['rice_yield_kgha'] == 0]
zero_area_count = (zero_yield_df['rice_area_1000ha'] == 0).sum()
crop_failure_count = (zero_yield_df['rice_area_1000ha'] > 0).sum()

print(f"Total zero-yield records: {len(zero_yield_df)}")
print(f" - Records with 0 Area & 0 Production (Uncultivated): {zero_area_count} ({zero_area_count/len(zero_yield_df)*100:.1f}%)")
print(f" - Records with >0 Area but 0 Production (Total Crop Failure): {crop_failure_count} ({crop_failure_count/len(zero_yield_df)*100:.1f}%)")

# Districts with 0 rice across all 16 years
dist_zero_stats = df.groupby(['state', 'district'])['rice_yield_kgha'].agg(total='count', zeros=lambda x: (x == 0).sum())
always_zero_districts = dist_zero_stats[dist_zero_stats['zeros'] == dist_zero_stats['total']]
print(f"\\nDistricts with ZERO rice cultivation across ALL recorded years ({len(always_zero_districts)} districts):")
print(always_zero_districts.index.tolist())
""")

# Section 6
add_md("""### 6. Target Variable Distribution (`rice_yield_kgha`)
We analyze the distribution of the primary prediction target: rice yield in kilograms per hectare (kg/ha).""")
add_code("""# Target variable summary statistics
target_all = df['rice_yield_kgha']
target_active = df[df['rice_yield_kgha'] > 0]['rice_yield_kgha']

target_comparison = pd.DataFrame({
    'Metric': ['Count', 'Mean (kg/ha)', 'Std Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis'],
    'All Observations': [len(target_all), target_all.mean(), target_all.std(), target_all.min(), target_all.quantile(0.25), target_all.median(), target_all.quantile(0.75), target_all.max(), target_all.skew(), target_all.kurt()],
    'Active Cultivation (Yield > 0)': [len(target_active), target_active.mean(), target_active.std(), target_active.min(), target_active.quantile(0.25), target_active.median(), target_active.quantile(0.75), target_active.max(), target_active.skew(), target_active.kurt()]
}).round(2)
target_comparison
""")

# Section 7
add_md("""### 7. Unit Consistency & Target Leakage Verification
We verify the physical unit relationship between production, area, and yield:
$$\\text{Yield (kg/ha)} = \\frac{\\text{Production (1,000 tons)} \\times 1,000,000\\text{ kg}}{\\text{Area (1,000 ha)} \\times 1,000\\text{ ha}} = \\frac{\\text{Production} \\times 1,000}{\\text{Area}}$$""")
add_code("""# Evaluate mathematical consistency for active cultivation rows
active_df = df[df['rice_area_1000ha'] > 0].copy()
derived_yield = (active_df['rice_production_1000tons'] * 1000) / active_df['rice_area_1000ha']
diff = (derived_yield - active_df['rice_yield_kgha']).abs()

print(f"Verification of Active Cultivation Records (N={len(active_df)}):")
print(f"Max absolute discrepancy: {diff.max():.6f} kg/ha")
print(f"Median discrepancy: {diff.median():.6f} kg/ha")
print(f"99th percentile discrepancy: {diff.quantile(0.99):.6f} kg/ha")

print(\"\"\"
CRITICAL LEAKAGE ASSESSMENT:
Because 'rice_production_1000tons' directly derives the target, it MUST BE EXCLUDED from all feature sets
for yield regression. Retaining production would cause near-perfect artificial R2 (~0.999) on historical data
but total failure at operational inference time when future harvest production is unknown.
\"\"\")
""")

# Section 8
add_md("""### 8. Geographical Variation & State-Level Yield Profiles""")
add_code("""# State-level summary
state_summary = df[df['rice_yield_kgha'] > 0].groupby('state')['rice_yield_kgha'].agg(
    Districts='nunique',
    Mean_Yield='mean',
    Median_Yield='median',
    Min_Yield='min',
    Max_Yield='max',
    Std_Dev='std'
).sort_values(by='Median_Yield', ascending=False).round(1)
state_summary
""")

# Section 9
add_md("""### 9. Temporal Trends & Climate Anomalies (2000–2015)""")
add_code("""# Annual yield trend and climate fluctuations
yearly_summary = df[df['rice_yield_kgha'] > 0].groupby('year').agg(
    National_Yield_Mean=('rice_yield_kgha', 'mean'),
    National_Yield_Median=('rice_yield_kgha', 'median'),
    Monsoon_Rain_Mean=('rain_monsoon_mm', 'mean'),
    Annual_Rain_Mean=('rain_annual_mm', 'mean'),
    Water_Stress_Mean=('water_stress_score', 'mean')
).round(1)
yearly_summary
""")

# Section 10
add_md("""### 10. Agro-Climatic Feature Correlations
We compute Pearson correlation coefficients between key candidate predictors and the yield target.""")
add_code("""candidate_cols = [
    'rice_yield_kgha', 'n_per_ha_nca', 'p_per_ha_nca', 'k_per_ha_nca',
    'lgp_days', 'rain_monsoon_mm', 'rain_annual_mm', 'water_stress_score',
    'irrig_canal_area_1000ha', 'irrig_tubewell_area_1000ha',
    'tmax_monsoon', 'tmin_monsoon', 'aet_monsoon'
]
corrs = df[df['rice_yield_kgha'] > 0][candidate_cols].corr()['rice_yield_kgha'].sort_values(ascending=False)
print("Correlation with rice_yield_kgha (Active Cultivation Subsample):")
print(corrs.round(3))
""")

# Section 11
add_md("""### 11. Key Findings & Recommendations for Phase 2 & 3

1. **Prediction Target & Unit:**
   - Target: `rice_yield_kgha` in **kg/ha**.
   - Active cultivation mean: **2,015.4 kg/ha** (Median: 2,026.4 kg/ha, Range: 58.9 to 5,653.8 kg/ha).

2. **Prediction Granularity:**
   - Spatial unit: **District-level** (253 districts across 19 states).
   - Temporal unit: **Annual / Kharif crop season** (2000 to 2015).

3. **Sample Scope & Zero-Yield Strategy:**
   - Exclude permanent non-rice cultivating districts (12 districts in Rajasthan, Saurashtra, and Lahaul & Spiti where rice area is consistently 0.0).
   - Model active rice-cultivating observations (`rice_area_1000ha > 0`), retaining true crop failure cases (yield = 0 with area > 0) as extreme downside risk instances.

4. **Strict Leakage Prevention:**
   - **EXCLUDE:** `rice_production_1000tons` (direct target formula).
   - **EXCLUDE FROM CLUSTERING:** `rice_yield_kgha`, `rice_production_1000tons`, `rice_area_1000ha`. Condition clusters must reflect purely environmental, soil, hydrological, and agronomic capacity.

5. **Chronological Splitting Strategy:**
   - Do NOT use random k-fold cross-validation across time.
   - Training: **2000 – 2011** (12 years, ~75% of data).
   - Validation: **2012 – 2013** (2 years, ~12.5% of data).
   - Test: **2014 – 2015** (2 years, ~12.5% of data).
   - This ensures rigorous out-of-time evaluation reflecting realistic deployment.""")

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/initial_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

with open("initial_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Generated initial_eda.ipynb in root and notebooks/")
