# Phase 1 Data Audit Report: RiPred (Rice Intelligence & Yield Prediction Engine)

**Project:** RiPred — Indian Rice Yield Prediction & Crop-Condition Intelligence  
**Dataset:** Indian Rice Yield and Climate Dataset (2000–2015)  
**Audit Date:** November 2026  
**Author:** RiPred Engineering & Data Science Team  
**Status:** Phase 1 Complete — Awaiting Approval for Phase 2  

---

## Executive Summary

This report documents the rigorous Phase 1 audit of the primary dataset for Project RiPred. The objective of RiPred is to deliver an enterprise-grade, modular machine-learning engine combining supervised yield regression, unsupervised agro-ecological condition discovery, hybrid architecture benchmarking, and yield-risk classification for Indian rice cultivation.

In strict compliance with project governance:
- **No models have been trained.**
- The raw dataset was inspected directly from `data/raw/rice.csv`.
- Complete data dictionaries, missing-value analyses, analytical figures, and an end-to-end EDA notebook (`initial_eda.ipynb`) have been generated.
- A critical mathematical leakage vector was discovered and quarantined.
- The precise observation granularity and modeling subpopulation have been mathematically and agronomically justified.

---

## 1. Dataset Dimensionality and Column Taxonomy

The dataset contains **3,744 observations** across **69 features**.

```
Dimensions: 3,744 rows × 69 columns
Memory Usage: ~2.0 MB
Primary Data Types:
  - float64: 66 columns
  - int64:    1 column  ('year')
  - object:   2 columns ('state', 'district')
```

### Functional Domain Categorization

| Domain Category | Column Count | Representative Features |
|---|---|---|
| **Temporal** | 1 | `year` (2000–2015) |
| **Geographical** | 2 | `state` (19 states), `district` (253 districts) |
| **Production & Target** | 3 | `rice_area_1000ha`, `rice_production_1000tons`, `rice_yield_kgha` |
| **Irrigation Infrastructure** | 4 | `irrig_canal_area_1000ha`, `irrig_tank_area_1000ha`, `irrig_tubewell_area_1000ha`, `irrig_other_area_1000ha` |
| **Nutrient & Fertilizer** | 9 | `n_per_ha_nca`, `p_per_ha_nca`, `k_per_ha_nca`, `n_kharif_tons`, `n_rabi_tons`, `p_kharif_tons`, `p_rabi_tons`, `k_kharif_tons`, `k_rabi_tons` |
| **Agro-climatic Phenology** | 1 | `lgp_days` (Length of Growing Period) |
| **Soil Taxonomy Orders** | 10 | `soil_loamy_alfisols`, `soil_vertisols`, `soil_inceptisols`, `soil_orthids`, etc. |
| **Seasonal Precipitation** | 10 | `rain_annual_mm`, `rain_pre_monsoon_mm`, `rain_monsoon_mm`, `rain_post_monsoon_mm`, `rain_winter_mm`, and corresponding deviations (`rain_diff_*`) |
| **Seasonal Thermal Stress** | 8 | `tmax_winter`, `tmax_pre_monsoon`, `tmax_monsoon`, `tmax_post_monsoon`, `tmin_winter`, `tmin_pre_monsoon`, `tmin_monsoon`, `tmin_post_monsoon` |
| **Monthly Water Deficit** | 11 | `wd_jan_mm` through `wd_dec_mm` (omitting September) |
| **Evapotranspiration (AET)** | 8 | `aet_winter`, `aet_pre_monsoon`, `aet_monsoon`, `aet_post_monsoon`, and deviations (`aet_diff_*`) |
| **Hydrological Stress Indices** | 2 | `water_stress_score`, `stress_group` |

---

## 2. Integrity, Duplicates, and Key Uniqueness

1. **Exact Row Duplicates:** **0 rows**.
2. **Composite Key Uniqueness:** Every observation represents a unique `(year, state, district)` combination (**0 duplicates** across composite key).
3. **District Count:** 253 unique districts across 19 states.
4. **State Hierarchy:** No district belongs to multiple states; administrative hierarchy is 100% clean.
5. **Text Standardization:** All state and district names are lowercase strings without leading/trailing whitespace, punctuation errors, or invalid characters.

---

## 3. Missing Value Analysis (Explicit vs. Implicit)

### Explicit Missing Values
- **Total `NaN` / `null` values:** **0 across all 3,744 rows and 69 columns** (0.00%).

### Implicit Missingness and Omitted Variables
1. **Omitted September Water Deficit (`wd_sep_mm`):**
   - The monthly water deficit features cover 11 months: `jan, feb, mar, apr, may, jun, jul, aug, oct, nov, dec`.
   - `wd_sep_mm` (September) is **absent** from the dataset schema.
   - *Impact:* September coincides with the grain-filling stage of Kharif rice. Water stress in this critical period must be inferred through `rain_monsoon_mm`, `aet_monsoon`, and the composite `water_stress_score`.
2. **Soil Composition Sums:**
   - The 10 soil percentage columns sum to 100.0% in 3,608 observations (96.37%).
   - In 136 observations, the sum is < 100.0% (mean = 77.4%, min = 0.0%). These represent districts where land area includes unclassified soils, rocky terrain, urban settlements, or water bodies.
3. **Temporal Incompleteness (District-Year Unbalance):**
   - A fully balanced panel would yield $253 \times 16 = 4,048$ observations. The dataset contains 3,744 observations (92.5% completeness).
   - District reporting varies by year:
     - 2000: 243 districts
     - 2001: 193 districts (gap in reporting)
     - 2002: 232 districts
     - 2003: 197 districts (gap in reporting)
     - 2004–2009: 244–252 districts
     - 2010–2011: 184 districts (significant drop in recorded districts)
     - 2012–2015: 252–253 districts (near complete coverage)

---

## 4. Zero-Yield and Non-Cultivation Boundary Analysis

A critical finding of the audit is the presence of **344 zero-yield observations** (`rice_yield_kgha == 0.0`, representing 9.19% of the dataset).

### Decomposition of Zero-Yield Cases

$$\text{Zero Yield Observations} = 344$$
- **Uncultivated District-Years (Area = 0, Production = 0):** **336 cases (97.7%)**
- **Complete Crop Failure (Area > 0, Production = 0):** **8 cases (2.3%)**

### Analysis of Always-Zero Districts
There are **12 districts** where rice cultivation is recorded as **0.0 across all 16 years**:
- **Rajasthan (8 districts):** Barmer, Churu, Jaisalmer, Jalore, Jodhpur, Nagaur, Pali, Sikar (Arid Thar Desert biome; rice is not an agro-ecological crop here).
- **Gujarat (3 districts):** Amreli, Jamnagar, Rajkot (Arid/semi-arid Saurashtra region).
- **Himachal Pradesh (1 district):** Lahul & Spiti (High-altitude trans-Himalayan cold desert).

In addition, **24 districts** cultivate rice intermittently (e.g. Jaipur, Sirohi, Indore, Mandsaur, Solapur), where rice is sown only during high-rainfall monsoon years and left uncultivated during dry cycles.

---

## 5. Target Distribution Analysis (`rice_yield_kgha`)

The target variable is **`rice_yield_kgha`**, measured in **kilograms per hectare (kg/ha)**.

| Metric | All Observations ($N=3,744$) | Active Cultivation ($N=3,400$, Yield > 0) |
|---|---|---|
| **Mean** | 1,830.25 kg/ha | **2,015.43 kg/ha** |
| **Standard Deviation** | 1,104.15 kg/ha | **984.48 kg/ha** |
| **Minimum** | 0.00 kg/ha | **58.88 kg/ha** |
| **25th Percentile ($Q_1$)** | 1,000.00 kg/ha | **1,257.42 kg/ha** |
| **Median ($Q_2$)** | 1,904.67 kg/ha | **2,026.38 kg/ha** |
| **75th Percentile ($Q_3$)** | 2,559.92 kg/ha | **2,642.82 kg/ha** |
| **Maximum** | 5,653.83 kg/ha | **5,653.83 kg/ha** |
| **Interquartile Range (IQR)**| 1,559.92 kg/ha | **1,385.40 kg/ha** |
| **Skewness** | 0.15 | **0.31** |
| **Kurtosis** | -0.50 | **-0.31** |

### Key Observations on Distribution:
1. When zero-yield uncultivated districts are isolated, the active yield distribution is remarkably well-behaved: nearly zero skewness (+0.31) and slight platykurtosis (-0.31).
2. The yield exhibits substantial spatial variance across India, ranging from low rainfed levels (~500 kg/ha in dry upland districts) to over 5,000 kg/ha in intensive irrigated belts of Punjab, Haryana, and Tamil Nadu.

---

## 6. Unit Consistency & Leakage Risk Assessment

### Unit Mathematical Consistency
The dataset includes both production totals and harvested areas:
- `rice_area_1000ha`: Cultivated area in thousands of hectares ($10^3 \text{ ha}$).
- `rice_production_1000tons`: Harvested production in thousands of metric tons ($10^6 \text{ kg}$).
- `rice_yield_kgha`: Target in kilograms per hectare ($\text{kg/ha}$).

Testing the identity:
$$\text{Calculated Yield} = \frac{\text{rice\_production\_1000tons} \times 1,000}{\text{rice\_area\_1000ha}}$$

- **Median absolute discrepancy:** **0.0027 kg/ha**
- **99th percentile discrepancy:** **0.701 kg/ha**
- **Conclusion:** The identity holds with mathematical exactness across all non-zero records. The tiny residual discrepancies are standard round-off artifacts in government statistical reporting tables.

### Severe Leakage Vector: `rice_production_1000tons`
- **Assessment:** **CRITICAL TARGET LEAKAGE.**
- **Finding:** If `rice_production_1000tons` is retained as an explanatory feature, any linear or tree-based model can reconstruct the target almost perfectly ($R^2 \approx 0.999$), rendering all climate, soil, and management signals irrelevant. In operational deployment at sowing or mid-season, total production is an unknown future outcome.
- **Mandate:** `rice_production_1000tons` **must be completely quarantined and excluded** from all feature engineering, regression, classification, and clustering pipelines.

---

## 7. Agro-Climatic Correlation & Feature Dynamics

Correlation analysis between candidate predictors and active rice yield reveals clear agronomic signals:

### Top Positive Correlations with Rice Yield
1. **Nutrient Application Rates:**
   - `n_per_ha_nca` ($r = +0.590$): Nitrogen application rate per hectare of net cropped area is the single strongest linear predictor of yield.
   - `p_per_ha_nca` ($r = +0.506$): Phosphate application rate.
   - `k_per_ha_nca` ($r = +0.373$): Potash application rate.
2. **Irrigation Infrastructure:**
   - `irrig_tubewell_area_1000ha` ($r = +0.312$): Tubewell irrigation protects against monsoon failure.
   - `irrig_canal_area_1000ha` ($r = +0.185$): Canal command networks.
3. **Hydrological Buffer:**
   - `aet_winter` ($r = +0.328$): Winter actual evapotranspiration indicates year-round surface water availability and double-cropping capacity.

### Top Negative Correlations with Rice Yield
1. **Soil Constraints:**
   - `soil_vertic_soils` ($r = -0.386$): Heavy clay/swelling vertic soils in semi-arid rainfed tracts (Deccan plateau) correlate with waterlogging or drought cracking without irrigation.
2. **Moisture Deficits:**
   - `aet_diff_post_monsoon` ($r = -0.356$): Deficit during crop maturation.
   - `wd_dec_mm` ($r = -0.345$), `wd_nov_mm` ($r = -0.325$): High post-monsoon water deficits indicate dry end-of-season conditions.
   - `stress_group` ($r = -0.272$) and `water_stress_score` ($r = -0.231$): High water stress directly suppresses yield.

---

## 8. Outlier & Physical Plausibility Assessment

Outliers were identified using the Interquartile Range method ($1.5 \times \text{IQR}$):

1. **Monsoon Deficits (`wd_jul_mm`, `wd_aug_mm`):** ~19% IQR outliers.
   - *Physical Verification:* In normal monsoon tracts, July and August have 0 mm water deficit. Non-zero deficits occur only in rain-shadow districts (interior Maharashtra, Saurashtra, Western Rajasthan). These represent genuine climatic droughts, not measurement errors.
2. **Rainfall Deviations (`rain_diff_*`):** 10–13% IQR outliers.
   - *Physical Verification:* India experiences high monsoon volatility. Historic drought years (2002, 2004, 2009, 2014, 2015) produce large negative departures (-500 mm to -2000 mm), while cyclonic events create massive positive spikes. These must be preserved as authentic climate signals.
3. **Sub-Zero Temperatures in Winter:**
   - Minimum temperatures reach -27.4°C in `tmin_winter` (Lahul & Spiti, Himachal Pradesh).
   - *Physical Verification:* High-altitude Himalayan district. The values are authentic climatological records.

---

## 9. Recommendations for Subsequent Phases

### Recommendation 1: Observation Granularity & Target Definition
- **Prediction Target:** `rice_yield_kgha` (Unit: **kg/ha**).
- **Target Scale:** Continuous numeric regression.
- **Granularity:** District-Crop Year level (`(year, state, district)`).

### Recommendation 2: Target Subpopulation Scope
- **Rule:** Filter out the 12 permanent non-rice cultivating districts where rice area is consistently 0.0 across all 16 years.
- **Modeling Universe:** Focus on active rice cultivation observations ($N = 3,400$), while preserving true crop failures ($N = 8$, where area > 0 but yield = 0).
- **Alternative Two-Stage Production Engine:** In the full product deployment, implement a Stage 1 binary gating classifier ("Is rice cultivated in this district-year?") followed by Stage 2 conditional yield regression on active land.

### Recommendation 3: Feature Space Separation for Unsupervised Clustering (Phase 3)
In strict accordance with project requirements, condition discovery must use **only environmental, soil, hydrological, and agronomic capacity features**:
- **Excluded:** `rice_yield_kgha`, `rice_production_1000tons`, `rice_area_1000ha`.
- **Clustering Inputs:**
  - 10 Soil taxonomy percentages (`soil_*`).
  - 10 Rainfall totals and anomalies (`rain_*`).
  - 8 Temperature features (`tmax_*`, `tmin_*`).
  - 11 Monthly water deficits (`wd_*`).
  - 8 Actual evapotranspiration variables (`aet_*`).
  - Length of Growing Period (`lgp_days`).
  - Water stress index (`water_stress_score`).
  - Irrigation source allocations (`irrig_*`).
  - Base fertilizer consumption rates (`n_per_ha_nca`, `p_per_ha_nca`, `k_per_ha_nca`).

### Recommendation 4: Chronological Splitting Protocol (Phase 4–6)
To prevent temporal data leakage and realistically mirror operational agricultural forecasting:
- **Never use random row splitting or randomized k-fold CV.**
- **Training Set:** Crop Years **2000 – 2011** (12 years, 2,525 active observations, ~74.3%).
- **Validation Set:** Crop Years **2012 – 2013** (2 years, 442 active observations, ~13.0%).
- **Test Set:** Crop Years **2014 – 2015** (2 years, 433 active observations, ~12.7%).
- *Agronomic Note:* 2014 and 2015 were consecutive severe El Niño drought years in India, providing the ultimate stress-test for model generalization on out-of-sample climate shocks.

---

## 10. Audit Artifact Deliverables

| Deliverable File | Description | Location |
|---|---|---|
| `data_audit_report.md` | Comprehensive technical audit report | Project Root & `reports/` |
| `data_dictionary.csv` | Full data dictionary for all 69 columns with units, ranges, and leakage classifications | Project Root & `reports/` |
| `missing_values_report.csv` | Granular missing value and IQR outlier statistics for all features | Project Root & `reports/` |
| `initial_eda.ipynb` | Executable Jupyter notebook with complete data exploration workflows | Project Root & `notebooks/` |
| Analytical Visualizations | 6 high-resolution charts depicting distributions, correlations, and trends | `reports/figures/` |

---
*Phase 1 Audit complete. Standing by for user review and approval before starting Phase 2.*
