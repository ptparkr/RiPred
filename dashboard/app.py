"""RiPred — Rice Intelligence & Yield Prediction Engine
Interactive Multi-Page Streamlit Dashboard.
Pages:
1. Project Overview
2. Rice Yield Prediction
3. Crop Condition Analysis
4. Cluster Visualization
5. Yield Risk Analysis
6. Historical Yield Trends
7. Model Performance
8. Feature Importance
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="RiPred — Rice Intelligence & Yield Prediction Engine",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1B5E20; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #424242; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F1F8E9; border-radius: 8px; padding: 15px; border-left: 5px solid #2E7D32; margin-bottom: 10px; }
    .metric-title { font-size: 0.9rem; color: #558B2F; font-weight: 600; text-transform: uppercase; }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #1B5E20; }
    .metric-desc { font-size: 0.8rem; color: #616161; }
    .risk-low { background-color: #FFEBEE; border-left: 5px solid #C62828; color: #B71C1C; padding: 10px; border-radius: 6px; }
    .risk-normal { background-color: #FFF8E1; border-left: 5px solid #F57F17; color: #E65100; padding: 10px; border-radius: 6px; }
    .risk-high { background-color: #E8F5E9; border-left: 5px solid #2E7D32; color: #1B5E20; padding: 10px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    data_dir = os.path.join(base_dir, "data", "raw")
    
    art = {}
    art['preprocessor'] = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    art['regressor'] = joblib.load(os.path.join(models_dir, "best_yield_regressor.joblib"))
    art['classifier'] = joblib.load(os.path.join(models_dir, "best_risk_classifier.joblib"))
    art['clusterer'] = joblib.load(os.path.join(models_dir, "crop_condition_clusterer.joblib"))
    art['conformal'] = joblib.load(os.path.join(models_dir, "conformal_predictor.joblib"))
    
    with open(os.path.join(models_dir, "feature_schema.json"), "r") as f:
        art['schema'] = json.load(f)
    with open(os.path.join(models_dir, "cluster_profiles.json"), "r") as f:
        art['cluster_profiles'] = json.load(f)
    with open(os.path.join(models_dir, "regression_metrics.json"), "r") as f:
        art['reg_metrics'] = json.load(f)
    with open(os.path.join(models_dir, "classification_metrics.json"), "r") as f:
        art['clf_metrics'] = json.load(f)
    with open(os.path.join(models_dir, "hybrid_comparison_report.json"), "r") as f:
        art['hybrid_report'] = json.load(f)
    with open(os.path.join(models_dir, "conformal_evaluation.json"), "r") as f:
        art['conformal_eval'] = json.load(f)
        
    raw_csv_path = os.path.join(data_dir, "rice.csv")
    if os.path.exists(raw_csv_path):
        art['df_raw'] = pd.read_csv(raw_csv_path)
    else:
        art['df_raw'] = None
        
    perm_csv = os.path.join(reports_dir, "permutation_importance.csv")
    if os.path.exists(perm_csv):
        art['perm_df'] = pd.read_csv(perm_csv)
    else:
        art['perm_df'] = None
        
    drift_csv = os.path.join(reports_dir, "feature_drift_report.csv")
    if os.path.exists(drift_csv):
        art['drift_df'] = pd.read_csv(drift_csv)
    else:
        art['drift_df'] = None
        
    return art


artifacts = load_all_artifacts()

# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1536657464919-892534f60d6e?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.title("🌾 RiPred Navigation")
page = st.sidebar.radio(
    "Select Intelligence Module:",
    [
        "1. Project Overview",
        "2. Rice Yield Prediction",
        "3. Crop Condition Analysis",
        "4. Cluster Visualization",
        "5. Yield Risk Analysis",
        "6. Historical Yield Trends",
        "7. Model Performance",
        "8. Feature Importance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Engine Specs:**
- **Version:** RiPred v1.0 Production
- **Observation:** District-Crop Year
- **Target Unit:** kg/ha
- **Split Protocol:** Chronological Out-of-Time
""")

# ==============================================================================
# PAGE 1: PROJECT OVERVIEW
# ==============================================================================
if page == "1. Project Overview":
    st.markdown('<div class="main-header">🌾 RiPred: Rice Intelligence & Yield Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Indian Rice Yield Regression, Agro-Ecological Clustering, and Risk Analytics System</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Observations</div>
            <div class="metric-val">3,744</div>
            <div class="metric-desc">253 Districts across 19 States</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Features</div>
            <div class="metric-val">69</div>
            <div class="metric-desc">Climate, Hydrology, Soil, NPK</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best Regressor</div>
            <div class="metric-val">420.8</div>
            <div class="metric-desc">Val MAE (kg/ha) — Gradient Boosting</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Reliability</div>
            <div class="metric-val">87.3%</div>
            <div class="metric-desc">Empirical Test Conformal Coverage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏛️ Product Architecture & Core Methodologies")
    
    st.markdown("""
    RiPred is engineered as a production-grade machine-learning system designed specifically for Indian rice agriculture. It addresses core forecasting challenges through five foundational pillars:

    1. **Strict Leakage Prevention:** Eliminates artificial target derivations (`rice_production_1000tons`), ensuring models only use information available before harvest.
    2. **Unsupervised Condition Discovery:** Identifies 4 distinct agro-climatic clusters representing water stress, humid monsoon tracts, deep clay vertisols, and high-input irrigated plains.
    3. **Hybrid Architecture Decision:** Empirically tests whether latent cluster labels boost predictive accuracy. Following empirical testing showing negligible delta, RiPred retains clustering as an independent diagnostic component rather than forcing it into the prediction model.
    4. **Yield-Risk Categorization:** Classifies harvests into **Low (< 1,386 kg/ha)**, **Normal (1,386 - 2,301 kg/ha)**, and **High (> 2,301 kg/ha)** tiers using training-derived percentiles with full class probabilities.
    5. **Conformal Uncertainty Quantification:** Supplies calibrated 90% finite-sample confidence intervals ($[\\hat{y} - 863.9, \\hat{y} + 863.9]$) that maintain empirical coverage even during severe out-of-sample El Niño droughts.
    """)

    st.markdown("---")
    st.markdown("### 🗺️ Geographical & Temporal Coverage")
    st.write("Spanning **2000 to 2015**, RiPred models all major rice agro-ecosystems across 19 Indian states:")
    if artifacts['df_raw'] is not None:
        state_counts = artifacts['df_raw']['state'].value_counts()
        fig, ax = plt.subplots(figsize=(12, 4))
        state_counts.plot(kind='bar', color='#2E7D32', ax=ax)
        ax.set_ylabel("Records")
        ax.set_title("Records per State in Source Panel Dataset")
        st.pyplot(fig)

# ==============================================================================
# PAGE 2: RICE YIELD PREDICTION
# ==============================================================================
elif page == "2. Rice Yield Prediction":
    st.markdown('<div class="main-header">🌾 Interactive Rice Yield Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input district agricultural indicators to generate predicted yield with calibrated uncertainty bounds.</div>', unsafe_allow_html=True)

    with st.form("yield_form"):
        st.markdown("#### 📍 Location & Land Profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            state_in = st.selectbox("State", sorted(artifacts['schema']['categorical_cols'] and [
                'punjab', 'haryana', 'tamil nadu', 'andhra pradesh', 'west bengal',
                'uttar pradesh', 'bihar', 'assam', 'karnataka', 'kerala', 'maharashtra',
                'madhya pradesh', 'rajasthan', 'gujarat', 'telangana', 'chhattisgarh', 'jharkhand', 'uttarakhand', 'himachal pradesh'
            ]))
        with c2:
            district_in = st.text_input("District Name", "ludhiana")
        with c3:
            year_in = st.number_input("Crop Year", min_value=2000, max_value=2030, value=2015)
        with c4:
            area_in = st.number_input("Rice Cultivated Area (1000 ha)", min_value=0.1, max_value=2000.0, value=250.0)

        st.markdown("#### 🧪 Fertilizer Application & Irrigation Capacity")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_in = st.slider("Nitrogen (N) kg/ha NCA", 0.0, 350.0, 160.0)
        with c2:
            p_in = st.slider("Phosphorus (P) kg/ha NCA", 0.0, 200.0, 45.0)
        with c3:
            k_in = st.slider("Potassium (K) kg/ha NCA", 0.0, 150.0, 25.0)
        with c4:
            lgp_in = st.slider("Length of Growing Period (Days)", 30.0, 300.0, 180.0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tubewell_in = st.number_input("Tubewell Area (1000 ha)", 0.0, 500.0, 180.0)
        with c2:
            canal_in = st.number_input("Canal Area (1000 ha)", 0.0, 500.0, 40.0)
        with c3:
            tank_in = st.number_input("Tank/Pond Area (1000 ha)", 0.0, 100.0, 0.0)
        with c4:
            other_irrig_in = st.number_input("Other Irrigation (1000 ha)", 0.0, 100.0, 2.0)

        st.markdown("#### 🌦️ Seasonal Climate & Water Stress")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rain_monsoon_in = st.number_input("Monsoon Rainfall (mm)", 0.0, 3500.0, 550.0)
        with c2:
            rain_annual_in = st.number_input("Annual Rainfall (mm)", 0.0, 5000.0, 750.0)
        with c3:
            tmax_monsoon_in = st.slider("Monsoon Max Temp (°C)", 15.0, 42.0, 34.0)
        with c4:
            stress_score_in = st.slider("Water Stress Score", 0.0, 250.0, 110.0)

        submit_btn = st.form_submit_button("🚀 Compute Prediction & Risk Analysis")

    if submit_btn:
        # Build dataframe matching schema
        num_cols = artifacts['schema']['numerical_cols']
        input_dict = {col: 0.0 for col in num_cols}
        
        input_dict['rice_area_1000ha'] = area_in
        input_dict['n_per_ha_nca'] = n_in
        input_dict['p_per_ha_nca'] = p_in
        input_dict['k_per_ha_nca'] = k_in
        input_dict['lgp_days'] = lgp_in
        input_dict['irrig_tubewell_area_1000ha'] = tubewell_in
        input_dict['irrig_canal_area_1000ha'] = canal_in
        input_dict['irrig_tank_area_1000ha'] = tank_in
        input_dict['irrig_other_area_1000ha'] = other_irrig_in
        input_dict['rain_monsoon_mm'] = rain_monsoon_in
        input_dict['rain_annual_mm'] = rain_annual_in
        input_dict['tmax_monsoon'] = tmax_monsoon_in
        input_dict['water_stress_score'] = stress_score_in
        input_dict['state'] = state_in
        input_dict['district'] = district_in
        input_dict['year'] = year_in

        df_input = pd.DataFrame([input_dict])
        
        # Transform through fitted preprocessor
        X_trans = artifacts['preprocessor'].transform(df_input)
        
        # Conformal prediction
        pred_yield, lower_b, upper_b = artifacts['conformal'].predict_intervals(X_trans)
        point_val = float(pred_yield[0])
        low_val = float(lower_b[0])
        up_val = float(upper_b[0])
        margin = artifacts['conformal'].quantile_residual_
        
        # Risk classification
        risk_details = artifacts['classifier'].predict_with_details(X_trans)[0]
        risk_tier = risk_details["predicted_risk"]
        
        # Cluster assignment
        cluster_id = int(artifacts['clusterer'].predict_cluster(df_input)[0])
        cluster_name = artifacts['clusterer'].cluster_names_.get(cluster_id, f"Cluster {cluster_id}")
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results & Reliability Assessment")
        
        r1, r2, r3 = st.columns([1.5, 1.2, 1.3])
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Rice Yield</div>
                <div class="metric-val">{point_val:,.1f} kg/ha</div>
                <div class="metric-desc">Model: Gradient Boosting Regressor (v1.0)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**90% Calibrated Conformal Interval:**")
            st.info(f"**[{low_val:,.1f} kg/ha  —  {up_val:,.1f} kg/ha]** (Margin: ±{margin:.1f} kg/ha)")
            
        with r2:
            badge_class = "risk-high" if risk_tier == "High" else ("risk-normal" if risk_tier == "Normal" else "risk-low")
            st.markdown(f"""
            <div class="{badge_class}">
                <h4 style="margin:0;">Risk Tier: {risk_tier} Yield</h4>
                <p style="margin:5px 0 0 0; font-size:0.85rem;">Confidence: {risk_details['confidence']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Class Probability Distribution:**")
            prob_df = pd.DataFrame(list(risk_details["class_probabilities"].items()), columns=["Tier", "Probability"])
            st.dataframe(prob_df, hide_index=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Crop-Condition Archetype</div>
                <div style="font-size:1.1rem; font-weight:700; color:#1B5E20; margin-top:5px;">{cluster_name}</div>
                <div class="metric-desc" style="margin-top:5px;">Assigned via Unsupervised Clustering</div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: CROP CONDITION ANALYSIS
# ==============================================================================
elif page == "3. Crop Condition Analysis":
    st.markdown('<div class="main-header">🌾 Crop Condition & Agro-Climatic Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Profiles of 4 latent crop-condition archetypes discovered through unsupervised learning.</div>', unsafe_allow_html=True)

    profiles = artifacts['cluster_profiles']['profiles']
    
    st.markdown("### 📋 Empirical Cluster Archetypes")
    
    for cid, prof in profiles.items():
        with st.expander(f"📍 {prof['name']} ({prof['percentage']}% of active area)", expanded=(cid in ['0', '2'])):
            c1, c2 = st.columns([1.5, 2])
            with c1:
                st.write(f"**Cluster Size:** {prof['count']} observations")
                st.write(f"**Top Represented States:** {', '.join([f'{k.title()} ({v})' for k, v in prof['top_states'].items()])}")
                if 'yield_actual_mean' in prof['stats']:
                    st.metric("Historical Mean Yield", f"{prof['stats']['yield_actual_mean']} kg/ha")
            with c2:
                stat_items = []
                for feat, s in prof['stats'].items():
                    if feat not in ['yield_actual_mean', 'yield_actual_median']:
                        stat_items.append({"Indicator": feat, "Mean": s['mean'], "Median": s['median'], "IQR Range": f"[{s['p25']}, {s['p75']}]"})
                st.dataframe(pd.DataFrame(stat_items), height=220, hide_index=True)

# ==============================================================================
# PAGE 4: CLUSTER VISUALIZATION
# ==============================================================================
elif page == "4. Cluster Visualization":
    st.markdown('<div class="main-header">🌾 Cluster Visualization & PCA Projection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dimensionality reduction reveals the spatial separation of agro-climatic clusters across India.</div>', unsafe_allow_html=True)

    var_ratios = artifacts['cluster_profiles']['pca_explained_variance_ratio']
    st.write(f"**Principal Components Explained Variance:** PC1: {var_ratios[0]*100:.1f}%, PC2: {var_ratios[1]*100:.1f}%, PC3: {var_ratios[2]*100:.1f}% (Total: {sum(var_ratios)*100:.1f}%)")

    if artifacts['df_raw'] is not None:
        cond_cols = [c for c in artifacts['clusterer'].feature_names_ if c in artifacts['df_raw'].columns]
        df_sample = artifacts['df_raw'][artifacts['df_raw']['rice_area_1000ha'] > 0].copy()
        
        # PCA projection
        pca_coords = artifacts['clusterer'].transform_pca(df_sample)
        cluster_labels = artifacts['clusterer'].predict_cluster(df_sample)
        
        df_plot = pd.DataFrame({
            'PC1': pca_coords[:, 0],
            'PC2': pca_coords[:, 1],
            'Cluster': [artifacts['clusterer'].cluster_names_.get(c, f"Cluster {c}") for c in cluster_labels],
            'Yield': df_sample['rice_yield_kgha']
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='tab10', alpha=0.7, ax=ax)
        ax.set_title("2D PCA Projection of Crop-Condition Clusters", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"PC1 ({var_ratios[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_ratios[1]*100:.1f}% variance)")
        st.pyplot(fig)

# ==============================================================================
# PAGE 5: YIELD RISK ANALYSIS
# ==============================================================================
elif page == "5. Yield Risk Analysis":
    st.markdown('<div class="main-header">🌾 Yield Risk Categorization Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating classification performance across Low, Normal, and High harvest tiers.</div>', unsafe_allow_html=True)

    thresh = artifacts['clf_metrics']['thresholds_kg_ha']
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="risk-low">
            <h4 style="margin:0;">🚨 Low Yield Tier</h4>
            <div style="font-size:1.5rem; font-weight:700;">&lt; {thresh['low_threshold']} kg/ha</div>
            <p style="margin:5px 0 0 0; font-size:0.8rem;">Bottom tercile of training yield distribution</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="risk-normal">
            <h4 style="margin:0;">⚖️ Normal Yield Tier</h4>
            <div style="font-size:1.5rem; font-weight:700;">{thresh['low_threshold']} - {thresh['high_threshold']} kg/ha</div>
            <p style="margin:5px 0 0 0; font-size:0.8rem;">Middle tercile baseline band</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="risk-high">
            <h4 style="margin:0;">🌟 High Yield Tier</h4>
            <div style="font-size:1.5rem; font-weight:700;">&gt; {thresh['high_threshold']} kg/ha</div>
            <p style="margin:5px 0 0 0; font-size:0.8rem;">Top tercile high-productivity harvest</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Classifier Benchmark Leaderboard")
    
    clf_res = artifacts['clf_metrics']['results']
    table_data = []
    for m_name, m_data in clf_res.items():
        table_data.append({
            "Classifier": m_name.replace("_", " ").title(),
            "Val Precision (w)": m_data['validation']['precision_weighted'],
            "Val Recall (w)": m_data['validation']['recall_weighted'],
            "Val F1 (w)": m_data['validation']['f1_weighted'],
            "Val ROC-AUC": m_data['validation']['roc_auc_ovr_weighted'],
            "Test F1 (w)": m_data['test']['f1_weighted']
        })
    st.dataframe(pd.DataFrame(table_data), hide_index=True)

# ==============================================================================
# PAGE 6: HISTORICAL YIELD TRENDS
# ==============================================================================
elif page == "6. Historical Yield Trends":
    st.markdown('<div class="main-header">🌾 Historical Yield Trends & Climate Shocks (2000–2015)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploring chronological trends and the impact of historic droughts on Indian rice production.</div>', unsafe_allow_html=True)

    if artifacts['df_raw'] is not None:
        df_active = artifacts['df_raw'][artifacts['df_raw']['rice_area_1000ha'] > 0]
        
        yearly = df_active.groupby('year').agg(
            Mean_Yield=('rice_yield_kgha', 'mean'),
            Median_Yield=('rice_yield_kgha', 'median'),
            Monsoon_Rain=('rain_monsoon_mm', 'mean'),
            Water_Stress=('water_stress_score', 'mean')
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        
        ax1.plot(yearly['year'], yearly['Mean_Yield'], color='#2E7D32', marker='o', linewidth=2.5, label='Mean Yield (kg/ha)')
        ax2.plot(yearly['year'], yearly['Monsoon_Rain'], color='#1565C0', marker='s', linestyle='--', linewidth=2, label='Monsoon Rain (mm)')
        
        ax1.set_xlabel("Crop Year")
        ax1.set_ylabel("Yield (kg/ha)", color='#2E7D32')
        ax2.set_ylabel("Monsoon Rain (mm)", color='#1565C0')
        ax1.set_title("National Yield Trajectory vs. Monsoon Precipitation (2000–2015)", fontsize=13, fontweight='bold')
        st.pyplot(fig)

# ==============================================================================
# PAGE 7: MODEL PERFORMANCE
# ==============================================================================
elif page == "7. Model Performance":
    st.markdown('<div class="main-header">🌾 Comprehensive Model Performance & Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strict chronological validation (2012–2013) and out-of-time test evaluation (2014–2015 El Niño drought).</div>', unsafe_allow_html=True)

    st.markdown("### 📈 Supervised Regression Leaderboard")
    reg_res = artifacts['reg_metrics']['results']
    reg_rows = []
    for name, r in reg_res.items():
        reg_rows.append({
            "Model": name.replace("_", " ").title(),
            "Val MAE (kg/ha)": r['validation']['MAE'],
            "Val RMSE (kg/ha)": r['validation']['RMSE'],
            "Val R²": r['validation']['R2'],
            "Test MAE (kg/ha)": r['test']['MAE'],
            "Test RMSE (kg/ha)": r['test']['RMSE'],
            "Test R²": r['test']['R2']
        })
    st.dataframe(pd.DataFrame(reg_rows).sort_values(by="Val MAE (kg/ha)"), hide_index=True)

    st.markdown("---")
    st.markdown("### 🔬 Hybrid Architecture Comparative Benchmark")
    hyb = artifacts['hybrid_report']['gradient_boosting_experiment']
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Model A (Base Features Only):**")
        st.json(hyb['model_a_base'])
    with c2:
        st.write("**Model B (Base Features + Cluster Assignment):**")
        st.json(hyb['model_b_hybrid'])
        
    st.info(f"**Empirical Decision:** {hyb['architectural_decision']} (Δ MAE: {hyb['deltas']['delta_MAE']} kg/ha)")

# ==============================================================================
# PAGE 8: FEATURE IMPORTANCE
# ==============================================================================
elif page == "8. Feature Importance":
    st.markdown('<div class="main-header">🌾 Feature Importance & Distribution Drift Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Permutation feature importance and Kolmogorov-Smirnov drift analytics on test climate shocks.</div>', unsafe_allow_html=True)

    if artifacts['perm_df'] is not None:
        st.markdown("### 🔍 Permutation Feature Importance (Validation Drop in MAE)")
        top_perm = artifacts['perm_df'].head(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_perm, x='importance_mean_mae_drop', y='feature', palette='viridis', ax=ax)
        ax.set_xlabel("Mean MAE Degradation when Permuted (kg/ha)")
        ax.set_ylabel("Feature")
        ax.set_title("Top 15 Most Influential Features (Permutation Test)", fontsize=13, fontweight='bold')
        st.pyplot(fig)

    if artifacts['drift_df'] is not None:
        st.markdown("---")
        st.markdown("### 🌊 Input Feature Distribution Drift (Train vs. Test Period)")
        drift_count = int(artifacts['drift_df']['is_drift_detected'].sum())
        st.warning(f"⚠️ **{drift_count} of {len(artifacts['drift_df'])} features** showed statistically significant distribution shift (p < 0.05) during the 2014–2015 El Niño drought.")
        st.dataframe(artifacts['drift_df'].head(20), hide_index=True)
