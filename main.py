"""RiPred — Rice Intelligence & Yield Prediction Engine
Master Pipeline Orchestrator for Phase 2 through Phase 7.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline, load_and_split_data
from src.clustering.models import run_clustering_pipeline
from src.regression.models import train_and_evaluate_regressors, save_regression_artifacts
from src.hybrid.models import compare_hybrid_models
from src.classification.models import train_and_evaluate_classifiers, save_classification_artifacts
from src.explainability.explainer import (
    compute_permutation_importance,
    analyze_prediction_errors,
    monitor_feature_drift,
    CalibratedConformalPredictor
)


def run_full_pipeline():
    print("=" * 70)
    print("STARTING RIPRED V1 END-TO-END TRAINING AND BENCHMARKING PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # PHASE 2: DATA PREPROCESSING
    # -------------------------------------------------------------
    print("\n>>> PHASE 2: DATA PREPROCESSING & CHRONOLOGICAL ISOLATION")
    train_raw, val_raw, test_raw, split_meta = load_and_split_data(
        filepath="data/raw/rice.csv",
        train_years=list(range(2000, 2012)),
        val_years=[2012, 2013],
        test_years=[2014, 2015],
        filter_uncultivated=True
    )
    print(f"Dataset split: Train={len(train_raw)} (2000-2011), Val={len(val_raw)} (2012-2013), Test={len(test_raw)} (2014-2015)")

    preprocessor = PreprocessingPipeline(scale_numeric=True)
    X_train = preprocessor.fit_transform(train_raw)
    X_val = preprocessor.transform(val_raw)
    X_test = preprocessor.transform(test_raw)

    y_train = train_raw['rice_yield_kgha']
    y_val = val_raw['rice_yield_kgha']
    y_test = test_raw['rice_yield_kgha']

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save processed datasets
    train_processed = X_train.copy()
    train_processed['rice_yield_kgha'] = y_train
    train_processed.to_csv("data/processed/train.csv", index=False)

    val_processed = X_val.copy()
    val_processed['rice_yield_kgha'] = y_val
    val_processed.to_csv("data/processed/val.csv", index=False)

    test_processed = X_test.copy()
    test_processed['rice_yield_kgha'] = y_test
    test_processed.to_csv("data/processed/test.csv", index=False)

    preprocessor.save("models/preprocessor.joblib")
    print(f"Processed features: {X_train.shape[1]} columns. Preprocessor and schemas saved.")

    # -------------------------------------------------------------
    # PHASE 3: UNSUPERVISED CROP-CONDITION DISCOVERY
    # -------------------------------------------------------------
    print("\n>>> PHASE 3: UNSUPERVISED CROP-CONDITION DISCOVERY")
    clusterer, cluster_metrics = run_clustering_pipeline(
        train_df=train_raw,
        val_df=val_raw,
        test_df=test_raw,
        output_dir="models",
        n_clusters=4
    )
    train_clusters = clusterer.predict_cluster(train_raw)
    val_clusters = clusterer.predict_cluster(val_raw)
    test_clusters = clusterer.predict_cluster(test_raw)
    print(f"K-Means (k=4) fitted. Silhouette score: {cluster_metrics['kmeans_silhouette_train']:.4f}")
    print(f"Hierarchical Silhouette: {cluster_metrics['hierarchical_silhouette_train']:.4f}")
    print(f"DBSCAN detected {cluster_metrics['dbscan_noise_count']} ({cluster_metrics['dbscan_noise_pct']}%) unusual observations.")
    for cid, name in clusterer.cluster_names_.items():
        print(f" - {name}")

    # -------------------------------------------------------------
    # PHASE 4: SUPERVISED RICE YIELD REGRESSION
    # -------------------------------------------------------------
    print("\n>>> PHASE 4: SUPERVISED RICE YIELD REGRESSION BENCHMARKING")
    trained_regressors, reg_results = train_and_evaluate_regressors(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    # Pick best regressor based on validation MAE
    best_reg_name = min(
        [m for m in reg_results.keys() if m != 'baseline_mean'],
        key=lambda m: reg_results[m]['validation']['MAE']
    )
    save_regression_artifacts(trained_regressors, reg_results, best_reg_name, "models")
    print(f"Regression benchmarking complete. Best regressor: {best_reg_name.upper()}")
    for model_name, metrics in reg_results.items():
        print(f"  {model_name.ljust(22)} -> Val MAE: {metrics['validation']['MAE']} kg/ha | Val R2: {metrics['validation']['R2']} | Test MAE: {metrics['test']['MAE']} kg/ha | Test R2: {metrics['test']['R2']}")

    # -------------------------------------------------------------
    # PHASE 5: HYBRID MODEL EVALUATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 5: HYBRID MODEL COMPARATIVE BENCHMARK")
    hybrid_report = compare_hybrid_models(
        X_train_base=X_train, y_train=y_train,
        X_test_base=X_test, y_test=y_test,
        train_clusters=train_clusters, test_clusters=test_clusters,
        output_dir="models"
    )
    gb_exp = hybrid_report['gradient_boosting_experiment']
    print(f"Model A (Base GB)  -> Test MAE: {gb_exp['model_a_base']['MAE']} kg/ha | Test R2: {gb_exp['model_a_base']['R2']}")
    print(f"Model B (Hybrid GB)-> Test MAE: {gb_exp['model_b_hybrid']['MAE']} kg/ha | Test R2: {gb_exp['model_b_hybrid']['R2']}")
    print(f"Delta MAE: {gb_exp['deltas']['delta_MAE']} kg/ha | Delta R2: {gb_exp['deltas']['delta_R2']}")
    print(f"Architectural Decision: {gb_exp['architectural_decision']}")

    # -------------------------------------------------------------
    # PHASE 6: YIELD-RISK CLASSIFICATION
    # -------------------------------------------------------------
    print("\n>>> PHASE 6: YIELD-RISK CLASSIFICATION BENCHMARKING")
    trained_clfs, clf_results, risk_thresh = train_and_evaluate_classifiers(
        X_train=X_train, y_train_yield=y_train,
        X_val=X_val, y_val_yield=y_val,
        X_test=X_test, y_test_yield=y_test
    )
    best_clf_name = max(
        clf_results.keys(),
        key=lambda m: clf_results[m]['validation']['f1_weighted']
    )
    save_classification_artifacts(trained_clfs, clf_results, risk_thresh, best_clf_name, "models")
    print(f"Tiers: Low (< {risk_thresh[0]:.1f} kg/ha), Normal ({risk_thresh[0]:.1f} - {risk_thresh[1]:.1f} kg/ha), High (> {risk_thresh[1]:.1f} kg/ha)")
    print(f"Best Classifier: {best_clf_name.upper()}")
    for name, m in clf_results.items():
        print(f"  {name.ljust(22)} -> Val F1: {m['validation']['f1_weighted']:.4f} | Val ROC-AUC: {m['validation']['roc_auc_ovr_weighted']} | Test F1: {m['test']['f1_weighted']:.4f}")

    # -------------------------------------------------------------
    # PHASE 7: EXPLAINABILITY, DRIFT & RELIABILITY
    # -------------------------------------------------------------
    print("\n>>> PHASE 7: EXPLAINABILITY, DRIFT & CONFORMAL UNCERTAINTY")
    best_regressor = trained_regressors[best_reg_name]

    # 1. Permutation Feature Importance
    perm_df = compute_permutation_importance(best_regressor, X_val, y_val, n_repeats=5)
    perm_path = "reports/permutation_importance.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"Top 5 most impactful features by permutation test:")
    for _, row in perm_df.head(5).iterrows():
        print(f"  - {row['feature'].ljust(30)}: +{row['importance_mean_mae_drop']:.2f} kg/ha MAE degradation")

    # 2. Prediction Error Analysis Segmented by State & Year
    test_preds = best_regressor.predict(X_test)
    error_analysis = analyze_prediction_errors(y_test, test_preds, test_raw)
    with open("reports/prediction_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"Error analysis completed across {len(error_analysis['by_state'])} states and {len(error_analysis['by_year'])} test years.")

    # 3. Distribution Drift Monitoring
    drift_df = monitor_feature_drift(X_train, X_test)
    drift_path = "reports/feature_drift_report.csv"
    drift_df.to_csv(drift_path, index=False)
    drift_count = int(drift_df['is_drift_detected'].sum())
    print(f"Input drift monitoring: {drift_count} of {len(drift_df)} features exhibited statistically significant distribution drift in 2014-2015 El Niño drought.")

    # 4. Calibrated Conformal Prediction
    conformal = CalibratedConformalPredictor(best_regressor, confidence_level=0.90)
    conformal.calibrate(X_val, y_val)
    cov_eval = conformal.evaluate_coverage(X_test, y_test)
    conformal.save("models/conformal_predictor.joblib")
    with open("models/conformal_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(cov_eval, f, indent=2)
    print(f"Conformal Calibration (90% target confidence): Actual Test Coverage = {cov_eval['actual_test_coverage']*100:.1f}%, Mean Interval Margin = +/-{cov_eval['calibrated_residual_margin_kg_ha']:.1f} kg/ha")

    # -------------------------------------------------------------
    # SUMMARY REPORT GENERATION
    # -------------------------------------------------------------
    summary_report_path = "reports/model_evaluation_report.md"
    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write(f"""# RiPred v1.0 — Comprehensive Model Evaluation & Architecture Report

## 1. Executive Summary
RiPred v1 is fully trained, benchmarked, and serialized. All transformations, cluster assignments, and regression/classification models strictly observe chronological separation (Train: 2000–2011, Validation: 2012–2013, Test: 2014–2015).

## 2. Supervised Yield Regression Leaderboard

| Model Architecture | Train MAE | Val MAE (kg/ha) | Val RMSE (kg/ha) | Val R² | Test MAE (kg/ha) | Test RMSE (kg/ha) | Test R² |
|---|---|---|---|---|---|---|---|
""")
        for name, m in reg_results.items():
            f.write(f"| **{name}** | {m['train']['MAE']} | {m['validation']['MAE']} | {m['validation']['RMSE']} | {m['validation']['R2']} | {m['test']['MAE']} | {m['test']['RMSE']} | {m['test']['R2']} |\n")

        f.write(f"""
**Selected Best Regressor:** `{best_reg_name}`

## 3. Unsupervised Condition Discovery (k=4)
- **K-Means Silhouette:** {cluster_metrics['kmeans_silhouette_train']:.4f}
- **Hierarchical Silhouette:** {cluster_metrics['hierarchical_silhouette_train']:.4f}
- **DBSCAN Anomaly Count:** {cluster_metrics['dbscan_noise_count']} ({cluster_metrics['dbscan_noise_pct']}%)

### Condition Profiles:
""")
        for cid, prof in clusterer.cluster_profiles_.items():
            f.write(f"- **{prof['name']}** ({prof['percentage']}% of active area): Mean Rain = {prof['stats']['rain_annual_mm']['mean']} mm, Mean Stress = {prof['stats']['water_stress_score']['mean']}, Top States: {', '.join(prof['top_states'].keys())}\n")

        f.write(f"""
## 4. Hybrid Architecture Evaluation (Model A vs Model B)
- **Model A (Base {gb_exp['model_type']}):** Test MAE = {gb_exp['model_a_base']['MAE']} kg/ha, Test R² = {gb_exp['model_a_base']['R2']}
- **Model B (Hybrid {gb_exp['model_type']} with Cluster):** Test MAE = {gb_exp['model_b_hybrid']['MAE']} kg/ha, Test R² = {gb_exp['model_b_hybrid']['R2']}
- **Δ MAE:** {gb_exp['deltas']['delta_MAE']} kg/ha | **Δ R²:** {gb_exp['deltas']['delta_R2']}
- **Decision:** {gb_exp['architectural_decision']}

## 5. Yield-Risk Classification Leaderboard

| Classifier Architecture | Val Precision (w) | Val Recall (w) | Val F1 (w) | Val ROC-AUC | Test F1 (w) |
|---|---|---|---|---|---|
""")
        for name, m in clf_results.items():
            f.write(f"| **{name}** | {m['validation']['precision_weighted']} | {m['validation']['recall_weighted']} | {m['validation']['f1_weighted']} | {m['validation']['roc_auc_ovr_weighted']} | {m['test']['f1_weighted']} |\n")

        f.write(f"""
**Selected Best Classifier:** `{best_clf_name}`  
**Thresholds (derived from training percentiles):**
- Low: < {risk_thresh[0]:.1f} kg/ha
- Normal: {risk_thresh[0]:.1f} - {risk_thresh[1]:.1f} kg/ha
- High: > {risk_thresh[1]:.1f} kg/ha

## 6. Conformal Reliability & Uncertainty Intervals
- **Nominal Target Confidence:** 90%
- **Actual Empirical Coverage on Unseen Test (2014-2015):** {cov_eval['actual_test_coverage']*100:.1f}%
- **Calibrated Error Margin:** ± {cov_eval['calibrated_residual_margin_kg_ha']:.1f} kg/ha
""")

    print(f"\nReport generated at {summary_report_path}")
    print("=" * 70)
    print("RIPRED V1 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
