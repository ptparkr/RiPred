"""Hybrid Model Benchmarking Module for Project RiPred.
Directly evaluates whether unsupervised cluster assignment improves supervised yield prediction:
- Model A: Original input features -> yield prediction
- Model B: Original input features + cluster assignment -> yield prediction
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_cluster_augmentation(
    X_train_base: pd.DataFrame,
    y_train: pd.Series,
    X_test_base: pd.DataFrame,
    y_test: pd.Series,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    model_type: str = "gradient_boosting",
    random_state: int = 42
) -> Dict[str, Any]:
    """Fits and compares Model A (base features) vs Model B (base features + one-hot cluster assignment)."""
    
    # 1. One-hot encode cluster assignments
    ohe_cluster = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_clust_encoded = ohe_cluster.fit_transform(train_clusters.reshape(-1, 1))
    test_clust_encoded = ohe_cluster.transform(test_clusters.reshape(-1, 1))
    cluster_cols = [f"cluster_{i}" for i in range(train_clust_encoded.shape[1])]
    
    # Construct Model B feature sets
    X_train_hybrid = X_train_base.copy()
    X_test_hybrid = X_test_base.copy()
    for idx, col in enumerate(cluster_cols):
        X_train_hybrid[col] = train_clust_encoded[:, idx]
        X_test_hybrid[col] = test_clust_encoded[:, idx]

    # Instantiate base and hybrid regressors identically
    if model_type == "random_forest":
        model_a = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=8, random_state=random_state, n_jobs=-1)
        model_b = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=8, random_state=random_state, n_jobs=-1)
    else:
        model_a = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state)
        model_b = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state)

    # Train Model A
    model_a.fit(X_train_base, y_train)
    pred_a_test = model_a.predict(X_test_base)
    mae_a = mean_absolute_error(y_test, pred_a_test)
    rmse_a = np.sqrt(mean_squared_error(y_test, pred_a_test))
    r2_a = r2_score(y_test, pred_a_test)

    # Train Model B
    model_b.fit(X_train_hybrid, y_train)
    pred_b_test = model_b.predict(X_test_hybrid)
    mae_b = mean_absolute_error(y_test, pred_b_test)
    rmse_b = np.sqrt(mean_squared_error(y_test, pred_b_test))
    r2_b = r2_score(y_test, pred_b_test)

    # Compute improvement differences
    diff_mae = mae_b - mae_a      # Negative is improvement
    diff_rmse = rmse_b - rmse_a    # Negative is improvement
    diff_r2 = r2_b - r2_a          # Positive is improvement

    improves = (diff_mae < 0) and (diff_rmse < 0) and (diff_r2 > 0)
    
    recommendation = (
        "Adopt Hybrid Architecture (Cluster feature improves generalization)"
        if improves else
        "Retain Clustering as Independent Analytical Component (Do not force into prediction model)"
    )

    comparison = {
        "model_type": model_type,
        "model_a_base": {
            "MAE": round(float(mae_a), 2),
            "RMSE": round(float(rmse_a), 2),
            "R2": round(float(r2_a), 4)
        },
        "model_b_hybrid": {
            "MAE": round(float(mae_b), 2),
            "RMSE": round(float(rmse_b), 2),
            "R2": round(float(r2_b), 4)
        },
        "deltas": {
            "delta_MAE": round(float(diff_mae), 2),
            "delta_RMSE": round(float(diff_rmse), 2),
            "delta_R2": round(float(diff_r2), 4)
        },
        "cluster_feature_improves": bool(improves),
        "architectural_decision": recommendation
    }

    return comparison, model_a, model_b


def compare_hybrid_models(
    X_train_base: pd.DataFrame,
    y_train: pd.Series,
    X_test_base: pd.DataFrame,
    y_test: pd.Series,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    output_dir: str = "models"
) -> Dict[str, Any]:
    """Runs hybrid comparison across both Gradient Boosting and Random Forest architectures."""
    gb_comp, gb_a, gb_b = evaluate_cluster_augmentation(
        X_train_base, y_train, X_test_base, y_test, train_clusters, test_clusters, model_type="gradient_boosting"
    )
    rf_comp, rf_a, rf_b = evaluate_cluster_augmentation(
        X_train_base, y_train, X_test_base, y_test, train_clusters, test_clusters, model_type="random_forest"
    )

    full_results = {
        "gradient_boosting_experiment": gb_comp,
        "random_forest_experiment": rf_comp,
        "summary": {
            "gb_improves": gb_comp["cluster_feature_improves"],
            "rf_improves": rf_comp["cluster_feature_improves"],
            "final_architecture": (
                "Hybrid Model" if gb_comp["cluster_feature_improves"] or rf_comp["cluster_feature_improves"]
                else "Decoupled Architecture (Independent Clustering + Baseline Regression)"
            )
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "hybrid_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    return full_results
