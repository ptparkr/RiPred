"""Explainability, Drift Monitoring, and Conformal Uncertainty Module for Project RiPred.
Implements:
1. Permutation Feature Importance.
2. Prediction Error Analysis by State and Year.
3. Distribution Drift Monitoring via Kolmogorov-Smirnov statistics.
4. Calibrated Uncertainty Intervals (Split Conformal Prediction).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import ks_2samp
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Computes Permutation Feature Importance on unseen validation data."""
    perm = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_mean_absolute_error"
    )
    
    importance_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean_mae_drop": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values(by="importance_mean_mae_drop", ascending=False)
    
    return importance_df


def analyze_prediction_errors(
    y_true: pd.Series,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame
) -> Dict[str, Any]:
    """Analyzes prediction error metrics (MAE, RMSE, bias) segmented by State and Year."""
    eval_df = meta_df[['year', 'state', 'district']].copy()
    eval_df['actual'] = y_true.values
    eval_df['predicted'] = y_pred
    eval_df['residual'] = eval_df['actual'] - eval_df['predicted']
    eval_df['abs_error'] = eval_df['residual'].abs()

    # State-level error breakdown
    state_errors = eval_df.groupby('state').agg(
        records=('actual', 'count'),
        mean_actual=('actual', 'mean'),
        mean_pred=('predicted', 'mean'),
        mean_bias=('residual', 'mean'),
        mae=('abs_error', 'mean'),
        rmse=('residual', lambda r: np.sqrt(np.mean(r**2)))
    ).round(2).sort_values(by='mae', ascending=True).to_dict(orient='index')

    # Year-level error breakdown
    year_errors = eval_df.groupby('year').agg(
        records=('actual', 'count'),
        mean_actual=('actual', 'mean'),
        mean_pred=('predicted', 'mean'),
        mean_bias=('residual', 'mean'),
        mae=('abs_error', 'mean'),
        rmse=('residual', lambda r: np.sqrt(np.mean(r**2)))
    ).round(2).sort_values(by='year').to_dict(orient='index')

    overall_metrics = {
        "overall_mae": round(float(eval_df['abs_error'].mean()), 2),
        "overall_rmse": round(float(np.sqrt(np.mean(eval_df['residual']**2))), 2),
        "overall_mean_bias": round(float(eval_df['residual'].mean()), 2),
        "median_abs_error": round(float(eval_df['abs_error'].median()), 2)
    }

    return {
        "overall": overall_metrics,
        "by_state": state_errors,
        "by_year": year_errors
    }


def monitor_feature_drift(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """Monitors distribution drift between training and test sets using Kolmogorov-Smirnov tests."""
    drift_records = []
    for col in train_features.columns:
        if pd.api.types.is_numeric_dtype(train_features[col]):
            train_vals = train_features[col].dropna()
            test_vals = test_features[col].dropna()
            
            stat, p_val = ks_2samp(train_vals, test_vals)
            has_drift = (p_val < alpha)
            
            drift_records.append({
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 5),
                "is_drift_detected": bool(has_drift),
                "train_mean": round(float(train_vals.mean()), 2),
                "test_mean": round(float(test_vals.mean()), 2),
                "pct_mean_shift": round(float((test_vals.mean() - train_vals.mean()) / (train_vals.mean() + 1e-9) * 100), 2)
            })
            
    drift_df = pd.DataFrame(drift_records).sort_values(by="ks_statistic", ascending=False)
    return drift_df


class CalibratedConformalPredictor:
    """Split-conformal prediction for guaranteed finite-sample coverage intervals."""
    def __init__(self, base_model: Any, confidence_level: float = 0.90):
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.quantile_residual_: float = 0.0

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """Calculates conformity score threshold (quantile of absolute residuals) on calibration set."""
        preds = self.base_model.predict(X_cal)
        abs_residuals = np.abs(y_cal.values - preds)
        
        # Conformal quantile adjustment for finite sample coverage
        n = len(abs_residuals)
        q_level = min(1.0, np.ceil((n + 1) * self.confidence_level) / n)
        self.quantile_residual_ = float(np.quantile(abs_residuals, q_level))
        return self

    def predict_intervals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns point predictions, lower bounds, and upper bounds in kg/ha."""
        point_preds = self.base_model.predict(X)
        lower_bounds = np.maximum(0.0, point_preds - self.quantile_residual_)
        upper_bounds = point_preds + self.quantile_residual_
        return point_preds, lower_bounds, upper_bounds

    def evaluate_coverage(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates empirical coverage and mean interval width on out-of-sample test data."""
        point_preds, lower, upper = self.predict_intervals(X_test)
        y_true = y_test.values
        covered = (y_true >= lower) & (y_true <= upper)
        actual_coverage = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))
        
        return {
            "nominal_confidence": self.confidence_level,
            "actual_test_coverage": round(actual_coverage, 4),
            "calibrated_residual_margin_kg_ha": round(self.quantile_residual_, 2),
            "mean_interval_width_kg_ha": round(mean_width, 2),
            "coverage_guarantee_met": bool(actual_coverage >= (self.confidence_level - 0.05))
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CalibratedConformalPredictor':
        return joblib.load(filepath)
