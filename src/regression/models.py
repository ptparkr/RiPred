"""Supervised Rice Yield Regression Module for Project RiPred.
Evaluates Historical Baseline, Ridge, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
Computes MAE, RMSE, and R2 on chronological validation and out-of-sample test periods.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class HistoricalMeanBaseline(BaseEstimator, RegressorMixin):
    """Baseline predicting historical mean yield by state (with national fallback)."""
    def __init__(self):
        self.national_mean_: float = 0.0
        self.state_means_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.national_mean_ = float(y.mean())
        if 'state' in X.columns:
            temp_df = pd.DataFrame({'state': X['state'], 'y': y})
            self.state_means_ = temp_df.groupby('state')['y'].mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'state' in X.columns:
            preds = [self.state_means_.get(s, self.national_mean_) for s in X['state']]
            return np.array(preds)
        return np.full(len(X), self.national_mean_)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and R-squared for continuous yield predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4)
    }


def train_and_evaluate_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains regression models on training split and evaluates on validation and test sets."""
    
    models: Dict[str, Any] = {
        "baseline_mean": HistoricalMeanBaseline(),
        "ridge_regression": Ridge(alpha=10.0, random_state=random_state),
        "decision_tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=15, random_state=random_state),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=8, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state)
    }
    
    if HAS_XGBOOST:
        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
            n_jobs=-1
        )

    results: Dict[str, Any] = {}
    trained_models: Dict[str, Any] = {}
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on validation and test
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        train_m = compute_metrics(y_train, y_train_pred)
        val_m = compute_metrics(y_val, y_val_pred)
        test_m = compute_metrics(y_test, y_test_pred)
        
        results[name] = {
            "train": train_m,
            "validation": val_m,
            "test": test_m
        }

    return trained_models, results


def save_regression_artifacts(
    trained_models: Dict[str, Any],
    results: Dict[str, Any],
    best_model_name: str,
    output_dir: str = "models"
):
    """Saves all trained regression models and benchmark evaluation table."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual models
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(output_dir, f"regression_{name}.joblib"))
        
    # Save best regression model as primary predictor
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(output_dir, "best_yield_regressor.joblib"))
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "regression_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": "rice_yield_kgha",
            "unit": "kg/ha",
            "best_model": best_model_name,
            "models_evaluated": list(results.keys()),
            "results": results
        }, f, indent=2)
