"""Yield-Risk Classification Module for Project RiPred.
Categorizes district harvests into Low, Normal, and High Yield tiers based strictly on training yield percentiles.
Evaluates Logistic Regression, Random Forest, and Gradient Boosting Classifiers.
Outputs calibrated class probabilities alongside class assignments.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)


class YieldRiskClassifier:
    """Wrapper delivering class labels, risk probabilities, and confidence intervals."""
    def __init__(self, model: Any, classes: List[str], thresholds: Tuple[float, float]):
        self.model = model
        self.classes = classes
        self.low_threshold = thresholds[0]
        self.high_threshold = thresholds[1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_with_details(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        probs = self.predict_proba(X)
        preds = self.predict(X)
        results = []
        for pred, prob_arr in zip(preds, probs):
            prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, prob_arr)}
            results.append({
                "predicted_risk": pred,
                "confidence": round(float(np.max(prob_arr)), 4),
                "class_probabilities": prob_dict,
                "tier_definitions_kg_ha": {
                    "Low": f"< {self.low_threshold:.1f} kg/ha",
                    "Normal": f"{self.low_threshold:.1f} - {self.high_threshold:.1f} kg/ha",
                    "High": f"> {self.high_threshold:.1f} kg/ha"
                }
            })
        return results


def evaluate_classifier_performance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    classes: List[str]
) -> Dict[str, Any]:
    """Computes Precision, Recall, F1, Confusion Matrix, and ROC-AUC."""
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    prec_macro = precision_score(y, preds, average='macro', zero_division=0)
    rec_macro = recall_score(y, preds, average='macro', zero_division=0)
    f1_macro = f1_score(y, preds, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y, preds, average='weighted', zero_division=0)
    rec_weighted = recall_score(y, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y, preds, labels=classes).tolist()
    
    try:
        auc = roc_auc_score(y, probs, multi_class='ovr', average='weighted')
    except Exception:
        auc = None
        
    return {
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc_ovr_weighted": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": cm,
        "classes": classes
    }


def train_and_evaluate_classifiers(
    X_train: pd.DataFrame,
    y_train_yield: pd.Series,
    X_val: pd.DataFrame,
    y_val_yield: pd.Series,
    X_test: pd.DataFrame,
    y_test_yield: pd.Series,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], Tuple[float, float]]:
    """Calculates risk thresholds from training yield and benchmarks classifiers."""
    
    # 1. Derive risk thresholds strictly from training set
    low_thresh = float(np.percentile(y_train_yield, 33.3333))
    high_thresh = float(np.percentile(y_train_yield, 66.6667))
    classes = ["Low", "Normal", "High"]
    
    def assign_tier(y_val):
        if y_val < low_thresh:
            return "Low"
        elif y_val <= high_thresh:
            return "Normal"
        else:
            return "High"

    y_train_cat = y_train_yield.apply(assign_tier)
    y_val_cat = y_val_yield.apply(assign_tier)
    y_test_cat = y_test_yield.apply(assign_tier)

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=6, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state)
    }

    trained_models = {}
    eval_results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train_cat)
        trained_models[name] = clf
        
        val_eval = evaluate_classifier_performance(clf, X_val, y_val_cat, classes)
        test_eval = evaluate_classifier_performance(clf, X_test, y_test_cat, classes)
        train_eval = evaluate_classifier_performance(clf, X_train, y_train_cat, classes)
        
        eval_results[name] = {
            "train": train_eval,
            "validation": val_eval,
            "test": test_eval
        }

    return trained_models, eval_results, (low_thresh, high_thresh)


def save_classification_artifacts(
    trained_models: Dict[str, Any],
    eval_results: Dict[str, Any],
    thresholds: Tuple[float, float],
    best_model_name: str,
    output_dir: str = "models"
):
    """Saves classification models and benchmark evaluation reports."""
    os.makedirs(output_dir, exist_ok=True)
    classes = ["Low", "Normal", "High"]
    
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(output_dir, f"classifier_{name}.joblib"))
        
    best_raw = trained_models[best_model_name]
    wrapped_best = YieldRiskClassifier(best_raw, classes, thresholds)
    joblib.dump(wrapped_best, os.path.join(output_dir, "best_risk_classifier.joblib"))
    
    metrics_path = os.path.join(output_dir, "classification_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": "yield_risk_tier",
            "classes": classes,
            "thresholds_kg_ha": {
                "low_threshold": round(thresholds[0], 2),
                "high_threshold": round(thresholds[1], 2)
            },
            "best_model": best_model_name,
            "results": eval_results
        }, f, indent=2)
