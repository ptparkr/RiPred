"""Unit tests for RiPred Regression, Classification, and Uncertainty Models."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from src.regression.models import HistoricalMeanBaseline, compute_metrics
from src.classification.models import YieldRiskClassifier
from src.explainability.explainer import CalibratedConformalPredictor


def test_compute_metrics():
    y_true = np.array([2000.0, 3000.0, 4000.0])
    y_pred = np.array([2100.0, 2900.0, 4000.0])
    metrics = compute_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] == pytest.approx(66.67, 0.05)


def test_conformal_predictor():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y = pd.Series(2000 + 500 * X['f_0'] + np.random.randn(100) * 100)

    reg = GradientBoostingRegressor(n_estimators=20, random_state=42)
    reg.fit(X[:70], y[:70])

    conformal = CalibratedConformalPredictor(reg, confidence_level=0.90)
    conformal.calibrate(X[70:85], y[70:85])

    point_preds, lower, upper = conformal.predict_intervals(X[85:])
    assert len(point_preds) == 15
    assert np.all(lower <= upper)
    assert np.all(point_preds >= lower)
    assert np.all(point_preds <= upper)


def test_yield_risk_classifier():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=[f"c_{i}" for i in range(4)])
    y = np.random.choice(["Low", "Normal", "High"], size=60)

    clf = GradientBoostingClassifier(n_estimators=15, random_state=42)
    clf.fit(X, y)

    wrapper = YieldRiskClassifier(clf, classes=["Low", "Normal", "High"], thresholds=(1500.0, 2500.0))
    details = wrapper.predict_with_details(X[:5])
    assert len(details) == 5
    for item in details:
        assert item["predicted_risk"] in ["Low", "Normal", "High"]
        probs = item["class_probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, 0.01)
