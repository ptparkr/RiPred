"""Integration tests for RiPred FastAPI Service."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_healthcheck_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_meta_endpoint():
    res = client.get("/meta")
    assert res.status_code == 200
    data = res.json()
    assert data["target_variable"] == "rice_yield_kgha"
    assert data["target_unit"] == "kg/ha"


def test_predict_yield_endpoint():
    payload = {
        "state": "punjab",
        "district": "ludhiana",
        "year": 2015,
        "rice_area_1000ha": 250.0,
        "n_per_ha_nca": 165.0,
        "p_per_ha_nca": 45.0,
        "k_per_ha_nca": 20.0,
        "irrig_tubewell_area_1000ha": 200.0,
        "rain_monsoon_mm": 500.0,
        "tmax_monsoon": 34.5,
        "water_stress_score": 110.0
    }
    res = client.post("/predict/yield", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_yield_kgha" in data
    assert data["unit"] == "kg/ha"
    assert data["conformal_lower_bound_kgha"] <= data["conformal_upper_bound_kgha"]


def test_predict_risk_endpoint():
    payload = {
        "state": "bihar",
        "district": "patna",
        "year": 2015,
        "n_per_ha_nca": 90.0,
        "rain_monsoon_mm": 800.0,
        "water_stress_score": 80.0
    }
    res = client.post("/predict/risk", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_risk_tier"] in ["Low", "Normal", "High"]
    assert "probabilities" in data


def test_predict_comprehensive_endpoint():
    payload = {
        "state": "tamil nadu",
        "district": "thanjavur",
        "year": 2015,
        "n_per_ha_nca": 140.0,
        "rain_monsoon_mm": 400.0,
        "water_stress_score": 130.0
    }
    res = client.post("/predict/comprehensive", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "yield_prediction" in data
    assert "risk_classification" in data
    assert "crop_condition" in data
    assert "explainability_notes" in data
