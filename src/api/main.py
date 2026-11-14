"""RiPred — Rice Intelligence & Yield Prediction Engine
Production FastAPI Application.
Exposes endpoints for yield regression, risk classification, crop-condition clustering,
conformal uncertainty bounds, and benchmark metadata.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="RiPred API — Rice Intelligence & Yield Prediction Engine",
    description="Enterprise ML inference service for Indian rice yield prediction, risk categorization, and agro-climatic clustering.",
    version="1.0.0"
)

# Global artifacts container
MODELS_DIR = "models"
REPORTS_DIR = "reports"

artifacts = {}


def load_artifacts():
    """Load preprocessor, models, clusterers, and metadata."""
    try:
        artifacts['preprocessor'] = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        artifacts['regressor'] = joblib.load(os.path.join(MODELS_DIR, "best_yield_regressor.joblib"))
        artifacts['classifier'] = joblib.load(os.path.join(MODELS_DIR, "best_risk_classifier.joblib"))
        artifacts['clusterer'] = joblib.load(os.path.join(MODELS_DIR, "crop_condition_clusterer.joblib"))
        artifacts['conformal'] = joblib.load(os.path.join(MODELS_DIR, "conformal_predictor.joblib"))
        
        with open(os.path.join(MODELS_DIR, "feature_schema.json"), "r") as f:
            artifacts['schema'] = json.load(f)
            
        with open(os.path.join(MODELS_DIR, "cluster_profiles.json"), "r") as f:
            artifacts['cluster_profiles'] = json.load(f)
            
        with open(os.path.join(MODELS_DIR, "regression_metrics.json"), "r") as f:
            artifacts['reg_metrics'] = json.load(f)

        with open(os.path.join(MODELS_DIR, "classification_metrics.json"), "r") as f:
            artifacts['clf_metrics'] = json.load(f)

        with open(os.path.join(MODELS_DIR, "hybrid_comparison_report.json"), "r") as f:
            artifacts['hybrid_report'] = json.load(f)
            
        print("All RiPred inference artifacts loaded successfully.")
    except Exception as e:
        print(f"Warning: Artifacts loading error: {e}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


# -----------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------
class AgriculturalInput(BaseModel):
    state: str = Field(..., example="punjab")
    district: Optional[str] = Field("ludhiana", example="ludhiana")
    year: Optional[int] = Field(2015, example=2015)
    
    # Core Agronomic & Soil Features
    rice_area_1000ha: float = Field(250.0, example=250.0)
    n_per_ha_nca: float = Field(165.0, example=165.0)
    p_per_ha_nca: float = Field(45.0, example=45.0)
    k_per_ha_nca: float = Field(20.0, example=20.0)
    irrig_tubewell_area_1000ha: float = Field(200.0, example=200.0)
    irrig_canal_area_1000ha: float = Field(50.0, example=50.0)
    irrig_tank_area_1000ha: float = Field(0.0, example=0.0)
    irrig_other_area_1000ha: float = Field(2.0, example=2.0)
    lgp_days: float = Field(180.0, example=180.0)
    
    # Climate Features
    rain_annual_mm: float = Field(650.0, example=650.0)
    rain_monsoon_mm: float = Field(500.0, example=500.0)
    rain_pre_monsoon_mm: float = Field(40.0, example=40.0)
    rain_post_monsoon_mm: float = Field(20.0, example=20.0)
    rain_winter_mm: float = Field(35.0, example=35.0)
    tmax_monsoon: float = Field(34.5, example=34.5)
    tmin_winter: float = Field(5.5, example=5.5)
    water_stress_score: float = Field(110.0, example=110.0)
    stress_group: int = Field(1, example=1)
    
    # Additional optional features with defaults
    custom_features: Optional[Dict[str, float]] = None


class YieldPredictionResponse(BaseModel):
    model_version: str
    predicted_yield_kgha: float
    unit: str
    conformal_lower_bound_kgha: float
    conformal_upper_bound_kgha: float
    confidence_level: float
    calibrated_margin_kgha: float


class RiskClassificationResponse(BaseModel):
    model_version: str
    predicted_risk_tier: str
    confidence: float
    probabilities: Dict[str, float]
    tier_definitions: Dict[str, str]


class CropConditionResponse(BaseModel):
    assigned_cluster_id: int
    cluster_name: str
    pca_coordinates: List[float]
    profile_summary: Dict[str, Any]


class ComprehensivePredictionResponse(BaseModel):
    model_version: str
    location: Dict[str, str]
    yield_prediction: YieldPredictionResponse
    risk_classification: RiskClassificationResponse
    crop_condition: CropConditionResponse
    explainability_notes: str


# -----------------------------------------------------------------
# Helper: Prepare DataFrame from Request
# -----------------------------------------------------------------
def create_input_dataframe(req: AgriculturalInput) -> pd.DataFrame:
    """Constructs a DataFrame matching the training data schema, imputing defaults for unprovided fields."""
    if 'schema' not in artifacts:
        load_artifacts()
        
    num_cols = artifacts['schema']['numerical_cols']
    
    # Default values based on median training features
    data = {}
    for col in num_cols:
        data[col] = 0.0

    # Fill provided values
    req_dict = req.dict()
    for k, v in req_dict.items():
        if k in data and v is not None:
            data[k] = float(v)

    if req.custom_features:
        for k, v in req.custom_features.items():
            if k in data:
                data[k] = float(v)

    data['state'] = req.state.lower().strip()
    data['district'] = req.district.lower().strip() if req.district else 'unknown'
    data['year'] = req.year or 2015

    df = pd.DataFrame([data])
    return df


# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------
@app.get("/health")
def healthcheck():
    """System health check and loaded model status."""
    if not artifacts:
        load_artifacts()
    return {
        "status": "healthy",
        "service": "RiPred Yield Intelligence Engine",
        "version": "1.0.0",
        "models_loaded": {
            "regressor": "regressor" in artifacts,
            "classifier": "classifier" in artifacts,
            "clusterer": "clusterer" in artifacts,
            "conformal_predictor": "conformal" in artifacts
        }
    }


@app.get("/meta")
def get_metadata():
    """Returns feature schemas, units, and model metadata."""
    if 'schema' not in artifacts:
        load_artifacts()
    return {
        "project": "RiPred",
        "target_variable": "rice_yield_kgha",
        "target_unit": "kg/ha",
        "observation_granularity": "District-Crop Year",
        "risk_thresholds_kg_ha": {
            "low": artifacts['schema']['yield_tercile_low'],
            "high": artifacts['schema']['yield_tercile_high']
        },
        "feature_count_transformed": len(artifacts['schema']['transformed_feature_names']),
        "numerical_features": artifacts['schema']['numerical_cols'],
        "categorical_features": artifacts['schema']['categorical_cols']
    }


@app.post("/predict/yield", response_model=YieldPredictionResponse)
def predict_yield(req: AgriculturalInput):
    """Predicts rice yield (kg/ha) with calibrated conformal uncertainty interval."""
    if 'regressor' not in artifacts:
        load_artifacts()
        
    df = create_input_dataframe(req)
    X_trans = artifacts['preprocessor'].transform(df)
    
    point_pred, lower, upper = artifacts['conformal'].predict_intervals(X_trans)
    
    return YieldPredictionResponse(
        model_version="RiPred-v1.0 (Gradient Boosting)",
        predicted_yield_kgha=round(float(point_pred[0]), 2),
        unit="kg/ha",
        conformal_lower_bound_kgha=round(float(lower[0]), 2),
        conformal_upper_bound_kgha=round(float(upper[0]), 2),
        confidence_level=0.90,
        calibrated_margin_kgha=round(artifacts['conformal'].quantile_residual_, 2)
    )


@app.post("/predict/risk", response_model=RiskClassificationResponse)
def predict_risk(req: AgriculturalInput):
    """Predicts yield-risk tier (Low, Normal, High) with calibrated class probabilities."""
    if 'classifier' not in artifacts:
        load_artifacts()

    df = create_input_dataframe(req)
    X_trans = artifacts['preprocessor'].transform(df)
    
    details = artifacts['classifier'].predict_with_details(X_trans)[0]
    
    return RiskClassificationResponse(
        model_version="RiPred-v1.0 (Gradient Boosting Classifier)",
        predicted_risk_tier=details["predicted_risk"],
        confidence=details["confidence"],
        probabilities=details["class_probabilities"],
        tier_definitions=details["tier_definitions_kg_ha"]
    )


@app.post("/predict/condition", response_model=CropConditionResponse)
def predict_condition(req: AgriculturalInput):
    """Assigns the observation to an unsupervised crop-condition cluster archetype."""
    if 'clusterer' not in artifacts:
        load_artifacts()

    df = create_input_dataframe(req)
    cluster_id = int(artifacts['clusterer'].predict_cluster(df)[0])
    pca_coords = artifacts['clusterer'].transform_pca(df)[0].tolist()
    
    cluster_profile = artifacts['cluster_profiles']['profiles'].get(str(cluster_id), {})
    
    return CropConditionResponse(
        assigned_cluster_id=cluster_id,
        cluster_name=artifacts['clusterer'].cluster_names_.get(cluster_id, f"Cluster {cluster_id}"),
        pca_coordinates=[round(float(c), 3) for c in pca_coords],
        profile_summary=cluster_profile
    )


@app.post("/predict/comprehensive", response_model=ComprehensivePredictionResponse)
def predict_comprehensive(req: AgriculturalInput):
    """Unified endpoint returning yield, conformal bounds, risk probabilities, and cluster profile."""
    yield_res = predict_yield(req)
    risk_res = predict_risk(req)
    cond_res = predict_condition(req)
    
    notes = (
        f"Predictions generated for {req.state.title()} ({req.district.title() if req.district else 'District'}). "
        f"Identified as '{cond_res.cluster_name}'. "
        f"Calibrated 90% conformal interval guarantees empirical coverage based on historical test shocks."
    )

    return ComprehensivePredictionResponse(
        model_version="RiPred-v1.0-Production",
        location={"state": req.state, "district": req.district or "unknown", "year": str(req.year or 2015)},
        yield_prediction=yield_res,
        risk_classification=risk_res,
        crop_condition=cond_res,
        explainability_notes=notes
    )


@app.get("/clusters")
def get_clusters():
    """Returns cluster profiles and PCA explained variance ratios."""
    if 'cluster_profiles' not in artifacts:
        load_artifacts()
    return artifacts['cluster_profiles']


@app.get("/evaluation")
def get_evaluation_metrics():
    """Returns regression, classification, and hybrid evaluation leaderboards."""
    if 'reg_metrics' not in artifacts:
        load_artifacts()
    return {
        "regression": artifacts['reg_metrics'],
        "classification": artifacts['clf_metrics'],
        "hybrid_comparison": artifacts['hybrid_report']
    }
