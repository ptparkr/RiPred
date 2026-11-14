"""Unit tests for RiPred Unsupervised Clustering Module."""

import pytest
import numpy as np
import pandas as pd
from src.clustering.models import CropConditionClusterer, get_condition_features


def test_condition_features_exclusion():
    df = pd.DataFrame({
        'rice_yield_kgha': [2000.0],
        'rice_production_1000tons': [200.0],
        'rice_area_1000ha': [100.0],
        'state': ['punjab'],
        'rain_annual_mm': [700.0],
        'water_stress_score': [100.0]
    })
    cond_feats = get_condition_features(df)
    assert 'rice_yield_kgha' not in cond_feats
    assert 'rice_production_1000tons' not in cond_feats
    assert 'rice_area_1000ha' not in cond_feats
    assert 'rain_annual_mm' in cond_feats
    assert 'water_stress_score' in cond_feats


def test_clusterer_fit_predict():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        'state': ['punjab'] * 30 + ['tamil nadu'] * 30,
        'district': [f'd_{i}' for i in range(n)],
        'year': [2010] * n,
        'rice_yield_kgha': np.random.uniform(1500, 3500, n),
        'rice_production_1000tons': np.random.uniform(100, 500, n),
        'rice_area_1000ha': np.random.uniform(50, 200, n),
        'rain_annual_mm': np.random.uniform(400, 2000, n),
        'rain_monsoon_mm': np.random.uniform(300, 1500, n),
        'water_stress_score': np.random.uniform(30, 200, n),
        'lgp_days': np.random.uniform(100, 250, n),
        'tmax_monsoon': np.random.uniform(28, 38, n),
        'tmin_winter': np.random.uniform(5, 20, n),
        'n_per_ha_nca': np.random.uniform(50, 250, n),
        'irrig_tubewell_area_1000ha': np.random.uniform(0, 100, n)
    })

    clusterer = CropConditionClusterer(n_clusters=3, random_state=42)
    clusterer.fit(df)

    # Predict clusters
    clusters = clusterer.predict_cluster(df)
    assert len(clusters) == n
    assert set(np.unique(clusters)).issubset({0, 1, 2})

    # PCA transformation
    pca_coords = clusterer.transform_pca(df)
    assert pca_coords.shape == (n, 3)

    # Profiles exist
    assert len(clusterer.cluster_profiles_) == 3
    for cid in range(3):
        assert str(cid) in clusterer.cluster_profiles_
