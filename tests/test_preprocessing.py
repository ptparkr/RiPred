"""Unit tests for RiPred Preprocessing Pipeline."""

import os
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline, load_and_split_data, LEAKAGE_COLUMNS


def test_load_and_split_data():
    raw_path = "data/raw/rice.csv"
    if not os.path.exists(raw_path):
        pytest.skip("Raw data file not present")

    train_df, val_df, test_df, meta = load_and_split_data(
        raw_path,
        train_years=list(range(2000, 2012)),
        val_years=[2012, 2013],
        test_years=[2014, 2015],
        filter_uncultivated=True
    )

    # Check temporal isolation
    assert set(train_df['year'].unique()).issubset(set(range(2000, 2012)))
    assert set(val_df['year'].unique()).issubset({2012, 2013})
    assert set(test_df['year'].unique()).issubset({2014, 2015})

    # Ensure no overlap
    assert len(set(train_df['year']).intersection(set(val_df['year']))) == 0
    assert len(set(train_df['year']).intersection(set(test_df['year']))) == 0

    # Ensure uncultivated filtering removed 0 area rows
    assert (train_df['rice_area_1000ha'] == 0).sum() == 0


def test_preprocessing_pipeline_fit_transform():
    # Synthetic small dataset
    df = pd.DataFrame({
        'year': [2000, 2001, 2002],
        'state': ['punjab', 'haryana', 'punjab'],
        'district': ['d1', 'd2', 'd3'],
        'rice_area_1000ha': [100.0, 150.0, 120.0],
        'rice_production_1000tons': [300.0, 450.0, 360.0],
        'rice_yield_kgha': [3000.0, 3000.0, 3000.0],
        'n_per_ha_nca': [150.0, 160.0, 140.0],
        'rain_annual_mm': [600.0, 700.0, 650.0]
    })

    pipeline = PreprocessingPipeline(scale_numeric=True)
    X_trans = pipeline.fit_transform(df)

    # Verify leakage column is excluded
    assert 'rice_production_1000tons' not in X_trans.columns
    assert 'rice_yield_kgha' not in X_trans.columns

    # Verify state one-hot encoding exists
    assert any(c.startswith('state_') for c in X_trans.columns)

    # Transform on new data with unseen state
    new_df = pd.DataFrame({
        'year': [2003],
        'state': ['kerala'],
        'district': ['d4'],
        'rice_area_1000ha': [80.0],
        'n_per_ha_nca': [120.0],
        'rain_annual_mm': [1200.0]
    })
    X_new_trans = pipeline.transform(new_df)
    assert X_new_trans.shape[1] == X_trans.shape[1]
