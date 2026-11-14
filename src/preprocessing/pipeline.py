"""Preprocessing pipeline for Project RiPred.
Ensures zero data leakage, strict chronological isolation, and robust transformations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


PERMANENT_ZERO_DISTRICTS = [
    ('gujarat', 'amreli'),
    ('gujarat', 'jamnagar'),
    ('gujarat', 'rajkot'),
    ('himachal pradesh', 'lahul & spiti'),
    ('rajasthan', 'barmer'),
    ('rajasthan', 'churu'),
    ('rajasthan', 'jaisalmer'),
    ('rajasthan', 'jalore'),
    ('rajasthan', 'jodhpur'),
    ('rajasthan', 'nagaur'),
    ('rajasthan', 'pali'),
    ('rajasthan', 'sikar'),
]

LEAKAGE_COLUMNS = ['rice_production_1000tons']
TARGET_COLUMN = 'rice_yield_kgha'


def load_and_split_data(
    filepath: str = "data/raw/rice.csv",
    train_years: List[int] = list(range(2000, 2012)),
    val_years: List[int] = [2012, 2013],
    test_years: List[int] = [2014, 2015],
    filter_uncultivated: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Loads raw dataset, optionally filters uncultivated district-years,
    and performs chronological train/validation/test split.
    """
    df = pd.read_csv(filepath)
    
    total_raw = len(df)
    
    if filter_uncultivated:
        # Exclude permanent 12 zero districts
        is_perm_zero = df.apply(lambda r: (r['state'], r['district']) in PERMANENT_ZERO_DISTRICTS, axis=1)
        # Retain active cultivation (area > 0)
        df_active = df[(~is_perm_zero) & (df['rice_area_1000ha'] > 0)].copy()
    else:
        df_active = df.copy()

    # Chronological splits
    train_df = df_active[df_active['year'].isin(train_years)].copy()
    val_df = df_active[df_active['year'].isin(val_years)].copy()
    test_df = df_active[df_active['year'].isin(test_years)].copy()

    metadata = {
        "total_raw_rows": total_raw,
        "active_filtered_rows": len(df_active),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "filter_uncultivated": filter_uncultivated,
    }
    
    return train_df, val_df, test_df, metadata


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """Production preprocessing pipeline for RiPred.
    Fits all imputation, scaling, and categorical encoders strictly on training data.
    """
    def __init__(
        self,
        target_col: str = TARGET_COLUMN,
        leakage_cols: Optional[List[str]] = None,
        scale_numeric: bool = True
    ):
        self.target_col = target_col
        self.leakage_cols = leakage_cols or LEAKAGE_COLUMNS
        self.scale_numeric = scale_numeric
        
        self.meta_cols = ['year', 'state', 'district']
        self.categorical_cols = ['state']
        self.numerical_cols: List[str] = []
        
        self.imputer_num = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.feature_names_in_: List[str] = []
        self.transformed_feature_names_: List[str] = []
        
        # Risk classification thresholds derived from training yield percentiles
        self.yield_tercile_low_: float = 0.0
        self.yield_tercile_high_: float = 0.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit preprocessing transformations strictly on training set."""
        X = X.copy()
        
        # Calculate yield terciles if y is provided
        if y is not None:
            self.yield_tercile_low_ = float(np.percentile(y, 33.3333))
            self.yield_tercile_high_ = float(np.percentile(y, 66.6667))
        elif self.target_col in X.columns:
            target_series = X[self.target_col]
            self.yield_tercile_low_ = float(np.percentile(target_series, 33.3333))
            self.yield_tercile_high_ = float(np.percentile(target_series, 66.6667))
            
        # Determine candidate feature columns (exclude target, leakage columns, and district string to avoid over-parameterization)
        cols_to_drop = [self.target_col] + self.leakage_cols + ['district', 'year']
        candidate_cols = [c for c in X.columns if c not in cols_to_drop]
        
        self.categorical_cols = [c for c in ['state'] if c in candidate_cols]
        self.numerical_cols = [c for c in candidate_cols if c not in self.categorical_cols]
        self.feature_names_in_ = self.numerical_cols + self.categorical_cols
        
        # Fit numerical imputer and scaler
        if self.numerical_cols:
            num_data = X[self.numerical_cols].values
            imputed = self.imputer_num.fit_transform(num_data)
            if self.scale_numeric:
                self.scaler.fit(imputed)
                
        # Fit categorical OneHotEncoder
        if self.categorical_cols:
            self.ohe.fit(X[self.categorical_cols])
            cat_feature_names = list(self.ohe.get_feature_names_out(self.categorical_cols))
        else:
            cat_feature_names = []
            
        self.transformed_feature_names_ = list(self.numerical_cols) + cat_feature_names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into model-ready features."""
        X = X.copy()
        
        # 1. Transform numerical features
        num_data = X[self.numerical_cols].values
        imputed_num = self.imputer_num.transform(num_data)
        if self.scale_numeric:
            scaled_num = self.scaler.transform(imputed_num)
        else:
            scaled_num = imputed_num
            
        # 2. Transform categorical features
        if self.categorical_cols:
            encoded_cat = self.ohe.transform(X[self.categorical_cols])
            transformed_matrix = np.hstack([scaled_num, encoded_cat])
        else:
            transformed_matrix = scaled_num
            
        transformed_df = pd.DataFrame(
            transformed_matrix,
            columns=self.transformed_feature_names_,
            index=X.index
        )
        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_risk_categories(self, yields: pd.Series) -> pd.Series:
        """Classify yields into 'Low', 'Normal', 'High' based on training terciles."""
        def classify(y):
            if y < self.yield_tercile_low_:
                return "Low"
            elif y <= self.yield_tercile_high_:
                return "Normal"
            else:
                return "High"
        return yields.apply(classify)

    def save(self, filepath: str):
        """Save fitted pipeline object to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
        # Also save feature schema as JSON
        schema_path = os.path.join(os.path.dirname(filepath), "feature_schema.json")
        schema_data = {
            "feature_names_in": self.feature_names_in_,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "transformed_feature_names": self.transformed_feature_names_,
            "yield_tercile_low": self.yield_tercile_low_,
            "yield_tercile_high": self.yield_tercile_high_
        }
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PreprocessingPipeline':
        """Load fitted pipeline object from disk."""
        return joblib.load(filepath)
