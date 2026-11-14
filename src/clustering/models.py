"""Unsupervised Crop-Condition Discovery Pipeline for RiPred.
Evaluates K-Means, Hierarchical Clustering, and DBSCAN using purely condition-related features.
Generates empirical cluster profiles, PCA projections, and outlier diagnostics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_condition_features(df: pd.DataFrame) -> List[str]:
    """Extracts strictly environmental, hydrological, soil, and baseline agronomic features.
    Explicitly excludes target (yield), production, and cultivated area.
    """
    excluded = [
        'year', 'state', 'district',
        'rice_yield_kgha', 'rice_production_1000tons', 'rice_area_1000ha',
        'n_kharif_tons', 'n_rabi_tons', 'p_kharif_tons', 'p_rabi_tons', 'k_kharif_tons', 'k_rabi_tons'
    ]
    candidate_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    return candidate_cols


def evaluate_cluster_counts(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> pd.DataFrame:
    """Computes Inertia, Silhouette Score, Calinski-Harabasz, and Davies-Bouldin across k."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        
        results.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette_score": round(float(sil), 4),
            "calinski_harabasz": round(float(ch), 2),
            "davies_bouldin": round(float(db), 4)
        })
    return pd.DataFrame(results)


class CropConditionClusterer:
    """Manages clustering models, standardizers, PCA visualization, and profiling."""
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.dbscan = DBSCAN(eps=4.5, min_samples=10)
        
        self.feature_names_: List[str] = []
        self.cluster_profiles_: Dict[str, Any] = {}
        self.cluster_names_: Dict[int, str] = {}
        
    def fit(self, df: pd.DataFrame):
        """Fit condition scaler, PCA, and clusterers on condition features."""
        self.feature_names_ = get_condition_features(df)
        X_raw = df[self.feature_names_].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Fit PCA
        self.pca.fit(X_scaled)
        
        # Fit K-Means
        self.kmeans.fit(X_scaled)
        
        # Build profiles based on actual centroid characteristics
        self._build_cluster_profiles(df, X_scaled)
        return self

    def predict_cluster(self, df: pd.DataFrame) -> np.ndarray:
        """Assign cluster indices (K-Means) to new observations."""
        X_raw = df[self.feature_names_].values
        X_scaled = self.scaler.transform(X_raw)
        return self.kmeans.predict(X_scaled)

    def transform_pca(self, df: pd.DataFrame) -> np.ndarray:
        """Project features onto PCA components."""
        X_raw = df[self.feature_names_].values
        X_scaled = self.scaler.transform(X_raw)
        return self.pca.transform(X_scaled)

    def _build_cluster_profiles(self, df: pd.DataFrame, X_scaled: np.ndarray):
        """Analyze actual centroid values and characteristics to build evidence-based profiles."""
        labels = self.kmeans.labels_
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        profiles = {}
        cluster_names = {}
        
        # Key diagnostic indicators
        key_metrics = [
            'rain_annual_mm', 'rain_monsoon_mm', 'water_stress_score', 'lgp_days',
            'tmax_monsoon', 'tmin_winter', 'n_per_ha_nca', 'irrig_canal_area_1000ha',
            'irrig_tubewell_area_1000ha', 'soil_vertisols', 'soil_inceptisols', 'soil_orthids'
        ]
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        for c in range(self.n_clusters):
            c_mask = (labels == c)
            c_size = int(c_mask.sum())
            c_pct = round(float(c_size / len(df) * 100), 2)
            c_df = df_clustered[df_clustered['cluster'] == c]
            
            c_stats = {}
            for m in available_metrics:
                c_stats[m] = {
                    "mean": round(float(c_df[m].mean()), 2),
                    "median": round(float(c_df[m].median()), 2),
                    "p25": round(float(c_df[m].quantile(0.25)), 2),
                    "p75": round(float(c_df[m].quantile(0.75)), 2)
                }
                
            # If yield is in df, record yield statistics without using it in fitting
            if 'rice_yield_kgha' in df.columns:
                yield_mean = float(c_df['rice_yield_kgha'].mean())
                yield_median = float(c_df['rice_yield_kgha'].median())
                c_stats['yield_actual_mean'] = round(yield_mean, 1)
                c_stats['yield_actual_median'] = round(yield_median, 1)
                
            # Top states in cluster
            top_states = c_df['state'].value_counts().head(3).to_dict() if 'state' in c_df.columns else {}
            
            # Evidence-based agro-ecological naming
            avg_rain = c_stats['rain_annual_mm']['mean']
            avg_stress = c_stats['water_stress_score']['mean']
            avg_n = c_stats['n_per_ha_nca']['mean']
            avg_vert = c_stats.get('soil_vertisols', {}).get('mean', 0)
            
            if avg_stress > 150:
                name = f"Cluster {c}: Semi-Arid / High Moisture Stress Zone"
            elif avg_rain > 1400:
                name = f"Cluster {c}: Humid / High Monsoon Rainfall Zone"
            elif avg_n > 140 or c_stats.get('irrig_tubewell_area_1000ha', {}).get('mean', 0) > 80:
                name = f"Cluster {c}: Intensive Irrigated / High-Input Agronomic Zone"
            elif avg_vert > 20:
                name = f"Cluster {c}: Vertisol / Deep Clay Sub-Humid Zone"
            else:
                name = f"Cluster {c}: Moderate Rainfall / Balanced Agro-Climatic Zone"
                
            cluster_names[c] = name
            profiles[str(c)] = {
                "name": name,
                "count": c_size,
                "percentage": c_pct,
                "top_states": top_states,
                "stats": c_stats
            }
            
        self.cluster_profiles_ = profiles
        self.cluster_names_ = cluster_names

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
        # Save profiles report JSON
        profile_path = os.path.join(os.path.dirname(filepath), "cluster_profiles.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_clusters": self.n_clusters,
                "feature_count": len(self.feature_names_),
                "features": self.feature_names_,
                "pca_explained_variance_ratio": [round(float(v), 4) for v in self.pca.explained_variance_ratio_],
                "profiles": self.cluster_profiles_
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'CropConditionClusterer':
        return joblib.load(filepath)


def run_clustering_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "models",
    n_clusters: int = 4
) -> Tuple[CropConditionClusterer, Dict]:
    """Runs full unsupervised clustering analysis, evaluates elbow/silhouette,
    fits clusterer on training condition features, evaluates validation/test clustering,
    and returns clusterer and comprehensive metrics.
    """
    clusterer = CropConditionClusterer(n_clusters=n_clusters)
    clusterer.fit(train_df)
    
    # Evaluate optimal k on training set
    condition_features = get_condition_features(train_df)
    X_train_scaled = clusterer.scaler.transform(train_df[condition_features].values)
    k_eval_df = evaluate_cluster_counts(X_train_scaled, k_range=range(2, 8))
    
    # Hierarchical clustering comparison
    hier_labels = clusterer.hierarchical.fit_predict(X_train_scaled)
    hier_sil = float(silhouette_score(X_train_scaled, hier_labels))
    
    # DBSCAN outlier experiment
    db_labels = clusterer.dbscan.fit_predict(X_train_scaled)
    db_noise_count = int((db_labels == -1).sum())
    db_noise_pct = round(float(db_noise_count / len(train_df) * 100), 2)
    
    # Assign clusters across splits
    train_clusters = clusterer.predict_cluster(train_df)
    val_clusters = clusterer.predict_cluster(val_df)
    test_clusters = clusterer.predict_cluster(test_df)
    
    pca_var = [round(float(v), 4) for v in clusterer.pca.explained_variance_ratio_]
    
    metrics = {
        "k_evaluation": k_eval_df.to_dict(orient="records"),
        "chosen_k": n_clusters,
        "kmeans_silhouette_train": float(silhouette_score(X_train_scaled, train_clusters)),
        "hierarchical_silhouette_train": hier_sil,
        "dbscan_noise_count": db_noise_count,
        "dbscan_noise_pct": db_noise_pct,
        "pca_variance_ratio_3d": pca_var,
        "pca_total_variance_explained": round(float(sum(pca_var)), 4),
        "cluster_distribution_train": pd.Series(train_clusters).value_counts().to_dict(),
        "cluster_distribution_val": pd.Series(val_clusters).value_counts().to_dict(),
        "cluster_distribution_test": pd.Series(test_clusters).value_counts().to_dict(),
    }
    
    # Save artifacts
    model_path = os.path.join(output_dir, "crop_condition_clusterer.joblib")
    clusterer.save(model_path)
    
    metrics_path = os.path.join(output_dir, "clustering_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    return clusterer, metrics
