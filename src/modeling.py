"""
Modeling & Clustering Pipeline.
Handles preprocessing (PowerTransform/RobustScaler), PCA,
K-Means, Gaussian Mixture, Agglomerative clustering, evaluation,
model persistence and inference.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

CLUSTERING_FEATURES = [
    'recency_days',
    'sessions_per_week',
    'total_engagement_hours_90d',
    'feature_adoption_rate',
    'usage_momentum_30d',
    'monetary_revenue_90d',
    'support_ticket_friction',
    'weekend_session_share'
]

SEGMENT_NAMES = {
    0: {"name": "Power Champions", "desc": "High frequency, deep feature adoption & highest revenue generators."},
    1: {"name": "Core Active Users", "desc": "Steady, reliable mid-tier engagement with consistent workflows."},
    2: {"name": "At-Risk Dormant", "desc": "High recency (dormant), sharp drop in momentum, and high support friction."},
    3: {"name": "Casual Explorers", "desc": "Low duration, minimal feature exploration, primarily free/starter plans."},
    4: {"name": "Emerging / High-Potential", "desc": "Recent signups with high momentum and fast initial adoption."}
}

class SegmentationPipeline:
    def __init__(self, n_clusters: int = 5, random_seed: int = 42):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.features = CLUSTERING_FEATURES
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=2, random_state=random_seed)
        self.model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=random_seed)
        self.is_fitted = False

    def fit_transform_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_raw = df[self.features].values
        X_trans = self.transformer.fit_transform(X_raw)
        X_scaled = self.scaler.fit_transform(X_trans)
        pca_coords = self.pca.fit_transform(X_scaled)
        return X_scaled, pca_coords

    def transform_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_raw = df[self.features].values
        X_trans = self.transformer.transform(X_raw)
        X_scaled = self.scaler.transform(X_trans)
        pca_coords = self.pca.transform(X_scaled)
        return X_scaled, pca_coords

    def fit(self, df: pd.DataFrame):
        X_scaled, _ = self.fit_transform_features(df)
        self.model.fit(X_scaled)
        self.gmm.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet.")
        X_scaled, _ = self.transform_features(df)
        return self.model.predict(X_scaled)

    def evaluate_clustering(self, df: pd.DataFrame, max_k: int = 8) -> dict:
        X_scaled, _ = self.fit_transform_features(df)
        k_range = range(2, max_k + 1)
        results = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=5)
            preds = km.fit_predict(X_scaled)
            results['k'].append(k)
            results['inertia'].append(float(km.inertia_))
            results['silhouette'].append(float(silhouette_score(X_scaled, preds, sample_size=2000, random_state=self.random_seed)))
            results['davies_bouldin'].append(float(davies_bouldin_score(X_scaled, preds)))
            results['calinski_harabasz'].append(float(calinski_harabasz_score(X_scaled, preds)))
        return results

    def save(self, filepath: str = "models/segmentation_pipeline.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/segmentation_pipeline.joblib"):
        return joblib.load(filepath)

def train_and_export_pipeline():
    df = pd.read_parquet("data/processed/features_users.parquet")
    pipeline = SegmentationPipeline(n_clusters=5, random_seed=42)
    pipeline.fit(df)
    
    X_scaled, pca_coords = pipeline.transform_features(df)
    kmeans_labels = pipeline.model.predict(X_scaled)
    gmm_labels = pipeline.gmm.predict(X_scaled)

    df['cluster'] = kmeans_labels
    df['cluster_gmm'] = gmm_labels
    df['pca_x'] = pca_coords[:, 0]
    df['pca_y'] = pca_coords[:, 1]
    
    # Map cluster names based on revenue & recency ordering
    cluster_order = df.groupby('cluster')['monetary_revenue_90d'].mean().sort_values(ascending=False).index.tolist()
    # Name mapping: highest revenue = Champions, lowest momentum/high recency = At-Risk, etc.
    # Let's assign descriptive titles
    label_map = {}
    for idx, c in enumerate(cluster_order):
        c_mean_recency = df[df['cluster'] == c]['recency_days'].mean()
        c_mean_rev = df[df['cluster'] == c]['monetary_revenue_90d'].mean()
        c_mean_mom = df[df['cluster'] == c]['usage_momentum_30d'].mean()
        
        if c_mean_rev > 150:
            name = "Power Champions"
        elif c_mean_recency > 35 or c_mean_mom < 0.4:
            name = "At-Risk Dormant"
        elif c_mean_mom > 1.3:
            name = "Emerging / High-Growth"
        elif c_mean_rev > 50:
            name = "Core Active Users"
        else:
            name = "Casual Explorers"
        label_map[c] = name

    df['segment_name'] = df['cluster'].map(label_map)
    df.to_parquet("data/processed/segmented_users.parquet", index=False)
    pipeline.save("models/segmentation_pipeline.joblib")
    
    eval_metrics = pipeline.evaluate_clustering(df, max_k=7)
    pd.DataFrame(eval_metrics).to_csv("reports/cluster_evaluation_metrics.csv", index=False)
    print("Pipeline trained and exported. Evaluation saved.")

if __name__ == "__main__":
    train_and_export_pipeline()
