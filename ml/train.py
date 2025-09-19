import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from ml.ingest import DataIngestion
from ml.transform import DataTransformation

DATASET_PATH = os.getenv("DATASET_PATH", os.path.join("data", "customers.csv"))
MODEL_PATH = os.path.join("artifacts", "model.pkl")

def find_optimal_k(X, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1.0
    # Try different K and choose the one with best silhouette score
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def main():
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Place your CSV there or set DATASET_PATH env var."
        )

    print(f"[train] Reading dataset from {DATASET_PATH}")
    di = DataIngestion()
    saved_csv = di.start_data_ingestion(DATASET_PATH)

    df = pd.read_csv(saved_csv)
    dt = DataTransformation()
    X, processor_path = dt.fit_transform(df)

    print("[train] Selecting K with silhouette score…")
    k = find_optimal_k(X, k_min=2, k_max=10)
    print(f"[train] Chosen K = {k}")

    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    model.fit(X)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] Saved model → {MODEL_PATH}")
    print(f"[train] Saved processor → {processor_path}")

if __name__ == "__main__":
    main()
