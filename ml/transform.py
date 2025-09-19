import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class DataTransformation:
    def __init__(self, processor_path: str = os.path.join("artifacts", "processor.pkl")):
        self.processor_path = processor_path

    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()

        # Impute a couple of known columns if present
        if "MINIMUM_PAYMENTS" in df.columns:
            df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
        if "CREDIT_LIMIT" in df.columns:
            df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())

        # Drop ID column (not used in clustering)
        if "CUST_ID" in df.columns:
            df = df.drop(columns=["CUST_ID"], axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        os.makedirs(os.path.dirname(self.processor_path), exist_ok=True)
        joblib.dump(scaler, self.processor_path)
        return X_scaled, self.processor_path
