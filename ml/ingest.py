import os
import pandas as pd

class DataIngestion:
    def __init__(self, output_path: str = os.path.join("artifacts", "df.csv")):
        self.data_path = output_path

    def start_data_ingestion(self, src_path: str) -> str:
        df = pd.read_csv(src_path)
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False, header=True)
        return self.data_path
