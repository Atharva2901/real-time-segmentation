from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Segmentation API")

# Load artifacts at startup (will exist after you train the model)
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PROCESSOR_PATH = os.path.join("artifacts", "processor.pkl")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
processor = joblib.load(PROCESSOR_PATH) if os.path.exists(PROCESSOR_PATH) else None

class InputData(BaseModel):
    CUST_ID: str
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: int
    PURCHASES_TRX: int
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: int

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: InputData):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not trained yet. Run training first.")
    try:
        df = pd.DataFrame([input_data.dict()])
        df = df.drop(columns=["CUST_ID"])
        X = processor.transform(df)
        cluster = model.predict(X)
        return {"cluster_label": int(cluster[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
