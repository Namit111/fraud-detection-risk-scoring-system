from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# --------------------------------------------------
# App configuration
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection & Risk Scoring API",
    description="Machine learning API for detecting potentially fraudulent transactions.",
    version="1.0.0"
)


# --------------------------------------------------
# Load trained model package
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "fraud_detection_model_package.joblib"
)

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
FRAUD_THRESHOLD = model_package["threshold"]
FEATURES = model_package["features"]


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "application": "Fraud Detection & Risk Scoring API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": model_package["model_name"],
        "fraud_threshold": FRAUD_THRESHOLD,
        "number_of_features": len(FEATURES),
        "metrics": model_package["metrics"]
    }


@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request into DataFrame
    transaction_data = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Ensure feature order matches training
    transaction_data = transaction_data[FEATURES]

    # Get fraud probability
    fraud_probability = model.predict_proba(
        transaction_data
    )[0][1]

    # Convert probability to risk score
    risk_score = fraud_probability * 100

    # Fraud decision
    is_fraud = fraud_probability >= FRAUD_THRESHOLD

    # Risk level
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_score": round(float(risk_score), 2),
        "risk_level": risk_level,
        "is_fraud": bool(is_fraud)
    }