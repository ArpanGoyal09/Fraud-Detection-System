import joblib as jb
from fastapi import FastAPI as fa
from src.schemas import Transaction, PredictionResponse
from src.preprocessing import preprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = jb.load("models/fraud-model.pkl")

decisionThreshold = 0.41

app = fa(
    title = "Fraud Detection System",
    description = "Scores credit card transactions for fraud probability using a tuned Random Forest classifier.",
    verison = "1.0.0",
)

@app.get("/")
def health_check():
    return {"status": "ok", "message":"Fraud Detection API running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    featuresDF = preprocess(transaction.model_dump())
    fraudProbab = model.predict_proba(featuresDF)[0][1]
    is_fraud = fraudProbab >= decisionThreshold

    logger.info(
        "prediction: is_fraud=%s probability=%.4f threshold=%s",
        is_fraud, fraudProbab, decisionThreshold
    )

    return PredictionResponse(
        is_fraud=bool(is_fraud),
        fraud_probability=round(float(fraudProbab),4),
        threshold_used=decisionThreshold,
    )
