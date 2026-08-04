from fastapi.testclient import TestClient as tc
from src.main import app

client = tc(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] =="ok"

def test_predict_flags_known_fraud_case():

    fraud_transaction = {
        "Time": 406.0, "Amount": 9.99,
        "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
        "V5": -0.5222, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
        "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
        "V13": -0.5952, "V14": -4.2892, "V15": 0.3898, "V16": -1.1407,
        "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
        "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
        "V25": 0.0445, "V26": 0.1780, "V27": 0.2611, "V28": -0.1433,
    }

    response = client.post("/predict", json=fraud_transaction)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] >= 0.41

def test_predict_clears_normal_transaction():

    normal_transaction = {
        "Time": 50000, "Amount": 25.0,
        "V1": 0.1, "V2": -0.05, "V3": 0.08, "V4": -0.02, "V5": 0.03,
        "V6": -0.06, "V7": 0.01, "V8": 0.02, "V9": -0.03, "V10": 0.04,
        "V11": -0.01, "V12": 0.05, "V13": -0.02, "V14": 0.03, "V15": -0.04,
        "V16": 0.02, "V17": -0.01, "V18": 0.03, "V19": -0.02, "V20": 0.01,
        "V21": -0.03, "V22": 0.04, "V23": -0.01, "V24": 0.02, "V25": -0.03,
        "V26": 0.01, "V27": -0.02, "V28": 0.01,
    }

    response = client.post("/predict", json=normal_transaction)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is False
