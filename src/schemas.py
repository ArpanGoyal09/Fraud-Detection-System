from pydantic import BaseModel, Field

class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since the first transaction")
    Amount: float = Field(..., ge=0, description="Raw transaction amount")
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 406.0,
                "Amount": 9.99,
                "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
                "V5": -0.5222, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
                "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
                "V13": -0.5952, "V14": -4.2892, "V15": 0.3898, "V16": -1.1407,
                "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
                "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
                "V25": 0.0445, "V26": 0.1780, "V27": 0.2611, "V28": -0.1433
            }
        }
    }

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float