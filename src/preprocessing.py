import pandas as pd
import joblib as jb

amount_scaler = jb.load("models/amount-scaler.pkl")

featureOrder = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount_Scaled', 'Hour']

def preprocess(transaction: dict) -> pd.DataFrame:
    data = transaction.copy()

    data["Hour"] = (data["Time"] // 3600) % 24

    amount_df = pd.DataFrame({"Amount": [data["Amount"]]})
    data["Amount_Scaled"] = amount_scaler.transform(amount_df)[0][0]

    row = {col: data[col] for col in featureOrder}

    return pd.DataFrame([row])