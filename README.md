# Fraud Detection System

A credit card fraud detection system: a Random Forest classifier trained on transaction data, served through a FastAPI wrapper. 

## Status

- [x] EDA + preprocessing
- [x] Model training (Random Forest, tuned threshold)
- [x] FastAPI wrapper with tests
- [x] Model comparison (XGBoost, Isolation Forest)
- [ ] OCI deployment
- [ ] Live endpoint / deployment architecture

This is a work in progress. Everything below reflects what's actually built and tested; deployment is the one piece still outstanding.

## The problem

Credit card fraud detection is a classification problem with severe class imbalance. In this dataset, fraud makes up **0.173%** of transactions (492 out of 284,807). A model that just predicts "not fraud" every time would be 99.8% accurate and completely useless. The real challenge isn't accuracy, it's finding a small number of fraud cases in a huge pile of legitimate ones, without generating so many false alarms that the system becomes unusable.

## Dataset

[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions, 31 columns. `V1`–`V28` are PCA-transformed features (anonymized for privacy), plus `Time`, `Amount`, and the `Class` label. The raw CSV isn't included in this repo (it's ~150MB and the dataset's license doesn't cover redistribution). You can download it from Kaggle and place it at `data/creditcard.csv` to reproduce the notebooks.

## Approach

**Feature engineering:** `Time` (seconds since first transaction) carries no real signal on its own, so it's dropped in favor of a derived `Hour` (0–23) feature, which shows a mild but real difference between fraud and legitimate transactions. `Amount` is scaled with `StandardScaler` fit on the training set only.

**Imbalance handling:** Class weighting (`class_weight='balanced'`) rather than SMOTE to avoid synthetic data artifacts in a high-dimensional PCA space where interpolation between points is hard to reason about.

**Model:** Random Forest was chosen over Logistic Regression because fraud signal here depends on non-linear relationships between PCA components, which a linear model can't capture. The decision threshold was tuned from the default 0.5 down to **0.41**, the edge of the precision-recall curve's flat plateau, trading a small amount of precision for meaningfully better recall.

**Evaluation:** PR-AUC, not ROC-AUC, is the headline metric. Under this level of class imbalance, ROC-AUC is inflated by the huge number of true negatives and doesn't reflect real-world usefulness.

Full reasoning for every decision above — including what was tried and rejected — is in [`docs/model_comparison.md`](docs/model_comparison.md).

## Results

| Metric | Value |
|---|---|
| Precision | 0.95 |
| Recall | 0.80 |
| PR-AUC | 0.86 |
| ROC-AUC | 0.953 |
| False positives (test set) | 4 out of 56,864 legitimate transactions |
| False negatives (test set) | 20 out of 98 fraud cases |

Also benchmarked against XGBoost (marginally better, not deployed, see comparison doc for why) and Isolation Forest (unsupervised, notably worse, confirms this problem benefits from labeled data rather than pure anomaly detection).

## API

The trained model is served through a FastAPI app (`src/main.py`). It accepts a raw transaction (`Time`, `Amount`, `V1`–`V28`), replicates the training-time preprocessing internally (scaling, hour derivation), and returns a fraud prediction using the tuned 0.41 threshold and not the model's default.

**Endpoints:**
- `GET /` for health check
- `POST /predict` to score a transaction

### Running locally

```bash
git clone https://github.com/ArpanGoyal09/Fraud-Detection-System.git
cd Fraud-Detection-System
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for an interactive test page, or run the automated tests:

```bash
pytest tests/ -v
```

## Project structure

```
├── data/               # dataset (gitignored)
├── docs/                # model comparison writeup and reasoning
├── models/              # trained model + scaler artifacts
├── notebooks/           # EDA, training, and model comparison scripts
├── src/                 # FastAPI application
│   ├── main.py           # API endpoints
│   ├── preprocessing.py  # feature engineering pipeline (mirrors training)
│   └── schemas.py        # request/response validation
└── tests/                # automated API tests
```

## Limitations

- **`V1`–`V28` are specific to this dataset's PCA transformation** and aren't transferable to other fraud datasets because a different dataset's anonymized features, even if similarly named, would represent a different underlying transformation and can't be validly scored by this model.
- **No live deployment yet** but the API is fully built and tested locally.
- Trained on data from a fixed time window; a production system would need to handle model drift as fraud patterns evolve over time.

## What's next

- OCI deployment
- Deployment architecture diagram
- Live endpoint link and latency numbers, once deployed
