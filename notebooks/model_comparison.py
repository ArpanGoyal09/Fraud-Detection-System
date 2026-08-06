import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest


df = pd.read_csv('data/creditcard.csv')
df.head()

scaler = ss()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df = df.drop('Amount', axis=1)

df['Hour'] = (df['Time'] // 3600) % 24
df = df.drop('Time', axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42, stratify=y)


#XGBOOST

neg, pos = np.bincount(y_train)
scale_pos_weight = neg/pos
#print(scale_pos_weight)

xgb_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='aucpr',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]

y_pred_xgb_default = (y_probs_xgb >= 0.5).astype(int)
#print(confusion_matrix(y_test, y_pred_xgb_default))
#print(classification_report(y_test, y_pred_xgb_default))
#print("ROC_AUC:", roc_auc_score(y_test, y_probs_xgb))
#print("PR_AUC:", average_precision_score(y_test, y_probs_xgb))


precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_test, y_probs_xgb)
target_recall = 0.80
idx = np.argmin(np.abs(recalls_xgb[:-1] - target_recall))
xgb_threshold = thresholds_xgb[idx]

#print(f"\n=== XGBoost (tuned threshold={xgb_threshold:.4f}, matched to recall=0.80) ===")
y_pred_xgb_tuned = (y_probs_xgb >= xgb_threshold).astype(int)
#print(confusion_matrix(y_test, y_pred_xgb_tuned))
#print(classification_report(y_test, y_pred_xgb_tuned))


#ISOLATION FOREST

contamination_rate = y_train.mean()
#print("Contamination Rate: ", contamination_rate)

iso_model = IsolationForest(
    n_estimators=100,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1
)
iso_model.fit(X_train)

y_pred_iso = iso_model.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

print("=== Isolation Forest (unsupervised, contamination=actual fraud rate) ===")
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))
