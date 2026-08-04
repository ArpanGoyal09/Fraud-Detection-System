import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as logR
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier as randomF
import joblib as jb


df = pd.read_csv('fraud-detection-system/data/creditcard.csv')
df.head()
#df.info()

df['Class'].value_counts()

df['Amount'].describe()

df.groupby('Class')['Amount'].describe()

scaler = ss()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df = df.drop('Amount', axis=1)

df[['Amount_Scaled']].describe()


df['Hour'] = (df['Time'] // 3600) % 24
df.groupby('Class')['Hour'].describe()

df = df.drop('Time', axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42, stratify=y)

#print(y_train.value_counts())
#print(y_test.value_counts())


#LOGISTIC REGRESSION 
model = logR(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))


#RANDOM FOREST
rf_model = randomF(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

#print(confusion_matrix(y_test, y_pred_rf))
#print(classification_report(y_test, y_pred_rf))

y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

#print("ROC_AUC:", roc_auc_score(y_test, y_probs_rf))
#print("PR_AUC:", average_precision_score(y_test, y_probs_rf))

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_rf)

'''
plt.figure(figsize=(8,6))
plt.plot(recalls, precisions, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.grid(True)
plt.show()
'''

target_recall = 0.80
idx = np.argmin(np.abs(recalls[:-1] - target_recall))
#print(f"Threshold: {thresholds[idx]:.4f}")
#print(f"Precision at this point: {precisions[idx]:.4f}")
#print(f"Recall at this point: {recalls[idx]:.4f}")

final_threshold = 0.41
y_pred_final = (y_probs_rf >= final_threshold).astype(int)

#print(confusion_matrix(y_test, y_pred_final))
#print(classification_report(y_test, y_pred_final))

jb.dump(rf_model, 'fraud-model.pkl')
jb.dump(scaler, 'amount-scaler.pkl')