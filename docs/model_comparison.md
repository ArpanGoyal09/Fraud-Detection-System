# Model Comparison

**Purpose:** The deployed model (Random Forest) was benchmarked against two alternative approaches to validate that it was a deliberate choice, not the only option considered. All models were evaluated on the identical train/test split (`random_state=42`, `test_size=0.2`, stratified) and identical preprocessing, so differences reflect the models themselves, not data variance.

---

## Summary

| Model | Precision | Recall | PR-AUC | False Positives | False Negatives |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.06 | 0.91 | — | ~1,500 | ~9 |
| **Random Forest (deployed)** | **0.95** | **0.80** | **0.86** | **4** | **20** |
| XGBoost | 0.96 | 0.80 | 0.877 | 3 | 20 |
| Isolation Forest (unsupervised) | 0.31 | 0.35 | — | 75 | 64 |

Random Forest and XGBoost were both evaluated at a matched recall of 0.80, by tuning each model's decision threshold independently off its own precision-recall curve. This isolates the comparison to precision at equal recall, rather than comparing arbitrary default thresholds.

---

## Logistic Regression (baseline)

High recall (0.91) but unusably low precision (0.06) — roughly 1,500 false positives per ~57,000 legitimate transactions. Included only as a baseline to demonstrate why a linear model is a poor fit here: fraud signal in this dataset is driven by non-linear relationships among the PCA-transformed features, which Logistic Regression's linear decision boundary cannot capture.

## Random Forest — deployed model

Trained with `class_weight='balanced'` to address the 0.173% fraud rate without synthetic data. Default threshold (0.5) gave 0.96 precision / 0.74 recall; tuned to 0.41 (the edge of the precision-recall curve's flat plateau) to recover recall to 0.80 at minimal precision cost (0.96 → 0.95). See Section 5 of the main project write-up for the full threshold-selection reasoning.

**Why this remains the deployed model despite XGBoost's marginal edge:** see Decision below.

## XGBoost

Gradient-boosted trees, trained sequentially so each tree corrects the errors of the ones before it — a meaningfully different approach from Random Forest's independent, parallel trees. Class imbalance handled via `scale_pos_weight` (≈577, computed as the ratio of legit to fraud transactions in the training set — XGBoost's equivalent of `class_weight='balanced'`, though it must be calculated explicitly rather than passed as a keyword).

At matched recall (0.80), XGBoost edges out Random Forest slightly: one fewer false positive (3 vs. 4) and a higher PR-AUC (0.877 vs. 0.86), indicating marginally better overall ranking quality. Notably, **false negatives were identical (20) for both models** — on the metric that matters most for a bank (catching fraud), the two models perform the same at this operating point; the difference is confined to precision at the margin.

One calibration note worth flagging: XGBoost's tuned threshold (0.986) is far higher than Random Forest's (0.41). This is not evidence of XGBoost being "more confident" in any meaningful sense — it reflects that the two models' probability outputs sit on different internal scales (Random Forest's are softened by averaging across 100 independent trees; XGBoost's are pushed toward the extremes by sequential boosting combined with `scale_pos_weight`). This is why models were compared at matched recall rather than matched raw threshold — the threshold values themselves aren't directly comparable across model types.

## Isolation Forest (unsupervised)

Included to test a fundamentally different paradigm: anomaly detection based on geometric isolation, rather than supervised classification. Trained without labels (`y_train` never used in `.fit()`); `contamination` set to the true training-set fraud rate to give it the fairest possible operating point.

Performance was substantially worse (0.31 precision / 0.35 recall) than either supervised model. This is a meaningful negative result, not a failed experiment: it indicates that fraud in this dataset is not simply *statistical outlier* behavior separable by geometric isolation — it has label-dependent structure that supervised learning can exploit but pure anomaly detection cannot see. This confirms supervised classification was the correct approach given labeled data is available, rather than an assumption taken for granted.

---

## Decision: Random Forest remains the deployed model

XGBoost's improvement is real but marginal (1 fewer false positive out of ~57,000 transactions; ~0.017 PR-AUC gain) and comes with costs: an additional dependency, a less intuitive threshold to reason about and explain, and a need to re-validate the full FastAPI pipeline against a new model artifact. Given Random Forest is already deployed, tested end-to-end (unit tests, live API validation), and well understood, the marginal metric gain does not currently justify the switch.

This is treated as a live decision, not a closed one — if false positives become a measured problem in a more realistic deployment scenario, or if the dependency cost becomes worth it for other reasons, XGBoost is the documented next step, and the modular design of the API (`preprocessing.py` / `main.py` separated from the model artifact) makes swapping it in straightforward.
