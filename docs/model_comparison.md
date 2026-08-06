# Model Comparison

Random Forest is the model actually deployed in this project. Before settling on it, I benchmarked it against XGBoost and Isolation Forest to check whether a different approach would do better. All three were trained and evaluated on the same train/test split (`random_state=42`, `test_size=0.2`, stratified) and the same preprocessing, so the comparison is fair and any difference in results comes from the model, not the data.

## Results

| Model | Precision | Recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|
| Logistic Regression | 0.06 | 0.91 | — | ~1,500 | ~9 |
| Random Forest | 0.95 | 0.80 | 0.86 | 4 | 20 |
| XGBoost | 0.96 | 0.80 | 0.877 | 3 | 20 |
| Isolation Forest | 0.31 | 0.35 | — | 75 | 64 |

Random Forest and XGBoost are both shown at recall = 0.80, tuned to that point off each model's own precision-recall curve. Comparing them at their default thresholds wouldn't mean much since the two models' probability outputs aren't on the same scale.

## Logistic Regression

This was the original baseline, not something I seriously considered deploying. 0.91 recall looks good on paper but precision is 0.06 — about 1,500 false positives on ~57K legit transactions in the test set. Fraud here depends on non-linear interactions between the PCA components, and Logistic Regression can only draw a straight line through that space, so this result is expected rather than surprising.

## Random Forest

Same as documented in the main writeup: `class_weight='balanced'`, threshold tuned from the default 0.5 down to 0.41 to recover recall from 0.74 to 0.80 without losing much precision (0.96 → 0.95).

## XGBoost

Wanted to check if a boosting approach (trees built sequentially to fix the previous trees' mistakes) would beat Random Forest's bagging approach (independent trees averaged together) on this kind of imbalanced problem.

Class imbalance handling is different here. No `class_weight='balanced'` option, so I had to compute `scale_pos_weight` manually (ratio of legit to fraud counts in the training set, came out to ~577.3) and pass it in directly.

At matched recall (0.80), XGBoost is slightly ahead: 3 false positives vs Random Forest's 4, PR-AUC 0.877 vs 0.86. But false negatives are exactly the same (20) for both, so on the number that actually costs the bank money (missed frauds), there's no difference. The gap is only in precision at the margin.

One thing that initially looked odd is XGBoost's tuned threshold came out to 0.986, nowhere close to Random Forest's 0.41. Turns out this isn't XGBoost being "more sure" of anything, it's just that boosted trees combined with `scale_pos_weight` push probabilities toward the extremes, while Random Forest's averaging across 100 trees keeps its probabilities more spread out. The two thresholds aren't measuring the same thing, which is exactly why I compared both models at matched recall instead of matched threshold.

## Isolation Forest

Wanted to test an unsupervised approach too. No labels used during training (`.fit(X_train)` only, no `y_train`). `contamination` was set to the actual fraud rate in the training data (`y_train.mean()`), which is a bit of a shortcut since a real unsupervised setup wouldn't have that number available, but it gives the model its best possible shot for this comparison.

Results were clearly worse: 0.31 precision, 0.35 recall. Makes sense once you think about what the model is actually doing. It flags points that are geometrically easy to isolate from the rest of the data. Since fraud here has a real learnable relationship to the labels, it isn't just generic statistical weirdness, so this result is basically confirmation that using the labels (i.e., supervised learning) was the right call.

## Why Random Forest, not XGBoost

XGBoost is marginally better on paper but the gap is small. One fewer false positive out of ~57K transactions, and a PR-AUC difference of about 0.017. That's not enough on its own to justify swapping out a model that's already deployed, tested end-to-end, and has a threshold I can explain in one sentence. Adding XGBoost as a dependency and re-validating the whole pipeline against a new model file is real work for a marginal gain.

If false positives turn out to matter more in practice, or there's a specific reason to prioritize that extra bit of precision, XGBoost is the documented next option and since the model is decoupled from the API code, swapping it in later isn't a big lift.
