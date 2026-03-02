# Leakage-Safe Benchmark Summary

- Best model (holdout F1): **random_forest**
- Samples: total **3204**, holdout **641**

| Model | Holdout Acc | Holdout Prec | Holdout Recall | Holdout F1 | Holdout ROC-AUC | CV Train F1 | CV Val F1 | F1 Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest | 0.9828 | 0.9946 | 0.9763 | 0.9854 | 0.9972 | 0.9870 | 0.9814 | 0.0057 |
| decision_tree | 0.9828 | 1.0000 | 0.9711 | 0.9853 | 0.9972 | 0.9809 | 0.9763 | 0.0046 |
| logistic_regression | 0.9704 | 0.9839 | 0.9658 | 0.9748 | 0.9953 | 0.9718 | 0.9683 | 0.0036 |