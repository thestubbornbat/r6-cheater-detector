# Holdout Benchmark Summary

- Dataset: **3204** samples
- Holdout test set: **641** samples
- Split: stratified train/test (80/20), random_state=42
- Best model by F1: **random_forest**

## Model performance (holdout test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.9828 | 0.9946 | 0.9763 | 0.9854 | 0.9978 |
| decision_tree | 0.9750 | 0.9840 | 0.9737 | 0.9788 | 0.9974 |
| logistic_regression | 0.9704 | 0.9813 | 0.9684 | 0.9748 | 0.9957 |

## Generated visuals

- `reports/figures/model_metric_comparison.png`
- `reports/figures/roc_curves.png`
- `reports/figures/confusion_matrix_decision_tree.png`
- `reports/figures/confusion_matrix_random_forest.png`
- `reports/figures/confusion_matrix_logistic_regression.png`
- `reports/figures/top_features_best_model.png`