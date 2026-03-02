# Model Benchmark Summary

Run date: 2026-03-03
Dataset: `Scripts/overview_data.csv`
Split: 80/20 stratified (`random_state=42`)

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9828 | 0.9843 | 0.9868 | **0.9855** | 0.9937 |
| Random Forest | 0.9828 | **0.9946** | 0.9763 | 0.9854 | **0.9978** |
| Decision Tree | 0.9750 | 0.9840 | 0.9737 | 0.9788 | 0.9974 |

## Winner
By F1 score, **Logistic Regression** is currently the top model on this split.

## Notes
- Random Forest has strongest precision and ROC-AUC, with slightly lower recall than Logistic Regression.
- Logistic Regression reported a convergence warning in this run; scaling + solver tuning may improve stability.
