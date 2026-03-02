# R6 Cheater Detector

Binary ML classifier that predicts whether a Rainbow Six Siege account is likely cheating based on public stat profiles.

## Project status
This repository has been cleaned for public presentation and includes a reproducible training/evaluation script.

## Repository layout
- `Scripts/train_and_evaluate.py` - reproducible model training + evaluation
- `Scripts/overview_data.csv` - labeled dataset used for training (`is_cheater` target)
- `Scripts/detect_cheater.py` - existing live profile inference script (tracker.gg scraping)
- `reports/metrics.json` - latest benchmark output (generated)
- `models/decision_tree_overview.pkl` - latest trained model (generated)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn joblib
python Scripts/train_and_evaluate.py
```

## Current benchmark
The latest run uses an 80/20 stratified split (`random_state=42`) on `Scripts/overview_data.csv`.

- Accuracy: **0.9750**
- Precision: **0.9840**
- Recall: **0.9737**
- F1: **0.9788**
- ROC-AUC: **0.9974**

## Notes
- This is a statistical risk model, not a ban decision engine.
- False positives are possible.
- Data quality and labeling quality directly affect performance.
- Please avoid using this model for harassment or automated punitive action.
