#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'Scripts' / 'overview_data.csv'
REPORTS_DIR = ROOT / 'reports'
MODELS_DIR = ROOT / 'models'

def load_data(path: Path):
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    y = df['is_cheater'].astype(int)
    X = df.drop(columns=['is_cheater']).apply(pd.to_numeric, errors='coerce')
    return X, y

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=20, class_weight='balanced')),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    metrics = {
        'samples_total': int(len(y)),
        'samples_train': int(len(y_train)),
        'samples_test': int(len(y_test)),
        'class_balance': {'cheater': int(y.sum()), 'non_cheater': int((1 - y).sum())},
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
    }
    model_path = MODELS_DIR / 'decision_tree_overview.pkl'
    metrics_path = REPORTS_DIR / 'metrics.json'
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
