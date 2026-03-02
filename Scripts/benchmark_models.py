#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba))
    else:
        auc = None

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )

    return {
        'model': name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': auc,
    }, pipeline


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'decision_tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=20, class_weight='balanced')),
        ]),
        'random_forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight='balanced_subsample',
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ]),
        'logistic_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=2000,
                n_jobs=-1,
            )),
        ]),
    }

    results = []
    trained = {}
    for name, pipe in candidates.items():
        metrics, fitted = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained[name] = fitted

    results_sorted = sorted(results, key=lambda r: (r['f1'], r['roc_auc'] or 0), reverse=True)
    best_name = results_sorted[0]['model']

    summary = {
        'dataset': {
            'samples_total': int(len(y)),
            'samples_train': int(len(y_train)),
            'samples_test': int(len(y_test)),
            'class_balance': {'cheater': int(y.sum()), 'non_cheater': int((1 - y).sum())},
        },
        'split': {'test_size': 0.2, 'random_state': 42, 'stratified': True},
        'results': results_sorted,
        'best_model': best_name,
    }

    (REPORTS_DIR / 'benchmark_results.json').write_text(json.dumps(summary, indent=2))
    for name, model in trained.items():
        import joblib
        joblib.dump(model, MODELS_DIR / f'{name}.pkl')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
