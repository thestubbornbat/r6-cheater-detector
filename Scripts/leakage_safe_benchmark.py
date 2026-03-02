#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'Scripts' / 'overview_data.csv'
REPORTS_DIR = ROOT / 'reports'
MODELS_DIR = ROOT / 'models'

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

LEAKY_OR_UNSTABLE_COLUMNS = {
    'Unnamed: 0',  # row index artifact
}


def load_data(path: Path):
    df = pd.read_csv(path)

    # 1) Remove obvious non-feature columns
    drop_cols = [c for c in LEAKY_OR_UNSTABLE_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2) Remove exact duplicate rows to reduce memorization risk
    df = df.drop_duplicates().reset_index(drop=True)

    y = df['is_cheater'].astype(int)
    X = df.drop(columns=['is_cheater']).apply(pd.to_numeric, errors='coerce')
    return X, y


def build_models():
    return {
        'decision_tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', DecisionTreeClassifier(
                random_state=SEED,
                max_depth=4,
                min_samples_leaf=30,
                class_weight='balanced',
            )),
        ]),
        'random_forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(
                n_estimators=250,
                random_state=SEED,
                class_weight='balanced_subsample',
                n_jobs=-1,
                max_depth=8,
                min_samples_leaf=5,
                max_features='sqrt',
            )),
        ]),
        'logistic_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=SEED,
                class_weight='balanced',
                C=0.5,
                max_iter=5000,
            )),
        ]),
    }


def summarize_scores(cv_results: dict, prefix: str):
    out = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        key = f'{prefix}_{metric}'
        vals = cv_results[key]
        out[metric] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }
    return out


def evaluate_holdout(model: Pipeline, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
    }, model


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    rows = []
    trained = {}

    for name, model in build_models().items():
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        holdout_metrics, fitted = evaluate_holdout(model, X_train, y_train, X_test, y_test)
        trained[name] = fitted

        train_cv = summarize_scores(cv_results, 'train')
        val_cv = summarize_scores(cv_results, 'test')

        row = {
            'model': name,
            'holdout': holdout_metrics,
            'cv_train': train_cv,
            'cv_validation': val_cv,
            'overfit_gap_f1': float(train_cv['f1']['mean'] - val_cv['f1']['mean']),
            'overfit_gap_auc': float(train_cv['roc_auc']['mean'] - val_cv['roc_auc']['mean']),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r['holdout']['f1'], r['holdout']['roc_auc']), reverse=True)

    summary = {
        'dataset': {
            'samples_total': int(len(y)),
            'samples_train': int(len(y_train)),
            'samples_holdout': int(len(y_test)),
            'class_balance': {
                'cheater': int(y.sum()),
                'non_cheater': int((1 - y).sum()),
            },
            'deduplicated': True,
        },
        'protocol': {
            'split': 'stratified train/holdout',
            'holdout_test_size': TEST_SIZE,
            'random_state': SEED,
            'cross_validation': f'{CV_FOLDS}-fold stratified on train only',
            'leakage_controls': [
                'removed non-feature index columns',
                'dropped exact duplicate rows before split',
                'final holdout remained untouched until final evaluation',
                'reported train-vs-validation CV gaps',
            ],
        },
        'results': rows,
        'best_model': rows[0]['model'],
    }

    (REPORTS_DIR / 'benchmark_results_leakage_safe.json').write_text(json.dumps(summary, indent=2))

    md = [
        '# Leakage-Safe Benchmark Summary',
        '',
        f"- Best model (holdout F1): **{summary['best_model']}**",
        f"- Samples: total **{summary['dataset']['samples_total']}**, holdout **{summary['dataset']['samples_holdout']}**",
        '',
        '| Model | Holdout Acc | Holdout Prec | Holdout Recall | Holdout F1 | Holdout ROC-AUC | CV Train F1 | CV Val F1 | F1 Gap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['holdout']['accuracy']:.4f} | {r['holdout']['precision']:.4f} | {r['holdout']['recall']:.4f} | {r['holdout']['f1']:.4f} | {r['holdout']['roc_auc']:.4f} | {r['cv_train']['f1']['mean']:.4f} | {r['cv_validation']['f1']['mean']:.4f} | {r['overfit_gap_f1']:.4f} |"
        )

    (REPORTS_DIR / 'benchmark_summary_leakage_safe.md').write_text('\n'.join(md))

    for name, model in trained.items():
        joblib.dump(model, MODELS_DIR / f'{name}_leakage_safe.pkl')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
