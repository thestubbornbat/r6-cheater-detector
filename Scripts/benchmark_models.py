#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Scripts" / "overview_data.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.2


def load_data(path: Path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    y = df["is_cheater"].astype(int)
    X = df.drop(columns=["is_cheater"]).apply(pd.to_numeric, errors="coerce")
    return X, y


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }, pipeline


def save_comparison_plot(results_sorted: list[dict], out_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = [r["model"] for r in results_sorted]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results_sorted):
        vals = [r[m] for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width=width, label=r["model"])

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Holdout test-set performance comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_roc_plot(results_sorted: list[dict], y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results_sorted:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on holdout test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_confusion_matrices(results_sorted: list[dict], y_test) -> None:
    for r in results_sorted:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            r["y_pred"],
            display_labels=["non-cheater", "cheater"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"Confusion matrix: {r['model']}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{r['model']}.png", dpi=180)
        plt.close(fig)


def save_feature_plot(best_name: str, best_model: Pipeline, feature_names: pd.Index):
    estimator = best_model.named_steps["model"]
    fig, ax = plt.subplots(figsize=(8, 5))

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        kind = "importance"
    else:
        importances = np.abs(estimator.coef_[0])
        kind = "|coefficient|"

    top_n = min(12, len(importances))
    order = np.argsort(importances)[-top_n:]
    ax.barh(np.array(feature_names)[order], importances[order])
    ax.set_title(f"Top {top_n} features for {best_name} ({kind})")
    ax.set_xlabel(kind)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_features_best_model.png", dpi=180)
    plt.close(fig)


def write_summary_markdown(summary: dict) -> None:
    lines = [
        "# Holdout Benchmark Summary",
        "",
        f"- Dataset: **{summary['dataset']['samples_total']}** samples",
        f"- Holdout test set: **{summary['dataset']['samples_test']}** samples",
        f"- Split: stratified train/test ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}), random_state={SEED}",
        f"- Best model by F1: **{summary['best_model']}**",
        "",
        "## Model performance (holdout test set)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |"
        )

    lines += [
        "",
        "## Generated visuals",
        "",
        "- `reports/figures/model_metric_comparison.png`",
        "- `reports/figures/roc_curves.png`",
        "- `reports/figures/confusion_matrix_decision_tree.png`",
        "- `reports/figures/confusion_matrix_random_forest.png`",
        "- `reports/figures/confusion_matrix_logistic_regression.png`",
        "- `reports/figures/top_features_best_model.png`",
    ]

    (REPORTS_DIR / "benchmark_summary.md").write_text("\n".join(lines))


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    candidates = {
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    random_state=SEED,
                    max_depth=6,
                    min_samples_leaf=20,
                    class_weight="balanced",
                ),
            ),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=SEED,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]),
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=SEED,
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]),
    }

    results = []
    trained = {}
    predictions = pd.DataFrame({"y_true": y_test.values}, index=y_test.index)

    for name, pipe in candidates.items():
        metrics, fitted = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained[name] = fitted
        predictions[f"pred_{name}"] = metrics["y_pred"]
        predictions[f"proba_{name}"] = metrics["y_proba"]

    results_sorted = sorted(results, key=lambda r: (r["f1"], r["roc_auc"]), reverse=True)
    best_name = results_sorted[0]["model"]

    summary = {
        "dataset": {
            "samples_total": int(len(y)),
            "samples_train": int(len(y_train)),
            "samples_test": int(len(y_test)),
            "class_balance": {"cheater": int(y.sum()), "non_cheater": int((1 - y).sum())},
        },
        "split": {"test_size": TEST_SIZE, "random_state": SEED, "stratified": True, "holdout": True},
        "results": [
            {
                k: v
                for k, v in r.items()
                if k not in {"y_pred", "y_proba"}
            }
            for r in results_sorted
        ],
        "best_model": best_name,
    }

    (REPORTS_DIR / "benchmark_results.json").write_text(json.dumps(summary, indent=2))
    predictions.to_csv(REPORTS_DIR / "holdout_predictions.csv", index=True)

    for name, model in trained.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    save_comparison_plot(results_sorted, FIGURES_DIR / "model_metric_comparison.png")
    save_roc_plot(results_sorted, y_test, FIGURES_DIR / "roc_curves.png")
    save_confusion_matrices(results_sorted, y_test)
    save_feature_plot(best_name, trained[best_name], X.columns)
    write_summary_markdown(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
