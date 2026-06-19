from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import get_settings


TARGET = "fraud_bool"
ID_COLUMNS = {"application_id"}


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train fraud model and log with MLflow.")
    parser.add_argument("--dataset", type=Path, default=settings.dataset_path)
    parser.add_argument("--model-path", type=Path, default=settings.model_path)
    parser.add_argument("--metadata-path", type=Path, default=settings.model_metadata_path)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=10)
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    # sklearn renamed sparse -> sparse_output. This supports modern versions and keeps compatibility.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float = 0.01) -> float:
    k = max(1, int(len(y_true) * k_fraction))
    top_idx = np.argsort(y_score)[-k:]
    return float(np.mean(y_true[top_idx]))


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}. Run scripts/download_dataset.py first.")

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    df = pd.read_csv(args.dataset)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found.")

    y = df[TARGET].astype(int)
    feature_columns = [c for c in df.columns if c not in ID_COLUMNS and c != TARGET]
    X = df[feature_columns]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    with mlflow.start_run(run_name="baf-random-forest") as run:
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics: dict[str, Any] = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision_pr_auc": float(average_precision_score(y_test, y_proba)),
            "precision_at_1pct": precision_at_k(y_test.to_numpy(), y_proba, 0.01),
            "precision_at_5pct": precision_at_k(y_test.to_numpy(), y_proba, 0.05),
            "precision_threshold_0_5": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_threshold_0_5": float(recall_score(y_test, y_pred, zero_division=0)),
            "fraud_rate": float(y.mean()),
        }
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.model_path)

        metadata = {
            "model_version": run.info.run_id,
            "target": TARGET,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "metrics": metrics,
            "dataset": str(args.dataset),
        }
        args.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(args.model_path), artifact_path="model_artifacts")
        mlflow.log_artifact(str(args.metadata_path), artifact_path="model_artifacts")

        print("Training complete.")
        print(f"MLflow run_id: {run.info.run_id}")
        print(f"Saved model: {args.model_path}")
        print(f"Saved metadata: {args.metadata_path}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
