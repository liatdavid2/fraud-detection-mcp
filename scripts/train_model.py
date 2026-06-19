from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from app.config import get_settings


TARGET = "fraud_bool"
ID_COLUMNS = {"application_id"}


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Train XGBoost and LightGBM fraud models and log results with MLflow."
    )
    parser.add_argument("--dataset", type=Path, default=settings.dataset_path)
    parser.add_argument("--model-path", type=Path, default=settings.model_path)
    parser.add_argument("--metadata-path", type=Path, default=settings.model_metadata_path)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)

    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    k = max(1, int(len(y_true) * k_fraction))
    top_idx = np.argsort(y_score)[::-1][:k]

    return float(np.mean(y_true[top_idx]))


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocess, numeric_features, categorical_features


def evaluate_predictions(y_true: pd.Series, y_proba: np.ndarray) -> dict[str, Any]:
    y_true_np = y_true.to_numpy(dtype=int)

    y_pred_05 = (y_proba >= 0.5).astype(int)

    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(
        y_true_np,
        y_pred_05,
        labels=[0, 1],
    ).ravel()

    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_proba)

    if len(thresholds) == 0:
        best_threshold = 0.5
        best_precision = float(precision_score(y_true_np, y_pred_05, zero_division=0))
        best_recall = float(recall_score(y_true_np, y_pred_05, zero_division=0))
        best_f1 = float(f1_score(y_true_np, y_pred_05, zero_division=0))
        y_pred_best = y_pred_05
    else:
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-12
        )

        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_precision = float(precisions[best_idx])
        best_recall = float(recalls[best_idx])
        best_f1 = float(f1_scores[best_idx])
        y_pred_best = (y_proba >= best_threshold).astype(int)

    tn_best, fp_best, fn_best, tp_best = confusion_matrix(
        y_true_np,
        y_pred_best,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true_np, y_proba)),
        "average_precision_pr_auc": float(average_precision_score(y_true_np, y_proba)),
        "precision_at_1pct": precision_at_k(y_true_np, y_proba, 0.01),
        "precision_at_5pct": precision_at_k(y_true_np, y_proba, 0.05),

        "precision_threshold_0_5": float(
            precision_score(y_true_np, y_pred_05, zero_division=0)
        ),
        "recall_threshold_0_5": float(
            recall_score(y_true_np, y_pred_05, zero_division=0)
        ),
        "f1_threshold_0_5": float(
            f1_score(y_true_np, y_pred_05, zero_division=0)
        ),

        "best_f1_threshold": best_threshold,
        "best_f1": best_f1,
        "best_f1_precision": best_precision,
        "best_f1_recall": best_recall,

        "tp_threshold_0_5": int(tp_05),
        "fp_threshold_0_5": int(fp_05),
        "tn_threshold_0_5": int(tn_05),
        "fn_threshold_0_5": int(fn_05),

        "tp_best_f1_threshold": int(tp_best),
        "fp_best_f1_threshold": int(fp_best),
        "tn_best_f1_threshold": int(tn_best),
        "fn_best_f1_threshold": int(fn_best),
    }


def build_models(args: argparse.Namespace, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "xgboost": XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def log_numeric_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(key, float(value))


def train_single_model(
    model_name: str,
    model: Any,
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    args: argparse.Namespace,
    common_params: dict[str, Any],
) -> dict[str, Any]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    run_name = f"baf-{model_name}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(common_params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_class", model.__class__.__name__)

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_proba)

        log_numeric_metrics(metrics)

        model_file = args.model_path.parent / f"{args.model_path.stem}_{model_name}{args.model_path.suffix}"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_file)

        mlflow.log_artifact(str(model_file), artifact_path="model_artifacts")

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "model_path": str(model_file),
            "metrics": metrics,
            "pipeline": pipeline,
        }


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}. Run scripts/download_dataset.py first."
        )

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = pd.read_csv(args.dataset)

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found.")

    y = df[TARGET].astype(int)

    feature_columns = [c for c in df.columns if c not in ID_COLUMNS and c != TARGET]
    X = df[feature_columns]

    preprocess, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    common_params = {
        "target": TARGET,
        "dataset": str(args.dataset),
        "rows": len(df),
        "features": len(feature_columns),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "test_size": args.test_size,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "scale_pos_weight": scale_pos_weight,
        "train_positive_count": positive_count,
        "train_negative_count": negative_count,
        "train_fraud_rate": float(y_train.mean()),
        "full_dataset_fraud_rate": float(y.mean()),
    }

    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(df):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print(f"Train class ratio non-fraud/fraud: {scale_pos_weight:.2f}")
    print()

    models = build_models(args, scale_pos_weight)

    results: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        result = train_single_model(
            model_name=model_name,
            model=model,
            preprocess=preprocess,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            args=args,
            common_params=common_params,
        )

        results[model_name] = {
            "model_name": result["model_name"],
            "run_id": result["run_id"],
            "model_path": result["model_path"],
            "metrics": result["metrics"],
        }

        print(f"{model_name} metrics:")
        print(json.dumps(result["metrics"], indent=2))
        print()

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"]["average_precision_pr_auc"],
    )

    best_model_path = Path(results[best_model_name]["model_path"])

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    best_pipeline = joblib.load(best_model_path)
    joblib.dump(best_pipeline, args.model_path)

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "average_precision_pr_auc",
        "model_version": results[best_model_name]["run_id"],
        "target": TARGET,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dataset": str(args.dataset),
        "class_balance": {
            "full_dataset_fraud_rate": float(y.mean()),
            "train_fraud_rate": float(y_train.mean()),
            "train_positive_count": positive_count,
            "train_negative_count": negative_count,
            "scale_pos_weight": float(scale_pos_weight),
        },
        "results": results,
        "best_metrics": results[best_model_name]["metrics"],
    }

    args.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="baf-model-selection"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("selection_metric", "average_precision_pr_auc")
        mlflow.log_metric(
            "selected_average_precision_pr_auc",
            results[best_model_name]["metrics"]["average_precision_pr_auc"],
        )
        mlflow.log_metric(
            "selected_roc_auc",
            results[best_model_name]["metrics"]["roc_auc"],
        )
        mlflow.log_artifact(str(args.model_path), artifact_path="selected_model")
        mlflow.log_artifact(str(args.metadata_path), artifact_path="selected_model")

    print("Training complete.")
    print(f"Selected model: {best_model_name}")
    print(f"Selection metric: average_precision_pr_auc")
    print(f"Saved selected model: {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")
    print()
    print("All results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()