from __future__ import annotations

import argparse
import json
import shutil
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
from sklearn.base import clone
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

TOP_K_FRACTIONS = {
    "0_5pct": 0.005,
    "1pct": 0.01,
    "5pct": 0.05,
    "10pct": 0.10,
}

TARGET_PRECISIONS = {
    "10pct": 0.10,
    "20pct": 0.20,
    "30pct": 0.30,
}


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Train and compare XGBoost and LightGBM fraud models with MLflow tracking."
    )

    parser.add_argument("--dataset", type=Path, default=settings.dataset_path)
    parser.add_argument("--model-path", type=Path, default=settings.model_path)
    parser.add_argument("--metadata-path", type=Path, default=settings.model_metadata_path)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--models",
        choices=["all", "xgboost", "lightgbm"],
        default="all",
        help="Which model family to train.",
    )
    parser.add_argument(
        "--selection-metric",
        default="average_precision_pr_auc",
        help="Metric used to choose the final saved model.",
    )

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)

    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, numeric_features, categorical_features


def precision_recall_at_fraction(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float,
) -> tuple[float, float, int]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    k = max(1, int(np.ceil(len(y_true) * fraction)))
    top_idx = np.argsort(y_score)[::-1][:k]
    positives_in_top_k = int(y_true[top_idx].sum())
    total_positives = int(y_true.sum())

    precision = positives_in_top_k / k
    recall = positives_in_top_k / total_positives if total_positives > 0 else 0.0

    return float(precision), float(recall), int(k)


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fn": int(fn),
        f"{prefix}_flagged_count": int(tp + fp),
        f"{prefix}_flagged_rate": float((tp + fp) / len(y_true)),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) == 0:
        return threshold_metrics(y_true, y_score, threshold=0.5, prefix="best_f1")

    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-12
    )
    best_idx = int(np.nanargmax(f1_scores))
    threshold = float(thresholds[best_idx])

    metrics = threshold_metrics(y_true, y_score, threshold=threshold, prefix="best_f1")
    metrics["best_f1_precision"] = float(precisions[best_idx])
    metrics["best_f1_recall"] = float(recalls[best_idx])
    metrics["best_f1_f1"] = float(f1_scores[best_idx])

    return metrics


def threshold_for_target_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float,
    prefix: str,
) -> dict[str, Any]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    if len(thresholds) == 0:
        return {
            f"{prefix}_target_precision": float(target_precision),
            f"{prefix}_threshold": None,
            f"{prefix}_precision": None,
            f"{prefix}_recall": None,
            f"{prefix}_f1": None,
        }

    candidate_indices = np.where(precisions[:-1] >= target_precision)[0]

    if len(candidate_indices) == 0:
        return {
            f"{prefix}_target_precision": float(target_precision),
            f"{prefix}_threshold": None,
            f"{prefix}_precision": None,
            f"{prefix}_recall": None,
            f"{prefix}_f1": None,
        }

    # Among thresholds that reach the desired precision, choose the one with the highest recall.
    best_idx = int(candidate_indices[np.argmax(recalls[candidate_indices])])
    selected_threshold = float(thresholds[best_idx])

    metrics = threshold_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=selected_threshold,
        prefix=prefix,
    )
    metrics[f"{prefix}_target_precision"] = float(target_precision)

    return metrics


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate_predictions(y_true: pd.Series, y_score: np.ndarray) -> dict[str, Any]:
    y_true_np = y_true.to_numpy(dtype=int)
    positive_count = int(y_true_np.sum())
    negative_count = int(len(y_true_np) - positive_count)
    fraud_rate = float(y_true_np.mean())

    metrics: dict[str, Any] = {
        "test_rows": int(len(y_true_np)),
        "test_positive_count": positive_count,
        "test_negative_count": negative_count,
        "test_fraud_rate": fraud_rate,
        "random_pr_auc_baseline": fraud_rate,
        "roc_auc": safe_roc_auc(y_true_np, y_score),
        "average_precision_pr_auc": float(average_precision_score(y_true_np, y_score)),
    }

    if fraud_rate > 0:
        metrics["pr_auc_lift_vs_random"] = float(
            metrics["average_precision_pr_auc"] / fraud_rate
        )
    else:
        metrics["pr_auc_lift_vs_random"] = None

    for label, fraction in TOP_K_FRACTIONS.items():
        precision, recall, k = precision_recall_at_fraction(y_true_np, y_score, fraction)
        metrics[f"precision_at_{label}"] = precision
        metrics[f"recall_at_{label}"] = recall
        metrics[f"k_at_{label}"] = k

    metrics.update(threshold_metrics(y_true_np, y_score, threshold=0.5, prefix="threshold_0_5"))
    metrics.update(best_f1_threshold(y_true_np, y_score))

    for label, target_precision in TARGET_PRECISIONS.items():
        metrics.update(
            threshold_for_target_precision(
                y_true=y_true_np,
                y_score=y_score,
                target_precision=target_precision,
                prefix=f"target_precision_{label}",
            )
        )

    return metrics


def build_models(args: argparse.Namespace, scale_pos_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {}

    if args.models in {"all", "xgboost"}:
        models["xgboost"] = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=args.min_child_weight,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=-1,
        )

    if args.models in {"all", "lightgbm"}:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_data_in_leaf,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            objective="binary",
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )

    if not models:
        raise ValueError("No models were selected for training.")

    return models


def log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, json.dumps(value, ensure_ascii=False))


def log_numeric_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            value_float = float(value)
            if np.isfinite(value_float):
                mlflow.log_metric(key, value_float)


def train_single_model(
    model_name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    args: argparse.Namespace,
    common_params: dict[str, Any],
) -> dict[str, Any]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", model),
        ]
    )

    run_name = f"baf-{model_name}"

    with mlflow.start_run(run_name=run_name) as run:
        log_params(common_params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_class", model.__class__.__name__)

        pipeline.fit(X_train, y_train)

        y_score = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_score)
        log_numeric_metrics(metrics)

        model_file = (
            args.model_path.parent
            / f"{args.model_path.stem}_{model_name}{args.model_path.suffix}"
        )
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_file)
        mlflow.log_artifact(str(model_file), artifact_path="model_artifacts")

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "model_path": str(model_file),
            "metrics": metrics,
        }


def validate_dataset(df: pd.DataFrame) -> None:
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found.")

    labels = set(df[TARGET].dropna().unique().tolist())
    if not labels.issubset({0, 1, False, True}):
        raise ValueError(
            f"Target column '{TARGET}' must contain binary labels 0/1. Found: {labels}"
        )

    if df[TARGET].nunique(dropna=True) < 2:
        raise ValueError(f"Target column '{TARGET}' must contain both classes.")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


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
    validate_dataset(df)

    y = df[TARGET].astype(int)
    feature_columns = [c for c in df.columns if c not in ID_COLUMNS and c != TARGET]
    X = df[feature_columns]

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    train_positive_count = int((y_train == 1).sum())
    train_negative_count = int((y_train == 0).sum())
    scale_pos_weight = train_negative_count / max(train_positive_count, 1)

    common_params = {
        "target": TARGET,
        "dataset": str(args.dataset),
        "rows": int(len(df)),
        "features": int(len(feature_columns)),
        "numeric_features": int(len(numeric_features)),
        "categorical_features": int(len(categorical_features)),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "models": args.models,
        "selection_metric": args.selection_metric,
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_alpha": float(args.reg_alpha),
        "reg_lambda": float(args.reg_lambda),
        "scale_pos_weight": float(scale_pos_weight),
        "train_positive_count": int(train_positive_count),
        "train_negative_count": int(train_negative_count),
        "train_fraud_rate": float(y_train.mean()),
        "full_dataset_fraud_rate": float(y.mean()),
    }

    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_columns):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print(f"Train positives: {train_positive_count:,}")
    print(f"Train negatives: {train_negative_count:,}")
    print(f"Train class ratio non-fraud/fraud: {scale_pos_weight:.2f}")
    print()

    models = build_models(args, scale_pos_weight)
    results: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        result = train_single_model(
            model_name=model_name,
            model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            args=args,
            common_params=common_params,
        )

        results[model_name] = result

        print(f"{model_name} metrics:")
        print(json.dumps(to_jsonable(result["metrics"]), indent=2))
        print()

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"].get(args.selection_metric) or float("-inf"),
    )
    best_model_path = Path(results[best_model_name]["model_path"])
    best_metrics = results[best_model_name]["metrics"]

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_model_path, args.model_path)

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": args.selection_metric,
        "model_version": results[best_model_name]["run_id"],
        "selected_model_source_path": str(best_model_path),
        "selected_model_output_path": str(args.model_path),
        "recommended_threshold": {
            "type": "best_f1_threshold",
            "value": best_metrics.get("best_f1_threshold"),
            "precision": best_metrics.get("best_f1_precision"),
            "recall": best_metrics.get("best_f1_recall"),
            "f1": best_metrics.get("best_f1_f1"),
        },
        "target": TARGET,
        "id_columns_excluded": sorted(ID_COLUMNS),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dataset": str(args.dataset),
        "class_balance": {
            "full_dataset_fraud_rate": float(y.mean()),
            "train_fraud_rate": float(y_train.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "train_positive_count": train_positive_count,
            "train_negative_count": train_negative_count,
            "scale_pos_weight": float(scale_pos_weight),
        },
        "results": results,
        "best_metrics": best_metrics,
    }

    args.metadata_path.write_text(
        json.dumps(to_jsonable(metadata), indent=2),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="baf-model-selection"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("selection_metric", args.selection_metric)
        mlflow.log_param("selected_model_run_id", results[best_model_name]["run_id"])
        log_numeric_metrics({f"selected_{k}": v for k, v in best_metrics.items()})
        mlflow.log_artifact(str(args.model_path), artifact_path="selected_model")
        mlflow.log_artifact(str(args.metadata_path), artifact_path="selected_model")

    printable_results = {
        name: {
            "model_name": result["model_name"],
            "run_id": result["run_id"],
            "model_path": result["model_path"],
            "metrics": result["metrics"],
        }
        for name, result in results.items()
    }

    print("Training complete.")
    print(f"Selected model: {best_model_name}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Saved selected model: {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")
    print()
    print("All results:")
    print(json.dumps(to_jsonable(printable_results), indent=2))


if __name__ == "__main__":
    main()
