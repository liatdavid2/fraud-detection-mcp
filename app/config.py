from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    dataset_path: Path
    model_path: Path
    model_metadata_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    app_host: str
    app_port: int


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        dataset_path=Path(os.getenv("FRAUD_DATASET_PATH", "data/processed/baf_base_sample.csv")),
        model_path=Path(os.getenv("FRAUD_MODEL_PATH", "models/fraud_model.joblib")),
        model_metadata_path=Path(os.getenv("FRAUD_MODEL_METADATA_PATH", "models/fraud_model_metadata.json")),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-mcp"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
    )
