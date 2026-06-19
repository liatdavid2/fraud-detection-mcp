from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class FraudScorer:
    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "Trained model artifacts not found. Run: python scripts/train_model.py"
            )
        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.feature_columns: list[str] = self.metadata["feature_columns"]
        self.model_version = self.metadata.get("model_version", "local")

    def score(self, application: dict[str, Any]) -> dict[str, Any]:
        row = {col: application.get(col) for col in self.feature_columns}
        X = pd.DataFrame([row], columns=self.feature_columns)
        probability = float(self.model.predict_proba(X)[0][1])
        return {
            "application_id": str(application.get("application_id")),
            "fraud_probability": probability,
            "risk_band": self.risk_band(probability),
            "top_reasons": self._rule_reasons(application, probability),
            "model_version": self.model_version,
        }

    @staticmethod
    def risk_band(probability: float) -> str:
        if probability >= 0.90:
            return "CRITICAL"
        if probability >= 0.70:
            return "HIGH"
        if probability >= 0.40:
            return "MEDIUM"
        return "LOW"

    def _rule_reasons(self, application: dict[str, Any], probability: float) -> list[str]:
        """Human-readable reasons for the agent.

        This is not SHAP yet. It is intentionally simple for the first version.
        The model is still real; these reasons provide analyst-friendly context.
        """
        reasons: list[str] = []
        numeric_checks = [
            ("income", "Income value is unusual relative to the learned fraud pattern"),
            ("name_email_similarity", "Name/email similarity indicator contributes to identity-risk context"),
            ("velocity_6h", "High short-term velocity can indicate suspicious repeated attempts"),
            ("velocity_24h", "High daily velocity can indicate suspicious repeated attempts"),
            ("customer_age", "Customer age is part of the account-opening risk profile"),
            ("proposed_credit_limit", "Requested credit limit contributes to the risk profile"),
        ]
        for key, text in numeric_checks:
            value = application.get(key)
            if value is not None:
                try:
                    float(value)
                    reasons.append(text)
                except (TypeError, ValueError):
                    pass
            if len(reasons) >= 4:
                break

        if probability >= 0.70:
            reasons.insert(0, "The trained fraud model assigns a high probability to this application")
        elif probability >= 0.40:
            reasons.insert(0, "The trained fraud model assigns a medium probability to this application")
        else:
            reasons.insert(0, "The trained fraud model assigns a low probability to this application")

        return reasons[:5]
