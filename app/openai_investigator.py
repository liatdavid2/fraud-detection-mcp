from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import Settings


class OpenAIInvestigator:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
            raise RuntimeError(
                "OPENAI_API_KEY is required. Create .env from .env.example and set a real key."
            )
        self.model = settings.openai_model
        self.client = OpenAI(api_key=settings.openai_api_key)

    def write_investigation_summary(
        self,
        application: dict[str, Any],
        score: dict[str, Any],
        policy: dict[str, Any],
    ) -> str:
        evidence = {
            "application": self._compact(application),
            "score": score,
            "policy": policy,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior financial fraud analyst. "
                    "Write concise, evidence-based fraud investigation summaries. "
                    "Do not claim facts that are not present in the evidence. "
                    "Use clear sections: Recommendation, Evidence, Risk, Next action."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Investigate this bank account opening fraud case and produce an analyst summary.\n\n"
                    f"Evidence JSON:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or "No investigation summary returned."

    @staticmethod
    def _compact(application: dict[str, Any]) -> dict[str, Any]:
        # Keep the prompt small while preserving useful fields.
        preferred = [
            "application_id",
            "fraud_bool",
            "income",
            "name_email_similarity",
            "prev_address_months_count",
            "current_address_months_count",
            "customer_age",
            "days_since_request",
            "intended_balcon_amount",
            "payment_type",
            "zip_count_4w",
            "velocity_6h",
            "velocity_24h",
            "velocity_4w",
            "bank_branch_count_8w",
            "date_of_birth_distinct_emails_4w",
            "employment_status",
            "credit_risk_score",
            "email_is_free",
            "housing_status",
            "phone_home_valid",
            "phone_mobile_valid",
            "bank_months_count",
            "has_other_cards",
            "proposed_credit_limit",
            "foreign_request",
            "source",
            "session_length_in_minutes",
            "device_os",
            "keep_alive_session",
            "device_distinct_emails_8w",
            "month",
        ]
        compact = {key: application.get(key) for key in preferred if key in application}
        if len(compact) < 8:
            for key, value in application.items():
                if key not in compact and len(compact) < 25:
                    compact[key] = value
        return compact
