from __future__ import annotations

import json
from typing import Any

from app.scoring import FraudScorer
from app.sqlite_store import FraudSQLiteStore


class FraudTools:
    def __init__(self, store: FraudSQLiteStore, scorer: FraudScorer):
        self.store = store
        self.scorer = scorer

    def get_application(self, application_id: str) -> dict[str, Any]:
        application = self.store.query_one(
            """
            SELECT *
            FROM applications
            WHERE application_id = ?
            """,
            (str(application_id),),
        )
        if not application:
            raise ValueError(f"Application {application_id} not found")
        self.store.insert_audit("fraud_tools", "get_application", {"application_id": application_id})
        return application

    def list_high_risk_applications(self, limit: int = 20) -> list[dict[str, Any]]:
        # If scores exist, use them. Otherwise list true fraud rows first for exploration.
        scored = self.store.query_many(
            """
            SELECT a.application_id, a.fraud_bool, s.fraud_probability, s.risk_band, s.top_reasons
            FROM model_scores s
            JOIN applications a ON a.application_id = s.application_id
            ORDER BY s.fraud_probability DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        if scored:
            return scored
        return self.store.query_many(
            """
            SELECT *
            FROM applications
            ORDER BY fraud_bool DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def score_application(self, application_id: str) -> dict[str, Any]:
        application = self.get_application(application_id)
        score = self.scorer.score(application)
        self.store.execute(
            """
            INSERT OR REPLACE INTO model_scores(
                application_id,
                fraud_probability,
                risk_band,
                top_reasons,
                model_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(application_id),
                float(score["fraud_probability"]),
                score["risk_band"],
                json.dumps(score["top_reasons"], ensure_ascii=False),
                score.get("model_version", "local"),
            ),
        )
        self.store.insert_audit("fraud_tools", "score_application", score)
        return score

    def get_policy_decision(self, fraud_probability: float, amount: float | None = None) -> dict[str, Any]:
        score = float(fraud_probability)
        if score >= 0.90:
            decision = {
                "decision": "HUMAN_REVIEW_REQUIRED",
                "priority": "CRITICAL",
                "should_create_case": True,
                "auto_decline_enabled": False,
                "reason": "Very high model probability. Demo policy requires human approval, not auto-decline.",
            }
        elif score >= 0.70:
            decision = {
                "decision": "HUMAN_REVIEW_REQUIRED",
                "priority": "HIGH",
                "should_create_case": True,
                "auto_decline_enabled": False,
                "reason": "High model probability. Route to analyst review.",
            }
        elif score >= 0.40:
            decision = {
                "decision": "MONITOR",
                "priority": "MEDIUM",
                "should_create_case": False,
                "auto_decline_enabled": False,
                "reason": "Medium model probability. Monitor unless other signals are present.",
            }
        else:
            decision = {
                "decision": "APPROVE_SIMULATION",
                "priority": "LOW",
                "should_create_case": False,
                "auto_decline_enabled": False,
                "reason": "Low model probability under the demo policy.",
            }
        if amount is not None:
            decision["amount"] = amount
        self.store.insert_audit("policy", "policy_decision", decision)
        return decision

    def create_review_case(
        self,
        application_id: str,
        priority: str,
        reason: str,
        analyst_summary: str | None = None,
    ) -> dict[str, Any]:
        case_id = f"CASE-{application_id}"
        self.store.execute(
            """
            INSERT OR REPLACE INTO review_cases(
                case_id,
                application_id,
                priority,
                status,
                reason,
                analyst_summary,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (case_id, str(application_id), priority, "OPEN", reason, analyst_summary),
        )
        case = {
            "case_id": case_id,
            "application_id": str(application_id),
            "priority": priority,
            "status": "OPEN",
            "reason": reason,
            "analyst_summary": analyst_summary,
        }
        self.store.insert_audit("fraud_tools", "create_review_case", case)
        return case

    def get_review_cases(self, status: str = "OPEN", limit: int = 50) -> list[dict[str, Any]]:
        return self.store.query_many(
            """
            SELECT *
            FROM review_cases
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, int(limit)),
        )

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.query_many(
            """
            SELECT *
            FROM audit_log
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def safe_select_query(self, sql: str, limit: int = 100) -> list[dict[str, Any]]:
        result = self.store.safe_select_query(sql, limit=limit)
        self.store.insert_audit("fraud_tools", "safe_select_query", {"sql": sql, "limit": limit})
        return result
