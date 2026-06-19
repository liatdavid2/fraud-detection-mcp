from __future__ import annotations

from typing import Any, TypedDict


class FraudInvestigationState(TypedDict, total=False):
    application_id: str
    application: dict[str, Any]
    score: dict[str, Any]
    policy: dict[str, Any]
    investigation_summary: str
    review_case: dict[str, Any] | None
    errors: list[str]
