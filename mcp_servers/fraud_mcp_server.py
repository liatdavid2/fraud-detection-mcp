from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.service_factory import create_services


mcp = FastMCP("fraud-detection-mcp")


@mcp.tool()
def get_application(application_id: str) -> dict[str, Any]:
    """Return one bank account opening application by application_id."""
    services = create_services()
    return services.tools.get_application(application_id)


@mcp.tool()
def list_high_risk_applications(limit: int = 20) -> list[dict[str, Any]]:
    """List high-risk applications. If no model scores exist yet, fraud-labelled rows are shown first."""
    services = create_services()
    return services.tools.list_high_risk_applications(limit=limit)


@mcp.tool()
def score_application(application_id: str) -> dict[str, Any]:
    """Run the trained fraud model and persist the score in SQLite."""
    services = create_services()
    return services.tools.score_application(application_id)


@mcp.tool()
def investigate_application(application_id: str) -> dict[str, Any]:
    """Run the LangGraph fraud investigation workflow, including OpenAI summary."""
    services = create_services()
    return dict(services.graph.invoke(application_id))


@mcp.tool()
def create_review_case(application_id: str, priority: str, reason: str) -> dict[str, Any]:
    """Create or update a human-review fraud case."""
    services = create_services()
    return services.tools.create_review_case(
        application_id=application_id,
        priority=priority,
        reason=reason,
        analyst_summary=None,
    )


@mcp.tool()
def get_review_cases(status: str = "OPEN", limit: int = 50) -> list[dict[str, Any]]:
    """Return review cases from the in-memory SQLite operational table."""
    services = create_services()
    return services.tools.get_review_cases(status=status, limit=limit)


@mcp.tool()
def get_audit_log(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent audit events from the in-memory SQLite audit log."""
    services = create_services()
    return services.tools.get_audit_log(limit=limit)


@mcp.tool()
def safe_select_query(sql: str, limit: int = 100) -> list[dict[str, Any]]:
    """Run a read-only SELECT query against the in-memory SQLite database."""
    services = create_services()
    return services.tools.safe_select_query(sql=sql, limit=limit)


if __name__ == "__main__":
    mcp.run()
