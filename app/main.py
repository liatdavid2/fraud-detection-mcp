from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.service_factory import create_services


app = FastAPI(
    title="Fraud Detection MCP",
    description="OpenAI + LangGraph + MCP-style fraud investigation platform with SQLite in-memory and MLflow.",
    version="0.1.0",
)


class SafeSQLRequest(BaseModel):
    sql: str
    limit: int = 100


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Fraud Detection MCP</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.5; }
          code, pre { background: #f5f5f5; padding: 2px 5px; border-radius: 4px; }
          li { margin-bottom: 8px; }
        </style>
      </head>
      <body>
        <h1>Fraud Detection MCP</h1>
        <p>OpenAI + LangGraph + SQLite in-memory + MLflow + MCP tools.</p>
        <h2>Try endpoints</h2>
        <ul>
          <li><a href="/health">/health</a></li>
          <li><a href="/applications/high-risk?limit=10">/applications/high-risk?limit=10</a></li>
          <li><code>POST /applications/{application_id}/investigate</code></li>
          <li><a href="/cases">/cases</a></li>
          <li><a href="/audit">/audit</a></li>
        </ul>
        <p>Example:</p>
        <pre>curl -X POST http://127.0.0.1:8000/applications/10/investigate</pre>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, Any]:
    services = _services_or_500()
    return {
        "status": "ok",
        "dataset_path": str(services.settings.dataset_path),
        "model_path": str(services.settings.model_path),
        "openai_model": services.settings.openai_model,
    }


@app.get("/applications/high-risk")
def list_high_risk_applications(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    services = _services_or_500()
    return services.tools.list_high_risk_applications(limit=limit)


@app.get("/applications/{application_id}")
def get_application(application_id: str) -> dict[str, Any]:
    services = _services_or_500()
    try:
        return services.tools.get_application(application_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/applications/{application_id}/score")
def score_application(application_id: str) -> dict[str, Any]:
    services = _services_or_500()
    try:
        return services.tools.score_application(application_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/applications/{application_id}/investigate")
def investigate_application(application_id: str) -> dict[str, Any]:
    services = _services_or_500()
    try:
        result = services.graph.invoke(application_id)
        return dict(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/cases")
def get_cases(status: str = "OPEN", limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    services = _services_or_500()
    return services.tools.get_review_cases(status=status, limit=limit)


@app.get("/audit")
def get_audit(limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
    services = _services_or_500()
    return services.tools.get_audit_log(limit=limit)


@app.post("/sql/select")
def safe_select(request: SafeSQLRequest) -> list[dict[str, Any]]:
    services = _services_or_500()
    try:
        return services.tools.safe_select_query(request.sql, limit=request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _services_or_500():
    try:
        return create_services()
    except Exception as exc:  # startup depends on dataset, model, OpenAI key
        raise HTTPException(status_code=500, detail=str(exc)) from exc
