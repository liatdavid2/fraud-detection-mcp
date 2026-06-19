# Fraud Detection MCP

AI-powered fraud detection and investigation platform using a real financial fraud dataset, SQLite in-memory SQL, MLflow model tracking, MCP tools, and LangGraph agents.

The project is intentionally built like a production-style AI/ML system rather than a notebook:

- **Real dataset**: Bank Account Fraud Dataset Suite from Kaggle / NeurIPS 2022.
- **SQL layer**: CSV is loaded into **SQLite in-memory**, so all fraud tools query SQL tables.
- **MLflow**: model training logs metrics, parameters, artifacts, and the trained model.
- **OpenAI**: investigation summaries require `OPENAI_API_KEY` from `.env`.
- **LangGraph**: orchestrates the fraud investigation workflow.
- **MCP**: exposes fraud tools as a Model Context Protocol server.
- **FastAPI**: serves investigation endpoints and a small dashboard page.

---

## Architecture

```text
BAF Kaggle Dataset
  ↓
scripts/download_dataset.py
  ↓
data/processed/baf_base_sample.csv
  ↓
scripts/train_model.py
  ↓
MLflow run + models/fraud_model.joblib
  ↓
FastAPI app
  ↓
SQLite :memory: SQL database
  ↓
Fraud tools
  ↓
LangGraph Fraud Agent
  ↓
OpenAI investigation summary

MCP Server exposes the same fraud tools to MCP-compatible clients.
```

---

## 1. Create and activate venv

Windows:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Mac / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Configure OpenAI key

Create `.env` from the example:

```bash
copy .env.example .env
```

Mac / Linux:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4.1-mini
```

Do not commit `.env` to GitHub.

---

## 3. Download the real BAF dataset

This uses KaggleHub, not a fake dataset.

```bash
python scripts/download_dataset.py
```

It downloads:

```text
sgpjesus/bank-account-fraud-dataset-neurips-2022
```

Then it copies the base CSV and creates:

```text
data/processed/baf_base_sample.csv
```

By default it samples 50,000 rows for fast local development. You can change `--sample-size`.

Example:

```bash
python scripts/download_dataset.py --sample-size 100000
```

---

## 4. Train the model with MLflow

```bash
python scripts/train_model.py
```

Outputs:

```text
models/fraud_model.joblib
models/fraud_model_metadata.json
mlruns/
```

Open MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 5. Run the FastAPI app

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```text
GET  /health
GET  /applications/high-risk?limit=10
GET  /applications/{application_id}
POST /applications/{application_id}/score
POST /applications/{application_id}/investigate
GET  /cases
GET  /audit
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/applications/10/investigate
```

---

## 6. Run the MCP server

```bash
python -m mcp_servers.fraud_mcp_server
```

The MCP server exposes tools such as:

```text
get_application
list_high_risk_applications
score_application
investigate_application
create_review_case
get_review_cases
get_audit_log
safe_select_query
```

---

## LangGraph workflow

```text
START
  ↓
load_application
  ↓
score_application
  ↓
policy_decision
  ↓
llm_investigation
  ↓
maybe_create_review_case
  ↓
END
```

The agent uses deterministic tools for data access and scoring, then calls OpenAI to produce an analyst-style investigation summary based on evidence.

---

## Why SQLite in-memory?

The first version uses SQLite `:memory:` to provide a real SQL query layer without requiring PostgreSQL or Docker. The storage layer is abstracted so it can later be replaced with PostgreSQL while keeping the MCP tools and LangGraph agent almost unchanged.

---

## Suggested GitHub description

```text
Fraud detection and investigation platform using OpenAI, LangGraph, MCP tools, MLflow, SQLite in-memory, and the Bank Account Fraud Dataset.
```

Suggested topics:

```text
fraud-detection, mcp, ai-agents, langgraph, openai, mlflow, sqlite, fastapi, machine-learning, financial-ai
```
