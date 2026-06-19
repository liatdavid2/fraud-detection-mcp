from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class FraudSQLiteStore:
    """SQLite in-memory store loaded from the real BAF CSV.

    The database exists only while this Python process is alive. This gives us
    SQL semantics without requiring PostgreSQL or Docker for the first version.
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}. "
                "Run: python scripts/download_dataset.py"
            )

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._load_applications()
        self._create_operational_tables()

    def _load_applications(self) -> None:
        df = pd.read_csv(self.dataset_path)
        if "fraud_bool" not in df.columns:
            raise ValueError("Expected BAF label column 'fraud_bool' was not found.")

        if "application_id" not in df.columns:
            df.insert(0, "application_id", [str(i) for i in range(len(df))])
        else:
            df["application_id"] = df["application_id"].astype(str)

        # Normalize pandas NA values for SQLite.
        df = df.where(pd.notnull(df), None)
        df.to_sql("applications", self.conn, if_exists="replace", index=False)

        self.conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_applications_application_id
            ON applications(application_id);

            CREATE INDEX IF NOT EXISTS idx_applications_fraud_bool
            ON applications(fraud_bool);
            """
        )
        self.conn.commit()

    def _create_operational_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_scores (
                application_id TEXT PRIMARY KEY,
                fraud_probability REAL NOT NULL,
                risk_band TEXT NOT NULL,
                top_reasons TEXT,
                model_version TEXT,
                scored_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS review_cases (
                case_id TEXT PRIMARY KEY,
                application_id TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                analyst_summary TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return self._row_to_dict(row) if row else None

    def query_many(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def insert_audit(self, actor: str, action: str, details: dict[str, Any] | str) -> None:
        details_text = details if isinstance(details, str) else json.dumps(details, ensure_ascii=False)
        self.execute(
            """
            INSERT INTO audit_log(actor, action, details)
            VALUES (?, ?, ?)
            """,
            (actor, action, details_text),
        )

    def table_columns(self, table_name: str) -> list[str]:
        rows = self.query_many(f"PRAGMA table_info({table_name})")
        return [row["name"] for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    def safe_select_query(self, sql: str, limit: int = 100) -> list[dict[str, Any]]:
        """Allow read-only SQL for demos while blocking writes.

        This is useful because the project intentionally talks SQL, but we do
        not want an LLM or user to execute arbitrary mutations.
        """
        normalized = sql.strip().lower()
        blocked_tokens = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma"]
        if not normalized.startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")
        if any(token in normalized for token in blocked_tokens):
            raise ValueError("Unsafe SQL keyword detected.")
        if " limit " not in normalized:
            sql = f"{sql.rstrip(';')} LIMIT {int(limit)}"
        return self.query_many(sql)
