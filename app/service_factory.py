from __future__ import annotations

from dataclasses import dataclass

from agents.graph import FraudInvestigationGraph
from app.config import Settings, get_settings
from app.fraud_tools import FraudTools
from app.openai_investigator import OpenAIInvestigator
from app.scoring import FraudScorer
from app.sqlite_store import FraudSQLiteStore


@dataclass
class Services:
    settings: Settings
    store: FraudSQLiteStore
    scorer: FraudScorer
    tools: FraudTools
    graph: FraudInvestigationGraph


_services: Services | None = None


def create_services(force_reload: bool = False) -> Services:
    global _services
    if _services is not None and not force_reload:
        return _services

    settings = get_settings()
    store = FraudSQLiteStore(settings.dataset_path)
    scorer = FraudScorer(settings.model_path, settings.model_metadata_path)
    tools = FraudTools(store=store, scorer=scorer)
    investigator = OpenAIInvestigator(settings)
    graph = FraudInvestigationGraph(tools=tools, investigator=investigator)
    _services = Services(
        settings=settings,
        store=store,
        scorer=scorer,
        tools=tools,
        graph=graph,
    )
    return _services
