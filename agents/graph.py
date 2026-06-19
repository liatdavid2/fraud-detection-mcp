from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.state import FraudInvestigationState
from app.fraud_tools import FraudTools
from app.openai_investigator import OpenAIInvestigator


class FraudInvestigationGraph:
    def __init__(self, tools: FraudTools, investigator: OpenAIInvestigator):
        self.tools = tools
        self.investigator = investigator
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(FraudInvestigationState)

        workflow.add_node("load_application", self._load_application)
        workflow.add_node("score_application", self._score_application)
        workflow.add_node("policy_decision", self._policy_decision)
        workflow.add_node("llm_investigation", self._llm_investigation)
        workflow.add_node("maybe_create_review_case", self._maybe_create_review_case)

        workflow.add_edge(START, "load_application")
        workflow.add_edge("load_application", "score_application")
        workflow.add_edge("score_application", "policy_decision")
        workflow.add_edge("policy_decision", "llm_investigation")
        workflow.add_edge("llm_investigation", "maybe_create_review_case")
        workflow.add_edge("maybe_create_review_case", END)

        return workflow.compile()

    def invoke(self, application_id: str) -> FraudInvestigationState:
        return self.graph.invoke({"application_id": str(application_id), "errors": []})

    def _load_application(self, state: FraudInvestigationState) -> FraudInvestigationState:
        application_id = state["application_id"]
        application = self.tools.get_application(application_id)
        return {"application": application}

    def _score_application(self, state: FraudInvestigationState) -> FraudInvestigationState:
        application_id = state["application_id"]
        score = self.tools.score_application(application_id)
        return {"score": score}

    def _policy_decision(self, state: FraudInvestigationState) -> FraudInvestigationState:
        score = state["score"]
        policy = self.tools.get_policy_decision(score["fraud_probability"])
        return {"policy": policy}

    def _llm_investigation(self, state: FraudInvestigationState) -> FraudInvestigationState:
        summary = self.investigator.write_investigation_summary(
            application=state["application"],
            score=state["score"],
            policy=state["policy"],
        )
        return {"investigation_summary": summary}

    def _maybe_create_review_case(self, state: FraudInvestigationState) -> FraudInvestigationState:
        policy = state["policy"]
        if not policy.get("should_create_case"):
            return {"review_case": None}

        application_id = state["application_id"]
        case = self.tools.create_review_case(
            application_id=application_id,
            priority=policy["priority"],
            reason=policy["reason"],
            analyst_summary=state.get("investigation_summary"),
        )
        return {"review_case": case}
