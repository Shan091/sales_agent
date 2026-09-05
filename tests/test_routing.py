"""
Unit tests for LangGraph routing precedence (Phase 1/2). No DB / LLM / Redis required.

Regression focus: an adversarial turn that ALSO gets flagged TECHNICAL_RAG must be
deflected to adversarial_block and must NOT reach the RAG retrieval node.
"""
from src.graph.workflow import route_after_triage, route_after_rag


def _state(**kw):
    base = {
        "current_archetype": None,
        "requires_human_handoff": False,
        "data_routing_flag": "NONE",
    }
    base.update(kw)
    return base


class TestRouteAfterTriage:
    def test_adversarial_with_technical_rag_goes_to_block_not_rag(self):
        s = _state(current_archetype="MALICIOUS_ADVERSARIAL", data_routing_flag="TECHNICAL_RAG")
        assert route_after_triage(s) == "adversarial_block"

    def test_adversarial_takes_precedence_over_handoff(self):
        s = _state(current_archetype="MALICIOUS_ADVERSARIAL", requires_human_handoff=True)
        assert route_after_triage(s) == "adversarial_block"

    def test_handoff_routes_to_escalation(self):
        s = _state(requires_human_handoff=True, current_archetype="SALES_HIGH_INTENT")
        assert route_after_triage(s) == "human_escalation"

    def test_technical_rag_routes_to_retrieval(self):
        s = _state(current_archetype="SALES_HIGH_INTENT", data_routing_flag="TECHNICAL_RAG")
        assert route_after_triage(s) == "retrieve_technical_context"

    def test_archetype_dispatch(self):
        mapping = {
            "SALES_HIGH_INTENT": "high_intent",
            "SALES_WINDOW_SHOPPER": "window_shopper",
            "SALES_PROBLEM_SOLVER": "problem_solver",
            "B2B_ENTERPRISE": "b2b_enterprise",
            "POST_SALE_SUPPORT": "post_sale_support",
            "OUT_OF_DOMAIN": "out_of_domain",
            "GENERAL_GREETING": "general_greeting",
            "CONTEXTUAL_REWARM": "contextual_rewarm",
            "HUMAN_PROBE": "human_probe",
        }
        for arch, node in mapping.items():
            assert route_after_triage(_state(current_archetype=arch)) == node

    def test_human_probe_routes_to_probe_not_escalation(self):
        # A calm human request routes to the probe (handoff stays False); it must NOT be
        # forwarded straight to human_escalation.
        s = _state(current_archetype="HUMAN_PROBE", requires_human_handoff=False)
        assert route_after_triage(s) == "human_probe"

    def test_unknown_archetype_defaults_to_escalation(self):
        assert route_after_triage(_state(current_archetype="???")) == "human_escalation"


class TestRouteAfterRag:
    def test_handoff_after_rag(self):
        s = _state(requires_human_handoff=True, current_archetype="SALES_HIGH_INTENT")
        assert route_after_rag(s) == "human_escalation"

    def test_archetype_after_rag(self):
        assert route_after_rag(_state(current_archetype="SALES_PROBLEM_SOLVER")) == "problem_solver"
