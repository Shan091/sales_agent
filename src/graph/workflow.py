# file: src/graph/workflow.py
from langgraph.graph import StateGraph, END
from src.graph.state import ConversationState
from src.graph.checkpointer import get_checkpointer, get_checkpointer_async
from src.graph.nodes.triage import node_triage, node_human_escalation, node_adversarial_block
from src.graph.nodes.rag import node_retrieve_technical_context
from src.graph.nodes.sales import (
    node_high_intent,
    node_window_shopper,
    node_problem_solver,
    node_b2b_enterprise,
    node_post_sale_support,
    node_out_of_domain,
    node_general_greeting,
    node_contextual_rewarm,
    node_human_probe
)


# ═══════════════════════════════════════════════
#  Phase 3: RAG-Aware Routing
# ═══════════════════════════════════════════════

def route_after_triage(state: ConversationState):
    """
    Routes after triage. If data_routing_flag is TECHNICAL_RAG,
    the message passes through the shared RAG retrieval node FIRST,
    then continues to the archetype node.
    """
    # Priority order (per design: Adversarial > Frustrated > Natural). Adversarial
    # deflection and human-handoff are checked BEFORE the TECHNICAL_RAG flag so a
    # prompt-injection disguised as a technical/spec question cannot bypass
    # adversarial_block and run the full RAG pipeline on hostile input.
    if state.get("current_archetype") == "MALICIOUS_ADVERSARIAL":
        return "adversarial_block"

    # A person owns this thread until they release it (src/scripts/resolve_handoff.py). Checked
    # alongside the per-turn decision and BEFORE anything else, because the customer may well
    # send an ordinary-looking message while a colleague is mid-call with them — resuming there
    # means two voices answering on two numbers, which is worse than a short holding line.
    if state.get("requires_human_handoff") or state.get("handoff_active"):
        return "human_escalation"

    # Phase 3: If the triage flags a technical query, route upstream to RAG first
    if state.get("data_routing_flag") == "TECHNICAL_RAG":
        return "retrieve_technical_context"

    arch = state.get("current_archetype")
    if arch == "SALES_HIGH_INTENT":
        return "high_intent"
    elif arch == "SALES_WINDOW_SHOPPER":
        return "window_shopper"
    elif arch == "SALES_PROBLEM_SOLVER":
        return "problem_solver"
    elif arch == "B2B_ENTERPRISE":
        return "b2b_enterprise"
    elif arch == "POST_SALE_SUPPORT":
        return "post_sale_support"
    elif arch == "OUT_OF_DOMAIN":
        return "out_of_domain"
    elif arch == "GENERAL_GREETING":
        return "general_greeting"
    elif arch == "CONTEXTUAL_REWARM":
        return "contextual_rewarm"
    elif arch == "HUMAN_PROBE":
        return "human_probe"
    elif arch == "MALICIOUS_ADVERSARIAL":
        return "adversarial_block"
    return "human_escalation"  # Default fallback


def route_after_rag(state: ConversationState):
    """
    Routes AFTER the RAG retrieval node, back to the archetype node.

    The RAG node no longer escalates on its own: if query condensation fails it
    falls back to a raw-message embed, and empty retrieval simply proceeds with no
    context (the archetype node answers warmly from OTOHOM_OVERVIEW). The
    requires_human_handoff check below is retained purely as a defensive guard for
    any upstream flag that may have been set before retrieval.
    """
    if state.get("requires_human_handoff"):
        return "human_escalation"

    arch = state.get("current_archetype")
    if arch == "SALES_HIGH_INTENT":
        return "high_intent"
    elif arch == "SALES_WINDOW_SHOPPER":
        return "window_shopper"
    elif arch == "SALES_PROBLEM_SOLVER":
        return "problem_solver"
    elif arch == "B2B_ENTERPRISE":
        return "b2b_enterprise"
    elif arch == "POST_SALE_SUPPORT":
        return "post_sale_support"
    elif arch == "OUT_OF_DOMAIN":
        return "out_of_domain"
    elif arch == "GENERAL_GREETING":
        return "general_greeting"
    elif arch == "CONTEXTUAL_REWARM":
        return "contextual_rewarm"
    return "human_escalation"


def create_workflow() -> StateGraph:
    workflow = StateGraph(ConversationState)

    # Add Nodes
    workflow.add_node("triage", node_triage)
    workflow.add_node("retrieve_technical_context", node_retrieve_technical_context)  # Phase 3: Upstream RAG
    workflow.add_node("high_intent", node_high_intent)
    workflow.add_node("window_shopper", node_window_shopper)
    workflow.add_node("problem_solver", node_problem_solver)
    workflow.add_node("b2b_enterprise", node_b2b_enterprise)
    workflow.add_node("post_sale_support", node_post_sale_support)
    workflow.add_node("out_of_domain", node_out_of_domain)
    workflow.add_node("general_greeting", node_general_greeting)
    workflow.add_node("contextual_rewarm", node_contextual_rewarm)
    workflow.add_node("human_probe", node_human_probe)
    workflow.add_node("human_escalation", node_human_escalation)
    workflow.add_node("adversarial_block", node_adversarial_block)

    # Build Edges
    workflow.set_entry_point("triage")

    # Triage -> RAG or Archetype
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "retrieve_technical_context": "retrieve_technical_context",
            "high_intent": "high_intent",
            "window_shopper": "window_shopper",
            "problem_solver": "problem_solver",
            "b2b_enterprise": "b2b_enterprise",
            "post_sale_support": "post_sale_support",
            "out_of_domain": "out_of_domain",
            "general_greeting": "general_greeting",
            "contextual_rewarm": "contextual_rewarm",
            "human_probe": "human_probe",
            "human_escalation": "human_escalation",
            "adversarial_block": "adversarial_block"
        }
    )

    # RAG -> Archetype (Phase 3: second routing hop after context injection)
    workflow.add_conditional_edges(
        "retrieve_technical_context",
        route_after_rag,
        {
            "high_intent": "high_intent",
            "window_shopper": "window_shopper",
            "problem_solver": "problem_solver",
            "b2b_enterprise": "b2b_enterprise",
            "post_sale_support": "post_sale_support",
            "out_of_domain": "out_of_domain",
            "general_greeting": "general_greeting",
            "contextual_rewarm": "contextual_rewarm",
            "human_escalation": "human_escalation",
        }
    )

    # All paths end the current graph execution round
    workflow.add_edge("high_intent", END)
    workflow.add_edge("window_shopper", END)
    workflow.add_edge("problem_solver", END)
    workflow.add_edge("b2b_enterprise", END)
    workflow.add_edge("post_sale_support", END)
    workflow.add_edge("out_of_domain", END)
    workflow.add_edge("general_greeting", END)
    workflow.add_edge("contextual_rewarm", END)
    workflow.add_edge("human_probe", END)
    workflow.add_edge("human_escalation", END)
    workflow.add_edge("adversarial_block", END)

    return workflow


def compile_workflow(checkpointer=None):
    """
    Compiles the graph WITH a checkpointer so multi-turn conversations persist across
    messages via thread_id. Defaults to the process-local MemorySaver — fine for tests
    and local_chat.py. The worker uses compile_workflow_async() for durable state.
    """
    return create_workflow().compile(checkpointer=checkpointer or get_checkpointer())


async def compile_workflow_async():
    """
    Production entry point: compiles the graph against the durable Postgres checkpointer.

    Durability is load-bearing now, not just nice to have — a `pending_order` (a priced,
    about-to-be-charged quote) lives in graph state between the customer tapping
    "Confirm & pay" and the link being minted. Losing it to a worker restart would drop a
    sale mid-checkout. Falls back to MemorySaver if Postgres is unreachable.
    """
    checkpointer = await get_checkpointer_async()
    return create_workflow().compile(checkpointer=checkpointer)
