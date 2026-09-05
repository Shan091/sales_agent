from typing import TypedDict, Annotated, Any, Dict, List, Optional
from datetime import datetime
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages

class ConversationState(TypedDict):
    """
    State for the LangGraph workflow.

    `messages` uses LangGraph's add_messages reducer, which appends and also COERCES
    lightweight inputs into real message objects: the worker seeds a turn with
    ("user", text) and it lands in state as a HumanMessage. Downstream code that
    inspects the latest message — the deterministic checkout / escape-hatch gates in
    nodes/triage.py — therefore sees a message object rather than a raw tuple.
    add_messages also handles RemoveMessage, which plain concatenation does not.
    """
    messages: Annotated[List[AnyMessage], add_messages]

    # NLP & Routing
    language_preference: str
    current_archetype: str
    requires_human_handoff: bool
    data_routing_flag: str
    human_request_count: int  # Consecutive calm "talk to a human" requests; drives probe -> escalate. Reset when the user moves on.

    # ── Human ownership of the thread ───────────────────────────────────────
    # requires_human_handoff is the decision made THIS turn. handoff_active is stickier: it
    # means a person now owns this conversation on their own number, and the agent must not
    # quietly resume just because the next message looks ordinary. Cleared only by
    # src/scripts/resolve_handoff.py (the human releasing it) or by the
    # HANDOFF_MAX_HOLD_HOURS safety net.
    handoff_active: bool
    handoff_started_at: Optional[float]
    # Why we escalated — drives which acknowledgement the customer reads, so a repeated payment
    # failure and a request for a person don't produce the same sentence.
    handoff_reason: Optional[str]
    # True once the customer has been told a person is taking over, so further messages during
    # the hold get a short "passed along" line instead of the full notice again.
    handoff_notified: bool
    # What the human did, supplied when they release the thread. This is the ONLY way context
    # crosses back: the human works on a different WhatsApp number, so nothing they type there
    # is visible to this system.
    handoff_notes: List[str]

    # Extracted Dialog State Tracking
    property_type: Optional[str]
    budget_tier: Optional[str]
    timeline: Optional[str]
    pain_points: List[str]
    deferred_purchase_intent: Optional[str]
    primary_interest: Optional[str]  # Sticky product interest from triage

    # Lead details for the sales-team handoff (sticky; gathered invisibly across turns).
    # Mobile number is the thread_id, so it is not duplicated here.
    customer_name: Optional[str]
    city: Optional[str]
    preferred_contact_time: Optional[str]
    lead_ready_for_handoff: bool  # LLM sets True when this is a real lead worth handing to the team
    lead_sent: bool  # Backend dedup flag: the lead was already delivered to the sinks for this thread
    # The same dedup for the OTHER kind. A thread can legitimately be both — a hot lead that later
    # needs a person belongs in the leads sheet AND in the escalations sheet — so the two flags are
    # separate and each fires once. Both are reset by handoff_control.release_handoff, so a thread
    # that needs a person again for a NEW reason is delivered again.
    escalation_sent: bool
    # Set by a sales node when the customer asked for the brochure AND one is configured. The
    # worker attaches the file after the turn — the graph only ever asks.
    brochure_requested: bool

    # RAG Pipeline (Phase 3)
    context_chunks: List[str]  # Populated by the upstream RAG node; consumed by any downstream archetype node
    rag_query: Optional[str]  # The condensed, entity-resolved query used for retrieval
    # True when the customer NAMED a product and retrieval found nothing about it. The archetype
    # then answers deterministically instead of calling the LLM, because a model asked about a
    # specific product with no data will answer from pretraining rather than admit the gap.
    specs_unavailable: bool

    # ── Agentic checkout ────────────────────────────────────────────────────
    # The gate/order/dedup trio, deliberately shaped like lead_ready_for_handoff/lead_sent:
    # the graph only ever *proposes*; the worker performs the money action after the turn.
    #
    # pending_order is built by NODE CODE (PricingEngine + discounts.apply_offer), never by
    # the LLM, and holds the fully priced itemized order: line items with catalogue unit
    # prices, the offer that survived clamping, and the grand total.
    pending_order: Optional[Dict[str, Any]]
    # Set in CODE when the customer taps "Confirm & pay" (see _is_confirm_checkout). The
    # explicit gate: no tap, no link, ever.
    checkout_confirmed: bool
    # Set in CODE when that tap arrived with no customer_name on file. The client's rule: the name
    # is mandatory at the pay button, so the link is held for one turn while the agent asks for it.
    # The mint gate reads it too — a held tap must not mint — and the next inbound message is parsed
    # for the name by code (`sales.py::_parse_name_and_city`), never by the model, because an LLM
    # turn at the pay button costs seconds at the worst moment and could re-propose the order.
    awaiting_pay_details: bool
    # Set in CODE when the customer taps "Apply <offer>" on a quote that was priced at list
    # price while an offer was in fact available. A re-pricing request, NOT an authorisation to
    # charge — the re-priced quote still has to be confirmed separately.
    apply_offer_requested: bool
    # Set in CODE when the customer taps "Add <product>" on a quote. Adds the complement the
    # agent suggested (already validated and priced when the quote was built) to the existing
    # order and re-quotes. Like apply_offer_requested this is a re-pricing request, never an
    # authorisation to charge.
    add_complement_requested: bool
    # ── The consultative walkthrough between "I'll take it" and a price ──────────────────
    # A customer who has just chosen a product is not yet looking at a quote. The order is
    # priced and held in state while three beats run in order: the dearer model of what they
    # picked, then the product that pairs with it, then the offer to price it up. Each beat is
    # one message with its own buttons, and each is rendered in CODE from data validated when
    # the order was built — so none of them costs an LLM call or can invent a figure.
    #
    # 0 nothing shown yet · 1 step-up shown · 2 pairing shown · 3 price offered · 4 priced.
    # Monotonic, and per conversation rather than per order: once the customer has been through
    # the walkthrough, a later change of mind re-prices straight to a quote instead of asking
    # them to sit through it again. Beats with nothing validated to say are skipped, so an order
    # with no step-up and no pairing goes from 0 to 3 in one step.
    consult_stage: int
    # Which sale the stage above refers to: the order's canonical skus, sorted and joined,
    # quantity deliberately excluded. Written only by sales.py::_advance, alongside the stage, so
    # the two can never disagree. A newly priced order sharing NO product with this one is a
    # different sale whose own step-up has never been shown, and it restarts the beats; an order
    # that merely grew or changed quantity is the same sale and keeps its place.
    consult_order_key: str
    # Set in CODE when the customer taps the step-up button ("Switch to Premium"). Swaps one
    # line for the verified dearer model and re-prices — a re-pricing request, never an
    # authorisation to charge.
    swap_upgrade_requested: bool
    # Set in CODE when the customer taps past a beat without taking it ("Keep the Base",
    # "Just this for now"). Advances the walkthrough; changes no money.
    consult_next_requested: bool
    # Set in CODE when the customer taps the quote button at the end of the walkthrough. The
    # itemised quote is a code render of the order already in state, so this needs no LLM call
    # either — and it is still not an authorisation to charge.
    quote_now_requested: bool
    # Worker dedup flag — a graph replay or TaskIQ retry must not re-mint a payment link.
    payment_link_sent: bool
    # The live Razorpay short url, mirrored here from the audit row so a retry nudge can
    # re-offer it even if the audit write failed. A bookkeeping failure must not become a dead
    # end for the customer.
    payment_link_url: Optional[str]
    # Closed loop from the Razorpay webhook: "paid" | "failed" | None, so the next turn
    # knows what happened without asking the customer.
    last_payment_status: Optional[str]
    # {sku: qty} of the most recently PAID order. `pending_order` is cleared the moment payment
    # lands — a settled order is not pending, and leaving it there left a pay button one tap from
    # a second charge — so this is what remains: the minimum needed for a later turn to recognise
    # the model re-proposing something the customer has already bought. The order itself lives in
    # payment_orders, which is the trail.
    paid_line_items: Dict[str, int]
    # Consecutive declines. One is handled in-agent (retry / other method); reaching
    # settings.MAX_PAYMENT_FAILURES trips the critical human-escalation safety valve.
    payment_failure_count: int

    # Semantic memory (mem0) recall for this turn — durable facts about this person from
    # earlier sessions, injected into the prompt as {memory_block}.
    memory_facts: List[str]

    # Compaction Memory
    conversation_summary: str
    last_user_message_timestamp: float


def _user_text_of(message: Any) -> Optional[str]:
    """
    The user text of ONE message, whatever shape it arrived in — or None if it isn't a user turn.

    Shared by `last_user_text` and `user_texts` so the shape tolerance that keeps the deterministic
    gates working lives in exactly one place.
    """
    content: Any = None
    role: Any = None

    if isinstance(message, BaseMessage):
        if message.type != "human":
            return None
        content = message.content
    elif isinstance(message, tuple) and len(message) == 2:
        role, content = message
    elif isinstance(message, dict):
        role = message.get("role") or message.get("type")
        content = message.get("content")
    else:
        return None

    if role is not None and str(role).lower() not in ("user", "human"):
        return None

    if isinstance(content, list):
        # Multimodal content blocks: keep the text parts only.
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        return None
    return content.strip()


def last_user_text(state: "ConversationState", lower: bool = True) -> str:
    """
    The latest inbound user text, however that message reached state: a BaseMessage, a
    ("user", text) tuple, or a {"role": ..., "content": ...} dict.

    Deterministic gates read through this rather than type-checking the raw list. Those gates
    decide whether a payment link is minted and whether the human escape hatch works, so they
    must not silently return False because an input arrived in a different but valid shape —
    such a failure is indistinguishable from the customer never having tapped the button.
    An AI turn never qualifies, which is what stops a quote's own button text from satisfying
    the gate it advertises. Returns "" when there is nothing text-like to inspect.

    lower=True (the default) suits keyword matching. Pass lower=False when the original text
    matters, e.g. the RAG fallback that embeds the message as a search query.
    """
    messages = state.get("messages") or []
    if not messages:
        return ""
    content = _user_text_of(messages[-1])
    if content is None:
        return ""
    return content.lower() if lower else content


def user_texts(state: "ConversationState") -> List[str]:
    """
    Every inbound user text in order, skipping the agent's turns and anything text-free.

    Used to quote the customer back to a salesperson in their own words (see
    `crm_handoff.build_digest`), which is worth more on a callback than any paraphrase.
    """
    return [
        text
        for text in (_user_text_of(m) for m in (state.get("messages") or []))
        if text
    ]

