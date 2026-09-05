"""
What the Otohom team receives when the agent finds a lead, or when it hands a thread to a person.

One destination, one row shape, and a `kind` column that says which job the row is. The client's call:
a single spreadsheet and a single inbox, filtered rather than split, because two tabs are two places
to forget to look. What the split fixed remains fixed — an escalation is no longer a "lead" with
nothing in it to call about, it is a row that says ESCALATION and carries the reason, the stage and
the order.

Everything here is pure except `freeze_and_handoff`, which does the delivery. That matters: the
summary a salesperson reads is built from state rather than written by the model, so it costs no LLM
call, adds no latency to the customer's turn, and cannot invent a figure or a promise.
"""
import logging
from typing import Any, Dict, List, Optional

from src.core.text import dedupe_keeping_first, truncate_words
from src.graph.state import ConversationState, user_texts
from src.services.lead_sink import LeadSink

logger = logging.getLogger(__name__)

# What kind of job this row is. It is a column, not a destination — everything lands in the same
# sheet and the same inbox, so the team filters on this rather than looking in two places.
KIND_LEAD = "LEAD"                # somebody to call back about a product
KIND_ESCALATION = "ESCALATION"    # somebody waiting for a person, right now
KIND_SUPPORT = "SUPPORT"          # an existing customer with a problem
KIND_PARTIAL = "PARTIAL"          # a thread with no product interest — not a callable lead

# How much of the customer's own words to quote per message, and how many messages.
_QUOTE_CHARS = 90
_QUOTE_MESSAGES = 3


def _kind_for(state: ConversationState) -> str:
    """
    Which job this row is, in the order the team needs it.

    A person waiting always wins, whatever else the thread also is: that row is the one with somebody
    on the other end of it. The lead row is not lost — it has its own dedup flag and fires on its
    own, so a hot lead that later escalates appears as both, exactly once each.
    """
    if state.get("requires_human_handoff") or state.get("handoff_active"):
        return KIND_ESCALATION
    if state.get("current_archetype") == "POST_SALE_SUPPORT":
        return KIND_SUPPORT
    if state.get("primary_interest"):
        return KIND_LEAD
    return KIND_PARTIAL


def stage_phrase(state: ConversationState) -> str:
    """
    How far the sale actually got, as a phrase rather than an integer.

    `consult_stage` is 0-4 and means nothing to whoever picks up the phone. The order below is
    deliberate: payment outcomes outrank the walkthrough, because a paid order is the end of the
    story and a failed one is the most urgent thing on the row.
    """
    paid_status = (state.get("last_payment_status") or "").lower()
    if paid_status == "paid":
        return "paid"
    if paid_status in ("failed", "expired"):
        failures = state.get("payment_failure_count") or 0
        return f"payment {paid_status}" + (f" ({failures}x)" if failures > 1 else "")
    if state.get("checkout_confirmed") or state.get("payment_link_sent"):
        return "payment link sent"
    if state.get("pending_order"):
        return "saw the price"
    stage = state.get("consult_stage") or 0
    if stage >= 1:
        return "chose a product"
    if state.get("primary_interest"):
        return "asked about a product"
    return "browsing"


def order_line(state: ConversationState) -> str:
    """
    What is on the order, priced by code, or what was actually bought once paid.

    Figures here are safe for the same reason the quote's are: they come from `pending_order`, which
    `_build_pending_order` priced from the catalogue and `validate_payment_request` re-derived. The
    model never contributes one.
    """
    paid = state.get("paid_line_items") or {}
    if paid:
        # No total here: `paid_line_items` is {sku: qty} and the amount lives on the PaymentOrder
        # row, not in graph state. The `stage` column beside this one already says "paid", so the
        # marker is not repeated.
        return ", ".join(f"{sku} x{qty}" for sku, qty in paid.items())
    order = state.get("pending_order") or {}
    if not order:
        return ""
    summary = order.get("product_summary") or ""
    amount = order.get("amount")
    if amount:
        return f"{summary} — ₹{int(amount):,}".strip(" —")
    return summary


def build_digest(state: ConversationState) -> str:
    """
    The summary column, built in code from what is already known.

    `conversation_summary` was declared in state and written by nothing, anywhere — which is the
    whole reason the column had always been blank. It is filled here instead of by a model: this
    runs on the customer's turn, so an extra LLM call would be latency they wait through, and a
    paraphrase of a conversation is exactly the kind of text that quietly invents a commitment.

    Five short lines at most, in the order a salesperson reads them: what they want, what worries
    them, how far it got, what they actually said, and what a colleague already did about it.
    """
    lines: List[str] = []

    interest = (state.get("primary_interest") or "").strip()
    city = (state.get("city") or "").strip()
    if interest or city:
        where = f" In {city}." if city else ""
        lines.append((f"Asked about: {interest}." if interest else "No product named yet.") + where)

    concerns = dedupe_keeping_first(state.get("pain_points") or [])
    if concerns:
        lines.append("Concerns: " + "; ".join(concerns) + ".")

    stage = stage_phrase(state)
    order = order_line(state)
    lines.append(f"Got as far as: {stage}" + (f" — {order}." if order else "."))

    quotes = user_texts(state)
    if quotes:
        # The first message says why they came; the last two say where they are now.
        picked = [quotes[0]] + [q for q in quotes[-2:] if q != quotes[0]]
        rendered = " … ".join(f'"{truncate_words(q, _QUOTE_CHARS)}"' for q in picked[:_QUOTE_MESSAGES])
        lines.append("Their words: " + rendered)

    notes = [n for n in (state.get("handoff_notes") or []) if n]
    if notes:
        lines.append("Colleague's note: " + truncate_words(notes[-1], 220))

    return "\n".join(lines)


class CRMHandoffService:
    @staticmethod
    def build_summary(
        thread_id: str, state: ConversationState, kind: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assemble the row the team receives. Pure (no I/O), so it is easy to test and reuse.

        Mobile number is the WhatsApp thread_id, so it isn't duplicated in state. `route` keeps the
        client's original SALES/SUPPORT tag; `kind` is what a reader filters on. Pass `kind` to force
        one (the delivery blocks do, so each dedup flag governs its own row); leave it out and it is
        inferred.

        Five columns were dropped here — property type, budget, timeline, preferred contact time and
        deferred purchase intent. Nothing wrote them for most conversations, so they were empty in
        almost every row, and a column that is empty in almost every row teaches whoever reads the
        sheet to stop looking at that end of it. What replaced them is what code always knows:
        stage, order, escalation reason and the summary.
        """
        archetype = state.get("current_archetype")
        kind = kind or _kind_for(state)
        route = "SUPPORT" if archetype == "POST_SALE_SUPPORT" else "SALES"
        return {
            "kind": kind,
            "route": route,
            "mobile_number": thread_id,
            "customer_name": state.get("customer_name"),
            "city": state.get("city"),
            "products_interested": state.get("primary_interest"),
            "pain_points": dedupe_keeping_first(state.get("pain_points") or []),
            "stage": stage_phrase(state),
            "order": order_line(state),
            "escalation_reason": state.get("handoff_reason") if kind != KIND_LEAD else None,
            "archetype": archetype,
            "language": state.get("language_preference"),
            "summary": build_digest(state),
        }

    @staticmethod
    async def freeze_and_handoff(
        thread_id: str, state: ConversationState, kind: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compile the row and deliver it to every configured sink (webhook → sheet/WhatsApp group/CRM,
        and email). Returns the row. Delivery is fail-soft and never raises, so a downstream outage
        cannot break the customer's turn or trigger a task retry that would re-send WhatsApp replies.
        """
        summary = CRMHandoffService.build_summary(thread_id, state, kind=kind)
        logger.info(f"[{thread_id}] {summary['kind']} handoff initiated.")
        delivery = await LeadSink.deliver_lead(summary)
        logger.info(f"[{thread_id}] {summary['kind']} delivery result: {delivery}")
        return summary
