# file: src/graph/nodes/sales.py
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage
from src.core.llm_factory import LLMFactory, execute_vendor_agnostic_node
from src.core.schemas import NodeExecutionSchema
from src.logic.prompts import (
    HIGH_INTENT_PROMPT,
    WINDOW_SHOPPER_PROMPT,
    PROBLEM_SOLVER_PROMPT,
    B2B_PROMPT,
    SUPPORT_PROMPT,
    OUT_OF_DOMAIN_PROMPT,
    GENERAL_GREETING_PROMPT,
    REWARM_PROMPT,
    HUMAN_PROBE_PROMPT,
    RAG_GROUNDING_DIRECTIVE,
    PRICING_AUTONOMY,
    PRICING_LEGACY,
)
from src.core.guardrails import Guardrails
from src.core.text import fit_label, dedupe_keeping_first
from src.core.business_hours import business_status_line
from src.memory.semantic import format_memory_block
from src.core.database import async_session_maker
from src.logic.pricing import PricingEngine
from src.logic import discounts
from src.graph.state import ConversationState, last_user_text
from config.settings import settings

logger = logging.getLogger(__name__)

# The escape-hatch option guaranteed on every human-probe reply, so a customer who wants a
# person is never trapped behind the probe. Tapping it sends "Connect me now" back to triage,
# which escalates immediately (see _is_connect_now in triage.py).
CONNECT_NOW_OPTION = {"label": "Connect me now", "postback_id": "CONNECT_NOW"}

# The two checkout buttons attached to a code-built itemized quote. Tapping "Confirm & pay"
# is the explicit gate the worker requires before minting a Razorpay link (see
# _is_confirm_checkout in triage.py). "Explore more" is a normal message — it simply doesn't set
# the gate, so the customer is never trapped into paying. It is worded as a request rather than a
# deferral on purpose: a customer who taps it stays in the conversation and has told the agent what
# to do next, where a plain "no thanks" ends the turn with nothing to act on.
CONFIRM_CHECKOUT_OPTION = {"label": "Confirm & pay", "postback_id": "CONFIRM_CHECKOUT"}
CHECKOUT_NOT_YET_OPTION = {"label": "Explore more", "postback_id": "CHECKOUT_NOT_YET"}

# The apply-discount button is built in _quote_options with the real percentage in its label
# (discounts.available_offer_preview), because a button reading "Apply my discount" tells the
# customer neither which discount nor how much — which is the complaint it was meant to answer.
APPLY_OFFER_POSTBACK = "APPLY_OFFER"

# The add-a-product button. Its label is built per-order from the complement the agent suggested
# and code then validated, so unlike the others there is no fixed phrase — the gate in triage.py
# matches this postback id (and the stored label) instead.
ADD_COMPLEMENT_POSTBACK = "ADD_COMPLEMENT"

# ── The consultative walkthrough ────────────────────────────────────────────────────────────────
# Choosing a product is not the same as asking what it costs. Between the two, the customer is
# walked through the dearer model of what they picked (only for the two pairs in discounts.UPGRADES
# — for every other product that beat does not exist), then the product that pairs with it, then
# the offer to price it up — one message per beat, each with its own buttons, and every one of them
# rendered in code from data validated when the order was built. See state.py::consult_stage.
#
# SWAP_UPGRADE takes the step-up; CONSULT_NEXT moves past a beat without taking it; QUOTE_NOW asks
# for the itemised price. None of the three authorises a charge — that is still only "Confirm & pay".
SWAP_UPGRADE_POSTBACK = "SWAP_UPGRADE"
CONSULT_NEXT_POSTBACK = "CONSULT_NEXT"
QUOTE_NOW_POSTBACK = "QUOTE_NOW"

# The label on the button that answers the walkthrough's last question. Phrased as the customer's
# own "yes" rather than as an instruction to the agent, and it says PRICE, not "quote": a quote is
# what one business sends another, and this is a person buying a door lock. Only the LABEL changes —
# the gate matches QUOTE_NOW_POSTBACK, which is exactly why labels are free to be rewritten.
QUOTE_NOW_LABEL = "Yes, show the price"
QUOTE_NOW_OPTION = {"label": QUOTE_NOW_LABEL, "postback_id": QUOTE_NOW_POSTBACK}

# Words that make a button a promise about the total. Only QUOTE_NOW_POSTBACK can keep that promise,
# and the model invents its own postback ids freely, so any OTHER option offering the price is either
# a duplicate of the one code just put in slot one or a button that shows nothing when tapped. A live
# chat produced both at once — `Yes, show the price [QUOTE_NOW]` beside `Show me the price
# [SHOW_PRICE]`, the second of which lands on the hold and answers with prose. See
# _keep_price_reachable, which drops them on the labels alone.
_PRICE_WORDS = ("price", "pricing", "cost", "total", "how much", "howmuch", "quote", "figure")

# The "I'll stay with what I picked" label for the pairing beat. It deliberately does not make the
# customer state a refusal — the same reason "Not yet" became "Explore more" on the quote. A no is a
# negative note at the moment momentum matters most, and nothing is gained by extracting one.
CONSULT_NEXT_LABEL = "Just this for now"

# The curiosity option offered beside the price button. The agent writes the QUESTION — a problem
# someone in this customer's situation is likely to have, in the words they'd use themselves; code
# picks the short button label from the closed set below, because a label has 20 characters and a
# model given a length limit spends them badly ("Smart switches & dim" came out of a live chat).
# Tapping it is an ordinary message — it reuses the CHECKOUT_NOT_YET postback, which nothing gates
# on — so the reply is a normal LLM turn that can act on what they asked for.
#
# The set is wide because a label that misses its theme is worse than a generic one: the button has
# to answer the question printed directly above it, or the customer cannot tell what tapping does.
# Every label here is hand-written at 20 characters or fewer, so fit_label never has to trim one —
# a half-cut label on the message that decides whether a price gets shown is not recoverable.
_HOOK_LABELS = (
    (
        ("electric", "power bill", "energy", "consumption", "meter", "bill", "wast",
         "left on", "switched on", "running all"),
        "Save on electricity",
    ),
    (
        ("secur", "safe", "intrud", "burglar", "break-in", "camera", "alarm", "lock",
         "stranger", "theft", "doorbell", "who's at", "whos at", "visitor", "gate"),
        "Make it safer",
    ),
    (("water", "leak", "flood", "tank", "seep", "damp", "overflow"), "Stop water damage"),
    (("curtain", "blind", "shade", "drape", "sunlight"), "Curtains at a tap"),
    (("light", "mood", "ambien", "dim", "lamp", "bulb", "dark"), "Better lighting"),
    (
        ("voice", "alexa", "google", "phone", "app", "remote", "away", "travel",
         "holiday", "trip", "not at home", "check on"),
        "Control it anywhere",
    ),
    (
        ("easier", "convenien", "comfort", "hands", "routine", "morning",
         "forget", "remember", "hassle", "chore", "get up", "getting up"),
        "Make life easier",
    ),
)
# The fallback answers the question rather than changing the subject. "Show me more" read as a menu
# request next to a specific problem, which is the one thing the hook is not.
_HOOK_FALLBACK_LABEL = "Yes, tell me more"
_HOOK_FALLBACK_TEXT = (
    "One thing people usually wish they'd sorted at the same time — want me to show you what makes "
    "the biggest difference in a place like yours?"
)

# Words the trade uses and customers don't. A hook or a benefit clause carrying one of these is
# dropped, exactly as a figure-carrying one already is: "most homes lose more on standby power than
# on lighting" is a true sentence that a customer has to decode before they can agree with it, and a
# question they have to decode is a question they don't answer. The prompt asks for plain words; this
# is what makes it hold when the model reaches for the technical term anyway.
#
# One list, three callers (the hook, upgrade_reason, complement_reason) so the rule cannot hold on
# one card and lapse on another. Failure is always a downgrade, never an error: the hook falls back
# to a code-written question, a benefit clause is dropped and its card with it.
_JARGON_TERMS = (
    "standby", "stand-by", "phantom", "load", "kwh", "kilowatt", "wattage", "voltage",
    "amperage", "retrofit", "gang", "pir", "iot", "zigbee", "z-wave", "protocol",
    "firmware", "api", "sdk", "actuator", "topology", "latency", "bandwidth",
    "integration", "ecosystem", "mesh network", "wi-fi module", "provisioning",
    "scene", "scenes", "automation", "automations",
)
_JARGON_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in _JARGON_TERMS) + r")\b", re.IGNORECASE)


def _has_jargon(text: str) -> bool:
    """True when this customer-bound clause carries a term the customer wouldn't use out loud."""
    return bool(_JARGON_RE.search(text or ""))


def _hook_label(hook: str) -> str:
    """The 20-character button label for the agent's curiosity question, from a closed set."""
    text = (hook or "").lower()
    for keys, label in _HOOK_LABELS:
        if any(k in text for k in keys):
            return label
    return _HOOK_FALLBACK_LABEL


def _upgrade_button_labels(replaces: str, target: str) -> tuple:
    """
    ("Switch to Premium", "Keep the Base") for a verified step-up pair — the differing tail of each
    name, not the whole thing. Both registry pairs share a prefix with their step-up
    ("Smart Door Lock Base/Premium", "… Panel 7 inch/10 inch"), so dropping the shared words is
    what leaves the one word the choice actually turns on inside WhatsApp's 20 characters. A pair
    with nothing in common falls back to the full name, trimmed at a word boundary.
    """
    a = (replaces or "").split()
    b = (target or "").split()
    i = 0
    while i < len(a) and i < len(b) and a[i].lower() == b[i].lower():
        i += 1
    take = " ".join(b[i:]) or (target or "").strip()
    keep = " ".join(a[i:]) or (replaces or "").strip()
    return fit_label(f"Switch to {take}", 20), fit_label(f"Keep the {keep}", 20)


def _add_button_label(display: str, limit: int = 20) -> str:
    """
    "Add Smart Camera" for an Indoor Smart Camera — the longest TAIL of the name that fits, never an
    ellipsis.

    fit_label trims at a word boundary and marks the cut, which is right for prose and wrong here:
    "Add Indoor Smart…" went out in slot one of a priced order, on the one message where a half-read
    label sits beside the pay button. The distinguishing noun is at the END of every catalogue name
    ("Indoor Smart *Camera*", "Touch Screen Control *Panel*"), which is the same observation
    _upgrade_button_labels already uses from the other side of the name — so dropping leading words
    keeps the word the tap is actually about.

    Falls back to fit_label only when a single trailing word still doesn't fit, so this can never
    raise and never returns an empty label.
    """
    words = (display or "").split()
    for start in range(len(words)):
        candidate = f"Add {' '.join(words[start:])}"
        if len(candidate) <= limit:
            return candidate
    return fit_label(f"Add {display}", limit)



# Phrases that would attribute a care/medical/safety-monitoring capability to an Otohom product.
# Nothing in the catalogue supports any of them: the sensors report an event as it happens (a door
# opening, movement starting) and none watches for the ABSENCE of one. A live chat produced "PIR
# Motion Sensors … can be set up to alert you if there's no movement for a long period" to someone
# asking about an elderly parent — every product named was real, which is why `specs_unavailable`
# (rag.py::_mentions, a product-is-mentioned check) had nothing to catch: the product existed and the
# retrieved chunks discussed it. The invented part was the feature.
#
# This list is OBSERVABILITY, not a guarantee — a paraphrase walks straight past it, and the real
# defence is the absolute prohibition in prompts.GUARDRAIL_RULES. It is here because a claim of this
# kind is the one worth knowing about the same day it happens, and a silent prompt rule tells nobody.
_CARE_CLAIM_PHRASES = (
    "no movement",
    "lack of movement",
    "absence of movement",
    "inactivity",
    "hasn't moved",
    "has not moved",
    "fall detect",
    "detect a fall",
    "detects a fall",
    "medical alert",
    "emergency alert",
    "vital sign",
    "health monitor",
    "wellbeing monitor",
    "well-being monitor",
    "if they don't",
    "if she doesn't",
    "if he doesn't",
)


def _keep_price_reachable(model_options: Optional[list]) -> list:
    """
    Buttons for an LLM turn taken while "Shall I show you the price?" is still unanswered: the price
    first, then up to two of the model's own.

    The order is priced, validated and in state, and nothing on screen shows it — so the one thing
    that must not be lost is the way to see it. Slot one carries the same label the ask used, so a
    tap means what the customer already read. What follows is whatever the model offered next: it has
    just answered a request to explore, so its options are the products it showed, and dropping them
    would make the reply a dead end. `Explore more` backfills when it offered none, because a lone
    price button reads as being cornered, which is the exact thing CHECKOUT_NOT_YET exists to stop.

    Descriptions are stripped for the reason given in _quote_options: one description anywhere flips
    the whole message to WhatsApp's LIST layout, and that would bury the price button behind a
    "Choose one" tap on a turn whose entire job is to keep it reachable.

    A model option that ALSO offers the price is dropped, on the label alone (_PRICE_WORDS) rather
    than on an exact match with QUOTE_NOW_LABEL. Matching the label exactly is what let a live chat
    ship "Yes, show the price" and "Show me the price" side by side: two buttons reading the same,
    one of them carrying an invented postback id that nothing gates on, so it answered with prose and
    no total. Two buttons for one thing is a confusion; the one that cannot deliver is a lie.
    """
    options = [dict(QUOTE_NOW_OPTION)]
    for opt in model_options or []:
        if len(options) >= 3:
            break
        label = str((opt or {}).get("label") or "").strip()
        pid = str((opt or {}).get("postback_id") or "").strip()
        if not label or not pid:
            continue
        lowered = label.lower()
        if any(word in lowered for word in _PRICE_WORDS):
            logger.info(f"[walkthrough] dropped a second price button from the model: {label!r}")
            continue
        options.append({"label": label, "postback_id": pid})
    if len(options) == 1:
        options.append(dict(CHECKOUT_NOT_YET_OPTION))
    return options


def _quote_options(order) -> list:
    """
    Buttons for a code-built quote, at most three (WhatsApp's reply-button ceiling).

    Slot one is either the Apply button or the Add button, never both — they compete for the same
    tap, and the discount comes first because the customer has already earned it. That ordering
    also puts the add-a-product button on the quote that FOLLOWS the Apply tap, which is the
    moment of most goodwill in the conversation: they have just watched the total fall because of
    something they did.

    The Apply button appears only when code has confirmed an eligible offer that this order isn't
    receiving — never as decoration, so a tap always changes the total. Its label carries the
    actual percentage, which is safe precisely because code wrote it: the rule the agent is held
    to is that IT must not produce a figure, not that the customer mustn't see one.

    No option here carries a `description`. A description is only rendered by WhatsApp's LIST
    layout, and whatsapp.py switches to that layout as soon as one is present — which buries
    "Confirm & pay" behind a "Choose one" tap on the single message where the call to action has to
    be the first thing a thumb reaches. Anything worth saying about a button is said in the quote.
    """
    options = []
    preview = None if order.get("applied_offer") else discounts.available_offer_preview(order.get("line_items", []))
    complement = order.get("suggested_complement") or {}
    if preview:
        options.append({"label": preview["button_label"], "postback_id": APPLY_OFFER_POSTBACK})
    elif complement.get("button_label"):
        options.append({"label": complement["button_label"], "postback_id": ADD_COMPLEMENT_POSTBACK})
    options.append(dict(CONFIRM_CHECKOUT_OPTION))
    options.append(dict(CHECKOUT_NOT_YET_OPTION))
    return options


# Matches anything that looks like an XML/HTML tag: <tag>, </tag>, <tag attr="x">.
# RAG chunks are hand-curated markdown, so a real angle-bracket tag never appears
# legitimately in catalog text. Neutralizing them before injection stops a poisoned
# or mis-edited chunk from forging a </otohom_technical_context> boundary or smuggling
# a fake <system>/<instructions> block into the prompt. A literal comparison like
# "< 5W standby" is left intact (a "<" followed by a space/digit is not a tag).
_RAG_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")


def _sanitize_rag_chunk(text: str) -> str:
    """Neutralize prompt-injection surfaces in a single retrieved chunk before it is
    wrapped in the <otohom_technical_context> block. Strips tag-like sequences that
    could break out of the context boundary; leaves ordinary catalog prose untouched."""
    if not text:
        return ""
    return _RAG_TAG_RE.sub(" ", text)


def _build_rag_context_block(state: ConversationState) -> str:
    """
    Builds the XML-tagged RAG context block for prompt injection.
    If context_chunks is populated (from the upstream RAG node), formats them
    inside the grounding directive. Otherwise, returns an empty string.

    Each chunk is sanitized (tag-like sequences neutralized) so retrieved content
    cannot escape the context tags or inject instructions into the system prompt.
    """
    chunks = state.get("context_chunks", [])
    if not chunks:
        return ""

    # Sanitize each chunk, then join all parent chunks with clear separators.
    safe_chunks = [_sanitize_rag_chunk(c) for c in chunks if c]
    rag_context = "\n\n---\n\n".join(safe_chunks)
    return RAG_GROUNDING_DIRECTIVE.format(rag_context=rag_context)


def _build_pricing_policy_block(state: ConversationState, catalogue_names=None) -> str:
    """
    Build the {pricing_policy_block} every sales prompt receives — the single authority on money
    for this turn.

    settings.AGENT_FULL_AUTONOMY picks exactly one of two mutually exclusive policies:
      * True  -> PRICING_AUTONOMY: the agent quotes, negotiates within policy, discounts from a
                 closed offer registry and takes payment. The live offer menu AND the live
                 catalogue are appended so the model selects both the offer id and the product
                 sku from sets it can actually see, never ones it invented.
      * False -> PRICING_LEGACY: the merchant's original policy (no quoting, no discounting,
                 pricing bridges to the team). One flag, no code change. Neither the offer menu
                 nor the catalogue is shown — a discount the agent may not give shouldn't be
                 visible to it.

    A short payment-status line is appended when a link is already out or an order is paid, so the
    agent doesn't re-propose an order mid-payment.
    """
    if not settings.AGENT_FULL_AUTONOMY:
        return PRICING_LEGACY

    block = PRICING_AUTONOMY + discounts.offer_menu_for_prompt()
    block += catalogue_for_prompt(catalogue_names)
    # Filtered against the same live catalogue the order is priced from, so a pair whose product has
    # been deactivated is never proposed. Like the offer menu, it carries no figures.
    block += discounts.upgrade_menu_for_prompt(catalogue_names)
    block += _offer_standing_for_prompt(state)

    status = state.get("last_payment_status")
    if state.get("payment_link_sent") and status not in ("paid", "failed"):
        block += "\n\nNOTE: a payment link has already been sent for this order and is awaiting payment — do not create another; just help with anything else."
    elif status == "paid":
        block += (
            "\n\nNOTE: their order is PAID and settled. They have already received a confirmation "
            "AND an itemised receipt carrying their order reference and transaction id, so do not "
            "confirm the payment again and do not restate the order — repeating it reads as though "
            "something went wrong. Do NOT set `checkout_items` for anything they have already "
            "bought: a quote carries a pay button, and one under a settled order asks them to pay "
            "twice. Answer what they actually asked. If they want something NEW, treat that as a "
            "fresh order and quote it normally."
        )
    return block


def _offer_standing_for_prompt(state: ConversationState) -> str:
    """
    Tell the agent where the CURRENT order stands on the offer ladder, so an upsell can be
    honest instead of invented.

    Code computes eligibility and the gap; the agent only gets to decide which complementary
    product is worth suggesting. Deliberately carries no percentage and no rupee figure — the
    quote prints those — so the agent can motivate the suggestion without ever stating a number.

    It reads `pending_order`, which does not exist until AFTER the model has answered, so on the turn
    that first proposes an order this block is empty by construction — which is why the suggestion
    fields ride on that same response instead of depending on this. What it is actually for is the
    turns that come later: an "Explore more" tap, a follow-up question, a change of mind. On those
    the order IS in state and this is the only thing that tells the agent an offer ladder exists.
    """
    order = state.get("pending_order") or {}
    line_items = order.get("line_items") or []
    if not line_items:
        return ""

    parts = []
    current = discounts.best_eligible_offer(line_items)
    if order.get("applied_offer"):
        parts.append(f"This order currently has offer {order['applied_offer']} applied.")
    elif current:
        parts.append(
            f"This order already qualifies for offer {current[0]}, which is NOT applied yet — "
            "the customer has an apply-discount button on the quote."
        )

    hint = discounts.next_offer_hint(line_items)
    if hint:
        if hint.get("needs_products"):
            n = int(hint["needs_products"])
            parts.append(
                f"Adding {n} more distinct product would move it up to offer {hint['offer_id']} "
                f"(a bigger bundle discount). If there is a genuinely useful complement for what "
                f"they're building, name it in `suggested_complement` with a short reason — the "
                f"system prices it, gives it its own message with an 'add it' button and states the "
                f"reward, so do NOT pitch it in your own text and never say by how much. Drop it "
                f"immediately if they pass."
            )
        else:
            parts.append(
                f"A larger order bracket (offer {hint['offer_id']}) exists above this one. The quote "
                "states the threshold — do not restate or estimate it."
            )

    if not parts:
        return ""
    return "\n\nWHERE THIS ORDER STANDS (computed by the system, not by you):\n- " + "\n- ".join(parts)


def catalogue_for_prompt(names) -> str:
    """
    Render the priceable catalogue as a closed list for the prompt.

    The model is required to copy `checkout_items[].sku` verbatim from this list. Left empty
    when the catalogue can't be read, in which case the model has no sku to copy and proposes
    no order — the same fail-closed direction as an unresolvable price.
    """
    if not names:
        return ""
    lines = "\n".join(f"- {n}" for n in names)
    return (
        "\n\nPRICEABLE CATALOGUE — the ONLY product names the system can price.\n"
        f"{lines}\n"
        "When you set checkout_items, the `sku` MUST be copied character-for-character from this "
        "list. Your message to the customer can still use natural words (\"the 6-gang glass "
        "panel\"), but an sku that is not on this list resolves to nothing and the customer gets "
        "no quote at all. If what they want isn't listed, don't invent an sku — say you'll get the "
        "exact option confirmed and offer the free consultation."
    )


async def _load_catalogue_names() -> list:
    """Active priceable product names, or [] if the catalogue can't be read (fail-soft: the
    turn still happens, it just can't propose an order)."""
    if not settings.AGENT_FULL_AUTONOMY:
        return []
    try:
        async with async_session_maker() as session:
            return await PricingEngine(session).list_catalogue_names()
    except Exception as e:
        logger.warning(f"[checkout] catalogue name lookup failed: {e}")
        return []


_FIGURE_RE = re.compile(r"[\d₹$€£%]")

# Alphanumerics only, so "Smart Door Lock  Premium" matches "smart door lock premium" and "6 SW"
# matches "6SW" — the same normalisation rag.py::_mentions uses for the same kind of question.
_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _norm_alnum(text: str) -> str:
    """Alphanumerics only, lower-cased — the shape both containment checks below compare in."""
    return _ALNUM_RE.sub("", (text or "").lower())


def _already_named(product: str, prior_text: str) -> bool:
    """True when this product (or its plain-English gloss) has already been put in front of them."""
    haystack = _norm_alnum(prior_text)
    if not haystack:
        return False
    for name in (product, discounts.plain_product_name(product)):
        needle = _norm_alnum(name)
        if needle and needle in haystack:
            return True
    return False


def _step_up_already_shown(product: str, prior_text: str) -> bool:
    """
    True only when this customer has already been shown the step-up CARD for this product.

    Both halves are required: one of discounts.STEP_UP_CARD_MARKERS (a sentence nothing but a
    rendered card contains) AND the target's name. Mentioning the product is not enough, and that
    distinction is the whole point of the function — the version that dropped a step-up on any
    prior mention deleted the beat outright in a live chat, because discovery had described both
    lock models in prose one turn earlier.

    Prose and the card are not the same event. Prose can say a dearer model exists; only the card
    carries the exact price difference, the benefits as bullets, the swap framing and the
    Switch/Keep pair of buttons — which is to say, only the card is a decision the customer can
    actually make. What must never repeat is the DECIDED either/or, so the marker is what we look
    for. If the customer sees the card and taps `Keep the ‹X›`, the marker is in the transcript and
    the step-up is spent from then on.
    """
    haystack = _norm_alnum(prior_text)
    if not haystack:
        return False
    if not any(_norm_alnum(m) in haystack for m in discounts.STEP_UP_CARD_MARKERS):
        return False
    return _already_named(product, prior_text)


def _validate_complement(sku, reason, line_items, trusted_names) -> dict:
    """
    Turn the agent's suggested add-on into something safe to put a button on, or {} to drop it.

    The agent picks WHICH product to suggest — it has the conversation, so it can tell "I travel a
    lot" from "I have young kids". Code decides whether that suggestion may reach the customer:

      * the sku must be one the catalogue actually prices, matched the same fail-closed way the
        order's own lines are (a name the model half-remembered buys a dead button)
      * it must not already be a line in this order, or the button re-sells what they're buying
      * next_offer_hint must report a reachable tier, because the block the nudge renders claims a
        reward for adding it, and a claim without a reward is a bigger bill dressed as a saving

    `reason` is the model text that reaches the customer here — it becomes the card's opening line —
    so it is checked for digits, percentages, currency symbols and trade jargon. If it fails, the
    reason is dropped and the button kept: the rules that cannot bend are that no figure originates
    in the model and that no customer reads a word only an electrician uses, and dropping the whole
    suggestion over a stray clause would cost the cross-sell entirely. A card with no usable reason
    still has an honest heading to fall back on (discounts._complement_lines) and a real reward to
    state, which is what makes this the opposite call from the step-up's.
    """
    sku = (sku or "").strip()
    if not sku:
        return {}

    existing = {str(li.get("sku") or "").strip().lower() for li in (line_items or [])}
    if sku.lower() in existing:
        logger.info(f"[upsell] dropped complement '{sku}': already in the order.")
        return {}

    canonical = trusted_names.get(sku)
    if not canonical:
        logger.info(f"[upsell] dropped complement '{sku}': not priceable in the catalogue.")
        return {}
    if canonical.strip().lower() in existing:
        logger.info(f"[upsell] dropped complement '{sku}': resolves to a line already in the order.")
        return {}

    if not (discounts.next_offer_hint(line_items) or {}).get("needs_products"):
        logger.info(f"[upsell] dropped complement '{canonical}': no reachable offer tier to earn.")
        return {}

    display = discounts.plain_product_name(canonical) or canonical
    reason = (reason or "").strip()
    if reason and _FIGURE_RE.search(reason):
        logger.warning(f"[upsell] dropped complement reason containing a figure: {reason!r}")
        reason = ""
    if reason and _has_jargon(reason):
        logger.warning(f"[upsell] dropped complement reason containing trade jargon: {reason!r}")
        reason = ""

    return {
        "sku": canonical,
        "display_name": display,
        "reason": reason,
        # The tail of the name, not a trimmed head: "Add Smart Camera", never "Add Indoor Smart…".
        # Built from the SPEAKABLE name, so a button never reads "Add 6 SW" — trade shorthand on a
        # button is a tap the customer can't evaluate.
        "button_label": _add_button_label(discounts.speakable_name(display)),
    }


def _validate_upgrade(sku, replaces, reason, line_items, trusted, prior_agent_text: str = "") -> dict:
    """
    Turn the agent's suggested step-up into something safe to show, or {} to drop it.

    A complement ADDS a product; an upgrade SWAPS one. That difference drives every check here,
    and it is why this is a separate function rather than an argument to _validate_complement:

      * the pair must appear in `discounts.UPGRADES`, a closed hand-verified registry. This is the
        load-bearing check. "Costs more" was the first version of it and it is not sound: a 3-phase
        energy meter, an outdoor flood-light camera and an 8-gang switch panel are all dearer than
        the thing a customer picked and none of them is a dearer MODEL of it — the first is decided
        by the building's supply, the second does a different job, the third is a fitment size.
        Only code can hold that distinction, because it isn't in the prices and it isn't reliably
        in the product names either.
      * `replaces` must name a line that is actually in this order. Code will not guess which line
        the model meant — the nearest-name match would be a coin flip between "6 SW" and
        "6 SW FAN", and getting it wrong prints a swap the customer never asked about.
      * the upgrade must resolve in the catalogue and must not already be a line, exactly as a
        complement must.
      * the STEP-UP CARD for the target must not have been shown already. A card they have seen and
        passed over is a decided either/or, and putting it back on screen is the agent repeating
        itself. The check is deliberately narrow — a rendered-card marker AND the name, see
        _step_up_already_shown — because "was it mentioned?" is the wrong question and answering it
        cost a live sale the whole beat: discovery had described both lock models in prose, so the
        step-up was dropped and the customer never saw the exact difference or the swap framing at
        all. Prose can say a dearer model exists; only the card is a decision.
        Note the asymmetry with the complement, which gets NO such check: a pairing product that
        came up in conversation and was never bought is still a live, un-asked question, whereas a
        step-up that was declined is answered.
      * there must be a usable REASON. The card leads on what the dearer model gives them, so with
        no benefit clause there is nothing to lead on and only a price difference left — a bill with
        no reason attached. Dropped whole rather than shown weak, the same call the complement makes
        the other way round (see its docstring for why they differ). Checked through
        discounts.benefit_heading so this refusal and the renderer cannot disagree.
      * the registry pair must still be dearer at today's catalogue prices. Kept as a cheap sanity
        assertion, not as the definition of an upgrade: if a price edit ever inverts a pair, the
        block would print "₹0 more" or a negative, and silence beats a nonsense figure.

    Deliberately NO offer-tier requirement, unlike a complement. A swap leaves the line count
    unchanged, so it earns no new tier — demanding one here would reject every upgrade ever
    proposed. An upgrade justifies itself by being the better product, and the block that renders
    it claims nothing about a discount; code prints the exact price difference instead.
    """
    sku = (sku or "").strip()
    replaces = (replaces or "").strip()
    if not sku or not replaces:
        return {}

    lines_by_sku = {str(li.get("sku") or "").strip().lower(): li for li in (line_items or [])}

    target_pr = trusted.get(sku)
    if not target_pr:
        logger.info(f"[upsell] dropped upgrade '{sku}': not priceable in the catalogue.")
        return {}
    canonical = target_pr.product_name
    if canonical.strip().lower() in lines_by_sku or sku.lower() in lines_by_sku:
        logger.info(f"[upsell] dropped upgrade '{sku}': already in the order.")
        return {}

    # `replaces` is matched through the same batch resolution as everything else, because the model
    # copies it from checkout_items — which price_line_items has since rewritten to canonical names.
    replaced_pr = trusted.get(replaces)
    replaced_canonical = replaced_pr.product_name if replaced_pr else replaces
    current = lines_by_sku.get(replaced_canonical.strip().lower()) or lines_by_sku.get(replaces.lower())
    if not current:
        logger.info(f"[upsell] dropped upgrade '{sku}': '{replaces}' is not a line in this order.")
        return {}

    # The closed registry, checked on the canonical names both sides resolved to.
    if discounts.upgrade_target(replaced_canonical) != canonical:
        logger.info(
            f"[upsell] dropped upgrade '{replaced_canonical}' -> '{canonical}': not a verified "
            "step-up pair. Nothing in the catalogue is a dearer model of anything else outside "
            "discounts.UPGRADES — dearer is not better."
        )
        return {}

    if prior_agent_text and _step_up_already_shown(canonical, prior_agent_text):
        logger.info(
            f"[upsell] dropped upgrade '{canonical}': the step-up card for it has already been "
            "shown and passed over. That is a decided either/or, so the walkthrough opens on the "
            "pairing card instead."
        )
        return {}

    qty = max(int(current.get("qty", 1) or 1), 1)
    current_unit = float(current.get("unit_price", 0.0)) + float(current.get("installation_fee", 0.0))
    new_unit = float(target_pr.base_price) + float(target_pr.installation_fee or 0.0)
    if new_unit <= current_unit:
        logger.warning(
            f"[upsell] dropped verified pair '{replaced_canonical}' -> '{canonical}': the step up "
            "is not dearer at current catalogue prices — check seed_pricing/products_pricing."
        )
        return {}

    reason = (reason or "").strip()
    if reason and _FIGURE_RE.search(reason):
        logger.warning(f"[upsell] dropped upgrade reason containing a figure: {reason!r}")
        reason = ""
    if reason and _has_jargon(reason):
        logger.warning(f"[upsell] dropped upgrade reason containing trade jargon: {reason!r}")
        reason = ""
    if not discounts.benefit_heading(reason):
        logger.info(
            f"[upsell] dropped upgrade '{canonical}': no usable benefit to lead on. A price "
            "difference with nothing behind it is a bill with no reason attached."
        )
        return {}

    replaces_sku = str(current.get("sku") or "").strip()
    replaces_display = discounts.plain_product_name(replaces_sku) or replaces_sku
    # The two labels are built from the catalogue names, not from the plain-English glosses: the
    # glosses of a pair are often identical ("smart door lock" for both grades), and a button has to
    # name the thing that differs.
    take_label, keep_label = _upgrade_button_labels(replaces_sku, canonical)

    return {
        "sku": canonical,
        "display_name": discounts.plain_product_name(canonical) or canonical,
        "replaces_sku": replaces_sku,
        "replaces_display": replaces_display,
        "qty": qty,
        # The delta is per unit and for the whole line, both computed here so nothing downstream
        # has to re-derive a figure. At qty 4 a "₹10,000 more" that meant per-unit would understate
        # the ask by ₹30,000 — the sort of number this codebase would rather not print at all.
        "unit_delta": new_unit - current_unit,
        "line_delta": (new_unit - current_unit) * qty,
        "reason": reason,
        "button_label": take_label,
        "keep_label": keep_label,
    }


def _reproposes_paid_order(checkout_items, paid_line_items: Dict[str, int]) -> bool:
    """
    True when the model has proposed nothing the customer hasn't already bought and paid for.

    The paid order leaves `pending_order` the moment the webhook lands, so it can no longer be
    re-minted. This is the second layer, and it exists because the model demonstrably does
    re-propose: asked only for a name and city *after* paying, it set `checkout_items` to the
    camera it had just sold, and code dutifully built a fresh quote with a pay button under it.
    A quote is an invitation to pay; putting one under a settled order asks for the money twice.

    Compared on sku AND quantity so a genuine second order still gets quoted — a product they
    don't own, or more of one they do, differs and goes through. Only an exact restatement of the
    paid order is refused.
    """
    if not paid_line_items:
        return False

    proposed: Dict[str, int] = {}
    for it in (checkout_items or []):
        sku = str(getattr(it, "sku", "") or "").strip().lower()
        if not sku:
            continue
        proposed[sku] = proposed.get(sku, 0) + int(getattr(it, "qty", 1) or 1)
    if not proposed:
        return False

    paid = {str(k).strip().lower(): int(v) for k, v in paid_line_items.items()}
    return all(qty <= paid.get(sku, 0) for sku, qty in proposed.items())


async def _build_pending_order(
    checkout_items,
    applied_offer,
    suggested_complement=None,
    complement_reason=None,
    suggested_upgrade=None,
    upgrade_replaces=None,
    upgrade_reason=None,
    explore_hook=None,
    prior_agent_text: str = "",
):
    """
    Turn the LLM's product proposal into a fully priced, guardrail-approved order — entirely
    in code. Returns the order dict, or None if nothing could be priced or the final money
    guardrail rejected the result. Rendering is the caller's job: the same order is shown as a
    walkthrough beat or as an itemised quote depending on where the customer has got to.

    The LLM supplied only names + quantities + an offer *choice* + one suggested add-on; every
    rupee here comes from products_pricing via PricingEngine, is clamped by discounts.apply_offer,
    and is then independently re-checked by Guardrails.validate_payment_request before we ever
    show a total.

    `prior_agent_text` is everything the agent has already said in this conversation. It is used for
    one thing only — refusing a step-up whose *card* has already been put in front of this customer
    (see _validate_upgrade / _step_up_already_shown; prose that merely described the product does not
    count) — and it is passed in rather than read here so this function stays a pure function of its
    arguments and the catalogue.
    """
    raw_items = [
        {"sku": (ci.sku or "").strip(), "qty": ci.qty}
        for ci in (checkout_items or [])
        if getattr(ci, "sku", None)
    ]
    if not raw_items:
        return None

    # The complement and the upgrade are priced in the SAME batch lookup as the order, not extra
    # round trips: both have to clear the identical fail-closed resolution before they may reach the
    # customer, and the tap that adds a complement re-prices from the catalogue anyway.
    extras = {
        s.strip()
        for s in (suggested_complement, suggested_upgrade, upgrade_replaces)
        if (s or "").strip()
    }
    skus = list({it["sku"] for it in raw_items} | extras)
    try:
        async with async_session_maker() as session:
            engine = PricingEngine(session)
            price_map = await engine.get_product_prices_batch(skus)
    except Exception as e:
        logger.error(f"[checkout] price lookup failed: {e}")
        return None

    trusted = {name: pr for name, pr in price_map.items() if pr is not None}
    if not trusted:
        # Log both sides of the mismatch: the model's proposal and what the catalogue actually
        # offers. Without the second half this failure looks like a pricing outage rather than
        # what it usually is — the model naming a product that doesn't exist under that string.
        logger.error(
            f"[checkout] NO SKU RESOLVED — nothing quotable. proposed={skus} "
            f"catalogue={await _load_catalogue_names()}"
        )
        return None

    # price_line_items only ever prices the ORDERED lines; the complement was resolved in the same
    # batch above but must not become a line until the customer taps for it.
    line_items, unresolved, notes = discounts.price_line_items(raw_items, trusted)
    if not line_items:
        return None

    order = discounts.apply_offer(line_items, applied_offer)
    if notes:
        order["audit_notes"] = notes + order.get("audit_notes", [])

    # ── Both suggestions are kept, because each gets its own beat ────────────────────────────────
    # They used to compete for one slot on the quote, and the loser was thrown away — a live
    # transcript has the agent naming a lock AND a door phone, the customer taking the lock, and the
    # door phone never mentioned again with a better tier one product away. The walkthrough removes
    # the competition instead of arbitrating it: the step-up is offered first (it swaps something
    # they have already decided to buy, so there is no new "do I need this?" question), the pairing
    # second, and the two never share a message.
    upgrade = _validate_upgrade(
        suggested_upgrade,
        upgrade_replaces,
        upgrade_reason,
        line_items,
        trusted,
        prior_agent_text,
    )
    if upgrade:
        order["suggested_upgrade"] = upgrade

    complement = _validate_complement(
        suggested_complement,
        complement_reason,
        line_items,
        {name: pr.product_name for name, pr in trusted.items()},
    )
    if complement:
        order["suggested_complement"] = complement

    # The problem-framing question offered beside the price button, held on the order because the
    # beat it belongs to may be two taps away. Text only. Figure-free for the same reason every other
    # piece of model text on this path is — no amount may originate in the model — and jargon-free
    # because a question the customer has to decode is a question they don't answer.
    hook = (explore_hook or "").strip()
    if hook and _FIGURE_RE.search(hook):
        logger.warning(f"[upsell] dropped explore hook containing a figure: {hook!r}")
        hook = ""
    if hook and _has_jargon(hook):
        logger.warning(f"[upsell] dropped explore hook containing trade jargon: {hook!r}")
        hook = ""
    if hook:
        order["explore_hook"] = hook

    # Final fail-closed money gate. Re-derives the total and matches every unit price to the
    # catalogue to the paisa. If anything is off, we present NO quote (rather than a wrong one).
    # price_line_items rewrites each line's sku to the CANONICAL catalogue name, and the
    # guardrail matches unit prices by line-item sku — so it needs a trusted map keyed by that
    # canonical name, not the (possibly normalized) requested string used for price_map above.
    trusted_by_canonical = {pr.product_name: pr for pr in trusted.values()}
    ok, reasons = Guardrails.validate_payment_request(order, trusted_by_canonical)
    if not ok:
        logger.error(f"[checkout] payment guardrail rejected the built order: {reasons}")
        return None

    return order


def _quote_message(order: dict, show_suggestions: bool = True) -> AIMessage:
    """The itemised quote as a WhatsApp message — code-written text, code-built buttons."""
    return AIMessage(
        content=discounts.format_quote_message(order, show_suggestions=show_suggestions),
        response_metadata={
            "options": _quote_options(order),
            "internal_thought": "Code-built itemized quote; every amount is server-computed and guardrail-verified.",
        },
    )


# The beats of the walkthrough, and the consult_stage each one leaves behind. Monotonic, so a beat
# is never replayed and a customer who changes their mind later goes straight to a price.
#
# "hold" is not a beat the customer sees — it is the absence of one, and it earns 0 so that
# `max(stage, _STAGE_AFTER[beat])` leaves the stage exactly where it was, whatever it was. It exists
# because the itemised order must appear only when the customer asks for it, and while
# "Shall I show you the price?" is on screen every OTHER turn has to keep the price off it while
# still keeping the order priced in state. See the hold branch in _execute_sales_node.
_STAGE_AFTER = {"hold": 0, "upsell": 1, "crosssell": 2, "quote_ask": 3, "quote": 4}


def _order_key(order: dict) -> str:
    """
    Which sale a walkthrough belongs to: the order's canonical skus, lower-cased, sorted, joined.

    Quantity is deliberately not part of it. "Make it four" refines the sale in progress and must
    not send the customer back to beat one; a genuinely different product is a different sale and
    deserves its own step-up. Stored beside consult_stage as `consult_order_key` and written only by
    `_advance`, so the pair can never disagree about which order the stage refers to.
    """
    skus = {
        str(li.get("sku") or "").strip().lower()
        for li in (order.get("line_items") or [])
        if str(li.get("sku") or "").strip()
    }
    return "|".join(sorted(skus))


def _is_new_sale(order: dict, walked_key) -> bool:
    """
    True when a freshly priced order shares NO product with the one the walkthrough was walking.

    Disjoint rather than merely different, on purpose: adding a curtain motor to a lock order is the
    same sale getting larger and is owed the updated price it asked for, not a restarted walkthrough.
    Coming back next week and picking a panel instead shares nothing, and that customer has never
    been shown the panel's step-up.
    """
    walked = {part for part in str(walked_key or "").split("|") if part}
    current = {part for part in _order_key(order).split("|") if part}
    return bool(walked and current and not (walked & current))


def _next_beat(order: dict, stage) -> str:
    """
    Which beat this order is due next: "upsell", "crosssell", "quote_ask" or "quote".

    A beat with nothing validated to say is skipped entirely rather than filled with something
    weaker — an order with no verified step-up and no pairing goes from nothing shown straight to
    the offer of a price, which is one extra message, not three.
    """
    stage = int(stage or 0)
    if stage < _STAGE_AFTER["upsell"] and discounts.upgrade_pitch(order):
        return "upsell"
    if stage < _STAGE_AFTER["crosssell"] and discounts.complement_pitch(order):
        return "crosssell"
    if stage < _STAGE_AFTER["quote_ask"]:
        return "quote_ask"
    return "quote"


def _beat_message(order: dict, beat: str) -> AIMessage:
    """
    One walkthrough beat as a WhatsApp message: code-written body, at most three buttons.

    Nothing here calls a model. The product names, the price difference and the reward were all
    settled when the order was built; the only model text that reaches the customer is the benefit
    clause the validators already checked for figures.
    """
    if beat == "quote":
        return _quote_message(order, show_suggestions=True)

    if beat == "upsell":
        up = order.get("suggested_upgrade") or {}
        return AIMessage(
            content="\n".join(discounts.upgrade_pitch(order)),
            response_metadata={
                "options": [
                    {"label": up.get("button_label") or "Switch to this one", "postback_id": SWAP_UPGRADE_POSTBACK},
                    {"label": up.get("keep_label") or CONSULT_NEXT_LABEL, "postback_id": CONSULT_NEXT_POSTBACK},
                    dict(CHECKOUT_NOT_YET_OPTION),
                ],
                "internal_thought": "Walkthrough: verified step-up offered on its own, before any price.",
            },
        )

    if beat == "crosssell":
        comp = order.get("suggested_complement") or {}
        return AIMessage(
            content="\n".join(discounts.complement_pitch(order)),
            response_metadata={
                "options": [
                    {"label": comp.get("button_label") or "Add it", "postback_id": ADD_COMPLEMENT_POSTBACK},
                    {"label": CONSULT_NEXT_LABEL, "postback_id": CONSULT_NEXT_POSTBACK},
                    dict(CHECKOUT_NOT_YET_OPTION),
                ],
                "internal_thought": "Walkthrough: the pairing product on its own, with the tier it earns.",
            },
        )

    # quote_ask — the only beat that asks for anything, and it asks for permission to show a price
    # rather than for the sale. Two buttons, not three: the second is the customer's way out of
    # being priced at all, so a third would only dilute a choice that is genuinely binary.
    #
    # Two sentences and nothing else. What used to sit between them was an explainer ("I'll break it
    # down line by line…") and a reassurance ("No commitment either way") — the first describes the
    # next message instead of earning it, and the second answers an objection nobody made while
    # putting the word "commitment" in front of a customer who hadn't thought of one. Both went. Two
    # questions on one screen is fine here, and only here, because each has its own button under it.
    hook = str(order.get("explore_hook") or "").strip() or _HOOK_FALLBACK_TEXT
    body = ["*Shall I show you the price?*", "", hook]
    return AIMessage(
        content="\n".join(body),
        response_metadata={
            "options": [
                {"label": QUOTE_NOW_LABEL, "postback_id": QUOTE_NOW_POSTBACK},
                {"label": _hook_label(hook), "postback_id": CHECKOUT_NOT_YET_OPTION["postback_id"]},
            ],
            "internal_thought": "Walkthrough: asked before pricing, with a benefit-led way to keep exploring.",
        },
    )


def _advance(order: dict, stage, lead_in: Optional[str] = None, beat: Optional[str] = None) -> dict:
    """
    Render the next beat for this order and return the graph update that goes with it.

    The single place consult_stage moves, so it can only ever move forward and only ever alongside
    the message that earned the move. It is also the only writer of `consult_order_key`, so the
    stage and the order it refers to are always written together. `lead_in` is a short code-written
    line for the taps that changed something (a swap, an addition) — silence after those reads as a
    dropped message. `beat` forces a particular beat, which is how an explicit "how much is it?"
    reaches a price without being made to tap through anything first.

    `beat="hold"` is the one value that renders NOTHING: it stores the priced order and moves the
    stage nowhere, so a turn can leave the price ready without putting it on screen.
    """
    stage = int(stage or 0)
    beat = beat or _next_beat(order, stage)
    if beat == "hold":
        # No message at all. The order is still priced, still validated and still stored, so the
        # `Yes, show the price` tap works the moment it comes — but nothing on screen shows a total,
        # because the customer has not asked for one. The model's own reply is the whole of this
        # turn; the caller is what guarantees the price button rides along with it.
        messages = []
    elif beat == "quote":
        # The quote body carries the step-up / pairing blocks only when the walkthrough never got
        # to show them — i.e. the customer asked for a price straight away. Otherwise each has
        # already had its own message and its own button, and repeating both under the total is
        # exactly the menu the walkthrough exists to replace.
        messages = [_quote_message(order, show_suggestions=stage < _STAGE_AFTER["quote_ask"])]
    else:
        messages = [_beat_message(order, beat)]

    if lead_in and messages:
        messages.insert(0, AIMessage(
            content=lead_in,
            response_metadata={"options": None, "internal_thought": "Deterministic walkthrough step; no LLM call."},
        ))
    return {
        "messages": messages,
        "pending_order": order,
        "consult_stage": max(stage, _STAGE_AFTER[beat]),
        "consult_order_key": _order_key(order),
        # Every walkthrough step is a re-pricing or a re-render, never an authorisation: the new
        # state of the order has to be confirmed on its own, and must stay re-mintable.
        "checkout_confirmed": False,
        "payment_link_sent": False,
        "swap_upgrade_requested": False,
        "consult_next_requested": False,
        "quote_now_requested": False,
        "add_complement_requested": False,
        "apply_offer_requested": False,
    }


async def _reprice_with_best_offer(state: ConversationState) -> dict:
    """
    Re-price the existing pending_order with the best offer code says it qualifies for, in
    response to the "Apply my discount" tap. No LLM call and no new products: the customer
    asked for the discount they were already entitled to, not for a different order.

    Re-runs the full money chain (apply_offer -> validate_payment_request), so the re-priced
    quote carries exactly the same guarantees as the original. Sets checkout_confirmed False —
    a new total needs a new, explicit confirmation.
    """
    order = state.get("pending_order") or {}
    line_items = order.get("line_items") or []
    best = discounts.best_eligible_offer(line_items)
    if not line_items or not best:
        return {"messages": [AIMessage(
            content="Let me double-check that for you — I'll come right back.",
            response_metadata={"options": None, "internal_thought": "Apply-offer tap with no eligible offer."},
        )], "apply_offer_requested": False}

    offer_id, _spec = best
    repriced = discounts.apply_offer(line_items, offer_id, order.get("currency", "INR"))
    # apply_offer returns a fresh dict, so the validated add-on has to be carried over explicitly.
    # It matters here more than anywhere: with the offer now applied, _quote_options drops the
    # Apply button and this is the quote that gets the "Add <product>" one.
    if order.get("suggested_complement"):
        repriced["suggested_complement"] = order["suggested_complement"]
    # The step-up survives the same way, and stays valid by construction: this tap changes the
    # discount, not the lines, so the product it replaces and the price gap are both untouched.
    if order.get("suggested_upgrade"):
        repriced["suggested_upgrade"] = order["suggested_upgrade"]

    trusted = {
        li["sku"]: {
            "product_name": li["sku"],
            "base_price": float(li.get("unit_price", 0.0)),
            "installation_fee": float(li.get("installation_fee", 0.0)),
        }
        for li in line_items
    }
    ok, reasons = Guardrails.validate_payment_request(repriced, trusted)
    if not ok:
        logger.error(f"[checkout] re-priced order rejected by the money guardrail: {reasons}")
        return {"messages": [AIMessage(
            content="Let me get that checked properly before I confirm anything — one moment.",
            response_metadata={"options": None, "internal_thought": "Re-priced order failed the guardrail."},
        )], "apply_offer_requested": False}

    logger.info(f"[checkout] applied best eligible offer '{offer_id}' on customer request.")
    # Forced to the quote beat: the Apply button only ever appears on a quote, so the customer is
    # already past the walkthrough and is owed the updated figure, not another step.
    return _advance(
        repriced,
        state.get("consult_stage"),
        lead_in="Done — that's the best offer this order qualifies for:",
        beat="quote",
    )


async def _add_complement_to_order(state: ConversationState) -> dict:
    """
    Add the suggested complement to the existing order and re-quote at whatever tier that reaches,
    in response to the "Add <product>" tap. No LLM call: the product was chosen, validated and
    priced when the quote was built, so this is arithmetic on an order already in state — and a
    model turn here could re-propose the order mid-checkout.

    The new line's price is read from the catalogue rather than from anything in the checkpoint,
    the offer is re-selected by code from the enlarged order, and the whole result goes back
    through validate_payment_request. Sets checkout_confirmed False: a new total needs a new,
    explicit confirmation, exactly as the apply-offer tap does.
    """
    order = state.get("pending_order") or {}
    line_items = order.get("line_items") or []
    comp = order.get("suggested_complement") or {}
    sku = str(comp.get("sku") or "").strip()

    def _fallback(note: str, log: str) -> dict:
        # Re-offer the order WITHOUT the add button: whatever just failed would fail again, and a
        # button that does nothing twice reads as a broken checkout rather than an unavailable
        # product. The existing quote is still valid and still payable.
        intact = {k: v for k, v in order.items() if k != "suggested_complement"}
        logger.warning(f"[upsell] {log}")
        return {
            "messages": [AIMessage(
                content=note,
                response_metadata={"options": _quote_options(intact) if line_items else None,
                                   "internal_thought": "Add-complement tap could not be completed."},
            )],
            "pending_order": intact if line_items else order,
            "add_complement_requested": False,
        }

    if not line_items or not sku:
        return _fallback(
            "Tell me which one you'd like to add and I'll put it on the same order.",
            "add tap with no stored complement.",
        )

    try:
        async with async_session_maker() as session:
            price_map = await PricingEngine(session).get_product_prices_batch([sku])
    except Exception as e:
        return _fallback(
            "I couldn't add that just now — your existing order is still here, though:",
            f"price lookup failed while adding '{sku}': {e}",
        )

    priced = price_map.get(sku)
    if not priced:
        return _fallback(
            "I couldn't add that one to this order — here's where the order stands:",
            f"'{sku}' no longer resolves in the catalogue.",
        )

    # Existing lines keep the unit prices the guardrail already approved; only the new line is
    # priced from source. _process_checkout re-prices everything from the catalogue at mint time,
    # so the amount that reaches Razorpay is never taken from a checkpoint either way.
    trusted = {
        li["sku"]: {
            "product_name": li["sku"],
            "base_price": float(li.get("unit_price", 0.0)),
            "installation_fee": float(li.get("installation_fee", 0.0)),
        }
        for li in line_items
    }
    trusted[sku] = priced
    raw_items = [{"sku": li["sku"], "qty": int(li.get("qty", 1))} for li in line_items]
    raw_items.append({"sku": sku, "qty": 1})

    new_lines, _unresolved, notes = discounts.price_line_items(raw_items, trusted)
    if len(new_lines) <= len(line_items):
        return _fallback(
            "I couldn't add that one to this order — here's where the order stands:",
            f"'{sku}' priced to no new line.",
        )

    best = discounts.best_eligible_offer(new_lines)
    grown = discounts.apply_offer(new_lines, best[0] if best else None, order.get("currency", "INR"))
    if notes:
        grown["audit_notes"] = notes + grown.get("audit_notes", [])
    # apply_offer returns a fresh dict. The step-up survives because it is still true — this tap
    # added a line, it did not touch the one the step-up replaces or the gap between the grades.
    # The complement deliberately does NOT survive: it is a line now, not a suggestion.
    if order.get("suggested_upgrade"):
        grown["suggested_upgrade"] = order["suggested_upgrade"]
    # And neither does the hook. It was written beside the pairing product, about the gap that
    # product fills — so the moment the customer adds it, the hook is a question about something
    # they now own. Live, that pitched the same door phone twice: once as the pairing card, then
    # again as the "want to know more?" question under the price ask, seconds after it went in the
    # basket. Dropping it makes the next ask beat fall back to _HOOK_FALLBACK_TEXT, which is
    # deliberately about nothing in particular and therefore cannot be about a product they have.

    # Every sku in new_lines is canonical, and each unit price is either one the guardrail already
    # approved on this order or the one just read from the catalogue for the added line — so this
    # is the same trust basis _reprice_with_best_offer works from. What the guardrail is re-deriving
    # here is the arithmetic: subtotal, the discount clamp and the total.
    trusted_by_canonical = {
        li["sku"]: {
            "product_name": li["sku"],
            "base_price": float(li.get("unit_price", 0.0)),
            "installation_fee": float(li.get("installation_fee", 0.0)),
        }
        for li in new_lines
    }
    ok, reasons = Guardrails.validate_payment_request(grown, trusted_by_canonical)
    if not ok:
        logger.error(f"[checkout] order rejected by the money guardrail after adding '{sku}': {reasons}")
        return _fallback(
            "Let me get that checked properly before I confirm anything — one moment.",
            f"guardrail rejected the enlarged order: {reasons}",
        )

    logger.info(f"[upsell] added complement '{sku}' on customer tap; offer now {grown.get('applied_offer')}.")
    # Both versions of this sentence are written in code, so the claim of a better rate is made
    # only when the computed percentage actually rose.
    improved = float(grown.get("discount_pct", 0.0) or 0.0) > float(order.get("discount_pct", 0.0) or 0.0)
    # Not forced to a quote: tapping "add it" during the walkthrough answers the pairing beat, and
    # the next thing owed is the offer to price it up — not a price the customer hasn't asked for.
    # From a quote (the walkthrough already spent) the next beat IS the quote, so it re-quotes.
    return _advance(
        grown,
        state.get("consult_stage"),
        lead_in=(
            "Added — and that's moved the whole order to a better rate."
            if improved else "Added to your order."
        ),
    )


async def _swap_upgrade_in_order(state: ConversationState) -> dict:
    """
    Swap the verified step-up into the existing order, in response to the "Switch to ‹X›" tap
    (the label is the differing tail of the name — "Switch to Premium", "Switch to 10 inch").
    No LLM call: the pair came from the closed `discounts.UPGRADES` registry and was validated and
    priced when the order was built, so this is a substitution plus arithmetic.

    The replacement line's price is read from the catalogue rather than from the checkpoint, the
    offer is re-selected over the new lines and the whole result goes back through
    validate_payment_request. The line COUNT is unchanged, so the offer tier normally is too.
    """
    order = state.get("pending_order") or {}
    line_items = order.get("line_items") or []
    up = order.get("suggested_upgrade") or {}
    target = str(up.get("sku") or "").strip()
    replaces = str(up.get("replaces_sku") or "").strip()

    def _fallback(note: str, log: str) -> dict:
        # Carry on WITHOUT the step-up rather than re-offering a button that just failed: the
        # customer's order is untouched and still payable, so the walkthrough simply moves on.
        intact = {k: v for k, v in order.items() if k != "suggested_upgrade"}
        logger.warning(f"[upsell] {log}")
        update = _advance(intact, state.get("consult_stage"), lead_in=note)
        return update

    if not line_items or not target or not replaces:
        return _fallback(
            "Let me keep that as it is for now.",
            "swap tap with no stored step-up.",
        )

    try:
        async with async_session_maker() as session:
            price_map = await PricingEngine(session).get_product_prices_batch([target])
    except Exception as e:
        return _fallback(
            "I couldn't switch that one just now — I've left your order as it was.",
            f"price lookup failed while swapping in '{target}': {e}",
        )

    priced = price_map.get(target)
    if not priced:
        return _fallback(
            "I couldn't switch that one — I've left your order as it was.",
            f"'{target}' no longer resolves in the catalogue.",
        )

    # Existing lines keep the unit prices the guardrail already approved; only the incoming line is
    # priced from source. _process_checkout re-prices everything at mint time regardless.
    trusted = {
        li["sku"]: {
            "product_name": li["sku"],
            "base_price": float(li.get("unit_price", 0.0)),
            "installation_fee": float(li.get("installation_fee", 0.0)),
        }
        for li in line_items
    }
    trusted[target] = priced

    raw_items, swapped_any = [], False
    for li in line_items:
        if str(li.get("sku") or "").strip().lower() == replaces.lower():
            raw_items.append({"sku": target, "qty": int(li.get("qty", 1) or 1)})
            swapped_any = True
        else:
            raw_items.append({"sku": li["sku"], "qty": int(li.get("qty", 1) or 1)})
    if not swapped_any:
        return _fallback(
            "I couldn't switch that one — I've left your order as it was.",
            f"'{replaces}' is no longer a line in this order.",
        )

    new_lines, _unresolved, notes = discounts.price_line_items(raw_items, trusted)
    if len(new_lines) != len(line_items):
        return _fallback(
            "I couldn't switch that one — I've left your order as it was.",
            f"swapping '{replaces}' -> '{target}' changed the line count.",
        )

    best = discounts.best_eligible_offer(new_lines)
    swapped = discounts.apply_offer(new_lines, best[0] if best else None, order.get("currency", "INR"))
    if notes:
        swapped["audit_notes"] = notes + swapped.get("audit_notes", [])
    # The step-up is spent — it is the order now. The pairing is not, so it survives to its own beat.
    if order.get("suggested_complement"):
        swapped["suggested_complement"] = order["suggested_complement"]
    if order.get("explore_hook"):
        swapped["explore_hook"] = order["explore_hook"]

    trusted_by_canonical = {
        li["sku"]: {
            "product_name": li["sku"],
            "base_price": float(li.get("unit_price", 0.0)),
            "installation_fee": float(li.get("installation_fee", 0.0)),
        }
        for li in new_lines
    }
    ok, reasons = Guardrails.validate_payment_request(swapped, trusted_by_canonical)
    if not ok:
        logger.error(f"[checkout] order rejected by the money guardrail after swapping '{target}': {reasons}")
        return _fallback(
            "Let me get that checked properly before I confirm anything — one moment.",
            f"guardrail rejected the swapped order: {reasons}",
        )

    logger.info(f"[upsell] swapped '{replaces}' -> '{target}' on customer tap.")
    display = up.get("display_name") or discounts.plain_product_name(target) or target
    return _advance(
        swapped,
        state.get("consult_stage"),
        lead_in=f"Done — I've put the {display} down for you instead.",
    )


def _build_handoff_block(state: ConversationState) -> str:
    """
    What a colleague did while they held this thread, injected on the turns after they released it.

    This is the only context that crosses back. The human works on a different WhatsApp number, so
    nothing they said to the customer is visible here — `handoff_notes` (written by
    src/scripts/resolve_handoff.py) is the entire channel. Empty string when nothing was handed
    back, which keeps the block absent rather than announcing that we know nothing.
    """
    notes = [n for n in (state.get("handoff_notes") or []) if n]
    if not notes:
        return ""
    recent = notes[-3:]
    lines = "\n".join(f"- {n}" for n in recent)
    return (
        "\n\nWHILE A COLLEAGUE WAS HANDLING THIS CONVERSATION (you were not part of it — this "
        "summary is all you have):\n"
        f"{lines}\n"
        "Carry on from there. Do not ask the customer to repeat any of it, do not re-offer "
        "something already resolved, and do not pretend you were present for the call. If the note "
        "says the order was completed, treat it as completed."
    )


def _care_claim_in(text: str) -> Optional[str]:
    """
    The first care/medical/safety-monitoring phrase in the agent's own words, or None.

    Scans only what the customer is about to read. A hit is logged, never blocked: the phrasing that
    matters most here is the phrasing this list does not contain, so treating a miss as an all-clear
    would be the wrong lesson to build in. See _CARE_CLAIM_PHRASES for why the existing
    `specs_unavailable` guard cannot cover this class of error.
    """
    if not text:
        return None
    lowered = text.lower()
    for phrase in _CARE_CLAIM_PHRASES:
        if phrase in lowered:
            return phrase
    return None


# Openers that congratulate the customer for spending. Banned in CONVERSATION_STYLE, emitted anyway
# in the live transcript ("Good choice!"), and read as sales patter at precisely the moment a buyer
# is most alert to being sold to — just after committing. Longest variants first so
# "excellent choice" is matched before bare "excellent".
#
# Bare "great", "nice" and "perfect" are deliberately absent: CONVERSATION_STYLE allows a
# three-word acknowledgement ("Got it." / "Right —" / "Perfect.") and stripping that would leave the
# reply opening cold on a customer who has just told us something. What is banned is praising the
# CHOICE, not acknowledging the message.
_PRAISE_WORDS = ("excellent", "wonderful", "fantastic", "awesome", "brilliant", "lovely", "amazing")

# The same words plus the ones that are praise ONLY when they are paid to something. Each noun below
# names a decision rather than a product, so "Perfect choice" is a compliment while "Perfect for a
# rented flat" is a recommendation and survives untouched.
_PRAISE_QUALIFIERS = _PRAISE_WORDS + ("good", "great", "solid", "smart", "nice", "perfect")
_PRAISE_NOUNS = ("choice", "pick", "call", "idea")

# Longest first, and generated rather than hand-listed. Enumerating these by hand is what left
# "amazing" banned on its own while "amazing choice" was not: the bare word matched, the noun didn't
# go with it, and the customer read "Choice, that one's popular." A fragment is worse than the
# compliment it came from.
_PRAISE_OPENERS = tuple(sorted(
    {f"{word} {noun}" for word in _PRAISE_QUALIFIERS for noun in _PRAISE_NOUNS} | set(_PRAISE_WORDS),
    key=lambda opener: (-len(opener), opener),
))

# Openers where the whole SENTENCE is the compliment, so the sentence goes rather than the words:
# trimming "That's a great way to boost your security." down to its opener leaves "Way to boost your
# security.", which is worse than either the original or nothing.
_PRAISE_SENTENCES = (
    "that's a great", "thats a great", "that's a fantastic", "thats a fantastic",
    "that's an excellent", "thats an excellent", "that's a smart", "thats a smart",
    "that's a really good", "thats a really good", "what a great", "what a good",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Below this, what survives trimming is filler ("Sure.", "Got it.") sitting above a card that says
# the same thing better, so the bubble goes rather than being sent as a fragment.
_LEAD_IN_MIN_CHARS = 12


def _strip_praise_opener(text: str) -> str:
    """
    Remove an opening compliment ("Good choice!", "Excellent —") from the model's own words.

    What follows is re-capitalised, so a stripped opener doesn't leave the sentence starting
    mid-thought. Returns the text unchanged when it opens on something substantive.

    A bare praise word only counts when it ends on a word boundary — "Excellently quiet" is a
    description of a product, not a compliment paid to the customer. `_PRAISE_OPENERS` is ordered
    longest-first, so a two-word compliment is consumed whole rather than leaving its noun behind.
    """
    body = (text or "").lstrip("*_ \t")
    lowered = body.lower()

    for opener in _PRAISE_SENTENCES:
        if lowered.startswith(opener):
            parts = _SENTENCE_SPLIT_RE.split(body, maxsplit=1)
            rest = (parts[1] if len(parts) > 1 else "").strip()
            return (rest[0].upper() + rest[1:]) if rest else ""

    for opener in _PRAISE_OPENERS:
        if not lowered.startswith(opener):
            continue
        if lowered[len(opener):len(opener) + 1].isalnum():
            continue
        rest = body[len(opener):].lstrip("*_").lstrip(" !.,:;—–-").strip()
        return (rest[0].upper() + rest[1:]) if rest else ""
    return text


def _menued_registry_pair(text: str, options) -> Optional[tuple]:
    """
    The step-up pair this turn put on screen TOGETHER, as (from, to), or None.

    CONVERSATION_STYLE forbids showing both models of one product while neither has a price against
    it: it hands the customer a choice between tiers they cannot see, and it spends the step-up beat
    in advance — the beat is the only place the exact difference and the swap framing ever appear.
    A live chat did exactly that during discovery.

    Observability, not a guarantee. The offending words are the model's own prose, and rewriting a
    whole reply mid-turn is a far bigger risk than a chatty one; what this buys is a countable line
    in the worker log instead of an impression. Matched on the catalogue names only — never on the
    plain-English gloss, which is identical for both halves of a pair and would fire on either one
    alone.
    """
    parts = [text or ""]
    for opt in options or []:
        parts.append(str(getattr(opt, "label", "") or ""))
        parts.append(str(getattr(opt, "description", "") or ""))
    haystack = _norm_alnum(" ".join(parts))
    if not haystack:
        return None
    for frm, spec in discounts.UPGRADES.items():
        to = str((spec or {}).get("to") or "")
        if _norm_alnum(frm) in haystack and _norm_alnum(to) in haystack:
            return frm, to
    return None


def _foreign_product_norms(order: dict, catalogue: Optional[List[str]] = None) -> set:
    """
    Normalised names of the products this order does NOT contain: the pairing suggestion, the
    step-up target, and everything else the catalogue can price.

    A name that is part of one the order does own is left out, which is what lets prose say "the
    Smart Door Lock" to a customer who chose the Base while the grade word itself stays on the card
    and the buttons. Normalisation is alphanumeric-only, so "6 SW" and "6sw" are one string — the
    same trick rag.py::_mentions uses.
    """
    own = set()
    for li in (order or {}).get("line_items") or []:
        sku = str(li.get("sku") or "")
        for variant in (sku, discounts.plain_product_name(sku)):
            owned = _ALNUM_RE.sub("", (variant or "").lower())
            if owned:
                own.add(owned)

    candidates: List[str] = []
    for block in ("suggested_complement", "suggested_upgrade"):
        info = (order or {}).get(block) or {}
        candidates.extend([str(info.get("sku") or ""), str(info.get("display_name") or "")])
    candidates.extend(str(name or "") for name in (catalogue or []))

    foreign = set()
    for name in candidates:
        for variant in (name, discounts.plain_product_name(name)):
            norm = _ALNUM_RE.sub("", (variant or "").lower())
            if not norm or any(norm in owned for owned in own):
                continue
            foreign.add(norm)
    return foreign


def beat_lead_in(
    text: str,
    beat: str = "",
    order: Optional[dict] = None,
    catalogue: Optional[List[str]] = None,
) -> str:
    """
    Trim the model's own words down to something that belongs above the code-built message that
    follows it — a walkthrough beat or a quote — or to nothing at all.

    Three things come out, in this order:

    * an opening compliment, which congratulates the customer for spending before they have even
      seen a price;
    * any sentence naming a product this order does not contain. This is the one that matters: in
      the live transcript the model pitched the Video Door Phone in prose a full turn BEFORE its
      button existed, so the customer was asked about something they had no way to say yes to. A
      lead-in may only talk about the beat underneath it, and that beat's body already names the
      product it is about;
    * a question. Every beat ends in buttons, and a question above them leaves the customer unsure
      whether to answer or tap.

    If what is left is too short to be saying anything, the bubble is dropped entirely — one clean
    message beats a fragment plus a card.

    Only the quote beat gets a stitched hand-off, and only when something was actually removed: a
    sentence that has just lost its ending may no longer read as an introduction, while an untouched
    one already does. It reads "Here's what it comes to:" — the wording it replaces promised "the
    full breakdown" above every beat, including a step-up card with no breakdown anywhere near it.
    """
    original = (text or "").strip()
    if not original:
        return ""

    foreign = _foreign_product_norms(order or {}, catalogue)
    kept: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(_strip_praise_opener(original)):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?"):
            continue
        norm = _ALNUM_RE.sub("", sentence.lower())
        if norm and any(name in norm for name in foreign):
            continue
        kept.append(sentence)

    trimmed = " ".join(kept).strip()
    if len(trimmed) < _LEAD_IN_MIN_CHARS:
        return ""
    if beat == "quote" and trimmed != original:
        return f"{trimmed}\n\nHere's what it comes to:"
    return trimmed


def _build_brochure_block(state: ConversationState) -> str:
    """
    Whether the agent may offer the brochure at all, and how to actually send it.

    Config-gated on settings.brochure_url because the failure this fixes was the agent promising a
    lookbook, being asked to send it to WhatsApp, and replying "here's our digital lookbook" while
    sending nothing. An offer the system cannot honour is worse than no offer: it spends the
    customer's trust on the one thing they explicitly asked for.
    """
    if not settings.brochure_url:
        return (
            "\n\nBROCHURE: there is NO brochure or lookbook you can send right now. Do not offer "
            "one, do not say you will send one, and never claim you have. If they ask for a "
            "brochure, say plainly that you can't send it here yet and offer what you CAN do "
            "instead — answer anything they want to know, or arrange a free consultation, a demo, "
            "or an experience-centre visit."
        )
    return (
        "\n\nBROCHURE: you can send the Otohom lookbook as a PDF. To do it, set `send_brochure` "
        "to true — the system attaches the file for you. Say you're sending it in the SAME message "
        "that sets the flag, never in an earlier one, and don't ask whether they'd like it by "
        "WhatsApp or email: this is WhatsApp, so it arrives here. Only set the flag when they've "
        "actually asked for it."
    )


def _build_grounding_block(state: ConversationState) -> str:
    """
    A hard stop on inventing product detail when retrieval brought back nothing.

    The prose rules elsewhere say "stay grounded", and in testing the model still credited a lock
    with a video doorbell it does not have. The difference here is that this block is *conditional*:
    it appears only on the turns where there is genuinely nothing to ground against, which is
    exactly when the temptation to fill the gap is strongest, and it names the escape route so the
    model has something safe to do instead of guessing.
    """
    if state.get("context_chunks"):
        return ""
    return (
        "\n\nNO PRODUCT DATA WAS RETRIEVED FOR THIS TURN.\n"
        "You therefore do NOT know, and must not state, any specific detail about any product: no "
        "features, no unlock methods, no materials, colours, dimensions, battery, compatibility, "
        "model differences or comparisons. Not even ones you feel certain about.\n"
        "What you may still do: talk about categories and outcomes at the level of the overview "
        "above, ask what matters most to them, and offer to confirm the exact specifics — \"let me "
        "get you the exact details on that\" is always a better answer than a confident guess. If "
        "they asked you to compare two products, say you'll pull the exact spec rather than "
        "describing either from memory."
    )


# The two asks at the pay button. A blank line after the question on purpose: these arrive at the
# most decisive moment of the sale, and a two-line block with no gap reads as one sentence.
#
# Neither of them says the city is optional. It is, in the sense that nothing blocks on it — but
# telling the customer so guarantees they skip it, and then the team has an order with a name on it
# and nowhere to send an installer. So the second ask names the easiest way to answer instead, and
# the code moves on with whatever comes back.
_PAY_DETAILS_ASK = (
    "Almost there — what name should I put on the order?\n\n"
    "Your city too, so we can book the right installation team."
)
_PAY_CITY_ASK = (
    "Thanks, {name}. Which city is this going to?\n\n"
    "_You can drop a pin instead if that's easier — the 📎 icon, then Location._"
)

# Replies that mean "moving on", not a place. Storing one of these would put "nope" on the order,
# the sheet and the installer's job card.
_CITY_DECLINES = frozenset({
    "no", "nope", "nah", "skip", "later", "na", "n/a", "-", "none", "nothing",
    "not now", "don't", "dont", "rather not", "next", "pass",
})


def _parse_city(text: str) -> Optional[str]:
    """
    Take a place out of the reply to "which city?" — or None when they moved on instead.

    Deliberately generous: whatever they put in that box is what they meant, including a dropped pin
    (which `processing.py` turns into the place name, the address or coordinates before the graph
    sees it). It refuses only the shapes that are plainly not an answer, because the cost of a wrong
    value here is an installer sent to the wrong place, and the cost of refusing is nothing — the
    link is minted either way.
    """
    raw = " ".join((text or "").split()).strip(" ,.")
    if not raw or "[" in raw or "?" in raw or len(raw) > 80:
        return None
    if raw.lower() in _CITY_DECLINES:
        return None
    if not any(ch.isalnum() for ch in raw):
        return None
    return raw


def _parse_name_and_city(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Read a name (and a city, if they gave one) out of the reply to the ask above.

    Deliberately narrow and deliberately code, not a model: this runs at the pay button, where an
    LLM turn would cost the customer several seconds at the worst possible moment and could
    re-propose the order instead of letting it complete. The ask is worded to invite exactly this
    shape, so the rule is "the reply IS the answer" — split on a comma, first part the name, second
    the city.

    It refuses rather than guesses when the reply is clearly not a name: a button tap (`[POSTBACK]`),
    a question, something long enough to be a sentence, or text with no letters in it. Those fall
    through to an ordinary turn so the agent can answer whatever they actually said, and the ask is
    still outstanding. Storing "why do you need my name?" as a customer's name would put it on the
    order, the sheet and the receipt.
    """
    raw = " ".join((text or "").split())
    if not raw or "[" in raw or "?" in raw or len(raw) > 60:
        return None, None
    if not any(ch.isalpha() for ch in raw):
        return None, None

    # "I'm Anil", "my name is Anil", "this is Anil from Kochi"
    lowered = raw.lower()
    for lead in ("my name is", "name is", "i am", "i'm", "im", "this is", "it's", "its"):
        if lowered.startswith(lead + " "):
            raw = raw[len(lead) + 1:].strip()
            break

    name, city = raw, None
    for separator in (",", " from ", " in "):
        if separator in raw or separator in raw.lower():
            head, _, tail = raw.partition(separator) if separator == "," else raw.lower().partition(separator)
            if separator != ",":
                # Recover the original casing for both halves.
                head = raw[:len(head)]
                tail = raw[len(raw) - len(tail):]
            name, city = head.strip(" ,"), tail.strip(" ,")
            break

    name = name.strip(" ,.")
    city = (city or "").strip(" ,.") or None
    if not name or len(name) > 40:
        return None, None
    return name, city


async def _execute_sales_node(state: ConversationState, prompt_template) -> dict:



    # ── The replies we asked for at the pay button: the name, then the city ────────────────
    # Checked BEFORE the confirm path, because `checkout_confirmed` is still True on these turns
    # (triage returns only the keys it changes) and the confirm path would otherwise ask again
    # forever. Which of the two we are waiting for is decided by whether a name is on file, so one
    # flag covers both steps.
    if (
        settings.AGENT_FULL_AUTONOMY
        and state.get("awaiting_pay_details")
        and state.get("pending_order")
        and not state.get("payment_link_sent")
    ):
        reply = last_user_text(state, lower=False)
        on_file = (state.get("customer_name") or "").strip()

        # Step two: we have the name and asked where it's going. Whatever comes back, the link goes
        # out — the city is worth asking for once and never worth blocking a payment over.
        if on_file:
            city = _parse_city(reply) if "[" not in reply else None
            logger.info(f"[checkout] pay details complete (city {'given' if city else 'not given'}); minting.")
            return {
                "city": city or state.get("city"),
                "awaiting_pay_details": False,
                "checkout_confirmed": True,
                "messages": [AIMessage(
                    content="Generating your secure payment link now — one moment.",
                    response_metadata={
                        "options": None,
                        "internal_thought": "City step answered (or skipped); worker mints the link.",
                    },
                )],
            }

        given_name, given_city = _parse_name_and_city(reply)
        if given_name and given_city:
            logger.info("[checkout] name and city captured at the pay button; minting now.")
            return {
                "customer_name": given_name,
                "city": given_city,
                "awaiting_pay_details": False,
                # Still the same authorisation from the same tap — nothing about the order changed.
                "checkout_confirmed": True,
                "messages": [AIMessage(
                    content=f"Thanks, {given_name} — generating your secure payment link now.",
                    response_metadata={
                        "options": None,
                        "internal_thought": "Name and city captured deterministically; worker mints the link.",
                    },
                )],
            }
        if given_name:
            # A name and nothing else. Ask once where it's going, keeping the flag set so the next
            # message is read as the answer — and keeping the authorisation, since the customer has
            # done nothing to withdraw it.
            logger.info("[checkout] name captured; asking where the order is going before minting.")
            return {
                "customer_name": given_name,
                "awaiting_pay_details": True,
                "checkout_confirmed": True,
                "messages": [AIMessage(
                    content=_PAY_CITY_ASK.format(name=given_name),
                    response_metadata={
                        "options": None,
                        "internal_thought": "Name on file; asking for the city once before the link.",
                    },
                )],
            }
        if "[" in reply:
            # They tapped a button again instead of answering. Re-ask rather than fall through to a
            # model turn that would re-quote and quietly drop the authorisation.
            return {
                "awaiting_pay_details": True,
                "messages": [AIMessage(
                    content=_PAY_DETAILS_ASK,
                    response_metadata={
                        "options": None,
                        "internal_thought": "Pay tapped again while the name is still outstanding; re-asking.",
                    },
                )],
            }

    # ── The confirm-tap turn needs no model at all ───────────────────────────────────────
    # triage has already set checkout_confirmed in code, the order is priced and sitting in
    # state, and the worker mints the link straight after this turn. An LLM call here buys
    # nothing, costs the customer several seconds at the single most important moment of the
    # sale, and risks the model re-proposing the order instead of letting it complete.
    if (
        settings.AGENT_FULL_AUTONOMY
        and state.get("checkout_confirmed")
        and state.get("pending_order")
        and not state.get("payment_link_sent")
        # Asked once already: a reply that wasn't a name goes to the model rather than being met
        # with the same question again.
        and not state.get("awaiting_pay_details")
    ):
        # One thing has to be true before money changes hands: we must know who is buying. The
        # client's rule — the name is mandatory at the pay button. Asked HERE rather than after the
        # receipt, where it used to live: a question that arrives after the money is done gets
        # ignored, and the team then has an order with nobody's name on it.
        if not (state.get("customer_name") or "").strip():
            logger.info("[checkout] confirm tap held: asking for the name before minting.")
            return {
                "awaiting_pay_details": True,
                "messages": [AIMessage(
                    content=_PAY_DETAILS_ASK,
                    response_metadata={
                        "options": None,
                        "internal_thought": "Confirm tap held: name is mandatory before a link is minted.",
                    },
                )],
            }
        logger.info("[checkout] confirm tap: deterministic ack, no LLM call.")
        return {
            "awaiting_pay_details": False,
            "messages": [AIMessage(
                content="Perfect. Generating your secure payment link now — one moment.",
                response_metadata={
                    "options": None,
                    "internal_thought": "Confirm tap acknowledged deterministically; worker mints the link.",
                },
            )]
        }

    # ── The apply-discount tap is also pure code ─────────────────────────────────────────
    # Same reasoning: the answer is arithmetic on an order already in state, and a model turn
    # here could talk about a discount whose size it doesn't get to decide.
    if settings.AGENT_FULL_AUTONOMY and state.get("apply_offer_requested") and state.get("pending_order"):
        return await _reprice_with_best_offer(state)

    # ── So is adding the product the agent suggested ──────────────────────────────────────
    # The product was chosen by the model, then validated and priced by code when the quote was
    # built, so the tap is arithmetic too. Placed above the specs_unavailable check below for the
    # same reason as the two taps above it: a grounding refusal must never swallow a checkout tap.
    if settings.AGENT_FULL_AUTONOMY and state.get("add_complement_requested") and state.get("pending_order"):
        return await _add_complement_to_order(state)

    # ── The three walkthrough taps are pure code as well ──────────────────────────────────
    # Every beat between "I'll take it" and a price is rendered from data validated when the order
    # was built, so a swap, a pass and a request to be priced are all answered here. That keeps the
    # walkthrough at zero added LLM calls — it costs latency nowhere — and, more importantly, means
    # the model cannot re-propose the order in the middle of it.
    if settings.AGENT_FULL_AUTONOMY and state.get("pending_order"):
        if state.get("swap_upgrade_requested"):
            return await _swap_upgrade_in_order(state)
        if state.get("consult_next_requested"):
            # Tapped past a beat without taking it. Nothing about the money changes; the customer
            # keeps exactly what they already chose and the walkthrough moves to the next beat.
            return _advance(state["pending_order"], state.get("consult_stage"))
        if state.get("quote_now_requested"):
            return _advance(state["pending_order"], state.get("consult_stage"), beat="quote")

    # ── Named a product we have no data on: answer honestly, in code ─────────────────────
    # Set by the RAG node when the customer named a specific product and retrieval returned
    # nothing. An LLM turn here fills the gap from pretraining — observed live, it credited a lock
    # with a video doorbell it doesn't have — and a prohibition buried in a 24k-character prompt did
    # not hold. So the reply is written here, where it cannot drift, and the conversation still
    # moves: they get a real next step rather than a dead "I don't know".
    if state.get("specs_unavailable"):
        logger.info("[grounding] named product with no retrieved data: deterministic honest reply.")
        return {
            "messages": [AIMessage(
                content=(
                    "I don't want to guess at the exact specs on that one — let me get them "
                    "confirmed for you rather than tell you something that turns out to be wrong.\n\n"
                    "In the meantime, tell me what matters most for your place and I'll point you to "
                    "the right fit."
                ),
                response_metadata={
                    "options": [
                        {"label": "Book a free call", "postback_id": "INTENT_CONSULTATION",
                         "description": "Someone from the team walks you through the options"},
                        {"label": "See what we do", "postback_id": "INTENT_OVERVIEW",
                         "description": "A quick tour of the categories we cover"},
                        {"label": "Talk to a person", "postback_id": "CONNECT_NOW",
                         "description": "Hand this straight to a colleague"},
                    ],
                    "internal_thought": "specs_unavailable: refused to state unverified product detail.",
                },
            )],
            "specs_unavailable": False,
        }

    # Warmth comes from the prompt, not from sampling randomness. At 0.6 the same input produced a
    # well-structured two-model breakdown on one run and a one-line brush-off on the next; 0.4 keeps
    # replies varied enough to feel human while actually following the formatting rules.
    llm = LLMFactory.get_llm(temperature=0.4)

    # Build the RAG context block (empty string if no RAG context available)
    rag_context_block = _build_rag_context_block(state)
    catalogue_names = await _load_catalogue_names()

    # Build the full candidate variable set, then pass ONLY the variables THIS template
    # declares. Keeps every prompt KeyError-proof regardless of the LangChain formatter's
    # strictness and drops dead vars (e.g. GENERAL_GREETING needs only chat_history;
    # language_preference is referenced by no prompt).
    candidate_vars = {
        "chat_history": state["messages"],
        "property_type": state.get("property_type", "Unknown"),
        "budget_tier": state.get("budget_tier", "Unknown"),
        "pain_point": state.get("pain_points", ["Unknown"])[-1] if state.get("pain_points") else "Unknown",
        "primary_interest": state.get("primary_interest") or "Smart Home Options",
        "deferred_purchase_intent": state.get("deferred_purchase_intent", "None"),
        "rag_context_block": rag_context_block,
        # Durable facts recalled from mem0 for this person (empty string when semantic memory
        # is off, unavailable, or has nothing on them yet) — see src/memory/semantic.py.
        "memory_block": format_memory_block(state.get("memory_facts") or []),
        # What a colleague did while they held the thread — the only context that crosses back
        # from their separate number. Empty string when no handoff has happened.
        "handoff_block": _build_handoff_block(state),
        # Whether a brochure can actually be sent, plus a hard grounding stop on the turns where
        # retrieval returned nothing. Both are appended to the same slot so a prompt only has to
        # declare one variable.
        "brochure_block": _build_brochure_block(state) + _build_grounding_block(state),
        # The single authority on money for this turn — full autonomy or the merchant's original
        # pricing policy, decided by AGENT_FULL_AUTONOMY. Every sales prompt declares it.
        "pricing_policy_block": _build_pricing_policy_block(state, catalogue_names),
        # Live availability guidance; only prompts that declare {business_status} receive it.
        "business_status": business_status_line(),
    }
    prompt_vars = {k: v for k, v in candidate_vars.items() if k in prompt_template.input_variables}
    formatted_prompt = prompt_template.format_messages(**prompt_vars)

    response = await execute_vendor_agnostic_node(llm, formatted_prompt, NodeExecutionSchema, "sales_node")

    if not response:
        logger.error("Sales execution failed. Escaping to Human.")
        fallback_msg = AIMessage(
            content="Something went wrong on my side just now. Rather than waste your time, I'm bringing in a colleague who can pick this up properly.",
            response_metadata={"options": None, "internal_thought": "LLM failed. Graceful fallback to human."}
        )
        return {"messages": [fallback_msg], "requires_human_handoff": True, "handoff_reason": "error"}

    # Two options that look identical after WhatsApp's 20-character cut are worse than one: the
    # customer is asked to choose between "Smart Door Lock…" and "Smart Door Lock…". Logged rather
    # than blocked — losing the whole reply over a label would cost more than the ambiguity.
    if response.options and len(response.options) > 1:
        shortened = [fit_label(o.label, 20) for o in response.options]
        if len(set(shortened)) < len(shortened):
            logger.warning(f"[style] option labels collide once shortened: {shortened}")

    # A care/medical/monitoring claim is the one hallucination class worth seeing the day it lands,
    # because the person acting on it may be caring for someone. Logged at ERROR, deliberately not
    # blocked: a keyword list cannot tell a claim from a correct refusal ("these don't alert you if
    # there's no movement" contains the same words), and suppressing the reply would break the honest
    # answer as often as the invented one. The prohibition itself lives in prompts.GUARDRAIL_RULES.
    care_phrase = _care_claim_in(response.conversational_text)
    if care_phrase:
        logger.error(
            f"[grounding] possible care/monitoring capability claim ({care_phrase!r}) — "
            "verify against docs/catalog/sensors/; Otohom sensors report events, not their absence."
        )
    # Everything this agent has already said on this thread. One check needs it: a step-up must not
    # be offered for a product the customer has already been shown and passed over, and that is a
    # decided either/or rather than an open question (_validate_upgrade). Built from the agent's own
    # turns only — a customer naming a product is them asking about it, not us having pitched it.
    prior_agent_text = "\n".join(
        str(m.content) for m in (state.get("messages") or [])
        if isinstance(m, AIMessage) and getattr(m, "content", None)
    )

    # If the LLM proposed products AND this isn't already the confirm-tap turn, price the order in
    # CODE and show the customer the NEXT BEAT of the walkthrough — which is a price only if they
    # actually asked for one. On the confirm turn itself (triage already set checkout_confirmed) we
    # deliberately do NOT rebuild: that would reset the dedup flags below and cancel the pending
    # mint. We just let the worker act on the order already in state.
    checkout_updates: dict = {}
    walkthrough_msgs: list = []
    order: dict = {}
    beat: str = ""
    is_confirm_turn = state.get("checkout_confirmed", False)
    # A quote carries a pay button, so it must never be built for something already bought. The
    # paid order is out of state by now, so this cannot re-mint anything — it stops the model
    # walking the customer back to a checkout they have already been through.
    reproposal = _reproposes_paid_order(
        getattr(response, "checkout_items", None), state.get("paid_line_items") or {}
    )
    if reproposal:
        logger.warning("[checkout] model re-proposed an already-paid order; quote suppressed.")
    if (
        settings.AGENT_FULL_AUTONOMY
        and getattr(response, "checkout_items", None)
        and not is_confirm_turn
        and not reproposal
    ):
        order = await _build_pending_order(
            response.checkout_items,
            response.applied_offer,
            getattr(response, "suggested_complement", None),
            getattr(response, "complement_reason", None),
            getattr(response, "suggested_upgrade", None),
            getattr(response, "upgrade_replaces", None),
            getattr(response, "upgrade_reason", None),
            getattr(response, "explore_hook", None),
            prior_agent_text,
        ) or {}
        if order:
            # A walkthrough belongs to the sale it is walking. An order sharing nothing with that
            # one is a new sale — they came back and picked something else — and its own step-up has
            # never been shown, so the beats start again. An order that merely grew or changed
            # quantity is the same sale and keeps its place, because being sent back to beat one
            # after asking for one more panel would read as the agent losing the thread.
            stage = state.get("consult_stage")
            if _is_new_sale(order, state.get("consult_order_key")):
                logger.info("[walkthrough] priced order shares no product with the walked one; beats restart.")
                stage = 0
            # Choosing a product is not asking what it costs. The walkthrough shows the dearer
            # model, then the product that pairs with it, and only then offers to price it up —
            # UNLESS the customer asked for the figure in so many words, in which case they get it
            # immediately and the rest is skipped. Whichever it is, code writes the message.
            asked_outright = bool(getattr(response, "quote_requested", False))
            beat = "quote" if asked_outright else _next_beat(order, stage)
            # "Shall I show you the price?" has exactly two answers: the `Yes, show the price` tap
            # (a deterministic fast path that never reaches this function) or saying so in words.
            # Every OTHER turn while that question is on screen HOLDS — the order stays priced in
            # state and the total stays off it. Without this the button documented as the way out of
            # being priced did the pricing: tapping the hook is an ordinary LLM turn, the model
            # re-proposed the same products, and _next_beat saw a stage already past the ask and
            # went straight to a quote. The customer was priced by the button that exists so they
            # can't be.
            #
            # Narrowed to the ask itself (stage 3) on purpose. Once the price HAS been shown, a
            # typed "make it two" must come back re-priced — that is the editable-order guarantee,
            # and holding there would silently swallow the edit the customer just asked for.
            if beat == "quote" and not asked_outright and stage == _STAGE_AFTER["quote_ask"]:
                logger.info(
                    "[walkthrough] holding the price: the order is priced and stored, but the "
                    "customer has not tapped for it or asked in words."
                )
                beat = "hold"
            checkout_updates = _advance(order, stage, beat=beat)
            walkthrough_msgs = checkout_updates.pop("messages", [])

    # Observability backstop for the one-question rule, counted on the model's own words only. A
    # turn that ends in buttons can answer one question, and the quote-ask beat's two questions are
    # code's, each with its own button — so a code-built body must never trip a warning aimed at the
    # model. Gated on the beats as well as on `options`, because before this it could not see a
    # walkthrough turn at all: a beat carries its buttons on its own message, so `response.options`
    # is None on exactly the turns the rule matters most. Logged, never blocked — a chatty reply is
    # a much smaller problem than a dropped turn.
    asked = response.conversational_text.count("?")
    if asked > 1 and (response.options or walkthrough_msgs or beat == "hold"):
        logger.warning(
            f"[style] {asked} questions in the model's own words on a turn that ends in buttons — "
            "they can only answer one of them."
        )

    # Same kind of backstop for the other rule the live transcript broke: a discovery turn that puts
    # both models of one product on screen. That costs the step-up beat, because a step-up card the
    # customer has already been shown is not offered again — so the menu trades the one message that
    # carries the exact difference for a choice between two prices they have not seen.
    menued = _menued_registry_pair(response.conversational_text, response.options)
    if menued:
        logger.warning(
            f"[style] both models of one product on screen together: '{menued[0]}' and "
            f"'{menued[1]}'. Recommend one with a line of why and let the step-up beat introduce "
            "the dearer model, where the price difference is."
        )

    # Pack the LLM response into an AIMessage with response_metadata so TaskIQ can read the UI
    # options. When a code-built beat or quote follows, the buttons live on THAT message, so the
    # warm lead-in carries no options of its own — and beat_lead_in decides what, if anything, it is
    # still allowed to say above them.
    lead_in = response.conversational_text
    if walkthrough_msgs:
        trimmed = beat_lead_in(lead_in, beat, order, catalogue_names)
        if trimmed != (lead_in or "").strip():
            logger.info(f"[style] trimmed a lead-in that precedes buttons: {lead_in!r} -> {trimmed!r}")
        lead_in = trimmed

    out_messages = list(walkthrough_msgs)
    # The way back to the price has to survive a browse turn. While "Shall I show you the price?" is
    # the question on the table and no total has been shown, every LLM reply carries the answer to it
    # in slot one — whether or not the model re-proposed the order this turn, because a turn that
    # quietly left `checkout_items` out would otherwise leave the tap that shows the total reachable
    # only by typing. Once the price HAS been shown the stage is past the ask and this stops.
    own_options = [opt.model_dump() for opt in response.options] if response.options else None
    price_ask_unanswered = (
        not walkthrough_msgs
        and settings.AGENT_FULL_AUTONOMY
        and int(state.get("consult_stage") or 0) == _STAGE_AFTER["quote_ask"]
        and bool(checkout_updates.get("pending_order") or state.get("pending_order"))
    )
    if beat == "hold" or price_ask_unanswered:
        own_options = _keep_price_reachable(own_options)
    # An empty lead-in means nothing survived the trim, so the bubble is dropped and the beat speaks
    # for itself. Without beats there is nothing else to send, so the reply goes out as it came.
    if (lead_in or "").strip() or not walkthrough_msgs:
        out_messages.insert(0, AIMessage(
            content=lead_in,
            response_metadata={
                "options": None if walkthrough_msgs else own_options,
                "internal_thought": response.internal_thought,
            },
        ))

    # Append the turn's pain point, then dedupe. The model re-states the same concern with different
    # casing on nearly every turn ("front door safety" / "Front door safety"), so an unconditional
    # append put "front door safety; Front door safety; Front door safety" in one spreadsheet cell.
    # dedupe_keeping_first also caps the list, because a twenty-turn chat otherwise writes a
    # paragraph there. One helper, so the sheet, the prompt and the digest all see the same list.
    pain_points_list = list(state.get("pain_points", []) or [])
    if response.extracted_pain_point:
        pain_points_list.append(response.extracted_pain_point)
    pain_points_list = dedupe_keeping_first(pain_points_list)

    # Return updated state
    return {
        "messages": out_messages,
        "property_type": response.extracted_property_type or state.get("property_type"),
        "budget_tier": response.extracted_budget_tier or state.get("budget_tier"),
        "timeline": response.extracted_timeline or state.get("timeline"),
        "pain_points": pain_points_list,
        # Sticky lead details — keep a value once captured; a later turn that doesn't restate
        # it must not wipe it (mirrors property_type/budget_tier above).
        "customer_name": response.extracted_customer_name or state.get("customer_name"),
        "city": response.extracted_city or state.get("city"),
        "preferred_contact_time": response.extracted_preferred_contact_time or state.get("preferred_contact_time"),
        # Sticky: once the model flags this a real lead, keep it flagged so the worker delivers it.
        "lead_ready_for_handoff": response.lead_ready_for_handoff or state.get("lead_ready_for_handoff", False),
        # The model ASKS for the brochure to go out; the worker attaches the actual file. Gated on
        # BROCHURE_URL, so a model that offers one when none is configured still sends nothing —
        # and the prompt in that mode tells it not to offer.
        "brochure_requested": bool(getattr(response, "send_brochure", False)) and bool(settings.brochure_url),
        # Server-built priced order + dedup flags (empty dict when no checkout this turn).
        **checkout_updates,
    }

async def node_high_intent(state: ConversationState):
    return await _execute_sales_node(state, HIGH_INTENT_PROMPT)

async def node_window_shopper(state: ConversationState):
    return await _execute_sales_node(state, WINDOW_SHOPPER_PROMPT)

async def node_problem_solver(state: ConversationState):
    return await _execute_sales_node(state, PROBLEM_SOLVER_PROMPT)

async def node_b2b_enterprise(state: ConversationState):
    return await _execute_sales_node(state, B2B_PROMPT)

async def node_post_sale_support(state: ConversationState):
    return await _execute_sales_node(state, SUPPORT_PROMPT)

async def node_out_of_domain(state: ConversationState):
    return await _execute_sales_node(state, OUT_OF_DOMAIN_PROMPT)

async def node_general_greeting(state: ConversationState):
    return await _execute_sales_node(state, GENERAL_GREETING_PROMPT)

async def node_contextual_rewarm(state: ConversationState):
    return await _execute_sales_node(state, REWARM_PROMPT)


async def node_human_probe(state: ConversationState):
    """
    Calm human request handler. Instead of forwarding to a human on the first ask, warmly probe for
    the reason (LLM-driven, contextual) and ALWAYS offer a "Connect me now" escape hatch so the
    customer is never trapped. Does NOT set requires_human_handoff — triage escalates on a repeat ask
    or when the hatch is tapped.
    """
    result = await _execute_sales_node(state, HUMAN_PROBE_PROMPT)
    messages = result.get("messages") or []
    ai_msg = messages[0] if messages else None

    # If the LLM path failed, _execute_sales_node's graceful fallback sets requires_human_handoff.
    # For a first calm ask we don't want to escalate — send a warm deterministic probe that still
    # carries the escape hatch, and drop the handoff flag.
    if ai_msg is None or result.get("requires_human_handoff"):
        fallback = AIMessage(
            content="Of course — I'm happy to bring in our team. Quick thing first: tell me a little "
                    "about what you'd like help with, and I'll either sort it right here or pass it "
                    "along with the details so you won't have to repeat yourself. 😊",
            response_metadata={
                "options": [dict(CONNECT_NOW_OPTION)],
                "internal_thought": "Human-probe LLM fallback: warm probe with guaranteed escape hatch, no escalation.",
            },
        )
        return {"messages": [fallback]}

    # Guarantee the escape hatch on the LLM's own message.
    meta = ai_msg.response_metadata if isinstance(ai_msg.response_metadata, dict) else {}
    options = meta.get("options") or []
    if not any((opt or {}).get("postback_id") == "CONNECT_NOW" for opt in options):
        meta["options"] = options + [dict(CONNECT_NOW_OPTION)]
        ai_msg.response_metadata = meta

    return result
