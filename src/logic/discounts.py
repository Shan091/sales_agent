# file: src/logic/discounts.py
"""
Bounded dynamic pricing — the trust boundary between the LLM and the money.

The agent is allowed to *grow the sale* by choosing a discount, but it is NOT allowed to
invent one. The split:

  LLM  decides  -> which SKUs, what quantity, and WHICH PREDEFINED OFFER to apply (an id).
  CODE decides  -> the unit prices (read from products_pricing), whether that offer is
                   actually eligible, how much of it survives the policy clamp, and the
                   final rupee total.

So the widest possible failure — a jailbroken model "agreeing" to 90% off, or naming an
offer that doesn't exist — costs at most `MAX_DISCOUNT_PCT`, because that is the only
number code will ever honour. Every clamp writes a human-readable note into the returned
order so the audit row explains itself (see PaymentOrder.audit_notes).

Two further hard rules encoded here:
  1. The installation fee is NEVER discounted — it is real labour cost, not margin.
  2. An ineligible offer degrades to no discount; it never falls through to "apply anyway".
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


# ── The offer registry. Code-owned and closed: an offer id the LLM invents is not here,
#    so it cannot be applied. Percentages are ceilings in their own right and are ALSO
#    clamped to settings.MAX_DISCOUNT_PCT at apply time.
#
#    Labels deliberately carry NO percentage. The model sees this registry in its prompt, and
#    it is forbidden to state a figure — the surest way to hold that line is to never show it
#    one. Code prints the real percentage on the quote, where it is computed.
OFFERS: Dict[str, Dict[str, Any]] = {
    "NONE": {
        "pct": 0.0,
        "label": "No offer",
        "min_line_items": 0,
        "min_subtotal": 0.0,
        "reason": "List price.",
    },
    "FESTIVE5": {
        "pct": 5.0,
        "label": "Festive offer",
        "min_line_items": 1,
        "min_subtotal": 0.0,
        "reason": "Seasonal offer — applies to any order.",
    },
    "BUNDLE8": {
        "pct": 8.0,
        "label": "Bundle offer",
        "min_line_items": 2,
        "min_subtotal": 0.0,
        "reason": "Two or more different products in one order.",
    },
    "BUNDLE10": {
        "pct": 10.0,
        "label": "Full-setup offer",
        "min_line_items": 3,
        "min_subtotal": 0.0,
        "reason": "Three or more different products — a full-room or whole-home setup.",
    },
    "PROJECT12": {
        "pct": 12.0,
        "label": "Project offer",
        "min_line_items": 2,
        "min_subtotal": 100000.0,
        "reason": "Multi-product project order — the largest bracket.",
    },
}

# The ids the LLM is allowed to name, published into the prompt so the model sees a closed set.
OFFER_IDS = tuple(OFFERS.keys())


# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  UPGRADE PAIRS — a closed, hand-verified registry, for the same reason OFFERS is one
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# A step-up may only be proposed for a pair that appears here. The obvious alternative — let the
# model name any product it thinks is "the better version" and have code check it costs more — was
# written first and is unsound: dearer does not mean better-version-of-the-same-thing, and every
# wrong pair below passes a price test comfortably.
#
# REJECTED CANDIDATES, and why (this list is the useful half of this registry — do not add any of
# them back without a catalogue fact that overturns the reason):
#
#   Energy Meter Single Phase -> Energy Meter 3 Phase
#       NOT a tier. Which one is correct is decided by the building's incoming supply. A home on a
#       single-phase connection cannot use the 3-phase unit at all. Selling it as an upgrade is an
#       electrical mismatch dressed as advice, and it is the most consequential wrong pair here.
#   Indoor Smart Camera -> Smart Flood Light Camera
#       Different jobs, not two grades of one job. The indoor unit watches a room from a shelf; the
#       flood light camera is an outdoor fixture that lights an approach. Someone who asked to keep
#       an eye on the living room is not served by an outdoor light. It's a cross-sell if anything,
#       which the complement path already covers.
#   4 SW -> 6 SW -> 8 SW (and every other gang-count step)
#       Gang count is FITMENT, not quality. It is set by how many circuits are in that wall box; a
#       6-gang panel on a 4-circuit box is two dead buttons and may not fit. Sizing a part up is
#       not an upgrade.
#   6 SW -> 6 SW FAN / 6 SW - DIMMER / 6 SW - SOCKET
#       Variants that are only correct given a fact about the room — a ceiling fan on that circuit,
#       dimmable fittings, a socket needed in that box. The right one is a specification question
#       for the site survey, and the agent cannot see the room. Left to the human who can.
#   PIR Motion Sensor -> Microwave Sensor
#       Different sensing technology chosen per application, not a better sensor.
#   Smart Door Lock * -> Biometric Access Control
#       A different product class (multi-user access control), not a grade of a residential lock.
#
# WHAT SURVIVED: exactly two pairs, both the same product in a better grade, both safe to propose
# without knowing anything about the site. Two is not a disappointing number — it is what the
# catalogue honestly contains, and a registry that admitted more would be admitting the wrong ones.
UPGRADES: Dict[str, Dict[str, str]] = {
    "Smart Door Lock Base": {
        "to": "Smart Door Lock Premium",
        "gains": "remote access from the app and more ways to open it",
    },
    "Touch Screen Control Panel 7 inch": {
        "to": "Touch Screen Control Panel 10 inch",
        "gains": "a larger screen for whole-home control",
    },
}


def upgrade_target(sku: str) -> Optional[str]:
    """The catalogue name this sku may be stepped up to, or None when no verified pair exists."""
    return (UPGRADES.get((sku or "").strip()) or {}).get("to")


def family_name(sku: str) -> str:
    """
    The name to use for this product in a SENTENCE: the part its step-up pair has in common
    ("Smart Door Lock Base" -> "Smart Door Lock"), or the whole name for a product in no pair.

    Customer-facing prose must not lean on the grade word alone. "The Base" is a tier nobody has
    been quoted for, and someone who simply asked for a door lock does not think of themselves as
    having chosen "the Base" — so the pairing card says "most people fitting a Smart Door Lock",
    which is true of either model. The grade word survives only where the contrast IS the subject:
    the step-up card names its target in full, and the two 20-character buttons say Premium / Base
    because that is the word the tap turns on.

    A product outside UPGRADES has no shared prefix to find and keeps its whole name — a Video
    Door Phone must never become "Video Door".

    Electrician shorthand is glossed rather than repeated: `6 SW` in a sentence tells the customer
    nothing, and someone who doesn't understand a word usually goes quiet instead of asking, so this
    returns "6-switch glass panel". The raw sku still appears on the itemised order, where it sits
    beside its gloss and earns its place as the thing the team can look up.
    """
    name = (sku or "").strip()
    if not name:
        return ""
    partner = upgrade_target(name) or next(
        (left for left, spec in UPGRADES.items() if spec.get("to") == name), ""
    )
    if not partner:
        return speakable_name(name)
    a, b = name.split(), partner.split()
    i = 0
    while i < len(a) and i < len(b) and a[i].lower() == b[i].lower():
        i += 1
    return " ".join(a[:i]) or name


def speakable_name(sku: str) -> str:
    """
    The product as you would SAY it to a customer: the plain gloss when the sku is pure trade
    shorthand, otherwise the sku itself.

    One helper for every place a product is named in prose or on a button — the suggestion cards, the
    `Add ‹product›` label, the sentence a card is built around. "Add 6 SW" and "Most people fitting a
    6 SW want that too" both went out before this existed, and neither means anything to anyone who
    hasn't wired a switchboard.

    Narrow on purpose: only the `‹n› SW` family is replaced. A product with a readable NAME keeps it
    even though `plain_product_name` can gloss it — "Zigbee Hub" is what the customer will say back to
    the team and what they'd search for, so swapping it for "hub that lets everything talk to each
    other" costs more than the shorthand ever did. The gloss belongs beside such a name on the quote,
    not instead of it.
    """
    name = (sku or "").strip()
    if not name:
        return ""
    if not (_SW_RE.match(name) or _TWO_WAY_RE.match(name)):
        return name
    gloss = plain_product_name(name)
    if not gloss:
        return name
    # The gloss is written for the quote's parenthetical, where a leading article reads naturally. In
    # prose the card supplies its own article, so leaving it produces "The the …".
    lowered = gloss.lower()
    for article in ("the ", "a ", "an "):
        if lowered.startswith(article):
            return gloss[len(article):]
    return gloss


def upgrade_menu_for_prompt(names=None) -> str:
    """
    The step-up pairs injected into the sales prompt.

    Same discipline as offer_menu_for_prompt: the model chooses from a set it can see, and the set
    contains no figures — the price difference is computed and printed by code. Filtered to what
    the live catalogue can actually price, so a pair whose product has been deactivated is never
    offered. Returns "" when nothing is available, which leaves the model with nothing to propose.
    """
    priceable = {str(n).strip() for n in (names or [])}
    pairs = [
        (frm, spec) for frm, spec in UPGRADES.items()
        if not priceable or (frm in priceable and spec["to"] in priceable)
    ]
    if not pairs:
        return ""
    lines = "\n".join(f"- {frm}  ->  {spec['to']} ({spec['gains']})" for frm, spec in pairs)
    return (
        "\n\nSTEP-UP PAIRS — the ONLY upgrades you may propose.\n"
        f"{lines}\n"
        "If a product in the order appears on the LEFT and the customer's own reason would be "
        "better served by the right-hand one, set `suggested_upgrade` to the right-hand name and "
        "`upgrade_replaces` to the left-hand one, both copied character-for-character. A pair that "
        "is not on this list is not an upgrade — nothing else in the catalogue is a dearer model of "
        "anything else, whatever the prices might suggest. Sizing a switch panel up, swapping an "
        "indoor camera for an outdoor one, or changing an energy meter's phase are all the WRONG "
        "part rather than a better one, and the system rejects them."
    )


def offer_menu_for_prompt() -> str:
    """
    The offer list injected into the sales prompt, so the agent knows what it may select.

    Notice what is absent: percentages and rupee figures. The agent picks an id; the system
    turns that into money. Offers are listed weakest-to-strongest so relative value is clear
    without a single number being shown — a figure in this block is a figure the model could
    repeat to a customer before code has verified it.
    """
    ordered = sorted(
        ((oid, spec) for oid, spec in OFFERS.items() if oid != "NONE"),
        key=lambda kv: float(kv[1]["pct"]),
    )
    lines = [f"- {oid}: {spec['label']} — {spec['reason']}" for oid, spec in ordered]
    return (
        "OFFERS YOU MAY SELECT (by id, in the applied_offer field — never type a percentage "
        "or a rupee figure yourself). Listed smallest to largest:\n"
        + "\n".join(lines)
        + "\n- NONE: no discount.\n"
        "Eligibility and the hard discount ceiling are enforced by the system. If you pick an "
        "offer the order does not qualify for, the system quietly drops it to list price — so "
        "pick honestly. The quote the customer receives states the discount and the reason for "
        "it; you do not need to, and must not, restate either."
    )


def _round2(value: float) -> float:
    return round(value + 1e-9, 2)


def _field(obj: Any, name: str, default: Any) -> Any:
    """Read `name` off either a PriceResult-style object or a plain dict.

    Written as an explicit branch rather than `getattr(...) or obj.get(...)` because a
    legitimate 0.0 installation_fee is falsy — the `or` form would fall through to
    `.get()` and blow up on a Pydantic model.
    """
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def price_line_items(raw_items: List[Dict[str, Any]], trusted_prices: Dict[str, Any]) -> tuple:
    """
    Turn (sku, qty) proposals into fully priced line items using ONLY trusted prices.

    `trusted_prices` maps sku -> object/dict carrying base_price + installation_fee, i.e.
    the PriceResult values returned by PricingEngine.get_product_prices_batch. Anything the
    caller could not resolve must be absent or None.

    Returns (line_items, unresolved_skus, notes). Quantities are clamped to
    settings.MAX_LINE_QTY and duplicate SKUs are merged, so a "qty: 9999" or a repeated
    line can't inflate the order.
    """
    notes: List[str] = []
    unresolved: List[str] = []
    merged: Dict[str, int] = {}

    for item in raw_items or []:
        sku = (item.get("sku") or "").strip() if isinstance(item, dict) else ""
        if not sku:
            continue
        try:
            qty = int(item.get("qty") or 1)
        except (TypeError, ValueError):
            qty = 1
        if qty < 1:
            qty = 1
        if qty > settings.MAX_LINE_QTY:
            notes.append(f"Quantity for '{sku}' clamped {qty} -> {settings.MAX_LINE_QTY} (per-line cap).")
            qty = settings.MAX_LINE_QTY
        merged[sku] = merged.get(sku, 0) + qty

    line_items: List[Dict[str, Any]] = []
    for sku, qty in merged.items():
        priced = trusted_prices.get(sku)
        if not priced:
            unresolved.append(sku)
            continue
        base = float(_field(priced, "base_price", 0.0))
        install = float(_field(priced, "installation_fee", 0.0))
        # Re-clamp the merged quantity: two separate lines for the same SKU could each be
        # under the cap yet sum above it.
        if qty > settings.MAX_LINE_QTY:
            notes.append(f"Merged quantity for '{sku}' clamped {qty} -> {settings.MAX_LINE_QTY}.")
            qty = settings.MAX_LINE_QTY
        name = str(_field(priced, "product_name", sku))
        line_items.append({
            "sku": name,
            "qty": qty,
            "unit_price": _round2(base),
            "installation_fee": _round2(install),
            "line_total": _round2((base + install) * qty),
        })

    if unresolved:
        notes.append(f"Dropped unpriced SKU(s): {', '.join(sorted(unresolved))}.")

    return line_items, unresolved, notes


def apply_offer(
    line_items: List[Dict[str, Any]],
    offer_id: Optional[str],
    currency: str = "INR",
) -> Dict[str, Any]:
    """
    Build the final priced order from already-trusted line items plus an offer *choice*.

    Pure function — no I/O, no LLM, fully unit-testable. Every deviation from what was
    asked for is recorded in `audit_notes` rather than silently applied, because the
    explanation is the deliverable as much as the number is.
    """
    notes: List[str] = []
    requested = (offer_id or "NONE").strip().upper() or "NONE"

    subtotal = _round2(sum(float(li.get("line_total", 0.0)) for li in line_items))
    # Installation labour is never discounted — only the product value is discountable.
    install_total = _round2(sum(
        float(li.get("installation_fee", 0.0)) * int(li.get("qty", 1)) for li in line_items
    ))
    discountable = _round2(max(subtotal - install_total, 0.0))

    spec = OFFERS.get(requested)
    if spec is None:
        notes.append(f"Unknown offer '{requested}' proposed — not in the code-owned registry; dropped to list price.")
        requested, spec = "NONE", OFFERS["NONE"]

    pct = float(spec["pct"])

    # Eligibility, checked in code against the real order (not the model's claim).
    distinct_lines = len(line_items)
    if pct > 0 and distinct_lines < int(spec.get("min_line_items", 0)):
        notes.append(
            f"Offer {requested} needs {spec['min_line_items']}+ distinct products, order has "
            f"{distinct_lines} — offer not applied."
        )
        requested, pct = "NONE", 0.0
    elif pct > 0 and subtotal < float(spec.get("min_subtotal", 0.0)):
        notes.append(
            f"Offer {requested} needs a subtotal of at least ₹{spec['min_subtotal']:,.0f}, order is "
            f"₹{subtotal:,.0f} — offer not applied."
        )
        requested, pct = "NONE", 0.0

    # The hard ceiling. This is the line that makes a 90%-off jailbreak worthless.
    if pct > settings.MAX_DISCOUNT_PCT:
        notes.append(f"Discount clamped {pct:.1f}% -> {settings.MAX_DISCOUNT_PCT:.1f}% (policy ceiling).")
        pct = float(settings.MAX_DISCOUNT_PCT)
    if pct < 0:
        pct = 0.0

    discount_amount = _round2(discountable * pct / 100.0)
    if install_total > 0 and pct > 0:
        notes.append(f"Discount applied to product value only; ₹{install_total:,.2f} installation not discounted.")

    amount = _round2(subtotal - discount_amount)

    order = {
        "line_items": line_items,
        "product_summary": summarize(line_items),
        "applied_offer": requested if pct > 0 else None,
        "offer_label": OFFERS.get(requested, {}).get("label") if pct > 0 else None,
        "discount_pct": pct,
        "discount_amount": discount_amount,
        "subtotal": subtotal,
        "amount": amount,
        "currency": currency,
        "audit_notes": notes,
        "policy": {
            "max_discount_pct": float(settings.MAX_DISCOUNT_PCT),
            "order_min": float(settings.RAZORPAY_MIN_AMOUNT),
            "order_cap": float(settings.RAZORPAY_MAX_AMOUNT),
            "max_line_qty": int(settings.MAX_LINE_QTY),
        },
    }
    return order


def summarize(line_items: List[Dict[str, Any]]) -> str:
    """Short one-line description of the order, used as the Razorpay link description."""
    parts = [f"{int(li.get('qty', 1))} x {li.get('sku')}" for li in line_items]
    text = ", ".join(parts) or "Otohom order"
    return text[:255]


def _effective_pct(spec: Dict[str, Any]) -> float:
    """An offer's real value after the policy ceiling — the registry percentage is only a
    request. Ranking offers by their raw pct would otherwise claim a tier is better when the
    clamp makes the two identical."""
    return min(float(spec.get("pct", 0.0)), float(settings.MAX_DISCOUNT_PCT))


def _order_totals(line_items: List[Dict[str, Any]]) -> tuple:
    """(distinct_line_count, subtotal) — the two quantities every eligibility rule reads."""
    subtotal = _round2(sum(float(li.get("line_total", 0.0)) for li in line_items))
    return len(line_items or []), subtotal


def best_eligible_offer(line_items: List[Dict[str, Any]]) -> Optional[tuple]:
    """
    The most valuable offer this order ALREADY qualifies for, as (offer_id, spec), or None.

    Eligibility is evaluated in code against the real order, exactly as apply_offer does — so
    what this promises and what apply_offer grants cannot drift apart. Used to answer a
    question the customer would otherwise never get answered: "was there a discount available
    that I didn't get?"
    """
    distinct, subtotal = _order_totals(line_items)
    if not line_items:
        return None

    best: Optional[tuple] = None
    for oid, spec in OFFERS.items():
        if oid == "NONE" or _effective_pct(spec) <= 0:
            continue
        if distinct < int(spec.get("min_line_items", 0)):
            continue
        if subtotal < float(spec.get("min_subtotal", 0.0)):
            continue
        if best is None or _effective_pct(spec) > _effective_pct(best[1]):
            best = (oid, spec)
    return best


def next_offer_hint(line_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The nearest better tier this order does NOT yet qualify for, with the concrete gap.

    Returns e.g. {"offer_id": "BUNDLE10", "pct": 10.0, "needs_products": 1} or
    {"offer_id": "PROJECT12", "pct": 12.0, "needs_subtotal": 12500.0}, or None when the order
    is already on the best available tier.

    This is the honest version of an upsell: it states what is actually true about the offer
    ladder instead of inventing pressure. Deliberately returns STRUCTURED data — the caller
    renders the percentage, so no figure has to travel through the model to reach the customer.
    Only single-dimension gaps are reported; an order needing both more products and a higher
    subtotal is too far away to be a useful nudge.
    """
    if not line_items:
        return None
    distinct, subtotal = _order_totals(line_items)
    current = best_eligible_offer(line_items)
    current_pct = _effective_pct(current[1]) if current else 0.0

    candidates: List[Dict[str, Any]] = []
    for oid, spec in OFFERS.items():
        if oid == "NONE":
            continue
        pct = _effective_pct(spec)
        if pct <= current_pct:
            continue

        product_gap = max(int(spec.get("min_line_items", 0)) - distinct, 0)
        subtotal_gap = _round2(max(float(spec.get("min_subtotal", 0.0)) - subtotal, 0.0))
        if product_gap and subtotal_gap:
            continue  # two gaps at once: not a nudge, a different conversation
        if not product_gap and not subtotal_gap:
            continue  # already eligible; best_eligible_offer would have picked it up

        hint = {"offer_id": oid, "pct": pct, "label": spec.get("label", oid)}
        if product_gap:
            hint["needs_products"] = product_gap
            # Effort relative to the order in hand: one more product is a small step on a
            # 2-line order and a big one on a single-line order.
            hint["_effort"] = product_gap / max(distinct, 1)
        else:
            hint["needs_subtotal"] = subtotal_gap
            hint["_effort"] = subtotal_gap / max(subtotal, 1.0)
        # Absolute magnitudes of the two gap types aren't comparable, and a nudge that asks the
        # customer to more than double their order isn't a nudge — drop it rather than look greedy.
        if hint["_effort"] > 1.0:
            continue
        candidates.append(hint)

    if not candidates:
        return None
    # Least additional effort first; better value breaks a tie.
    candidates.sort(key=lambda h: (h["_effort"], -h["pct"]))
    chosen = dict(candidates[0])
    chosen.pop("_effort", None)
    return chosen


def available_offer_preview(line_items: List[Dict[str, Any]], currency_symbol: str = "₹") -> Optional[Dict[str, Any]]:
    """
    What the customer would save by applying the best offer they already qualify for.

    Answers the question a list-price quote otherwise leaves hanging: "is there a discount here,
    and how much is it?" Computed by re-pricing the real order, so the figure shown next to the
    button is the figure the button will actually produce — not an estimate.

    Returns None when nothing is eligible or the saving rounds to zero.
    """
    best = best_eligible_offer(line_items)
    if not best:
        return None
    offer_id, spec = best
    priced = apply_offer(line_items, offer_id)
    saving = float(priced.get("discount_amount", 0.0) or 0.0)
    if saving <= 0:
        return None
    return {
        "offer_id": offer_id,
        "label": spec.get("label", offer_id),
        "reason": spec.get("reason", ""),
        "pct": float(priced.get("discount_pct", 0.0) or 0.0),
        "saving": saving,
        "new_total": float(priced.get("amount", 0.0) or 0.0),
        # Code writes this button label, so it may carry a figure — the prohibition is on the
        # MODEL producing one, not on the customer seeing a verified number.
        "button_label": f"Apply {float(priced.get('discount_pct', 0.0)):.0f}% off",
    }


_SW_RE = re.compile(r"^(\d+)\s*SW\b(.*)$", re.IGNORECASE)
_TWO_WAY_RE = re.compile(r"^2\s*way\s*(\d+)\s*SW$", re.IGNORECASE)

# Catalogue names are written for electricians ("6 SW", "2 Way 2 SW"). On a quote the customer
# reads, they need plain words next to them. Display only — the sku itself is still printed, and
# is still what every amount was resolved against, so the trail stays intact.
_PLAIN_SUFFIXES = {
    "dimmer": "with dimming",
    "socket": "with a socket",
    "fan": "with fan speed control",
}
_PLAIN_NAMES = {
    "Hider Retrofit Module": "in-wall module (fits behind your existing switch plate)",
    "Zigbee Hub": "the hub that lets everything talk to each other",
    "IR Blaster": "controls your AC/TV remotes from the app",
    "Smart MCB Controller": "main-switch control from the app",
    "Grande Socket": "smart plug socket",
    # "PIR" is trade jargon and "Microwave Sensor" actively misleads — people think of the oven.
    # "Door Window Sensor" is how the catalogue spells it; nobody says it that way out loud.
    "PIR Motion Sensor": "detects movement in a room",
    "Microwave Sensor": "presence sensor for automatic lighting",
    "Door Window Sensor": "tells the app when a door or window opens",
}


def plain_product_name(sku: str) -> str:
    """
    A plain-English gloss for a catalogue sku, or "" when the name already reads normally.

    "6 SW" means nothing to someone who isn't an electrician, and a customer who doesn't
    understand a line on a quote usually goes quiet rather than asking. Anything already
    self-explanatory ("Curtain Motor", "Smart Door Lock Premium") gets no gloss — restating a
    clear name as a clear name is noise.
    """
    name = (sku or "").strip()
    if not name:
        return ""

    two_way = _TWO_WAY_RE.match(name)
    if two_way:
        n = int(two_way.group(1))
        return f"{n}-switch panel, two-way (works from two places, like both ends of a stair)"

    match = _SW_RE.match(name)
    if match:
        n = int(match.group(1))
        rest = (match.group(2) or "").strip(" -").lower()
        base = f"{n}-switch glass panel" if n > 1 else "1-switch glass panel"
        for key, phrase in _PLAIN_SUFFIXES.items():
            if key in rest:
                return f"{base} {phrase}"
        return base

    return _PLAIN_NAMES.get(name, "")


# ── Wording shared by both suggestion cards ─────────────────────────────────────────────────────
# Each card opens on what the customer GAINS rather than on a label, so the two have to read alike
# and there has to be one place to fix the voice. The agent supplies a `;`-separated list of
# benefits; everything below is how that list becomes a heading, bullets and — for the step-up — a
# deliberately small price tag.
_BENEFIT_LEAD_INS = (
    "so that you can ",
    "so that you'll ",
    "so you can ",
    "so you'll ",
    "so you ",
    "you can ",
    "you'll ",
)

# Longer than this and the heading renders as a plain first line instead of bold. It is never
# trimmed: a claim cut in half reads worse than an unbolded one.
_HEADING_MAX = 64
_MAX_BULLETS = 3


def _benefit_clause(text: str) -> str:
    """One benefit, without the brief's own scaffolding or its trailing punctuation."""
    clause = (text or "").strip().lstrip("•").strip()
    lowered = clause.lower()
    for lead in _BENEFIT_LEAD_INS:
        if lowered.startswith(lead):
            clause = clause[len(lead) :].strip()
            break
    return clause.rstrip(" .!,;")


def benefit_heading(reason: str) -> str:
    """
    The first benefit, as the line the customer reads before anything else.

    Shared by both cards on purpose: they arrive one after the other, and a difference in voice
    between them reads as two different senders. Public because sales.py::_validate_upgrade has to
    agree with it — a step-up whose reason yields no heading is dropped at validation, so the
    walkthrough never schedules a beat this function would render empty.
    """
    clause = _benefit_clause((reason or "").split(";")[0])
    return clause[0].upper() + clause[1:] if clause else ""


def _heading_line(text: str) -> str:
    """Bold when it fits a heading; otherwise the same words, unbolded and whole."""
    if not text:
        return ""
    return f"*{text}*" if len(text) <= _HEADING_MAX else text


def _benefit_bullets(reason: str, limit: int = _MAX_BULLETS) -> List[str]:
    """The benefits after the first, capped so the card stays glanceable. A fourth is dropped."""
    out: List[str] = []
    for part in (reason or "").split(";")[1:]:
        clause = _benefit_clause(part)
        if clause:
            out.append(f"• {clause}")
        if len(out) >= max(int(limit), 0):
            break
    return out


def _delta_tag(unit_delta: float, line_delta: float, qty: int) -> str:
    """
    The price difference as a small tag on the product name, never as a second price.

    `+₹10,000` in italics beside the product reads as a property of that product; the same figure on
    a line of its own reads as another amount being asked for. Valuation is relative (Ariely), so at
    qty > 1 the per-unit figure comes first — that is the figure the comparison turns on — with the
    line figure after it so the quote still reconciles.
    """
    cur = "₹"
    qty = max(int(qty or 1), 1)
    if qty == 2:
        return f"_(+{cur}{unit_delta:,.0f} each, {cur}{line_delta:,.0f} for both)_"
    if qty > 2:
        return f"_(+{cur}{unit_delta:,.0f} each, {cur}{line_delta:,.0f} for all {qty})_"
    return f"_(+{cur}{line_delta:,.0f})_"


def _article(name: str) -> str:
    """"a" or "an" for a product name; a wrong article is the kind of slip that reads as a bot."""
    return "an" if (name or "").strip()[:1].lower() in ("a", "e", "i", "o", "u") else "a"


def _anchor_family(order: Dict[str, Any]) -> str:
    """
    What this order is about, named the way a sentence should name it: the first line item through
    family_name, so the pairing card says "a Smart Door Lock" and not "a Smart Door Lock Base".
    """
    for li in order.get("line_items") or []:
        name = family_name(str(li.get("sku") or ""))
        if name:
            return name
    return ""


# The last line of a standalone step-up card, which is also the thing that makes a rendered card
# recognisable later. Kept as one pair of literals so the renderer below and the "have we already
# shown this?" check in sales.py::_step_up_already_shown cannot drift apart.
_STEP_UP_SWAP_LINES: Tuple[str, str] = (
    "_You'd get this one instead of the one you picked._",
    "_You'd get these instead of the ones you picked._",
)

# What only a RENDERED step-up card contains. sales.py::_step_up_already_shown requires one of
# these in prior agent text before it will treat a step-up as spent, because the question is not
# "has this product been mentioned?" but "has this customer been given the card — with its exact
# price difference and its Switch/Keep buttons — and turned it down?". A live transcript proved
# those are different events: discovery described both lock models in prose, the older any-mention
# check read that as a decided either/or, and the customer never saw the difference at all.
STEP_UP_CARD_MARKERS: Tuple[str, ...] = tuple(
    line.strip("_").rstrip(".") for line in _STEP_UP_SWAP_LINES
)


def _upgrade_lines(order: Dict[str, Any], standalone: bool = False) -> List[str]:
    """
    The step-up suggestion for this order, or [] when there is nothing validated to show.

    Unlike the complement block this claims no discount, so it needs no reachable offer tier — a
    swap keeps the line count and earns nothing new. What it claims instead is a price difference,
    computed in sales.py::_validate_upgrade from the catalogue on both sides and printed here. The
    agent supplies the benefits; code owns every figure.

    `standalone=True` is the walkthrough beat: this block IS the whole message, sent the moment the
    customer settles on a product and before any price has been shown, with its own buttons under
    it. So it cannot talk about something already quoted, and it does not need to explain that words
    are enough — the buttons carry that.

    Every clause of the wording is deliberate, because the obvious phrasing gets every one of these
    backwards:

      * IT OPENS ON THE BENEFIT. This reverses an earlier draft that opened on the label "Also
        available", and the reason that draft existed still holds: buyers "inherently mistrust
        vendors and fear being up-sold", so a heading must not announce an upsell. An availability
        label serves that badly — it tells the customer nothing they wanted, and the one thing they
        need in order to choose (what the dearer model does that theirs doesn't) was left to a
        single clause further down. Leading with the gain informs instead of announcing, which
        reaches the same goal properly. Still no "?": the agent is not asking for anything here,
        the buttons are the ask.
      * THE FEATURES, PLURAL, AS BULLETS. "What am I missing by staying put?" is answered item by
        item. One sentence about being without something is not an answer.
      * THE DIFFERENCE IS A TAG, never a new total — see _delta_tag.
      * THE CLOSING LINE FRAMES THE FIGURE AS A SWAP, which is also literally what the code does:
        sales.py::_swap_upgrade_in_order substitutes the line at the same qty and asserts the line
        count is unchanged. So the sentence describes a guarantee rather than selling one. It
        carries no hedge and needs none — the `Keep the ‹X›` button under it is what makes declining
        free, and a hedge on top of a button is the vendor reassuring themselves.
      * NO USABLE BENEFIT MEANS NO CARD. A price difference with nothing behind it is a bill with no
        reason attached, so the whole step-up is dropped rather than shown weak — the same rule the
        complement already followed. That is why there is no no-reason fallback line.
    """
    up = order.get("suggested_upgrade") or {}
    sku = str(up.get("sku") or "").strip()
    line_delta = float(up.get("line_delta", 0.0) or 0.0)
    reason = str(up.get("reason") or "").strip()
    heading = benefit_heading(reason)
    if not sku or line_delta <= 0 or not heading:
        return []

    display = speakable_name(str(up.get("display_name") or sku).strip())
    qty = max(int(up.get("qty", 1) or 1), 1)
    tag = _delta_tag(float(up.get("unit_delta", 0.0) or 0.0), line_delta, qty)
    replaced = speakable_name(str(up.get("replaces_display") or up.get("replaces_sku") or "").strip())

    # A blank line under the heading and above the closing line. On a phone the card is five or six
    # short lines with no punctuation to break them up, and run together they read as one block that
    # gets skimmed — the heading is the part that has to land on its own. Only in the STANDALONE card:
    # inside a quote this block is one item among several and the spacing would push the total off
    # the first screen.
    out = [_heading_line(heading)]
    if standalone:
        out.append("")
    bullets = _benefit_bullets(reason)
    if bullets:
        out.append(f"The {display} {tag} also gives you:")
        out.extend(bullets)
    else:
        out.append(f"That's the {display} {tag}.")

    if standalone:
        # Recasts the figure as the difference between two things rather than the price of an extra
        # one, and answers the only question a customer has at this point: one lock or two?
        out.append("")
        out.append(_STEP_UP_SWAP_LINES[1 if qty > 1 else 0])
    else:
        out.append(
            f"Your {replaced} stays as it is — just say the word if you'd rather have this one."
            if replaced
            else "Happy with what's here? Leave it as it is — just say the word if you'd rather switch."
        )
    return out



def _complement_lines(order: Dict[str, Any], hint: Optional[Dict[str, Any]], standalone: bool = False) -> List[str]:
    """
    The named add-on suggestion for this order, or [] when there is nothing honest to say.

    Requires BOTH a complement that code has already validated against the catalogue and a real
    reward for adding it. A suggestion with no reward attached is just a bigger bill, and this
    block claims a reward — so no reachable tier means no block.

    Every word here is written in code except the benefits, which the agent supplies and which the
    caller drops if they carry a figure or trade jargon.

    `standalone=True` is the walkthrough beat, where this block is the entire message and — for
    every product outside the two UPGRADES pairs — the FIRST thing the customer sees after choosing.
    Three parts, in this order:

      1. the strongest benefit, in the customer's own terms, as the opening line. Not a label:
         "Goes well with it" and "a common pairing" are categories, and a category is not a reason.
      2. "Most people fitting a ‹family› want that too." Social proof in the blueprint's SPIN form,
         written in CODE and claiming a common WANT rather than a purchase statistic — we hold no
         purchase data, and a figure we cannot stand behind does not belong in a public repo. It
         says "a Smart Door Lock", via _anchor_family, and never "a Smart Door Lock Base".
      3. the reward for adding it now, last, because it is the reason to do it today rather than the
         reason to do it at all.

    Nothing in it is product-specific — the family name, the benefits and the reward all come from
    the order — so it reads the same for a camera or a curtain motor as it does for a lock.

    Inside a quote the "?" heading stays: there the block is one item among several, the button
    directly beneath answers it on the same screen, and brevity wins.
    """
    comp = order.get("suggested_complement") or {}
    sku = str(comp.get("sku") or "").strip()
    if not sku or not hint or not hint.get("needs_products"):
        return []

    display = speakable_name(str(comp.get("display_name") or sku).strip())
    reason = str(comp.get("reason") or "").strip()
    heading = benefit_heading(reason)
    reward = f"takes this order to {float(hint['pct']):.0f}% off"

    if not standalone:
        second = f"{heading} — and it {reward}." if heading else f"Adding it {reward}."
        return [f"*Add the {display}?*", second]

    family = _anchor_family(order)
    social = f"Most people fitting {_article(family)} {family} want" if family else ""
    # Blank line under the heading and above the reward, for the reason given in _upgrade_lines: the
    # benefit has to land on its own or the whole card reads as one block and gets skimmed.
    out = [_heading_line(heading) if heading else f"*Worth adding: {display}*", ""]
    bullets = _benefit_bullets(reason)
    if heading and bullets:
        lead = f"The {display} also gives you:"
        out.append(f"{social} that too. {lead}".strip() if social else lead)
        out.extend(bullets)
    elif heading:
        out.append(f"{social} that too. That's the {display}.".strip() if social else f"That's the {display}.")
    elif social:
        out.append(f"{social} one of these too.")
    out.append("")
    out.append(f"Adding it {reward}.")
    return out


def upgrade_pitch(order: Dict[str, Any]) -> List[str]:
    """The step-up beat of the walkthrough, as its own message. [] when nothing is validated."""
    return _upgrade_lines(order, standalone=True)


def complement_pitch(order: Dict[str, Any]) -> List[str]:
    """The pairing beat of the walkthrough, as its own message. [] when nothing is validated."""
    return _complement_lines(order, next_offer_hint(order.get("line_items") or []), standalone=True)


def format_quote_message(order: Dict[str, Any], show_suggestions: bool = True) -> str:
    """
    The customer-facing itemised quote, written entirely in code so the figure on screen is
    provably the figure code computed.

    `show_suggestions=False` drops the step-up / pairing blocks from the body. It is passed when
    the consultative walkthrough has already shown them, each as its own message with its own
    button: repeating them here would be the third time of asking, and the quote is the one
    message that has to be read at a glance. The BUTTONS still appear — sales.py::_quote_options
    reads the same validated data — so nothing becomes unreachable, it just stops being re-pitched.
    """
    cur = "₹"
    lines = ["*Your Otohom order*", ""]
    for li in order.get("line_items", []):
        qty = int(li.get("qty", 1))
        unit = float(li.get("unit_price", 0.0))
        install = float(li.get("installation_fee", 0.0))
        total = float(li.get("line_total", 0.0))
        plain = plain_product_name(str(li.get("sku") or ""))
        # Two lines per item, never three. A quote is read at a glance, and each extra line pushes
        # the total — the thing they are actually looking for — further off screen.
        lines.append(f"{li.get('sku')}  ×{qty}   {cur}{total:,.0f}")
        # installation_fee is per unit and ADDITIONAL to unit_price (line_total = (unit+install)*qty),
        # so the per-unit figure shown here has to be the all-in one or the line stops reconciling.
        # "₹3,500 each, incl. ₹500 fitting" against a ₹4,000 line said the ₹500 was already inside
        # the ₹3,500 and left the ₹4,000 unexplainable — on the one message where every figure has
        # to be defensible. Printed all-in, `each × qty` equals the line total at any quantity.
        #
        # At qty 1 there is nothing to multiply, so "each" earns nothing and the all-in figure is
        # the line total repeated: one ₹19,500 order printed that figure three times (line, "each",
        # Total). Only the split survives there, which is the part the line total doesn't already
        # say. With no fee and no gloss there is nothing left to print at all.
        if qty > 1:
            detail = (
                f"{cur}{unit + install:,.0f} each ({cur}{unit:,.0f} + {cur}{install:,.0f} fitting)"
                if install
                else f"{cur}{unit:,.0f} each"
            )
        else:
            detail = f"({cur}{unit:,.0f} + {cur}{install:,.0f} fitting)" if install else ""
        if plain:
            detail = f"{plain} · {detail}" if detail else plain
        if detail:
            lines.append(f"_{detail}_")

    line_items = order.get("line_items", [])
    subtotal = float(order.get("subtotal", 0.0))
    discount = float(order.get("discount_amount", 0.0))
    hint = next_offer_hint(line_items)

    # ── The suggestions, above the total ────────────────────────────────────────────────────────
    # Below the total they are not read: the customer reaches the figure they were waiting for, sees
    # the pay button under it and taps. Sitting here they read as what they are — one more thing
    # they could change — and the total still closes the message immediately above the buttons.
    #
    # Only ever shown to a customer who asked for a price outright and so never saw the walkthrough;
    # after the beats, repeating them here is the menu the walkthrough exists to replace.
    nudge = (_upgrade_lines(order) or _complement_lines(order, hint)) if show_suggestions else []
    if nudge:
        lines.append("")
        lines.extend(nudge)

    lines.append("")

    if discount > 0:
        # A subtotal line only earns its place when something is deducted from it.
        lines.append(f"Subtotal   {cur}{subtotal:,.0f}")
        label = order.get("offer_label") or order.get("applied_offer")
        lines.append(f"{label} ({float(order.get('discount_pct', 0.0)):.0f}% off)   −{cur}{discount:,.0f}")
    lines.append(f"*Total*   {cur}{float(order.get('amount', 0.0)):,.0f}")

    # ── The offer ladder ────────────────────────────────────────────────────────────────────
    # "Is there a discount I'm not getting?" is answered here, in code, with real figures, and
    # kept to two lines because this sits directly above the pay button where length costs
    # conversions.
    preview = None if order.get("applied_offer") else available_offer_preview(line_items)
    if preview:
        lines.append("")
        lines.append(f"*You qualify for {preview['pct']:.0f}% off*")
        # The SAVING only — never the discounted total. Printing both put a better figure two lines
        # under the line labelled *Total*, so the message contradicted its own heading and the
        # customer had to work out which number they were actually being charged. The real total
        # arrives the instant they tap, from the same function that computed this saving.
        lines.append(
            f"Our {preview['label'].lower()} saves you {cur}{preview['saving']:,.0f}. "
            f"Tap *{preview['button_label']}* below."
        )

    # "How do I get a better one?" — asked generically, and only when there was no specific
    # product to name above. Making the customer pick a product themselves, from a catalogue they
    # cannot see, is the weakest form of this ask; it is the fallback, not the default. It still
    # ends on a button rather than on homework: "tell me what else you're weighing up" put the work
    # back on the customer at the one moment they were ready to act.
    if hint and not nudge and show_suggestions:
        lines.append("")
        lines.append("*Want a bigger discount?*")
        if hint.get("needs_products"):
            n = int(hint["needs_products"])
            noun = "product" if n == 1 else "products"
            lines.append(
                f"{'One' if n == 1 else n} more {noun} takes this to {hint['pct']:.0f}% off — "
                f"tap *Explore more* and I'll show you what fits your place."
            )
        elif hint.get("needs_subtotal"):
            threshold = float(hint["needs_subtotal"]) + subtotal
            lines.append(f"Orders over {cur}{threshold:,.0f} get {hint['pct']:.0f}% off.")

    # ── Changing the order ──────────────────────────────────────────────────────────────────────
    # Typing an edit already works — it is an ordinary LLM turn that rebuilds checkout_items and
    # re-prices through the same guardrail — and nothing told the customer so. Four messages of
    # tapping teaches a thumb to look for buttons, so the line names the GESTURE ("the message
    # box"); "just tell me" was true and useless because it doesn't say where. It sits under the
    # Total and above the pay prompt: below the CTA it competes with Confirm & pay, above the Total
    # it reads as an apology for a price they haven't seen. Every re-price — Apply, swap, add, a
    # typed edit — comes back through this function, which is what makes it unconditional.
    lines.append("")
    lines.append(
        "_Need anything different — a different quantity, another product, one taken off? "
        "Just type it in the message box._"
    )

    lines.append("")
    lines.append("_Test-mode order. Tap below for a secure payment link._")
    return "\n".join(lines)
