import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.settings import settings

logger = logging.getLogger(__name__)


# Currency-anchored price detector: matches ₹ / Rs / INR / rupees / "/-" adjacent to a
# number (either order). Deliberately NOT matching bare numbers, so spec figures like
# "800W" or "100-240V" are never mistaken for prices.
_PRICE_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*\d[\d,]*(?:\.\d+)?"      # ₹5,000 / Rs. 5000 / INR 5000
    r"|\d[\d,]*(?:\.\d+)?\s*(?:INR|rupees|/-)\b",     # 5000 INR / 5,000 rupees / 5000/-
    re.IGNORECASE,
)

# Prompt-injection payloads neutralized before user text reaches the graph.
# Defense-in-depth only, and NOT a complete control: pattern matching cannot enumerate every
# phrasing. The real protection is that the model is never trusted with a figure (see
# validate_pricing_output below) — this only lowers the noise reaching it.
_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|messages?|context)",
    r"disregard\s+(?:the\s+|all\s+|any\s+)?(?:above|previous|prior|instructions?)",
    r"forget\s+(?:everything|all|your\s+instructions?|previous\s+instructions?)",
    r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
    r"(?:what\s+is|show\s+me|print)\s+your\s+system\s+prompt",
    r"system\s+prompt",
    r"you\s+are\s+(?:now\s+)?an?\s+(?:unrestricted|jailbroken|dan\b|developer[-\s]mode)",
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:dan\b|an?\s+unrestricted|a\s+jailbroken)",
    r"developer\s+mode\s+(?:enabled|on)",
)]


def _price_digits(text: str) -> set:
    """Normalized numeric values of every currency-anchored price mention in `text`."""
    out = set()
    for match in _PRICE_RE.findall(text or ""):
        digits = re.sub(r"[^\d]", "", match)
        if digits:
            out.add(digits.lstrip("0") or "0")
    return out


class Guardrails:
    """
    Deterministic Input/Output Guardrails.
    Ensures the LLM doesn't hallucinate pricing or break safety constraints.
    """

    @staticmethod
    def validate_pricing_output(llm_text: str, verified_prices: Iterable[str] = ()) -> bool:
        """
        Fail-closed pricing check. Returns True only if EVERY currency amount quoted in
        `llm_text` also appears in `verified_prices` (values sourced from the DB /
        PricingEngine this turn). A response with no price mention is always valid.

        `verified_prices` may be raw strings (e.g. RAG / pricing-fallback context lines);
        their numeric values are extracted and compared.
        """
        quoted = _price_digits(llm_text)
        if not quoted:
            return True
        allowed = set()
        for price in verified_prices:
            allowed |= _price_digits(str(price))
        hallucinated = quoted - allowed
        if hallucinated:
            logger.warning(
                f"Pricing guardrail: unverified price(s) {sorted(hallucinated)} "
                f"not in verified set {sorted(allowed) or '[]'}."
            )
            return False
        return True

    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """
        Neutralizes obvious prompt-injection payloads before they reach the graph,
        preserving the rest of the user's text. Defense-in-depth, not a full control.
        """
        sanitized = user_input
        for pattern in _INJECTION_PATTERNS:
            sanitized = pattern.sub("***", sanitized)
        return sanitized

    # ═══════════════════════════════════════════════
    #  The money gate
    # ═══════════════════════════════════════════════

    @staticmethod
    def validate_payment_request(
        order: Optional[Dict[str, Any]],
        trusted_prices: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Final fail-closed check, run immediately before a Razorpay link is minted.

        Re-derives the whole order from its own line items and refuses it on ANY
        mismatch. This is deliberately redundant with discounts.apply_offer: that function
        computes the amount, this one independently re-checks it, so a future bug in the
        pricing path cannot quietly turn into a wrong charge. Pure and sync — no DB, no
        LLM — so the money rules are unit-testable without infrastructure.

        `trusted_prices` (optional) maps sku -> PriceResult/dict freshly read from
        products_pricing at mint time. When supplied, every unit price in the order must
        match it to the paisa; that is what stops a tampered or stale checkpoint value
        from being charged.

        Returns (ok, reasons). `reasons` is non-empty on rejection and is written into the
        PaymentOrder audit trail.
        """
        reasons: List[str] = []

        if not isinstance(order, dict) or not order:
            return False, ["No pending order to charge."]

        line_items = order.get("line_items") or []
        if not line_items:
            return False, ["Order has no line items."]

        currency = str(order.get("currency") or "INR").upper()
        if currency != "INR":
            reasons.append(f"Unsupported currency '{currency}'.")

        recomputed_subtotal = 0.0
        discountable = 0.0

        for li in line_items:
            sku = str(li.get("sku", "")).strip()
            if not sku:
                reasons.append("Line item with an empty SKU.")
                continue

            try:
                qty = int(li.get("qty", 0))
                unit = float(li.get("unit_price", 0.0))
                install = float(li.get("installation_fee", 0.0))
                line_total = float(li.get("line_total", 0.0))
            except (TypeError, ValueError):
                reasons.append(f"Non-numeric amounts on line '{sku}'.")
                continue

            if qty < 1 or qty > settings.MAX_LINE_QTY:
                reasons.append(f"Quantity {qty} for '{sku}' is outside 1..{settings.MAX_LINE_QTY}.")
            if unit <= 0:
                reasons.append(f"Non-positive unit price for '{sku}'.")
            if install < 0:
                reasons.append(f"Negative installation fee for '{sku}'.")

            expected_line = round((unit + install) * max(qty, 0) + 1e-9, 2)
            if abs(expected_line - line_total) > 0.01:
                reasons.append(
                    f"Line total for '{sku}' is {line_total}, expected {expected_line} "
                    f"from {qty} x ({unit} + {install})."
                )

            if trusted_prices is not None:
                priced = trusted_prices.get(sku)
                if not priced:
                    reasons.append(f"'{sku}' is not in the trusted price catalogue.")
                else:
                    t_unit = priced.get("base_price") if isinstance(priced, dict) else getattr(priced, "base_price", None)
                    t_install = priced.get("installation_fee") if isinstance(priced, dict) else getattr(priced, "installation_fee", None)
                    if t_unit is None or abs(float(t_unit) - unit) > 0.01:
                        reasons.append(f"Unit price for '{sku}' ({unit}) does not match the catalogue ({t_unit}).")
                    if t_install is not None and abs(float(t_install) - install) > 0.01:
                        reasons.append(f"Installation fee for '{sku}' ({install}) does not match the catalogue ({t_install}).")

            recomputed_subtotal += line_total
            discountable += unit * max(qty, 0)

        recomputed_subtotal = round(recomputed_subtotal + 1e-9, 2)
        discountable = round(discountable + 1e-9, 2)

        stated_subtotal = float(order.get("subtotal", 0.0) or 0.0)
        if abs(stated_subtotal - recomputed_subtotal) > 0.01:
            reasons.append(f"Subtotal {stated_subtotal} does not match the sum of line totals ({recomputed_subtotal}).")

        pct = float(order.get("discount_pct", 0.0) or 0.0)
        discount = float(order.get("discount_amount", 0.0) or 0.0)
        if pct < 0 or discount < 0:
            reasons.append("Negative discount.")
        if pct > settings.MAX_DISCOUNT_PCT + 1e-9:
            reasons.append(f"Discount {pct}% exceeds the {settings.MAX_DISCOUNT_PCT}% policy ceiling.")
        expected_discount = round(discountable * pct / 100.0 + 1e-9, 2)
        if abs(expected_discount - discount) > 0.01:
            reasons.append(
                f"Discount amount {discount} does not match {pct}% of the discountable "
                f"product value ({discountable} -> {expected_discount})."
            )
        if discount > discountable + 0.01:
            reasons.append("Discount exceeds the discountable product value (installation is never discounted).")

        amount = float(order.get("amount", 0.0) or 0.0)
        expected_amount = round(recomputed_subtotal - discount + 1e-9, 2)
        if abs(expected_amount - amount) > 0.01:
            reasons.append(f"Grand total {amount} does not equal subtotal − discount ({expected_amount}).")
        if amount < settings.RAZORPAY_MIN_AMOUNT:
            reasons.append(f"Total {amount} is below the minimum chargeable amount ({settings.RAZORPAY_MIN_AMOUNT}).")
        if amount > settings.RAZORPAY_MAX_AMOUNT:
            reasons.append(f"Total {amount} exceeds the per-order cap ({settings.RAZORPAY_MAX_AMOUNT}).")

        if reasons:
            logger.error(f"Payment guardrail REJECTED order: {reasons}")
            return False, reasons
        return True, []
