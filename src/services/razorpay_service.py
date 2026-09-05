# file: src/services/razorpay_service.py
"""
Razorpay Payment Links — the agent's money action.

Deliberately raw `httpx` + stdlib `hmac` instead of the Razorpay SDK, mirroring how the
Meta webhook is verified in src/api/webhooks.py. Two reasons: one less dependency in the
money path, and the signature check stays something a reviewer can read end to end.

Every method is config-gated and fail-soft. If keys are missing or Razorpay is down,
create_payment_link returns None and the caller tells the customer warmly that the link
is on its way from the team — a payment outage must never break the conversation or
trigger a TaskIQ retry (which would re-send WhatsApp messages).

TEST MODE: use rzp_test_* keys. Razorpay's hosted page then accepts the documented test
cards, and no real money moves.
"""
import base64
import hashlib
import hmac
import logging
from typing import Any, Dict, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Razorpay works in the smallest currency unit.
PAISE_PER_RUPEE = 100


class RazorpayService:

    @staticmethod
    def _auth_header() -> str:
        raw = f"{settings.RAZORPAY_KEY_ID}:{settings.RAZORPAY_KEY_SECRET}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    @staticmethod
    def to_paise(amount_rupees: float) -> int:
        """Rupees -> integer paise. Rounded, never truncated, so ₹15,119.99 bills correctly."""
        return int(round(float(amount_rupees) * PAISE_PER_RUPEE))

    @staticmethod
    async def create_payment_link(
        thread_id: str,
        order: Dict[str, Any],
        reference_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Mint a payment link for an ALREADY-VALIDATED order.

        The caller must have run Guardrails.validate_payment_request first; this method
        re-checks only the outer envelope (config present, amount inside the hard cap) as a
        belt-and-braces stop, then sends `order["amount"]` — it never recomputes a total,
        so there is exactly one place in the codebase that decides what a customer pays.

        Returns {"link_id", "short_url", "amount", "raw"} or None on any failure.
        """
        if not settings.razorpay_enabled:
            logger.warning(f"[{thread_id}] Razorpay not configured; skipping link creation.")
            return None

        amount = float(order.get("amount", 0.0) or 0.0)
        if not (settings.RAZORPAY_MIN_AMOUNT <= amount <= settings.RAZORPAY_MAX_AMOUNT):
            logger.error(
                f"[{thread_id}] Refusing to mint a link for {amount} — outside the "
                f"[{settings.RAZORPAY_MIN_AMOUNT}, {settings.RAZORPAY_MAX_AMOUNT}] policy window."
            )
            return None

        description = (order.get("product_summary") or "Otohom smart home order")[:255]
        payload = {
            "amount": RazorpayService.to_paise(amount),
            "currency": str(order.get("currency") or "INR").upper(),
            "accept_partial": False,
            "description": description,
            # WhatsApp numbers arrive from Meta without a '+'; Razorpay wants E.164.
            "customer": {"contact": f"+{thread_id.lstrip('+')}"},
            # We confirm in-chat off the webhook, so silence Razorpay's own notifications.
            "notify": {"sms": False, "email": False},
            "reminder_enable": False,
            # notes come back verbatim on the webhook — this is how a paid event is
            # attributed to a conversation even if the link id lookup ever fails.
            "notes": {
                "thread_id": str(thread_id),
                "reference_id": str(reference_id or ""),
                "applied_offer": str(order.get("applied_offer") or "NONE"),
                "discount_pct": str(order.get("discount_pct", 0.0)),
            },
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{settings.RAZORPAY_API_BASE}/payment_links",
                    json=payload,
                    headers={
                        "Authorization": RazorpayService._auth_header(),
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code >= 400:
                logger.error(f"[{thread_id}] Razorpay link creation failed: {resp.status_code} {resp.text}")
                return None
            data = resp.json()
        except Exception as e:
            logger.error(f"[{thread_id}] Razorpay link creation error: {e}")
            return None

        link_id = data.get("id")
        short_url = data.get("short_url")
        if not link_id or not short_url:
            logger.error(f"[{thread_id}] Razorpay response missing id/short_url: {data}")
            return None

        logger.info(f"[{thread_id}] Payment link {link_id} created for ₹{amount:,.2f}.")
        return {"link_id": link_id, "short_url": short_url, "amount": amount, "raw": data}

    # ── Webhook verification ──────────────────────────────────────────────
    @staticmethod
    def verify_webhook_signature(raw_body: bytes, signature: str) -> bool:
        """
        HMAC-SHA256 of the RAW request body, hex-digested, compared in constant time with
        the X-Razorpay-Signature header. Fail-closed: no configured secret or no header
        means reject, so an unconfigured deploy cannot be told an order was paid.

        The body must be the exact bytes received — re-serializing the parsed JSON would
        change whitespace/key order and break the digest.
        """
        secret = settings.RAZORPAY_WEBHOOK_SECRET
        if not secret:
            logger.error("Razorpay webhook secret not configured; rejecting inbound event.")
            return False
        if not signature:
            return False
        expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature.strip())

    @staticmethod
    def parse_event(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten the parts of a Razorpay webhook we act on, so the task code doesn't have to
        walk the nested envelope. Tolerant by design: an unknown or partial event yields
        empty fields rather than raising, and the task then no-ops.

        Handled events: payment_link.paid, payment_link.expired, payment.failed,
        payment.captured.
        """
        event = str(payload.get("event") or "")
        entities = payload.get("payload") or {}
        link = (entities.get("payment_link") or {}).get("entity") or {}
        payment = (entities.get("payment") or {}).get("entity") or {}
        notes = link.get("notes") or payment.get("notes") or {}

        return {
            "event": event,
            "link_id": link.get("id"),
            "payment_id": payment.get("id"),
            "thread_id": notes.get("thread_id"),
            "amount_paise": link.get("amount") or payment.get("amount"),
            "error_description": payment.get("error_description") or payment.get("error_reason"),
            "method": payment.get("method"),
        }
