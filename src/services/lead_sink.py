# file: src/services/lead_sink.py
"""
Lead delivery — sends a captured lead, or an escalation, to the destinations the Otohom client asked
for: one spreadsheet, one inbox, and (later) a CRM.

Two independent, fail-soft sinks (a delivery failure must NEVER break the customer's turn
or trigger a task retry — that would re-send WhatsApp replies):

1. Webhook  — POST the row JSON to settings.LEADS_WEBHOOK_URL. This is the flexible sink; an
   automation behind it (Google Sheets Apps Script / Zapier / Make / n8n) fans it out to the Sheet,
   the WhatsApp Sales Leads Group, and the CRM. The WhatsApp *group* can ONLY be reached this way —
   the Meta Cloud API cannot post to groups.
2. Email    — optional direct SMTP send to settings.LEADS_EMAIL_TO, in one readable format for every
   kind (see `build_email`).

Everything lands in the same sheet and the same inbox: the `kind` column says whether a row is a lead
or somebody waiting for a person, and the team filters on it. That is the client's call — two tabs is
two places to forget to look. Each sink is skipped silently when unconfigured, so partial setups
(webhook-only, e.g.) work without error. See `crm_handoff.py` for the row itself.
"""
import asyncio
import logging
import smtplib
from email.message import EmailMessage
from typing import Any, Dict

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class LeadSink:
    @staticmethod
    async def deliver_lead(summary: Dict[str, Any]) -> Dict[str, bool]:
        """
        Fan the lead out to every configured sink. Returns a per-sink success map, e.g.
        {"webhook": True, "email": False}. Never raises — every failure is logged and swallowed.
        """
        results: Dict[str, bool] = {}
        results["webhook"] = await LeadSink._send_webhook(summary)
        results["email"] = await LeadSink._send_email(summary)
        if not any(results.values()):
            # Nothing configured (or everything failed): keep the full lead in the log so it is
            # never silently lost — an operator can still recover it.
            logger.warning(f"Lead not delivered to any sink; logging in full: {summary}")
        return results

    # ── Sink 1: webhook ──────────────────────────────────────────────
    @staticmethod
    async def _send_webhook(summary: Dict[str, Any]) -> bool:
        url = settings.LEADS_WEBHOOK_URL
        if not url:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.post(url, json=summary)
                resp.raise_for_status()
            logger.info(
                f"{summary.get('kind')} webhook delivered for {summary.get('mobile_number')}."
            )
            return True
        except Exception as e:
            logger.error(f"Lead webhook delivery failed: {e}")
            return False

    # ── Sink 2: email (optional, blocking SMTP offloaded to a thread) ─
    @staticmethod
    async def _send_email(summary: Dict[str, Any]) -> bool:
        if not settings.SMTP_HOST or not settings.LEADS_EMAIL_TO:
            return False
        try:
            await asyncio.to_thread(LeadSink._send_email_sync, summary)
            logger.info(
                f"{summary.get('kind')} email delivered to {settings.LEADS_EMAIL_TO} "
                f"for {summary.get('mobile_number')}."
            )
            return True
        except Exception as e:
            logger.error(f"Lead email delivery failed: {e}")
            return False

    @staticmethod
    def _handback_instruction(number: str) -> str:
        """
        How to give the thread back — the one thing an escalation alert must not leave out.

        The colleague works the customer on their OWN number, so nothing they say reaches this
        system: whatever they write in the note is exactly what the agent will know when it picks the
        conversation back up. Shows the WhatsApp form when a staff allowlist is configured, because a
        salesperson has a phone and not a terminal, and the CLI form when it is not — so the
        instruction is never wrong about what will actually work.
        """
        if settings.staff_numbers:
            return (
                "When you're done, text the Otohom agent's WhatsApp number from your own:\n"
                f"  #done {number} <what you did / what it must know>\n"
                f"  #back {number} <...>   same, and tells the customer the agent is back\n"
                f"  #status {number}       check the hold, change nothing"
            )
        return (
            "When you're done, release the thread and tell the agent what happened:\n"
            f'  python -m src.scripts.resolve_handoff {number} --note "<what you did>"'
        )

    @staticmethod
    def build_email(summary: Dict[str, Any]) -> tuple[str, str]:
        """
        `(subject, body)` — one readable format for every kind, written to be read on a phone.

        The old body was every key as `k: v` followed by the whole payload as indented JSON. Nobody
        reads JSON in an inbox, and the one line that mattered — what to do next — was at the bottom
        under two hundred characters of punctuation. So: a subject that says which kind and who, the
        facts a salesperson acts on in a fixed order, the code-built summary indented under it, and
        for anything waiting on a person, the exact command that hands it back.
        """
        kind = summary.get("kind") or "LEAD"
        number = summary.get("mobile_number", "")
        name = (summary.get("customer_name") or "").strip()
        who = f"{name} — {number}" if name else number
        reason = summary.get("escalation_reason")
        waiting = kind in ("ESCALATION", "SUPPORT")

        if waiting:
            tag = f"{kind.lower()} · {reason}" if reason else kind.lower()
            subject = f"[Otohom {tag}] {who}"
            opening = "Somebody is waiting for a person on WhatsApp."
        else:
            subject = f"[Otohom {kind.lower()}] {who}"
            opening = "The agent captured a lead on WhatsApp."

        rows = [
            ("Customer", who),
            ("City", summary.get("city")),
            ("Interested in", summary.get("products_interested")),
            ("How far it got", summary.get("stage")),
            ("Order", summary.get("order")),
            ("Concerns", "; ".join(summary.get("pain_points") or [])),
            ("Reason", reason),
            ("Language", summary.get("language")),
        ]
        width = max(len(label) for label, _ in rows)
        facts = "\n".join(
            f"{label.ljust(width)} : {value}" for label, value in rows if value
        )

        parts = [opening, "", facts]
        digest = (summary.get("summary") or "").strip()
        if digest:
            parts += ["", "Summary", "\n".join(f"  {line}" for line in digest.splitlines())]
        if waiting:
            parts += ["", LeadSink._handback_instruction(number)]
        return subject, "\n".join(parts) + "\n"

    @staticmethod
    def _send_email_sync(summary: Dict[str, Any]) -> None:
        """Blocking SMTP send. Runs in a worker thread via asyncio.to_thread."""
        subject, body = LeadSink.build_email(summary)
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.LEADS_EMAIL_FROM or settings.SMTP_USERNAME
        msg["To"] = settings.LEADS_EMAIL_TO
        msg.set_content(body)

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
            if settings.SMTP_USE_TLS:
                server.starttls()
            if settings.SMTP_USERNAME:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.send_message(msg)
