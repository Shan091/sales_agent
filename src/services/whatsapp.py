import asyncio
import logging
import time
import httpx
from typing import Dict, Any, List, Optional
from config.settings import settings
from src.core.text import fit_label
from src.storage.cache import CacheService

logger = logging.getLogger(__name__)

# WhatsApp's hard ceilings on interactive elements.
_BUTTON_TITLE_MAX = 20   # reply buttons
_LIST_TITLE_MAX = 24     # list row titles
_LIST_DESC_MAX = 72      # list row descriptions

# Outbound send retries. A blip on one socket is not a reason to re-run a whole conversation turn,
# which is what happened before: a single ConnectTimeout escaped the turn, TaskIQ re-ran it from the
# top (LLM call included), and the failed bubble was skipped as already-sent on the way back.
_SEND_ATTEMPTS = 3
_SEND_BACKOFF_SECONDS = 0.6
_TRANSIENT_SEND_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError,
                          httpx.ReadError, httpx.WriteError, httpx.PoolTimeout)
# Meta's own transient answers. Everything else in the 4xx range is a wrong payload or a bad token
# and will be exactly as wrong on the next attempt.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class WhatsAppService:
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
        self.api_url = f"https://graph.facebook.com/v19.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
        self.headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
            "Content-Type": "application/json"
        }

    async def send_typing_indicator(
        self,
        thread_id: str,
        inbound_message_id: Optional[str] = None,
        received_at: Optional[float] = None,
        repost: bool = False,
    ):
        """
        Show the "typing…" dots and mark the customer's message read.

        Both are the same Cloud API call, and it is keyed to the INBOUND message id — there is
        no "start typing for this conversation" endpoint. Without a wamid there is nothing to
        send, so we skip rather than fire a request that would 400.

        Meta displays the indicator for up to ~25 seconds from the POST and dismisses it as soon
        as our reply lands. There is no endpoint that extends that window, so a turn that runs
        longer shows dots and then silence: **that ceiling is Meta's and this code cannot fix it**
        — the only real remedy is a shorter turn. What it can do is not spend the window queued and
        not fall silent while work is still in flight, which is what `repost` is for (see
        `processing.py::typing_heartbeat`).

        `received_at` is a `time.monotonic()` reading from when the turn picked the message up;
        it makes the log line say how long the customer had already been waiting when the dots
        went out, so "the indicator sometimes doesn't show" can be answered with a number instead
        of an impression. `repost` only changes the wording, so the two are countable apart.

        Fail-soft by design: cosmetic feedback must never cost a turn, so every error is logged
        and swallowed.
        """
        if not inbound_message_id:
            logger.debug(f"[{thread_id}] No inbound message id; skipping typing indicator.")
            return

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": inbound_message_id,
            "typing_indicator": {"type": "text"},
        }
        try:
            started = time.monotonic()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.api_url, json=payload, headers=self.headers)
            now = time.monotonic()
            what = "Typing indicator re-posted" if repost else "Typing indicator + read receipt"
            timing = f"POST {(now - started) * 1000:.0f}ms"
            if received_at is not None:
                timing += f", {(now - received_at) * 1000:.0f}ms into the turn"
            if response.status_code != 200:
                logger.warning(
                    f"[{thread_id}] {what} rejected: {response.status_code} - {response.text} ({timing})"
                )
            else:
                # INFO, not DEBUG: in production logs a silent turn was indistinguishable from a
                # working one, which is most of why "sometimes" was all anyone could say about it.
                logger.info(f"[{thread_id}] {what} sent ({timing}).")
        except Exception as e:
            logger.warning(f"[{thread_id}] Typing indicator failed (non-fatal): {e}")

    async def send_document(
        self,
        thread_id: str,
        webhook_msg_id: str,
        url: str,
        filename: str,
        caption: str = "",
    ) -> bool:
        """
        Send a PDF (the brochure / lookbook) as a WhatsApp document.

        Exists because the prompts used to offer a lookbook the system had no way to deliver — the
        agent said "here's our digital lookbook" and sent nothing, which is worse than never
        offering it. Returns True only if Meta accepted the document, so the caller can tell the
        customer the truth either way.

        `link` must be publicly fetchable by Meta's servers; a localhost or tunnel-only URL will be
        rejected. Config-gated on settings.brochure_url, so with nothing configured the agent is
        told not to offer it at all rather than promising and failing.
        """
        is_new = await self.cache.check_and_set_idempotency(thread_id, webhook_msg_id, "document", 0)
        if not is_new:
            logger.info(f"[{thread_id}] Document already sent for this key. Skipping.")
            return True

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": thread_id,
            "type": "document",
            "document": {"link": url, "filename": filename},
        }
        if caption:
            payload["document"]["caption"] = caption

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, json=payload, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"[{thread_id}] Document send failed: {response.status_code} - {response.text}")
                return False
            logger.info(f"[{thread_id}] Document '{filename}' sent.")
            return True
        except Exception as e:
            logger.error(f"[{thread_id}] Document send failed: {e}")
            return False

    async def dispatch_message(
        self,
        thread_id: str,
        webhook_msg_id: str,
        node_name: str,
        msg_index: int,
        text: str,
        options: Optional[List[Dict[str, Any]]] = None,
        last_user_message_timestamp: float = 0.0
    ):
        """
        Sends a WhatsApp message with Idempotency and 24h SLA compliance.
        
        FIX: Now correctly routes to buttons (1-3 options) vs list messages (4-10 options).
        """
        # 1. Hyper-Specific Idempotency Check
        is_new = await self.cache.check_and_set_idempotency(
            thread_id, webhook_msg_id, node_name, msg_index
        )
        if not is_new:
            logger.info(f"[{thread_id}] Idempotency hash matched. Skipping duplicate dispatch.")
            return

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": thread_id,
        }

        # 2. The 24-Hour Rule Check
        current_time = time.time()
        hours_elapsed = (current_time - last_user_message_timestamp) / 3600.0

        if hours_elapsed > 24:
            logger.warning(f"[{thread_id}] 24h SLA breached. Enforcing Template Fallback.")
            payload["type"] = "template"
            payload["template"] = {
                "name": "nurture_checkin_01",
                "language": {"code": "en"}
            }
        else:
            # Buttons show ONLY a title — WhatsApp drops the description entirely. So the layout
            # is chosen by whether the descriptions carry meaning, not just by how many options
            # there are. Presenting "Base lock" / "Premium lock" as bare buttons is what left a
            # customer choosing between two products whose difference was never shown.
            has_descriptions = bool(options) and any((opt.get("description") or "").strip() for opt in options)

            if options and len(options) <= 3 and not has_descriptions:
                # Interactive BUTTON Message — best for actions (Confirm & pay / Explore more),
                # where the label alone says everything.
                payload["type"] = "interactive"
                buttons = [
                    {
                        "type": "reply",
                        "reply": {
                            "id": opt["postback_id"],
                            "title": fit_label(opt["label"], _BUTTON_TITLE_MAX)
                        }
                    }
                    for opt in options[:3]
                ]
                payload["interactive"] = {
                    "type": "button",
                    "body": {"text": text},
                    "action": {"buttons": buttons}
                }
            elif options:
                # Interactive LIST Message — up to 10 rows, and the only layout that renders the
                # description line under each title. Used whenever the options need explaining.
                rows = []
                for opt in options[:10]:  # WhatsApp limit: 10 rows
                    row = {
                        "id": opt["postback_id"],
                        "title": fit_label(opt["label"], _LIST_TITLE_MAX),
                    }
                    description = fit_label(opt.get("description") or "", _LIST_DESC_MAX)
                    if description:
                        row["description"] = description
                    rows.append(row)
                payload["type"] = "interactive"
                payload["interactive"] = {
                    "type": "list",
                    "body": {"text": text},
                    "action": {
                        "button": "Choose one",
                        "sections": [
                            {
                                "title": "Tap to continue",
                                "rows": rows
                            }
                        ]
                    }
                }
            else:
                # Standard Text (no options)
                payload["type"] = "text"
                payload["text"] = {"body": text}

        # 3. HTTP Request
        logger.info(f"[{thread_id}] Sending payload to Meta API... type={payload.get('type')}")
        try:
            await self._post_with_retries(thread_id, payload)
        except Exception:
            # The claim was taken before the send. Give it back, or a TaskIQ retry of this turn will
            # skip the bubble as "already sent" and the customer never receives it at all.
            await self.cache.release_idempotency(thread_id, webhook_msg_id, node_name, msg_index)
            raise

    async def _post_with_retries(self, thread_id: str, payload: Dict[str, Any]) -> None:
        """
        POST to Meta, retrying the failures that are worth retrying.

        Observed live: a single `httpx.ConnectTimeout` — a TCP connect that never completed —
        propagated out of a turn, was logged as a pipeline error, and handed the whole turn back to
        TaskIQ to re-run from the top, LLM call included. A blip on one socket is not a reason to
        re-do a conversation turn, so it is retried here where the only cost is the request.

        Retried: connect/read timeouts, connection resets, and Meta's own 429/5xx. NOT retried: 4xx,
        which means the payload or the token is wrong and will be just as wrong next time — those
        raise immediately so the log shows the real reason rather than three copies of it.

        The explicit timeout matters too. This client had none, so it used httpx's 5-second default
        for every phase; the rest of this file already passes 10-30s. `connect` is the phase that
        fails on a flaky link, so it gets the most room.
        """
        timeout = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=5.0)
        last_error: Optional[Exception] = None

        for attempt in range(1, _SEND_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.api_url, json=payload, headers=self.headers)
                if response.status_code == 200:
                    logger.info(f"[{thread_id}] Message successfully sent to Meta API: {response.json()}")
                    return
                logger.error(
                    f"[{thread_id}] Meta API Error: {response.status_code} - {response.text}"
                )
                if response.status_code not in _RETRYABLE_STATUS:
                    response.raise_for_status()
                last_error = httpx.HTTPStatusError(
                    f"Meta returned {response.status_code}", request=response.request, response=response
                )
            except _TRANSIENT_SEND_ERRORS as e:
                last_error = e
                logger.warning(
                    f"[{thread_id}] Send attempt {attempt}/{_SEND_ATTEMPTS} failed "
                    f"({type(e).__name__}: {e})."
                )

            if attempt < _SEND_ATTEMPTS:
                await asyncio.sleep(_SEND_BACKOFF_SECONDS * attempt)

        logger.error(f"[{thread_id}] Giving up after {_SEND_ATTEMPTS} send attempts.")
        raise last_error if last_error else RuntimeError("Meta send failed with no recorded error")
