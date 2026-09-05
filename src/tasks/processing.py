import asyncio
import logging
import time
from typing import Optional
from sqlmodel import select
from taskiq.exceptions import TaskRejectedError
from src.services.whatsapp import WhatsAppService
from src.services.transcoder import TranscoderService
from src.services.crm_handoff import CRMHandoffService, KIND_ESCALATION, KIND_LEAD
from src.services.handoff_control import (
    STAFF_COMMAND_HELP,
    handoff_status,
    looks_like_staff_command,
    parse_staff_command,
    release_handoff,
    resolve_thread,
)
from src.services.razorpay_service import RazorpayService
from src.storage.cache import CacheService
from src.storage.models import PaymentOrder
from src.core.guardrails import Guardrails
from src.graph.workflow import compile_workflow_async
from src.core.database import async_session_maker
from src.logic.pricing import PricingEngine
from src.logic import discounts
from src.memory.semantic import get_semantic_memory
from src.rag.ingestion import get_embedding_client
from src.core.tracing import langfuse_config, new_request_id
import redis.asyncio as aioredis
from config.settings import settings
from src.tasks.broker import broker

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
#  C5 FIX: Lazy initialization — no module-level connections.
#  These are set during TaskIQ WORKER_STARTUP event in broker.py.
# ═══════════════════════════════════════════════
_cache: Optional[CacheService] = None
_whatsapp: Optional[WhatsAppService] = None
_graph_app = None


async def initialize_worker_services(redis_pool: aioredis.Redis):
    """Called by broker.py WORKER_STARTUP event to inject live connections."""
    global _cache, _whatsapp, _graph_app
    _cache = CacheService(redis_pool)
    _whatsapp = WhatsAppService(_cache)
    _graph_app = await compile_workflow_async()
    logger.info("Worker services initialized (Redis, WhatsApp, Graph).")
    await _warm_heavy_clients()


async def _warm_heavy_clients() -> None:
    """
    Load the embedding model and build the mem0 client at BOOT, not on a customer's turn.

    Measured before this existed: a first message took ~36 seconds, ~22 of them spent loading
    BAAI/bge-m3 twice in the same process — once when mem0 built its embedder, once when the RAG
    client built its own. Both were lazy, so the cost landed on whoever happened to message first,
    and it landed again in every additional worker process.

    Nobody is waiting during startup, so that is where it belongs. Both are best-effort: a warm-up
    failure must not stop the worker accepting messages, it just means the old lazy path applies.
    """
    try:
        embed_client = get_embedding_client()
        # _load_model is CPU-blocking; keep it off the event loop.
        await asyncio.to_thread(embed_client._load_model)
        # Then actually encode something. Loading the weights is not the whole cost: the first real
        # forward pass builds the compute graph and costs ~2s more, which would otherwise land on
        # the first customer. Steady-state encoding after this is 87-161ms for 1-3 queries.
        await embed_client.aembed_documents(["warm up"])
        logger.info("Embedding model warm.")
    except Exception as e:
        logger.warning(f"Embedding warm-up failed (will load lazily instead): {e}")

    if settings.MEM0_ENABLED:
        try:
            await asyncio.to_thread(get_semantic_memory)
            logger.info("Semantic memory client warm.")
        except Exception as e:
            logger.warning(f"Semantic memory warm-up failed (will build lazily instead): {e}")


async def close_worker_services():
    """
    Called by broker WORKER_SHUTDOWN to close the worker's Redis pool and prevent
    connection leaks across worker restarts (mirrors the DB engine dispose).
    """
    global _cache
    if _cache is not None:
        closer = getattr(_cache.redis, "aclose", None) or getattr(_cache.redis, "close", None)
        if closer is not None:
            try:
                await closer()
            except Exception as e:
                logger.warning(f"Error closing worker Redis pool: {e}")
        _cache = None


# How long a turn may sit in the wa_mutex queue before its typing dots are re-posted. Meta holds
# the indicator ~25s from the POST, so a turn that waited most of that out for the previous turn to
# finish would start its own work in silence. 15s leaves the re-post inside the original window
# rather than after it, which is the case we can reason about.
_TYPING_REPOST_AFTER_SECONDS = 15.0

# How long to wait before posting the dots at all. Six turns in this app answer with no model call
# and hand their first message to Meta 44–83ms in (measured), where the dots and the reply then race
# each other at Meta — and when the reply wins, "typing…" is drawn AFTER the answer and sits there
# with nothing coming. Anything the customer waits on is still slower than this by an order of
# magnitude (a triage call alone is ~1.3s), so the grace costs a real turn a quarter-second and buys
# the instant ones silence. Deliberately not keyed off the route: this function runs before the
# graph, the fast paths are only known inside triage, and an event means a seventh one added later
# gets this for free.
_TYPING_GRACE_SECONDS = 0.25

# Cadence and ceiling for keeping the dots alive on a turn that outruns Meta's ~25s window. There is
# no endpoint that extends it, so re-posting just under the ceiling is the only lever there is.
# HONEST CAVEAT: a re-post while the window is still open is known not to extend it, and whether one
# after expiry opens a fresh window is untested against Meta — the cost of being wrong is one
# swallowed request per 20s of an already-pathological turn, and the log answers it on the next run.
_TYPING_REPOST_EVERY_SECONDS = 20.0
_TYPING_REPOST_MAX = 3


async def _wait_for(seconds: float, *events: Optional[asyncio.Event]) -> bool:
    """
    Sleep `seconds` and report whether none of `events` fired — True when the wait ran out untouched.

    Polled rather than awaited on the events themselves so one helper serves both the grace window
    and the re-post cadence, and so a caller that passes no events simply sleeps.
    """
    deadline = time.monotonic() + seconds
    live = [e for e in events if e is not None]
    step = 0.05 if seconds <= 1.0 else 0.5
    while True:
        if any(e.is_set() for e in live):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        await asyncio.sleep(min(step, remaining))


async def typing_heartbeat(
    thread_id: str,
    inbound_message_id: Optional[str],
    stop_event: asyncio.Event,
    acquired_event: Optional[asyncio.Event] = None,
    received_at: Optional[float] = None,
    reply_started: Optional[asyncio.Event] = None,
):
    """
    Show the typing dots for as long as the customer is actually waiting — and not at all when they
    aren't.

    Three cases, in the order they happen:

    1. **An instant answer needs no dots.** After `_TYPING_GRACE_SECONDS`, if the turn has already
       begun sending (`reply_started`) or has finished (`stop_event`), nothing is posted. Otherwise
       the dots go out. Before this, a deterministic tap turn posted dots that raced its own reply to
       Meta and could be drawn after the answer, then hold for ~25s with nothing coming.
    2. **A queued turn must not spend its window waiting.** The dots fire before `acquire_lock`, which
       polls for up to 30s while a previous turn finishes, so a customer who sends two messages in a
       row could have the second turn's window expire before its work even began. If that wait ran
       past `_TYPING_REPOST_AFTER_SECONDS`, the same wamid is posted again the moment real work starts.
    3. **A long turn must not fall silent.** Meta holds the indicator ~25s from the POST, clears it
       when our reply lands, and offers no way to extend it — so the dots are re-posted every
       `_TYPING_REPOST_EVERY_SECONDS` (capped at `_TYPING_REPOST_MAX`) until the turn ends. That is
       the only lever available for a turn slower than the window; a shorter turn is still the real
       fix, and the caveat on `_TYPING_REPOST_EVERY_SECONDS` says what is and isn't known here.

    `received_at` is a `time.monotonic()` reading from when the turn picked the message up, so the log
    line says how long the customer had already been waiting when the dots went out — "it sometimes
    doesn't show" is then a number rather than an impression.

    Resilient: a WhatsApp API failure here must never become an unretrieved task exception that
    surfaces during cleanup and skips the lock release.
    """
    started = time.monotonic()

    if not await _wait_for(_TYPING_GRACE_SECONDS, stop_event, reply_started):
        logger.debug(f"[{thread_id}] Answered inside the grace window; no typing dots needed.")
        return

    try:
        await _whatsapp.send_typing_indicator(thread_id, inbound_message_id, received_at=received_at)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"[{thread_id}] Typing heartbeat failed (non-fatal): {e}")

    if not inbound_message_id:
        return

    try:
        if acquired_event is not None:
            # Poll rather than await the event outright, so `stop_event` — set on every exit path the
            # caller takes, including the one where the lock is never acquired — ends the wait even if
            # the cancellation that follows it is slow to arrive.
            while not acquired_event.is_set():
                if stop_event.is_set():
                    return
                await asyncio.sleep(0.5)
            waited = time.monotonic() - started
            if waited >= _TYPING_REPOST_AFTER_SECONDS:
                logger.info(
                    f"[{thread_id}] Waited {waited:.1f}s for the mutex; re-posting the typing dots "
                    "before the turn's own work starts."
                )
                await _whatsapp.send_typing_indicator(
                    thread_id, inbound_message_id, received_at=received_at, repost=True
                )

        for _ in range(_TYPING_REPOST_MAX):
            if await _wait_for(_TYPING_REPOST_EVERY_SECONDS, stop_event):
                await _whatsapp.send_typing_indicator(
                    thread_id, inbound_message_id, received_at=received_at, repost=True
                )
            else:
                return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"[{thread_id}] Typing re-post failed (non-fatal): {e}")


_LOW_SIGNAL_INPUTS = {
    "hi", "hii", "hello", "hey", "yo", "hai", "namaste", "start",
    "ok", "okay", "k", "yes", "yeah", "yep", "no", "nope", "sure",
    "thanks", "thank you", "ty", "👍", "🙏", "😊",
}


def _is_worth_recalling(text_input: str, msg_type: Optional[str]) -> bool:
    """
    Whether a semantic-memory search on this input can plausibly return anything useful.

    A button tap carries a postback id, and a bare "hi" or "ok" carries no topic — embedding
    either produces a query with nothing to match, so the search is pure added latency on the
    turn the customer is waiting for. Real questions and product talk still get full recall.
    """
    if msg_type == "interactive":
        return False
    cleaned = (text_input or "").strip().strip("!.?").lower()
    if not cleaned or cleaned in _LOW_SIGNAL_INPUTS:
        return False
    return len(cleaned) > 3


async def lock_keepalive(thread_id: str, stop_event: asyncio.Event):
    """Dynamically extends the Mutex lock TTL while the worker is processing."""
    lock_key = f"wa_mutex:{thread_id}"
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            if not stop_event.is_set() and _cache:
                await _cache.redis.expire(lock_key, 30)


# ═══════════════════════════════════════════════
#  Agentic checkout — the money action, run OUTSIDE the graph
# ═══════════════════════════════════════════════

async def _write_payment_order(thread_id: str, order: dict, status: str, link: Optional[dict] = None, audit_extra=None) -> Optional[int]:
    """Persist one PaymentOrder audit row. Isolated + fail-soft: an audit-write failure must
    never break the money flow (the link may already be live)."""
    try:
        async with async_session_maker() as session:
            row = PaymentOrder(
                thread_id=thread_id,
                line_items=list(order.get("line_items", [])),
                product_summary=order.get("product_summary", ""),
                applied_offer=order.get("applied_offer"),
                discount_pct=float(order.get("discount_pct", 0.0) or 0.0),
                discount_amount=float(order.get("discount_amount", 0.0) or 0.0),
                subtotal=float(order.get("subtotal", 0.0) or 0.0),
                amount=float(order.get("amount", 0.0) or 0.0),
                currency=order.get("currency", "INR"),
                status=status,
                razorpay_link_id=(link or {}).get("link_id"),
                payment_link_url=(link or {}).get("short_url"),
                audit_notes=list(order.get("audit_notes", [])) + list(audit_extra or []),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            logger.info(f"[{thread_id}] PaymentOrder #{row.id} recorded (status={status}).")
            return row.id
    except Exception as e:
        logger.error(f"[{thread_id}] Failed to write PaymentOrder ({status}): {e}")
        return None


async def _process_checkout(thread_id: str, config: dict, order: dict, webhook_msg_id: str, last_timestamp: float):
    """
    Mint a Razorpay link for a confirmed order and send it in chat.

    Called ONLY after an explicit "Confirm & pay" tap, on a server-built pending_order, once
    (payment_link_sent dedup). Re-validates against FRESH catalogue prices at mint time — the
    amount is never trusted straight from the checkpoint. Every branch records a PaymentOrder
    row so the audit trail is complete whether the mint succeeds or fails.

    payment_link_sent is set True ONLY on a successful mint, so a transient Razorpay/DB failure
    leaves the "Confirm & pay" button re-tappable for a clean retry rather than dead-ending.
    """
    # Re-price from the catalogue NOW. If the lookup fails we fall back to trusted=None, which
    # makes the guardrail re-check internal consistency only (the amount was already matched to
    # the catalogue when the order was built this session).
    skus = [li.get("sku") for li in order.get("line_items", []) if li.get("sku")]
    trusted = None
    try:
        async with async_session_maker() as session:
            price_map = await PricingEngine(session).get_product_prices_batch(skus)
        trusted = {name: pr for name, pr in price_map.items() if pr is not None} or None
    except Exception as e:
        logger.error(f"[{thread_id}] Checkout re-pricing lookup failed (proceeding on built order): {e}")

    ok, reasons = Guardrails.validate_payment_request(order, trusted)
    if not ok:
        await _write_payment_order(thread_id, order, status="failed", audit_extra=[f"Mint-time guardrail: {r}" for r in reasons])
        await _whatsapp.dispatch_message(
            thread_id=thread_id, webhook_msg_id=f"checkout_fail:{webhook_msg_id}", node_name="checkout", msg_index=0,
            text="Give me a moment — I need to double-check a couple of details on that order before I send a secure link.",
            options=None, last_user_message_timestamp=last_timestamp,
        )
        return

    link = await RazorpayService.create_payment_link(thread_id, order, reference_id=str(thread_id))
    if not link:
        await _write_payment_order(thread_id, order, status="failed", audit_extra=["Link creation returned None (Razorpay disabled or API error)."])
        await _whatsapp.dispatch_message(
            thread_id=thread_id, webhook_msg_id=f"checkout_fail:{webhook_msg_id}", node_name="checkout", msg_index=0,
            text="I couldn't generate the secure payment link just now. Let me get this sorted and send it across shortly.",
            options=None, last_user_message_timestamp=last_timestamp,
        )
        return

    await _write_payment_order(thread_id, order, status="link_created", link=link)
    amount = float(order.get("amount", 0.0) or 0.0)
    pay_text = (
        f"All set! Your total is ₹{amount:,.0f}.\n\n"
        f"Tap to pay securely:\n{link['short_url']}\n\n"
        "_Test-mode payment — you can use a Razorpay test card._"
    )
    await _whatsapp.dispatch_message(
        thread_id=thread_id, webhook_msg_id=link["link_id"], node_name="checkout", msg_index=0,
        text=pay_text, options=None, last_user_message_timestamp=last_timestamp,
    )
    # Dedup: only NOW is the money action complete, so only now do we mark it sent.
    # The link url is mirrored into graph state as well as the audit row, so a retry nudge can
    # always re-offer the live link even if the DB write failed — those are separate failures
    # and the customer shouldn't inherit a bookkeeping problem as a dead end.
    await _graph_app.aupdate_state(config, {
        "payment_link_sent": True,
        "last_payment_status": "link_created",
        "payment_link_url": link["short_url"],
    })
    logger.info(f"[{thread_id}] Payment link {link['link_id']} sent for ₹{amount:,.2f}.")


async def _handle_staff_command(thread_id: str, msg: dict, log_tag: str) -> bool:
    """
    A colleague controlling somebody else's thread — not a customer. True when consumed.

    Runs before the typing dots and before the customer mutex, because none of the ordinary turn
    applies: a staff command is ABOUT another conversation, so it must never build graph state for
    the sender's own number, never run triage on it, and never leave them holding a customer's lock.

    The allowlist is the authorisation. A number that is not on it falls through to the normal sales
    path and gets an ordinary reply — deliberately not an error and not a hint, because anyone who
    learned the syntax could otherwise release any hold on any thread. The commands are also
    invisible when `STAFF_WHATSAPP_NUMBERS` is unset, which is the default.
    """
    staff = settings.staff_numbers
    if not staff or thread_id[-10:] not in staff:
        return False

    text = (msg.get("text", {}) or {}).get("body", "") or ""
    if not looks_like_staff_command(text):
        return False

    command, target, note = parse_staff_command(text)
    logger.info(f"{log_tag} Staff command '{command}' for {target or '<no number>'}.")

    async def _notify(customer: str, body: str) -> bool:
        return await _whatsapp.dispatch_message(
            thread_id=customer, webhook_msg_id=f"handback:{msg.get('id') or int(time.time())}",
            node_name="handback", msg_index=0, text=body,
            options=None, last_user_message_timestamp=time.time(),
        )

    try:
        if not target:
            reply = STAFF_COMMAND_HELP
        else:
            # The number as typed is usually not the thread id: a colleague writes "9812345678" and
            # the conversation is keyed "919812345678". Resolving both spellings is what stops
            # "No conversation found" arriving while the customer is visibly still being held.
            resolved, tried = await resolve_thread(_graph_app, target, sender=thread_id)
            if not resolved:
                also = [c for c in tried if c != target]
                reply = (
                    f"I couldn't find a conversation for {target}"
                    + (f" (also tried {', '.join(also)})" if also else "")
                    + ". Check the number, or send #status <number> to look again."
                )
            elif command == "status":
                reply = await handoff_status(_graph_app, resolved)
            else:
                reply = await release_handoff(
                    _graph_app, resolved, note,
                    tell_customer=(command == "back"), notify=_notify,
                )
    except Exception as e:
        logger.error(f"{log_tag} Staff command failed: {e}", exc_info=True)
        reply = "Something went wrong running that. The thread has not been changed."

    await _whatsapp.dispatch_message(
        thread_id=thread_id, webhook_msg_id=f"staffcmd:{msg.get('id') or int(time.time())}",
        node_name="staff_command", msg_index=0, text=reply,
        options=None, last_user_message_timestamp=time.time(),
    )
    return True


@broker.task(max_retries=3, queue_name=settings.TASKIQ_QUEUE_NAME)
async def taskiq_process_message(msg: dict):
    thread_id = msg.get("from")
    msg_type = msg.get("type")
    webhook_msg_id = msg.get("id", "internal_triggered_job")
    # One id per turn, stamped on every log line below. thread_id says WHICH conversation;
    # request_id says which TURN of it, so concurrent chats can be untangled in the logs.
    rid = new_request_id()
    log_tag = f"[{thread_id}|{rid}]"

    if not thread_id:
        logger.error("Message has no 'from' field. Dropping.")
        return

    # A handoff command from an allowlisted colleague is not a conversational turn: no dots, no
    # mutex, no graph. Everything below this line assumes the sender is a customer.
    if await _handle_staff_command(thread_id, msg, log_tag):
        return

    # Typing dots FIRST, before the mutex. The customer should see acknowledgement within a
    # second even when a previous turn still holds the lock — that wait is exactly when silence
    # feels worst. Only a real inbound wamid can carry it, so internally-triggered jobs skip it.
    # `lock_acquired` lets the heartbeat re-post the dots if that wait ate Meta's ~25s window, and
    # `reply_started` is what keeps them off a turn that answers instantly: the six no-LLM paths are
    # sending before the grace window is even up, and dots drawn after the answer are worse than none.
    inbound_wamid = msg.get("id")
    turn_started_at = time.monotonic()
    stop_heartbeat = asyncio.Event()
    lock_acquired = asyncio.Event()
    reply_started = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        typing_heartbeat(
            thread_id, inbound_wamid, stop_heartbeat, lock_acquired, turn_started_at, reply_started
        )
    )

    # 1. Mutex Lock with Graceful Async Polling (owner-safe)
    lock_value = None
    for attempt in range(1, 31):
        lock_value = await _cache.acquire_lock(thread_id, timeout=30)
        if lock_value:
            break
        if attempt == 1:
            logger.warning(f"{log_tag} Lock held by previous message. Waiting to acquire...")
        await asyncio.sleep(1)

    if not lock_value:
        logger.error(f"{log_tag} Could not acquire lock after 30s. Dropping message to prevent pileup.")
        stop_heartbeat.set()
        heartbeat_task.cancel()
        return

    lock_acquired.set()
    keepalive_task = asyncio.create_task(lock_keepalive(thread_id, stop_heartbeat))

    try:
        # 2. Input Extraction (H5 FIX: handle interactive payloads correctly)
        if msg_type == "audio":
            media_id = msg.get("audio", {}).get("id")  # Meta sends media ID, not direct link
            text_input = await TranscoderService.transcribe_audio(media_id)
            if not text_input:
                # Transcription is unavailable. Say so instead of guessing what they said: an
                # invented transcript would have the agent confidently answer a question the
                # customer never asked, and in autonomy mode it could be acted on.
                await _whatsapp.dispatch_message(
                    thread_id=thread_id,
                    webhook_msg_id=f"noaudio:{webhook_msg_id}",
                    node_name="transcoder",
                    msg_index=0,
                    text=(
                        "Thanks for that! I can't listen to voice notes just yet — could you "
                        "type it out for me, or tap an option? I'll take it from there."
                    ),
                    last_user_message_timestamp=time.time(),
                )
                return
        elif msg_type == "interactive":
            # H5 FIX: Interactive messages have button_reply or list_reply, NOT text.body
            interactive = msg.get("interactive", {})
            button_reply = interactive.get("button_reply", {})
            list_reply = interactive.get("list_reply", {})
            title = button_reply.get("title") or list_reply.get("title") or ""
            postback_id = button_reply.get("id") or list_reply.get("id") or ""
            # Inject the postback ID into the text so deterministic routing (like checkout gates) can match it robustly.
            text_input = f"{title} [{postback_id}]" if postback_id else title
        else:
            text_input = msg.get("text", {}).get("body", "")

        if not text_input:
            logger.warning(f"{log_tag} Empty text input extracted from msg_type={msg_type}. Skipping.")
            return

        # M4 FIX: Sanitize input before LangGraph injection
        text_input = Guardrails.sanitize_input(text_input)

        # 3. LangGraph Streaming Execution
        # `configurable.thread_id` is what the checkpointer keys on; the tracing fragment adds the
        # Langfuse callback + turn metadata and degrades to an empty callback list when unset.
        config = {"configurable": {"thread_id": thread_id}}
        run_config = {
            **config,
            **langfuse_config(thread_id, rid, extra_metadata={"msg_type": msg_type or "text"}),
        }

        # Semantic recall (mem0) — durable facts about this person from EARLIER sessions, fetched
        # before the graph runs and passed in as state so the archetype prompts can personalise
        # via {memory_block}. Best-effort: no mem0, no facts, no difference to the turn.
        # Skipped for button taps and bare greetings/affirmations: there is no query to match on,
        # so the search is latency the customer pays for nothing.
        memory_facts = []
        _mem = get_semantic_memory()
        if _mem and _is_worth_recalling(text_input, msg_type):
            memory_facts = await _mem.search(text_input, user_id=thread_id)
            if memory_facts:
                logger.info(f"{log_tag} Recalled {len(memory_facts)} semantic fact(s).")

        # Track the timestamp for 24h rule enforcement
        last_timestamp = msg.get("timestamp", time.time())
        if isinstance(last_timestamp, str):
            last_timestamp = float(last_timestamp)

        logger.info(f"{log_tag} Turn start (msg_type={msg_type}).")
        reply_texts = []
        async for output in _graph_app.astream(
            {"messages": [("user", text_input)], "memory_facts": memory_facts},
            config=run_config,
            stream_mode="updates",
        ):
            for node_name, state_updates in output.items():
                messages = state_updates.get("messages", [])
                if messages:
                    for idx, m in enumerate(messages):
                        if hasattr(m, "type") and m.type == "ai":
                            options = m.response_metadata.get("options") if hasattr(m, "response_metadata") else None
                            reply_texts.append(m.content)
                            # Tells the typing heartbeat the wait is over. Set before the send, not
                            # after: on a no-LLM path the send itself is the slowest thing in the
                            # turn, and dots posted during it would land after the answer.
                            reply_started.set()
                            await _whatsapp.dispatch_message(
                                thread_id=thread_id,
                                webhook_msg_id=webhook_msg_id,
                                node_name=node_name,
                                msg_index=idx,
                                text=m.content,
                                options=options,
                                last_user_message_timestamp=last_timestamp
                            )
                            # Brief pause so a multi-message reply reads as consecutive chat
                            # bubbles rather than one wall of text. Kept short: it is dead time
                            # the customer spends watching nothing happen.
                            await asyncio.sleep(0.6)

        # ── Lead capture delivery + checkout, off ONE state snapshot ───────────
        # Both blocks read post-turn state; taking a single snapshot avoids paying for the
        # checkpointer round trip twice while the customer's mutex is still held.
        try:
            snapshot = await _graph_app.aget_state(config)
            fstate = snapshot.values if snapshot else {}
        except Exception as snap_err:
            logger.error(f"{log_tag} Post-turn state read failed (non-fatal): {snap_err}")
            fstate = {}

        # After the turn, deliver what the team needs — as TWO independent things, because they are
        # two jobs for two people. A lead is somebody to call back; an escalation is somebody
        # waiting right now. One flag each, so a hot lead that later needs a person reaches both
        # sheets exactly once, in whichever order it happened. Fully isolated: a delivery failure
        # must never retry the task (that would re-send WhatsApp replies) or break the turn.
        try:
            deliveries = []
            if fstate.get("lead_ready_for_handoff") and not fstate.get("lead_sent"):
                deliveries.append((KIND_LEAD, "lead_sent"))
            if fstate.get("requires_human_handoff") and not fstate.get("escalation_sent"):
                deliveries.append((KIND_ESCALATION, "escalation_sent"))
            for kind, flag in deliveries:
                await CRMHandoffService.freeze_and_handoff(thread_id, fstate, kind=kind)
                await _graph_app.aupdate_state(config, {flag: True})
        except Exception as lead_err:
            logger.error(f"{log_tag} Lead delivery failed (non-fatal): {lead_err}")

        # ── Brochure ────────────────────────────────────────────────────────────
        # The graph only ever *asks*; the file is attached here. This exists because the agent used
        # to say "here's our digital lookbook" and send nothing — an unkept promise about the one
        # thing the customer explicitly asked for.
        if fstate.get("brochure_requested") and settings.brochure_url:
            sent = await _whatsapp.send_document(
                thread_id=thread_id,
                webhook_msg_id=f"brochure:{webhook_msg_id}",
                url=settings.brochure_url,
                filename=settings.BROCHURE_FILENAME,
            )
            if not sent:
                # Say so rather than leaving them waiting for a file that isn't coming.
                await _whatsapp.dispatch_message(
                    thread_id=thread_id, webhook_msg_id=f"brochure_fail:{webhook_msg_id}",
                    node_name="brochure", msg_index=0,
                    text=("I couldn't attach the lookbook just now — I'll get it across shortly. "
                          "In the meantime, ask me anything you'd have looked for in it."),
                    options=None, last_user_message_timestamp=last_timestamp,
                )
            await _graph_app.aupdate_state(config, {"brochure_requested": False})

        # ── Agentic checkout: mint the payment link (the money action) ─────────
        # Runs OUTSIDE the graph, beside the lead block, so a checkpointer replay or a
        # TaskIQ retry can never re-mint a link. Fires only after an explicit "Confirm & pay"
        # tap (checkout_confirmed, set in code by triage) on a server-built pending_order, and
        # exactly once (payment_link_sent dedup). Fully isolated + fail-soft: a payment-step
        # failure must never retry the task (that would re-send the whole turn's replies).
        try:
            if (
                fstate.get("checkout_confirmed")
                and fstate.get("pending_order")
                and not fstate.get("payment_link_sent")
                # The name is mandatory at the pay button (the client's rule). While the agent is
                # waiting for it, the tap is authorised but the link is held — one turn, then it
                # mints on the same authorisation.
                and not fstate.get("awaiting_pay_details")
            ):
                await _process_checkout(thread_id, config, fstate["pending_order"], webhook_msg_id, last_timestamp)
        except Exception as pay_err:
            logger.error(f"{log_tag} Checkout/payment step failed (non-fatal): {pay_err}")

        # ── Semantic memory write (enqueued, not awaited) ───────────────────────
        # mem0 runs its own LLM extraction pass and is the slowest thing in the turn. Awaiting
        # it here delayed nothing the customer can see — but it happened INSIDE wa_mutex, so it
        # delayed their NEXT message by however long it took. Hand it to its own task instead.
        if _mem:
            try:
                exchange = f"Customer: {text_input}"
                if reply_texts:
                    exchange += "\nAgent: " + "\n".join(t for t in reply_texts if t)
                await taskiq_store_memory.kiq(thread_id, exchange)
            except Exception as mem_err:
                logger.warning(f"{log_tag} Could not enqueue semantic memory write (non-fatal): {mem_err}")

        logger.info(
            f"{log_tag} Turn complete ({len(reply_texts)} message(s) sent, "
            f"{time.monotonic() - turn_started_at:.1f}s)."
        )

    except TaskRejectedError:
        raise  # Let TaskIQ handle retries
    except Exception:
        logger.error(f"{log_tag} Pipeline error", exc_info=True)
        raise  # preserve the original traceback
    finally:
        stop_heartbeat.set()
        # Cancel + drain the background tasks defensively. A failing or hung heartbeat/
        # keepalive must NEVER skip the lock release below (which would hold wa_mutex
        # until TTL and block the thread). release_lock is owner-safe and always runs.
        for task in (heartbeat_task, keepalive_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as bg_err:
                logger.warning(f"{log_tag} Background task error during cleanup: {bg_err}")
        await _cache.release_lock(thread_id, lock_value)  # H6 FIX: pass lock_value
        logger.info(f"{log_tag} Lock released.")


# ═══════════════════════════════════════════════
#  Razorpay webhook → close the checkout loop (the payment-confirm consumer)
# ═══════════════════════════════════════════════

async def _find_payment_order(session, link_id: Optional[str], thread_id: Optional[str]):
    """Locate the PaymentOrder a webhook refers to: by Razorpay link id first (exact), else the
    thread's most recent order (a payment.failed event often carries no link id). Returns row|None."""
    if link_id:
        row = (await session.execute(
            select(PaymentOrder)
            .where(PaymentOrder.razorpay_link_id == link_id)
            .order_by(PaymentOrder.created_at.desc())
        )).scalars().first()
        if row is not None:
            return row
    if thread_id:
        return (await session.execute(
            select(PaymentOrder)
            .where(PaymentOrder.thread_id == thread_id)
            .order_by(PaymentOrder.created_at.desc())
        )).scalars().first()
    return None


async def _handle_payment_failure(session, row, event: dict, parsed: dict, thread_id: str, config: dict):
    """
    Payment-decline policy — the graceful-failure half of the autonomy story.

    A single decline is a normal, recoverable event: nudge once, keep the SAME link live, and offer a
    human escape hatch. Repeated declines are where the agent stops trying alone and trips the CRITICAL
    escalation safety valve (requires_human_handoff + a one-time human alert) — the "human stays for
    critical cases" rule. The running count lives in graph state; it relies on the edge dedup
    (idem_rzp:{event_id} in webhooks.py) so a Razorpay redelivery doesn't double-count.
    """
    try:
        snap = await _graph_app.aget_state(config)
        state = snap.values if snap else {}
    except Exception:
        state = {}
    count = int(state.get("payment_failure_count", 0) or 0) + 1
    reason = parsed.get("error_description") or "the payment couldn't be completed"
    # Prefer the audit row, fall back to graph state: the retry has to offer a link even if the
    # row is missing for an unrelated reason.
    link_url = (getattr(row, "payment_link_url", None) if row is not None else None) \
        or state.get("payment_link_url")
    escalate = count >= settings.MAX_PAYMENT_FAILURES

    # Record the attempt on the order. Only flip status to 'failed' when we actually give up
    # (escalate); below the threshold the link is still live, so leave its status intact.
    if row is not None and row.status != "paid":
        try:
            row.audit_notes = list(row.audit_notes or []) + [f"payment.failed x{count}: {reason}"]
        except Exception:
            pass
        row.raw_event = event or {}
        if escalate:
            row.status = "failed"
        await session.commit()

    if not escalate:
        await _graph_app.aupdate_state(config, {"last_payment_status": "failed", "payment_failure_count": count})

        # The ladder is deliberate. A first decline is usually a bank or card quirk, not a
        # problem needing a person — offering a human that early implies something is broken and
        # pulls the customer out of a sale they were completing. So: retry the live link only.
        # From the SECOND decline onward, add the hatch, because now it plausibly isn't them.
        retry_line = (
            f"The secure link is still live — give it another go:\n{link_url}"
            if link_url else
            "Give it another try in a moment and it should go through."
        )
        if count == 1:
            text = (
                f"That payment didn't go through — {reason}. Nothing to worry about, it happens.\n\n"
                f"{retry_line}"
            )
            options = None
        else:
            text = (
                f"Still not going through — {reason}.\n\n"
                f"{retry_line}\n\n"
                "If it fails again, tap below and I'll bring in a teammate who can sort it with you."
            )
            options = [{"label": "Talk to a person", "postback_id": "CONNECT_NOW"}]

        await _whatsapp.dispatch_message(
            thread_id=thread_id, webhook_msg_id=f"payfail:{parsed.get('payment_id') or count}",
            node_name="payment_failed", msg_index=0, text=text,
            options=options,
            last_user_message_timestamp=time.time(),
        )
        logger.info(f"[{thread_id}] Payment decline #{count} handled in-agent (recoverable).")
        return

    # Repeated declines -> CRITICAL human escalation (the safety valve).
    await _graph_app.aupdate_state(config, {
        "last_payment_status": "failed",
        "payment_failure_count": count,
        "requires_human_handoff": True,
        "handoff_reason": "payment",
        "handoff_active": True,
        "handoff_notified": True,
        "handoff_started_at": time.time(),
    })
    await _whatsapp.dispatch_message(
        thread_id=thread_id, webhook_msg_id=f"payfail_escalate:{thread_id}:{count}",
        node_name="payment_failed", msg_index=0,
        text=("I'm sorry — that still didn't go through, and I don't want you to keep trying. "
              "I've asked a colleague to take this over and complete the order with you; they have "
              "your full order details, so there's nothing for you to repeat."),
        options=None, last_user_message_timestamp=time.time(),
    )
    # Alert a person once. This is an escalation, not a lead — a customer whose card has failed
    # three times is waiting, not browsing — so it uses the escalation flag and lands in the
    # escalations sheet with the reason on it.
    try:
        snap = await _graph_app.aget_state(config)
        fstate = snap.values if snap else {}
        if not fstate.get("escalation_sent"):
            await CRMHandoffService.freeze_and_handoff(thread_id, fstate, kind=KIND_ESCALATION)
            await _graph_app.aupdate_state(config, {"escalation_sent": True})
    except Exception as e:
        logger.error(f"[{thread_id}] Payment-failure escalation alert failed (non-fatal): {e}")
    logger.info(f"[{thread_id}] Payment failed {count}x — escalated to human (critical safety valve).")


def _paid_line_items(row) -> dict:
    """
    {sku: qty} for a settled order, from the audit row that is the trail for it.

    This is what survives clearing `pending_order` on payment, and it exists for exactly one
    consumer: sales.py::_reproposes_paid_order, which needs to tell a model restating an order the
    customer already owns from a genuine second purchase. Quantities are kept for that reason — the
    same sku at a higher quantity is a new order, not a repeat of the old one.

    Tolerant of a partial row: the audit write is fail-soft, so a missing or malformed `line_items`
    must degrade to {} rather than raise inside the paid handler and abandon the rest of it.
    """
    out: dict = {}
    for li in (getattr(row, "line_items", None) or []):
        if not isinstance(li, dict):
            continue
        sku = str(li.get("sku") or "").strip()
        if not sku:
            continue
        try:
            qty = int(li.get("qty", 1) or 1)
        except (TypeError, ValueError):
            qty = 1
        out[sku] = out.get(sku, 0) + max(qty, 1)
    return out


def _payment_celebration(row, customer_name: str = "") -> str:
    """
    The moment message, sent on its own immediately before the receipt block.

    Someone has just spent real money on their home, and "Payment received, thank you" treats that
    like a vending machine. This is the one point in the conversation where warmth is the correct
    professional response — so it leads with the status, thanks them, and then stops.

    Deliberately not product-specific: this is code that runs for every order, and a confident
    generic line beats a specific one that might not fit what they bought. The name is used only
    when the customer actually gave it — normally they have, since no link is minted for an order
    with nobody's name on it, but a link resent from an older order may predate that rule, so a
    greeting is never guessed.
    """
    name = (customer_name or "").strip()
    thanks = f"Thank you, {name} —" if name else "Thank you —"
    return (
        "*Order confirmed* 🎉\n\n"
        f"{thanks} your payment has gone through and your order is locked in.\n"
        "We're on it from here."
    )


def _format_payment_receipt(row, amount: float) -> str:
    """
    The evidence block, sent as its own message so it stays scrollable-to.

    Someone who has just parted with money should be left holding a reference, not a reassurance.
    The order reference leads, on its own line, because it is the first thing anyone actually comes
    back to this message for — theirs, the team's, or their bank's. Amount, payment mode and
    transaction id follow for the same reason.

    WhatsApp bold sits on short standalone labels only; markers adjacent to digits or punctuation
    render unreliably, so no figure is wrapped in them. Monospace has no such problem on a
    standalone token, and it makes the reference long-press-copyable — which is the only thing a
    reference number is for.
    """
    lines = ["*Your Otohom receipt*"]

    if getattr(row, "id", None):
        lines.append("")
        lines.append("Order reference")
        lines.append(f"```OTO-{row.id}```")

    items = list(getattr(row, "line_items", None) or [])
    if items or getattr(row, "product_summary", None):
        lines.append("")
        lines.append("*What you ordered*")
    if items:
        for li in items:
            sku = str(li.get("sku") or "")
            lines.append(f"{sku}  ×{int(li.get('qty', 1) or 1)}")
            # The same gloss the quote carried, so the receipt doesn't suddenly read like
            # electrician shorthand for a product they recognised by name an hour ago.
            plain = discounts.plain_product_name(sku)
            if plain:
                lines.append(f"_{plain}_")
    elif getattr(row, "product_summary", None):
        lines.append(str(row.product_summary))

    lines.append("")
    lines.append("*Payment*")
    if amount > 0:
        lines.append(f"Amount paid: ₹{amount:,.0f}")
    mode = "Razorpay (test mode)" if str(settings.RAZORPAY_KEY_ID or "").startswith("rzp_test") else "Razorpay"
    lines.append(f"Payment mode: {mode}")
    reference = getattr(row, "razorpay_payment_id", None) or getattr(row, "razorpay_link_id", None)
    if reference:
        lines.append(f"Transaction ID: ```{reference}```")

    lines.append("")
    lines.append("*What happens next*")
    lines.append("Someone from our team will call you to confirm the details and book your "
                 "installation slot.")
    lines.append("")
    lines.append("Anything at all before then, just message here.")
    return "\n".join(lines)


async def _send_payment_confirmation(
    thread_id: str, config: dict, row, amount: float, paid_name: str, dedup_key: str
) -> bool:
    """
    The two post-payment messages, in order, sent by exactly one caller.

    Razorpay sends more than one event for the same money — `payment_link.paid` AND
    `payment.captured` — with different event ids, so the `X-Razorpay-Event-Id` dedup in the webhook
    cannot collapse them and both reach the paid branch. Each individual send was already idempotent,
    which turned out to be the wrong grain: the two runs interleaved, the loser's receipt was skipped
    as a duplicate while the winner's was still in flight, and the loser then ran straight on to the
    name-and-city ask — which overtook the receipt. The customer read the question, then a long
    receipt on top of it. So the CLAIM is over the whole sequence, and it is what makes "the ask is
    last" true rather than likely.

    Deliberately claimed here and not in the caller's money block: the DB status write and the
    aupdate_state that clears `pending_order` must run for BOTH events, because losing a race must
    never be what leaves a paid order sitting in state one tap from a second link. Only the chat has
    a single owner.

    Returns True when this call owned the sequence. Fail-soft throughout: a chat failure here must
    not retry the task and re-charge anything.
    """
    is_owner = await _cache.check_and_set_idempotency(
        thread_id, f"paid-notify:{dedup_key}", "payment_confirm_sequence", 0
    )
    if not is_owner:
        logger.info(
            f"[{thread_id}] Another paid event already sent the confirmation for {dedup_key}; "
            "skipping the chat sequence."
        )
        return False

    # Two messages, deliberately. The first is the moment; the second is the evidence they keep.
    # Merged, the transaction id is buried inside a celebration and the celebration makes the receipt
    # read like marketing. The pause between them is the same one a multi-message turn uses.
    #
    # The receipt is LAST on purpose and nothing follows it. The name-and-city question used to sit
    # here, after the money was done, where it was routinely ignored — it now runs at the pay button,
    # where the name is mandatory, so by this point we already know who bought.
    for index, text in enumerate((
        _payment_celebration(row, paid_name),
        _format_payment_receipt(row, amount),
    )):
        await _whatsapp.dispatch_message(
            thread_id=thread_id, webhook_msg_id=f"paid:{dedup_key}",
            node_name="payment_confirm", msg_index=index, text=text,
            options=None, last_user_message_timestamp=time.time(),
        )
        if index == 0:
            await asyncio.sleep(0.6)
    return True


@broker.task(max_retries=2, queue_name=settings.TASKIQ_QUEUE_NAME)
async def taskiq_store_memory(thread_id: str, exchange: str):
    """
    Distil durable facts from one exchange into mem0, for FUTURE sessions.

    Its own task, deliberately: mem0 runs an LLM extraction pass, and doing that inside the
    conversational turn held wa_mutex for the duration — so the cost landed on the customer's
    NEXT message rather than on the one being answered. Nothing in the current turn depends on
    the result, so it belongs off the critical path.

    Best-effort by nature: if a fact fails to store, the conversation is unaffected — the agent
    simply won't remember that detail next session. Never raises.
    """
    mem = get_semantic_memory()
    if not mem:
        return
    try:
        await mem.extract_and_store(exchange, user_id=thread_id)
        logger.info(f"[{thread_id}] Semantic memory updated.")
    except Exception as e:
        logger.warning(f"[{thread_id}] Semantic memory write failed (non-fatal): {e}")


@broker.task(max_retries=3, queue_name=settings.TASKIQ_QUEUE_NAME)
async def taskiq_confirm_payment(event: dict):
    """
    Close the checkout loop from a VERIFIED Razorpay webhook (the HMAC was already checked at the
    edge in webhooks.py). Runs in the worker so it shares _whatsapp / _graph_app, but OUTSIDE the
    wa_mutex — a payment event is not a conversational turn and must not queue behind one.

    Idempotent by construction so a Razorpay redelivery is harmless: PaymentOrder status writes are
    absolute and every chat send is keyed (webhook_msg_id) through dispatch_message's dedup.

      paid    -> PaymentOrder.status='paid' (+ razorpay_payment_id, raw_event), in-chat confirmation,
                 state last_payment_status='paid', payment_failure_count reset to 0.
      failed  -> single decline nudged warmly in-agent (link stays live); repeated declines trip the
                 CRITICAL human-escalation safety valve. See _handle_payment_failure.
      expired -> recorded quietly; no chat spam.
    """
    parsed = RazorpayService.parse_event(event or {})
    etype = parsed.get("event") or ""
    link_id = parsed.get("link_id")
    payment_id = parsed.get("payment_id")
    thread_id = parsed.get("thread_id")

    if not thread_id and not link_id:
        logger.warning(f"Razorpay event '{etype}' carries no thread_id/link_id to attribute; ignoring.")
        return

    async with async_session_maker() as session:
        row = await _find_payment_order(session, link_id, thread_id)
        if thread_id is None and row is not None:
            thread_id = row.thread_id
        if not thread_id:
            logger.warning(f"Razorpay event '{etype}' (link {link_id}) matched no known order; ignoring.")
            return

        config = {"configurable": {"thread_id": thread_id}}

        # ── Paid ──────────────────────────────────────────────────────────────
        if etype in ("payment_link.paid", "payment.captured", "order.paid"):
            if row is None:
                # No proposal row (e.g. the link-id lookup missed) — still record the money movement.
                paid_amount = float(parsed.get("amount_paise") or 0) / 100.0
                row = PaymentOrder(
                    thread_id=thread_id, amount=paid_amount, currency="INR", status="paid",
                    razorpay_link_id=link_id, razorpay_payment_id=payment_id,
                    product_summary="(order confirmed via webhook)",
                )
                session.add(row)
            else:
                row.status = "paid"
                row.razorpay_payment_id = payment_id or row.razorpay_payment_id
            row.raw_event = event or {}
            await session.commit()
            await session.refresh(row)

            amount = float(row.amount or 0.0)
            # Greet by name ONLY if the thread actually holds one. It normally does — the name is
            # asked for at the pay button and no link mints without it — but a guessed name on a
            # receipt is worse than no name at all.
            paid_name = ""
            try:
                snap = await _graph_app.aget_state(config)
                paid_name = ((snap.values if snap else {}).get("customer_name") or "").strip()
            except Exception as e:
                logger.warning(f"[{thread_id}] Could not read customer_name for the receipt: {e}")
            # The state write runs for EVERY paid event, ahead of the claim inside the send helper,
            # because a settled order must stop being a pending one whether or not this run owns the
            # chat. The mint block fires on `checkout_confirmed and pending_order and not
            # payment_link_sent`, and any later turn that rebuilds a quote sets payment_link_sent
            # back to False — so a paid order left in state sat one "Confirm & pay" tap away from a
            # SECOND live link for something already bought. Observed: answering the name/city
            # question re-sent the whole quote, pay CTA and buttons included.
            await _graph_app.aupdate_state(config, {
                "last_payment_status": "paid",
                "payment_failure_count": 0,
                "pending_order": None,
                "checkout_confirmed": False,
                "payment_link_url": None,
                # What was bought, so a later turn can tell a re-proposal from a new order.
                "paid_line_items": _paid_line_items(row),
            })
            # Celebration → receipt → the name-and-city ask, as one sequence owned by one event.
            # Two Razorpay events for the same payment used to interleave here and leave the question
            # stranded in the middle.
            await _send_payment_confirmation(
                thread_id, config, row, amount, paid_name, payment_id or link_id
            )
            logger.info(f"[{thread_id}] Payment confirmed (PaymentOrder #{getattr(row, 'id', None)}).")
            return

        # ── Failed ──────────────────────────────────────────────────────────────
        if etype == "payment.failed":
            await _handle_payment_failure(session, row, event, parsed, thread_id, config)
            return

        # ── Expired / cancelled ───────────────────────────────────────────────────
        if etype in ("payment_link.expired", "payment_link.cancelled"):
            if row is not None and row.status != "paid":
                row.status = "expired"
                row.raw_event = event or {}
                await session.commit()
            logger.info(f"[{thread_id}] Payment link {link_id} {etype}; recorded, no chat sent.")
            return

    logger.info(f"[{thread_id}] Razorpay event '{etype}' not actioned.")