import json
import logging
from fastapi import APIRouter, Request, Depends, Response
from src.api.dependencies import get_redis
import hmac
import hashlib
from config.settings import settings
import redis.asyncio as redis

# Deep-path work is enqueued to the TaskIQ worker (Meta Cloud ingress → async graph run).
from src.tasks.processing import taskiq_process_message, taskiq_confirm_payment
from src.services.razorpay_service import RazorpayService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
@router.get("")
async def verify_webhook(request: Request):
    """WhatsApp webhook verification endpoint (H4 FIX)."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WHATSAPP_VERIFY_TOKEN:
        return Response(content=challenge, status_code=200)
    logger.warning(f"Webhook verification failed. mode={mode}, token_match={token == settings.WHATSAPP_VERIFY_TOKEN}")
    return Response(status_code=403)


@router.post("/")
@router.post("")
async def receive_webhook(request: Request, redis_pool: redis.Redis = Depends(get_redis)):
    """
    Ingests WhatsApp payloads.
    Implements Fast/Deep path partitioning, Mutex checking, and SLA compliance.
    """
    # FIX: Read body ONCE and reuse for both HMAC and JSON parsing
    body = await request.body()

    signature = request.headers.get("X-Hub-Signature-256", "")
    expected_hash = hmac.HMAC(
        key=settings.META_APP_SECRET.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, f"sha256={expected_hash}"):
        logger.error("Invalid Meta signature. Dropping malicious payload.")
        return Response(status_code=401)

    try:
        payload = json.loads(body)
    except Exception as e:
        logger.warning(f"Malformed webhook JSON body dropped: {e}")
        return Response(status_code=400)

    # Extract entries
    entries = payload.get("entry", [])
    for entry in entries:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})

            # Handle status updates (delivery receipts, read receipts) — ignore them
            statuses = value.get("statuses", [])
            if statuses:
                continue

            messages = value.get("messages", [])

            for msg in messages:
                thread_id = msg.get("from")
                msg_type = msg.get("type")
                msg_id = msg.get("id")

                if not thread_id or not msg_id:
                    continue

                # 0. Ingestion-Level Idempotency (Stress-Test Fix)
                # Prevents Meta retry storms from hitting TaskIQ and wasting LLM tokens.
                idem_key = f"idem_in:{msg_id}"
                is_new = await redis_pool.set(idem_key, "1", nx=True, ex=86400)
                if not is_new:
                    logger.info(f"[{thread_id}] Duplicate webhook payload {msg_id} blocked at ingestion.")
                    continue

                # 1. Audio Media Ingestion (SLA Compliant)
                if msg_type == "audio":
                    logger.info(f"[{thread_id}] Audio detected. Deferring straight to Deep Path TaskIQ.")
                    await taskiq_process_message.kiq(msg)
                    continue

                # 2. Route all messages (text + interactive button/list replies) to Deep Path TaskIQ.
                # The Fast Path optimization (deterministic handlers bypassing LangGraph) is
                # intentionally deferred until the graph has stable, production-ready deterministic
                # nodes to plug in. Until then, ALL message types go through LangGraph.
                if msg_type == "interactive":
                    logger.info(f"[{thread_id}] Interactive reply detected. Pushing to Deep Path TaskIQ.")
                else:
                    logger.info(f"[{thread_id}] Text detected. Pushing to Deep Path TaskIQ.")
                await taskiq_process_message.kiq(msg)

    # ALWAYS return 202 immediately to protect Meta < 5s SLA
    return Response(status_code=202)


@router.post("/razorpay")
async def receive_razorpay_webhook(request: Request, redis_pool: redis.Redis = Depends(get_redis)):
    """
    Razorpay payment webhook — closes the checkout loop.

    Mirrors the Meta handler: read the raw body ONCE (the HMAC is over exact bytes),
    verify the signature fail-closed, dedupe on Razorpay's delivery id, then hand the
    parsed event to the worker and return fast. The worker marks the PaymentOrder paid/
    failed and sends the in-chat confirmation. We never do DB/LLM work on this hot path.
    """
    # Raw bytes for HMAC — re-serializing parsed JSON would change the digest.
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not RazorpayService.verify_webhook_signature(body, signature):
        logger.error("Invalid Razorpay signature. Dropping payload.")
        return Response(status_code=401)

    try:
        payload = json.loads(body)
    except Exception as e:
        logger.warning(f"Malformed Razorpay webhook body dropped: {e}")
        return Response(status_code=400)

    # Idempotency: Razorpay may redeliver the same event. Dedupe on its unique delivery id
    # so a paid/failed transition is processed once. The worker task is idempotent anyway
    # (status writes are absolute, chat sends are keyed), so a missing header just skips dedupe.
    event_id = request.headers.get("X-Razorpay-Event-Id", "")
    if event_id:
        idem_key = f"idem_rzp:{event_id}"
        is_new = await redis_pool.set(idem_key, "1", nx=True, ex=86400)
        if not is_new:
            logger.info(f"Duplicate Razorpay event {event_id} blocked at ingestion.")
            return Response(status_code=200)

    await taskiq_confirm_payment.kiq(payload)
    return Response(status_code=202)