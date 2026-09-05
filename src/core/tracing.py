"""
Observability — Langfuse tracing + a per-turn request id.

Two small, independent things live here, both config-gated and both no-ops when unset:

1. `langfuse_config(...)` returns the dict you splat into a LangGraph/LangChain call
   (`config={...}`). When LANGFUSE_* keys are present it carries the callback handler plus
   trace metadata (thread id, request id, message type); when they aren't, it carries just the
   metadata and the handler list is empty. Callers never have to branch on whether tracing is on.

2. `new_request_id()` — a short id minted once per turn and stamped into every log line for
   that turn. The thread_id tells you WHICH conversation; the request_id tells you which TURN
   of it, so interleaved worker logs from concurrent chats can be untangled after the fact.
   This is what makes the money path auditable from logs alone, independent of Langfuse.

The Langfuse handler is built ONCE per process (it opens an HTTP client and a background
flush thread). A failure to construct it is logged and swallowed — a tracing outage must never
cost a customer their turn.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_handler: Optional[Any] = None
_client: Optional[Any] = None
_handler_built = False


def new_request_id() -> str:
    """Short, log-friendly id for one turn (one inbound WhatsApp message)."""
    return uuid.uuid4().hex[:12]


def _build_client() -> Optional[Any]:
    """
    Construct (and thereby register) the Langfuse client singleton from settings.

    Done FIRST and unconditionally, before the callback handler, because from v3 onward the
    handler carries no credentials — it resolves them from the process-wide client. Building
    the handler first would hand us an unconfigured handler that silently drops every trace,
    which is the worst outcome: tracing looks wired and records nothing.

    The credential kwarg for the endpoint was `host` through v3 and `base_url` in v4, so we
    try both rather than pinning a version. requirements.txt only floors at >=2.36.
    """
    from langfuse import Langfuse

    creds = {
        "public_key": settings.LANGFUSE_PUBLIC_KEY,
        "secret_key": settings.LANGFUSE_SECRET_KEY,
    }
    for endpoint_kwarg in ("host", "base_url"):
        try:
            return Langfuse(**creds, **{endpoint_kwarg: settings.LANGFUSE_HOST})
        except TypeError:
            continue
    return Langfuse(**creds)  # last resort: let it read the endpoint from its own env defaults


def get_langfuse_handler() -> Optional[Any]:
    """
    Process-wide Langfuse CallbackHandler, or None when tracing is off/unavailable.

    Tolerates every published layout: the LangChain handler lives at `langfuse.langchain`
    from v3 and at `langfuse.callback` in v2; the handler takes credentials in v2 and takes
    none (reading the client singleton) from v3 on.
    """
    global _handler, _client, _handler_built
    if _handler_built:
        return _handler
    _handler_built = True

    if not settings.langfuse_enabled:
        logger.info("Langfuse keys not set; tracing disabled.")
        return None

    try:
        try:
            from langfuse.langchain import CallbackHandler  # v3+
        except ImportError:
            from langfuse.callback import CallbackHandler  # v2

        try:
            _client = _build_client()
        except Exception as client_err:
            # v2 has a Langfuse client too, but there the handler owns its own — so a client
            # failure is not fatal on that line. Keep going and let the handler try.
            logger.warning(f"Langfuse client init failed, trying handler-owned credentials: {client_err}")

        try:
            _handler = CallbackHandler()
        except TypeError:
            # v2: the handler is the thing that carries credentials.
            _handler = CallbackHandler(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
        logger.info(f"Langfuse tracing enabled (host={settings.LANGFUSE_HOST}).")
    except Exception as e:
        logger.warning(f"Langfuse handler unavailable (tracing disabled, non-fatal): {e}")
        _handler = None
    return _handler


def langfuse_config(
    thread_id: str,
    request_id: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build the LangGraph/LangChain `config` fragment for one turn.

    Merge it into the call's config rather than replacing it — the graph needs its own
    `configurable.thread_id` to find the right checkpoint:

        config = {"configurable": {"thread_id": tid}, **langfuse_config(tid, rid)}
    """
    # `langfuse_session_id` / `langfuse_user_id` are the documented reserved keys the Python
    # LangChain integration lifts out of metadata onto the trace itself. Setting both to the
    # WhatsApp number is what groups a customer's many turns into ONE session in the UI and
    # attributes them to that person — without it every turn is an orphan trace. (Their bare
    # `session_id`/`user_id` spellings are JS-only; they'd sit in metadata doing nothing here.)
    metadata: Dict[str, Any] = {
        "thread_id": thread_id,
        "request_id": request_id,
        "environment": settings.APP_ENV,
        "agent_full_autonomy": settings.AGENT_FULL_AUTONOMY,
        "langfuse_session_id": thread_id,
        "langfuse_user_id": thread_id,
        "langfuse_tags": tags or ["otohom", "whatsapp"],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    handler = get_langfuse_handler()
    return {
        "callbacks": [handler] if handler else [],
        "metadata": metadata,
        "run_name": f"otohom_turn:{thread_id}",
        "tags": tags or ["otohom", "whatsapp"],
    }


def flush_langfuse() -> None:
    """
    Flush buffered traces at worker shutdown so the last turns aren't lost.

    Tracing is batched on a background thread, so a worker that exits promptly drops whatever
    is still queued — including the payment turn, which is the one you most want in the trace.
    Where the flush lives moved between versions (the v2 handler owns a client; from v3 the
    client is a module-level singleton), so try each known holder in turn.
    """
    for holder in (_client, _handler, getattr(_handler, "client", None)):
        flush = getattr(holder, "flush", None)
        if callable(flush):
            try:
                flush()
                return
            except Exception as e:
                logger.warning(f"Langfuse flush failed (non-fatal): {e}")
                return
    # v3+ with no client handle of our own: fall back to the SDK's singleton accessor.
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass
