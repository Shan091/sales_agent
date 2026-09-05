"""
Release a thread a human took over, and hand the agent whatever context it needs to carry on.

Handoff is human-driven: after `node_human_escalation` marks a thread `handoff_active`, the agent
answers further messages with a short holding line and nothing else until a colleague releases it
here. That is deliberate — the colleague is working on a DIFFERENT WhatsApp number, so nothing
they say to the customer is visible to this system. If the agent resumed on its own it would be
answering blind, in parallel with a real person, on two numbers at once.

Which is also the answer to "how does the agent get the context of the human conversation": it
does not, and it cannot. The note is the only channel. Whatever you write there is what the agent
knows; anything you leave out, it will not know. Write the outcome, not the transcript:

    python -m src.scripts.resolve_handoff 919812345678 \
        --note "Card was blocked for international. Paid by UPI on the phone, order confirmed."

    python -m src.scripts.resolve_handoff 919812345678 \
        --note "Wanted a 12-panel villa quote. Site visit booked Thu 4pm." --tell-customer

A colleague with a phone and no terminal does the same thing by texting the agent's own WhatsApp
number — `#done 919812345678 <what you did>` — from a number on `STAFF_WHATSAPP_NUMBERS`. Both
channels call `src/services/handoff_control.py`, so they cannot drift apart.

Usage:
    python -m src.scripts.resolve_handoff <whatsapp_number> --note "<what you did>" [--tell-customer]
    python -m src.scripts.resolve_handoff <whatsapp_number> --status
"""
import argparse
import asyncio
import logging
import time

from src.graph.workflow import compile_workflow_async
from src.services.handoff_control import handoff_status, release_handoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def show_status(thread_id: str) -> None:
    app = await compile_workflow_async()
    print(await handoff_status(app, thread_id))


async def resolve(thread_id: str, note: str, tell_customer: bool) -> None:
    app = await compile_workflow_async()

    notify = None
    if tell_customer:
        # Imported lazily: the common path (release only) needs no Redis connection.
        import redis.asyncio as aioredis
        from config.settings import settings
        from src.services.whatsapp import WhatsAppService
        from src.storage.cache import CacheService

        redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

        async def notify(target: str, text: str) -> bool:
            try:
                wa = WhatsAppService(CacheService(redis))
                await wa.dispatch_message(
                    thread_id=target,
                    webhook_msg_id=f"handback:{int(time.time())}",
                    node_name="handback",
                    msg_index=0,
                    text=text,
                    options=None,
                    last_user_message_timestamp=time.time(),
                )
                return True
            finally:
                closer = getattr(redis, "aclose", None) or getattr(redis, "close", None)
                if closer:
                    await closer()

    print(await release_handoff(app, thread_id, note, tell_customer=tell_customer, notify=notify))


def main() -> None:
    parser = argparse.ArgumentParser(description="Release a human-held conversation back to the agent.")
    parser.add_argument("thread_id", help="WhatsApp number in full international form, e.g. 919812345678")
    parser.add_argument("--note", help="What you did / what the agent must know to carry on. Required to release.")
    parser.add_argument("--status", action="store_true", help="Show the hold status and exit; change nothing.")
    parser.add_argument("--tell-customer", action="store_true",
                        help="Also send the customer a short 'I'm back' message.")
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status(args.thread_id))
        return
    if not args.note:
        parser.error(
            "--note is required. It is the ONLY way what you did reaches the agent — it cannot "
            "see your conversation with the customer."
        )
    asyncio.run(resolve(args.thread_id, args.note, args.tell_customer))


if __name__ == "__main__":
    main()
