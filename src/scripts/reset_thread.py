"""
Reset one WhatsApp number's memory for a clean demo re-run.

The agent has three memories (see src/graph/checkpointer.py):
  * EPISODIC  — LangGraph checkpoints in Postgres, keyed by thread_id. This is what makes a
                re-run "remember" the previous chat (and, crucially, keeps dedup flags like
                payment_link_sent / lead_sent set). Cleared here by default.
  * SEMANTIC  — mem0 facts, keyed by user_id (== the number). Cleared best-effort when
                MEM0_ENABLED (the mem0 store may not be running in every environment).
  * RECORDS   — PaymentOrder / CRMLead audit rows. These are an intentional audit trail, so
                they are LEFT ALONE unless you pass --purge-records.

thread_id is the WhatsApp number exactly as Meta sends it (digits, no '+'), e.g. 919812345678.

Usage
-----
    python -m src.scripts.reset_thread 919812345678
    python -m src.scripts.reset_thread 919812345678 --purge-records   # also drop PaymentOrder/CRMLead
"""
import argparse
import asyncio
import logging

from sqlalchemy import text, delete

from config.settings import settings
from src.core.database import async_session_maker, engine
from src.storage.models import PaymentOrder, CRMLead

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# LangGraph's Postgres checkpointer tables. All carry a thread_id column. Deleted child-first
# so nothing dangles; each is skipped if absent (e.g. a MemorySaver-only environment).
_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


async def _clear_episodic(session, thread_id: str) -> None:
    for table in _CHECKPOINT_TABLES:
        # table names come from the fixed tuple above (never user input); thread_id is bound.
        exists = (await session.execute(
            text("SELECT to_regclass(:qualified)"), {"qualified": f"public.{table}"}
        )).scalar()
        if not exists:
            logger.info(f"Checkpoint table '{table}' not present; skipping.")
            continue
        result = await session.execute(
            text(f"DELETE FROM {table} WHERE thread_id = :tid"), {"tid": thread_id}
        )
        logger.info(f"[{thread_id}] Cleared {result.rowcount or 0} row(s) from {table}.")
    await session.commit()


async def _clear_records(session, thread_id: str) -> None:
    orders = await session.execute(delete(PaymentOrder).where(PaymentOrder.thread_id == thread_id))
    leads = await session.execute(delete(CRMLead).where(CRMLead.whatsapp_id == thread_id))
    await session.commit()
    logger.info(
        f"[{thread_id}] Purged {orders.rowcount or 0} PaymentOrder + {leads.rowcount or 0} CRMLead row(s)."
    )


async def _clear_semantic(thread_id: str) -> None:
    if not settings.MEM0_ENABLED:
        logger.info("MEM0_ENABLED is false; skipping semantic (mem0) reset.")
        return
    try:
        from src.memory.semantic import SemanticMemory
        mem = SemanticMemory()
        if not mem.enabled:
            logger.warning("mem0 unavailable (see log above); skipping semantic reset.")
            return
        await mem.reset(user_id=thread_id)
        logger.info(f"[{thread_id}] Semantic (mem0) memory cleared.")
    except Exception as e:
        logger.warning(f"[{thread_id}] Semantic memory reset skipped/failed (non-fatal): {e}")


async def reset_thread(thread_id: str, purge_records: bool = False) -> None:
    async with async_session_maker() as session:
        await _clear_episodic(session, thread_id)
        if purge_records:
            await _clear_records(session, thread_id)
    await _clear_semantic(thread_id)
    await engine.dispose()
    logger.info(f"[{thread_id}] Reset complete — next message starts a fresh conversation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset one WhatsApp number's agent memory.")
    parser.add_argument("thread_id", help="WhatsApp number as Meta sends it, digits only (e.g. 919812345678).")
    parser.add_argument(
        "--purge-records", action="store_true",
        help="Also delete this number's PaymentOrder + CRMLead audit rows (off by default).",
    )
    args = parser.parse_args()
    asyncio.run(reset_thread(args.thread_id, purge_records=args.purge_records))


if __name__ == "__main__":
    main()
