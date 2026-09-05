"""
Graph checkpointer — episodic memory for the conversation.

This is memory #1 of three (see docs/buildathon/architecture.md):
  * EPISODIC  (here)  — every turn of every chat, keyed by thread_id (the WhatsApp number).
                        Written automatically by LangGraph after each node.
  * SEMANTIC  (mem0)  — durable facts about a person, keyed by user_id (also the number).
  * RECORDS   (SQL)   — PaymentOrder / CRMLead / DocumentChunk business rows.

Two backends:
  * AsyncPostgresSaver — durable. Survives a worker restart mid-conversation, which matters
    now that a `pending_order` (an about-to-be-charged quote) lives in graph state.
  * MemorySaver — process-local fallback used by tests, `local_chat.py`, and any environment
    where psycopg/langgraph-checkpoint-postgres or the DB is unavailable. Losing state on
    restart is acceptable there; silently losing it in production is not, so the fallback
    logs loudly.

Inspecting / resetting memory: see src/scripts/reset_thread.py and the "Memory" section of
the README.
"""
import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings

logger = logging.getLogger(__name__)

# Module-level so close_checkpointer() can dispose it on worker shutdown.
_pool: Optional[Any] = None


def get_checkpointer():
    """
    Synchronous, process-local checkpointer. Used by tests and local_chat.py, and as the
    fallback inside get_checkpointer_async(). NOT durable.
    """
    return MemorySaver()


async def get_checkpointer_async():
    """
    Durable AsyncPostgresSaver backed by its own psycopg connection pool, with the
    `checkpoints` / `checkpoint_writes` / `checkpoint_blobs` tables created on first use.

    Note the separate pool: the app's SQLAlchemy engine speaks asyncpg, but
    langgraph-checkpoint-postgres speaks psycopg3, so the two cannot share connections.
    settings.psycopg_dsn strips the '+asyncpg' driver suffix for us.

    Falls back to MemorySaver — never raises — because a checkpointer problem must not stop
    the worker from answering customers.
    """
    global _pool

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
        from psycopg.rows import dict_row
    except ImportError as e:
        logger.warning(
            f"Postgres checkpointer unavailable ({e}); falling back to in-memory MemorySaver. "
            "Conversation state will NOT survive a restart."
        )
        return get_checkpointer()

    try:
        _pool = AsyncConnectionPool(
            conninfo=settings.psycopg_dsn,
            max_size=10,
            open=False,
            # AsyncPostgresSaver requires autocommit (it manages its own transactions) and
            # dict rows; without these it raises at the first checkpoint write.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await _pool.open(wait=True, timeout=15.0)
        saver = AsyncPostgresSaver(_pool)
        await saver.setup()  # idempotent DDL
        logger.info("Using durable AsyncPostgresSaver checkpointer (episodic memory persisted).")
        return saver
    except Exception as e:
        logger.error(
            f"Failed to open the Postgres checkpointer ({e}); falling back to MemorySaver. "
            "Conversation state will NOT survive a restart."
        )
        if _pool is not None:
            try:
                await _pool.close()
            except Exception:
                pass
            _pool = None
        return get_checkpointer()


async def close_checkpointer() -> None:
    """Dispose the checkpointer pool on worker shutdown (mirrors close_db_engine)."""
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
            logger.info("Checkpointer connection pool closed.")
        except Exception as e:
            logger.warning(f"Error closing checkpointer pool: {e}")
        _pool = None
