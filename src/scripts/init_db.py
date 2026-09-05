"""
Idempotent database initialization.

Creates the pgvector extension, then all SQLModel tables plus their declared HNSW /
GIN indexes (from src/storage/models.py). Safe to run repeatedly — the extension uses
IF NOT EXISTS and create_all is checkfirst.

Usage:
    python -m src.scripts.init_db
"""
import asyncio
import logging

from sqlalchemy import text
from sqlmodel import SQLModel

from src.core.database import engine
import src.storage.models  # noqa: F401 — registers tables on SQLModel.metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


async def init_db() -> None:
    async with engine.begin() as conn:
        # The vector column type + HNSW index require the extension to exist first.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)
        await _align_audit_timestamps(conn)
    await engine.dispose()
    logger.info("DB initialized: pgvector extension + tables + HNSW/GIN indexes.")


async def _align_audit_timestamps(conn) -> None:
    """
    Bring the audit tables' timestamps to `timestamptz` on databases created before the
    models declared it.

    `create_all` is checkfirst — it creates missing tables but never alters an existing one, and
    there is no Alembic here. A table left with `timestamp without time zone` rejects every
    insert (the models supply timezone-aware values), and because the audit write is fail-soft
    that shows up as rows silently never appearing. Idempotent: re-running is a no-op once the
    types already match.
    """
    for table in ("payment_orders", "crm_leads"):
        for column in ("created_at", "updated_at"):
            await conn.execute(text(f"""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{table}'
                          AND column_name = '{column}'
                          AND data_type = 'timestamp without time zone'
                    ) THEN
                        ALTER TABLE {table}
                            ALTER COLUMN {column} TYPE timestamptz
                            USING {column} AT TIME ZONE 'UTC';
                    END IF;
                END $$;
            """))
    logger.info("Audit timestamps verified as timestamptz.")


if __name__ == "__main__":
    asyncio.run(init_db())
