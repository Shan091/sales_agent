# file: src/core/database.py
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from config.settings import settings

logger = logging.getLogger(__name__)

# Create the async engine globally
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True for SQL query logging in development
    future=True,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True, # Prevent stale connections
)

# Create a sessionmaker factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency to inject database sessions into routes.
    """
    async with async_session_maker() as session:
        yield session

async def close_db_engine():
    """
    Gracefully disposes of the connection pool.
    Must be called on app/worker shutdown.
    """
    logger.info("Disposing async database engine...")
    await engine.dispose()
