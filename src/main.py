import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import redis.asyncio as redis

from config.settings import settings
from src.api.router import api_router
from src.core.database import close_db_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Refuse to boot a production deploy with placeholder secrets (forgeable HMAC, etc.).
    settings.assert_production_secrets()

    # Initialize connection pools
    logger.info("Initializing Redis pool...")
    app.state.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Start the TaskIQ broker on the API (kicker) side so webhook handlers can enqueue
    # deep-path work with .kiq(). Guarded by is_worker_process so the worker process
    # (which starts the broker itself) is unaffected.
    from src.tasks.broker import broker
    if not broker.is_worker_process:
        await broker.startup()

    yield

    # Cleanup connection pools
    logger.info("Closing Redis pool...")
    if not broker.is_worker_process:
        await broker.shutdown()
    await app.state.redis.aclose()

    await close_db_engine()

app = FastAPI(title="Otohom Sales Agent", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
