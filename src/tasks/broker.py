import os
import redis.asyncio as aioredis
from taskiq_redis import ListQueueBroker
from taskiq import TaskiqEvents

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ListQueueBroker is heavily optimized for high-throughput, async-native task queuing.
# We explicitly disable socket read timeout on the broker connection because BRPOP is a long-polling blocking operation.
broker = ListQueueBroker(
    url=REDIS_URL,
    queue_name="salesforge_wa_queue",
    socket_timeout=None,
    socket_connect_timeout=10,
)


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup_event(state: TaskiqEvents) -> None:
    """
    C5 FIX: Initialize connections inside the worker's event loop, not at module import.
    This prevents the orphaned-event-loop anti-pattern.
    """
    from src.tasks.processing import initialize_worker_services
    from config.settings import settings

    # Refuse to boot a production worker with placeholder secrets.
    settings.assert_production_secrets()

    redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    await initialize_worker_services(redis_pool)

    print("TaskIQ Worker started: Redis pool, WhatsApp service, and LangGraph initialized.")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown_event(state: TaskiqEvents) -> None:
    print("TaskIQ Worker shutting down: closing Redis pool + DB connections...")

    from src.core.database import close_db_engine
    from src.graph.checkpointer import close_checkpointer
    from src.tasks.processing import close_worker_services
    from src.core.tracing import flush_langfuse

    # Push any buffered traces out before the process dies, so the last turns aren't lost.
    flush_langfuse()

    # Close the worker's own Redis pool (created in startup_event) before the DB engine,
    # so repeated worker restarts don't leak Redis connections. The checkpointer keeps a
    # separate psycopg pool of its own, so it needs its own dispose.
    await close_worker_services()
    await close_checkpointer()
    await close_db_engine()