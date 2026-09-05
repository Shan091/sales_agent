from fastapi import Request
import redis.asyncio as redis

async def get_redis(request: Request) -> redis.Redis:
    """Dependency to retrieve the Redis connection pool from app state."""
    return request.app.state.redis
