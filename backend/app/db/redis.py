from typing import Optional
import redis.asyncio as aioredis
from app.core.config import settings

redis_pool: Optional[aioredis.ConnectionPool] = None
redis_client: Optional[aioredis.Redis] = None


async def init_redis() -> aioredis.Redis:
    """Initializes the global async Redis connection pool and client."""
    global redis_pool, redis_client
    if redis_client is None:
        redis_pool = aioredis.ConnectionPool.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        redis_client = aioredis.Redis(connection_pool=redis_pool)
    return redis_client


async def get_redis() -> aioredis.Redis:
    """Dependency / accessor to get the initialized Redis client instance."""
    if redis_client is None:
        return await init_redis()
    return redis_client


async def close_redis() -> None:
    """Closes the Redis client and connection pool."""
    global redis_pool, redis_client
    if redis_client is not None:
        if hasattr(redis_client, "aclose"):
            await redis_client.aclose()
        else:
            await redis_client.close()
        redis_client = None
    if redis_pool is not None:
        if hasattr(redis_pool, "adisconnect"):
            await redis_pool.adisconnect()
        else:
            await redis_pool.disconnect()
        redis_pool = None

