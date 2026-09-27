"""
Redis-backed Caching, Semantic Cache, and Rate Limiting for ZICO.

Provides:
    - Safe caching for deterministic travel knowledge and policy answers.
    - Policy ensuring volatile live flight queries are NEVER cached.
    - Explicit TTL expiration.
    - Sliding window / bucket rate limiting per IP or session.
    - Graceful degradation: all operations fail open / bypass if Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from app.core.config import settings
from app.db.redis import get_redis

logger = logging.getLogger(__name__)

# Categories that are safe to cache (stable travel policies, destination facts)
CACHEABLE_INTENTS = {"research", "general_travel"}
# Non-cacheable categories (volatile real-time flight statuses and tracking)
NON_CACHEABLE_INTENTS = {"flight"}

# In-memory rate limiting fallback when Redis is offline
_IN_MEMORY_RATE_LIMITS: Dict[str, list[float]] = {}


def _normalize_cache_key(prompt: str, prefix: str = "zico_cache") -> str:
    """Produces a deterministic normalized cache key from the prompt text."""
    normalized = " ".join(prompt.strip().lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


class CacheService:
    """Manages Redis caching, semantic caching, and request rate limiting."""

    def __init__(self, default_ttl: int = 3600) -> None:
        self.default_ttl = default_ttl

    async def get_cached_response(
        self,
        prompt: str,
        intent: str = "general_travel",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached response payload if prompt was previously processed and intent is cache-safe.
        Returns None on cache miss, non-cacheable intent, or Redis downtime.
        """
        if intent not in CACHEABLE_INTENTS:
            return None

        key = _normalize_cache_key(prompt)
        try:
            r = await get_redis()
            cached_data = await r.get(key)
            if cached_data:
                logger.info(f"Cache hit for key {key}")
                return json.loads(cached_data)
        except Exception as exc:
            logger.debug(f"Redis cache lookup bypassed gracefully: {exc}")
        return None

    async def set_cached_response(
        self,
        prompt: str,
        payload: Dict[str, Any],
        intent: str = "general_travel",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Stores response in Redis with explicit TTL if intent is cache-safe.
        Never caches live flight responses.
        """
        if intent not in CACHEABLE_INTENTS:
            return False

        key = _normalize_cache_key(prompt)
        expiration = ttl or self.default_ttl
        try:
            r = await get_redis()
            serialized = json.dumps(payload, default=str)
            await r.set(key, serialized, ex=expiration)
            logger.info(f"Cached response for key {key} with TTL {expiration}s")
            return True
        except Exception as exc:
            logger.debug(f"Redis cache store bypassed gracefully: {exc}")
            return False

    async def check_rate_limit(
        self,
        client_identifier: str,
        max_requests: Optional[int] = None,
        window_seconds: int = 60,
    ) -> bool:
        """
        Evaluates whether client_identifier has exceeded request threshold.
        Returns True if request is ALLOWED, False if RATE LIMITED.
        Fails open if Redis is unavailable.
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True

        limit = max_requests or settings.RATE_LIMIT_PER_MINUTE
        rate_key = f"rate_limit:{client_identifier}"
        now = time.time()

        try:
            r = await get_redis()
            # Redis sorted set sliding-window rate limiter
            pipe = r.pipeline()
            pipe.zremrangebyscore(rate_key, 0, now - window_seconds)
            pipe.zadd(rate_key, {f"{now}": now})
            pipe.zcard(rate_key)
            pipe.expire(rate_key, window_seconds + 5)
            results = await pipe.execute()
            current_count = results[2]
            return current_count <= limit
        except Exception as exc:
            logger.debug(f"Redis rate limiter falling back to memory: {exc}")
            # In-memory sliding window fallback
            history = _IN_MEMORY_RATE_LIMITS.setdefault(client_identifier, [])
            cutoff = now - window_seconds
            _IN_MEMORY_RATE_LIMITS[client_identifier] = [t for t in history if t > cutoff]
            if len(_IN_MEMORY_RATE_LIMITS[client_identifier]) < limit:
                _IN_MEMORY_RATE_LIMITS[client_identifier].append(now)
                return True
            return False


_cache_service_instance: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Returns singleton CacheService instance."""
    global _cache_service_instance
    if _cache_service_instance is None:
        _cache_service_instance = CacheService()
    return _cache_service_instance
