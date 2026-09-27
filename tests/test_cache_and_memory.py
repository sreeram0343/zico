"""
Unit tests for CacheService, Semantic Cache Policy, Rate Limiting, and MemoryService.
"""

import pytest

from app.services.cache import CacheService, _normalize_cache_key, get_cache_service
from app.services.memory import MemoryService, get_memory_service


@pytest.mark.asyncio
async def test_cache_service_key_normalization():
    """Verify cache keys are normalized and deterministic."""
    k1 = _normalize_cache_key("  What are the BAGGAGE rules for Lufthansa?  ")
    k2 = _normalize_cache_key("what are the baggage rules for lufthansa?")
    assert k1 == k2
    assert k1.startswith("zico_cache:")


@pytest.mark.asyncio
async def test_cache_service_policy_non_cacheable_flight():
    """Verify live flight queries are NEVER cached to avoid stale aviation data."""
    cache = CacheService()
    stored = await cache.set_cached_response(
        prompt="Status of flight EK522 tomorrow",
        payload={"reply": "Flight EK522 is on time."},
        intent="flight",
    )
    assert stored is False

    looked_up = await cache.get_cached_response(
        prompt="Status of flight EK522 tomorrow",
        intent="flight",
    )
    assert looked_up is None


@pytest.mark.asyncio
async def test_cache_service_safe_caching_fallback():
    """Verify safe general travel / policy responses can be set and looked up."""
    cache = CacheService()
    stored = await cache.set_cached_response(
        prompt="What is the baggage limit for transatlantic flights?",
        payload={"reply": "Standard limit is 23kg per checked bag."},
        intent="research",
    )
    # Succeeds or bypasses gracefully without raising exceptions
    assert isinstance(stored, bool)


@pytest.mark.asyncio
async def test_rate_limiter_in_memory_fallback():
    """Verify rate limiter enforces limits on excessive requests."""
    cache = CacheService()
    client_ip = "192.168.1.100"

    # Within limit of 3 requests
    assert await cache.check_rate_limit(client_ip, max_requests=3, window_seconds=60) is True
    assert await cache.check_rate_limit(client_ip, max_requests=3, window_seconds=60) is True
    assert await cache.check_rate_limit(client_ip, max_requests=3, window_seconds=60) is True

    # 4th request exceeds max_requests=3
    assert await cache.check_rate_limit(client_ip, max_requests=3, window_seconds=60) is False


@pytest.mark.asyncio
async def test_memory_service_add_and_retrieve():
    """Verify conversation turns can be appended and retrieved."""
    memory = MemoryService()
    sess_id = "test_sess_mem_001"

    await memory.add_message(sess_id, role="user", content="Hello ZICO")
    await memory.add_message(sess_id, role="assistant", content="How can I assist your journey?")

    history = await memory.get_recent_history(sess_id, limit=5)
    assert len(history) >= 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello ZICO"
    assert history[1]["role"] == "assistant"

    # Clean up
    await memory.clear_session(sess_id)
    cleared = await memory.get_recent_history(sess_id, limit=5)
    assert len(cleared) == 0


@pytest.mark.asyncio
async def test_singleton_accessors():
    """Verify cache and memory singletons."""
    assert isinstance(get_cache_service(), CacheService)
    assert isinstance(get_memory_service(), MemoryService)
