"""
Unit tests for database models and redis connection helper.
"""

from unittest.mock import AsyncMock, patch
import pytest
from app.db.models import AuditLog, Base, Trip
from app.db.redis import close_redis, get_redis, init_redis


def test_trip_model_instantiation():
    """Verify Trip ORM model instantiation and attributes."""
    trip = Trip(
        id="trip_test_01",
        user_id="user_123",
        state_json={"trip_id": "trip_test_01", "itinerary": []},
    )
    assert trip.id == "trip_test_01"
    assert trip.user_id == "user_123"
    assert trip.state_json["trip_id"] == "trip_test_01"


def test_audit_log_model_instantiation():
    """Verify AuditLog ORM model instantiation."""
    log = AuditLog(
        trip_id="trip_test_01",
        action_type="BOOKING",
        payload={"status": "APPROVED"},
    )
    assert log.trip_id == "trip_test_01"
    assert log.action_type == "BOOKING"
    assert log.payload == {"status": "APPROVED"}


@pytest.mark.asyncio
async def test_redis_initialization():
    """Verify Redis client pool initialization and cleanup."""
    with patch("redis.asyncio.ConnectionPool.from_url") as mock_pool_factory, \
         patch("redis.asyncio.Redis") as mock_redis_cls:

        mock_pool = AsyncMock()
        mock_pool_factory.return_value = mock_pool

        mock_client = AsyncMock()
        mock_redis_cls.return_value = mock_client

        # Reset global state
        await close_redis()

        client = await init_redis()
        assert client is not None
        mock_pool_factory.assert_called_once()

        # get_redis returns same instance
        client2 = await get_redis()
        assert client2 == client

        # Cleanup
        await close_redis()
        assert mock_client.close.await_count == 1 or getattr(mock_client, "aclose").await_count == 1
        assert mock_pool.disconnect.await_count == 1 or getattr(mock_pool, "adisconnect").await_count == 1

