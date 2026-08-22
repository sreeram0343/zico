import pytest
from app.db.checkpointer import _get_postgres_dsn, get_postgres_checkpointer, setup_checkpoint_tables


def test_get_postgres_dsn_normalization():
    """Verify conversion of SQLAlchemy asyncpg connection strings to standard DSN."""
    dsn = _get_postgres_dsn()
    assert not dsn.startswith("postgresql+asyncpg://")
    assert dsn.startswith("postgresql://") or "user:pass" in dsn


@pytest.mark.asyncio
async def test_get_postgres_checkpointer_context_manager():
    """Verify checkpointer context manager yields a valid LangGraph checkpointer."""
    async with get_postgres_checkpointer() as checkpointer:
        assert checkpointer is not None
        # Verify checkpoint methods exist
        assert hasattr(checkpointer, "get") or hasattr(checkpointer, "aget")
        assert hasattr(checkpointer, "put") or hasattr(checkpointer, "aput")


@pytest.mark.asyncio
async def test_setup_checkpoint_tables():
    """Verify setup_checkpoint_tables completes without exceptions."""
    result = await setup_checkpoint_tables()
    assert result is True
