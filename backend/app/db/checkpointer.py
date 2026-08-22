from contextlib import asynccontextmanager
import logging
import os
from typing import Any, AsyncGenerator, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_postgres_dsn() -> str:
    """Extracts a valid PostgreSQL DSN from the application settings DATABASE_URL."""
    url = settings.DATABASE_URL
    # Normalize sqlalchemy driver prefixes to standard postgresql://
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql+psycopg://", "postgresql://", 1)
    return url


# Connection pool singleton
_pool: Optional[AsyncConnectionPool] = None


def get_connection_pool() -> AsyncConnectionPool:
    """Initializes or returns the singleton psycopg AsyncConnectionPool."""
    global _pool
    if _pool is None:
        dsn = _get_postgres_dsn()
        _pool = AsyncConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=10,
            open=False,
            timeout=5.0,
        )
    return _pool


@asynccontextmanager
async def get_postgres_checkpointer() -> AsyncGenerator[BaseCheckpointSaver, None]:
    """
    Yields an AsyncPostgresSaver connected to PostgreSQL.
    Falls back gracefully to MemorySaver if PostgreSQL is offline or during testing.
    """
    if os.getenv("PYTEST_CURRENT_TEST") is not None or settings.APP_ENV == "test":
        yield MemorySaver()
        return

    pool = get_connection_pool()
    try:
        if not pool.opened:
            await pool.open()

        async with pool.connection() as conn:
            saver = AsyncPostgresSaver(conn)
            yield saver
    except Exception as exc:
        logger.warning(f"PostgreSQL checkpointer unavailable, using in-memory fallback: {exc}")
        yield MemorySaver()


async def setup_checkpoint_tables() -> bool:
    """
    Ensures all LangGraph checkpoint tables (checkpoints, checkpoint_blobs,
    checkpoint_writes, checkpoint_migrations) exist in the PostgreSQL database.
    """
    if os.getenv("PYTEST_CURRENT_TEST") is not None or settings.APP_ENV == "test":
        return True

    try:
        pool = get_connection_pool()
        if not pool.opened:
            await pool.open()

        async with pool.connection() as conn:
            saver = AsyncPostgresSaver(conn)
            await saver.setup()
            logger.info("LangGraph PostgreSQL checkpoint tables and migrations verified.")
            return True
    except Exception as exc:
        logger.warning(f"Could not initialize PostgreSQL checkpoint tables: {exc}")
        return False
