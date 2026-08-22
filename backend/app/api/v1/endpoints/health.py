import asyncio
from typing import Any, Dict
from fastapi import APIRouter
from sqlalchemy import text
from app.db.redis import get_redis
from app.db.session import engine
from app.rag.service import get_rag_service

router = APIRouter()


@router.get("", response_model=Dict[str, Any])
@router.get("/", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check verifying readiness of API, PostgreSQL, Redis, and Qdrant services.
    """
    db_status = "healthy"
    redis_status = "healthy"
    qdrant_status = "healthy"

    # 1. Database Check with fast timeout
    try:
        async with asyncio.timeout(0.5):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unavailable: {str(exc)}"

    # 2. Redis Check with fast timeout
    try:
        async with asyncio.timeout(0.5):
            r = await get_redis()
            await r.ping()
    except Exception as exc:
        redis_status = f"unavailable: {str(exc)}"

    # 3. Qdrant Check
    try:
        rag = get_rag_service()
        rag.client.get_collections()
    except Exception as exc:
        qdrant_status = f"unavailable: {str(exc)}"

    return {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" and qdrant_status == "healthy" else "degraded",
        "service": "zico-backend",
        "components": {
            "database": db_status,
            "redis": redis_status,
            "qdrant": qdrant_status,
        },
    }

