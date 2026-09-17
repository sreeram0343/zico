"""
ZICO - AI-Powered Travel Operations Assistant.

FastAPI Application Entry Point.
Provides the ASGI application instance, exposes the liveness health endpoint,
and mounts the travel operations API router.
"""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, status

from app.api.routes import router as api_router
from app.core.logging import configure_logging, get_logger

# Ensure centralized logging is configured for ASGI servers (e.g., uvicorn app.main:app)
configure_logging()
logger = get_logger(__name__)

# Primary FastAPI application instance
app = FastAPI(
    title="ZICO",
    description="AI-powered Travel Operations Assistant",
    version="0.1.0",
)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Application Liveness Health Check",
    tags=["Health"],
)
async def health_check() -> Dict[str, str]:
    """
    Liveness health check endpoint.

    Returns a simple status indicating that the FastAPI application process is alive.
    Performs zero external API, LLM, or database calls.
    """
    return {"status": "ok"}


# Mount the travel operations router (already defines prefix="/api/v1")
app.include_router(api_router)

__all__ = [
    "app",
    "health_check",
]
