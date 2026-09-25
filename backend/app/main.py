"""
ZICO - AI-Powered Travel Operations Assistant.

FastAPI Application Entry Point.
Provides the ASGI application instance, exposes the liveness health endpoint,
mounts the travel operations API router, and enforces structured request tracing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request, Response, status

from app.api.routes import router as api_router
from app.core.logging import configure_logging, get_logger
from app.core.request_context import (
    clear_request_context,
    set_request_context,
)

# Ensure centralized logging is configured for ASGI servers (e.g., uvicorn app.main:app)
configure_logging()
logger = get_logger(__name__)

# Primary FastAPI application instance
app = FastAPI(
    title="ZICO",
    description="AI-powered Travel Operations Assistant",
    version="0.1.0",
)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next: Any) -> Response:
    """
    HTTP middleware for end-to-end request tracing and structured lifecycle logging.

    Extracts or generates request/session identifiers, establishes async request context,
    measures request execution duration, and attaches the canonical X-Request-ID header.
    """
    # 1. Establish deterministic request ID
    raw_req_id = request.headers.get("X-Request-ID")
    if raw_req_id and raw_req_id.strip():
        req_id = raw_req_id.strip()
    else:
        req_id = f"req_{uuid.uuid4().hex[:12]}"

    # 2. Extract optional session ID from headers
    raw_sess_id = request.headers.get("X-Session-ID")
    sess_id = raw_sess_id.strip() if raw_sess_id and raw_sess_id.strip() else None

    # 3. Bind context variables to active async task
    set_request_context(request_id=req_id, session_id=sess_id)

    # 4. Log request start event
    start_time = time.perf_counter()
    logger.info(
        "request_started method=%s path=%s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if response.status_code >= 500:
            error_type = (
                "ServiceUnavailable" if response.status_code == 503 else "InternalServerError"
            )
            logger.error(
                "request_failed method=%s path=%s status_code=%d duration_ms=%.2f error_type=%s",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                error_type,
            )
        else:
            logger.info(
                "request_completed method=%s path=%s status_code=%d duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )

        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        error_type = type(exc).__name__
        logger.error(
            "request_failed method=%s path=%s status_code=500 duration_ms=%.2f error_type=%s",
            request.method,
            request.url.path,
            duration_ms,
            error_type,
        )
        raise
    finally:
        clear_request_context()


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
    "request_tracing_middleware",
]
