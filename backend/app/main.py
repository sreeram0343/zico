"""
ZICO - AI-Powered Travel Operations Assistant.

FastAPI Application Entry Point.
Provides the ASGI application instance, exposes the liveness health endpoint,
mounts the travel operations API router, establishes centralized exception handlers,
and enforces structured request tracing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.api.stream import router as stream_router
from app.api.v1.endpoints.actions import router as actions_router
from app.api.v1.endpoints.flights import router as flights_router
from app.api.v1.endpoints.health import router as v1_health_router
from app.api.v1.endpoints.rag import router as rag_router
from app.api.v1.endpoints.trips import router as trips_router
from app.api.v1.endpoints.voice import router as voice_router
from app.core.exceptions import (
    ZICO_INTERNAL_ERROR,
    ZICO_VALIDATION_ERROR,
    ZicoError,
    sanitize_error_message,
)
from app.core.logging import configure_logging, get_logger
from app.core.request_context import (
    clear_request_context,
    get_request_id,
    get_session_id,
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


@app.exception_handler(ZicoError)
async def zico_exception_handler(request: Request, exc: ZicoError) -> JSONResponse:
    """
    Centralized handler for all domain-specific ZicoError exceptions.

    Maps domain exceptions to their registered HTTP status codes and machine-readable
    error codes while ensuring clients never receive unsanitized internal state.
    """
    req_id = (
        get_request_id() or request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    )
    sess_id = get_session_id() or request.headers.get("X-Session-ID") or "-"

    logger.error(
        "zico_error code=%s status_code=%d request_id=%s session_id=%s message=%s",
        exc.code,
        exc.http_status_code,
        req_id,
        sess_id,
        exc.message,
    )

    error_payload: Dict[str, Any] = {
        "code": exc.code,
        "message": exc.get_safe_message(),
        "request_id": req_id,
    }
    if exc.details:
        error_payload["details"] = exc.details

    return JSONResponse(
        status_code=exc.http_status_code,
        content={"error": error_payload},
        headers={"X-Request-ID": req_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Centralized handler for FastAPI and Pydantic input validation failures.

    Translates raw validation errors into sanitized, structured error responses
    without exposing backend file paths, code internals, or raw user secrets.
    """
    req_id = (
        get_request_id() or request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    )
    sess_id = get_session_id() or request.headers.get("X-Session-ID") or "-"

    safe_details: List[Dict[str, Any]] = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if x not in ("body", "__root__")]
        field_name = ".".join(loc) if loc else "body"
        msg = sanitize_error_message(err.get("msg", "Invalid request parameter."))
        err_type = err.get("type", "value_error")
        safe_details.append(
            {
                "field": field_name,
                "message": msg,
                "type": err_type,
            }
        )

    logger.warning(
        "request_validation_failed request_id=%s session_id=%s errors=%s",
        req_id,
        sess_id,
        safe_details,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": ZICO_VALIDATION_ERROR,
                "message": "Invalid request parameters.",
                "request_id": req_id,
                "details": safe_details,
            }
        },
        headers={"X-Request-ID": req_id},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler for unexpected application failures.

    Shields clients from internal Python exceptions, tracebacks, and runtime defects.
    """
    req_id = (
        get_request_id() or request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
    )
    sess_id = get_session_id() or request.headers.get("X-Session-ID") or "-"
    exc_type = type(exc).__name__

    logger.error(
        "unhandled_exception type=%s request_id=%s session_id=%s",
        exc_type,
        req_id,
        sess_id,
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": ZICO_INTERNAL_ERROR,
                "message": "An unexpected error occurred while processing the travel request.",
                "request_id": req_id,
            }
        },
        headers={"X-Request-ID": req_id},
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

    # Handle CORS Preflight OPTIONS requests
    if request.method == "OPTIONS":
        return Response(
            status_code=status.HTTP_204_NO_CONTENT,
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin") or "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "X-Request-ID": req_id,
            },
        )

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Attach CORS headers to response
        client_origin = request.headers.get("origin")
        if client_origin:
            response.headers["Access-Control-Allow-Origin"] = client_origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"

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
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": ZICO_INTERNAL_ERROR,
                    "message": "An unexpected error occurred while processing the travel request.",
                    "request_id": req_id,
                }
            },
            headers={"X-Request-ID": req_id},
        )
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

# Mount WebSocket real-time event streaming router (/ws/stream/{trip_id})
app.include_router(stream_router)

# Mount supplementary v1 domain routers
app.include_router(v1_health_router, prefix="/api/v1/health", tags=["Component Health"])
app.include_router(trips_router, prefix="/api/v1/trips", tags=["Trips"])
app.include_router(actions_router, prefix="/api/v1/actions", tags=["HITL Actions"])
app.include_router(flights_router, prefix="/api/v1/flights", tags=["Flights"])
app.include_router(rag_router, prefix="/api/v1/rag", tags=["Policy RAG"])
app.include_router(voice_router, prefix="/api/v1/voice", tags=["Voice"])

__all__ = [
    "app",
    "health_check",
    "request_tracing_middleware",
    "zico_exception_handler",
    "validation_exception_handler",
    "generic_exception_handler",
]
