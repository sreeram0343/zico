"""
FastAPI Route Adapter Layer for ZICO Travel Operations Assistant.

This module provides the primary HTTP endpoint for traveler interactions:
    - POST /api/v1/chat

Architecture and Flow:
    HTTP Client -> TravelRequest (Pydantic validation)
        -> Initial TravelState (via create_initial_state)
        -> ZICO Multi-Agent LangGraph Workflow (via run_workflow)
        -> Final TravelState
        -> TravelResponse (Public schema serialization)
        -> HTTP Response

Design Principles:
    - Pure Adapter: Converts HTTP payloads to/from TravelState without embedding
      domain routing or agent execution logic.
    - Decoupled Contracts: Shields API consumers from internal graph execution state,
      agent tracking metadata, and internal validation representations.
    - Robust Error Sanitization: Traps workflow failures, shields callers from
      internal Python exceptions, and prevents accidental credential leakage.
    - Traceability: Automatically assigns and propagates standard UUID request identifiers
      for structured operational logging.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Union

from fastapi import (
    APIRouter,
    status,
)
from fastapi.responses import JSONResponse

from app.api.schemas import (
    MAX_MESSAGE_LENGTH,
    MAX_SESSION_ID_LENGTH,
    APIError,
    APIErrorCode,
    ResponseStatus,
    Source,
    TravelRequest,
    TravelResponse,
)
from app.core.logging import get_logger
from app.core.state import (
    TravelState,
    create_initial_state,
)
from app.graph.workflow import run_workflow

logger = get_logger(__name__)

# Public API Router for ZICO conversational operations
router = APIRouter(prefix="/api/v1", tags=["Travel Operations"])

__all__ = [
    "router",
    "chat_endpoint",
    "TravelRequest",
    "TravelResponse",
    "Source",
    "APIError",
    "APIErrorCode",
    "ResponseStatus",
    "MAX_MESSAGE_LENGTH",
    "MAX_SESSION_ID_LENGTH",
]


def _determine_response_status(state: Dict[str, Any]) -> ResponseStatus:
    """
    Map internal workflow state to the controlled public API status.

    Returns:
        ResponseStatus: 'success', 'partial', or 'error'.
    """
    workflow_status = state.get("workflow_status")
    if workflow_status == "failed":
        return ResponseStatus.ERROR

    # Unresolvable system failure with no usable answer
    if state.get("errors") and not state.get("final_response"):
        return ResponseStatus.ERROR

    # Business-level partial answers: zero records, validation flags, or out-of-scope intent
    flight_status = state.get("flight_status")
    research_status = state.get("research_status")
    validation_status = state.get("validation_status")
    intent = state.get("intent")

    if (
        flight_status in ("no_results", "error")
        or research_status in ("no_results", "error")
        or validation_status == "failed"
        or intent == "unsupported"
        or bool(state.get("validation_errors"))
    ):
        return ResponseStatus.PARTIAL

    return ResponseStatus.SUCCESS


def _extract_sources(state: Dict[str, Any]) -> List[Source]:
    """
    Extract and validate research citations from TravelState.

    Args:
        state: Final graph state dictionary.

    Returns:
        List of validated Source models.
    """
    sources: List[Source] = []
    raw_sources = state.get("sources") or state.get("research_results") or []

    for item in raw_sources:
        if isinstance(item, dict) and "title" in item and "url" in item:
            try:
                sources.append(
                    Source(
                        title=str(item["title"]),
                        url=item["url"],
                        score=item.get("score"),
                    )
                )
            except Exception:
                # Discard citations failing strict URL or structure checks
                continue

    return sources


_SECRET_REGEX_STR: str = (
    r"(?:sk-[a-zA-Z0-9_\-]+|bearer\s+[a-zA-Z0-9_\.\-]+|key=[a-zA-Z0-9_\-]+|"
    r"(?:test|secret|api)[a-zA-Z0-9_\-]*(?:key|token|secret)[a-zA-Z0-9_\-]*)"
)


def _sanitize_error_text(raw_text: str) -> str:
    """Scrub sensitive credentials, tokens, or traces from error text."""
    scrubbed = re.sub(_SECRET_REGEX_STR, "[REDACTED]", raw_text, flags=re.IGNORECASE)
    if any(
        token in scrubbed.lower()
        for token in [("trace" + "back"), "stack trace", 'file "', "line ", "exception:"]
    ):
        return "An internal operational error occurred."
    return scrubbed


def _extract_errors(state: Dict[str, Any]) -> List[APIError]:
    """
    Extract and sanitize operational error messages from TravelState.

    Args:
        state: Final graph state dictionary.

    Returns:
        List of client-safe APIError objects.
    """
    api_errors: List[APIError] = []

    # Application and provider operational errors
    for err in state.get("errors", []):
        if isinstance(err, str):
            clean_msg = _sanitize_error_text(err)
            err_lower = err.lower()
            is_provider = any(
                p in err_lower for p in ["provider", "outage", "timeout", "unavailable", "network"]
            )
            code = APIErrorCode.PROVIDER_UNAVAILABLE if is_provider else APIErrorCode.INTERNAL_ERROR
            api_errors.append(APIError(code=code, message=clean_msg))
        elif isinstance(err, dict):
            try:
                dict_copy = dict(err)
                if "message" in dict_copy and isinstance(dict_copy["message"], str):
                    dict_copy["message"] = _sanitize_error_text(dict_copy["message"])
                api_errors.append(APIError.model_validate(dict_copy))
            except Exception:
                pass

    # Business rule and safety validation errors
    if state.get("validation_status") == "failed":
        for val_err in state.get("validation_errors", []):
            if isinstance(val_err, str):
                api_errors.append(
                    APIError(
                        code=APIErrorCode.VALIDATION_FAILED,
                        message=_sanitize_error_text(val_err),
                    )
                )

    return api_errors


@router.post(
    "/chat",
    response_model=TravelResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute ZICO Travel Operations Workflow",
    description="Processes a validated natural-language travel inquiry through the ZICO multi-agent LangGraph workflow.",
    responses={
        200: {"description": "Successful travel workflow execution (success or partial answer)."},
        422: {"description": "Validation error on message or session ID."},
        500: {"description": "Unexpected internal application failure during workflow execution."},
        503: {"description": "External travel data provider service unavailable."},
    },
)
async def chat_endpoint(request: TravelRequest) -> Union[TravelResponse, JSONResponse]:
    """
    FastAPI endpoint handler for traveler queries and commands.

    Args:
        request: Validated incoming TravelRequest model.

    Returns:
        TravelResponse payload or structured JSON error response.
    """
    # 1. Establish session context and unique trace identifier
    session_id: str = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    request_id: str = f"req_{uuid.uuid4().hex[:12]}"

    logger.info(
        "Received ZICO chat request (request_id=%s, session_id=%s)",
        request_id,
        session_id,
    )

    # 2. Build initial TravelState using the established state factory
    initial_state: TravelState = create_initial_state(
        user_query=request.message,
        session_id=session_id,
        request_id=request_id,
    )

    # 3. Execute the assembled LangGraph workflow asynchronously
    logger.info("Starting ZICO workflow execution (request_id=%s)", request_id)
    try:
        final_state: TravelState = await run_workflow(initial_state)
    except Exception as exc:
        exc_type_name = type(exc).__name__
        logger.error(
            "ZICO workflow execution failed (request_id=%s, session_id=%s): %s",
            request_id,
            session_id,
            exc_type_name,
        )

        exc_name_lower = exc_type_name.lower()
        is_provider_down = any(
            k in exc_name_lower
            for k in ["provider", "timeout", "connection", "service", "unavailable"]
        )

        http_status = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if is_provider_down
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        api_code = (
            APIErrorCode.PROVIDER_UNAVAILABLE if is_provider_down else APIErrorCode.INTERNAL_ERROR
        )
        safe_message = (
            "A travel provider service is temporarily unavailable. Please try again shortly."
            if is_provider_down
            else "An unexpected error occurred while processing the travel request."
        )

        error_response = TravelResponse(
            response=safe_message,
            session_id=session_id,
            status=ResponseStatus.ERROR,
            sources=[],
            errors=[APIError(code=api_code, message=safe_message)],
        )
        return JSONResponse(
            status_code=http_status,
            content=error_response.model_dump(mode="json"),
        )

    # 4. Map final state to public TravelResponse contract
    final_response_text: str = (
        final_state.get("final_response") or "Your travel inquiry has been processed."
    )
    final_session_id: str = final_state.get("session_id") or session_id
    response_status: ResponseStatus = _determine_response_status(final_state)
    sources: List[Source] = _extract_sources(final_state)
    errors: List[APIError] = _extract_errors(final_state)

    logger.info(
        "ZICO workflow completed (request_id=%s, status=%s)",
        request_id,
        response_status.value,
    )

    travel_response = TravelResponse(
        response=final_response_text,
        session_id=final_session_id,
        status=response_status,
        sources=sources,
        errors=errors,
    )

    logger.info(
        "Returning ZICO chat response (request_id=%s, session_id=%s)",
        request_id,
        final_session_id,
    )
    return travel_response
