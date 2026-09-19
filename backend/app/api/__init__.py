"""
ZICO Public API Package.

Exposes the primary API router, chat endpoint, and public Pydantic request/response schemas.
"""

from app.api.routes import chat_endpoint, router
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
