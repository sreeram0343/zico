"""
Public API Request and Response Schema Layer for ZICO Travel Operations.

This module defines the strict, strongly-typed Pydantic schemas serving as the
contract between external HTTP consumers and the internal ZICO multi-agent system:
    - TravelRequest: Validated natural-language user query and optional session ID.
    - TravelResponse: Client-safe synthesized response, session, status, sources, and errors.
    - Source: Normalized, URL-validated citation reference for research findings.
    - APIError: Standardized, machine-readable operational error payload.
    - ResponseStatus: Controlled set of operational response statuses.
    - APIErrorCode: Enumerated machine-readable error codes.

Design Principles:
    - Decoupled API Contract: Internal graph state (TravelState) is strictly isolated
      from public API models to avoid leaking operational metadata or agent internals.
    - Strict Input Sanitization: Rejects empty or whitespace-only inputs, trims whitespace,
      and enforces sensible length limits to protect backend services.
    - Pure Data Schemas: Zero external service dependencies, zero network calls,
      and zero environment or configuration access.
    - Strict Field Governance: Rejects unknown fields via extra='forbid' to prevent
      malformed payloads or accidental parameter injection.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
)

# Sensible bounds for public API inputs
# 4,000 characters accommodates complex travel queries while preventing buffer abuse.
MAX_MESSAGE_LENGTH: int = 4000
MAX_SESSION_ID_LENGTH: int = 128

__all__ = [
    "MAX_MESSAGE_LENGTH",
    "MAX_SESSION_ID_LENGTH",
    "ResponseStatus",
    "APIErrorCode",
    "APIError",
    "Source",
    "TravelRequest",
    "TravelResponse",
]


class ResponseStatus(str, Enum):
    """
    Controlled set of operational statuses for ZICO API responses.

    Attributes:
        SUCCESS: Request was processed completely with an authoritative answer.
        PARTIAL: Request was processed but has limitations, missing data, or empty search findings.
        ERROR: Request could not be processed due to a failure or unresolvable error.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class APIErrorCode(str, Enum):
    """
    Stable, machine-readable error codes for ZICO API errors.

    Attributes:
        INVALID_REQUEST: Malformed or unprocessable input parameters.
        MISSING_INFORMATION: Incomplete parameters required for the requested operation.
        PROVIDER_UNAVAILABLE: External flight or research provider service outage.
        VALIDATION_FAILED: Safety or business rule violation on traveler parameters.
        INTERNAL_ERROR: Unexpected system or execution failure.
    """

    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIError(BaseModel):
    """
    Client-safe structured error description.

    Attributes:
        code: Machine-readable standardized error code.
        message: Human-readable, non-sensitive explanation of the error.
    """

    code: Union[APIErrorCode, str] = Field(
        ...,
        description="Standardized machine-readable error code.",
        examples=["INVALID_REQUEST", "PROVIDER_UNAVAILABLE"],
    )
    message: str = Field(
        ...,
        description="Human-readable, non-sensitive explanation of the error.",
        examples=["Flight provider service temporarily unavailable."],
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class Source(BaseModel):
    """
    Normalized source citation for verified research findings.

    Attributes:
        title: Descriptive title or label for the referenced source.
        url: Syntactically valid HTTP or HTTPS URL.
        score: Optional relevance or confidence score.
    """

    title: str = Field(
        ...,
        description="Descriptive title or label for the referenced source.",
        examples=["German Federal Foreign Office - Visa Regulations"],
    )
    url: HttpUrl = Field(
        ...,
        description="Syntactically valid HTTP or HTTPS URL of the referenced source.",
        examples=["https://www.auswaertiges-amt.de/en/visa-service"],
    )
    score: Optional[float] = Field(
        default=None,
        description="Optional relevance or confidence score for the research finding.",
        examples=[0.95],
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class TravelRequest(BaseModel):
    """
    Public incoming request payload for the ZICO travel operations assistant.

    Attributes:
        message: Natural-language travel inquiry or command from the traveler.
        session_id: Optional conversation session identifier for ongoing turns.
    """

    message: str = Field(
        ...,
        description="Natural-language travel inquiry or instruction from the traveler.",
        examples=["Find flights from Trivandrum to Dubai tomorrow."],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional conversation session identifier for ongoing interaction.",
        examples=["test-session-01"],
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("message", mode="before")
    @classmethod
    def validate_and_sanitize_message(cls, value: Any) -> str:
        """
        Validate and normalize the user request message.

        Enforces non-empty content after trimming and verifies the message
        does not exceed MAX_MESSAGE_LENGTH.
        """
        if not isinstance(value, str):
            raise ValueError("message must be a string.")

        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message cannot be empty or whitespace-only.")

        if len(trimmed) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"message exceeds maximum permitted length of {MAX_MESSAGE_LENGTH} characters."
            )

        return trimmed

    @field_validator("session_id", mode="before")
    @classmethod
    def validate_and_sanitize_session_id(cls, value: Any) -> Optional[str]:
        """
        Validate and normalize the optional session ID.

        If provided, rejects empty/whitespace strings and enforces length constraints.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("session_id must be a string.")

        trimmed = value.strip()
        if not trimmed:
            raise ValueError("session_id cannot be an empty or whitespace-only string.")

        if len(trimmed) > MAX_SESSION_ID_LENGTH:
            raise ValueError(
                f"session_id exceeds maximum permitted length of {MAX_SESSION_ID_LENGTH} characters."
            )

        return trimmed


class TravelResponse(BaseModel):
    """
    Public response payload returned to the traveler.

    Attributes:
        response: Final synthesized natural-language response.
        session_id: Conversation session identifier associated with the request turn.
        status: High-level operational status ('success', 'partial', 'error').
        sources: List of verified research source citations.
        errors: List of standardized operational errors, if any occurred.
    """

    response: str = Field(
        ...,
        description="Final user-facing natural language response generated by ZICO.",
        examples=["Flight EK522 from DXB to TRV is scheduled on time."],
    )
    session_id: str = Field(
        ...,
        description="Conversation session identifier associated with this request.",
        examples=["test-session-01"],
    )
    status: ResponseStatus = Field(
        default=ResponseStatus.SUCCESS,
        description="Controlled operational status of the response.",
        examples=["success", "partial", "error"],
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Verified research citations supporting the response.",
    )
    errors: List[APIError] = Field(
        default_factory=list,
        description="Standardized client-safe operational errors, if any.",
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("response", mode="before")
    @classmethod
    def validate_response_text(cls, value: Any) -> str:
        """Validate that response is a valid string."""
        if not isinstance(value, str):
            raise ValueError("response must be a string.")
        return value

    @field_validator("session_id", mode="before")
    @classmethod
    def validate_response_session_id(cls, value: Any) -> str:
        """Validate that session_id is a non-empty string."""
        if not isinstance(value, str):
            raise ValueError("session_id must be a string.")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("session_id cannot be empty.")
        return trimmed

    @field_validator("errors", mode="before")
    @classmethod
    def coerce_errors(cls, value: Any) -> Any:
        """Coerce raw strings or dictionaries into standardized APIError instances."""
        if not isinstance(value, list):
            return value
        coerced: List[Any] = []
        for item in value:
            if isinstance(item, str):
                coerced.append(
                    APIError(code=APIErrorCode.INTERNAL_ERROR, message=item)
                )
            else:
                coerced.append(item)
        return coerced
