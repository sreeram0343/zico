"""
Centralized Exception Hierarchy and Machine-Readable Error Codes for ZICO.

This module provides a unified, strongly-typed exception system for ZICO travel
operations, establishing:
    - Base ZicoError with safe client-facing messaging, status mapping, and sanitized details.
    - Specialized exception domain subclasses (Configuration, Validation, Routing, Workflow,
      Agent, Provider, Tool, ExternalService, Internal).
    - Stable, machine-readable, uppercase error codes.
    - Strict redaction of secrets, tokens, credentials, and raw tracebacks.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

__all__ = [
    "ErrorCode",
    "ZICO_VALIDATION_ERROR",
    "ZICO_ROUTING_ERROR",
    "ZICO_WORKFLOW_ERROR",
    "ZICO_AGENT_ERROR",
    "ZICO_PROVIDER_ERROR",
    "ZICO_TOOL_ERROR",
    "ZICO_EXTERNAL_SERVICE_ERROR",
    "ZICO_CONFIGURATION_ERROR",
    "ZICO_INTERNAL_ERROR",
    "sanitize_error_message",
    "ZicoError",
    "ConfigurationError",
    "ValidationError",
    "RoutingError",
    "WorkflowError",
    "AgentError",
    "ProviderError",
    "ToolError",
    "ExternalServiceError",
    "InternalError",
]

# ---------------------------------------------------------------------------
# Machine-Readable Error Codes
# ---------------------------------------------------------------------------


class ErrorCode:
    """Stable, uppercase machine-readable error codes for ZICO."""

    VALIDATION_ERROR: str = "ZICO_VALIDATION_ERROR"
    ROUTING_ERROR: str = "ZICO_ROUTING_ERROR"
    WORKFLOW_ERROR: str = "ZICO_WORKFLOW_ERROR"
    AGENT_ERROR: str = "ZICO_AGENT_ERROR"
    PROVIDER_ERROR: str = "ZICO_PROVIDER_ERROR"
    TOOL_ERROR: str = "ZICO_TOOL_ERROR"
    EXTERNAL_SERVICE_ERROR: str = "ZICO_EXTERNAL_SERVICE_ERROR"
    CONFIGURATION_ERROR: str = "ZICO_CONFIGURATION_ERROR"
    INTERNAL_ERROR: str = "ZICO_INTERNAL_ERROR"


# Canonical module-level aliases for direct imports
ZICO_VALIDATION_ERROR: str = ErrorCode.VALIDATION_ERROR
ZICO_ROUTING_ERROR: str = ErrorCode.ROUTING_ERROR
ZICO_WORKFLOW_ERROR: str = ErrorCode.WORKFLOW_ERROR
ZICO_AGENT_ERROR: str = ErrorCode.AGENT_ERROR
ZICO_PROVIDER_ERROR: str = ErrorCode.PROVIDER_ERROR
ZICO_TOOL_ERROR: str = ErrorCode.TOOL_ERROR
ZICO_EXTERNAL_SERVICE_ERROR: str = ErrorCode.EXTERNAL_SERVICE_ERROR
ZICO_CONFIGURATION_ERROR: str = ErrorCode.CONFIGURATION_ERROR
ZICO_INTERNAL_ERROR: str = ErrorCode.INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Secret & Traceback Redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"bearer\s+[a-zA-Z0-9_\.\-]{8,}", re.IGNORECASE),
    re.compile(r"(?:api_?key|access_?key|token|secret|password)=([^\s&]+)", re.IGNORECASE),
    re.compile(r"(?:postgres|postgresql|mysql|sqlite|redis)://[^\s\"']+", re.IGNORECASE),
]

_TRACEBACK_PATTERNS = [
    re.compile(r"traceback\s+\(most recent call last\):", re.IGNORECASE),
    re.compile(r'File\s+"[^"]+",\s+line\s+\d+', re.IGNORECASE),
    re.compile(r"stack trace:", re.IGNORECASE),
]


def sanitize_error_message(text: str) -> str:
    """
    Scrub sensitive credentials, tokens, connection strings, and tracebacks from error text.

    Args:
        text: Raw error text or message.

    Returns:
        Scrubbed, client-safe error message.
    """
    if not isinstance(text, str):
        return str(text)

    scrubbed = text

    # Redact secret patterns
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)

    # Check for traceback signatures
    for pattern in _TRACEBACK_PATTERNS:
        if pattern.search(scrubbed):
            return "An internal operational error occurred."

    return scrubbed


# ---------------------------------------------------------------------------
# Base ZICO Exception
# ---------------------------------------------------------------------------


class ZicoError(Exception):
    """
    Base exception for all ZICO travel operations failures.

    Attributes:
        code: Machine-readable uppercase error code.
        http_status_code: Standard HTTP status code mapped to this error category.
        message: Sanitized human-readable error description.
        safe_message: Safe client-facing message guaranteed to leak zero internal details.
        details: Optional dictionary of sanitized diagnostic context.
    """

    code: str = ErrorCode.INTERNAL_ERROR
    http_status_code: int = 500
    default_safe_message: str = "An unexpected error occurred while processing the travel request."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        http_status_code: Optional[int] = None,
        safe_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_message = message if message is not None else self.default_safe_message
        self.message: str = sanitize_error_message(raw_message)
        super().__init__(self.message)

        if code is not None:
            self.code = code
        if http_status_code is not None:
            self.http_status_code = http_status_code

        self.safe_message: str = sanitize_error_message(safe_message or self.default_safe_message)
        self.details: Dict[str, Any] = self._sanitize_details(details) if details else {}

    @classmethod
    def _sanitize_details(cls, details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Scrub secret keys or token values from error detail dictionaries."""
        if not details:
            return {}
        cleaned: Dict[str, Any] = {}
        for key, value in details.items():
            key_str = str(key).lower()
            if any(
                secret_token in key_str
                for secret_token in ["key", "token", "secret", "auth", "password", "credential"]
            ):
                cleaned[key] = "[REDACTED]"
            elif isinstance(value, str):
                cleaned[key] = sanitize_error_message(value)
            elif isinstance(value, dict):
                cleaned[key] = cls._sanitize_details(value)
            else:
                cleaned[key] = value
        return cleaned

    def get_safe_message(self) -> str:
        """Return a verified safe message for public API responses."""
        return self.safe_message or self.default_safe_message


# ---------------------------------------------------------------------------
# Specialized Exception Domain Classes
# ---------------------------------------------------------------------------


class ConfigurationError(ZicoError):
    """Raised when environment variables, API configurations, or system settings are invalid."""

    code: str = ErrorCode.CONFIGURATION_ERROR
    http_status_code: int = 500
    default_safe_message: str = "A system configuration error occurred."


class ValidationError(ZicoError):
    """Raised when request payload, travel inquiry parameters, or business invariants fail."""

    code: str = ErrorCode.VALIDATION_ERROR
    http_status_code: int = 400
    default_safe_message: str = "Invalid request parameters."


class RoutingError(ZicoError):
    """Raised when the router agent cannot resolve or route the travel inquiry."""

    code: str = ErrorCode.ROUTING_ERROR
    http_status_code: int = 400
    default_safe_message: str = "Unable to determine routing for the travel inquiry."


class WorkflowError(ZicoError):
    """Raised when the LangGraph workflow execution, transitions, or state assembly fail."""

    code: str = ErrorCode.WORKFLOW_ERROR
    http_status_code: int = 500
    default_safe_message: str = "An error occurred during travel workflow execution."


class AgentError(ZicoError):
    """Raised when an individual domain agent encounters a processing failure."""

    code: str = ErrorCode.AGENT_ERROR
    http_status_code: int = 500
    default_safe_message: str = "An error occurred while executing the travel agent."


class ProviderError(ZicoError, RuntimeError):
    """Raised when an external model or service provider (e.g. OpenAI) is unavailable or fails."""

    code: str = ErrorCode.PROVIDER_ERROR
    http_status_code: int = 503
    default_safe_message: str = (
        "The travel provider service is temporarily unavailable. Please try again shortly."
    )


class ToolError(ZicoError):
    """Raised when an operational tool (e.g., flight lookup, location resolver) fails."""

    code: str = ErrorCode.TOOL_ERROR
    http_status_code: int = 502
    default_safe_message: str = "A travel operations tool encountered an error."


class ExternalServiceError(ProviderError):
    """Raised when a 3rd-party external API (AviationStack, Tavily, etc.) encounters an outage."""

    code: str = ErrorCode.EXTERNAL_SERVICE_ERROR
    http_status_code: int = 503
    default_safe_message: str = (
        "An external travel service is temporarily unavailable. Please try again shortly."
    )


class InternalError(ZicoError):
    """Raised for unexpected internal application or runtime errors."""

    code: str = ErrorCode.INTERNAL_ERROR
    http_status_code: int = 500
    default_safe_message: str = "An unexpected error occurred while processing the travel request."
