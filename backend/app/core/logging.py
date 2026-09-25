"""
Centralized structured logging infrastructure for the ZICO project.

This module provides a unified, structured logging system for all ZICO
subsystems, including agents, tools, LangGraph workflows, and FastAPI routes.

Features:
    - Standard Library: Built purely on Python's standard `logging` library.
    - Structured Request Tracing: Injects `request_id` and `session_id` into every log record.
    - Automated Secret Sanitization: Actively scrubs credentials, API keys, tokens,
      passwords, and authorization headers from logs and exception messages.
    - Idempotency & Safety: Guards against duplicate handler creation across
      repeated configuration calls, pytest runs, or server reloads.
    - Non-Invasive: Respects the broader Python logging hierarchy without
      disabling third-party loggers (e.g. Uvicorn, FastAPI).
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Optional, TextIO, Union

from app.core.config import settings
from app.core.request_context import get_request_id, get_session_id

# ---------------------------------------------------------------------------
# Constants & Formats
# ---------------------------------------------------------------------------

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "request_id=%(request_id)s session_id=%(session_id)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

VALID_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
}

FALLBACK_LOG_LEVEL = logging.INFO
_ZICO_HANDLER_MARKER = "_is_zico_handler"

# ---------------------------------------------------------------------------
# Secret & Sensitive Data Sanitization
# ---------------------------------------------------------------------------

# Matches key-value or header-like credential definitions (case-insensitive)
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|apikey|authorization|access[_-]?token|refresh[_-]?token|token|password|passwd|secret|cookie)\b"
    r"(\s*[:=]\s*)"
    r"(['\"][^'\"]*['\"]|Bearer\s+[^\s,;]+|[^\s,;]+)"
)

# Matches standalone Bearer authentication tokens
_BEARER_PATTERN = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9_\.\-]+")

# Matches OpenAI and generic vendor API secret prefixes (e.g. sk-...)
_SK_KEY_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]+)\b")


def sanitize_text(text: str) -> str:
    """
    Scrub sensitive credentials, tokens, passwords, and authorization headers.

    Replaces discovered sensitive values with '[REDACTED]'.

    Args:
        text: Raw log message, stack trace, or formatted string.

    Returns:
        Sanitized text safe for operational storage and logging.
    """
    if not isinstance(text, str):
        return text

    # 1. Scrub key-value pairs (e.g. api_key=secret, password: "xyz")
    sanitized = _SENSITIVE_KEY_PATTERN.sub(r"\1\2[REDACTED]", text)

    # 2. Scrub standalone Bearer tokens
    sanitized = _BEARER_PATTERN.sub(r"\1[REDACTED]", sanitized)

    # 3. Scrub standalone vendor secret keys (e.g. sk-...)
    sanitized = _SK_KEY_PATTERN.sub(r"[REDACTED]", sanitized)

    return sanitized


# ---------------------------------------------------------------------------
# Context Filter & Structured Formatter
# ---------------------------------------------------------------------------


class RequestContextFilter(logging.Filter):
    """
    Logging filter that injects active request and session context into LogRecords
    and sanitizes log messages and arguments at source.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Determine request_id: priority to contextvars, then record attr, then "-"
        ctx_req_id = get_request_id()
        if ctx_req_id:
            record.request_id = ctx_req_id
        elif not hasattr(record, "request_id") or record.request_id is None:
            record.request_id = "-"

        # Determine session_id: priority to contextvars, then record attr, then "-"
        ctx_sess_id = get_session_id()
        if ctx_sess_id:
            record.session_id = ctx_sess_id
        elif not hasattr(record, "session_id") or record.session_id is None:
            record.session_id = "-"

        # Sanitize message if it is a string
        if isinstance(record.msg, str):
            record.msg = sanitize_text(record.msg)

        # Sanitize string args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: sanitize_text(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(
                    sanitize_text(a) if isinstance(a, str) else a for a in record.args
                )

        return True


class StructuredFormatter(logging.Formatter):
    """
    Structured formatter that generates uniform, timestamped log lines
    with request/session tracking and applies output-level credential redaction.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Ensure context attributes exist on the record before formatting
        if not hasattr(record, "request_id") or record.request_id is None:
            record.request_id = get_request_id() or "-"
        if not hasattr(record, "session_id") or record.session_id is None:
            record.session_id = get_session_id() or "-"

        formatted = super().format(record)
        return sanitize_text(formatted)


# ---------------------------------------------------------------------------
# Stream Handler & Configuration
# ---------------------------------------------------------------------------


class _DynamicStreamHandler(logging.StreamHandler):
    """
    StreamHandler that dynamically references the active sys.stdout.

    Ensures that stream redirection, pytest capture fixtures (capsys),
    and server environments receive output reliably without stale stream handles.
    """

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        super().__init__()
        self._custom_stream: Optional[TextIO] = stream

    @property
    def stream(self) -> Any:  # type: ignore[override]
        return self._custom_stream if self._custom_stream is not None else sys.stdout

    @stream.setter
    def stream(self, value: Optional[TextIO]) -> None:
        self._custom_stream = value


def resolve_log_level(level: Optional[Union[str, int]] = None) -> int:
    """
    Resolve a log level string or integer into a standard logging level constant.

    Falls back safely to logging.INFO if the provided level is unrecognized
    or invalid, ensuring the logging subsystem never crashes on bad input.

    Args:
        level: Optional log level string (e.g. 'DEBUG') or integer constant.
               If None, retrieves settings.LOG_LEVEL from app.core.config.

    Returns:
        Integer logging level constant (e.g. logging.INFO).
    """
    if level is None:
        level = getattr(settings, "LOG_LEVEL", "INFO")

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        cleaned = level.strip().upper()
        if cleaned in VALID_LOG_LEVELS:
            return VALID_LOG_LEVELS[cleaned]

    return FALLBACK_LOG_LEVEL


def configure_logging(
    level: Optional[Union[str, int]] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    stream: Optional[TextIO] = None,
) -> None:
    """
    Configure the centralized application logging system.

    Sets the logging level and attaches a structured StreamHandler with
    credential redaction and request context filtering to the root logger.
    Idempotent: safe to call multiple times without attaching duplicate handlers.

    Args:
        level: Optional override for log level. If omitted, uses settings.LOG_LEVEL.
        log_format: Format string for log records.
        date_format: Date/time format string.
        stream: Optional custom output stream. Defaults to dynamic sys.stdout.
    """
    resolved_level = resolve_log_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    formatter = StructuredFormatter(fmt=log_format, datefmt=date_format)

    # Check for an existing ZICO handler to prevent duplicate handlers
    zico_handler: Optional[logging.Handler] = None
    for handler in root_logger.handlers:
        if getattr(handler, _ZICO_HANDLER_MARKER, False):
            zico_handler = handler
            break

    if zico_handler is not None:
        # Update existing handler configuration idempotently
        zico_handler.setLevel(resolved_level)
        zico_handler.setFormatter(formatter)
        if isinstance(zico_handler, _DynamicStreamHandler):
            zico_handler.stream = stream

        # Ensure RequestContextFilter is attached
        if not any(isinstance(f, RequestContextFilter) for f in zico_handler.filters):
            zico_handler.addFilter(RequestContextFilter())
    else:
        # Create and register a single marked DynamicStreamHandler
        handler = _DynamicStreamHandler(stream=stream)
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        handler.addFilter(RequestContextFilter())
        setattr(handler, _ZICO_HANDLER_MARKER, True)
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Obtain a configured logger instance for a module.

    Child loggers inherit configuration and formatting from the root logger
    via standard hierarchy propagation. Calling this function never attaches
    duplicate handlers.

    Args:
        name: Logger identifier, typically __name__ of the calling module.

    Returns:
        Standard library Logger instance.
    """
    return logging.getLogger(name)


__all__ = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
    "FALLBACK_LOG_LEVEL",
    "_ZICO_HANDLER_MARKER",
    "resolve_log_level",
    "configure_logging",
    "get_logger",
    "sanitize_text",
    "RequestContextFilter",
    "StructuredFormatter",
]

# Automatically initialize default logging configuration on module import
configure_logging()
