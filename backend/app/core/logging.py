"""
Centralized logging infrastructure for the ZICO project.

This module provides a unified, consistent logging system for all ZICO
subsystems, including agents, tools, LangGraph workflows, and FastAPI routes.

Usage:
    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting flight search workflow")
    logger.warning("No direct routes found; exploring alternatives")
    logger.error("Flight provider request timed out")

Architecture:
    - Standard Library: Built purely on Python's standard `logging` library.
    - Centralized Configuration: `configure_logging()` sets formatting and
      root log level based on `app.core.config.settings.LOG_LEVEL`.
    - Idempotency & Safety: Guards against duplicate handler creation across
      repeated configuration calls or multiple module imports.
    - Non-Invasive: Respects the broader Python logging hierarchy without
      disabling third-party loggers (e.g. Uvicorn, FastAPI).
    - Secret Protection: Never logs or prints environment variables, API keys,
      or authentication credentials.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional, TextIO, Union

from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants & Formats
# ---------------------------------------------------------------------------

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
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

    Sets the logging level and attaches a human-readable StreamHandler
    to the root logger. Idempotent: safe to call multiple times without
    attaching duplicate handlers or producing duplicate log output.

    Args:
        level: Optional override for log level. If omitted, uses settings.LOG_LEVEL.
        log_format: Format string for log records.
        date_format: Date/time format string.
        stream: Optional custom output stream. Defaults to dynamic sys.stdout.
    """
    resolved_level = resolve_log_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

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
    else:
        # Create and register a single marked DynamicStreamHandler
        handler = _DynamicStreamHandler(stream=stream)
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
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


# Automatically initialize default logging configuration on module import
configure_logging()
