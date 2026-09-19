"""
Unit tests for ZICO centralized logging infrastructure (app.core.logging).

Covers:
- Test 1: Logger creation returns a valid logging.Logger
- Test 2: Logger name is preserved exactly as requested
- Test 3: Logging configuration respects app config settings
- Test 4: Duplicate-handler protection across multiple calls
- Test 5: Log formatting contains timestamp, level, name, message
- Test 6: Different severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Test 7: Invalid log level fails safely with sensible fallback
- Test 8: Logger reuse returns the same underlying logger object
- Test 9: No secret leakage during logging configuration and operation
"""

import logging
from typing import Generator

import pytest

from app.core.config import settings
from app.core.logging import (
    _ZICO_HANDLER_MARKER,
    FALLBACK_LOG_LEVEL,
    configure_logging,
    get_logger,
    resolve_log_level,
)


@pytest.fixture(autouse=True)
def reset_logging_state() -> Generator[None, None, None]:
    """Ensure clean logging state before and after each test."""
    root = logging.getLogger()
    original_level = root.level
    yield
    root.setLevel(original_level)
    # Reconfigure back to original settings level
    configure_logging()


def test_logger_creation():
    """Test 1 — Logger creation: returns a valid logging.Logger instance."""
    logger = get_logger("zico.test")
    assert isinstance(logger, logging.Logger)


def test_logger_name():
    """Test 2 — Logger name: preserves the requested name."""
    name = "zico.test.custom_name"
    logger = get_logger(name)
    assert logger.name == name


def test_logging_configuration(monkeypatch: pytest.MonkeyPatch):
    """Test 3 — Logging configuration: sets intended level from application config."""
    # Test setting DEBUG level
    monkeypatch.setattr(settings, "LOG_LEVEL", "DEBUG")
    configure_logging()
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG

    # Test setting WARNING level
    monkeypatch.setattr(settings, "LOG_LEVEL", "WARNING")
    configure_logging()
    assert root_logger.level == logging.WARNING

    # Test explicit level override in configure_logging
    configure_logging(level="ERROR")
    assert root_logger.level == logging.ERROR


def test_duplicate_handler_protection(capsys: pytest.CaptureFixture):
    """Test 4 — Duplicate-handler protection: multiple calls do not duplicate output."""
    root_logger = logging.getLogger()

    # Call configure_logging multiple times
    configure_logging()
    configure_logging()
    configure_logging()

    # Verify only one ZICO handler exists on root logger
    zico_handlers = [h for h in root_logger.handlers if getattr(h, _ZICO_HANDLER_MARKER, False)]
    assert len(zico_handlers) == 1

    # Call get_logger multiple times
    logger1 = get_logger("zico.test.duplicate")
    logger2 = get_logger("zico.test.duplicate")
    assert logger1 is logger2

    # Emit a message and verify it is printed exactly once to stdout
    capsys.readouterr()  # Clear previous capture
    logger1.info("Unique verification message for duplicate test")
    captured = capsys.readouterr()

    # Should contain message exactly once
    assert captured.out.count("Unique verification message for duplicate test") == 1


def test_log_formatting(capsys: pytest.CaptureFixture):
    """Test 5 — Log formatting: contains level, logger name, and message."""
    configure_logging(level="INFO")
    logger = get_logger("zico.agents.flight")

    capsys.readouterr()
    logger.info("Starting flight lookup for test")
    captured = capsys.readouterr().out

    assert "INFO" in captured
    assert "zico.agents.flight" in captured
    assert "Starting flight lookup for test" in captured
    assert "|" in captured


def test_different_severity_levels(caplog: pytest.LogCaptureFixture):
    """Test 6 — Different severity levels: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    configure_logging(level="DEBUG")
    logger = get_logger("zico.test.severity")

    with caplog.at_level(logging.DEBUG):
        logger.debug("Debug event")
        logger.info("Info event")
        logger.warning("Warning event")
        logger.error("Error event")
        logger.critical("Critical event")

    records = [r for r in caplog.records if r.name == "zico.test.severity"]
    assert len(records) == 5

    levels = [r.levelname for r in records]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    messages = [r.message for r in records]
    assert messages == [
        "Debug event",
        "Info event",
        "Warning event",
        "Error event",
        "Critical event",
    ]


def test_invalid_log_level(monkeypatch: pytest.MonkeyPatch):
    """Test 7 — Invalid log level: system fails safely to fallback level."""
    # Test unrecognized string
    resolved = resolve_log_level("SUPER_VERBOSE")
    assert resolved == FALLBACK_LOG_LEVEL

    # Test bad config setting
    monkeypatch.setattr(settings, "LOG_LEVEL", "NONEXISTENT_LEVEL")
    configure_logging()
    root_logger = logging.getLogger()
    assert root_logger.level == FALLBACK_LOG_LEVEL

    # Test with None or invalid types
    assert resolve_log_level(None) == resolve_log_level(settings.LOG_LEVEL)


def test_logger_reuse():
    """Test 8 — Logger reuse: multiple get_logger calls return the same object."""
    logger_a = get_logger("zico.test.reuse")
    logger_b = get_logger("zico.test.reuse")
    assert logger_a is logger_b
    assert logger_a.name == "zico.test.reuse"


def test_no_secret_leakage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
):
    """Test 9 — No secret leakage: configuration and logging do not expose API credentials."""
    fake_openai = "fake-openai-secret-key-999"
    fake_tavily = "fake-tavily-secret-token-888"
    fake_aviation = "fake-aviationstack-secret-777"

    monkeypatch.setattr(settings, "OPENAI_API_KEY", fake_openai)
    monkeypatch.setattr(settings, "TAVILY_API_KEY", fake_tavily)
    monkeypatch.setattr(settings, "AVIATIONSTACK_API_KEY", fake_aviation)

    capsys.readouterr()
    with caplog.at_level(logging.DEBUG):
        configure_logging()
        logger = get_logger("zico.security.test")
        logger.info("Normal operational log event")

    captured_out = capsys.readouterr().out
    captured_err = capsys.readouterr().err
    log_text = caplog.text

    all_output = f"{captured_out}\n{captured_err}\n{log_text}"
    assert fake_openai not in all_output
    assert fake_tavily not in all_output
    assert fake_aviation not in all_output
