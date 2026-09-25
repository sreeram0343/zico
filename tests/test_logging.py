"""
Unit and integration tests for ZICO structured logging and request tracing.

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
- Test 10: Request context set, get, clear, and missing behavior
- Test 11: Asynchronous context isolation between concurrent tasks
- Test 12: Request and session IDs appear in log records and formatted output
- Test 13: Sensitive credential redaction (api_key, Authorization, password, access_token, etc.)
- Test 14: HTTP middleware request_started and request_completed lifecycle tracing
- Test 15: Client-supplied X-Request-ID reuse and missing X-Request-ID generation
- Test 16: HTTP middleware request_failed on server error
- Test 17: Session ID and request ID propagation into chat endpoint and workflow state
- Test 18: Workflow lifecycle logging (workflow_started, workflow_completed, workflow_failed)
- Test 19: Agent lifecycle logging (agent_started, agent_completed, agent_failed)
- Test 20: No traceback leakage to API response bodies
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import TravelResponse
from app.core.config import settings
from app.core.logging import (
    _ZICO_HANDLER_MARKER,
    FALLBACK_LOG_LEVEL,
    configure_logging,
    get_logger,
    resolve_log_level,
    sanitize_text,
)
from app.core.request_context import (
    clear_request_context,
    get_request_context,
    get_request_id,
    get_session_id,
    set_request_context,
    set_request_id,
    set_session_id,
)
from app.core.state import TravelState, create_initial_state
from app.graph.nodes import router_node
from app.graph.workflow import run_workflow
from app.main import app


@pytest.fixture(autouse=True)
def reset_logging_state() -> Generator[None, None, None]:
    """Ensure clean logging and request context state before and after each test."""
    clear_request_context()
    root = logging.getLogger()
    original_level = root.level
    yield
    clear_request_context()
    root.setLevel(original_level)
    configure_logging()


@pytest.fixture
def client() -> TestClient:
    """Synchronous test client for the FastAPI application."""
    return TestClient(app)


# ===========================================================================
# 1. Base Logging Tests (Preserved 1 - 9)
# ===========================================================================


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
    monkeypatch.setattr(settings, "LOG_LEVEL", "DEBUG")
    configure_logging()
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG

    monkeypatch.setattr(settings, "LOG_LEVEL", "WARNING")
    configure_logging()
    assert root_logger.level == logging.WARNING

    configure_logging(level="ERROR")
    assert root_logger.level == logging.ERROR


def test_duplicate_handler_protection(capsys: pytest.CaptureFixture):
    """Test 4 — Duplicate-handler protection: multiple calls do not duplicate output."""
    root_logger = logging.getLogger()

    configure_logging()
    configure_logging()
    configure_logging()

    zico_handlers = [h for h in root_logger.handlers if getattr(h, _ZICO_HANDLER_MARKER, False)]
    assert len(zico_handlers) == 1

    logger1 = get_logger("zico.test.duplicate")
    logger2 = get_logger("zico.test.duplicate")
    assert logger1 is logger2

    capsys.readouterr()
    logger1.info("Unique verification message for duplicate test")
    captured = capsys.readouterr()

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
    resolved = resolve_log_level("SUPER_VERBOSE")
    assert resolved == FALLBACK_LOG_LEVEL

    monkeypatch.setattr(settings, "LOG_LEVEL", "NONEXISTENT_LEVEL")
    configure_logging()
    root_logger = logging.getLogger()
    assert root_logger.level == FALLBACK_LOG_LEVEL

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


# ===========================================================================
# 2. Request Context Tests
# ===========================================================================


def test_request_context_basic():
    """Test 10 — Request context: set, get, clear, and missing behavior."""
    assert get_request_id() is None
    assert get_session_id() is None
    assert get_request_context() == {"request_id": None, "session_id": None}

    set_request_context(request_id="req_101", session_id="sess_202")
    assert get_request_id() == "req_101"
    assert get_session_id() == "sess_202"
    assert get_request_context() == {"request_id": "req_101", "session_id": "sess_202"}

    clear_request_context()
    assert get_request_id() is None
    assert get_session_id() is None

    set_request_id("req_standalone")
    assert get_request_id() == "req_standalone"
    assert get_session_id() is None

    set_session_id("sess_standalone")
    assert get_session_id() == "sess_standalone"


@pytest.mark.asyncio
async def test_request_context_async_isolation():
    """Test 11 — Async context isolation: concurrent tasks maintain isolated context."""
    results: Dict[str, Dict[str, Any]] = {}

    async def worker(task_name: str, req_id: str, sess_id: str):
        set_request_context(request_id=req_id, session_id=sess_id)
        # Yield execution to allow interleaved async scheduling
        await asyncio.sleep(0.01)
        results[task_name] = {
            "req": get_request_id(),
            "sess": get_session_id(),
        }

    await asyncio.gather(
        worker("task_1", "req_alpha", "sess_alpha"),
        worker("task_2", "req_beta", "sess_beta"),
        worker("task_3", "req_gamma", "sess_gamma"),
    )

    assert results["task_1"] == {"req": "req_alpha", "sess": "sess_alpha"}
    assert results["task_2"] == {"req": "req_beta", "sess": "sess_beta"}
    assert results["task_3"] == {"req": "req_gamma", "sess": "sess_gamma"}


# ===========================================================================
# 3. Structured Logging & Context Records
# ===========================================================================


def test_structured_log_record_fields(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
):
    """Test 12 — Structured log fields: request_id and session_id appear in records and format."""
    set_request_context(request_id="req_xyz123", session_id="sess_abc789")
    logger = get_logger("zico.structured.test")

    capsys.readouterr()
    with caplog.at_level(logging.INFO):
        logger.info("Executing structured logging check")

    # Record-level verification
    record = [r for r in caplog.records if r.name == "zico.structured.test"][0]
    assert record.request_id == "req_xyz123"
    assert record.session_id == "sess_abc789"
    assert record.levelname == "INFO"
    assert record.name == "zico.structured.test"

    # Formatted output verification
    out = capsys.readouterr().out
    assert "request_id=req_xyz123" in out
    assert "session_id=sess_abc789" in out
    assert "Executing structured logging check" in out


# ===========================================================================
# 4. Sensitive Credential Redaction Tests
# ===========================================================================


@pytest.mark.parametrize(
    "raw_input,expected_redacted",
    [
        ("api_key=secret123", "api_key=[REDACTED]"),
        ("apikey: 'secret123'", "apikey: [REDACTED]"),
        ("Authorization=Bearer secret-token", "Authorization=[REDACTED]"),
        ("Authorization: Bearer secret-token", "Authorization: [REDACTED]"),
        ("Bearer secret-token-without-header", "Bearer [REDACTED]"),
        ("password=secret", "password=[REDACTED]"),
        ('passwd: "super_secret_password"', "passwd: [REDACTED]"),
        ("access_token=abc123", "access_token=[REDACTED]"),
        ("refresh_token=xyz789", "refresh_token=[REDACTED]"),
        ("cookie=session_id=456", "cookie=[REDACTED]"),
        ("Using key sk-proj-1234567890abcdef for OpenAI", "Using key [REDACTED] for OpenAI"),
        ("Normal text with tokenized word", "Normal text with tokenized word"),
    ],
)
def test_sensitive_credential_redaction(raw_input: str, expected_redacted: str):
    """Test 13 — Sensitive credential redaction across key formats."""
    result = sanitize_text(raw_input)
    assert result == expected_redacted


def test_sensitive_data_redacted_in_logging(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
):
    """Test 13b — Sensitive data logged through Logger is scrubbed in records and stream."""
    logger = get_logger("zico.leak.test")
    secret_payload = (
        "api_key=sk-secret-leak-999 Authorization=Bearer leak-token password=secretpass"
    )

    capsys.readouterr()
    with caplog.at_level(logging.INFO):
        logger.info("Checking leakage: %s", secret_payload)

    record = [r for r in caplog.records if r.name == "zico.leak.test"][0]
    assert "sk-secret-leak-999" not in record.message
    assert "leak-token" not in record.message
    assert "secretpass" not in record.message

    out = capsys.readouterr().out
    assert "sk-secret-leak-999" not in out
    assert "leak-token" not in out
    assert "secretpass" not in out
    assert "[REDACTED]" in out


# ===========================================================================
# 5. HTTP Middleware & Request Tracing Tests
# ===========================================================================


def test_request_lifecycle_tracing_success(client: TestClient, caplog: pytest.LogCaptureFixture):
    """Test 14 — HTTP request lifecycle: records request_started and request_completed."""
    with caplog.at_level(logging.INFO):
        response = client.get(
            "/health",
            headers={"X-Request-ID": "req_trace_test_01", "X-Session-ID": "sess_trace_test_01"},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "req_trace_test_01"

    messages = [r.message for r in caplog.records]
    assert any("request_started method=GET path=/health" in m for m in messages)
    assert any(
        "request_completed method=GET path=/health status_code=200 duration_ms=" in m
        for m in messages
    )


def test_request_id_reuse_and_auto_generation(client: TestClient):
    """Test 15 — Request ID: client-provided is reused, missing generates new UUID ID."""
    # 1. Reusing client-provided
    res1 = client.get("/health", headers={"X-Request-ID": "custom-uuid-999"})
    assert res1.headers.get("X-Request-ID") == "custom-uuid-999"

    # 2. Generating new when missing
    res2 = client.get("/health")
    req_id = res2.headers.get("X-Request-ID")
    assert req_id is not None
    assert req_id.startswith("req_")


def test_request_lifecycle_tracing_failure(client: TestClient, caplog: pytest.LogCaptureFixture):
    """Test 16 — HTTP middleware logs request_failed on 500 error."""
    with (
        patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_wf,
        caplog.at_level(logging.ERROR),
    ):
        mock_wf.side_effect = RuntimeError("Simulated internal workflow explosion")
        response = client.post("/api/v1/chat", json={"message": "Crash this request"})

        assert response.status_code == 500
        assert "X-Request-ID" in response.headers

        # Verify request_failed was logged
        messages = [r.message for r in caplog.records]
        assert any("request_failed" in m and "status_code=500" in m for m in messages)


def test_session_id_propagation_to_chat(client: TestClient, caplog: pytest.LogCaptureFixture):
    """Test 17 — Session ID and request ID propagate into chat endpoint and TravelState."""
    mock_final_state: TravelState = create_initial_state(
        user_query="Find flights",
        session_id="sess_client_provided",
        request_id="req_client_provided",
    )
    mock_final_state["final_response"] = "Direct flight AI-967 found."
    mock_final_state["workflow_status"] = "completed"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        with caplog.at_level(logging.INFO):
            response = client.post(
                "/api/v1/chat",
                json={"message": "Find flights", "session_id": "sess_client_provided"},
                headers={"X-Request-ID": "req_client_provided"},
            )

        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == "req_client_provided"

        data = response.json()
        validated = TravelResponse.model_validate(data)
        assert validated.session_id == "sess_client_provided"

        # Verify initial TravelState received the IDs
        called_state = mock_run.call_args[0][0]
        assert called_state["request_id"] == "req_client_provided"
        assert called_state["session_id"] == "sess_client_provided"


# ===========================================================================
# 6. Workflow & Agent Lifecycle Logging Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_workflow_lifecycle_logging(caplog: pytest.LogCaptureFixture):
    """Test 18 — Workflow lifecycle logging: started, completed, and failed events."""
    base_state = create_initial_state(
        user_query="Test query",
        session_id="sess_wf_test",
        request_id="req_wf_test",
    )

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = dict(base_state)

    # 1. Success execution
    with caplog.at_level(logging.INFO):
        await run_workflow(base_state, graph=mock_graph)

    messages = [r.message for r in caplog.records]
    assert any(
        "workflow_started workflow=zico_workflow request_id=req_wf_test session_id=sess_wf_test"
        in m
        for m in messages
    )
    assert any(
        "workflow_completed workflow=zico_workflow request_id=req_wf_test session_id=sess_wf_test duration_ms="
        in m
        for m in messages
    )

    # 2. Failure execution
    mock_failing_graph = AsyncMock()
    mock_failing_graph.ainvoke.side_effect = ValueError("Fatal graph failure")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            await run_workflow(base_state, graph=mock_failing_graph)

    err_messages = [r.message for r in caplog.records]
    assert any(
        "workflow_failed workflow=zico_workflow request_id=req_wf_test session_id=sess_wf_test duration_ms="
        in m
        and "error_type=ValueError" in m
        for m in err_messages
    )


@pytest.mark.asyncio
async def test_agent_lifecycle_logging(caplog: pytest.LogCaptureFixture):
    """Test 19 — Agent lifecycle logging: agent_started, agent_completed, agent_failed."""
    state = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess_agent_test",
        request_id="req_agent_test",
    )
    set_request_context(request_id="req_agent_test", session_id="sess_agent_test")

    # 1. Success execution
    with (
        patch("app.graph.nodes.aroute_request", new_callable=AsyncMock) as m_agent,
        caplog.at_level(logging.INFO),
    ):
        m_agent.return_value = {"intent": "flight"}
        await router_node(state)

    messages = [r.message for r in caplog.records]
    assert any(
        "agent_started agent=router_agent request_id=req_agent_test session_id=sess_agent_test" in m
        for m in messages
    )
    assert any(
        "agent_completed agent=router_agent request_id=req_agent_test session_id=sess_agent_test duration_ms="
        in m
        for m in messages
    )

    # 2. Failure execution
    with (
        patch("app.graph.nodes.aroute_request", new_callable=AsyncMock) as m_fail,
        caplog.at_level(logging.ERROR),
    ):
        m_fail.side_effect = RuntimeError("Agent execution crashed")
        with pytest.raises(RuntimeError):
            await router_node(state)

    err_messages = [r.message for r in caplog.records]
    assert any(
        "agent_failed agent=router_agent request_id=req_agent_test session_id=sess_agent_test duration_ms="
        in m
        and "error_type=RuntimeError" in m
        for m in err_messages
    )


def test_no_traceback_in_client_response(client: TestClient):
    """Test 20 — Client responses do not expose internal python tracebacks."""
    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_wf:
        mock_wf.side_effect = ZeroDivisionError("division by zero internal defect")
        response = client.post("/api/v1/chat", json={"message": "trigger division error"})

        assert response.status_code == 500
        text = response.text
        assert "ZeroDivisionError" not in text
        assert "Traceback" not in text
        assert 'File "' not in text
        assert "line " not in text
