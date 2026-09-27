"""
Comprehensive Test Suite for ZICO Centralized Exception & Error Handling.

Validates Prompt 23 requirements:
    - ZICO exception hierarchy structure and inheritance.
    - Stable, uppercase, machine-readable error codes.
    - Distinction between internal errors and safe client-facing errors.
    - Zero credential or traceback leakage in exception messages, details, and logs.
    - Extended API error schemas (APIError, ErrorResponse, TravelResponse.error).
    - Centralized FastAPI exception handlers (ZicoError, RequestValidationError, generic Exception).
    - Request ID and Session ID availability in error responses and logging.
    - Backward-compatible behavior with existing successful API operations.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.schemas import (
    APIError,
    APIErrorCode,
    ErrorResponse,
    ResponseStatus,
    TravelResponse,
)
from app.core.exceptions import (
    ZICO_AGENT_ERROR,
    ZICO_CONFIGURATION_ERROR,
    ZICO_EXTERNAL_SERVICE_ERROR,
    ZICO_INTERNAL_ERROR,
    ZICO_PROVIDER_ERROR,
    ZICO_ROUTING_ERROR,
    ZICO_TOOL_ERROR,
    ZICO_VALIDATION_ERROR,
    ZICO_WORKFLOW_ERROR,
    AgentError,
    ConfigurationError,
    ErrorCode,
    ExternalServiceError,
    InternalError,
    ProviderError,
    RoutingError,
    ToolError,
    ValidationError,
    WorkflowError,
    ZicoError,
    sanitize_error_message,
)
from app.core.llm import LLMConfigurationError
from app.core.state import create_initial_state
from app.main import app as main_app
from app.tools.aviationstack import AviationStackConfigError, AviationStackError
from app.tools.flight_tool import FlightToolError, FlightToolValidationError
from app.tools.location import LocationResolutionError
from app.tools.tavily_search import TavilyConfigError, TavilySearchError


@pytest.fixture
def client() -> TestClient:
    """Synchronous test client for the full ZICO application."""
    return TestClient(main_app)


# ===========================================================================
# 1. Exception Hierarchy & Inheritance
# ===========================================================================


def test_zico_exception_hierarchy():
    """Verify that all domain exception classes inherit from ZicoError and Exception."""
    subclasses = [
        ConfigurationError,
        ValidationError,
        RoutingError,
        WorkflowError,
        AgentError,
        ProviderError,
        ToolError,
        ExternalServiceError,
        InternalError,
    ]
    for sub in subclasses:
        assert issubclass(sub, ZicoError), f"{sub.__name__} must be a subclass of ZicoError"
        assert issubclass(sub, Exception), f"{sub.__name__} must be a subclass of Exception"

    # ExternalServiceError is a specialized ProviderError
    assert issubclass(ExternalServiceError, ProviderError)


def test_tool_and_core_exceptions_inherit_from_hierarchy():
    """Verify existing tool and core exceptions integrate with the ZICO hierarchy."""
    assert issubclass(AviationStackError, ExternalServiceError)
    assert issubclass(AviationStackError, ZicoError)
    assert issubclass(AviationStackConfigError, ConfigurationError)

    assert issubclass(TavilySearchError, ExternalServiceError)
    assert issubclass(TavilySearchError, ZicoError)
    assert issubclass(TavilyConfigError, ConfigurationError)

    assert issubclass(LocationResolutionError, ToolError)
    assert issubclass(LocationResolutionError, ZicoError)

    assert issubclass(FlightToolError, ToolError)
    assert issubclass(FlightToolValidationError, ValidationError)

    assert issubclass(LLMConfigurationError, ConfigurationError)


# ===========================================================================
# 2. Stable Machine-Readable Error Codes
# ===========================================================================


def test_error_codes_are_stable_uppercase():
    """Verify error codes are uppercase, string-based, and decoupled from class names."""
    codes = [
        ErrorCode.VALIDATION_ERROR,
        ErrorCode.ROUTING_ERROR,
        ErrorCode.WORKFLOW_ERROR,
        ErrorCode.AGENT_ERROR,
        ErrorCode.PROVIDER_ERROR,
        ErrorCode.TOOL_ERROR,
        ErrorCode.EXTERNAL_SERVICE_ERROR,
        ErrorCode.CONFIGURATION_ERROR,
        ErrorCode.INTERNAL_ERROR,
    ]
    for code in codes:
        assert isinstance(code, str)
        assert code.isupper(), f"Error code {code!r} must be strictly uppercase"
        assert code.startswith("ZICO_"), f"Error code {code!r} must start with 'ZICO_'"

    # Verify canonical constant bindings
    assert ZICO_VALIDATION_ERROR == "ZICO_VALIDATION_ERROR"
    assert ZICO_ROUTING_ERROR == "ZICO_ROUTING_ERROR"
    assert ZICO_WORKFLOW_ERROR == "ZICO_WORKFLOW_ERROR"
    assert ZICO_AGENT_ERROR == "ZICO_AGENT_ERROR"
    assert ZICO_PROVIDER_ERROR == "ZICO_PROVIDER_ERROR"
    assert ZICO_TOOL_ERROR == "ZICO_TOOL_ERROR"
    assert ZICO_EXTERNAL_SERVICE_ERROR == "ZICO_EXTERNAL_SERVICE_ERROR"
    assert ZICO_CONFIGURATION_ERROR == "ZICO_CONFIGURATION_ERROR"
    assert ZICO_INTERNAL_ERROR == "ZICO_INTERNAL_ERROR"


def test_default_status_and_code_mappings():
    """Verify standard HTTP status and code defaults for each exception type."""
    assert ValidationError().http_status_code == 400
    assert ValidationError().code == ZICO_VALIDATION_ERROR

    assert RoutingError().http_status_code == 400
    assert RoutingError().code == ZICO_ROUTING_ERROR

    assert ProviderError().http_status_code == 503
    assert ProviderError().code == ZICO_PROVIDER_ERROR

    assert ExternalServiceError().http_status_code == 503
    assert ExternalServiceError().code == ZICO_EXTERNAL_SERVICE_ERROR

    assert ToolError().http_status_code == 502
    assert ToolError().code == ZICO_TOOL_ERROR

    assert WorkflowError().http_status_code == 500
    assert WorkflowError().code == ZICO_WORKFLOW_ERROR

    assert AgentError().http_status_code == 500
    assert AgentError().code == ZICO_AGENT_ERROR

    assert ConfigurationError().http_status_code == 500
    assert ConfigurationError().code == ZICO_CONFIGURATION_ERROR

    assert InternalError().http_status_code == 500
    assert InternalError().code == ZICO_INTERNAL_ERROR


# ===========================================================================
# 3. Secret Redaction & Traceback Scrubbing
# ===========================================================================


@pytest.mark.parametrize(
    "leaked_secret,pattern_name",
    [
        ("sk-proj-1234567890abcdef12345678", "OpenAI key"),
        ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "Bearer token"),
        ("api_key=my_super_secret_token_123", "Query param API key"),
        ("access_key=secret_aviation_key_456", "Query param access key"),
        ("postgresql://user:secretpass@localhost:5432/zico", "Database URL"),
        ("redis://:redispass@localhost:6379/0", "Redis URL"),
    ],
)
def test_sanitize_error_message_redacts_credentials(leaked_secret: str, pattern_name: str):
    """Verify sensitive tokens and connection strings are redacted from error text."""
    raw = f"Fatal transport failure with credential: {leaked_secret}"
    cleaned = sanitize_error_message(raw)
    assert leaked_secret not in cleaned, f"Failed to redact {pattern_name}"
    assert "[REDACTED]" in cleaned


def test_sanitize_error_message_scrubs_tracebacks():
    """Verify raw Python tracebacks are completely scrubbed from error messages."""
    traceback_sample = (
        'Traceback (most recent call last):\n  File "app/tools/flight.py", line 42, in search\n'
        "ZeroDivisionError: division by zero"
    )
    cleaned = sanitize_error_message(traceback_sample)
    assert "ZeroDivisionError" not in cleaned
    assert 'File "' not in cleaned
    assert "line 42" not in cleaned
    assert cleaned == "An internal operational error occurred."


def test_zico_error_details_redaction():
    """Verify ZicoError details dictionary automatically redacts secrets in keys and values."""
    err = ZicoError(
        "Service failure",
        details={
            "api_key": "secret-123",
            "access_token": "token-456",
            "normal_field": "safe_value",
            "nested": {"client_secret": "nested-secret-789", "metric": 42},
        },
    )
    assert err.details["api_key"] == "[REDACTED]"
    assert err.details["access_token"] == "[REDACTED]"
    assert err.details["normal_field"] == "safe_value"
    assert err.details["nested"]["client_secret"] == "[REDACTED]"
    assert err.details["nested"]["metric"] == 42


# ===========================================================================
# 4. API Error Schemas
# ===========================================================================


def test_api_error_schema_supports_request_id_and_details():
    """Verify APIError accepts request_id, details, and validates correctly."""
    err = APIError(
        code=ZICO_PROVIDER_ERROR,
        message="Aviation provider timeout.",
        request_id="req_test_abc123",
        details={"retry_after": 30},
    )
    dumped = err.model_dump(mode="json")
    assert dumped["code"] == ZICO_PROVIDER_ERROR
    assert dumped["message"] == "Aviation provider timeout."
    assert dumped["request_id"] == "req_test_abc123"
    assert dumped["details"] == {"retry_after": 30}


def test_error_response_wrapper():
    """Verify ErrorResponse top-level model complies with { 'error': APIError } structure."""
    err = APIError(
        code=ZICO_PROVIDER_ERROR,
        message="The travel information service is temporarily unavailable.",
        request_id="req_wrap_999",
    )
    wrapped = ErrorResponse(error=err)
    dumped = wrapped.model_dump(mode="json")
    assert "error" in dumped
    assert dumped["error"]["code"] == ZICO_PROVIDER_ERROR
    assert dumped["error"]["request_id"] == "req_wrap_999"


def test_travel_response_with_typed_api_errors():
    """Verify TravelResponse preserves typed APIError objects with request identifiers."""
    api_err = APIError(
        code=APIErrorCode.PROVIDER_UNAVAILABLE,
        message="Flight provider unavailable.",
        request_id="req_single_err",
        details={"provider": "aviationstack"},
    )
    resp = TravelResponse(
        response="Unable to fulfill inquiry.",
        session_id="sess_err_test",
        status=ResponseStatus.ERROR,
        errors=[api_err],
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["status"] == "error"
    assert len(dumped["errors"]) == 1
    assert dumped["errors"][0]["code"] == "PROVIDER_UNAVAILABLE"
    assert dumped["errors"][0]["request_id"] == "req_single_err"
    assert dumped["errors"][0]["details"] == {"provider": "aviationstack"}


# ===========================================================================
# 5. FastAPI Centralized Exception Handlers
# ===========================================================================


def test_fastapi_zico_error_handler():
    """Verify ZicoError raised in endpoint is captured by centralized handler with mapped status."""
    test_app = FastAPI()

    # Import handlers from main
    from app.main import (
        generic_exception_handler,
        request_tracing_middleware,
        zico_exception_handler,
    )

    test_app.middleware("http")(request_tracing_middleware)
    test_app.add_exception_handler(ZicoError, zico_exception_handler)
    test_app.add_exception_handler(Exception, generic_exception_handler)

    @test_app.get("/test-provider-error")
    def prov_err():
        raise ProviderError(
            "External flight provider crashed with sk-secretkey12345",
            safe_message="Travel flight provider is currently unavailable.",
        )

    client = TestClient(test_app)
    response = client.get("/test-provider-error", headers={"X-Request-ID": "req_custom_503"})

    assert response.status_code == 503
    assert response.headers.get("X-Request-ID") == "req_custom_503"

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "ZICO_PROVIDER_ERROR"
    assert "sk-secretkey12345" not in response.text
    assert data["error"]["message"] == "Travel flight provider is currently unavailable."
    assert data["error"]["request_id"] == "req_custom_503"


def test_fastapi_validation_error_handler(client: TestClient):
    """Verify FastAPI RequestValidationError returns 422 with structured, safe error details."""
    response = client.post(
        "/api/v1/chat",
        json={"message": "", "session_id": "s"},
        headers={"X-Request-ID": "req_val_422"},
    )
    assert response.status_code == 422
    assert response.headers.get("X-Request-ID") == "req_val_422"

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "ZICO_VALIDATION_ERROR"
    assert data["error"]["request_id"] == "req_val_422"
    assert isinstance(data["error"]["details"], list)
    assert len(data["error"]["details"]) > 0
    assert any(d["field"] == "message" for d in data["error"]["details"])


def test_fastapi_unhandled_exception_handler():
    """Verify unexpected internal exception returns 500 without leaking stack or traceback."""
    test_app = FastAPI()
    from app.main import (
        generic_exception_handler,
        request_tracing_middleware,
        zico_exception_handler,
    )

    test_app.middleware("http")(request_tracing_middleware)
    test_app.add_exception_handler(ZicoError, zico_exception_handler)
    test_app.add_exception_handler(Exception, generic_exception_handler)

    @test_app.get("/test-unhandled-crash")
    def crash():
        raise KeyError("sensitive_internal_dict_key")

    client = TestClient(test_app)
    response = client.get("/test-unhandled-crash", headers={"X-Request-ID": "req_unhandled_500"})

    assert response.status_code == 500
    assert response.headers.get("X-Request-ID") == "req_unhandled_500"

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "ZICO_INTERNAL_ERROR"
    assert data["error"]["request_id"] == "req_unhandled_500"
    assert "sensitive_internal_dict_key" not in response.text
    assert "KeyError" not in response.text
    assert "Traceback" not in response.text


# ===========================================================================
# 6. Chat Endpoint Integration with Typed Exceptions
# ===========================================================================


def test_chat_endpoint_provider_exception(client: TestClient):
    """Verify chat endpoint converts ProviderError into HTTP 503 with typed error payload."""
    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = ProviderError("AviationStack gateway connection reset")

        response = client.post(
            "/api/v1/chat",
            json={"message": "Flight status EK522"},
            headers={"X-Request-ID": "req_prov_int"},
        )

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "temporarily unavailable" in data["response"]
        assert data["errors"][0]["code"] == "PROVIDER_UNAVAILABLE"
        assert data["errors"][0]["request_id"] == "req_prov_int"


def test_chat_endpoint_routing_exception(client: TestClient):
    """Verify chat endpoint converts RoutingError into HTTP 400 with typed error payload."""
    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = RoutingError("Could not resolve inquiry intent")

        response = client.post(
            "/api/v1/chat",
            json={"message": "Indecipherable query"},
            headers={"X-Request-ID": "req_route_400"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert data["errors"][0]["code"] == "INVALID_REQUEST"
        assert data["errors"][0]["request_id"] == "req_route_400"


def test_chat_endpoint_state_errors_with_request_id(client: TestClient):
    """Verify operational errors in TravelState are tagged with the active request ID."""
    mock_state = create_initial_state("Flights to London", "sess_req_test", "req_flow_01")
    mock_state["final_response"] = "Partial answer with flight warnings."
    mock_state["flight_status"] = "error"
    mock_state["errors"] = ["Partner gateway timeout for Heathrow"]

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_state

        response = client.post(
            "/api/v1/chat",
            json={"message": "Flights to London"},
            headers={"X-Request-ID": "req_flow_01"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["errors"]) == 1
        assert data["errors"][0]["code"] == "PROVIDER_UNAVAILABLE"
        assert data["errors"][0]["request_id"] == "req_flow_01"
