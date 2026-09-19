"""
Comprehensive test suite for ZICO API Request/Response Schemas.

Validates:
    - Test 34: Valid TravelRequest instantiation and acceptance
    - Test 35: Leading/trailing whitespace trimming on message
    - Test 36: Empty string message rejection
    - Test 37: Whitespace-only message rejection
    - Test 38: Excessive message length rejection
    - Test 39: Valid session ID validation and trimming
    - Test 40: Empty or whitespace-only session ID rejection
    - Test 41: Omitted session ID behavior (no auto-generation)
    - Test 42: Unknown request fields rejection (extra="forbid")
    - Test 43: Valid TravelResponse validation
    - Test 44: Valid 'success' status response
    - Test 45: Valid 'error' status response
    - Test 46: Invalid status rejection
    - Test 47: Source model validation with valid title and URL
    - Test 48: Source model URL validation (rejects malformed URLs without network calls)
    - Test 49: Source model serialization to JSON
    - Test 50: APIError serialization and field validation
    - Test 51: Mutable default isolation between distinct TravelResponse instances
    - Test 52: Secret rejection (extra="forbid" blocks credentials) and zero secret fields
    - Test 53: Internal graph state isolation (blocks TravelState fields from TravelResponse)
    - Test 54: Pydantic JSON schema generation and structural properties
    - Test 55: Full round-trip serialization (Model -> JSON -> Model)
    - Test 56: Public field naming consistency (snake_case)
    - Test 61: Architecture compliance (zero tool, LLM, network, or env imports)
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict

import pytest
from pydantic import ValidationError

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

# ---------------------------------------------------------------------------
# Test 34: Valid TravelRequest
# ---------------------------------------------------------------------------


def test_valid_travel_request() -> None:
    """Verify TravelRequest accepts a standard valid message."""
    query = "Find flights from Trivandrum to Dubai."
    req = TravelRequest(message=query)

    assert req.message == query
    assert req.session_id is None


# ---------------------------------------------------------------------------
# Test 35: Whitespace Trimming
# ---------------------------------------------------------------------------


def test_travel_request_whitespace_trimming() -> None:
    """Verify leading and trailing whitespace is trimmed while preserving internal spaces."""
    raw_message = "   Find flights from TRV to DXB.   "
    req = TravelRequest(message=raw_message)

    assert req.message == "Find flights from TRV to DXB."

    # Internal formatting preserved
    multi_space_message = "  Find   flights  to   London  "
    req_multi = TravelRequest(message=multi_space_message)
    assert req_multi.message == "Find   flights  to   London"


# ---------------------------------------------------------------------------
# Test 36 & 37: Empty and Whitespace-Only Message Rejection
# ---------------------------------------------------------------------------


def test_travel_request_empty_message_rejected() -> None:
    """Verify empty string message raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TravelRequest(message="")
    assert "message" in str(exc_info.value)


@pytest.mark.parametrize("ws_query", ["   ", " ", "\t", "\n", " \t \n "])
def test_travel_request_whitespace_only_message_rejected(ws_query: str) -> None:
    """Verify whitespace-only messages raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TravelRequest(message=ws_query)
    assert "message" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 38: Excessive Message Length Rejection
# ---------------------------------------------------------------------------


def test_travel_request_excessive_message_length() -> None:
    """Verify messages exceeding MAX_MESSAGE_LENGTH are rejected with clear error."""
    oversized_message = "A" * (MAX_MESSAGE_LENGTH + 1)
    with pytest.raises(ValidationError) as exc_info:
        TravelRequest(message=oversized_message)
    assert "maximum permitted length" in str(exc_info.value)

    # Verify boundary value is accepted
    exact_max_message = "A" * MAX_MESSAGE_LENGTH
    req = TravelRequest(message=exact_max_message)
    assert len(req.message) == MAX_MESSAGE_LENGTH


# ---------------------------------------------------------------------------
# Test 39: Valid Session ID
# ---------------------------------------------------------------------------


def test_travel_request_valid_session_id() -> None:
    """Verify a valid session_id is accepted and trimmed."""
    req = TravelRequest(message="Flight status", session_id="test-session-01")
    assert req.session_id == "test-session-01"

    # Trimming
    req_trimmed = TravelRequest(message="Flight status", session_id="  session-with-spaces  ")
    assert req_trimmed.session_id == "session-with-spaces"


# ---------------------------------------------------------------------------
# Test 40: Empty Session ID Rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_session", ["", "   ", "\t", "\n"])
def test_travel_request_empty_session_id_rejected(invalid_session: str) -> None:
    """Verify empty or whitespace-only session_id raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TravelRequest(message="Flight status", session_id=invalid_session)
    assert "session_id" in str(exc_info.value)


def test_travel_request_excessive_session_id_length() -> None:
    """Verify session_id exceeding MAX_SESSION_ID_LENGTH is rejected."""
    oversized_session = "s" * (MAX_SESSION_ID_LENGTH + 1)
    with pytest.raises(ValidationError) as exc_info:
        TravelRequest(message="Flight status", session_id=oversized_session)
    assert "session_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 41: Omitted Session ID
# ---------------------------------------------------------------------------


def test_travel_request_omitted_session_id() -> None:
    """Verify omitted session_id defaults to None without auto-generating a fake ID."""
    req = TravelRequest(message="Find flights")
    assert req.session_id is None

    req_explicit_none = TravelRequest(message="Find flights", session_id=None)
    assert req_explicit_none.session_id is None


# ---------------------------------------------------------------------------
# Test 42: Unknown Request Fields Rejection
# ---------------------------------------------------------------------------


def test_travel_request_unknown_fields_rejected() -> None:
    """Verify unexpected fields trigger validation failure via extra='forbid'."""
    with pytest.raises(ValidationError) as exc_info:
        TravelRequest.model_validate(
            {
                "message": "Find a flight.",
                "unknown_field": "unexpected_payload",
            }
        )
    assert "extra_forbidden" in str(exc_info.value) or "unknown_field" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 43, 44, 45, 46: TravelResponse and Status Validation
# ---------------------------------------------------------------------------


def test_valid_travel_response_partial() -> None:
    """Verify constructing a valid TravelResponse with status='partial'."""
    response_text = "No matching flight information was found."
    session_id = "test-session"
    status = "partial"

    resp = TravelResponse(
        response=response_text,
        session_id=session_id,
        status=status,
        sources=[],
        errors=[],
    )

    assert resp.response == response_text
    assert resp.session_id == session_id
    assert resp.status == ResponseStatus.PARTIAL
    assert resp.status == "partial"
    assert resp.sources == []
    assert resp.errors == []


def test_valid_travel_response_success() -> None:
    """Verify TravelResponse accepts status='success' and defaults empty collections."""
    resp = TravelResponse(
        response="Flight EK522 is on time.",
        session_id="session-123",
        status="success",
    )
    assert resp.status == ResponseStatus.SUCCESS
    assert resp.status == "success"
    assert resp.sources == []
    assert resp.errors == []


def test_valid_travel_response_error() -> None:
    """Verify TravelResponse accepts status='error'."""
    resp = TravelResponse(
        response="An error occurred while processing flight data.",
        session_id="session-err-01",
        status="error",
        errors=[APIError(code=APIErrorCode.PROVIDER_UNAVAILABLE, message="Service timeout")],
    )
    assert resp.status == ResponseStatus.ERROR
    assert resp.status == "error"
    assert len(resp.errors) == 1
    assert resp.errors[0].code == APIErrorCode.PROVIDER_UNAVAILABLE


def test_travel_response_invalid_status_rejected() -> None:
    """Verify unapproved status strings raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        TravelResponse(
            response="Some text",
            session_id="sess-01",
            status="banana",  # type: ignore[arg-type]
        )
    assert "status" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 47 & 48: Source Validation and URL Validation
# ---------------------------------------------------------------------------


def test_source_model_valid() -> None:
    """Verify Source model validates title, HttpUrl, and optional score."""
    source = Source(
        title="German Visa Portal",
        url="https://germany.diplo.de/visa",  # type: ignore[arg-type]
        score=0.95,
    )
    assert source.title == "German Visa Portal"
    assert str(source.url) == "https://germany.diplo.de/visa"
    assert source.score == 0.95


def test_source_model_score_optional() -> None:
    """Verify Source score is optional and defaults to None."""
    source = Source(
        title="Flight Advisory",
        url="https://flightradar24.com/live",  # type: ignore[arg-type]
    )
    assert source.score is None


@pytest.mark.parametrize(
    "invalid_url", ["not-a-url", "ftp://unsupported", "htp:/typo", "http://", ""]
)
def test_source_model_malformed_url_rejected(invalid_url: str) -> None:
    """Verify syntactically invalid URLs raise ValidationError without network calls."""
    with pytest.raises(ValidationError) as exc_info:
        Source(title="Bad Link", url=invalid_url)  # type: ignore[arg-type]
    assert "url" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 49: Source Serialization
# ---------------------------------------------------------------------------


def test_source_serialization() -> None:
    """Verify Source serializes cleanly to dict and JSON without raw provider payloads."""
    source = Source(
        title="Emirates Schedule",
        url="https://www.emirates.com/flights/ek522",  # type: ignore[arg-type]
        score=0.98,
    )

    dump_dict = source.model_dump(mode="json")
    assert dump_dict == {
        "title": "Emirates Schedule",
        "url": "https://www.emirates.com/flights/ek522",
        "score": 0.98,
    }

    json_str = source.model_dump_json()
    assert '"title":"Emirates Schedule"' in json_str
    assert '"url":"https://www.emirates.com/flights/ek522"' in json_str


def test_source_rejects_raw_provider_fields() -> None:
    """Verify extra raw provider fields (e.g. Tavily content) are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Source.model_validate(
            {
                "title": "Search Result",
                "url": "https://example.com",
                "raw_provider_content": "unfiltered text",
            }
        )
    assert "extra_forbidden" in str(exc_info.value) or "raw_provider_content" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 50: APIError Serialization
# ---------------------------------------------------------------------------


def test_api_error_serialization() -> None:
    """Verify APIError validates, serializes code and message individually, and dumps cleanly."""
    err = APIError(
        code=APIErrorCode.INVALID_REQUEST,
        message="Origin airport IATA code could not be resolved.",
    )

    assert err.code == APIErrorCode.INVALID_REQUEST
    assert err.code == "INVALID_REQUEST"
    assert err.message == "Origin airport IATA code could not be resolved."

    dump = err.model_dump(mode="json")
    assert dump["code"] == "INVALID_REQUEST"
    assert dump["message"] == "Origin airport IATA code could not be resolved."

    json_str = err.model_dump_json()
    assert '"code":"INVALID_REQUEST"' in json_str
    assert '"message":"Origin airport IATA code could not be resolved."' in json_str


def test_travel_response_coerces_error_formats() -> None:
    """Verify TravelResponse coerces plain strings and dictionaries into APIError instances."""
    resp = TravelResponse(
        response="Operation completed with warnings.",
        session_id="sess-coerce-01",
        status="partial",
        errors=[
            "Network latency experienced",
            {"code": "PROVIDER_UNAVAILABLE", "message": "Secondary search timed out"},
            APIError(code=APIErrorCode.VALIDATION_FAILED, message="Date mismatch"),
        ],  # type: ignore[arg-type]
    )

    assert len(resp.errors) == 3
    assert resp.errors[0].code == APIErrorCode.INTERNAL_ERROR
    assert resp.errors[0].message == "Network latency experienced"
    assert resp.errors[1].code == APIErrorCode.PROVIDER_UNAVAILABLE
    assert resp.errors[2].code == APIErrorCode.VALIDATION_FAILED


# ---------------------------------------------------------------------------
# Test 51: Mutable Default Isolation
# ---------------------------------------------------------------------------


def test_mutable_default_isolation() -> None:
    """Verify list defaults in separate TravelResponse instances do NOT share state."""
    r1 = TravelResponse(response="Response 1", session_id="s1")
    r2 = TravelResponse(response="Response 2", session_id="s2")

    # Mutate collections of r1
    r1.sources.append(Source(title="Src 1", url="https://example.com/1"))  # type: ignore[arg-type]
    r1.errors.append(APIError(code=APIErrorCode.INTERNAL_ERROR, message="Err 1"))

    # r2 collections must remain completely unaffected
    assert len(r1.sources) == 1
    assert len(r1.errors) == 1
    assert len(r2.sources) == 0
    assert len(r2.errors) == 0
    assert r1.sources is not r2.sources
    assert r1.errors is not r2.errors


# ---------------------------------------------------------------------------
# Test 52: Secret Rejection and Schema Safety
# ---------------------------------------------------------------------------


def test_secret_rejection_in_requests_and_responses() -> None:
    """Verify credentials and secret fields cannot be passed into API models."""
    secret_payload = {"message": "Plan trip", "api_key": "sk-fake-secret-key"}
    with pytest.raises(ValidationError):
        TravelRequest.model_validate(secret_payload)

    response_payload = {
        "response": "Here is your plan",
        "session_id": "s-1",
        "access_token": "bearer-token-secret",
    }
    with pytest.raises(ValidationError):
        TravelResponse.model_validate(response_payload)

    # Models must have zero credential fields defined
    for model_cls in [TravelRequest, TravelResponse, Source, APIError]:
        fields = set(model_cls.model_fields.keys())
        assert "api_key" not in fields
        assert "token" not in fields
        assert "secret" not in fields


# ---------------------------------------------------------------------------
# Test 53: Internal Graph State Isolation
# ---------------------------------------------------------------------------


def test_internal_state_isolation_from_travel_response() -> None:
    """Verify internal TravelState fields are rejected by TravelResponse via extra='forbid'."""
    forbidden_internal_fields = [
        ("active_agent", "flight_agent"),
        ("completed_agents", ["router_agent", "flight_agent"]),
        ("validation_errors", ["bad date"]),
        ("metadata", {"env": "prod"}),
        ("flight_results", [{"flight_iata": "EK522"}]),
        ("research_results", [{"title": "Visa"}]),
        ("workflow_status", "running"),
    ]

    for field_name, sample_val in forbidden_internal_fields:
        payload: Dict[str, Any] = {
            "response": "Final answer",
            "session_id": "sess-iso-01",
            "status": "success",
            field_name: sample_val,
        }
        with pytest.raises(ValidationError, match=field_name):
            TravelResponse.model_validate(payload)


# ---------------------------------------------------------------------------
# Test 54: JSON Schema Generation
# ---------------------------------------------------------------------------


def test_json_schema_generation() -> None:
    """Verify JSON schemas reflect required fields, status constraints, and source definitions."""
    req_schema = TravelRequest.model_json_schema()
    assert req_schema["type"] == "object"
    assert "message" in req_schema["required"]
    assert "properties" in req_schema
    assert "session_id" in req_schema["properties"]

    resp_schema = TravelResponse.model_json_schema()
    assert resp_schema["type"] == "object"
    assert "response" in resp_schema["required"]
    assert "session_id" in resp_schema["required"]

    # Status constraints visible in schema definitions
    defs = resp_schema.get("$defs", {})
    status_def = defs.get("ResponseStatus", {})
    assert set(status_def.get("enum", [])) == {"success", "partial", "error"}

    # Sources definition visible
    assert "sources" in resp_schema["properties"]
    assert "errors" in resp_schema["properties"]


# ---------------------------------------------------------------------------
# Test 55: Round-Trip Serialization
# ---------------------------------------------------------------------------


def test_round_trip_serialization() -> None:
    """Verify models survive full Model -> JSON -> Model round-trip without loss."""
    original_resp = TravelResponse(
        response="Flight EK522 is scheduled.",
        session_id="roundtrip-sess-01",
        status=ResponseStatus.SUCCESS,
        sources=[Source(title="Emirates", url="https://emirates.com/info", score=0.99)],  # type: ignore[arg-type]
        errors=[APIError(code=APIErrorCode.INVALID_REQUEST, message="Informational note")],
    )

    json_str = original_resp.model_dump_json()
    reloaded = TravelResponse.model_validate_json(json_str)

    assert reloaded.response == original_resp.response
    assert reloaded.session_id == original_resp.session_id
    assert reloaded.status == original_resp.status
    assert len(reloaded.sources) == len(original_resp.sources)
    assert reloaded.sources[0].title == original_resp.sources[0].title
    assert str(reloaded.sources[0].url) == str(original_resp.sources[0].url)
    assert reloaded.sources[0].score == original_resp.sources[0].score
    assert len(reloaded.errors) == len(original_resp.errors)
    assert reloaded.errors[0].code == original_resp.errors[0].code
    assert reloaded.errors[0].message == original_resp.errors[0].message


# ---------------------------------------------------------------------------
# Test 56: API Design Consistency (snake_case)
# ---------------------------------------------------------------------------


def test_api_field_naming_snake_case() -> None:
    """Verify all public model fields adhere to strict snake_case naming conventions."""
    snake_case_pattern = re.compile(r"^[a-z][a-z0-9_]*$")

    models_to_check = [TravelRequest, TravelResponse, Source, APIError]
    for model_cls in models_to_check:
        for field_name in model_cls.model_fields.keys():
            assert snake_case_pattern.match(field_name), (
                f"Field {field_name!r} in {model_cls.__name__} does not follow snake_case naming."
            )


# ---------------------------------------------------------------------------
# Test 61: Architecture Compliance (Zero Tool, LLM, or Env Access)
# ---------------------------------------------------------------------------


def test_schema_architecture_compliance() -> None:
    """
    Verify app.api.schemas strictly functions as a data contract and does NOT:
        - import or call LLMs (OpenAI / ChatOpenAI)
        - import or execute external tools (AviationStack, Tavily, requests, httpx)
        - construct StateGraph workflows
        - access environment variables (os.getenv, os.environ)
    """
    import app.api.schemas as schemas_module

    source_code = inspect.getsource(schemas_module)

    prohibited_tokens = [
        "OpenAI",
        "ChatOpenAI",
        "AviationStack",
        "Tavily",
        "httpx",
        "requests",
        "StateGraph",
        "os.getenv",
        "os.environ",
    ]

    for token in prohibited_tokens:
        assert token not in source_code, (
            f"app.api.schemas must not contain prohibited token: {token!r}"
        )
