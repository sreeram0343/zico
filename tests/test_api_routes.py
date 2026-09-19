"""
Comprehensive test suite for ZICO FastAPI Chat Route Layer.

Validates:
    - Test 36: Valid chat request execution (POST /api/v1/chat)
    - Test 37: Valid session ID propagation into workflow
    - Test 38: Request message whitespace trimming prior to workflow execution
    - Test 39: Empty string request body rejection (HTTP 422, workflow untouched)
    - Test 40: Whitespace-only request rejection (HTTP 422, workflow untouched)
    - Test 41: Workflow receives complete, valid initial TravelState
    - Test 42: Successful workflow response mapping to TravelResponse
    - Test 43: Business no-results state produces HTTP 200 with status='partial'
    - Test 44: Business validation failure produces HTTP 200 with status='partial' and sanitized errors
    - Test 45: Provider failure returns sanitized error response and HTTP 503
    - Test 46: Unexpected workflow exception returns HTTP 500 with zero exception leakage
    - Test 47: Custom session ID preservation in final API response
    - Test 48: Dynamic unique request ID generation and state propagation
    - Test 49: Exactly-once workflow invocation per HTTP request
    - Test 50: No direct domain agent or external tool execution from route layer
    - Test 51: Strict TravelResponse Pydantic validation of returned payload
    - Test 52: Complete isolation from internal TravelState fields
    - Test 53: Zero secret leakage in response bodies, error messages, or logs
    - Test 54: Application/json response content type verification
    - Test 55: OpenAPI documentation generation for /api/v1/chat
    - Test 56: Absence of /health endpoint on the chat router
    - Test 57: Structured APIError model usage in response errors
    - Test 62 & 63: Architectural compliance (no LLM, tool, StateGraph, or os.environ access)
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.api.schemas import ResponseStatus, TravelResponse
from app.core.state import TravelState, create_initial_state

# ---------------------------------------------------------------------------
# Test Application Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    """Create an isolated FastAPI application mounting only the ZICO chat router."""
    test_app = FastAPI(title="ZICO Test App")
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Provide a synchronous TestClient bound to the isolated test app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 36: Valid Chat Request
# ---------------------------------------------------------------------------


def test_valid_chat_request(client: TestClient) -> None:
    """Verify POST /api/v1/chat executes workflow and returns valid TravelResponse."""
    mock_final_state: TravelState = create_initial_state(
        user_query="Find flights from TRV to DXB.",
        session_id="sess-test-36",
        request_id="req-test-36",
    )
    mock_final_state["final_response"] = "Flight EK522 from TRV to DXB is on time."
    mock_final_state["workflow_status"] = "completed"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:

        async def mock_exec(state: TravelState) -> TravelState:
            enriched = dict(state)
            enriched["final_response"] = "Flight EK522 from TRV to DXB is on time."
            enriched["workflow_status"] = "completed"
            return enriched

        mock_run.side_effect = mock_exec

        response = client.post(
            "/api/v1/chat",
            json={"message": "Find flights from TRV to DXB."},
        )

        assert response.status_code == 200
        data = response.json()

        # Validate against public Pydantic model
        validated = TravelResponse.model_validate(data)
        assert validated.response == "Flight EK522 from TRV to DXB is on time."
        assert validated.status == ResponseStatus.SUCCESS
        assert validated.session_id.startswith("sess_")
        assert validated.sources == []
        assert validated.errors == []

        # Ensure no internal graph fields leaked
        assert "active_agent" not in data
        assert "flight_results" not in data
        assert "validation_errors" not in data


# ---------------------------------------------------------------------------
# Test 37: Valid Session Request
# ---------------------------------------------------------------------------


def test_valid_session_request_propagation(client: TestClient) -> None:
    """Verify provided session_id is forwarded to initial state and preserved in response."""
    custom_session = "my-custom-session-123"

    mock_final_state: TravelState = create_initial_state(
        user_query="Check my flight.",
        session_id=custom_session,
        request_id="req-test-37",
    )
    mock_final_state["final_response"] = "Flight confirmed."
    mock_final_state["session_id"] = custom_session

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Check my flight.",
                "session_id": custom_session,
            },
        )

        assert response.status_code == 200
        passed_state: TravelState = mock_run.call_args[0][0]
        assert passed_state["session_id"] == custom_session
        assert response.json()["session_id"] == custom_session


# ---------------------------------------------------------------------------
# Test 38: Request Trimming
# ---------------------------------------------------------------------------


def test_request_message_trimming(client: TestClient) -> None:
    """Verify message whitespace is trimmed by schema before reaching workflow state."""
    raw_query = "   Find flights to Dubai   "
    expected_query = "Find flights to Dubai"

    mock_final_state: TravelState = create_initial_state(
        user_query=expected_query,
        session_id="s1",
        request_id="r1",
    )
    mock_final_state["final_response"] = "Done."

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post(
            "/api/v1/chat",
            json={"message": raw_query},
        )

        assert response.status_code == 200
        passed_state: TravelState = mock_run.call_args[0][0]
        assert passed_state["user_query"] == expected_query


# ---------------------------------------------------------------------------
# Test 39 & 40: Invalid and Whitespace-Only Request Rejection
# ---------------------------------------------------------------------------


def test_empty_request_body_rejected(client: TestClient) -> None:
    """Verify empty string message is rejected with HTTP 422 without invoking workflow."""
    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422
        assert mock_run.call_count == 0


@pytest.mark.parametrize("ws", ["   ", " ", "\t", "\n", " \t \n "])
def test_whitespace_only_request_rejected(client: TestClient, ws: str) -> None:
    """Verify whitespace-only messages are rejected with HTTP 422 without invoking workflow."""
    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        response = client.post("/api/v1/chat", json={"message": ws})
        assert response.status_code == 422
        assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# Test 41 & 48: Initial TravelState Verification and Request ID Generation
# ---------------------------------------------------------------------------


def test_workflow_receives_complete_initial_state(client: TestClient) -> None:
    """Verify initial state contains user_query, valid session_id, and dynamic request_id."""
    mock_final_state = create_initial_state("Query", "s", "r")
    mock_final_state["final_response"] = "Ok"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post(
            "/api/v1/chat",
            json={"message": "Flights to London tomorrow"},
        )

        assert response.status_code == 200
        passed_state: TravelState = mock_run.call_args[0][0]

        # Invariants from create_initial_state
        assert passed_state["user_query"] == "Flights to London tomorrow"
        assert passed_state["session_id"].startswith("sess_")
        assert passed_state["request_id"].startswith("req_")
        assert passed_state["flight_status"] == "not_requested"
        assert passed_state["research_status"] == "not_requested"
        assert passed_state["validation_status"] == "pending"


# ---------------------------------------------------------------------------
# Test 42: Successful Workflow Response
# ---------------------------------------------------------------------------


def test_successful_workflow_response(client: TestClient) -> None:
    """Verify successful workflow produces TravelResponse with status='success'."""
    mock_final_state = create_initial_state("Query", "s", "r")
    mock_final_state["final_response"] = "Flight EK522 is scheduled."
    mock_final_state["flight_status"] = "success"
    mock_final_state["validation_status"] = "passed"
    mock_final_state["workflow_status"] = "completed"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Flight EK522 status"})

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Flight EK522 is scheduled."
        assert data["status"] == "success"
        assert data["errors"] == []


# ---------------------------------------------------------------------------
# Test 43: Business No-Results Response
# ---------------------------------------------------------------------------


def test_no_results_response_partial_status(client: TestClient) -> None:
    """Verify zero flight records return HTTP 200 with status='partial' instead of HTTP 500."""
    mock_final_state = create_initial_state("Query", "s", "r")
    mock_final_state["final_response"] = "No matching flights were found for your route."
    mock_final_state["flight_status"] = "no_results"
    mock_final_state["validation_status"] = "passed"
    mock_final_state["workflow_status"] = "completed"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Find flights from JFK to ORD"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert "No matching flights" in data["response"]


# ---------------------------------------------------------------------------
# Test 44: Business Validation Failure
# ---------------------------------------------------------------------------


def test_validation_failure_handling(client: TestClient) -> None:
    """Verify validation_status='failed' returns status='partial' with sanitized error payload."""
    mock_final_state = create_initial_state("Query", "s", "r")
    mock_final_state["final_response"] = "Departure date cannot be in the past."
    mock_final_state["validation_status"] = "failed"
    mock_final_state["validation_errors"] = ["Departure date 2020-01-01 is in the past"]
    mock_final_state["workflow_status"] = "completed"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Book flight yesterday"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert len(data["errors"]) == 1
        assert data["errors"][0]["code"] == "VALIDATION_FAILED"
        assert "2020-01-01" in data["errors"][0]["message"]


# ---------------------------------------------------------------------------
# Test 45: Provider Failure Handling
# ---------------------------------------------------------------------------


def test_provider_failure_returns_sanitized_503(client: TestClient) -> None:
    """Verify external provider outage raises sanitized HTTP 503 without secret leakage."""
    secret_token = "secret_provider_token_xyz"

    class ProviderTimeoutError(RuntimeError):
        pass

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = ProviderTimeoutError(f"AviationStack timeout with {secret_token}")

        response = client.post("/api/v1/chat", json={"message": "Status of flight EK522"})

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert "provider service is temporarily unavailable" in data["response"]
        assert secret_token not in response.text


# ---------------------------------------------------------------------------
# Test 46: Unexpected Workflow Exception Handling
# ---------------------------------------------------------------------------


def test_unexpected_exception_returns_sanitized_500(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify generic unexpected workflow exception returns HTTP 500 without leaking traceback."""
    caplog.set_level(logging.ERROR)

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = ZeroDivisionError("division by zero internal bug")

        response = client.post("/api/v1/chat", json={"message": "Trigger unexpected crash"})

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "An unexpected error occurred" in data["response"]
        assert "ZeroDivisionError" not in response.text
        assert "division by zero" not in response.text

        # Verify logged safely
        assert any("ZeroDivisionError" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 47: Session ID Preservation
# ---------------------------------------------------------------------------


def test_session_id_preservation_in_final_response(client: TestClient) -> None:
    """Verify client-supplied session ID is preserved throughout route conversion."""
    custom_session = "sess-preserve-47"
    mock_final_state = create_initial_state("Q", custom_session, "r")
    mock_final_state["final_response"] = "Done"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello", "session_id": custom_session},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == custom_session


# ---------------------------------------------------------------------------
# Test 49: Exactly-Once Workflow Invocation
# ---------------------------------------------------------------------------


def test_workflow_called_exactly_once(client: TestClient) -> None:
    """Verify one HTTP request triggers the LangGraph workflow exactly once."""
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Response"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Flights"})
        assert response.status_code == 200
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# Test 50: No Direct Agent or External Tool Invocation
# ---------------------------------------------------------------------------


def test_no_direct_agent_or_tool_calls_from_route(client: TestClient) -> None:
    """Verify route handler does NOT bypass workflow to invoke agents or tools directly."""
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Processed"

    with (
        patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run,
        patch("app.agents.router_agent.aroute_request") as mock_router,
        patch("app.agents.flight_agent.run_flight_agent") as mock_flight,
        patch("app.agents.research_agent.run_research_agent") as mock_research,
        patch("app.agents.validator_agent.run_validator_agent") as mock_validator,
        patch("app.agents.response_agent.run_response_agent") as mock_response,
        patch("app.tools.aviationstack.AviationStackClient") as mock_aviation,
        patch("app.tools.tavily_search.TavilySearchClient") as mock_tavily,
    ):
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Flight status"})

        assert response.status_code == 200
        assert mock_run.call_count == 1

        # Direct domain agent/tool calls must be zero
        assert not mock_router.called
        assert not mock_flight.called
        assert not mock_research.called
        assert not mock_validator.called
        assert not mock_response.called
        assert not mock_aviation.called
        assert not mock_tavily.called


# ---------------------------------------------------------------------------
# Test 51: Response Schema Validation
# ---------------------------------------------------------------------------


def test_response_conforms_to_travel_response_schema(client: TestClient) -> None:
    """Verify the endpoint output strictly satisfies TravelResponse schema."""
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Schedule confirmed."
    mock_final_state["sources"] = [
        {"title": "Emirates", "url": "https://emirates.com/ek522", "score": 0.95}
    ]

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Flight check"})
        assert response.status_code == 200

        # Strict parsing
        parsed = TravelResponse.model_validate(response.json())
        assert parsed.response == "Schedule confirmed."
        assert len(parsed.sources) == 1
        assert str(parsed.sources[0].url) == "https://emirates.com/ek522"


# ---------------------------------------------------------------------------
# Test 52: Internal Fields Not Exposed
# ---------------------------------------------------------------------------


def test_internal_travel_state_fields_not_exposed(client: TestClient) -> None:
    """Verify internal operational fields are completely omitted from public response."""
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Flight status: on time."
    mock_final_state["active_agent"] = "flight_agent"
    mock_final_state["completed_agents"] = ["router_agent", "flight_agent"]
    mock_final_state["metadata"] = {"private_metric": 42, "provider_id": "avi_99"}
    mock_final_state["flight_results"] = [{"flight_iata": "EK522", "raw": "data"}]
    mock_final_state["research_results"] = [{"raw_content": "some text"}]
    mock_final_state["workflow_status"] = "completed"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Flight check"})
        data = response.json()

        for forbidden_key in [
            "active_agent",
            "completed_agents",
            "metadata",
            "flight_results",
            "research_results",
            "workflow_status",
            "flight_query",
            "research_query",
        ]:
            assert forbidden_key not in data, (
                f"Internal key {forbidden_key!r} leaked to public response"
            )


# ---------------------------------------------------------------------------
# Test 53: Secrets Not Exposed
# ---------------------------------------------------------------------------


def test_secrets_not_exposed_in_response_or_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify simulated secrets in state or errors never appear in HTTP response or logs."""
    caplog.set_level(logging.INFO)

    secret_key = "sk-proj-super-secret-key-12345"
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Normal user response."
    mock_final_state["errors"] = [f"Transport timeout involving {secret_key}"]
    mock_final_state["metadata"] = {"api_key": secret_key}

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Flight status"})

        # Response body must not contain secret
        assert secret_key not in response.text

        # Route operational logs must not contain secret
        for record in caplog.records:
            assert secret_key not in record.message


# ---------------------------------------------------------------------------
# Test 54: Content Type
# ---------------------------------------------------------------------------


def test_response_content_type_is_json(client: TestClient) -> None:
    """Verify endpoint responds with application/json header."""
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Hello"

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Hello"})
        assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Test 55: OpenAPI Contract Generation
# ---------------------------------------------------------------------------


def test_openapi_schema_contains_chat_endpoint(app: FastAPI) -> None:
    """Verify FastAPI automatically generates the OpenAPI schema for POST /api/v1/chat."""
    openapi = app.openapi()
    assert "/api/v1/chat" in openapi["paths"]
    assert "post" in openapi["paths"]["/api/v1/chat"]

    post_op = openapi["paths"]["/api/v1/chat"]["post"]
    assert "TravelRequest" in str(post_op) or "requestBody" in post_op
    assert "200" in post_op["responses"]


# ---------------------------------------------------------------------------
# Test 56: No Health Endpoint on Router
# ---------------------------------------------------------------------------


def test_no_health_endpoint_on_chat_router() -> None:
    """Verify chat router defines strictly the chat endpoint and does not include /health."""
    registered_paths = [r.path for r in router.routes]
    assert "/health" not in registered_paths
    assert "/api/v1/health" not in registered_paths


# ---------------------------------------------------------------------------
# Test 57: Error Response Design
# ---------------------------------------------------------------------------


def test_error_response_design_uses_typed_api_error(client: TestClient) -> None:
    """Verify errors list in response uses typed APIError structure."""
    mock_final_state = create_initial_state("Q", "s", "r")
    mock_final_state["final_response"] = "Warning during search."
    mock_final_state["errors"] = ["Provider timeout occurred on partner gateway"]

    with patch("app.api.routes.run_workflow", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_final_state

        response = client.post("/api/v1/chat", json={"message": "Search flight"})
        data = response.json()

        assert len(data["errors"]) == 1
        err_item = data["errors"][0]
        assert "code" in err_item
        assert "message" in err_item
        assert err_item["code"] == "PROVIDER_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Test 62 & 63: Architectural Compliance
# ---------------------------------------------------------------------------


def test_routes_architecture_compliance() -> None:
    """
    Verify app.api.routes strictly functions as an HTTP adapter and does NOT:
        - import or construct StateGraph workflows
        - call LLMs (OpenAI / ChatOpenAI)
        - execute external tools (AviationStack, Tavily, requests, httpx)
        - access environment variables (os.getenv, os.environ)
        - access credentials (OPENAI_API_KEY, TAVILY_API_KEY, AVIATIONSTACK_API_KEY)
    """
    import app.api.routes as routes_module

    source = inspect.getsource(routes_module)

    prohibited_tokens = [
        "StateGraph",
        "add_node",
        "add_edge",
        "add_conditional_edges",
        ".compile(",
        "ChatOpenAI",
        "OpenAI",
        "TavilyClient",
        "AviationStack",
        "httpx",
        "requests",
        "os.getenv",
        "os.environ",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "AVIATIONSTACK_API_KEY",
        "traceback",
        "str(exc)",
        "repr(exc)",
    ]

    for token in prohibited_tokens:
        assert token not in source, f"app.api.routes must not contain prohibited token: {token!r}"
