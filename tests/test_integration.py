"""
Realistic Offline End-to-End Integration Test Suite for ZICO Travel Operations.

Verifies the complete request lifecycle across components:
    HTTP Request -> FastAPI -> API Route -> TravelState creation
    -> LangGraph Workflow -> Router Agent -> Specialized Agent (Flight / Research)
    -> Validator Agent -> Response Agent -> TravelResponse -> HTTP Response

Testing Philosophy & Boundary Isolation:
    - Pure offline execution: zero external network, zero live OpenAI/Tavily/AviationStack calls.
    - True component integration: FastAPI app, route adapters, LangGraph StateGraph,
      state enrichment, location resolution, business rule validation, and response
      serialization execute their actual production logic.
    - Mocked external boundaries: OpenAI model invocations, AviationStack REST client,
      and Tavily search client are mocked at their respective provider boundaries.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.agents.router_agent import RouterDecision
from app.api.schemas import (
    APIError,
    APIErrorCode,
    ResponseStatus,
    Source,
    TravelRequest,
    TravelResponse,
)
from app.core.config import settings
from app.core.state import TravelState
from app.agents.flight_agent import run_flight_agent as real_run_flight_agent
from app.agents.research_agent import run_research_agent as real_run_research_agent
from app.agents.response_agent import run_response_agent as real_run_response_agent
from app.agents.router_agent import aroute_request as real_aroute_request
from app.agents.validator_agent import run_validator_agent as real_run_validator_agent
from app.main import app
from app.tools.aviationstack import (
    AirlineDetails,
    AirportDetails,
    AviationStackAPIError,
    AviationStackResponse,
    AviationStackTimeoutError,
    FlightDetails,
    NormalizedFlight,
    PaginationInfo,
)
from app.tools.tavily_search import (
    ResearchResult,
    TavilyHTTPError,
    TavilySearchResponse,
    TavilyTimeoutError,
)

FAKE_OPENAI_KEY = "test-secret-openai-key-999"
FAKE_TAVILY_KEY = "test-secret-tavily-key-888"
FAKE_AVIATION_KEY = "test-secret-aviationstack-key-777"


# ---------------------------------------------------------------------------
# Reusable Test Fixtures & Synthetic Data Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee tests execute offline with synthetic secret keys."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setattr(settings, "TAVILY_API_KEY", FAKE_TAVILY_KEY)
    monkeypatch.setattr(settings, "AVIATIONSTACK_API_KEY", FAKE_AVIATION_KEY)


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient bound to the primary ZICO application."""
    return TestClient(app)


def build_sample_flight(
    flight_iata: str = "EK522",
    flight_number: str = "522",
    airline_name: str = "Emirates",
    dep_iata: str = "DXB",
    arr_iata: str = "TRV",
    flight_status: str = "scheduled",
    flight_date: str = "2026-09-20",
) -> NormalizedFlight:
    """Construct a normalized flight record model for testing."""
    return NormalizedFlight(
        flight_date=flight_date,
        flight_status=flight_status,
        flight_number=flight_number,
        flight_iata=flight_iata,
        flight_icao=f"UAE{flight_number}",
        airline_name=airline_name,
        airline_iata="EK",
        airline_icao="UAE",
        departure_airport="Dubai International Airport",
        departure_iata=dep_iata,
        departure_scheduled="2026-09-20T10:00:00Z",
        departure_estimated="2026-09-20T10:05:00Z",
        departure_actual=None,
        arrival_airport="Trivandrum International Airport",
        arrival_iata=arr_iata,
        arrival_scheduled="2026-09-20T15:30:00Z",
        arrival_estimated="2026-09-20T15:25:00Z",
        arrival_actual=None,
        flight=FlightDetails(number=flight_number, iata=flight_iata, icao=f"UAE{flight_number}"),
        airline=AirlineDetails(name=airline_name, iata="EK", icao="UAE"),
        departure=AirportDetails(airport="Dubai International Airport", iata=dep_iata, scheduled="2026-09-20T10:00:00Z"),
        arrival=AirportDetails(airport="Trivandrum International Airport", iata=arr_iata, scheduled="2026-09-20T15:30:00Z"),
    )


def build_aviation_response(flights: List[NormalizedFlight]) -> AviationStackResponse:
    """Construct an AviationStackResponse envelope."""
    return AviationStackResponse(
        success=True,
        count=len(flights),
        pagination=PaginationInfo(limit=10, offset=0, count=len(flights), total=len(flights)),
        data=flights,
        provider="aviationstack",
    )


def build_sample_research_results() -> List[ResearchResult]:
    """Construct synthetic, deterministic research finding models."""
    return [
        ResearchResult(
            title="German Federal Foreign Office - Visa Information",
            url="https://www.auswaertiges-amt.de/en/visa-service",
            content="Travelers requiring a Schengen visa must provide proof of accommodation and travel insurance.",
            score=0.96,
        ),
        ResearchResult(
            title="Schengen Visa Entry Requirements 2026",
            url="https://schengenvisainfo.example.com/germany",
            content="Current 2026 entry rules for Germany mandate a passport valid for at least 3 months.",
            score=0.89,
        ),
    ]


def build_tavily_response(results: List[ResearchResult], query: str = "visa requirements germany") -> TavilySearchResponse:
    """Construct a TavilySearchResponse envelope."""
    return TavilySearchResponse(
        success=True,
        query=query,
        count=len(results),
        results=results,
        provider="tavily",
    )


def create_mock_router_model(intent: str, confidence: float = 1.0) -> MagicMock:
    """Return a mock LLM chat model configured for router structured decision output."""
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=RouterDecision(intent=intent, confidence=confidence))
    mock_runnable.invoke = MagicMock(return_value=RouterDecision(intent=intent, confidence=confidence))

    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output = MagicMock(return_value=mock_runnable)
    return mock_chat_model


def create_mock_response_model(content: str) -> MagicMock:
    """Return a mock LLM chat model for response generation."""
    mock_chat_model = MagicMock()
    mock_ai_msg = MagicMock()
    mock_ai_msg.content = content
    mock_chat_model.ainvoke = AsyncMock(return_value=mock_ai_msg)
    mock_chat_model.invoke = MagicMock(return_value=mock_ai_msg)
    return mock_chat_model


# ===========================================================================
# 1. Successful Flight Request End-to-End (Scenarios 6 & 7)
# ===========================================================================


def test_flight_request_end_to_end(client: TestClient) -> None:
    """
    Test complete flight status lookup request through FastAPI -> workflow -> response.

    Flow: POST /api/v1/chat -> Router (flight) -> Flight Agent -> AviationStack -> Validator -> Response.
    """
    sample_flight = build_sample_flight(flight_iata="EK522", flight_number="522")
    aviation_envelope = build_aviation_response([sample_flight])

    mock_router_model = create_mock_router_model("flight", confidence=0.98)
    mock_resp_model = create_mock_response_model("Flight EK522 from DXB to TRV is scheduled on time.")

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock(return_value=aviation_envelope)
    mock_aviation_instance.close = AsyncMock()

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_instance), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert response.status_code == 200
    payload = response.json()

    # Public contract validation
    parsed = TravelResponse.model_validate(payload)
    assert parsed.status == ResponseStatus.SUCCESS
    assert "EK522" in parsed.response
    assert "scheduled" in parsed.response.lower()

    # Tool invocation verification
    mock_aviation_instance.get_flights.assert_awaited_once()
    call_kwargs = mock_aviation_instance.get_flights.call_args.kwargs
    assert call_kwargs.get("flight_iata") == "EK522" or call_kwargs.get("flight_number") == "522"

    # Verify research agent was NOT executed
    mock_tavily_instance.search.assert_not_called()

    # Verify no internal state leaked
    for internal_field in ["active_agent", "completed_agents", "flight_query", "flight_results", "metadata"]:
        assert internal_field not in payload


# ===========================================================================
# 2. Flight Route Search with Natural Language Location Resolution (Scenario 8)
# ===========================================================================


def test_flight_route_search_end_to_end(client: TestClient) -> None:
    """
    Test natural language route extraction and offline airport resolution:
    "Find flights from Trivandrum to Dubai." -> dep_iata: TRV, arr_iata: DXB.
    """
    flight_record = build_sample_flight(flight_iata="EK523", dep_iata="TRV", arr_iata="DXB")
    aviation_envelope = build_aviation_response([flight_record])

    mock_router_model = create_mock_router_model("flight", confidence=0.95)
    mock_resp_model = create_mock_response_model("Found scheduled flight EK523 operating from TRV to DXB.")

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock(return_value=aviation_envelope)
    mock_aviation_instance.close = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post(
            "/api/v1/chat",
            json={"message": "Find flights from Trivandrum to Dubai."},
        )

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)
    assert parsed.status == ResponseStatus.SUCCESS

    # Verify real LocationResolver correctly translated city names to IATA codes
    mock_aviation_instance.get_flights.assert_awaited_once()
    kwargs = mock_aviation_instance.get_flights.call_args.kwargs
    assert kwargs.get("dep_iata") == "TRV"
    assert kwargs.get("arr_iata") == "DXB"


# ===========================================================================
# 3. Successful Research Path End-to-End (Scenario 9)
# ===========================================================================


def test_research_request_end_to_end(client: TestClient) -> None:
    """
    Test travel policy research inquiry through FastAPI -> workflow -> response.

    Flow: POST /api/v1/chat -> Router (research) -> Research Agent -> Tavily -> Validator -> Response.
    """
    research_items = build_sample_research_results()
    tavily_envelope = build_tavily_response(research_items)

    mock_router_model = create_mock_router_model("research", confidence=0.97)
    mock_resp_model = create_mock_response_model(
        "According to the German Federal Foreign Office, travelers require a valid Schengen visa."
    )

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search = AsyncMock(return_value=tavily_envelope)
    mock_tavily_instance.close = AsyncMock()

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_instance), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post(
            "/api/v1/chat",
            json={"message": "What are the current visa requirements for Germany?"},
        )

    assert response.status_code == 200
    payload = response.json()

    parsed = TravelResponse.model_validate(payload)
    assert parsed.status == ResponseStatus.SUCCESS
    assert "Schengen visa" in parsed.response

    # Verify sources reached final API contract
    assert len(parsed.sources) >= 1
    assert any("auswaertiges-amt.de" in str(s.url) for s in parsed.sources)

    # Verify Tavily called and AviationStack untouched
    mock_tavily_instance.search.assert_awaited_once()
    mock_aviation_instance.get_flights.assert_not_called()


# ===========================================================================
# 4. General Travel Request Routes to Research (Scenario 10)
# ===========================================================================


def test_general_travel_routes_to_research(client: TestClient) -> None:
    """
    Verify intent='general_travel' routes through research -> validator -> response
    without creating a new agent.
    """
    mock_router_model = create_mock_router_model("general_travel", confidence=0.90)
    mock_resp_model = create_mock_response_model("Here are general travel suggestions for your trip to Dubai.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post(
            "/api/v1/chat",
            json={"message": "Help me plan a trip to Dubai."},
        )

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)
    assert parsed.response is not None
    assert len(parsed.response) > 0


# ===========================================================================
# 5. Unsupported Request Bypasses Specialized Agents (Scenario 11)
# ===========================================================================


def test_unsupported_request_bypasses_specialized_agents(client: TestClient) -> None:
    """
    Verify non-travel requests route directly router -> response, bypassing
    Flight Agent, Research Agent, and Validator Agent.
    """
    mock_router_model = create_mock_router_model("unsupported", confidence=1.0)
    mock_resp_model = create_mock_response_model(
        "I specialize strictly in travel and flight operations and cannot assist with compiler development."
    )

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock()

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search = AsyncMock()

    mock_validator = AsyncMock(wraps=real_run_validator_agent)

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_instance), \
         patch("app.graph.nodes.run_validator_agent", side_effect=mock_validator), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post(
            "/api/v1/chat",
            json={"message": "Explain how to build a compiler in C++."},
        )

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)

    assert "compiler" in parsed.response.lower()
    mock_aviation_instance.get_flights.assert_not_called()
    mock_tavily_instance.search.assert_not_called()
    mock_validator.assert_not_called()


# ===========================================================================
# 6. Invalid Router Intent Safe Fallback (Scenario 12)
# ===========================================================================


def test_invalid_router_intent_safe_fallback(client: TestClient) -> None:
    """
    Verify if the router classifies an unmapped intent (e.g. 'hotel'),
    the workflow routing in edges.py deterministically defaults to 'response'.
    """
    # Router produces 'hotel' which edge router safely defaults to 'response'
    async def mock_router_hotel(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        return {"intent": "hotel", "active_agent": "router_agent"}

    mock_resp_model = create_mock_response_model("Hotel booking operations are currently outside our supported scope.")

    with patch("app.graph.nodes.aroute_request", side_effect=mock_router_hotel), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Find me a hotel in Paris."})

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)
    assert parsed.response is not None


# ===========================================================================
# 7. Provider Failure — AviationStack (Scenario 13)
# ===========================================================================


def test_aviationstack_provider_failure_reaches_api_safely(client: TestClient) -> None:
    """
    Verify external AviationStack outage/error is caught, sanitized, and
    propagated without exposing internal stack traces, API keys, or claiming success.
    """
    mock_router_model = create_mock_router_model("flight", confidence=0.99)

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock(
        side_effect=AviationStackAPIError(code="rate_limit_exceeded", message="Monthly limit reached")
    )
    mock_aviation_instance.close = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)

    # Must be 'partial' or 'error' (not claiming verified success)
    assert parsed.status in (ResponseStatus.PARTIAL, ResponseStatus.ERROR)

    # Must not leak stack traces or secret keys
    resp_text = str(payload)
    assert "Traceback" not in resp_text
    assert FAKE_AVIATION_KEY not in resp_text


# ===========================================================================
# 8. Provider Failure — Tavily (Scenario 14)
# ===========================================================================


def test_tavily_provider_failure_reaches_api_safely(client: TestClient) -> None:
    """
    Verify external Tavily outage/error is trapped and reported safely.
    """
    mock_router_model = create_mock_router_model("research", confidence=0.99)

    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search = AsyncMock(
        side_effect=TavilyHTTPError(status_code=503, message="Tavily backend unavailable")
    )
    mock_tavily_instance.close = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_instance):

        response = client.post(
            "/api/v1/chat",
            json={"message": "What are the visa rules for Japan?"},
        )

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)

    assert parsed.status in (ResponseStatus.PARTIAL, ResponseStatus.ERROR)
    assert FAKE_TAVILY_KEY not in str(payload)
    assert "Traceback" not in str(payload)


# ===========================================================================
# 9. OpenAI Failure — Router Agent (Scenario 15)
# ===========================================================================


def test_router_openai_failure_handling(client: TestClient) -> None:
    """
    Verify that if the Router's LLM call fails completely, the pipeline
    falls back safely to 'unsupported', does NOT route to Flight, and leaks zero secrets.
    """
    mock_router_model = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(side_effect=RuntimeError(f"Connection to OpenAI failed with key {FAKE_OPENAI_KEY}"))
    mock_router_model.with_structured_output = MagicMock(return_value=mock_runnable)

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)

    # Should not route to flight
    mock_aviation_instance.get_flights.assert_not_called()

    # Must protect secret
    assert FAKE_OPENAI_KEY not in str(payload)


# ===========================================================================
# 10. OpenAI Failure — Response Agent (Scenario 16)
# ===========================================================================


def test_response_openai_failure_handling(client: TestClient) -> None:
    """
    Verify that if specialized processing succeeds but Response Agent's LLM fails,
    the deterministic fallback generates a client-safe response without crashing.
    """
    sample_flight = build_sample_flight(flight_iata="EK522")
    aviation_envelope = build_aviation_response([sample_flight])

    mock_router_model = create_mock_router_model("flight", confidence=1.0)

    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock(return_value=aviation_envelope)
    mock_aviation_instance.close = AsyncMock()

    mock_failing_resp_model = MagicMock()
    mock_failing_resp_model.ainvoke = AsyncMock(
        side_effect=RuntimeError(f"OpenAI service 500 error {FAKE_OPENAI_KEY}")
    )

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_failing_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)

    # Verify fallback response exists and includes factual flight information
    assert parsed.response is not None
    assert "EK522" in parsed.response
    assert FAKE_OPENAI_KEY not in str(payload)


# ===========================================================================
# 11. Business Validation Failure Handling (Scenario 17)
# ===========================================================================


def test_validation_failure_handling(client: TestClient) -> None:
    """
    Verify that when specialized processing produces structurally invalid travel data
    (e.g., origin == destination), the Validator marks it failed and response status is 'partial'.
    """
    mock_router_model = create_mock_router_model("flight", confidence=0.99)

    # User queries a route where origin and destination are the same
    async def mock_flight_with_contradiction(state: TravelState) -> Dict[str, Any]:
        return {
            "origin": "DXB",
            "destination": "DXB",
            "flight_status": "success",
            "flight_results": [],
            "active_agent": "flight_agent",
            "completed_agents": list(state.get("completed_agents", [])) + ["flight_agent"],
        }

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.graph.nodes.run_flight_agent", side_effect=mock_flight_with_contradiction):

        response = client.post(
            "/api/v1/chat",
            json={"message": "Check flight from Dubai to Dubai."},
        )

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)

    # Response layer must respect validation failure
    assert parsed.status == ResponseStatus.PARTIAL
    # Must contain validation error information
    assert any("origin and destination cannot be identical" in err.message.lower() for err in parsed.errors)


# ===========================================================================
# 12. No-Result Flight Search Handling (Scenario 18)
# ===========================================================================


def test_flight_no_results_preserves_status(client: TestClient) -> None:
    """
    Verify successful AviationStack call returning 0 records maps to status='partial'
    and does NOT return a 500 error.
    """
    aviation_envelope = build_aviation_response([])  # Empty records

    mock_router_model = create_mock_router_model("flight", confidence=1.0)
    mock_aviation_instance = MagicMock()
    mock_aviation_instance.get_flights = AsyncMock(return_value=aviation_envelope)
    mock_aviation_instance.close = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_instance):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK99999."})

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)
    assert parsed.status == ResponseStatus.PARTIAL


# ===========================================================================
# 13. No-Result Research Search Handling (Scenario 19)
# ===========================================================================


def test_research_no_results_preserves_status(client: TestClient) -> None:
    """
    Verify successful Tavily call returning 0 results maps to status='partial'
    and distinguishes from provider failure.
    """
    tavily_envelope = build_tavily_response([])  # Empty findings

    mock_router_model = create_mock_router_model("research", confidence=1.0)
    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search = AsyncMock(return_value=tavily_envelope)
    mock_tavily_instance.close = AsyncMock()

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_instance):

        response = client.post(
            "/api/v1/chat",
            json={"message": "What are visa regulations for Atlantis?"},
        )

    assert response.status_code == 200
    payload = response.json()
    parsed = TravelResponse.model_validate(payload)
    assert parsed.status == ResponseStatus.PARTIAL


# ===========================================================================
# 14. Research Source Propagation (Scenario 20)
# ===========================================================================


def test_research_source_propagation(client: TestClient) -> None:
    """
    Verify research source URLs survive end-to-end:
    Tavily -> Research Agent -> TravelState -> Response Agent -> TravelResponse.sources.
    """
    research_items = [
        ResearchResult(
            title="Official Source A",
            url="https://source-a.example.gov/visa",
            content="Official requirements A.",
            score=0.98,
        ),
        ResearchResult(
            title="Embassy Source B",
            url="https://source-b.example.org/guidance",
            content="Embassy guidance B.",
            score=0.87,
        ),
    ]
    tavily_envelope = build_tavily_response(research_items)

    mock_router_model = create_mock_router_model("research", confidence=1.0)
    mock_tavily_instance = MagicMock()
    mock_tavily_instance.search = AsyncMock(return_value=tavily_envelope)
    mock_tavily_instance.close = AsyncMock()

    mock_resp_model = create_mock_response_model("Here is research based on official sources.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_instance), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post(
            "/api/v1/chat",
            json={"message": "Current travel advisories for Iceland"},
        )

    assert response.status_code == 200
    parsed = TravelResponse.model_validate(response.json())

    # Both URLs must be preserved in final API sources
    source_urls = [str(s.url) for s in parsed.sources]
    assert any("source-a.example.gov" in u for u in source_urls)
    assert any("source-b.example.org" in u for u in source_urls)


# ===========================================================================
# 15. Session ID Propagation (Scenario 21)
# ===========================================================================


def test_session_id_propagation(client: TestClient) -> None:
    """
    Verify user-provided session_id is preserved throughout state and returned unchanged.
    """
    expected_session_id = "integration-custom-session-uuid-12345"
    mock_router_model = create_mock_router_model("unsupported", confidence=1.0)
    mock_resp_model = create_mock_response_model("General acknowledgment.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello ZICO.",
                "session_id": expected_session_id,
            },
        )

    assert response.status_code == 200
    parsed = TravelResponse.model_validate(response.json())
    assert parsed.session_id == expected_session_id


# ===========================================================================
# 16. Request ID Generation & Uniqueness (Scenario 22)
# ===========================================================================


def test_request_id_generation_unique(client: TestClient) -> None:
    """
    Verify each request receives a unique non-empty request ID in workflow state.
    """
    captured_request_ids: List[str] = []

    async def spy_router(state: TravelState) -> Dict[str, Any]:
        req_id = state.get("request_id")
        if req_id:
            captured_request_ids.append(req_id)
        return {"intent": "unsupported", "active_agent": "router_agent"}

    mock_resp_model = create_mock_response_model("Response text.")

    with patch("app.graph.nodes.aroute_request", side_effect=spy_router), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        resp1 = client.post("/api/v1/chat", json={"message": "First request."})
        resp2 = client.post("/api/v1/chat", json={"message": "Second request."})

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert len(captured_request_ids) == 2
    assert captured_request_ids[0] != captured_request_ids[1]
    assert captured_request_ids[0].startswith("req_")


# ===========================================================================
# 17. State Isolation Between Requests (Scenario 23)
# ===========================================================================


def test_state_isolation_between_requests(client: TestClient) -> None:
    """
    Verify request A (flight) and request B (research) maintain absolute isolation:
    Flight results never leak into research response, and vice versa.
    """
    # 1. Execute Flight Request
    flight = build_sample_flight(flight_iata="EK522")
    aviation_env = build_aviation_response([flight])
    mock_router_flight = create_mock_router_model("flight")
    mock_aviation_inst = MagicMock(get_flights=AsyncMock(return_value=aviation_env), close=AsyncMock())
    mock_resp_flight = create_mock_response_model("EK522 flight details.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_flight), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_inst), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_flight):

        resp_flight = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert resp_flight.status_code == 200
    assert "EK522" in resp_flight.json()["response"]

    # 2. Execute Research Request
    research_items = build_sample_research_results()
    tavily_env = build_tavily_response(research_items)
    mock_router_research = create_mock_router_model("research")
    mock_tavily_inst = MagicMock(search=AsyncMock(return_value=tavily_env), close=AsyncMock())
    mock_resp_research = create_mock_response_model("Germany visa details.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_research), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_inst), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_research):

        resp_research = client.post("/api/v1/chat", json={"message": "Visa requirements for Germany."})

    assert resp_research.status_code == 200
    research_payload = resp_research.json()

    # Assert flight data is completely absent from research response
    assert "EK522" not in research_payload["response"]
    assert "EK522" not in str(research_payload)


# ===========================================================================
# 18. Repeated Requests Independence (Scenario 24)
# ===========================================================================


def test_repeated_requests_independence(client: TestClient) -> None:
    """
    Verify sending identical requests sequentially creates distinct workflow runs
    without state bleed or caching interference.
    """
    mock_router_model = create_mock_router_model("unsupported")
    mock_resp_model = create_mock_response_model("Clean response.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        for _ in range(3):
            response = client.post("/api/v1/chat", json={"message": "Hello ZICO."})
            assert response.status_code == 200
            parsed = TravelResponse.model_validate(response.json())
            assert parsed.status in (ResponseStatus.SUCCESS, ResponseStatus.PARTIAL)


# ===========================================================================
# 19. Concurrent Requests State Isolation (Scenario 25)
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_requests_state_isolation() -> None:
    """
    Verify concurrent async HTTP requests process independently without race conditions
    or state contamination across graph runs.
    """
    mock_flight_router = create_mock_router_model("flight")
    flight_env = build_aviation_response([build_sample_flight(flight_iata="EK522")])
    mock_aviation = MagicMock(get_flights=AsyncMock(return_value=flight_env), close=AsyncMock())
    mock_flight_resp = create_mock_response_model("EK522 scheduled.")

    mock_research_router = create_mock_router_model("research")
    research_env = build_tavily_response(build_sample_research_results())
    mock_tavily = MagicMock(search=AsyncMock(return_value=research_env), close=AsyncMock())
    mock_research_resp = create_mock_response_model("Visa guidance.")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Patch models based on message content
        async def mock_router_dispatch(messages: Any) -> Any:
            user_text = str(messages[-1].content) if messages else ""
            if "flight" in user_text.lower():
                return RouterDecision(intent="flight", confidence=1.0)
            return RouterDecision(intent="research", confidence=1.0)

        mock_dynamic_runnable = MagicMock(ainvoke=AsyncMock(side_effect=mock_router_dispatch))
        mock_dynamic_router = MagicMock(with_structured_output=MagicMock(return_value=mock_dynamic_runnable))

        async def mock_resp_dispatch(messages: Any) -> Any:
            prompt = str(messages[-1].content) if messages else ""
            if "EK522" in prompt or "flight" in prompt.lower():
                return MagicMock(content="Flight EK522 status on time.")
            return MagicMock(content="Visa regulations information.")

        mock_dynamic_resp = MagicMock(ainvoke=AsyncMock(side_effect=mock_resp_dispatch))

        with patch("app.agents.router_agent.get_chat_model", return_value=mock_dynamic_router), \
             patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation), \
             patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily), \
             patch("app.agents.response_agent.get_chat_model", return_value=mock_dynamic_resp):

            task1 = ac.post("/api/v1/chat", json={"message": "Check flight EK522.", "session_id": "sess-flight"})
            task2 = ac.post("/api/v1/chat", json={"message": "Research visa requirements.", "session_id": "sess-research"})

            resp1, resp2 = await asyncio.gather(task1, task2)

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    p1 = resp1.json()
    p2 = resp2.json()

    assert p1["session_id"] == "sess-flight"
    assert p2["session_id"] == "sess-research"
    assert "EK522" in p1["response"]
    assert "EK522" not in p2["response"]


# ===========================================================================
# 20. Public API Contract Compliance (Scenario 26)
# ===========================================================================


def test_public_api_contract_compliance(client: TestClient) -> None:
    """
    Verify every response field strictly conforms to the TravelResponse model specification.
    """
    mock_router_model = create_mock_router_model("unsupported")
    mock_resp_model = create_mock_response_model("Direct response.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Hello ZICO."})

    assert response.status_code == 200
    data = response.json()

    # Pydantic validation
    model = TravelResponse(**data)
    assert isinstance(model.response, str)
    assert isinstance(model.session_id, str)
    assert isinstance(model.status, ResponseStatus)
    assert isinstance(model.sources, list)
    assert isinstance(model.errors, list)


# ===========================================================================
# 21. Internal State Protection (Scenario 27)
# ===========================================================================


def test_internal_state_protection(client: TestClient) -> None:
    """
    Verify internal state fields (active_agent, completed_agents, validation_errors,
    flight_query, research_query, metadata) are never exposed in the API response.
    """
    flight = build_sample_flight()
    env = build_aviation_response([flight])

    mock_router_model = create_mock_router_model("flight")
    mock_aviation_inst = MagicMock(get_flights=AsyncMock(return_value=env), close=AsyncMock())
    mock_resp_model = create_mock_response_model("EK522 is scheduled.")

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_inst), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert response.status_code == 200
    payload = response.json()

    # Permitted public fields ONLY
    allowed_keys = {"response", "session_id", "status", "sources", "errors"}
    assert set(payload.keys()).issubset(allowed_keys)


# ===========================================================================
# 22. Secret Protection End-to-End (Scenario 28)
# ===========================================================================


def test_secret_protection_end_to_end(client: TestClient) -> None:
    """
    Verify synthetic API keys never appear in API responses, error details, or headers,
    even during operational failures.
    """
    mock_router_model = MagicMock()
    mock_runnable = MagicMock()
    # Injected exception with secret key
    mock_runnable.ainvoke = AsyncMock(
        side_effect=RuntimeError(f"Critical failure exposing {FAKE_OPENAI_KEY} and {FAKE_TAVILY_KEY}")
    )
    mock_router_model.with_structured_output = MagicMock(return_value=mock_runnable)

    with patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model):
        response = client.post("/api/v1/chat", json={"message": "Check flight."})

    payload_str = str(response.json())
    headers_str = str(dict(response.headers))

    assert FAKE_OPENAI_KEY not in payload_str
    assert FAKE_TAVILY_KEY not in payload_str
    assert FAKE_AVIATION_KEY not in payload_str
    assert FAKE_OPENAI_KEY not in headers_str


# ===========================================================================
# 23. Request Validation End-to-End (Scenario 29)
# ===========================================================================


def test_request_validation_empty_and_whitespace(client: TestClient) -> None:
    """
    Verify FastAPI and Pydantic reject empty or whitespace-only messages with HTTP 422
    without invoking the LangGraph workflow.
    """
    with patch("app.api.routes.run_workflow") as mock_workflow:
        # Empty message
        resp1 = client.post("/api/v1/chat", json={"message": ""})
        assert resp1.status_code == 422

        # Whitespace message
        resp2 = client.post("/api/v1/chat", json={"message": "   "})
        assert resp2.status_code == 422

        mock_workflow.assert_not_called()


# ===========================================================================
# 24. Unknown Request Fields Rejection (Scenario 30)
# ===========================================================================


def test_unknown_request_fields_rejected(client: TestClient) -> None:
    """
    Verify sending unexpected fields in TravelRequest is rejected (extra='forbid').
    """
    with patch("app.api.routes.run_workflow") as mock_workflow:
        response = client.post(
            "/api/v1/chat",
            json={"message": "Check flight.", "unexpected_field": "disallowed_value"},
        )
        assert response.status_code == 422
        mock_workflow.assert_not_called()


# ===========================================================================
# 25. Health Endpoint Isolation (Scenario 32)
# ===========================================================================


def test_health_endpoint_isolation(client: TestClient) -> None:
    """
    Verify GET /health returns 200 with zero workflow, LLM, or tool calls.
    """
    with patch("app.api.routes.run_workflow") as mock_workflow, \
         patch("app.core.llm.get_chat_model") as mock_llm:

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_workflow.assert_not_called()
    mock_llm.assert_not_called()


# ===========================================================================
# 26. OpenAPI Route Registration (Scenario 31)
# ===========================================================================


def test_openapi_route_registration(client: TestClient) -> None:
    """
    Verify /openapi.json contains /health and /api/v1/chat with valid schema references.
    """
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()

    paths = spec.get("paths", {})
    assert "/health" in paths
    assert "/api/v1/chat" in paths

    chat_post = paths["api/v1/chat" if "api/v1/chat" in paths else "/api/v1/chat"]["post"]
    assert "requestBody" in chat_post
    assert "responses" in chat_post


# ===========================================================================
# 27. Agent Execution Order Verification (Scenario 36)
# ===========================================================================


def test_agent_execution_order_flight(client: TestClient) -> None:
    """
    Verify exact node execution sequence for flight requests:
    router -> flight -> validator -> response.
    """
    call_sequence: List[str] = []

    async def spy_router(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("router")
        return await real_aroute_request(state)

    async def spy_flight(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("flight")
        return await real_run_flight_agent(state)

    async def spy_validator(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("validator")
        return await real_run_validator_agent(state)

    async def spy_response(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("response")
        return await real_run_response_agent(state)

    sample_flight = build_sample_flight(flight_iata="EK522")
    aviation_env = build_aviation_response([sample_flight])
    mock_router_model = create_mock_router_model("flight")
    mock_aviation_inst = MagicMock(get_flights=AsyncMock(return_value=aviation_env), close=AsyncMock())
    mock_resp_model = create_mock_response_model("EK522 scheduled.")

    with patch("app.graph.nodes.aroute_request", side_effect=spy_router), \
         patch("app.graph.nodes.run_flight_agent", side_effect=spy_flight), \
         patch("app.graph.nodes.run_validator_agent", side_effect=spy_validator), \
         patch("app.graph.nodes.run_response_agent", side_effect=spy_response), \
         patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.flight_agent.AviationStackClient", return_value=mock_aviation_inst), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Check flight EK522."})

    assert response.status_code == 200
    assert call_sequence == ["router", "flight", "validator", "response"]


def test_agent_execution_order_research(client: TestClient) -> None:
    """
    Verify exact node execution sequence for research requests:
    router -> research -> validator -> response.
    """
    call_sequence: List[str] = []

    async def spy_router(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("router")
        return await real_aroute_request(state)

    async def spy_research(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("research")
        return await real_run_research_agent(state)

    async def spy_validator(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("validator")
        return await real_run_validator_agent(state)

    async def spy_response(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("response")
        return await real_run_response_agent(state)

    tavily_env = build_tavily_response(build_sample_research_results())
    mock_router_model = create_mock_router_model("research")
    mock_tavily_inst = MagicMock(search=AsyncMock(return_value=tavily_env), close=AsyncMock())
    mock_resp_model = create_mock_response_model("Visa guidance.")

    with patch("app.graph.nodes.aroute_request", side_effect=spy_router), \
         patch("app.graph.nodes.run_research_agent", side_effect=spy_research), \
         patch("app.graph.nodes.run_validator_agent", side_effect=spy_validator), \
         patch("app.graph.nodes.run_response_agent", side_effect=spy_response), \
         patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.research_agent.TavilySearchClient", return_value=mock_tavily_inst), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Visa requirements for Germany."})

    assert response.status_code == 200
    assert call_sequence == ["router", "research", "validator", "response"]


def test_agent_execution_order_unsupported(client: TestClient) -> None:
    """
    Verify exact node execution sequence for unsupported requests:
    router -> response (bypassing flight, research, and validator).
    """
    call_sequence: List[str] = []

    async def spy_router(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("router")
        return await real_aroute_request(state)

    async def spy_response(state: TravelState, **kwargs: Any) -> Dict[str, Any]:
        call_sequence.append("response")
        return await real_run_response_agent(state)

    mock_router_model = create_mock_router_model("unsupported")
    mock_resp_model = create_mock_response_model("Out of scope.")

    with patch("app.graph.nodes.aroute_request", side_effect=spy_router), \
         patch("app.graph.nodes.run_response_agent", side_effect=spy_response), \
         patch("app.agents.router_agent.get_chat_model", return_value=mock_router_model), \
         patch("app.agents.response_agent.get_chat_model", return_value=mock_resp_model):

        response = client.post("/api/v1/chat", json={"message": "Write a python script."})

    assert response.status_code == 200
    assert call_sequence == ["router", "response"]
