"""
Unit tests for ZICO's Flight Operations Agent.

Verifies:
- Translation of TravelState into AviationStack queries (flight number, route search, dates).
- Location resolution integration (cities to IATA codes, existing IATA code bypass).
- Error handling for unresolved and ambiguous locations without tool execution.
- Handling of missing flight information, provider errors, timeouts, and empty results.
- State preservation, agent tracking, repeated execution safety, and JSON serialization.
- Zero unintended calls to OpenAI, Tavily, or external network APIs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.agents.flight_agent import FlightAgent, run_flight_agent
from app.core.state import TravelState, create_initial_state
from app.tools.aviationstack import (
    AirportDetails,
    AviationStackAPIError,
    AviationStackNetworkError,
    AviationStackResponse,
    AviationStackTimeoutError,
    NormalizedFlight,
)
from app.tools.location import LocationResolver, ResolvedLocation


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def create_sample_flight(
    flight_iata: str = "EK522",
    dep_iata: str = "TRV",
    arr_iata: str = "DXB",
    status: str = "scheduled",
) -> NormalizedFlight:
    """Create a populated NormalizedFlight fixture."""
    return NormalizedFlight(
        flight_date="2026-10-01",
        flight_status=status,
        flight_number="522",
        flight_iata=flight_iata,
        airline_name="Emirates",
        departure_airport="Trivandrum International Airport",
        departure_iata=dep_iata,
        departure_scheduled="2026-10-01T10:00:00+00:00",
        arrival_airport="Dubai International Airport",
        arrival_iata=arr_iata,
        arrival_scheduled="2026-10-01T12:30:00+00:00",
        departure=AirportDetails(iata=dep_iata, airport="Trivandrum International Airport"),
        arrival=AirportDetails(iata=arr_iata, airport="Dubai International Airport"),
    )


def mock_aviation_client(flights: List[NormalizedFlight] = None) -> AsyncMock:
    """Create a mock AviationStackClient returning specified flights."""
    client = AsyncMock()
    data = flights if flights is not None else [create_sample_flight()]
    client.get_flights.return_value = AviationStackResponse(
        success=True,
        count=len(data),
        data=data,
    )
    return client


# ---------------------------------------------------------------------------
# Test 34: Valid Flight Number
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_flight_number() -> None:
    """Verify flight number query invokes AviationStack and updates state results."""
    client = mock_aviation_client([create_sample_flight("EK522")])

    state: TravelState = create_initial_state(
        user_query="Check EK522",
        session_id="sess-fnum",
        request_id="req-fnum",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "success"
    assert len(update["flight_results"]) == 1
    assert update["flight_results"][0]["flight_iata"] == "EK522"
    assert update["flight_query"]["flight_iata"] == "EK522"
    assert update["active_agent"] == "flight_agent"
    assert "flight_agent" in update["completed_agents"]
    client.get_flights.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 35: Route Search with Location Resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_search_with_location_resolution() -> None:
    """Verify route search resolves natural city names (Trivandrum -> TRV, Dubai -> DXB)."""
    client = mock_aviation_client([create_sample_flight(dep_iata="TRV", arr_iata="DXB")])

    state: TravelState = create_initial_state(
        user_query="Find flights from Trivandrum to Dubai",
        session_id="sess-route",
        request_id="req-route",
    )
    state["intent"] = "flight"
    state["origin"] = "Trivandrum"
    state["destination"] = "Dubai"
    state["departure_date"] = "2026-10-01"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "success"
    assert update["flight_query"]["dep_iata"] == "TRV"
    assert update["flight_query"]["arr_iata"] == "DXB"
    assert update["flight_query"]["flight_date"] == "2026-10-01"
    client.get_flights.assert_awaited_once_with(
        flight_iata=None,
        flight_number=None,
        dep_iata="TRV",
        arr_iata="DXB",
        flight_date="2026-10-01",
    )


# ---------------------------------------------------------------------------
# Test 36: Route Search with Existing Codes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_search_with_existing_codes() -> None:
    """Verify origin='TRV' and destination='DXB' are directly used without re-resolving."""
    client = mock_aviation_client([create_sample_flight(dep_iata="TRV", arr_iata="DXB")])

    state: TravelState = create_initial_state(
        user_query="Flights from TRV to DXB",
        session_id="sess-codes",
        request_id="req-codes",
    )
    state["intent"] = "flight"
    state["origin"] = "TRV"
    state["destination"] = "DXB"

    # Mock resolver to verify it is bypassed for standard 3-letter IATA codes
    mock_resolver = MagicMock()

    update = await run_flight_agent(
        state,
        aviation_client=client,
        location_resolver=mock_resolver,
    )

    assert update["flight_status"] == "success"
    assert update["flight_query"]["dep_iata"] == "TRV"
    assert update["flight_query"]["arr_iata"] == "DXB"
    assert not mock_resolver.resolve.called
    client.get_flights.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 37: Unresolved Origin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresolved_origin() -> None:
    """Verify unresolved origin returns error state without calling AviationStack."""
    client = AsyncMock()

    state: TravelState = create_initial_state(
        user_query="Flights from NonexistentCityXYZ to Dubai",
        session_id="sess-unresolved-orig",
        request_id="req-unresolved-orig",
    )
    state["intent"] = "flight"
    state["origin"] = "NonexistentCityXYZ"
    state["destination"] = "Dubai"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert not client.get_flights.called
    assert any("could not be resolved (not_found)" in err for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 38: Unresolved Destination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresolved_destination() -> None:
    """Verify unresolved destination returns error state without calling AviationStack."""
    client = AsyncMock()

    state: TravelState = create_initial_state(
        user_query="Flights from Trivandrum to NonexistentCityABC",
        session_id="sess-unresolved-dest",
        request_id="req-unresolved-dest",
    )
    state["intent"] = "flight"
    state["origin"] = "Trivandrum"
    state["destination"] = "NonexistentCityABC"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert not client.get_flights.called
    assert any("destination 'NonexistentCityABC' could not be resolved (not_found)" in err for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 39: Ambiguous Location
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ambiguous_location() -> None:
    """Verify ambiguous locations (e.g. Springfield) do not arbitrarily pick an airport."""
    client = AsyncMock()

    state: TravelState = create_initial_state(
        user_query="Flights from Springfield to Dubai",
        session_id="sess-ambig",
        request_id="req-ambig",
    )
    state["intent"] = "flight"
    state["origin"] = "Springfield"
    state["destination"] = "Dubai"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert not client.get_flights.called
    assert any("origin 'Springfield' is ambiguous and matches multiple airports" in err for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 40: Missing Flight Information
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_flight_information() -> None:
    """Verify insufficient information returns a clear error without querying AviationStack."""
    client = AsyncMock()

    state: TravelState = create_initial_state(
        user_query="Check my flight",
        session_id="sess-missing",
        request_id="req-missing",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert not client.get_flights.called
    assert any("insufficient flight information" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 41: AviationStack No Results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aviationstack_no_results() -> None:
    """Verify zero matching records sets flight_status='no_results' without raising an exception."""
    client = mock_aviation_client([])

    state: TravelState = create_initial_state(
        user_query="Check flight AI9999",
        session_id="sess-nores",
        request_id="req-nores",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "no_results"
    assert update["flight_results"] == []
    assert "errors" not in update


# ---------------------------------------------------------------------------
# Test 42: AviationStack Provider Failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aviationstack_failure() -> None:
    """Verify provider API errors set flight_status='error' and preserve error details."""
    client = AsyncMock()
    client.get_flights.side_effect = AviationStackAPIError("usage_limit_reached", "Account quota reached")

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-apifail",
        request_id="req-apifail",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert update["flight_results"] == []
    assert any("usage_limit_reached" in err for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 43: Timeout and Network Failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout() -> None:
    """Verify request timeout is caught and logged cleanly into state errors."""
    client = AsyncMock()
    client.get_flights.side_effect = AviationStackTimeoutError("AviationStack request timed out after 10s")

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-timeout",
        request_id="req-timeout",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert any("timed out" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 44: State Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_preservation() -> None:
    """Verify unrelated state fields (session_id, request_id, user_query, metadata) remain intact."""
    client = mock_aviation_client()

    original_state: TravelState = {
        "user_query": "Check flight EK522",
        "session_id": "session-unique-789",
        "request_id": "request-unique-012",
        "intent": "flight",
        "research_results": [{"title": "Visa Info", "url": "https://example.com/visa"}],
        "metadata": {"tier": "platinum", "loyalty_id": "EM-999"},
        "workflow_status": "running",
        "errors": [],
    }

    update = await run_flight_agent(original_state, aviation_client=client)

    # Agent should only return keys it updates
    assert "research_results" not in update
    assert "session_id" not in update
    assert "request_id" not in update

    merged = {**original_state, **update}
    assert merged["session_id"] == "session-unique-789"
    assert merged["request_id"] == "request-unique-012"
    assert merged["research_results"] == [{"title": "Visa Info", "url": "https://example.com/visa"}]
    assert merged["metadata"]["loyalty_id"] == "EM-999"
    assert merged["flight_status"] == "success"


# ---------------------------------------------------------------------------
# Test 45: Active and Completed Agent Tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_completed_agent_tracking() -> None:
    """Verify active_agent and completed_agents are updated appropriately."""
    client = mock_aviation_client()

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-agents",
        request_id="req-agents",
    )
    state["intent"] = "flight"
    state["completed_agents"] = ["router_agent"]

    update = await run_flight_agent(state, aviation_client=client)

    assert update["active_agent"] == "flight_agent"
    assert update["completed_agents"] == ["router_agent", "flight_agent"]


# ---------------------------------------------------------------------------
# Test 46: No OpenAI Call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_openai_call() -> None:
    """Verify the Flight Agent does not import or invoke OpenAI / ChatOpenAI."""
    client = mock_aviation_client()

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-no-llm",
        request_id="req-no-llm",
    )
    state["intent"] = "flight"

    with patch("app.core.llm.get_chat_model") as mock_get_chat:
        update = await run_flight_agent(state, aviation_client=client)
        assert update["flight_status"] == "success"
        assert not mock_get_chat.called


# ---------------------------------------------------------------------------
# Test 47: No Tavily Call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tavily_call() -> None:
    """Verify the Flight Agent does not invoke Tavily search operations."""
    client = mock_aviation_client()

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-no-tavily",
        request_id="req-no-tavily",
    )
    state["intent"] = "flight"

    with patch("app.tools.tavily_search.TavilySearchClient") as mock_tavily:
        update = await run_flight_agent(state, aviation_client=client)
        assert update["flight_status"] == "success"
        assert not mock_tavily.called


# ---------------------------------------------------------------------------
# Test 48: JSON Serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serialization() -> None:
    """Verify flight update dictionary is completely JSON-serializable."""
    client = mock_aviation_client()

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-json",
        request_id="req-json",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    # Must serialize without TypeError
    serialized = json.dumps(update)
    reconstructed = json.loads(serialized)
    assert reconstructed["flight_status"] == "success"
    assert reconstructed["flight_results"][0]["flight_iata"] == "EK522"


# ---------------------------------------------------------------------------
# Test 49: Repeated Execution Safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_execution() -> None:
    """Verify executing Flight Agent repeatedly does not duplicate completed_agents entries."""
    client = mock_aviation_client()

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-repeat",
        request_id="req-repeat",
    )
    state["intent"] = "flight"

    # First run
    update1 = await run_flight_agent(state, aviation_client=client)
    state1 = {**state, **update1}
    assert state1["completed_agents"] == ["flight_agent"]

    # Second run against updated state
    update2 = await run_flight_agent(state1, aviation_client=client)
    state2 = {**state1, **update2}
    assert state2["completed_agents"] == ["flight_agent"]
    assert state2["completed_agents"].count("flight_agent") == 1
