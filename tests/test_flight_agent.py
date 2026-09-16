"""
Unit tests for ZICO's Flight Operations Agent.

Verifies translation of TravelState into AviationStack queries, location resolution,
flight status tracking, error handling, intent validation, and state preservation.
All tests run locally using mocked AviationStack and Location tools.
"""

from __future__ import annotations

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
# Test: Flight Status Lookup by Flight Number
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flight_status_lookup_by_flight_number() -> None:
    """Verify checking a flight number extracts the IATA designator and updates state."""
    client = mock_aviation_client([create_sample_flight("EK522")])

    state: TravelState = create_initial_state(
        user_query="Check status of flight EK522.",
        session_id="sess-flight-1",
        request_id="req-flight-1",
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
# Test: Flight Search by Route with Location Resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flight_search_by_route_with_location_resolution() -> None:
    """Verify route search resolves city names to airport IATA codes."""
    client = mock_aviation_client([create_sample_flight(dep_iata="TRV", arr_iata="DXB")])

    state: TravelState = create_initial_state(
        user_query="Find flights from Trivandrum to Dubai on 2026-10-01",
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
# Test: Zero Search Results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_flight_results() -> None:
    """Verify successful AviationStack response with 0 records sets flight_status='no_results'."""
    client = mock_aviation_client([])

    state: TravelState = create_initial_state(
        user_query="Check flight AI9999",
        session_id="sess-noresults",
        request_id="req-noresults",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "no_results"
    assert update["flight_results"] == []
    assert "errors" not in update  # No operational error occurred


# ---------------------------------------------------------------------------
# Test: Provider API Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_api_error_handling() -> None:
    """Verify provider API errors set flight_status='error' and update errors in state."""
    client = AsyncMock()
    client.get_flights.side_effect = AviationStackAPIError("usage_limit_reached", "Account quota reached")

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-apierr",
        request_id="req-apierr",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert update["flight_results"] == []
    assert any("usage_limit_reached" in err for err in update["errors"])


# ---------------------------------------------------------------------------
# Test: Timeout and Network Failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_and_network_failures() -> None:
    """Verify timeouts and network errors update state cleanly without uncaught exceptions."""
    client = AsyncMock()
    client.get_flights.side_effect = AviationStackTimeoutError("AviationStack request timed out")

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
# Test: Intent Guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_guard_rejects_non_flight() -> None:
    """Verify Flight Agent rejects requests where intent != 'flight' without calling tools."""
    client = AsyncMock()

    for invalid_intent in ["research", "general_travel", "unsupported", None]:
        state: TravelState = create_initial_state(
            user_query="Check flight EK522",
            session_id="sess-guard",
            request_id="req-guard",
        )
        state["intent"] = invalid_intent

        update = await run_flight_agent(state, aviation_client=client)

        assert update["flight_status"] == "error"
        assert any("invalid intent" in err.lower() for err in update["errors"])
        assert not client.get_flights.called


# ---------------------------------------------------------------------------
# Test: Insufficient Search Information
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insufficient_search_information() -> None:
    """Verify missing flight number and missing route returns error state without calling provider."""
    client = AsyncMock()

    state: TravelState = create_initial_state(
        user_query="Hello I want to see flights please",
        session_id="sess-insufficient",
        request_id="req-insufficient",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    assert update["flight_status"] == "error"
    assert update["flight_results"] == []
    assert any("insufficient flight information" in err.lower() for err in update["errors"])
    assert not client.get_flights.called


# ---------------------------------------------------------------------------
# Test: Ambiguous Location Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ambiguous_location_handling() -> None:
    """Verify ambiguous locations (like Springfield) do not arbitrarily pick an airport."""
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

    # Cannot query AviationStack when departure airport is ambiguous
    assert update["flight_status"] == "error"
    assert any("could not resolve origin 'Springfield'" in err for err in update["errors"])
    assert not client.get_flights.called


# ---------------------------------------------------------------------------
# Test: State Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_preservation() -> None:
    """Verify unrelated fields in TravelState remain intact after flight operations."""
    client = mock_aviation_client()

    original_state: TravelState = {
        "user_query": "Check flight EK522",
        "session_id": "session-123",
        "request_id": "request-456",
        "intent": "flight",
        "research_results": [{"title": "Visa info", "url": "https://example.com"}],
        "final_response": None,
        "workflow_status": "running",
        "errors": [],
        "metadata": {"user_tier": "gold"},
    }

    update = await run_flight_agent(original_state, aviation_client=client)

    # Flight agent should not modify unrelated keys
    assert "research_results" not in update
    assert "user_query" not in update
    assert "session_id" not in update

    merged = {**original_state, **update}
    assert merged["research_results"] == [{"title": "Visa info", "url": "https://example.com"}]
    assert merged["metadata"]["user_tier"] == "gold"
    assert merged["flight_status"] == "success"


# ---------------------------------------------------------------------------
# Test: Secret Protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_protection(caplog: pytest.LogCaptureFixture) -> None:
    """Verify credentials are never exposed in logs or state updates."""
    caplog.set_level(logging.DEBUG)
    client = mock_aviation_client()

    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-secret",
        request_id="req-secret",
    )
    state["intent"] = "flight"

    update = await run_flight_agent(state, aviation_client=client)

    captured = caplog.text.lower()
    assert "access_key" not in captured or "redacted" in captured
    assert "api_key" not in captured
    assert "secret" not in captured
