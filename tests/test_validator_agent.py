"""
Comprehensive offline test suite for ZICO Workflow Validation Agent.

Validates:
    - Test 36: Valid flight state passes
    - Test 37: Flight no-results state passes (zero matching flights is valid)
    - Test 38: Flight provider failure fails validation while preserving tool errors
    - Test 39: Valid research state passes
    - Test 40: Research no-results state passes (successful search with 0 matches is valid)
    - Test 41: Malformed research result fails validation (missing/invalid URL)
    - Test 42: Invalid intent rejection (e.g. 'hotel') without mutating intent
    - Test 43: Missing or whitespace user_query reporting
    - Test 44: Invalid date relationship (return_date < departure_date)
    - Test 45: Invalid passenger count (passengers <= 0)
    - Test 46: Contradictory route for flight intent (origin == destination)
    - Test 47: State preservation of unrelated fields
    - Test 48: Existing tool/runtime errors preserved independently
    - Test 49: Validation error deduplication on repeated execution
    - Test 50: Deterministic validation results and error ordering
    - Test 51: Unsupported intent recognition without rewriting
    - Test 52: General travel intent passes without requiring specialized agent outputs
    - Test 53: Zero external network or tool calls (AviationStack, Tavily, OpenAI, Location)
    - Test 54: JSON serialization compatibility
    - Test 55: Active and completed agent tracking idempotency
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.agents.validator_agent import (
    ValidatorAgent,
    run_validator_agent,
)
from app.core.state import TravelState, create_initial_state

# ---------------------------------------------------------------------------
# Test 36: Valid Flight State
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_flight_state() -> None:
    """Verify realistic successful flight state passes validation with 0 errors."""
    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-val-01",
        request_id="req-val-01",
    )
    state["intent"] = "flight"
    state["origin"] = "DXB"
    state["destination"] = "TRV"
    state["flight_status"] = "success"
    state["flight_results"] = [
        {
            "flight_iata": "EK522",
            "airline_name": "Emirates",
            "flight_status": "scheduled",
            "departure": {"iata": "DXB", "scheduled": "2026-10-01T21:45:00"},
            "arrival": {"iata": "TRV", "scheduled": "2026-10-02T03:20:00"},
        }
    ]

    update = await run_validator_agent(state)

    assert update["validation_status"] == "passed"
    assert update["validation_errors"] == []
    assert update["active_agent"] == "validator_agent"
    assert "validator_agent" in update["completed_agents"]


# ---------------------------------------------------------------------------
# Test 37: Flight No-Results State
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flight_no_results_state() -> None:
    """Verify successful flight query returning 0 records is not treated as a validation failure."""
    state: TravelState = create_initial_state(
        user_query="Find flights from TRV to DXB on 2026-12-25",
        session_id="sess-val-02",
        request_id="req-val-02",
    )
    state["intent"] = "flight"
    state["origin"] = "TRV"
    state["destination"] = "DXB"
    state["flight_status"] = "no_results"
    state["flight_results"] = []

    update = await run_validator_agent(state)

    assert update["validation_status"] == "passed"
    assert update["validation_errors"] == []


# ---------------------------------------------------------------------------
# Test 38: Flight Provider Failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flight_provider_failure() -> None:
    """Verify provider error fails validation while preserving original execution errors."""
    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-val-03",
        request_id="req-val-03",
    )
    state["intent"] = "flight"
    state["flight_status"] = "error"
    state["flight_results"] = []
    state["errors"] = ["AviationStack provider timeout after 10000ms"]

    update = await run_validator_agent(state)

    assert update["validation_status"] == "failed"
    assert any("flight operation" in err.lower() for err in update["validation_errors"])

    # Merged state preserves original errors
    merged = {**state, **update}
    assert merged["errors"] == ["AviationStack provider timeout after 10000ms"]


# ---------------------------------------------------------------------------
# Test 39: Valid Research State
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_research_state() -> None:
    """Verify research state with valid titles, URLs, and content passes validation."""
    state: TravelState = create_initial_state(
        user_query="What are the current visa rules for Germany?",
        session_id="sess-val-04",
        request_id="req-val-04",
    )
    state["intent"] = "research"
    state["research_query"] = "Germany visa rules current official"
    state["research_status"] = "success"
    state["research_results"] = [
        {
            "title": "German Federal Foreign Office - Visa Information",
            "url": "https://www.auswaertiges-amt.de/en/visa-service",
            "content": "Official requirements for short stay Schengen visas.",
            "score": 0.98,
        },
        {
            "title": "Schengen Visa Requirements Guide",
            "url": "https://schengenvisainfo.com/germany-visa/",
            "content": "Step-by-step documentation guide.",
            "score": 0.89,
        },
    ]

    update = await run_validator_agent(state)

    assert update["validation_status"] == "passed"
    assert update["validation_errors"] == []


# ---------------------------------------------------------------------------
# Test 40: Research No-Results State
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_no_results_state() -> None:
    """Verify successful research query returning zero matches passes validation."""
    state: TravelState = create_initial_state(
        user_query="Extremely obscure local transit decree in 1840",
        session_id="sess-val-05",
        request_id="req-val-05",
    )
    state["intent"] = "research"
    state["research_status"] = "no_results"
    state["research_results"] = []

    update = await run_validator_agent(state)

    assert update["validation_status"] == "passed"
    assert update["validation_errors"] == []


# ---------------------------------------------------------------------------
# Test 41: Malformed Research Result (Missing or Invalid URL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_research_result() -> None:
    """Verify research results missing source URLs or containing invalid URLs fail validation."""
    # Case A: Missing URL key entirely
    state_a: TravelState = create_initial_state(
        user_query="Baggage allowance for Emirates",
        session_id="sess-val-06a",
        request_id="req-val-06a",
    )
    state_a["intent"] = "research"
    state_a["research_status"] = "success"
    state_a["research_results"] = [
        {
            "title": "Emirates Baggage Allowance",
            "content": "Passengers may check up to 30kg in Economy.",
            # URL missing
        }
    ]

    update_a = await run_validator_agent(state_a)
    assert update_a["validation_status"] == "failed"
    assert any("source url" in err.lower() for err in update_a["validation_errors"])

    # Case B: Malformed / non-HTTP URL
    state_b: TravelState = create_initial_state(
        user_query="Baggage allowance for Emirates",
        session_id="sess-val-06b",
        request_id="req-val-06b",
    )
    state_b["intent"] = "research"
    state_b["research_status"] = "success"
    state_b["research_results"] = [
        {
            "title": "Emirates Baggage Allowance",
            "url": "not-a-valid-url",
            "content": "Passengers may check up to 30kg in Economy.",
        }
    ]

    update_b = await run_validator_agent(state_b)
    assert update_b["validation_status"] == "failed"
    assert any("source url" in err.lower() for err in update_b["validation_errors"])


# ---------------------------------------------------------------------------
# Test 42: Invalid Intent Rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_intent() -> None:
    """Verify unrecognized or missing intent fails validation without mutating intent."""
    state: TravelState = create_initial_state(
        user_query="Book a 5-star hotel in Dubai",
        session_id="sess-val-07",
        request_id="req-val-07",
    )
    state["intent"] = "hotel"  # Unsupported intent

    update = await run_validator_agent(state)

    assert update["validation_status"] == "failed"
    assert any(
        "invalid or missing intent 'hotel'" in err.lower() for err in update["validation_errors"]
    )
    # Must NOT overwrite or replace intent in update
    assert "intent" not in update


# ---------------------------------------------------------------------------
# Test 43: Missing User Query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_user_query() -> None:
    """Verify missing or whitespace-only user_query fails validation without guessing."""
    for empty_query in ["", "   ", "\t\n"]:
        state: TravelState = create_initial_state(
            user_query=empty_query,
            session_id="sess-val-08",
            request_id="req-val-08",
        )
        state["intent"] = "general_travel"

        update = await run_validator_agent(state)

        assert update["validation_status"] == "failed"
        assert any(
            "missing or empty user_query" in err.lower() for err in update["validation_errors"]
        )


# ---------------------------------------------------------------------------
# Test 44: Invalid Date Relationship
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_date_relationship() -> None:
    """Verify return_date earlier than departure_date fails validation."""
    state: TravelState = create_initial_state(
        user_query="Flight from London to New York",
        session_id="sess-val-09",
        request_id="req-val-09",
    )
    state["intent"] = "flight"
    state["flight_status"] = "success"
    state["departure_date"] = "2026-11-20"
    state["return_date"] = "2026-11-10"  # Earlier than departure

    update = await run_validator_agent(state)

    assert update["validation_status"] == "failed"
    assert any("earlier than departure_date" in err.lower() for err in update["validation_errors"])


# ---------------------------------------------------------------------------
# Test 45: Invalid Passenger Count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_passenger_count() -> None:
    """Verify passenger count <= 0 or invalid type fails validation."""
    for invalid_passengers in [0, -2]:
        state: TravelState = create_initial_state(
            user_query="Find flights for 0 passengers",
            session_id="sess-val-10",
            request_id="req-val-10",
        )
        state["intent"] = "general_travel"
        state["passengers"] = invalid_passengers

        update = await run_validator_agent(state)

        assert update["validation_status"] == "failed"
        assert any(
            "passengers must be an integer >= 1" in err.lower()
            for err in update["validation_errors"]
        )


# ---------------------------------------------------------------------------
# Test 46: Contradictory Route for Flight Intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contradictory_route() -> None:
    """Verify origin == destination for flight operations triggers validation failure."""
    state: TravelState = create_initial_state(
        user_query="Flight from DXB to DXB",
        session_id="sess-val-11",
        request_id="req-val-11",
    )
    state["intent"] = "flight"
    state["origin"] = "DXB"
    state["destination"] = "DXB"
    state["flight_status"] = "success"

    update = await run_validator_agent(state)

    assert update["validation_status"] == "failed"
    assert any("cannot be identical" in err.lower() for err in update["validation_errors"])


# ---------------------------------------------------------------------------
# Test 47: State Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_preservation() -> None:
    """Verify all unrelated fields in TravelState remain completely untouched."""
    original_state: TravelState = {
        "user_query": "Check flight EK522",
        "session_id": "sess-preserve-123",
        "request_id": "req-preserve-456",
        "intent": "flight",
        "origin": "DXB",
        "destination": "TRV",
        "departure_date": "2026-10-01",
        "return_date": "2026-10-15",
        "passengers": 2,
        "flight_status": "success",
        "flight_results": [{"flight_iata": "EK522"}],
        "research_results": [{"title": "Visa info"}],
        "metadata": {"user_id": "cust-888", "tier": "platinum"},
        "workflow_status": "running",
        "errors": ["Minor non-fatal network warning"],
    }

    update = await run_validator_agent(original_state)

    # Must NOT return unrelated keys
    assert "user_query" not in update
    assert "session_id" not in update
    assert "request_id" not in update
    assert "metadata" not in update
    assert "flight_results" not in update

    merged = {**original_state, **update}
    assert merged["session_id"] == "sess-preserve-123"
    assert merged["request_id"] == "req-preserve-456"
    assert merged["metadata"]["tier"] == "platinum"
    assert merged["validation_status"] == "passed"


# ---------------------------------------------------------------------------
# Test 48: Existing Errors Preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_errors_preserved() -> None:
    """Verify execution/tool errors in state['errors'] are kept strictly intact."""
    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-val-errors",
        request_id="req-val-errors",
    )
    state["intent"] = "flight"
    state["flight_status"] = "error"
    state["errors"] = ["AviationStack provider HTTP 500"]

    update = await run_validator_agent(state)

    # Validator should not clear or replace state['errors']
    assert "errors" not in update
    merged = {**state, **update}
    assert merged["errors"] == ["AviationStack provider HTTP 500"]
    assert len(merged["validation_errors"]) > 0


# ---------------------------------------------------------------------------
# Test 49: Validation Error Deduplication on Repeated Execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_error_deduplication() -> None:
    """Verify running validator multiple times on invalid state does not duplicate errors."""
    state: TravelState = create_initial_state(
        user_query="",
        session_id="sess-val-repeat",
        request_id="req-val-repeat",
    )
    state["intent"] = "hotel"

    # Run 1
    update1 = await run_validator_agent(state)
    state1 = {**state, **update1}
    count1 = len(state1["validation_errors"])

    # Run 2
    update2 = await run_validator_agent(state1)
    state2 = {**state1, **update2}
    count2 = len(state2["validation_errors"])

    assert count1 == count2
    assert len(state2["validation_errors"]) == len(set(state2["validation_errors"]))


# ---------------------------------------------------------------------------
# Test 50: Deterministic Validation Results and Ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_results() -> None:
    """Verify repeated runs on unchanged state yield identical status and stably ordered errors."""
    state: TravelState = create_initial_state(
        user_query="",
        session_id="sess-val-deter",
        request_id="req-val-deter",
    )
    state["intent"] = "unknown_category"
    state["passengers"] = -1

    agent = ValidatorAgent()
    res1 = agent.validate(state)
    res2 = agent.validate(state)

    assert res1["validation_status"] == res2["validation_status"]
    assert res1["validation_errors"] == res2["validation_errors"]


# ---------------------------------------------------------------------------
# Test 51: Unsupported Intent Recognition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_request() -> None:
    """Verify unsupported intent is recognized as valid domain classification without mutating."""
    state: TravelState = create_initial_state(
        user_query="Write Python code for quicksort",
        session_id="sess-val-unsupported",
        request_id="req-val-unsupported",
    )
    state["intent"] = "unsupported"

    update = await run_validator_agent(state)

    assert update["validation_status"] == "passed"
    assert update["validation_errors"] == []
    assert "intent" not in update


# ---------------------------------------------------------------------------
# Test 52: General Travel Intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_general_travel_request() -> None:
    """Verify general_travel intent passes without requiring flight or research outputs."""
    state: TravelState = create_initial_state(
        user_query="What should I pack for a 7-day trip to Paris?",
        session_id="sess-val-general",
        request_id="req-val-general",
    )
    state["intent"] = "general_travel"
    state["flight_results"] = []
    state["research_results"] = []

    update = await run_validator_agent(state)

    assert update["validation_status"] == "passed"
    assert update["validation_errors"] == []


# ---------------------------------------------------------------------------
# Test 53: Zero External Network / Tool Calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_external_tool_usage() -> None:
    """Verify validator executes without calling AviationStack, Tavily, OpenAI, or Location."""
    state: TravelState = create_initial_state(
        user_query="Check flight EK522",
        session_id="sess-val-no-tools",
        request_id="req-val-no-tools",
    )
    state["intent"] = "flight"
    state["flight_status"] = "success"
    state["flight_results"] = [{"flight_iata": "EK522"}]

    with (
        patch("app.tools.aviationstack.AviationStackClient") as mock_aviation,
        patch("app.tools.tavily_search.TavilySearchClient") as mock_tavily,
        patch("app.tools.location.LocationResolver") as mock_location,
        patch("app.core.llm.get_chat_model") as mock_llm,
    ):
        update = await run_validator_agent(state)

        assert update["validation_status"] == "passed"
        assert not mock_aviation.called
        assert not mock_tavily.called
        assert not mock_location.called
        assert not mock_llm.called


# ---------------------------------------------------------------------------
# Test 54: JSON Serialization Compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serialization() -> None:
    """Verify validator update dictionary is completely JSON-serializable."""
    state: TravelState = create_initial_state(
        user_query="Find flights",
        session_id="sess-val-json",
        request_id="req-val-json",
    )
    state["intent"] = "flight"
    state["flight_status"] = "success"
    state["flight_results"] = [{"flight_iata": "EK522"}]

    update = await run_validator_agent(state)

    serialized = json.dumps(update)
    reconstructed = json.loads(serialized)
    assert reconstructed["validation_status"] == "passed"
    assert reconstructed["validation_errors"] == []
    assert reconstructed["active_agent"] == "validator_agent"


# ---------------------------------------------------------------------------
# Test 55: Active and Completed Agent Tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_completed_agent_tracking() -> None:
    """Verify active_agent and completed_agents are updated idempotently."""
    state: TravelState = create_initial_state(
        user_query="Plan trip to Tokyo",
        session_id="sess-val-track",
        request_id="req-val-track",
    )
    state["intent"] = "general_travel"
    state["completed_agents"] = ["router_agent"]

    update = await run_validator_agent(state)

    assert update["active_agent"] == "validator_agent"
    assert update["completed_agents"] == ["router_agent", "validator_agent"]

    # Run again to ensure no duplicate entries
    state2 = {**state, **update}
    update2 = await run_validator_agent(state2)
    assert update2["completed_agents"] == ["router_agent", "validator_agent"]
    assert update2["completed_agents"].count("validator_agent") == 1
