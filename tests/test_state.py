"""
Unit tests for ZICO shared workflow state module (app.core.state).

Covers:
- Test 1: Minimal state initialization and defaults
- Test 2: Travel information representation
- Test 3: Flight results storage
- Test 4: Research results and source metadata
- Test 5: Validation errors handling
- Test 6: Workflow errors vs validation errors separation
- Test 7: State isolation and mutable default safety
- Test 8: Partial state updates for LangGraph node execution
"""

import pytest
from app.core.state import (
    FlightResult,
    ResearchResult,
    TravelState,
    create_initial_state,
)


def test_minimal_state():
    """Test 1 — Minimal state: creation and default values."""
    state = create_initial_state(
        user_query="Find flights to London",
        session_id="session_abc_123",
        request_id="req_xyz_789",
    )

    # Required information exists and matches
    assert state["user_query"] == "Find flights to London"
    assert state["session_id"] == "session_abc_123"
    assert state["request_id"] == "req_xyz_789"

    # Optional travel fields are empty/None
    assert state["origin"] is None
    assert state["destination"] is None
    assert state["departure_date"] is None
    assert state["return_date"] is None
    assert state["passengers"] is None

    # Intent fields are unpopulated
    assert state["intent"] is None
    assert state["intent_confidence"] is None

    # Operations are in their initial state
    assert state["flight_status"] == "not_requested"
    assert state["flight_results"] == []
    assert state["research_status"] == "not_requested"
    assert state["research_results"] == []

    # Validation and workflow status
    assert state["validation_status"] == "pending"
    assert state["validation_errors"] == []
    assert state["workflow_status"] == "initialized"
    assert state["errors"] == []
    assert state["final_response"] is None
    assert state["sources"] == []
    assert state["metadata"] == {}


def test_travel_information():
    """Test 2 — Travel information: extraction and representation."""
    state = create_initial_state(
        user_query="Book flight from Trivandrum to Dubai tomorrow for 2 people",
        session_id="session_1",
        request_id="req_1",
    )

    state["origin"] = "Trivandrum"
    state["destination"] = "Dubai"
    state["departure_date"] = "2026-09-17"
    state["return_date"] = "2026-09-24"
    state["passengers"] = 2

    assert state["origin"] == "Trivandrum"
    assert state["destination"] == "Dubai"
    assert state["departure_date"] == "2026-09-17"
    assert state["return_date"] == "2026-09-24"
    assert state["passengers"] == 2


def test_flight_results():
    """Test 3 — Flight results: storing multiple normalized results."""
    state = create_initial_state(
        user_query="Flights to Paris",
        session_id="session_flight",
        request_id="req_flight",
    )

    flight_1: FlightResult = {
        "airline": "Air France",
        "flight_number": "AF123",
        "origin": "JFK",
        "destination": "CDG",
        "departure_time": "2026-10-01T18:00:00Z",
        "arrival_time": "2026-10-02T07:30:00Z",
        "price": 650.00,
        "currency": "USD",
        "metadata": {"aircraft": "Boeing 777"},
    }

    flight_2: FlightResult = {
        "airline": "Delta",
        "flight_number": "DL456",
        "origin": "JFK",
        "destination": "CDG",
        "departure_time": "2026-10-01T21:00:00Z",
        "arrival_time": "2026-10-02T10:15:00Z",
        "price": 720.50,
        "currency": "USD",
        "metadata": {"aircraft": "Airbus A350"},
    }

    state["flight_query"] = {"origin": "JFK", "destination": "CDG", "date": "2026-10-01"}
    state["flight_results"] = [flight_1, flight_2]
    state["flight_status"] = "success"

    assert len(state["flight_results"]) == 2
    assert state["flight_results"][0]["flight_number"] == "AF123"
    assert state["flight_results"][1]["price"] == 720.50
    assert state["flight_status"] == "success"
    assert state["flight_query"]["destination"] == "CDG"


def test_research_results():
    """Test 4 — Research results: preserving source metadata."""
    state = create_initial_state(
        user_query="What are the baggage rules for Emirates?",
        session_id="session_res",
        request_id="req_res",
    )

    res_item: ResearchResult = {
        "title": "Emirates Baggage Allowance & Guidelines",
        "url": "https://www.emirates.com/english/before-you-fly/baggage/",
        "content": "Economy Class passengers are permitted up to 30kg of checked baggage.",
        "score": 0.94,
        "metadata": {
            "source_type": "official_airline_policy",
            "retrieved_at": "2026-09-16T10:00:00Z",
            "provider": "tavily",
        },
    }

    state["research_query"] = "Emirates baggage allowance policy"
    state["research_results"].append(res_item)
    state["research_status"] = "success"

    assert len(state["research_results"]) == 1
    retrieved = state["research_results"][0]
    assert retrieved["title"] == "Emirates Baggage Allowance & Guidelines"
    assert retrieved["url"] == "https://www.emirates.com/english/before-you-fly/baggage/"
    assert retrieved["score"] == 0.94
    assert retrieved["metadata"]["provider"] == "tavily"
    assert state["research_status"] == "success"


def test_validation_errors():
    """Test 5 — Validation errors: representing multiple validation errors."""
    state = create_initial_state(
        user_query="Book flight",
        session_id="session_val",
        request_id="req_val",
    )

    state["validation_status"] = "failed"
    state["validation_errors"].append("Departure date cannot be in the past.")
    state["validation_errors"].append("Return date must be strictly after departure date.")
    state["validation_errors"].append("Passenger count must be at least 1.")

    assert state["validation_status"] == "failed"
    assert len(state["validation_errors"]) == 3
    assert "Departure date cannot be in the past." in state["validation_errors"]
    assert "Return date must be strictly after departure date." in state["validation_errors"]
    assert "Passenger count must be at least 1." in state["validation_errors"]


def test_workflow_errors_separated_from_validation():
    """Test 6 — Workflow errors vs validation errors: strictly separated."""
    state = create_initial_state(
        user_query="Check flights",
        session_id="session_err",
        request_id="req_err",
    )

    # Add tool / external service error to errors
    state["errors"].append("Aviationstack API timeout after 5000ms")
    state["workflow_status"] = "failed"

    # Validation errors must remain empty and untouched
    assert len(state["errors"]) == 1
    assert "Aviationstack API timeout" in state["errors"][0]
    assert len(state["validation_errors"]) == 0
    assert state["validation_status"] == "pending"

    # Now simulate a business validation error occurring separately
    state["validation_errors"].append("Requested route TRV->DXB has no available seats.")
    state["validation_status"] = "failed"

    # Both error lists exist independently
    assert len(state["errors"]) == 1
    assert len(state["validation_errors"]) == 1
    assert state["errors"][0] != state["validation_errors"][0]


def test_state_isolation():
    """Test 7 — State isolation: modifying one state does not affect another."""
    state_a = create_initial_state(
        user_query="Query A",
        session_id="session_A",
        request_id="req_A",
        metadata={"client": "web"},
    )
    state_b = create_initial_state(
        user_query="Query B",
        session_id="session_B",
        request_id="req_B",
        metadata={"client": "mobile"},
    )

    # Mutate all collection fields on state_a
    state_a["flight_results"].append({"flight": "EK500"})
    state_a["research_results"].append({"title": "Dubai Guide"})
    state_a["completed_agents"].append("intent_agent")
    state_a["agent_messages"].append({"role": "system", "msg": "started"})
    state_a["validation_errors"].append("Validation error on A")
    state_a["errors"].append("Tool error on A")
    state_a["sources"].append({"url": "https://example.com"})
    state_a["metadata"]["custom_flag"] = True

    # Verify state_b collections remain completely empty / unaltered
    assert len(state_b["flight_results"]) == 0
    assert len(state_b["research_results"]) == 0
    assert len(state_b["completed_agents"]) == 0
    assert len(state_b["agent_messages"]) == 0
    assert len(state_b["validation_errors"]) == 0
    assert len(state_b["errors"]) == 0
    assert len(state_b["sources"]) == 0
    assert "custom_flag" not in state_b["metadata"]
    assert state_b["metadata"]["client"] == "mobile"


def test_partial_updates():
    """Test 8 — Partial updates: LangGraph nodes can return and merge partial state."""
    state = create_initial_state(
        user_query="Flights to London next week",
        session_id="session_update",
        request_id="req_update",
    )

    # Node 1: Intent detection node output
    intent_update: TravelState = {
        "intent": "flight",
        "intent_confidence": 0.97,
        "active_agent": "intent_agent",
        "completed_agents": ["intent_agent"],
    }
    state.update(intent_update)

    # Verify updated fields and un-updated fields
    assert state["intent"] == "flight"
    assert state["intent_confidence"] == 0.97
    assert state["active_agent"] == "intent_agent"
    assert state["completed_agents"] == ["intent_agent"]
    # Original fields preserved
    assert state["user_query"] == "Flights to London next week"
    assert state["workflow_status"] == "initialized"
    assert state["origin"] is None

    # Node 2: Information extraction node output
    extraction_update: TravelState = {
        "destination": "London",
        "departure_date": "2026-09-23",
        "active_agent": "extraction_agent",
        "completed_agents": state["completed_agents"] + ["extraction_agent"],
    }
    state.update(extraction_update)

    assert state["destination"] == "London"
    assert state["departure_date"] == "2026-09-23"
    assert state["intent"] == "flight"  # Prior node data preserved
    assert state["active_agent"] == "extraction_agent"
    assert state["completed_agents"] == ["intent_agent", "extraction_agent"]