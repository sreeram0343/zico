import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.state import ZicoGraphState, TripSegment, SegmentType, Location
from app.graph.engine import (
    input_node,
    flight_search_worker_node,
    research_worker_node,
    supervisor_router,
    validator_node
)
from app.tools.flight_search import search_flights

# 1. Test state cleanup isolation
def test_input_node_clears_flight_search_results():
    """Ensure previous request flight results do not leak into a new request."""
    state = {
        "messages": [HumanMessage(content="Hello")],
        "flight_search_results": {"status": "RESULTS", "flights": [{"id": "stale"}]}
    }
    result = input_node(state)
    assert result["flight_search_results"] == {}

# 2. Test flight isolation constraint (malformed request)
def test_flight_search_worker_requires_both_origin_and_destination():
    """Ensure the worker asks for clarification rather than hallucinating origins."""
    state = {"messages": [HumanMessage(content="Find me flights to London")]}
    result = flight_search_worker_node(state)
    assert "messages" in result
    msg = result["messages"][0].content
    assert "departing from" in msg.lower() or "departure city" in msg.lower()
    # It should not have attempted to set flight_search_results
    assert "flight_search_results" not in result

# 3. Test flight isolation constraint (no results)
@patch("app.graph.engine.search_flights")
def test_flight_search_worker_produces_no_results_structure(mock_search):
    """Ensure empty search results return strict NO_RESULTS state."""
    mock_search.invoke.return_value = []
    state = {"messages": [HumanMessage(content="Find me flights from Mumbai to Delhi on 2025-05-20")]}
    result = flight_search_worker_node(state)
    assert result["flight_search_results"]["status"] == "NO_RESULTS"
    assert result["flight_search_results"]["flights"] == []

# 4. Test flight isolation constraint (results)
@patch("app.graph.engine.search_flights")
def test_flight_search_worker_produces_results_structure(mock_search):
    """Ensure valid search results are properly structured for the frontend."""
    from datetime import datetime, timedelta
    now = datetime.now()
    mock_seg = TripSegment(
        id="f1",
        type=SegmentType.FLIGHT,
        title="Air India (AI-123)",
        start_time=now,
        end_time=now + timedelta(hours=2),
        location=Location(name="Delhi", iata_code="DEL"),
        cost=150.0,
        currency="USD",
        metadata={"airline": "Air India", "flight_number": "AI-123", "departure_iata": "BOM", "arrival_iata": "DEL"}
    )
    mock_search.invoke.return_value = [mock_seg]
    
    state = {"messages": [HumanMessage(content="Find me flights from Mumbai to Delhi on 2025-05-20")]}
    result = flight_search_worker_node(state)
    
    fsr = result["flight_search_results"]
    assert fsr["status"] == "RESULTS"
    assert len(fsr["flights"]) == 1
    assert fsr["flights"][0]["airline"] == "Air India"
    assert fsr["flights"][0]["flightNumber"] == "AI-123"

# 5. Test hotel research isolation
def test_research_worker_does_not_touch_flight_results():
    """Ensure hotel searches do not mutate or contaminate flight states."""
    state = {"messages": [HumanMessage(content="Find me hotels in Pune")], "itinerary": []}
    result = research_worker_node(state)
    # Research node should NOT return flight_search_results key
    assert "flight_search_results" not in result
    assert "messages" in result

# 6. Test supervisor routing to flight search
def test_supervisor_routes_to_flight_search():
    """Supervisor must route flight queries correctly."""
    # Since supervisor uses next_node output from supervisor_node in the actual graph,
    # we just test the router function here given a next_node state.
    state = {"next_node": "flight_search_worker"}
    assert supervisor_router(state) == "flight_search_worker"

# 7. Test supervisor routing to research
def test_supervisor_routes_to_research_worker():
    """Supervisor must route research queries correctly."""
    state = {"next_node": "research_worker"}
    assert supervisor_router(state) == "research_worker"

# 8. Test supervisor routing to disruption
def test_supervisor_routes_to_disruption_worker():
    """Supervisor must route disruption queries correctly."""
    state = {"next_node": "disruption_worker"}
    assert supervisor_router(state) == "disruption_worker"

# 9. Test validator conflict detection
def test_validator_node_identifies_conflicts():
    """Validator must strictly catch overlapping itinerary items."""
    from datetime import datetime, timedelta
    now = datetime.now()
    seg1 = TripSegment(
        id="f1", type=SegmentType.FLIGHT, title="AI1", start_time=now, end_time=now + timedelta(hours=2),
        location=Location(name="A", iata_code="A"), cost=100
    )
    seg2 = TripSegment(
        id="f2", type=SegmentType.FLIGHT, title="AI2", start_time=now + timedelta(hours=1), end_time=now + timedelta(hours=3),
        location=Location(name="B", iata_code="B"), cost=100
    )
    state = {"itinerary": [seg1, seg2], "constraints": {}, "active_disruptions": []}
    result = validator_node(state)
    
    assert "active_disruptions" in result
    conflict = [d for d in result["active_disruptions"] if d.get("type") == "ITINERARY_CONFLICT"]
    assert len(conflict) == 1

# 10. Test high-impact action approval routing
def test_supervisor_routes_to_booking_approval():
    """High impact pending actions must force routing to approval node."""
    state = {
        "pending_actions": [
            {"action_type": "BOOKING", "status": "PENDING", "requires_explicit_approval": True}
        ]
    }
    assert supervisor_router(state) == "booking_approval_node"
