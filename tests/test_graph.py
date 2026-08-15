"""
Unit tests for LangGraph state machine construction and execution.
"""

from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from app.graph.engine import create_zico_graph
from app.graph.state import Location, SegmentType, TripConstraints, TripSegment


def test_graph_compilation():
    """Verify that the state graph compiles without error."""
    app = create_zico_graph()
    assert app is not None


def test_graph_single_step_execution():
    """Verify complete execution flow through input_node, supervisor_node, and validator_node."""
    checkpointer = MemorySaver()
    app = create_zico_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "thread_test_01"}}

    base_time = datetime(2026, 9, 1, 10, 0)
    loc = Location(name="Paris CDG", iata_code="CDG")
    valid_segment = TripSegment(
        id="seg_01",
        type=SegmentType.FLIGHT,
        title="Flight to Paris",
        start_time=base_time,
        end_time=base_time + timedelta(hours=2),
        location=loc,
        cost=250.0,
    )

    initial_input = {
        "messages": [HumanMessage(content="Hello ZICO, validate my itinerary")],
        "trip_id": "trip_001",
        "user_id": "user_abc",
        "itinerary": [valid_segment],
        "constraints": TripConstraints(max_budget=500.0),
    }

    result = app.invoke(initial_input, config=config)

    assert result["trip_id"] == "trip_001"
    assert result["user_id"] == "user_abc"
    assert len(result["messages"]) == 1
    assert result["next_node"] == "validator_node"
    assert len(result["active_disruptions"]) == 0


def test_graph_validation_disruption_detection():
    """Verify that graph execution captures conflicts into active_disruptions."""
    checkpointer = MemorySaver()
    app = create_zico_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "thread_conflict_01"}}

    base_time = datetime(2026, 9, 1, 10, 0)
    loc = Location(name="JFK Airport", iata_code="JFK")

    # 2 overlapping flights
    seg1 = TripSegment(
        id="f1",
        type=SegmentType.FLIGHT,
        title="Flight 1",
        start_time=base_time,
        end_time=base_time + timedelta(hours=4),
        location=loc,
        cost=300.0,
    )
    seg2 = TripSegment(
        id="f2",
        type=SegmentType.FLIGHT,
        title="Flight 2",
        start_time=base_time + timedelta(hours=3),  # 1 hour overlap
        end_time=base_time + timedelta(hours=6),
        location=loc,
        cost=300.0,
    )

    input_data = {
        "messages": [HumanMessage(content="Check my overlapping flights")],
        "trip_id": "trip_002",
        "user_id": "user_xyz",
        "itinerary": [seg1, seg2],
        "constraints": TripConstraints(max_budget=500.0),  # Total cost 600 > 500
    }

    result = app.invoke(input_data, config=config)

    assert len(result["active_disruptions"]) >= 2
    types = [d["type"] for d in result["active_disruptions"]]
    assert "ITINERARY_CONFLICT" in types
    assert "BUDGET_EXCEEDED" in types


def test_graph_checkpoint_state_retrieval():
    """Verify that checkpoint state can be retrieved by thread_id after execution."""
    checkpointer = MemorySaver()
    app = create_zico_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "thread_state_retrieval"}}

    app.invoke(
        {
            "messages": [HumanMessage(content="Save my state")],
            "trip_id": "trip_state_123",
            "user_id": "user_456",
            "itinerary": [],
        },
        config=config,
    )

    state_snapshot = app.get_state(config)
    assert state_snapshot is not None
    assert state_snapshot.values["trip_id"] == "trip_state_123"
    assert state_snapshot.values["user_id"] == "user_456"
    assert len(state_snapshot.values["messages"]) == 1
