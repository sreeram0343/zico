from datetime import datetime, timedelta
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage
import pytest
from app.graph.engine import (
    build_zico_graph,
    create_zico_graph,
    disruption_worker_node,
    flight_search_worker_node,
    policy_rag_worker_node,
    supervisor_router,
)
from app.graph.state import (
    ActionStatus,
    ActionType,
    Location,
    SegmentType,
    TripConstraints,
    TripSegment,
    ZicoGraphState,
)


@pytest.fixture
def sample_state():
    base_time = datetime(2026, 9, 10, 8, 0)
    seg1 = TripSegment(
        id="f_01",
        type=SegmentType.FLIGHT,
        title="Flight to Rome",
        start_time=base_time,
        end_time=base_time + timedelta(hours=3),
        location=Location(name="Rome FCO", iata_code="FCO"),
        cost=320.0,
    )
    return {
        "messages": [HumanMessage(content="Find flights from JFK to LHR")],
        "trip_id": "trip_test_worker",
        "user_id": "user_worker_01",
        "itinerary": [seg1],
        "constraints": TripConstraints(max_budget=1000.0),
        "pending_actions": [],
        "active_disruptions": [],
        "next_node": None,
    }


def test_flight_search_worker_node_execution(sample_state):
    """Verify flight search worker invokes search and returns AIMessage with flight options."""
    mock_flight_data = {
        "best_flights": [
            {
                "flights": [
                    {
                        "airline": "British Airways",
                        "flight_number": "BA 112",
                        "departure_airport": {"name": "JFK", "id": "JFK", "time": "2026-09-10 18:30"},
                        "arrival_airport": {"name": "LHR", "id": "LHR", "time": "2026-09-11 06:30"},
                        "duration": 420,
                    }
                ],
                "total_duration": 420,
                "price": 540.0,
            }
        ]
    }

    with patch("app.tools.flight_search.client.search", return_value=mock_flight_data):
        result = flight_search_worker_node(sample_state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert "British Airways" in msg.content
        assert "540" in msg.content


def test_policy_rag_worker_node_execution(sample_state):
    """Verify policy RAG worker queries vector store and returns grounded guidance."""
    state = dict(sample_state)
    state["messages"] = [HumanMessage(content="What is the EU 261 flight cancellation compensation?")]

    result = policy_rag_worker_node(state)
    assert "messages" in result
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert "EU Regulation 261" in msg.content or "Policy" in msg.content


def test_disruption_worker_node_execution(sample_state):
    """Verify disruption worker detects delay impact and formulates PendingAction."""
    result = disruption_worker_node(sample_state)
    assert "pending_actions" in result
    assert len(result["pending_actions"]) >= 1
    action = result["pending_actions"][0]
    assert action.requires_explicit_approval is True
    assert action.status == ActionStatus.PENDING

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)


def test_supervisor_router_routing_logic():
    """Verify supervisor router matches valid destinations or defaults to validator_node."""
    assert supervisor_router({"next_node": "flight_search_worker"}) == "flight_search_worker"
    assert supervisor_router({"next_node": "policy_rag_worker"}) == "policy_rag_worker"
    assert supervisor_router({"next_node": "disruption_worker"}) == "disruption_worker"
    assert supervisor_router({"next_node": "validator_node"}) == "validator_node"
    assert supervisor_router({"next_node": "UNKNOWN_NODE"}) == "validator_node"


def test_end_to_end_graph_with_worker_routing(sample_state):
    """Verify full graph compilation and execution routing through worker branch."""
    with patch("app.graph.engine.supervisor_node", return_value={"next_node": "policy_rag_worker"}):
        graph = build_zico_graph().compile()
        config = {"configurable": {"thread_id": "thread_e2e_worker_test"}}

        result = graph.invoke(sample_state, config=config)
        assert result["trip_id"] == sample_state["trip_id"]
        # Output should have user message + worker AIMessage
        assert len(result["messages"]) >= 2
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1


