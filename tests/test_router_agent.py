"""
Unit tests for ZICO's Intent Router Agent.

Verifies deterministic classification of user requests into operational intents
(flight, research, general_travel, unsupported), structured output handling,
error resiliency, state preservation, and zero unintended tool execution.
All tests run locally with mocked models and zero network/provider access.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from app.agents.router_agent import (
    RouterAgent,
    RouterDecision,
    aroute_request,
    route_request,
)
from app.core.state import TravelState, create_initial_state


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


class MockStructuredModel:
    """Mock chat model simulating ChatOpenAI with structured output."""

    def __init__(self, return_decision: Any, side_effect: Any = None) -> None:
        self.return_decision = return_decision
        self.side_effect = side_effect
        self.invoke_mock = MagicMock()
        self.ainvoke_mock = MagicMock()

    def with_structured_output(self, schema: Any) -> "MockStructuredModel":
        self.schema = schema
        return self

    def invoke(self, messages: Any) -> Any:
        self.invoke_mock(messages)
        if self.side_effect:
            raise self.side_effect
        return self.return_decision

    async def ainvoke(self, messages: Any) -> Any:
        self.ainvoke_mock(messages)
        if self.side_effect:
            raise self.side_effect
        return self.return_decision


# ---------------------------------------------------------------------------
# Test 26: Flight Intent Routing
# ---------------------------------------------------------------------------


def test_flight_intent_routing() -> None:
    """Verify flight queries route to 'flight' without calling AviationStack."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="flight", confidence=1.0)
    )

    state: TravelState = create_initial_state(
        user_query="Check the status of flight EK522.",
        session_id="sess-001",
        request_id="req-001",
    )

    update = route_request(state, llm=mock_model)

    assert update["intent"] == "flight"
    assert update["intent_confidence"] == 1.0
    assert update["active_agent"] == "router_agent"
    assert mock_model.invoke_mock.called


# ---------------------------------------------------------------------------
# Test 27: Research Intent Routing
# ---------------------------------------------------------------------------


def test_research_intent_routing() -> None:
    """Verify research queries route to 'research' without calling Tavily."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="research", confidence=0.95)
    )

    state: TravelState = create_initial_state(
        user_query="What are the current visa requirements for Germany?",
        session_id="sess-002",
        request_id="req-002",
    )

    update = route_request(state, llm=mock_model)

    assert update["intent"] == "research"
    assert update["intent_confidence"] == 0.95
    assert update["active_agent"] == "router_agent"


# ---------------------------------------------------------------------------
# Test 28: General Travel Intent Routing
# ---------------------------------------------------------------------------


def test_general_travel_intent_routing() -> None:
    """Verify broad travel planning queries route to 'general_travel'."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="general_travel", confidence=0.9)
    )

    state: TravelState = create_initial_state(
        user_query="Help me plan a seven-day trip to Dubai.",
        session_id="sess-003",
        request_id="req-003",
    )

    update = route_request(state, llm=mock_model)

    assert update["intent"] == "general_travel"
    assert update["intent_confidence"] == 0.9
    assert update["active_agent"] == "router_agent"


# ---------------------------------------------------------------------------
# Test 29: Unsupported Intent Routing
# ---------------------------------------------------------------------------


def test_unsupported_intent_routing() -> None:
    """Verify non-travel queries route to 'unsupported'."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="unsupported", confidence=1.0)
    )

    state: TravelState = create_initial_state(
        user_query="Explain how to implement a compiler in C++.",
        session_id="sess-004",
        request_id="req-004",
    )

    update = route_request(state, llm=mock_model)

    assert update["intent"] == "unsupported"
    assert update["active_agent"] == "router_agent"


# ---------------------------------------------------------------------------
# Test 30: Empty Query Handling
# ---------------------------------------------------------------------------


def test_empty_query_handling() -> None:
    """Verify empty and whitespace-only queries are rejected without calling OpenAI."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="flight", confidence=1.0)
    )

    for empty_query in ["", "   ", "\t\n  "]:
        state: TravelState = create_initial_state(
            user_query=empty_query,
            session_id="sess-empty",
            request_id="req-empty",
        )

        update = route_request(state, llm=mock_model)

        # Model must NOT be called
        assert not mock_model.invoke_mock.called
        assert update["intent"] == "unsupported"
        assert update["intent_confidence"] is None
        assert update["active_agent"] == "router_agent"


# ---------------------------------------------------------------------------
# Test 31: Invalid Model Intent Fallback
# ---------------------------------------------------------------------------


def test_invalid_model_intent_fallback() -> None:
    """Verify unsupported intents like 'hotel' do not enter state and fall back safely."""
    # Simulate a raw dictionary returned with an invalid intent category
    mock_model = MockStructuredModel(
        return_decision={"intent": "hotel", "confidence": 0.8}
    )

    state: TravelState = create_initial_state(
        user_query="Book a 5 star hotel in Paris.",
        session_id="sess-hotel",
        request_id="req-hotel",
    )

    update = route_request(state, llm=mock_model)

    # Must NOT place "hotel" into the state
    assert update["intent"] == "unsupported"
    assert update["intent"] != "hotel"
    assert "errors" in update
    assert any("hotel" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 32: LLM Provider Failure Handling
# ---------------------------------------------------------------------------


def test_model_failure_handling() -> None:
    """Verify OpenAI exceptions do not crash routing and update state errors."""
    mock_model = MockStructuredModel(
        return_decision=None,
        side_effect=RuntimeError("OpenAI API rate limit exceeded"),
    )

    state: TravelState = create_initial_state(
        user_query="Check flight EK522.",
        session_id="sess-err",
        request_id="req-err",
    )

    update = route_request(state, llm=mock_model)

    # Must not invent a flight intent upon failure
    assert update["intent"] == "unsupported"
    assert update["intent_confidence"] is None
    assert "errors" in update
    assert any("rate limit" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 33: Malformed Structured Response Handling
# ---------------------------------------------------------------------------


def test_malformed_structured_response() -> None:
    """Verify invalid or unparseable responses fail safely without silent corruption."""
    for bad_response in [None, "Not a valid model decision", 12345, []]:
        mock_model = MockStructuredModel(return_decision=bad_response)

        state: TravelState = create_initial_state(
            user_query="What is the weather in Tokyo?",
            session_id="sess-malformed",
            request_id="req-malformed",
        )

        update = route_request(state, llm=mock_model)

        assert update["intent"] == "unsupported"
        assert "errors" in update


# ---------------------------------------------------------------------------
# Test 34: State Preservation
# ---------------------------------------------------------------------------


def test_state_preservation() -> None:
    """Verify unrelated fields in TravelState remain intact after routing."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="flight", confidence=1.0)
    )

    original_state: TravelState = {
        "user_query": "Check status of flight EK522.",
        "session_id": "session-xyz",
        "request_id": "request-abc",
        "origin": "TRV",
        "destination": "DXB",
        "departure_date": "2026-10-01",
        "passengers": 2,
        "workflow_status": "initialized",
        "errors": [],
        "metadata": {"custom_tag": "vip"},
    }

    # Execute router
    update = route_request(original_state, llm=mock_model)

    # Router must only update its own fields
    assert "origin" not in update
    assert "destination" not in update
    assert "session_id" not in update
    assert "request_id" not in update

    # Merge update into state as LangGraph would
    merged_state = {**original_state, **update}
    assert merged_state["intent"] == "flight"
    assert merged_state["origin"] == "TRV"
    assert merged_state["destination"] == "DXB"
    assert merged_state["session_id"] == "session-xyz"
    assert merged_state["request_id"] == "request-abc"
    assert merged_state["passengers"] == 2
    assert merged_state["metadata"]["custom_tag"] == "vip"


# ---------------------------------------------------------------------------
# Test 35: No Unintended Tool Execution
# ---------------------------------------------------------------------------


def test_no_tool_execution() -> None:
    """Verify the router never invokes AviationStack, Tavily, or Location tools."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="flight", confidence=1.0)
    )

    state: TravelState = create_initial_state(
        user_query="Check flight EK522.",
        session_id="sess-notools",
        request_id="req-notools",
    )

    with patch("app.tools.aviationstack.AviationStackClient") as mock_aviation, \
         patch("app.tools.tavily_search.TavilySearchClient") as mock_tavily, \
         patch("app.tools.location.LocationResolver") as mock_location:

        update = route_request(state, llm=mock_model)

        assert update["intent"] == "flight"
        assert not mock_aviation.called
        assert not mock_tavily.called
        assert not mock_location.called


# ---------------------------------------------------------------------------
# Test 36: Structured Output Construction
# ---------------------------------------------------------------------------


def test_structured_output_construction() -> None:
    """Verify with_structured_output is called with RouterDecision schema."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="research", confidence=1.0)
    )

    agent = RouterAgent(model=mock_model)
    state: TravelState = create_initial_state(
        user_query="Visa requirements for Japan?",
        session_id="sess-schema",
        request_id="req-schema",
    )

    agent.route(state)

    assert hasattr(mock_model, "schema")
    assert mock_model.schema == RouterDecision


# ---------------------------------------------------------------------------
# Test 37: Asynchronous Routing (LangGraph Async Compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_router_execution() -> None:
    """Verify asynchronous routing execution via aroute_request."""
    mock_model = MockStructuredModel(
        return_decision=RouterDecision(intent="research", confidence=0.98)
    )

    state: TravelState = create_initial_state(
        user_query="Current travel restrictions for UAE?",
        session_id="sess-async",
        request_id="req-async",
    )

    update = await aroute_request(state, llm=mock_model)

    assert update["intent"] == "research"
    assert update["intent_confidence"] == 0.98
    assert update["active_agent"] == "router_agent"
    assert mock_model.ainvoke_mock.called
