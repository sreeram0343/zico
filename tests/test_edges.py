"""
Comprehensive offline test suite for ZICO LangGraph Conditional Edge Routing.

Validates:
    - Test 25: Intent 'flight' routes to 'flight'
    - Test 26: Intent 'research' routes to 'research'
    - Test 27: Intent 'general_travel' routes to 'research'
    - Test 28: Intent 'unsupported' routes to 'response'
    - Test 29: Invalid/unknown intents fallback to 'response'
    - Test 30: Missing intent key fallback to 'response'
    - Test 31: Empty string intent fallback to 'response'
    - Test 32: Whitespace-only intent fallback to 'response'
    - Test 33: Case and whitespace normalization
    - Test 34: State immutability and complete preservation
    - Test 35: Deterministic routing across multiple iterations
    - Test 36: Zero external tool or API dependencies
    - Test 37: Valid return domain constrained strictly to allowed nodes
    - Test 38: Immutability of public ROUTE_MAP
    - Test 39: No graph compilation or execution upon importing module
    - Test 43: Architectural compliance (no LLM calls, tool execution, or StateGraph construction)
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest

from app.core.state import TravelState, create_initial_state
from app.graph.edges import (
    DEFAULT_FALLBACK_NODE,
    ROUTE_MAP,
    route_after_router,
)

# ---------------------------------------------------------------------------
# Fixture: Base TravelState
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state() -> TravelState:
    """Provide a realistic TravelState fixture."""
    state = create_initial_state(
        user_query="Flight EK522 status from DXB to TRV",
        session_id="sess-edge-01",
        request_id="req-edge-01",
    )
    state["origin"] = "DXB"
    state["destination"] = "TRV"
    state["metadata"] = {"tier": "platinum", "client": "web"}
    return state


# ---------------------------------------------------------------------------
# Test 25: Flight Intent
# ---------------------------------------------------------------------------


def test_route_flight_intent(base_state: TravelState) -> None:
    """Verify intent='flight' routes directly to the 'flight' node."""
    base_state["intent"] = "flight"
    assert route_after_router(base_state) == "flight"


# ---------------------------------------------------------------------------
# Test 26: Research Intent
# ---------------------------------------------------------------------------


def test_route_research_intent(base_state: TravelState) -> None:
    """Verify intent='research' routes directly to the 'research' node."""
    base_state["intent"] = "research"
    assert route_after_router(base_state) == "research"


# ---------------------------------------------------------------------------
# Test 27: General Travel Intent
# ---------------------------------------------------------------------------


def test_route_general_travel_intent(base_state: TravelState) -> None:
    """Verify intent='general_travel' routes to the 'research' node for MVP."""
    base_state["intent"] = "general_travel"
    assert route_after_router(base_state) == "research"


# ---------------------------------------------------------------------------
# Test 28: Unsupported Intent
# ---------------------------------------------------------------------------


def test_route_unsupported_intent(base_state: TravelState) -> None:
    """Verify intent='unsupported' routes to the 'response' node for explanation."""
    base_state["intent"] = "unsupported"
    assert route_after_router(base_state) == "response"


# ---------------------------------------------------------------------------
# Test 29: Invalid / Unknown Intents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_intent",
    [
        "hotel",
        "weather",
        "booking",
        "car_rental",
        "medical_advice",
        "something_random",
    ],
)
def test_route_invalid_intents_fallback(base_state: TravelState, invalid_intent: str) -> None:
    """Verify unmapped or unknown intents route to the safe fallback 'response' node."""
    base_state["intent"] = invalid_intent
    assert route_after_router(base_state) == "response"


# ---------------------------------------------------------------------------
# Test 30: Missing Intent Key
# ---------------------------------------------------------------------------


def test_route_missing_intent_key(base_state: TravelState) -> None:
    """Verify state with completely absent 'intent' key routes to safe fallback without error."""
    if "intent" in base_state:
        del base_state["intent"]
    assert route_after_router(base_state) == DEFAULT_FALLBACK_NODE
    assert route_after_router(base_state) == "response"


# ---------------------------------------------------------------------------
# Test 31: Empty String Intent
# ---------------------------------------------------------------------------


def test_route_empty_intent_string(base_state: TravelState) -> None:
    """Verify intent='' routes to safe fallback 'response' node."""
    base_state["intent"] = ""
    assert route_after_router(base_state) == "response"


# ---------------------------------------------------------------------------
# Test 32: Whitespace-Only Intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws", [" ", "   ", "\t", "\n", " \t \n "])
def test_route_whitespace_intent(base_state: TravelState, ws: str) -> None:
    """Verify whitespace-only intents route to safe fallback 'response' node."""
    base_state["intent"] = ws
    assert route_after_router(base_state) == "response"


# ---------------------------------------------------------------------------
# Test 33: Case and Whitespace Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_intent,expected_node",
    [
        ("FLIGHT", "flight"),
        ("Flight", "flight"),
        ("  flight  ", "flight"),
        ("RESEARCH", "research"),
        ("Research", "research"),
        ("  research  ", "research"),
        ("GENERAL_TRAVEL", "research"),
        ("General_Travel", "research"),
        ("  general_travel  ", "research"),
        ("UNSUPPORTED", "response"),
        ("Unsupported", "response"),
        ("  unsupported  ", "response"),
    ],
)
def test_route_normalization(base_state: TravelState, raw_intent: str, expected_node: str) -> None:
    """Verify case and leading/trailing whitespace are defensively normalized."""
    base_state["intent"] = raw_intent
    assert route_after_router(base_state) == expected_node


# ---------------------------------------------------------------------------
# Test 34: State Preservation and Immutability
# ---------------------------------------------------------------------------


def test_state_remains_completely_unmutated(base_state: TravelState) -> None:
    """Verify route_after_router does not add, remove, or modify any state fields."""
    base_state["intent"] = "flight"
    original_state_snapshot = copy.deepcopy(dict(base_state))

    result = route_after_router(base_state)

    assert result == "flight"
    assert dict(base_state) == original_state_snapshot


# ---------------------------------------------------------------------------
# Test 35: Deterministic Routing
# ---------------------------------------------------------------------------


def test_deterministic_routing_iterations(base_state: TravelState) -> None:
    """Verify routing produces identical output across multiple successive calls."""
    base_state["intent"] = "flight"
    results = [route_after_router(base_state) for _ in range(20)]
    assert all(r == "flight" for r in results)

    base_state["intent"] = "general_travel"
    results_gt = [route_after_router(base_state) for _ in range(20)]
    assert all(r == "research" for r in results_gt)


# ---------------------------------------------------------------------------
# Test 36 & 37: Valid Return Domain & Pure Execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_intent",
    [
        "flight",
        "research",
        "general_travel",
        "unsupported",
        "invalid_test",
        "",
        None,
    ],
)
def test_valid_return_domain(base_state: TravelState, test_intent: Any) -> None:
    """Verify all outputs strictly belong to the allowed set of registered graph nodes."""
    allowed_nodes = {"flight", "research", "response"}
    base_state["intent"] = test_intent
    destination_node = route_after_router(base_state)
    assert destination_node in allowed_nodes


# ---------------------------------------------------------------------------
# Test 38: Immutability of ROUTE_MAP
# ---------------------------------------------------------------------------


def test_route_map_is_immutable() -> None:
    """Verify public ROUTE_MAP cannot be modified at runtime."""
    with pytest.raises(TypeError):
        ROUTE_MAP["flight"] = "corrupted_node"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Test 39: No Graph Construction on Import
# ---------------------------------------------------------------------------


def test_no_graph_compilation_on_import() -> None:
    """Verify importing app.graph.edges does not construct or compile StateGraph."""
    import app.graph.edges as edges_mod

    assert hasattr(edges_mod, "route_after_router")
    assert callable(edges_mod.route_after_router)


# ---------------------------------------------------------------------------
# Test 43: Architectural Compliance
# ---------------------------------------------------------------------------


def test_edges_file_architecture_compliance() -> None:
    """
    Verify app.graph.edges strictly functions as a decision layer and does NOT:
        - import or instantiate StateGraph / compile
        - call LLMs (OpenAI / ChatOpenAI)
        - execute external tools (AviationStack, Tavily, requests, httpx)
    """
    import app.graph.edges as edges_mod

    source = inspect.getsource(edges_mod)

    # Must NOT construct StateGraph
    prohibited_graph_tokens = [
        "StateGraph",
        "add_node",
        "add_edge",
        "add_conditional_edges",
        ".compile(",
    ]
    for token in prohibited_graph_tokens:
        assert token not in source, (
            f"app.graph.edges must not contain graph building token: {token!r}"
        )

    # Must NOT execute LLMs or tools
    prohibited_tool_tokens = [
        "ChatOpenAI",
        "OpenAI(",
        "get_chat_model",
        "AviationStack",
        "Tavily",
        "httpx",
        "requests.get",
        "requests.post",
    ]
    for token in prohibited_tool_tokens:
        assert token not in source, (
            f"app.graph.edges must not contain tool/LLM execution token: {token!r}"
        )
