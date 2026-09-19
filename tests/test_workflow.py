"""
Comprehensive offline test suite for ZICO LangGraph Executable Workflow.

Validates:
    - Test 32: Workflow construction and compilation
    - Test 33: All core nodes exist in compiled graph topology
    - Test 34: Router -> Flight -> Validator -> Response -> END execution path
    - Test 35: Router -> Research -> Validator -> Response -> END execution path
    - Test 36: Router -> General Travel -> Research -> Validator -> Response -> END
    - Test 37: Router -> Unsupported -> Response -> END (bypasses tools & validator)
    - Test 38: Invalid intent fallback to Response -> END
    - Test 39: Missing intent fallback to Response -> END
    - Test 40: Validation invariant (always executes before response on specialized paths)
    - Test 41: Unsupported bypass invariant (validator never called for unsupported)
    - Test 42: Response node always terminates to END
    - Test 43: Progressive state update propagation through graph
    - Test 44: Initial state preservation across complete execution
    - Test 45: Node exception propagation through workflow
    - Test 46: Zero external tool or LLM calls during graph construction
    - Test 47: Concurrent invocation safety and request state isolation
    - Test 48: Workflow repeatability (independent fresh graph builds)
    - Test 49: No duplicate node registration on successive builds
    - Test 50: Clean, safe import of workflow module without side effects
    - Test 54: Architectural compliance (no LLM instantiation or tool execution)
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from app.core.state import TravelState, create_initial_state
from app.graph.workflow import (
    build_workflow,
    create_workflow,
    run_workflow,
)

# ---------------------------------------------------------------------------
# Fixture: Base TravelState
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state() -> TravelState:
    """Create a realistic TravelState fixture."""
    state = create_initial_state(
        user_query="Status of flight EK522 from DXB to TRV",
        session_id="session-wf-01",
        request_id="req-wf-01",
    )
    state["origin"] = "DXB"
    state["destination"] = "TRV"
    state["metadata"] = {"channel": "test_suite", "env": "offline"}
    return state


# ---------------------------------------------------------------------------
# Test 32 & 33: Workflow Construction and Node Verification
# ---------------------------------------------------------------------------


def test_workflow_construction_and_compilation() -> None:
    """Verify build_workflow constructs and compiles a valid CompiledStateGraph."""
    graph = build_workflow()
    assert isinstance(graph, CompiledStateGraph)
    assert graph is not None


def test_compiled_workflow_contains_expected_nodes() -> None:
    """Verify the compiled graph contains all 5 operational nodes."""
    graph = build_workflow()
    registered_nodes = set(graph.nodes.keys())

    expected_nodes = {"router", "flight", "research", "validator", "response"}
    assert expected_nodes.issubset(registered_nodes)


# ---------------------------------------------------------------------------
# Test 34: Router -> Flight Path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_flight_path_execution(base_state: TravelState) -> None:
    """Verify intent='flight' routes through router -> flight -> validator -> response -> END."""
    call_order: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        call_order.append("router")
        return {"intent": "flight", "active_agent": "router_agent"}

    async def mock_flight(state: TravelState) -> Dict[str, Any]:
        call_order.append("flight")
        return {
            "flight_status": "success",
            "flight_results": [{"flight_iata": "EK522", "flight_status": "scheduled"}],
            "active_agent": "flight_agent",
        }

    async def mock_research(state: TravelState) -> Dict[str, Any]:
        call_order.append("research")
        return {"research_status": "success"}

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        call_order.append("validator")
        return {"validation_status": "passed", "active_agent": "validator_agent"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        call_order.append("response")
        return {"final_response": "Flight EK522 is scheduled.", "active_agent": "response_agent"}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight),
        patch("app.graph.workflow.research_node", side_effect=mock_research),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        final_state = await run_workflow(base_state, graph=graph)

        assert call_order == ["router", "flight", "validator", "response"]
        assert "research" not in call_order
        assert final_state["intent"] == "flight"
        assert final_state["flight_status"] == "success"
        assert final_state["validation_status"] == "passed"
        assert "EK522" in final_state["final_response"]


# ---------------------------------------------------------------------------
# Test 35: Router -> Research Path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_research_path_execution(base_state: TravelState) -> None:
    """Verify intent='research' routes through router -> research -> validator -> response -> END."""
    call_order: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        call_order.append("router")
        return {"intent": "research", "active_agent": "router_agent"}

    async def mock_flight(state: TravelState) -> Dict[str, Any]:
        call_order.append("flight")
        return {"flight_status": "success"}

    async def mock_research(state: TravelState) -> Dict[str, Any]:
        call_order.append("research")
        return {
            "research_status": "success",
            "research_results": [
                {
                    "title": "Visa Info",
                    "url": "https://example.com",
                    "content": "Passport required.",
                }
            ],
            "active_agent": "research_agent",
        }

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        call_order.append("validator")
        return {"validation_status": "passed", "active_agent": "validator_agent"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        call_order.append("response")
        return {"final_response": "Passport required for entry.", "active_agent": "response_agent"}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight),
        patch("app.graph.workflow.research_node", side_effect=mock_research),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        final_state = await run_workflow(base_state, graph=graph)

        assert call_order == ["router", "research", "validator", "response"]
        assert "flight" not in call_order
        assert final_state["intent"] == "research"
        assert final_state["research_status"] == "success"
        assert final_state["validation_status"] == "passed"


# ---------------------------------------------------------------------------
# Test 36: Router -> General Travel Path (routes to research)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_general_travel_routes_to_research(base_state: TravelState) -> None:
    """Verify intent='general_travel' routes to research -> validator -> response."""
    call_order: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        call_order.append("router")
        return {"intent": "general_travel", "active_agent": "router_agent"}

    async def mock_research(state: TravelState) -> Dict[str, Any]:
        call_order.append("research")
        return {"research_status": "success", "active_agent": "research_agent"}

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        call_order.append("validator")
        return {"validation_status": "passed", "active_agent": "validator_agent"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        call_order.append("response")
        return {
            "final_response": "Here are general travel suggestions.",
            "active_agent": "response_agent",
        }

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.research_node", side_effect=mock_research),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        final_state = await run_workflow(base_state, graph=graph)

        assert call_order == ["router", "research", "validator", "response"]
        assert final_state["intent"] == "general_travel"


# ---------------------------------------------------------------------------
# Test 37: Router -> Unsupported Path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_unsupported_bypasses_tools_and_validator(base_state: TravelState) -> None:
    """Verify intent='unsupported' routes directly router -> response -> END (bypasses validator)."""
    call_order: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        call_order.append("router")
        return {"intent": "unsupported", "active_agent": "router_agent"}

    async def mock_flight(state: TravelState) -> Dict[str, Any]:
        call_order.append("flight")
        return {}

    async def mock_research(state: TravelState) -> Dict[str, Any]:
        call_order.append("research")
        return {}

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        call_order.append("validator")
        return {}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        call_order.append("response")
        return {
            "final_response": "This request is outside ZICO's travel scope.",
            "active_agent": "response_agent",
        }

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight),
        patch("app.graph.workflow.research_node", side_effect=mock_research),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        final_state = await run_workflow(base_state, graph=graph)

        assert call_order == ["router", "response"]
        assert "flight" not in call_order
        assert "research" not in call_order
        assert "validator" not in call_order
        assert "outside ZICO's travel scope" in final_state["final_response"]


# ---------------------------------------------------------------------------
# Test 38 & 39: Invalid and Missing Intent Fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_invalid_intent_fallback(base_state: TravelState) -> None:
    """Verify invalid intent ('hotel') routes directly to response via edge fallback."""
    call_order: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        call_order.append("router")
        return {"intent": "hotel"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        call_order.append("response")
        return {"final_response": "Handled by fallback response."}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        final_state = await run_workflow(base_state, graph=graph)

        assert call_order == ["router", "response"]
        assert final_state["final_response"] == "Handled by fallback response."


@pytest.mark.asyncio
async def test_workflow_missing_intent_fallback(base_state: TravelState) -> None:
    """Verify missing intent key in router output routes safely to response."""
    call_order: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        call_order.append("router")
        return {}  # No intent key

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        call_order.append("response")
        return {"final_response": "Handled by fallback response for missing intent."}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        final_state = await run_workflow(base_state, graph=graph)

        assert call_order == ["router", "response"]
        assert "missing intent" in final_state["final_response"]


# ---------------------------------------------------------------------------
# Test 40 & 41: Validation Invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_val", ["flight", "research", "general_travel"])
async def test_validator_always_executes_before_response(
    base_state: TravelState, intent_val: str
) -> None:
    """Verify Validator is ALWAYS executed immediately before Response on specialized paths."""
    visited: List[str] = []

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        visited.append("router")
        return {"intent": intent_val}

    async def mock_flight(state: TravelState) -> Dict[str, Any]:
        visited.append("flight")
        return {"flight_status": "success"}

    async def mock_research(state: TravelState) -> Dict[str, Any]:
        visited.append("research")
        return {"research_status": "success"}

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        visited.append("validator")
        return {"validation_status": "passed"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        visited.append("response")
        return {"final_response": "done"}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight),
        patch("app.graph.workflow.research_node", side_effect=mock_research),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        await run_workflow(base_state, graph=graph)

        assert "validator" in visited
        assert "response" in visited
        assert visited.index("validator") < visited.index("response")


# ---------------------------------------------------------------------------
# Test 42, 43, 44: State Propagation, Preservation, and Termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_propagation_and_preservation(base_state: TravelState) -> None:
    """Verify initial fields are preserved and node updates propagate to final state."""
    orig_session_id = base_state["session_id"]
    orig_request_id = base_state["request_id"]
    orig_user_query = base_state["user_query"]
    orig_metadata = dict(base_state["metadata"])

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        return {"intent": "flight"}

    async def mock_flight(state: TravelState) -> Dict[str, Any]:
        return {"flight_status": "success", "flight_results": [{"flight_iata": "EK522"}]}

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        return {"validation_status": "passed"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        return {"final_response": "Flight confirmed.", "sources": []}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()
        result = await run_workflow(base_state, graph=graph)

        # Initial fields preserved
        assert result["session_id"] == orig_session_id
        assert result["request_id"] == orig_request_id
        assert result["user_query"] == orig_user_query
        assert result["metadata"] == orig_metadata

        # Node enrichments propagated
        assert result["intent"] == "flight"
        assert result["flight_status"] == "success"
        assert result["validation_status"] == "passed"
        assert result["final_response"] == "Flight confirmed."


# ---------------------------------------------------------------------------
# Test 45: Node Failure Exception Propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_node_failure_propagates_exception(base_state: TravelState) -> None:
    """Verify that node exceptions bubble up without being swallowed."""

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        return {"intent": "flight"}

    async def mock_flight_failing(state: TravelState) -> Dict[str, Any]:
        raise RuntimeError("Simulated Flight Provider Transport Outage")

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight_failing),
    ):
        graph = build_workflow()
        with pytest.raises(RuntimeError, match="Simulated Flight Provider Transport Outage"):
            await run_workflow(base_state, graph=graph)


# ---------------------------------------------------------------------------
# Test 46 & 50: Zero External Network Calls During Construction / Import
# ---------------------------------------------------------------------------


def test_zero_external_calls_during_construction() -> None:
    """Verify constructing and compiling the workflow executes zero external network/tool calls."""
    with (
        patch("app.tools.aviationstack.AviationStackClient") as mock_av,
        patch("app.tools.tavily_search.TavilySearchClient") as mock_tav,
        patch("app.core.llm.get_chat_model") as mock_llm,
    ):
        graph = build_workflow()
        assert graph is not None

        assert not mock_av.called
        assert not mock_tav.called
        assert not mock_llm.called


# ---------------------------------------------------------------------------
# Test 47: Concurrent Invocation Safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_workflow_invocations() -> None:
    """Verify multiple concurrent requests run with isolated state without data leakage."""

    async def mock_router(state: TravelState) -> Dict[str, Any]:
        # Return intent corresponding to query
        q = state.get("user_query", "")
        if "flight" in q.lower():
            return {"intent": "flight"}
        return {"intent": "research"}

    async def mock_flight(state: TravelState) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"flight_results": [{"req": state.get("request_id")}]}

    async def mock_research(state: TravelState) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"research_results": [{"req": state.get("request_id")}]}

    async def mock_validator(state: TravelState) -> Dict[str, Any]:
        return {"validation_status": "passed"}

    async def mock_response(state: TravelState) -> Dict[str, Any]:
        return {"final_response": f"Done for {state.get('request_id')}"}

    with (
        patch("app.graph.workflow.router_node", side_effect=mock_router),
        patch("app.graph.workflow.flight_node", side_effect=mock_flight),
        patch("app.graph.workflow.research_node", side_effect=mock_research),
        patch("app.graph.workflow.validator_node", side_effect=mock_validator),
        patch("app.graph.workflow.response_node", side_effect=mock_response),
    ):
        graph = build_workflow()

        state_a = create_initial_state(
            user_query="Search flight", session_id="s1", request_id="req-A"
        )
        state_b = create_initial_state(
            user_query="Search research", session_id="s2", request_id="req-B"
        )

        res_a, res_b = await asyncio.gather(
            run_workflow(state_a, graph=graph),
            run_workflow(state_b, graph=graph),
        )

        assert res_a["request_id"] == "req-A"
        assert res_a["intent"] == "flight"
        assert "req-A" in res_a["final_response"]

        assert res_b["request_id"] == "req-B"
        assert res_b["intent"] == "research"
        assert "req-B" in res_b["final_response"]


# ---------------------------------------------------------------------------
# Test 42: Response Always Terminates to END
# ---------------------------------------------------------------------------


def test_response_always_terminates_to_end() -> None:
    """Verify the compiled workflow topology directly routes from response to END."""
    graph = build_workflow()
    graph_repr = graph.get_graph()
    edges = [(e.source, e.target) for e in graph_repr.edges]
    assert ("response", "__end__") in edges


# ---------------------------------------------------------------------------
# Test 48 & 49: Repeatability and No Duplicate Graph Nodes
# ---------------------------------------------------------------------------


def test_workflow_repeatability_creates_fresh_instances() -> None:
    """Verify build_workflow called multiple times produces fresh, working graphs without collision."""
    graph_1 = build_workflow()
    graph_2 = build_workflow()

    assert graph_1 is not graph_2
    assert isinstance(graph_1, CompiledStateGraph)
    assert isinstance(graph_2, CompiledStateGraph)

    assert set(graph_1.nodes.keys()) == set(graph_2.nodes.keys())


def test_no_duplicate_graph_nodes_on_rebuild() -> None:
    """Verify rebuilding the graph creates independent builder instances with exact node registration."""
    builder_1 = create_workflow()
    builder_2 = create_workflow()

    expected_nodes = {"router", "flight", "research", "validator", "response"}
    assert set(builder_1.nodes.keys()) == expected_nodes
    assert set(builder_2.nodes.keys()) == expected_nodes
    assert len(builder_1.nodes) == 5
    assert len(builder_2.nodes) == 5


# ---------------------------------------------------------------------------
# Test 50: Import Safety
# ---------------------------------------------------------------------------


def test_import_safety_no_network_or_llm_calls() -> None:
    """Verify importing or reloading app.graph.workflow causes zero external network or model calls."""
    import importlib

    with (
        patch("app.tools.aviationstack.AviationStackClient") as mock_av,
        patch("app.tools.tavily_search.TavilySearchClient") as mock_tav,
        patch("app.core.llm.get_chat_model") as mock_llm,
    ):
        import app.graph.workflow as wf

        importlib.reload(wf)
        assert not mock_av.called
        assert not mock_tav.called
        assert not mock_llm.called


# ---------------------------------------------------------------------------
# Test 54: Architectural Compliance
# ---------------------------------------------------------------------------


def test_workflow_architecture_compliance() -> None:
    """
    Verify app.graph.workflow strictly functions as an orchestration layer and does NOT:
        - instantiate LLMs directly
        - call external tools or HTTP clients
        - duplicate routing logic (must import route_after_router from app.graph.edges)
    """
    import app.graph.workflow as wf_module

    source = inspect.getsource(wf_module)

    # Must contain StateGraph orchestration tokens
    required_tokens = ["StateGraph", "add_node", "add_edge", "add_conditional_edges", ".compile("]
    for token in required_tokens:
        assert token in source, f"app.graph.workflow must contain orchestration token: {token!r}"

    # Must import route_after_router from edges
    assert "route_after_router" in source

    # Must NOT directly invoke tools or HTTP clients
    prohibited_tool_tokens = [
        "ChatOpenAI",
        "OpenAI(",
        "TavilySearchClient",
        "AviationStackClient",
        "requests.get",
        "requests.post",
        "httpx",
    ]
    for token in prohibited_tool_tokens:
        assert token not in source, (
            f"app.graph.workflow must not contain direct tool/LLM token: {token!r}"
        )
