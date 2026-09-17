"""
Comprehensive offline test suite for ZICO LangGraph Node Adapters.

Validates:
    - Test 25: Router node adapter execution and delegation
    - Test 26: Flight node adapter execution and delegation
    - Test 27: Research node adapter execution and delegation
    - Test 28: Validator node adapter execution and delegation
    - Test 29: Response node adapter execution and delegation
    - Test 30: Async invocation correctness across all nodes
    - Test 32: Unaltered state propagation and state preservation
    - Test 33: Exception propagation without swallowing errors
    - Test 34: Exactly-once agent invocation (no duplicate calls)
    - Test 35: Logging safety (zero credentials/secrets in operational logs)
    - Test 36: Clean exports and public interface compliance
    - Test 37: Zero graph compilation / workflow construction on import
    - Test 41: Architecture compliance (no direct tool calls or LLM instantiation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from app.core.state import TravelState, create_initial_state
from app.graph.nodes import (
    flight_node,
    research_node,
    response_node,
    router_node,
    validator_node,
)


# ---------------------------------------------------------------------------
# Fixture: Sample TravelState
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state() -> TravelState:
    """Create a realistic base TravelState for node adapter testing."""
    state = create_initial_state(
        user_query="Flight EK522 status from DXB to TRV",
        session_id="session-node-test-01",
        request_id="req-node-test-01",
    )
    state["origin"] = "DXB"
    state["destination"] = "TRV"
    state["departure_date"] = "2026-10-01"
    state["metadata"] = {"source": "unit_test", "tier": "gold"}
    return state


# ---------------------------------------------------------------------------
# Test 25: Router Node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_node_invokes_agent_and_returns_update(base_state: TravelState) -> None:
    """Verify router_node delegates to run_router_agent and returns update unchanged."""
    expected_update = {
        "intent": "flight",
        "intent_confidence": 0.98,
        "active_agent": "router_agent",
    }

    with patch("app.graph.nodes.aroute_request", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = expected_update

        result = await router_node(base_state)

        mock_agent.assert_awaited_once_with(base_state)
        assert result == expected_update
        assert result["intent"] == "flight"


# ---------------------------------------------------------------------------
# Test 26: Flight Node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flight_node_invokes_agent_and_returns_update(base_state: TravelState) -> None:
    """Verify flight_node delegates to run_flight_agent and returns update unchanged."""
    expected_update = {
        "flight_query": {"flight_iata": "EK522"},
        "flight_results": [{"flight_iata": "EK522", "flight_status": "scheduled"}],
        "flight_status": "success",
        "active_agent": "flight_agent",
        "completed_agents": ["flight_agent"],
    }

    with patch("app.graph.nodes.run_flight_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = expected_update

        result = await flight_node(base_state)

        mock_agent.assert_awaited_once_with(base_state)
        assert result == expected_update
        assert result["flight_status"] == "success"


# ---------------------------------------------------------------------------
# Test 27: Research Node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_node_invokes_agent_and_returns_update(base_state: TravelState) -> None:
    """Verify research_node delegates to run_research_agent and returns update unchanged."""
    base_state["intent"] = "research"
    base_state["user_query"] = "What are the Schengen visa requirements for Germany?"

    expected_update = {
        "research_query": "Germany Schengen visa requirements travel",
        "research_results": [{"title": "German Visa Portal", "url": "https://germany.diplo.de/visa", "content": "Valid passport required."}],
        "research_status": "success",
        "active_agent": "research_agent",
        "completed_agents": ["research_agent"],
    }

    with patch("app.graph.nodes.run_research_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = expected_update

        result = await research_node(base_state)

        mock_agent.assert_awaited_once_with(base_state)
        assert result == expected_update
        assert result["research_status"] == "success"


# ---------------------------------------------------------------------------
# Test 28: Validator Node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_node_invokes_agent_and_returns_update(base_state: TravelState) -> None:
    """Verify validator_node delegates to run_validator_agent and returns update unchanged."""
    base_state["intent"] = "flight"
    base_state["flight_status"] = "success"
    base_state["flight_results"] = [{"flight_iata": "EK522"}]

    expected_update = {
        "validation_status": "passed",
        "validation_errors": [],
        "active_agent": "validator_agent",
        "completed_agents": ["validator_agent"],
    }

    with patch("app.graph.nodes.run_validator_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = expected_update

        result = await validator_node(base_state)

        mock_agent.assert_awaited_once_with(base_state)
        assert result == expected_update
        assert result["validation_status"] == "passed"


# ---------------------------------------------------------------------------
# Test 29: Response Node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_node_invokes_agent_and_returns_update(base_state: TravelState) -> None:
    """Verify response_node delegates to run_response_agent and returns update unchanged."""
    base_state["intent"] = "flight"
    base_state["flight_status"] = "success"
    base_state["validation_status"] = "passed"

    expected_update = {
        "final_response": "Flight EK522 from DXB to TRV is scheduled on time.",
        "sources": [],
        "active_agent": "response_agent",
        "completed_agents": ["response_agent"],
    }

    with patch("app.graph.nodes.run_response_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = expected_update

        result = await response_node(base_state)

        mock_agent.assert_awaited_once_with(base_state)
        assert result == expected_update
        assert "EK522" in result["final_response"]


# ---------------------------------------------------------------------------
# Test 32: State Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_passed_intact_without_node_side_effects(base_state: TravelState) -> None:
    """Verify nodes pass the state object directly without mutating unrelated fields."""
    original_state_copy = dict(base_state)

    with patch("app.graph.nodes.aroute_request", new_callable=AsyncMock) as mock_router:
        mock_router.return_value = {"intent": "flight"}
        await router_node(base_state)
        passed_state = mock_router.call_args[0][0]
        # State passed to agent must retain all original fields intact
        assert passed_state["session_id"] == original_state_copy["session_id"]
        assert passed_state["request_id"] == original_state_copy["request_id"]
        assert passed_state["origin"] == original_state_copy["origin"]
        assert passed_state["destination"] == original_state_copy["destination"]
        assert passed_state["metadata"] == original_state_copy["metadata"]


# ---------------------------------------------------------------------------
# Test 33: Exception Propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_fn,agent_patch_path",
    [
        (router_node, "app.graph.nodes.aroute_request"),
        (flight_node, "app.graph.nodes.run_flight_agent"),
        (research_node, "app.graph.nodes.run_research_agent"),
        (validator_node, "app.graph.nodes.run_validator_agent"),
        (response_node, "app.graph.nodes.run_response_agent"),
    ],
)
async def test_agent_exception_propagation(node_fn: Any, agent_patch_path: str, base_state: TravelState) -> None:
    """Verify node adapters do not swallow agent exceptions."""
    custom_error = RuntimeError("Agent runtime failure simulation")

    with patch(agent_patch_path, new_callable=AsyncMock) as mock_agent:
        mock_agent.side_effect = custom_error

        with pytest.raises(RuntimeError, match="Agent runtime failure simulation"):
            await node_fn(base_state)


# ---------------------------------------------------------------------------
# Test 34: Exactly Once Execution (No Duplication)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_duplicate_agent_invocation(base_state: TravelState) -> None:
    """Verify invoking a node triggers the underlying agent exactly once."""
    with patch("app.graph.nodes.aroute_request", new_callable=AsyncMock) as m_router, \
         patch("app.graph.nodes.run_flight_agent", new_callable=AsyncMock) as m_flight, \
         patch("app.graph.nodes.run_research_agent", new_callable=AsyncMock) as m_research, \
         patch("app.graph.nodes.run_validator_agent", new_callable=AsyncMock) as m_validator, \
         patch("app.graph.nodes.run_response_agent", new_callable=AsyncMock) as m_response:

        m_router.return_value = {"intent": "flight"}
        m_flight.return_value = {"flight_status": "success"}
        m_research.return_value = {"research_status": "success"}
        m_validator.return_value = {"validation_status": "passed"}
        m_response.return_value = {"final_response": "done"}

        await router_node(base_state)
        assert m_router.call_count == 1

        await flight_node(base_state)
        assert m_flight.call_count == 1

        await research_node(base_state)
        assert m_research.call_count == 1

        await validator_node(base_state)
        assert m_validator.call_count == 1

        await response_node(base_state)
        assert m_response.call_count == 1


# ---------------------------------------------------------------------------
# Test 35: Logging Safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_safety_no_secret_leakage(caplog: pytest.LogCaptureFixture, base_state: TravelState) -> None:
    """Verify node execution logs do not contain API keys or private tokens."""
    caplog.set_level(logging.INFO)

    secret_key = "sk-proj-super-secret-key-12345"
    base_state["metadata"] = {"api_key": secret_key}

    with patch("app.graph.nodes.aroute_request", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = {"intent": "flight"}

        await router_node(base_state)

        for record in caplog.records:
            assert secret_key not in record.message
            assert "super-secret" not in record.message


# ---------------------------------------------------------------------------
# Test 36: Module Interface and Exports
# ---------------------------------------------------------------------------


def test_nodes_module_exports() -> None:
    """Verify app.graph.nodes defines clear, callable node adapters via __all__."""
    import app.graph.nodes as nodes_module

    expected_exports = {
        "router_node",
        "flight_node",
        "research_node",
        "validator_node",
        "response_node",
    }

    assert hasattr(nodes_module, "__all__")
    assert set(nodes_module.__all__) == expected_exports

    for name in expected_exports:
        fn = getattr(nodes_module, name)
        assert callable(fn), f"Export {name} must be a callable function"


# ---------------------------------------------------------------------------
# Test 37 & 41: Architecture Compliance (No Graph Construction or Tool Calls)
# ---------------------------------------------------------------------------


def test_no_graph_construction_or_tool_execution_in_nodes() -> None:
    """
    Verify app.graph.nodes strictly functions as an adapter layer and does NOT:
        - construct or compile StateGraph
        - define conditional edge routing
        - import or call external HTTP clients / providers directly
    """
    import inspect
    import app.graph.nodes as nodes_module

    source_code = inspect.getsource(nodes_module)

    # Must NOT construct or compile StateGraph
    prohibited_graph_tokens = [
        "StateGraph",
        "add_node",
        "add_edge",
        "add_conditional_edges",
        ".compile(",
    ]
    for token in prohibited_graph_tokens:
        assert token not in source_code, f"app.graph.nodes must not contain graph construction token: {token!r}"

    # Must NOT directly invoke tools or HTTP clients
    prohibited_tool_tokens = [
        "AviationStackClient",
        "TavilySearchClient",
        "httpx",
        "requests.get",
        "requests.post",
        "ChatOpenAI",
        "OpenAI(",
    ]
    for token in prohibited_tool_tokens:
        assert token not in source_code, f"app.graph.nodes must not directly call external tools/providers: {token!r}"
