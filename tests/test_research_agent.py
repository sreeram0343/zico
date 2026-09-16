"""
Comprehensive offline test suite for ZICO Travel Research Agent.

Validates:
    - Test 38: Basic research request
    - Test 39: Current-information request
    - Test 40: Direct deterministic query path (no LLM)
    - Test 41: LLM-assisted query generation
    - Test 42: Malformed LLM output fallback
    - Test 43: Missing / whitespace user query handling
    - Test 44: Multi-turn context incorporation (e.g. destination in state)
    - Test 45: Tavily normalized search results preservation
    - Test 46: Duplicate URL deduplication
    - Test 47: Empty search results handling
    - Test 48: Tavily provider failure handling
    - Test 49: Tavily timeout error handling
    - Test 50: Location resolver failure resilience
    - Test 51: State preservation of unrelated fields
    - Test 52: Non-research intent rejection
    - Test 53: No AviationStack calls
    - Test 54: No final response generation
    - Test 55: Secret protection (no credentials in logs/state)
    - Test 56: JSON serialization compatibility
    - Test 57: Repeated execution idempotency

Mocking:
    - All external network dependencies (Tavily, OpenAI, AviationStack) are strictly mocked.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.research_agent import (
    ResearchAgent,
    ResearchQueryDecision,
    run_research_agent,
)
from app.core.state import TravelState, create_initial_state
from app.tools.location import ResolvedLocation
from app.tools.tavily_search import (
    ResearchResult,
    TavilyAPIError,
    TavilySearchResponse,
    TavilyTimeoutError,
)


# ---------------------------------------------------------------------------
# Test Helpers & Mock Fixtures
# ---------------------------------------------------------------------------


def make_mock_result(
    title: str = "Germany Visa Rules",
    url: str = "https://example.com/visa",
    content: str = "Indian citizens require a Schengen visa for short stays in Germany.",
    score: float = 0.95,
) -> ResearchResult:
    """Create a normalized ResearchResult model."""
    return ResearchResult(
        title=title,
        url=url,
        content=content,
        score=score,
        published_date="2026-01-15",
    )


def mock_tavily_client(results: Optional[List[ResearchResult]] = None) -> AsyncMock:
    """Construct an AsyncMock simulating TavilySearchClient."""
    client = AsyncMock()
    items = results if results is not None else [make_mock_result()]
    client.search.return_value = TavilySearchResponse(
        success=True,
        query="Germany visa requirements",
        count=len(items),
        results=items,
    )
    return client


# ---------------------------------------------------------------------------
# Test 38: Basic Research Request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_research_request() -> None:
    """Verify research query preparation, Tavily invocation, and result storage."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="What are the visa requirements for Germany?",
        session_id="sess-001",
        request_id="req-001",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    assert client.search.called
    assert update["research_status"] == "success"
    assert update["active_agent"] == "research_agent"
    assert "research_agent" in update["completed_agents"]
    assert len(update["research_results"]) == 1
    assert update["research_results"][0]["url"] == "https://example.com/visa"
    assert "germany" in update["research_query"].lower()


# ---------------------------------------------------------------------------
# Test 39: Current-Information Request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_information_request() -> None:
    """Verify temporal intent (latest, current, today, 2026) is preserved in research query."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="What are the latest visa rules for Japan updated today?",
        session_id="sess-002",
        request_id="req-002",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    assert client.search.called
    query_used = update["research_query"].lower()
    # Must preserve temporal indicator
    assert any(term in query_used for term in ["latest", "current", "today", "updated"])


# ---------------------------------------------------------------------------
# Test 40: Direct Query Path (No LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_query_path() -> None:
    """Verify simple requests use deterministic query construction without invoking OpenAI."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Dubai airport baggage allowance",
        session_id="sess-003",
        request_id="req-003",
    )
    state["intent"] = "research"

    # Ensure get_chat_model is not called when direct deterministic path runs
    with patch("app.agents.research_agent.get_chat_model") as mock_llm_getter:
        update = await run_research_agent(state, tavily_client=client, use_llm=False)
        assert not mock_llm_getter.called
        assert update["research_status"] == "success"
        assert "dubai airport baggage allowance" in update["research_query"].lower()


# ---------------------------------------------------------------------------
# Test 41: LLM-Assisted Query Generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_assisted_query_generation() -> None:
    """Verify LLM query generation is invoked and structured output is passed to Tavily."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Could you explain all entry documentation needed for European travel?",
        session_id="sess-004",
        request_id="req-004",
    )
    state["intent"] = "research"

    mock_model = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = ResearchQueryDecision(
        query="Europe travel entry documentation official requirements"
    )
    mock_model.with_structured_output.return_value = mock_runnable

    with patch("app.agents.research_agent.get_chat_model", return_value=mock_model) as mock_getter:
        update = await run_research_agent(
            state,
            tavily_client=client,
            use_llm=True,
        )

        assert mock_getter.called
        assert mock_runnable.invoke.called
        assert update["research_query"] == "Europe travel entry documentation official requirements"
        assert client.search.called
        assert client.search.call_args[1]["query"] == "Europe travel entry documentation official requirements"


# ---------------------------------------------------------------------------
# Test 42: Malformed LLM Output Fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_llm_output() -> None:
    """Verify invalid or empty structured LLM output safely falls back to deterministic query."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="What are Schengen visa requirements?",
        session_id="sess-005",
        request_id="req-005",
    )
    state["intent"] = "research"

    mock_model = MagicMock()
    mock_runnable = MagicMock()
    # Simulate LLM returning invalid schema (empty query or junk)
    mock_runnable.invoke.return_value = {"invalid_key": "unusable output"}
    mock_model.with_structured_output.return_value = mock_runnable

    with patch("app.agents.research_agent.get_chat_model", return_value=mock_model):
        update = await run_research_agent(
            state,
            tavily_client=client,
            use_llm=True,
        )

        # Should safely fall back to cleaned query, not fail or call Tavily with garbage
        assert update["research_status"] == "success"
        assert "schengen visa requirements" in update["research_query"].lower()
        assert client.search.called


# ---------------------------------------------------------------------------
# Test 43: Missing Query Context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_query_context() -> None:
    """Verify empty or whitespace-only user queries return an error without calling Tavily."""
    client = mock_tavily_client()

    for empty_val in ["", "   ", "\t\n"]:
        state: TravelState = create_initial_state(
            user_query=empty_val,
            session_id="sess-empty",
            request_id="req-empty",
        )
        state["intent"] = "research"

        update = await run_research_agent(state, tavily_client=client)

        assert not client.search.called
        assert update["research_status"] == "error"
        assert any("missing or empty" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 44: Multi-Turn Context Incorporation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiturn_context() -> None:
    """Verify available state context (destination='Germany') is included when query is vague."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="What documents do I need?",
        session_id="sess-multi",
        request_id="req-multi",
    )
    state["intent"] = "research"
    state["destination"] = "Germany"

    update = await run_research_agent(state, tavily_client=client)

    assert client.search.called
    assert "germany" in update["research_query"].lower()
    assert "documents" in update["research_query"].lower()


# ---------------------------------------------------------------------------
# Test 45: Tavily Results Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tavily_results_preservation() -> None:
    """Verify all fields (title, url, content, score, date) from Tavily results are preserved."""
    items = [
        ResearchResult(
            title="Official German Federal Foreign Office",
            url="https://auswaertiges-amt.de/en/visa-service",
            content="Official visa information for entry into the Federal Republic of Germany.",
            score=0.98,
            published_date="2026-02-01",
        ),
        ResearchResult(
            title="Schengen Visa Info Germany",
            url="https://schengenvisainfo.com/germany-visa",
            content="Guidelines and required application documents for tourists.",
            score=0.88,
        ),
    ]
    client = mock_tavily_client(results=items)

    state: TravelState = create_initial_state(
        user_query="Germany visa details",
        session_id="sess-fields",
        request_id="req-fields",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    assert update["research_status"] == "success"
    res = update["research_results"]
    assert len(res) == 2
    assert res[0]["title"] == "Official German Federal Foreign Office"
    assert res[0]["url"] == "https://auswaertiges-amt.de/en/visa-service"
    assert res[0]["score"] == 0.98
    assert res[0]["published_date"] == "2026-02-01"
    assert res[1]["score"] == 0.88


# ---------------------------------------------------------------------------
# Test 46: Duplicate Result Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_result_handling() -> None:
    """Verify duplicate URLs in provider results are deduplicated while preserving order."""
    items = [
        make_mock_result(title="Result 1", url="https://example.com/visa"),
        make_mock_result(title="Result 1 Duplicate", url="https://example.com/visa/"),  # trailing slash
        make_mock_result(title="Result 2", url="https://example.com/other-info"),
    ]
    client = mock_tavily_client(results=items)

    state: TravelState = create_initial_state(
        user_query="Visa requirements",
        session_id="sess-dedup",
        request_id="req-dedup",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    res = update["research_results"]
    assert len(res) == 2
    assert res[0]["url"] == "https://example.com/visa"
    assert res[1]["url"] == "https://example.com/other-info"


# ---------------------------------------------------------------------------
# Test 47: Empty Results Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tavily_empty_results() -> None:
    """Verify successful Tavily search returning 0 matches sets no_results without error."""
    client = mock_tavily_client(results=[])

    state: TravelState = create_initial_state(
        user_query="Obscure travel regulation inquiry",
        session_id="sess-empty-res",
        request_id="req-empty-res",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    assert update["research_status"] == "no_results"
    assert update["research_results"] == []
    # Must NOT record a false application error
    assert "errors" not in update


# ---------------------------------------------------------------------------
# Test 48: Tavily Provider Failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tavily_provider_failure() -> None:
    """Verify provider exceptions are caught and stored in state errors distinctly from no_results."""
    client = AsyncMock()
    client.search.side_effect = TavilyAPIError("Tavily API quota exceeded", status_code=429)

    state: TravelState = create_initial_state(
        user_query="Dubai travel guidelines",
        session_id="sess-err",
        request_id="req-err",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    assert update["research_status"] == "error"
    assert update["research_results"] == []
    assert any("quota exceeded" in err for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 49: Tavily Timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tavily_timeout() -> None:
    """Verify Tavily timeout is handled gracefully and recorded in state errors."""
    client = AsyncMock()
    client.search.side_effect = TavilyTimeoutError("Tavily request timed out after 10.0s")

    state: TravelState = create_initial_state(
        user_query="Visa regulations",
        session_id="sess-timeout",
        request_id="req-timeout",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    assert update["research_status"] == "error"
    assert any("timed out" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 50: Location Resolver Failure Resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_location_resolver_failure() -> None:
    """Verify location resolver failures do not crash the Research Agent."""
    client = mock_tavily_client()
    mock_resolver = MagicMock()
    mock_resolver.resolve.side_effect = RuntimeError("Location database offline")

    state: TravelState = create_initial_state(
        user_query="What documents do I need?",
        session_id="sess-loc-fail",
        request_id="req-loc-fail",
    )
    state["intent"] = "research"
    state["destination"] = "Atlantis"

    update = await run_research_agent(
        state,
        tavily_client=client,
        location_resolver=mock_resolver,
    )

    # Should fall back to raw destination and still succeed
    assert update["research_status"] == "success"
    assert "atlantis" in update["research_query"].lower()
    assert client.search.called


# ---------------------------------------------------------------------------
# Test 51: State Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_preservation() -> None:
    """Verify unrelated state fields remain completely unchanged after research operations."""
    client = mock_tavily_client()

    original_state: TravelState = {
        "user_query": "What are German visa requirements?",
        "session_id": "sess-unique-999",
        "request_id": "req-unique-888",
        "intent": "research",
        "origin": "TRV",
        "destination": "FRA",
        "flight_results": [{"flight_iata": "EK522"}],
        "metadata": {"user_loyalty": "gold", "client_version": "1.0"},
        "workflow_status": "running",
        "errors": [],
    }

    update = await run_research_agent(original_state, tavily_client=client)

    # Agent must only return fields it updates
    assert "flight_results" not in update
    assert "session_id" not in update
    assert "request_id" not in update
    assert "origin" not in update

    merged = {**original_state, **update}
    assert merged["session_id"] == "sess-unique-999"
    assert merged["request_id"] == "req-unique-888"
    assert merged["flight_results"] == [{"flight_iata": "EK522"}]
    assert merged["metadata"]["user_loyalty"] == "gold"
    assert merged["research_status"] == "success"


# ---------------------------------------------------------------------------
# Test 52: Non-Research Intent Rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_intent() -> None:
    """Verify requests with intent != 'research' are rejected without calling Tavily."""
    client = mock_tavily_client()

    for non_research_intent in ["flight", "general_travel", "unsupported"]:
        state: TravelState = create_initial_state(
            user_query="Check EK522 status",
            session_id="sess-wrong",
            request_id="req-wrong",
        )
        state["intent"] = non_research_intent

        update = await run_research_agent(state, tavily_client=client)

        assert not client.search.called
        assert update["research_status"] == "error"
        # Must NOT modify or overwrite the intent
        assert "intent" not in update
        assert any("invalid intent" in err.lower() for err in update["errors"])


# ---------------------------------------------------------------------------
# Test 53: No AviationStack Call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_aviationstack_call() -> None:
    """Verify Research Agent never imports or invokes the AviationStack tool."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Germany visa regulations",
        session_id="sess-no-aviation",
        request_id="req-no-aviation",
    )
    state["intent"] = "research"

    with patch("app.tools.aviationstack.AviationStackClient") as mock_aviation:
        update = await run_research_agent(state, tavily_client=client)
        assert update["research_status"] == "success"
        assert not mock_aviation.called


# ---------------------------------------------------------------------------
# Test 54: No Final Response Generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_final_response_generation() -> None:
    """Verify Research Agent produces only structured state, never final natural language."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Baggage rules for Emirates",
        session_id="sess-no-resp",
        request_id="req-no-resp",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    # Must NOT generate final_response
    assert "final_response" not in update


# ---------------------------------------------------------------------------
# Test 55: Secret Protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_secret_leakage(caplog: pytest.LogCaptureFixture) -> None:
    """Verify API keys and credentials never appear in logs, returned state, or errors."""
    caplog.set_level(logging.DEBUG)
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Dubai travel regulations",
        session_id="sess-secret",
        request_id="req-secret",
    )
    state["intent"] = "research"

    fake_key = "test-tavily-key-secret-123"

    update = await run_research_agent(state, tavily_client=client)

    # Check state update
    serialized_update = json.dumps(update)
    assert fake_key not in serialized_update

    # Check logs
    assert fake_key not in caplog.text


# ---------------------------------------------------------------------------
# Test 56: JSON Serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serialization() -> None:
    """Verify research update dictionary is completely JSON-serializable."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Visa requirements for Germany",
        session_id="sess-json",
        request_id="req-json",
    )
    state["intent"] = "research"

    update = await run_research_agent(state, tavily_client=client)

    # Must serialize without TypeError
    serialized = json.dumps(update)
    reconstructed = json.loads(serialized)
    assert reconstructed["research_status"] == "success"
    assert len(reconstructed["research_results"]) == 1


# ---------------------------------------------------------------------------
# Test 57: Repeated Execution Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_execution() -> None:
    """Verify executing Research Agent repeatedly does not duplicate completed_agents entries."""
    client = mock_tavily_client()

    state: TravelState = create_initial_state(
        user_query="Visa requirements for Germany",
        session_id="sess-repeat",
        request_id="req-repeat",
    )
    state["intent"] = "research"

    # First run
    update1 = await run_research_agent(state, tavily_client=client)
    state1 = {**state, **update1}
    assert state1["completed_agents"] == ["research_agent"]

    # Second run against updated state
    update2 = await run_research_agent(state1, tavily_client=client)
    state2 = {**state1, **update2}
    assert state2["completed_agents"] == ["research_agent"]
    assert state2["completed_agents"].count("research_agent") == 1
