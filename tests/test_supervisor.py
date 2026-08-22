from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
import pytest
from app.graph.supervisor import RouteDecision, supervisor_node


def test_route_decision_schema():
    """Verify RouteDecision model validation and default confidence."""
    decision = RouteDecision(
        next_step="flight_search_worker",
        confidence=0.95,
        reasoning="User requested flight search",
    )
    assert decision.next_step == "flight_search_worker"
    assert decision.confidence == 0.95
    assert decision.reasoning == "User requested flight search"


@patch("app.graph.supervisor.ChatOpenAI")
def test_supervisor_node_routing_high_confidence(mock_chat_openai):
    """Test supervisor_node execution routing to specialized worker with high confidence."""
    mock_decision = RouteDecision(
        next_step="policy_rag_worker",
        confidence=0.9,
        reasoning="Inquiry regarding EU261 compensation rules",
    )
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = RunnableLambda(lambda x: mock_decision)
    mock_chat_openai.return_value = mock_llm_instance

    state = {
        "messages": [HumanMessage(content="What are my rights if my flight is delayed by 4 hours in Paris?")]
    }

    result = supervisor_node(state)
    assert result == {"next_node": "policy_rag_worker"}


@patch("app.graph.supervisor.ChatOpenAI")
def test_supervisor_node_ambiguous_low_confidence_fallback(mock_chat_openai):
    """Test that low-confidence or ambiguous intent defaults to validator_node."""
    mock_decision = RouteDecision(
        next_step="flight_search_worker",
        confidence=0.4,  # below 0.65 threshold
        reasoning="Ambiguous input with low confidence",
    )
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = RunnableLambda(lambda x: mock_decision)
    mock_chat_openai.return_value = mock_llm_instance

    state = {
        "messages": [HumanMessage(content="Maybe something about tomorrow?")]
    }

    result = supervisor_node(state)
    assert result == {"next_node": "validator_node"}


@patch("app.graph.supervisor.ChatOpenAI")
def test_supervisor_node_exception_fallback(mock_chat_openai):
    """Test that unexpected LLM or network exceptions gracefully fallback to validator_node."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.side_effect = Exception("API connection failure")
    mock_chat_openai.return_value = mock_llm_instance

    state = {
        "messages": [HumanMessage(content="Hello ZICO")]
    }

    result = supervisor_node(state)
    assert result == {"next_node": "validator_node"}
