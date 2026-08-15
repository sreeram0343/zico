from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from app.graph.supervisor import RouteDecision, supervisor_node


def test_route_decision_schema():
    """Verify RouteDecision model validation."""
    decision = RouteDecision(
        next_step="flight_search_worker",
        reasoning="User requested flight search"
    )
    assert decision.next_step == "flight_search_worker"
    assert decision.reasoning == "User requested flight search"


@patch("app.graph.supervisor.ChatOpenAI")
def test_supervisor_node_routing(mock_chat_openai):
    """Test supervisor_node execution with mocked LLM."""
    mock_decision = RouteDecision(
        next_step="flight_search_worker",
        reasoning="Routing to flight search"
    )
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = RunnableLambda(lambda x: mock_decision)
    mock_chat_openai.return_value = mock_llm_instance

    state = {
        "messages": [HumanMessage(content="Find me flights from NYC to LHR")]
    }

    result = supervisor_node(state)

    assert result == {"next_node": "flight_search_worker"}
