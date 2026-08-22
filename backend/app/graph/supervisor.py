from typing import Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.config import settings


# 1. Define allowed routing destinations
class RouteDecision(BaseModel):
    """Pydantic structured output model for supervisor routing decisions."""

    next_step: Literal[
        "flight_search_worker",
        "policy_rag_worker",
        "disruption_worker",
        "validator_node",
        "FINISH",
    ] = Field(
        description="The next worker node or terminal action to route the conversation to."
    )
    reasoning: str = Field(
        description="Short rationale for choosing this worker or finishing"
    )


# 2. System prompt enforcing role separation
SUPERVISOR_SYSTEM_PROMPT = """You are the orchestrator for the ZICO travel companion.
Decide which worker needs to act next based on traveler state and latest message.

Available Workers:
- flight_search_worker: For flight searches, availability, schedules, fares, and flight lookup queries.
- policy_rag_worker: For baggage allowance, cancellation policies, airline rules, and travel policy documents.
- disruption_worker: For handling flight delays, cancellations, rebooking assistance, weather issues, or urgent travel disruptions.
- validator_node: For checking itinerary conflicts, budget caps, and timeline buffer constraints.
- FINISH: Select FINISH when the conversation has completed, the user's intent is answered, or no further specialized worker action is required.
"""


# 3. Supervisor Node callable
def supervisor_node(state: Dict[str, Any] | Any) -> Dict[str, Any]:
    """
    Evaluates the conversation history and routes execution to the appropriate worker node or FINISH.
    Uses ChatOpenAI with structured output (RouteDecision).
    """
    # Extract messages defensively from state dictionary or state object
    if isinstance(state, dict):
        raw_messages = state.get("messages") or []
    else:
        raw_messages = getattr(state, "messages", None) or []

    messages: List[BaseMessage] = []
    for msg in raw_messages:
        if isinstance(msg, BaseMessage):
            messages.append(msg)
        elif isinstance(msg, str):
            messages.append(HumanMessage(content=msg))
        elif isinstance(msg, dict):
            messages.append(HumanMessage(content=msg.get("content", "")))

    if not messages:
        messages = [HumanMessage(content="Hello ZICO")]

    # Initialize LLM and bind structured output
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.OPENAI_API_KEY,
        timeout=3.0,
        max_retries=1,
    )
    structured_llm = llm.with_structured_output(RouteDecision)


    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | structured_llm

    try:
        decision: RouteDecision = chain.invoke({"messages": messages})
        next_step = getattr(decision, "next_step", "validator_node")
    except Exception:
        # Fallback to validator_node if external API fails or in offline tests
        next_step = "validator_node"

    if not isinstance(next_step, str):
        next_step = "validator_node"

    return {"next_node": next_step}

