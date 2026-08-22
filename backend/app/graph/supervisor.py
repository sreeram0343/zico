from typing import Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.config import settings

CONFIDENCE_THRESHOLD = 0.65
DEFAULT_FALLBACK_ROUTE = "validator_node"


class RouteDecision(BaseModel):
    """Pydantic structured output model for supervisor routing decisions."""

    next_step: Literal[
        "flight_search_worker",
        "policy_rag_worker",
        "disruption_worker",
        "validator_node",
        "FINISH",
    ] = Field(
        description="The next specialized worker node or terminal action to route the conversation to."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for this routing classification.",
    )
    reasoning: str = Field(
        description="Short rationale explaining the routing classification."
    )


SUPERVISOR_SYSTEM_PROMPT = """You are the Lead Routing Supervisor for the ZICO intelligent travel operations companion.
Analyze the traveler's conversation history, active state, and latest inquiry to classify intent and select the single most appropriate worker.

Available Workers:
1. flight_search_worker: For flight searches, schedules, airline availability, fares, and new flight reservations.
2. policy_rag_worker: For baggage dimensions/weight rules, cancellation policies, EU261 passenger compensation, visas, and insurance regulations.
3. disruption_worker: For flight delays, cancellations, missed connections, schedule collisions, and urgent rebooking assistance.
4. validator_node: For checking itinerary conflict overlaps, layover buffers, and budget compliance.
5. FINISH: When the user's intent is answered or no further worker action is required.

Safety Rules:
- If traveler intent is ambiguous, low-confidence, or unclear (confidence < 0.65), default to validator_node.
- Never guess or route to a specialized worker without explicit intent.
"""


def supervisor_node(state: Dict[str, Any] | Any) -> Dict[str, Any]:
    """
    Evaluates conversation history and deterministically routes execution to the appropriate worker node.
    Uses ChatOpenAI with structured output (RouteDecision) and explicit confidence-threshold fallback.
    """
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

    try:
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

        decision: RouteDecision = chain.invoke({"messages": messages})
        confidence = getattr(decision, "confidence", 1.0)
        next_step = getattr(decision, "next_step", DEFAULT_FALLBACK_ROUTE)

        # Enforce confidence threshold fallback for ambiguous intent
        if confidence < CONFIDENCE_THRESHOLD:
            next_step = DEFAULT_FALLBACK_ROUTE

    except Exception:
        # Graceful fallback to default route on offline tests / API disconnect
        next_step = DEFAULT_FALLBACK_ROUTE

    if not isinstance(next_step, str):
        next_step = DEFAULT_FALLBACK_ROUTE

    return {"next_node": next_step}

