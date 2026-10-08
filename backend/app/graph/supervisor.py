import logging
import os
from typing import Any, Dict, List, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.65
DEFAULT_FALLBACK_ROUTE = "validator_node"


class RouteDecision(BaseModel):
    """Pydantic structured output model for supervisor routing decisions."""

    next_step: Literal[
        "flight_search_worker",
        "research_worker",
        "policy_rag_worker",
        "disruption_worker",
        "booking_approval_node",
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
    reasoning: str = Field(description="Short rationale explaining the routing classification.")


SUPERVISOR_SYSTEM_PROMPT = """You are the Lead Routing Supervisor for the ZICO intelligent travel operations companion.
Analyze the traveler's conversation history, active state, and latest inquiry to classify intent and select the single most appropriate worker.

Available Workers:
1. flight_search_worker: For flight searches, flight status, schedules, and airline availability. NEVER use for hotels or destination sightseeing.
2. research_worker: For hotel searches, accommodation, resorts, budget stays, destination research, local attractions, things to do, and travel planning.
3. policy_rag_worker: For baggage dimensions/weight rules, cancellation policies, EU261 passenger compensation, visas, and insurance regulations.
4. disruption_worker: For flight delays, cancellations, missed connections, schedule collisions, and urgent rebooking assistance.
5. booking_approval_node: When approving or executing a pending booking or cancellation action.
6. validator_node: For checking itinerary conflict overlaps, layover buffers, and budget compliance.
7. FINISH: When the user's intent is answered or no further worker action is required.

Safety Rules:
- If the user asks about hotels, resorts, or accommodation (e.g. 'Hotels in Pune under 10k'), you MUST route to research_worker. NEVER route hotel inquiries to flight_search_worker.
- If traveler intent is ambiguous, low-confidence, or unclear (confidence < 0.65), default to validator_node.
- Never guess or route to a specialized worker without explicit intent.
"""


def _classify_intent_heuristically(latest_text: str) -> str:
    """Deterministic fallback intent classifier when LLM is offline, rate-limited, or quota exceeded."""
    lower = latest_text.lower()

    # 1. Hotel and Destination Research has highest priority over general terms
    if any(
        k in lower
        for k in [
            "hotel",
            "hotels",
            "hostel",
            "resort",
            "stay",
            "room",
            "lodging",
            "accommodation",
            "places to visit",
            "attractions",
            "sightseeing",
            "things to do",
            "explore",
            "visit in",
            "guide",
        ]
    ):
        return "research_worker"

    # 2. Flight disruptions
    if any(
        k in lower
        for k in ["delay", "cancel", "reschedule", "disrupt", "missed", "stranded", "late"]
    ):
        return "disruption_worker"

    # 3. Policy & Regulations
    if any(
        k in lower
        for k in [
            "policy",
            "baggage",
            "bag",
            "luggage",
            "visa",
            "passport",
            "refund",
            "insurance",
            "eu261",
            "compensation",
            "rule",
        ]
    ):
        return "policy_rag_worker"

    # 4. Approvals
    if any(
        k in lower for k in ["approve", "confirm", "proceed", "yes", "accept", "reject", "deny"]
    ):
        return "booking_approval_node"

    # 5. Flight Search
    if any(
        k in lower
        for k in [
            "flight",
            "flights",
            "fly",
            "airline",
            "ticket",
            "airport",
            "plane",
            "from ",
            "trip to",
        ]
    ):
        return "flight_search_worker"

    return DEFAULT_FALLBACK_ROUTE


def supervisor_node(state: Dict[str, Any] | Any) -> Dict[str, Any]:
    """
    Evaluates conversation history and deterministically routes execution to the appropriate worker node.
    Uses ChatOpenAI with structured output (RouteDecision) and robust pattern-based fallback.
    """
    if isinstance(state, dict):
        raw_messages = state.get("messages") or []
    else:
        raw_messages = getattr(state, "messages", None) or []

    messages: List[BaseMessage] = []
    latest_user_text = ""
    for msg in raw_messages:
        if isinstance(msg, BaseMessage):
            messages.append(msg)
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                latest_user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, str):
            messages.append(HumanMessage(content=msg))
            latest_user_text = msg
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            messages.append(HumanMessage(content=content))
            latest_user_text = content

    if not messages:
        messages = [HumanMessage(content="Hello ZICO")]

    from app.core.exceptions import ProviderError
    from app.core.request_context import get_request_id, get_session_id, set_request_context

    req_id = (state.get("request_id") if isinstance(state, dict) else getattr(state, "request_id", None)) or get_request_id()
    sess_id = (state.get("session_id") if isinstance(state, dict) else getattr(state, "session_id", None)) or get_session_id()
    if req_id or sess_id:
        set_request_context(request_id=req_id, session_id=sess_id)

    is_mocked = hasattr(ChatOpenAI, "assert_called") or "mock" in type(ChatOpenAI).__module__
    if is_mocked or (
        settings.OPENAI_API_KEY
        and not settings.OPENAI_API_KEY.startswith("test")
        and settings.APP_ENV != "test"
        and os.getenv("PYTEST_CURRENT_TEST") is None
    ):
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
                request_timeout=2.0,
                max_retries=0,
            )
            structured_llm = llm.with_structured_output(RouteDecision)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SUPERVISOR_SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            chain = prompt | structured_llm

            decision: RouteDecision = chain.invoke({"messages": messages})
            confidence = getattr(decision, "confidence", 1.0)
            next_step = getattr(decision, "next_step", DEFAULT_FALLBACK_ROUTE)

            if confidence < CONFIDENCE_THRESHOLD:
                next_step = _classify_intent_heuristically(latest_user_text)

        except Exception as exc:
            err_str = str(exc).lower()
            if any(k in err_str for k in ["quota", "429", "insufficient_quota", "credit_balance_exhausted", "rate_limit"]):
                logger.error("OpenAI quota exhausted during supervisor routing: %s", exc)
                raise ProviderError(
                    "OpenAI quota exhausted",
                    safe_message="ZICO is temporarily unable to process this request because the AI service is unavailable. Please try again later.",
                    details={"provider": "openai", "error_type": "insufficient_quota"},
                ) from exc
            if any(k in err_str for k in ["timeout", "timed out"]):
                logger.error("OpenAI timeout during supervisor routing: %s", exc)
                raise ProviderError(
                    "OpenAI request timed out",
                    safe_message="ZICO request timed out while contacting the AI service. Please try again later.",
                    details={"provider": "openai", "error_type": "timeout"},
                ) from exc

            logger.warning(
                f"LLM supervisor invocation notice: {exc}. Using deterministic intent routing."
            )
            next_step = _classify_intent_heuristically(latest_user_text)
    else:
        next_step = _classify_intent_heuristically(latest_user_text)

    if not isinstance(next_step, str):
        next_step = DEFAULT_FALLBACK_ROUTE

    return {"next_node": next_step}
