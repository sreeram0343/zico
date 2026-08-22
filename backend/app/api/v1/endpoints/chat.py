from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from app.graph.engine import graph_engine
from app.graph.state import (
    PendingAction,
    TripConstraints,
    TripSegment,
    ZicoGraphState,
)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(description="Traveler query, command, or update")
    trip_id: Optional[str] = Field(default=None, description="Active Trip ID for state persistence")
    user_id: Optional[str] = Field(default="user_default", description="Traveler User ID")
    constraints: Optional[TripConstraints] = None
    itinerary: Optional[List[TripSegment]] = None


class ChatResponse(BaseModel):
    trip_id: str
    user_id: str
    reply: str
    active_disruptions: List[Dict[str, Any]] = Field(default_factory=list)
    pending_actions: List[PendingAction] = Field(default_factory=list)
    itinerary: List[TripSegment] = Field(default_factory=list)


@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest) -> ChatResponse:
    """
    Executes the conversational travel operations workflow via the LangGraph state machine.
    """
    trip_id = request.trip_id or f"trip_{uuid.uuid4().hex[:8]}"
    user_id = request.user_id or "user_default"

    config = {"configurable": {"thread_id": trip_id}}

    # Fetch existing graph state if present
    try:
        current_snapshot = graph_engine.get_state(config)
        existing_values = current_snapshot.values if current_snapshot else {}
    except Exception:
        existing_values = {}

    current_itinerary = request.itinerary or existing_values.get("itinerary", [])
    current_constraints = request.constraints or existing_values.get("constraints", TripConstraints())
    current_actions = existing_values.get("pending_actions", [])
    current_disruptions = existing_values.get("active_disruptions", [])

    input_payload = {
        "messages": [HumanMessage(content=request.message)],
        "trip_id": trip_id,
        "user_id": user_id,
        "itinerary": current_itinerary,
        "constraints": current_constraints,
        "pending_actions": current_actions,
        "active_disruptions": current_disruptions,
    }

    try:
        result = graph_engine.invoke(input_payload, config=config)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow execution failed: {str(exc)}",
        )

    # Extract latest assistant message
    messages = result.get("messages", [])
    reply_text = "I have updated your travel state and verified your itinerary."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
            reply_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return ChatResponse(
        trip_id=result.get("trip_id", trip_id),
        user_id=result.get("user_id", user_id),
        reply=reply_text,
        active_disruptions=result.get("active_disruptions", []),
        pending_actions=result.get("pending_actions", []),
        itinerary=result.get("itinerary", []),
    )
