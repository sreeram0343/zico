from __future__ import annotations
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Domain Models (Strict Pydantic)
# ---------------------------------------------------------------------------

class SegmentType(str):
    FLIGHT = "flight"
    HOTEL = "hotel"
    ACTIVITY = "activity"
    TRANSFER = "transfer"

class Location(BaseModel):
    name: str
    iata_code: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class TripSegment(BaseModel):
    id: str
    type: Literal["flight", "hotel", "activity", "transfer"]
    title: str
    start_time: datetime
    end_time: datetime
    location: Location
    cost: float = Field(ge=0.0, default=0.0)
    currency: str = "USD"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_confirmed: bool = False

    @field_validator("end_time")
    @classmethod
    def validate_chronology(cls, v: datetime, values: Any) -> datetime:
        start = values.data.get("start_time")
        if start and v < start:
            raise ValueError("end_time must be strictly after start_time")
        return v

class TripConstraints(BaseModel):
    max_budget: Optional[float] = None
    min_connection_buffer_minutes: int = 90
    required_arrival_by: Optional[datetime] = None
    strict_dietary: List[str] = Field(default_factory=list)

class PendingAction(BaseModel):
    action_id: str
    action_type: Literal["BOOKING", "CANCELLATION", "PAYMENT", "RESCHEDULE"]
    description: str
    payload: Dict[str, Any]
    requires_explicit_approval: bool = True
    status: Literal["PENDING", "APPROVED", "REJECTED"] = "PENDING"

# ---------------------------------------------------------------------------
# LangGraph Runtime State
# ---------------------------------------------------------------------------

class ZicoGraphState(BaseModel):
    """The central state flowing across all nodes in the orchestrator."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    trip_id: str
    user_id: str
    itinerary: List[TripSegment] = Field(default_factory=list)
    constraints: TripConstraints = Field(default_factory=TripConstraints)
    pending_actions: List[PendingAction] = Field(default_factory=list)
    active_disruptions: List[Dict[str, Any]] = Field(default_factory=list)
    next_node: Optional[str] = None