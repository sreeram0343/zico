from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Domain Models (Strict Pydantic)
# ---------------------------------------------------------------------------


class SegmentType(str, Enum):
    FLIGHT = "FLIGHT"
    HOTEL = "HOTEL"
    ACTIVITY = "ACTIVITY"
    TRANSFER = "TRANSFER"

    @classmethod
    def _missing_(cls, value: object) -> Optional[SegmentType]:
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class ActionType(str, Enum):
    BOOKING = "BOOKING"
    CANCELLATION = "CANCELLATION"
    PAYMENT = "PAYMENT"
    RESCHEDULE = "RESCHEDULE"

    @classmethod
    def _missing_(cls, value: object) -> Optional[ActionType]:
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

    @classmethod
    def _missing_(cls, value: object) -> Optional[ActionStatus]:
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class Location(BaseModel):
    name: str
    iata_code: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class TripSegment(BaseModel):
    id: str
    type: SegmentType
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
        if start and v <= start:
            raise ValueError("end_time must be strictly after start_time")
        return v


class TripConstraints(BaseModel):
    max_budget: Optional[float] = None
    min_connection_buffer_minutes: int = 90
    required_arrival_by: Optional[datetime] = None
    strict_dietary: List[str] = Field(default_factory=list)


class PendingAction(BaseModel):
    action_id: str
    action_type: ActionType
    description: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    requires_explicit_approval: bool = True
    status: ActionStatus = ActionStatus.PENDING


# ---------------------------------------------------------------------------
# LangGraph Runtime State
# ---------------------------------------------------------------------------


class ZicoGraphState(BaseModel):
    """The central state flowing across all nodes in the orchestrator."""

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )
    trip_id: str
    user_id: str
    itinerary: List[TripSegment] = Field(default_factory=list)
    constraints: TripConstraints = Field(default_factory=TripConstraints)
    pending_actions: List[PendingAction] = Field(default_factory=list)
    active_disruptions: List[Dict[str, Any]] = Field(default_factory=list)
    next_node: Optional[str] = None