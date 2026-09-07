from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Enums & Value Objects
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
    name: str = Field(..., description="Location or airport display name.")
    iata_code: Optional[str] = Field(default=None, description="Standard 3-letter IATA airport code if applicable.")
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Geographic latitude in degrees.")
    lng: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Geographic longitude in degrees.")

    @field_validator("iata_code", mode="before")
    @classmethod
    def clean_iata(cls, v: Any) -> Optional[str]:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            return cleaned if cleaned else None
        return v


# ---------------------------------------------------------------------------
# User Preferences Schema
# ---------------------------------------------------------------------------


class UserPreferences(BaseModel):
    """Traveler preferences governing flight, hotel, and itinerary synthesis."""

    preferred_airlines: List[str] = Field(default_factory=list, description="Airlines preferred by traveler.")
    preferred_cabin_class: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = Field(
        default="ECONOMY",
        description="Default seating cabin class tier.",
    )
    seating_preference: Literal["WINDOW", "AISLE", "MIDDLE", "ANY"] = Field(
        default="ANY",
        description="Seating location preference.",
    )
    max_layover_hours: float = Field(
        default=8.0,
        ge=0.0,
        le=48.0,
        description="Maximum acceptable layover duration in hours.",
    )
    direct_flights_only: bool = Field(
        default=False,
        description="Whether to restrict flight searches to non-stop direct flights.",
    )
    hotel_rating_min: float = Field(
        default=3.5,
        ge=1.0,
        le=5.0,
        description="Minimum star rating for hotel accommodations.",
    )
    dietary_requirements: List[str] = Field(
        default_factory=list,
        description="Special dietary guidelines (e.g. Vegetarian, Halal, Kosher, Gluten-Free).",
    )
    loyalty_programs: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of airline/hotel brand to traveler membership number.",
    )
    currency: str = Field(default="USD", min_length=3, max_length=3, description="Preferred 3-letter currency code.")
    home_airport: Optional[str] = Field(default=None, description="Primary home/origin airport IATA code.")

    @field_validator("currency", mode="before")
    @classmethod
    def validate_currency(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if len(cleaned) == 3 and cleaned.isalpha():
                return cleaned
            raise ValueError(f"Currency must be a 3-letter ISO code, received '{v}'")
        return "USD"

    @field_validator("home_airport", mode="before")
    @classmethod
    def validate_home_airport(cls, v: Any) -> Optional[str]:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned and (len(cleaned) != 3 or not cleaned.isalpha()):
                raise ValueError(f"Home airport must be a 3-letter IATA code, received '{v}'")
            return cleaned if cleaned else None
        return v

    @field_validator("preferred_cabin_class", "seating_preference", mode="before")
    @classmethod
    def normalize_literals(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip().upper()
        return v


# ---------------------------------------------------------------------------
# Trip Segment Domain Schema
# ---------------------------------------------------------------------------


class TripSegment(BaseModel):
    """Represents an atomic, validated segment in a travel itinerary."""

    id: str = Field(..., min_length=1, description="Unique identifier for the segment.")
    type: SegmentType = Field(..., description="Category of travel segment.")
    title: str = Field(..., min_length=1, description="Human-readable title (e.g. flight number or hotel name).")
    start_time: datetime = Field(..., description="Departure, check-in, or commencement datetime.")
    end_time: datetime = Field(..., description="Arrival, check-out, or conclusion datetime.")
    location: Location = Field(..., description="Primary geographic destination or arrival location.")
    cost: float = Field(default=0.0, ge=0.0, description="Total monetary cost for this segment.")
    currency: str = Field(default="USD", min_length=3, max_length=3, description="3-letter currency code.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary domain-specific metadata.")
    is_confirmed: bool = Field(default=False, description="Whether booking/reservation is fully confirmed.")

    @field_validator("id", "title", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("String identifier cannot be blank.")
            return cleaned
        return str(v)

    @field_validator("currency", mode="before")
    @classmethod
    def validate_currency(cls, v: Any) -> str:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if len(cleaned) == 3 and cleaned.isalpha():
                return cleaned
        return "USD"

    @field_validator("end_time")
    @classmethod
    def validate_chronology(cls, v: datetime, values: Any) -> datetime:
        start = values.data.get("start_time")
        if start and v <= start:
            raise ValueError("end_time must be strictly after start_time")
        return v

    @property
    def duration_minutes(self) -> int:
        """Computes segment elapsed duration in integer minutes."""
        diff = self.end_time - self.start_time
        return max(0, int(diff.total_seconds() / 60))


# ---------------------------------------------------------------------------
# Trip Constraints & Pending Actions
# ---------------------------------------------------------------------------


class TripConstraints(BaseModel):
    max_budget: Optional[float] = Field(default=None, ge=0.0, description="Upper bound budget for journey.")
    min_connection_buffer_minutes: int = Field(default=90, ge=0, description="Minimum layover buffer minutes.")
    required_arrival_by: Optional[datetime] = Field(default=None, description="Hard deadline for final arrival.")
    strict_dietary: List[str] = Field(default_factory=list, description="Strict dietary constraints.")


class PendingAction(BaseModel):
    action_id: str = Field(..., min_length=1)
    action_type: ActionType
    description: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    requires_explicit_approval: bool = True
    status: ActionStatus = ActionStatus.PENDING


# ---------------------------------------------------------------------------
# Authoritative Trip Domain State (TripState)
# ---------------------------------------------------------------------------


class TripState(BaseModel):
    """
    Authoritative domain state representing an active, planned, or completed journey.
    Provides strict chronological validation across all contained segments.
    """

    trip_id: str = Field(..., min_length=1, description="Unique primary identifier for the trip.")
    user_id: str = Field(..., min_length=1, description="Owner user identifier.")
    title: Optional[str] = Field(default="Untitled Journey", description="Display name for the trip.")
    itinerary: List[TripSegment] = Field(default_factory=list, description="Chronological sequence of trip segments.")
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="Traveler preferences.")
    constraints: TripConstraints = Field(default_factory=TripConstraints, description="Operational constraints.")
    status: Literal["PLANNING", "CONFIRMED", "IN_PROGRESS", "COMPLETED", "CANCELLED", "DISRUPTED"] = Field(
        default="PLANNING",
        description="Lifecycle status of journey.",
    )
    total_cost: float = Field(default=0.0, ge=0.0, description="Aggregated cost across all segments.")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def compute_aggregates_and_validate(self) -> TripState:
        # Calculate total cost automatically if not explicitly overridden
        computed_cost = sum(seg.cost for seg in self.itinerary)
        if self.total_cost == 0.0 and computed_cost > 0.0:
            self.total_cost = round(computed_cost, 2)
        return self

    @property
    def start_date(self) -> Optional[datetime]:
        """Returns commencement datetime of earliest segment in itinerary."""
        if not self.itinerary:
            return None
        return min(seg.start_time for seg in self.itinerary)

    @property
    def end_date(self) -> Optional[datetime]:
        """Returns conclusion datetime of latest segment in itinerary."""
        if not self.itinerary:
            return None
        return max(seg.end_time for seg in self.itinerary)

    def add_segment(self, segment: TripSegment) -> None:
        """Appends a new TripSegment, maintains timestamp, and recomputes total cost."""
        self.itinerary.append(segment)
        self.total_cost = round(sum(s.cost for s in self.itinerary), 2)
        self.updated_at = datetime.now(timezone.utc)

    def remove_segment(self, segment_id: str) -> bool:
        """Removes segment by identifier. Returns True if removed, False if not found."""
        initial_len = len(self.itinerary)
        self.itinerary = [s for s in self.itinerary if s.id != segment_id]
        if len(self.itinerary) < initial_len:
            self.total_cost = round(sum(s.cost for s in self.itinerary), 2)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def has_temporal_overlap(self) -> List[Tuple[TripSegment, TripSegment]]:
        """
        Inspects itinerary for any temporal collisions where two segments of the
        same type (e.g. overlapping flights) occur concurrently.
        """
        overlaps: List[Tuple[TripSegment, TripSegment]] = []
        sorted_segs = sorted(self.itinerary, key=lambda s: s.start_time)
        for i in range(len(sorted_segs)):
            for j in range(i + 1, len(sorted_segs)):
                s1 = sorted_segs[i]
                s2 = sorted_segs[j]
                # If s2 starts before s1 ends, temporal overlap detected
                if s2.start_time < s1.end_time and s1.type == s2.type:
                    overlaps.append((s1, s2))
        return overlaps

    def get_segments_by_type(self, seg_type: SegmentType) -> List[TripSegment]:
        """Filters itinerary segments matching the specified segment type."""
        return [s for s in self.itinerary if s.type == seg_type]


# ---------------------------------------------------------------------------
# LangGraph Runtime State
# ---------------------------------------------------------------------------


class ZicoGraphState(BaseModel):
    """The central state flowing across all nodes in the orchestrator graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list
    )
    trip_id: str
    user_id: str
    itinerary: List[TripSegment] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    constraints: TripConstraints = Field(default_factory=TripConstraints)
    pending_actions: List[PendingAction] = Field(default_factory=list)
    active_disruptions: List[Dict[str, Any]] = Field(default_factory=list)
    next_node: Optional[str] = None

    def to_trip_state(self, title: str = "Active Journey") -> TripState:
        """Converts active graph runtime state into an authoritative TripState domain model."""
        return TripState(
            trip_id=self.trip_id,
            user_id=self.user_id,
            title=title,
            itinerary=list(self.itinerary),
            preferences=self.preferences,
            constraints=self.constraints,
            status="PLANNING",
        )