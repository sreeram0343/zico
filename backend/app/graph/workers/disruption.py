from datetime import datetime, timedelta, timezone

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import uuid
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.graph.state import (
    ActionStatus,
    ActionType,
    PendingAction,
    SegmentType,
    TripConstraints,
    TripSegment,
    ZicoGraphState,
)


class CollisionType(str, Enum):
    DIRECT_OVERLAP = "DIRECT_OVERLAP"
    INSUFFICIENT_BUFFER = "INSUFFICIENT_BUFFER"
    HOTEL_CHECKIN_CONFLICT = "HOTEL_CHECKIN_CONFLICT"
    ACTIVITY_MISSED = "ACTIVITY_MISSED"


class SchedulingCollision(BaseModel):
    """Structured representation of a detected downstream itinerary collision."""

    collision_id: str = Field(default_factory=lambda: f"col_{uuid.uuid4().hex[:8]}")
    collision_type: CollisionType
    trigger_segment_id: str
    impacted_segment_id: str
    impacted_segment_title: str
    actual_gap_minutes: int
    required_buffer_minutes: int
    time_deficit_minutes: int
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    description: str
    suggested_adjustment: str


class DisruptionWorkerResult(BaseModel):
    """Structured summary returned by the disruption reasoning worker."""

    has_collisions: bool
    total_conflicts_count: int
    collisions: List[SchedulingCollision] = Field(default_factory=list)
    max_time_deficit_minutes: int = 0
    summary: str = ""


def _find_downstream_chain(
    current_segment: TripSegment,
    all_segments: List[TripSegment],
) -> List[TripSegment]:
    """Returns all itinerary segments scheduled strictly after current_segment start time."""
    sorted_segments = sorted(all_segments, key=lambda s: s.start_time)
    downstream: List[TripSegment] = []
    for s in sorted_segments:
        if s.id != current_segment.id and s.start_time >= current_segment.start_time:
            downstream.append(s)
    return downstream


def detect_downstream_collisions(
    modified_segment: TripSegment,
    itinerary: List[TripSegment],
    constraints: Optional[TripConstraints] = None,
) -> List[SchedulingCollision]:
    """
    Recursively walks all downstream segments in the itinerary following a modified segment
    (e.g., delayed flight arrival or rescheduled departure) and identifies scheduling collisions:
    - Direct time overlaps (arrival after next departure)
    - Insufficient layover buffers (< min_connection_buffer_minutes)
    - Activity/Hotel check-in missed windows

    Recursively cascades the time shift downstream to check if subsequent legs also collide.
    """
    if not itinerary:
        return []

    min_buffer = constraints.min_connection_buffer_minutes if constraints else 90
    downstream_segments = _find_downstream_chain(modified_segment, itinerary)
    collisions: List[SchedulingCollision] = []

    # Current projected arrival time propagating downstream
    projected_departure = modified_segment.end_time
    last_trigger_id = modified_segment.id

    for downstream in downstream_segments:
        gap_minutes = int((downstream.start_time - projected_departure).total_seconds() / 60)

        # 1. Direct Overlap: arrival is strictly past downstream start time
        if gap_minutes < 0:
            time_deficit = abs(gap_minutes) + min_buffer
            collision = SchedulingCollision(
                collision_type=CollisionType.DIRECT_OVERLAP,
                trigger_segment_id=last_trigger_id,
                impacted_segment_id=downstream.id,
                impacted_segment_title=downstream.title,
                actual_gap_minutes=gap_minutes,
                required_buffer_minutes=min_buffer,
                time_deficit_minutes=time_deficit,
                severity="CRITICAL",
                description=(
                    f"Direct scheduling collision: Arrival of previous leg ({projected_departure.strftime('%H:%M')}) "
                    f"is {abs(gap_minutes)}m past the departure of '{downstream.title}' ({downstream.start_time.strftime('%H:%M')})."
                ),
                suggested_adjustment=f"Rebook '{downstream.title}' to a departure after {(projected_departure + timedelta(minutes=min_buffer)).strftime('%Y-%m-%d %H:%M')}.",
            )
            collisions.append(collision)
            # Propagate delay to next leg for recursive collision cascade
            projected_departure = downstream.end_time + timedelta(minutes=time_deficit)
            last_trigger_id = downstream.id

        # 2. Insufficient Buffer for transit connections
        elif gap_minutes < min_buffer:
            time_deficit = min_buffer - gap_minutes
            severity = "HIGH" if downstream.type == SegmentType.FLIGHT else "MEDIUM"
            collision = SchedulingCollision(
                collision_type=CollisionType.INSUFFICIENT_BUFFER,
                trigger_segment_id=last_trigger_id,
                impacted_segment_id=downstream.id,
                impacted_segment_title=downstream.title,
                actual_gap_minutes=gap_minutes,
                required_buffer_minutes=min_buffer,
                time_deficit_minutes=time_deficit,
                severity=severity,
                description=(
                    f"Insufficient connection buffer: Connection gap ({gap_minutes}m) between arrival "
                    f"and '{downstream.title}' is below required {min_buffer}m minimum."
                ),
                suggested_adjustment=f"Adjust connection or rebook '{downstream.title}' with at least {min_buffer}m buffer.",
            )
            collisions.append(collision)
            # Propagate buffer shift
            projected_departure = downstream.end_time + timedelta(minutes=time_deficit)
            last_trigger_id = downstream.id

        else:
            # Clean connection, update projected departure to downstream's normal end time
            projected_departure = downstream.end_time
            last_trigger_id = downstream.id

    return collisions


def disruption_reasoning_worker(
    state: ZicoGraphState | Dict[str, Any],
    modified_segment: Optional[TripSegment] = None,
) -> Dict[str, Any]:
    """
    Disruption reasoning worker node:
    Evaluates itinerary state, walks downstream segments, detects collisions,
    and formulates structured pending actions for traveler confirmation.
    """
    if isinstance(state, dict):
        itinerary = state.get("itinerary", [])
        constraints = state.get("constraints")
        active_disruptions = list(state.get("active_disruptions", []))
        pending_actions = list(state.get("pending_actions", []))
    else:
        itinerary = getattr(state, "itinerary", [])
        constraints = getattr(state, "constraints", None)
        active_disruptions = list(getattr(state, "active_disruptions", []))
        pending_actions = list(getattr(state, "pending_actions", []))

    if not itinerary:
        return {
            "messages": [AIMessage(content="No active itinerary segments found to analyze for disruptions.")],
            "active_disruptions": active_disruptions,
            "pending_actions": pending_actions,
        }

    # If no explicit modified segment provided, analyze latest disrupted segment or first segment
    target_segment = modified_segment or itinerary[0]

    collisions = detect_downstream_collisions(target_segment, itinerary, constraints)

    if not collisions:
        summary_msg = f"Schedule analysis clean: No cascading collisions detected for itinerary '{target_segment.title}'."
        return {
            "messages": [AIMessage(content=summary_msg)],
            "active_disruptions": active_disruptions,
            "pending_actions": pending_actions,
        }

    # Record active disruption event
    disruption_event = {
        "event_id": f"disrupt_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_segment": target_segment.id,
        "collisions_count": len(collisions),
        "collisions": [c.model_dump() for c in collisions],
    }

    active_disruptions.append(disruption_event)

    # Propose PendingAction for each critical collision
    for col in collisions:
        if col.severity in ("CRITICAL", "HIGH"):
            action = PendingAction(
                action_id=f"act_{uuid.uuid4().hex[:6]}",
                action_type=ActionType.RESCHEDULE,
                description=f"Resolve {col.collision_type.value}: {col.suggested_adjustment}",
                payload={
                    "collision_id": col.collision_id,
                    "impacted_segment_id": col.impacted_segment_id,
                    "time_deficit_minutes": col.time_deficit_minutes,
                    "suggested_adjustment": col.suggested_adjustment,
                },
                requires_explicit_approval=True,
                status=ActionStatus.PENDING,
            )
            pending_actions.append(action)

    formatted_collisions = "\n".join(
        f"- **[{c.severity}] {c.impacted_segment_title}**: {c.description} (Deficit: {c.time_deficit_minutes}m)"
        for c in collisions
    )

    response_text = (
        f"🚨 **Disruption Analysis Alert**: Detected {len(collisions)} cascading scheduling conflict(s):\n\n"
        f"{formatted_collisions}\n\n"
        f"I have created {len(collisions)} recovery action proposal(s) requiring your explicit confirmation."
    )

    return {
        "messages": [AIMessage(content=response_text)],
        "active_disruptions": active_disruptions,
        "pending_actions": pending_actions,
    }
