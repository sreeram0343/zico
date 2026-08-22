from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
import uuid
from pydantic import BaseModel, Field
from app.graph.state import (
    ActionStatus,
    ActionType,
    PendingAction,
    SegmentType,
    TripConstraints,
    TripSegment,
)


class DisruptionEvent(BaseModel):
    """External disruption signal impacting an itinerary segment."""

    segment_id: str
    event_type: Literal["DELAY", "CANCELLATION", "MISSED_CONNECTION", "WEATHER"]
    delay_minutes: int = Field(default=0, ge=0)
    reason: str = Field(default="Unspecified flight disruption")


class DisruptionImpact(BaseModel):
    """Calculated cascade impact on the overall journey."""

    affected_segment_id: str
    impacted_downstream_segment_ids: List[str] = Field(default_factory=list)
    time_deficit_minutes: int = Field(default=0, ge=0)
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    summary: str


class RecoveryProposal(BaseModel):
    """Proposed recovery modification for traveler review."""

    proposal_id: str = Field(default_factory=lambda: f"rec_{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    proposed_new_segments: List[TripSegment] = Field(default_factory=list)
    segment_ids_to_remove: List[str] = Field(default_factory=list)
    estimated_cost_delta: float = 0.0
    currency: str = "USD"


def analyze_disruption(
    itinerary: List[TripSegment],
    event: DisruptionEvent,
    constraints: Optional[TripConstraints] = None,
) -> DisruptionImpact:
    """
    Computes ripple effects of a segment delay or cancellation across all downstream itinerary segments.
    """
    constraints = constraints or TripConstraints()
    sorted_segments = sorted(itinerary, key=lambda s: s.start_time)

    # Locate target segment
    target_idx = None
    for i, seg in enumerate(sorted_segments):
        if seg.id == event.segment_id:
            target_idx = i
            break

    if target_idx is None:
        return DisruptionImpact(
            affected_segment_id=event.segment_id,
            impacted_downstream_segment_ids=[],
            time_deficit_minutes=0,
            severity="LOW",
            summary=f"Segment {event.segment_id} not found in itinerary.",
        )

    target_seg = sorted_segments[target_idx]
    impacted_ids: List[str] = []
    max_deficit = 0

    if event.event_type == "CANCELLATION":
        # All subsequent segments dependent on arrival at this destination are impacted
        for next_seg in sorted_segments[target_idx + 1:]:
            impacted_ids.append(next_seg.id)
        return DisruptionImpact(
            affected_segment_id=target_seg.id,
            impacted_downstream_segment_ids=impacted_ids,
            time_deficit_minutes=0,
            severity="CRITICAL",
            summary=(
                f"Segment '{target_seg.title}' was cancelled ({event.reason}). "
                f"Cascading cancellation affects {len(impacted_ids)} downstream segments."
            ),
        )

    # DELAY or MISSED_CONNECTION
    delayed_end_time = target_seg.end_time + timedelta(minutes=event.delay_minutes)
    transit_types = {SegmentType.FLIGHT, SegmentType.TRANSFER}

    current_arrival = delayed_end_time
    for next_seg in sorted_segments[target_idx + 1:]:
        gap_seconds = (next_seg.start_time - current_arrival).total_seconds()
        gap_minutes = int(gap_seconds / 60)

        required_buffer = (
            constraints.min_connection_buffer_minutes
            if (target_seg.type in transit_types and next_seg.type in transit_types)
            else 0
        )

        if gap_minutes < required_buffer:
            deficit = required_buffer - gap_minutes
            max_deficit = max(max_deficit, deficit)
            impacted_ids.append(next_seg.id)
            # Propagate delay downstream for transit segments
            if next_seg.type in transit_types:
                current_arrival = next_seg.end_time + timedelta(minutes=deficit)
        else:
            # Buffer absorbed the delay
            break

    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    if len(impacted_ids) > 1 or max_deficit > 120:
        severity = "CRITICAL"
    elif len(impacted_ids) == 1 or max_deficit > 30:
        severity = "HIGH"
    elif max_deficit > 0:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    summary = (
        f"Delay of {event.delay_minutes}m on '{target_seg.title}' creates a buffer deficit of "
        f"{max_deficit}m affecting {len(impacted_ids)} downstream connection(s)."
        if impacted_ids
        else f"Delay of {event.delay_minutes}m on '{target_seg.title}' is safely absorbed by existing buffers."
    )

    return DisruptionImpact(
        affected_segment_id=target_seg.id,
        impacted_downstream_segment_ids=impacted_ids,
        time_deficit_minutes=max_deficit,
        severity=severity,
        summary=summary,
    )


def create_recovery_action(
    itinerary: List[TripSegment],
    impact: DisruptionImpact,
    event: DisruptionEvent,
    constraints: Optional[TripConstraints] = None,
) -> Optional[PendingAction]:
    """
    Constructs a structured PendingAction with proposed recovery options requiring traveler approval.
    """
    if impact.severity == "LOW" and not impact.impacted_downstream_segment_ids:
        return None

    action_id = f"act_{uuid.uuid4().hex[:8]}"
    affected_seg = next((s for s in itinerary if s.id == impact.affected_segment_id), None)
    seg_title = affected_seg.title if affected_seg else impact.affected_segment_id

    if event.event_type == "CANCELLATION":
        action_type = ActionType.CANCELLATION
        description = (
            f"Flight cancellation recovery for '{seg_title}'. Review rebooking and hotel options."
        )
        payload = {
            "disruption_type": "CANCELLATION",
            "affected_segment_id": impact.affected_segment_id,
            "impacted_segments": impact.impacted_downstream_segment_ids,
            "recommended_action": "REBOOK_NEXT_AVAILABLE",
            "eu261_eligible": True,
            "estimated_compensation": "€400-€600 per passenger if within EU regulation",
        }
    else:
        action_type = ActionType.RESCHEDULE
        description = (
            f"Reschedule connection for '{seg_title}' due to {event.delay_minutes}m delay."
        )
        payload = {
            "disruption_type": "DELAY",
            "affected_segment_id": impact.affected_segment_id,
            "time_deficit_minutes": impact.time_deficit_minutes,
            "impacted_segments": impact.impacted_downstream_segment_ids,
            "recommended_action": "AUTO_RECONNECT_TRANSFER",
            "proposed_buffer_extension_minutes": impact.time_deficit_minutes + 30,
        }

    return PendingAction(
        action_id=action_id,
        action_type=action_type,
        description=description,
        payload=payload,
        requires_explicit_approval=True,
        status=ActionStatus.PENDING,
    )


def apply_recovery_action(
    itinerary: List[TripSegment],
    action: PendingAction,
    approved_replacement_segments: Optional[List[TripSegment]] = None,
) -> List[TripSegment]:
    """
    Applies an approved pending recovery action, updating or replacing itinerary segments.
    """
    if action.status != ActionStatus.APPROVED:
        raise ValueError(f"Cannot apply action with status '{action.status}'. Must be APPROVED.")

    impacted_ids = set(action.payload.get("impacted_segments", []))
    affected_id = action.payload.get("affected_segment_id")
    if affected_id:
        impacted_ids.add(affected_id)

    # Filter out replaced segments if new ones are provided
    if approved_replacement_segments:
        updated_itinerary = [seg for seg in itinerary if seg.id not in impacted_ids]
        updated_itinerary.extend(approved_replacement_segments)
        return sorted(updated_itinerary, key=lambda s: s.start_time)

    # If simple delay shift without replacement list, return confirmed current state
    return itinerary
