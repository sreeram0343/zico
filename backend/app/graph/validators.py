from typing import List, Optional
from pydantic import BaseModel, Field
from app.graph.state import SegmentType, TripConstraints, TripSegment


class ItineraryConflict(BaseModel):
    segment_a_id: str
    segment_b_id: str
    reason: str
    deficit_minutes: int = Field(ge=0, default=0)


def detect_itinerary_conflicts(
    segments: List[TripSegment],
    constraints: TripConstraints,
) -> List[ItineraryConflict]:
    """
    Deterministic detection of temporal overlaps and insufficient connection buffer
    times without relying on LLM inference.
    """
    if len(segments) < 2:
        return []

    # Sort segments chronologically by start_time
    sorted_segments = sorted(segments, key=lambda s: s.start_time)
    conflicts: List[ItineraryConflict] = []

    transit_types = {SegmentType.FLIGHT, SegmentType.TRANSFER}

    for i in range(len(sorted_segments) - 1):
        seg_a = sorted_segments[i]
        seg_b = sorted_segments[i + 1]

        # 1. Direct Time Overlap Check
        if seg_a.end_time > seg_b.start_time:
            overlap_seconds = (seg_a.end_time - seg_b.start_time).total_seconds()
            deficit_minutes = max(1, int(overlap_seconds / 60))
            conflicts.append(
                ItineraryConflict(
                    segment_a_id=seg_a.id,
                    segment_b_id=seg_b.id,
                    reason=(
                        f"Direct time overlap between '{seg_a.title}' "
                        f"and '{seg_b.title}' ({deficit_minutes}m overlap)"
                    ),
                    deficit_minutes=deficit_minutes,
                )
            )
        # 2. Connection Buffer Check between Transit Segments (Flight/Transfer)
        elif seg_a.type in transit_types and seg_b.type in transit_types:
            gap_seconds = (seg_b.start_time - seg_a.end_time).total_seconds()
            gap_minutes = int(gap_seconds / 60)
            if gap_minutes < constraints.min_connection_buffer_minutes:
                deficit_minutes = (
                    constraints.min_connection_buffer_minutes - gap_minutes
                )
                conflicts.append(
                    ItineraryConflict(
                        segment_a_id=seg_a.id,
                        segment_b_id=seg_b.id,
                        reason=(
                            f"Insufficient connection buffer between '{seg_a.title}' "
                            f"and '{seg_b.title}' ({gap_minutes}m layover < "
                            f"{constraints.min_connection_buffer_minutes}m minimum)"
                        ),
                        deficit_minutes=deficit_minutes,
                    )
                )

    return conflicts


def validate_budget_cap(
    segments: List[TripSegment],
    max_budget: Optional[float],
) -> bool:
    """
    Checks whether the total cost of all segments exceeds the specified budget cap.
    """
    if max_budget is None:
        return True

    total_cost = sum(seg.cost for seg in segments)
    return total_cost <= max_budget
