from datetime import datetime, timedelta
import pytest

from app.graph.state import Location, SegmentType, TripConstraints, TripSegment
from app.graph.workers.disruption import (
    CollisionType,
    detect_downstream_collisions,
    disruption_reasoning_worker,
)


def _make_segment(seg_id: str, title: str, start: datetime, end: datetime, seg_type=SegmentType.FLIGHT) -> TripSegment:
    return TripSegment(
        id=seg_id,
        type=seg_type,
        title=title,
        start_time=start,
        end_time=end,
        location=Location(name="Airport", iata_code="XYZ"),
        cost=200.0,
    )


def test_detect_downstream_collisions_clean_schedule():
    """Verify clean itinerary with sufficient layover returns no collisions."""
    now = datetime(2026, 9, 15, 8, 0)
    seg1 = _make_segment("seg_1", "Leg 1: JFK -> LHR", now, now + timedelta(hours=7))  # 08:00 - 15:00
    seg2 = _make_segment("seg_2", "Leg 2: LHR -> CDG", now + timedelta(hours=9), now + timedelta(hours=10))  # 17:00 - 18:00 (120m buffer)

    itinerary = [seg1, seg2]
    constraints = TripConstraints(min_connection_buffer_minutes=90)

    collisions = detect_downstream_collisions(seg1, itinerary, constraints)
    assert len(collisions) == 0


def test_detect_downstream_collisions_insufficient_buffer():
    """Verify detection of connection gap below minimum required layover buffer."""
    now = datetime(2026, 9, 15, 8, 0)
    # Leg 1 delayed, arrives at 16:30 instead of 15:00
    seg1_delayed = _make_segment("seg_1", "Leg 1 (Delayed)", now, now + timedelta(hours=8, minutes=30))  # 08:00 - 16:30
    seg2 = _make_segment("seg_2", "Leg 2: LHR -> CDG", now + timedelta(hours=9), now + timedelta(hours=10))  # 17:00 - 18:00 (30m gap vs 90m req)

    itinerary = [seg1_delayed, seg2]
    constraints = TripConstraints(min_connection_buffer_minutes=90)

    collisions = detect_downstream_collisions(seg1_delayed, itinerary, constraints)
    assert len(collisions) == 1
    assert collisions[0].collision_type == CollisionType.INSUFFICIENT_BUFFER
    assert collisions[0].actual_gap_minutes == 30
    assert collisions[0].time_deficit_minutes == 60
    assert collisions[0].impacted_segment_id == "seg_2"


def test_detect_downstream_collisions_direct_overlap():
    """Verify detection of direct overlap where delayed flight arrives after next departure."""
    now = datetime(2026, 9, 15, 8, 0)
    # Leg 1 delayed significantly, arrives at 17:45 (past 17:00 departure of Leg 2)
    seg1_severe = _make_segment("seg_1", "Leg 1 (Severely Delayed)", now, now + timedelta(hours=9, minutes=45))  # 17:45
    seg2 = _make_segment("seg_2", "Leg 2: LHR -> CDG", now + timedelta(hours=9), now + timedelta(hours=10))  # 17:00

    itinerary = [seg1_severe, seg2]
    constraints = TripConstraints(min_connection_buffer_minutes=90)

    collisions = detect_downstream_collisions(seg1_severe, itinerary, constraints)
    assert len(collisions) == 1
    assert collisions[0].collision_type == CollisionType.DIRECT_OVERLAP
    assert collisions[0].severity == "CRITICAL"
    assert collisions[0].actual_gap_minutes == -45
    assert collisions[0].time_deficit_minutes == 135


def test_disruption_reasoning_worker_generates_pending_actions():
    """Verify disruption reasoning worker updates state with active disruptions and pending actions."""
    now = datetime(2026, 9, 15, 8, 0)
    seg1 = _make_segment("seg_1", "Leg 1", now, now + timedelta(hours=9, minutes=30))  # arrives 17:30
    seg2 = _make_segment("seg_2", "Leg 2", now + timedelta(hours=9), now + timedelta(hours=10))  # departs 17:00

    state = {
        "itinerary": [seg1, seg2],
        "constraints": TripConstraints(min_connection_buffer_minutes=90),
        "active_disruptions": [],
        "pending_actions": [],
    }

    result = disruption_reasoning_worker(state, modified_segment=seg1)
    assert len(result["active_disruptions"]) == 1
    assert len(result["pending_actions"]) == 1
    assert result["pending_actions"][0].requires_explicit_approval is True
    assert "Disruption Analysis Alert" in result["messages"][0].content
