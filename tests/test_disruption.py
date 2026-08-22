from datetime import datetime, timedelta
import pytest
from app.graph.disruption import (
    DisruptionEvent,
    DisruptionImpact,
    analyze_disruption,
    apply_recovery_action,
    create_recovery_action,
)
from app.graph.state import (
    ActionStatus,
    ActionType,
    Location,
    PendingAction,
    SegmentType,
    TripConstraints,
    TripSegment,
)


@pytest.fixture
def sample_multi_leg_itinerary():
    base_time = datetime(2026, 9, 20, 8, 0)
    seg1 = TripSegment(
        id="flight_leg_1",
        type=SegmentType.FLIGHT,
        title="JFK to LHR (Flight BA178)",
        start_time=base_time,
        end_time=base_time + timedelta(hours=7),  # Arrives 15:00
        location=Location(name="London Heathrow", iata_code="LHR"),
        cost=650.0,
    )
    seg2 = TripSegment(
        id="flight_leg_2",
        type=SegmentType.FLIGHT,
        title="LHR to CDG (Flight BA308)",
        start_time=base_time + timedelta(hours=8, minutes=45),  # 16:45 (105m layover)
        end_time=base_time + timedelta(hours=10),  # 18:00
        location=Location(name="Paris CDG", iata_code="CDG"),
        cost=150.0,
    )
    seg3 = TripSegment(
        id="hotel_paris",
        type=SegmentType.HOTEL,
        title="Parisian Boutique Hotel",
        start_time=base_time + timedelta(hours=11),  # 19:00
        end_time=base_time + timedelta(days=2),
        location=Location(name="Paris", iata_code="PAR"),
        cost=400.0,
    )
    return [seg1, seg2, seg3]


def test_analyze_disruption_safe_delay(sample_multi_leg_itinerary):
    """Verify that a minor delay within buffer does not create critical disruption."""
    event = DisruptionEvent(
        segment_id="flight_leg_1",
        event_type="DELAY",
        delay_minutes=10,  # 105m buffer - 10m = 95m > 90m buffer
        reason="Minor taxiway congestion",
    )
    impact = analyze_disruption(sample_multi_leg_itinerary, event)
    assert impact.severity == "LOW"
    assert len(impact.impacted_downstream_segment_ids) == 0


def test_analyze_disruption_buffer_deficit(sample_multi_leg_itinerary):
    """Verify that delay causing connection buffer deficit marks downstream flight."""
    event = DisruptionEvent(
        segment_id="flight_leg_1",
        event_type="DELAY",
        delay_minutes=45,  # 105m layover - 45m = 60m layover (< 90m required buffer)
        reason="Late inbound aircraft",
    )
    constraints = TripConstraints(min_connection_buffer_minutes=90)
    impact = analyze_disruption(sample_multi_leg_itinerary, event, constraints)

    assert impact.severity in ("HIGH", "CRITICAL")
    assert "flight_leg_2" in impact.impacted_downstream_segment_ids
    assert impact.time_deficit_minutes == 30  # 90 - 60 = 30m deficit


def test_analyze_disruption_cancellation(sample_multi_leg_itinerary):
    """Verify that segment cancellation invalidates all downstream segments."""
    event = DisruptionEvent(
        segment_id="flight_leg_1",
        event_type="CANCELLATION",
        delay_minutes=0,
        reason="Engine maintenance cancellation",
    )
    impact = analyze_disruption(sample_multi_leg_itinerary, event)

    assert impact.severity == "CRITICAL"
    assert "flight_leg_2" in impact.impacted_downstream_segment_ids
    assert "hotel_paris" in impact.impacted_downstream_segment_ids


def test_create_recovery_action_pending_approval(sample_multi_leg_itinerary):
    """Verify generation of structured PendingAction requiring user approval."""
    event = DisruptionEvent(
        segment_id="flight_leg_1",
        event_type="DELAY",
        delay_minutes=50,
        reason="Severe weather delay",
    )
    impact = analyze_disruption(sample_multi_leg_itinerary, event)
    action = create_recovery_action(sample_multi_leg_itinerary, impact, event)

    assert action is not None
    assert action.requires_explicit_approval is True
    assert action.status == ActionStatus.PENDING
    assert action.action_type == ActionType.RESCHEDULE
    assert action.payload["time_deficit_minutes"] > 0


def test_apply_recovery_action_workflow(sample_multi_leg_itinerary):
    """Verify applying approved recovery action replaces segments properly."""
    base_time = datetime(2026, 9, 20, 8, 0)
    replacement_flight = TripSegment(
        id="flight_leg_2_rebooked",
        type=SegmentType.FLIGHT,
        title="LHR to CDG (Flight AF1481 - Rebooked)",
        start_time=base_time + timedelta(hours=10),
        end_time=base_time + timedelta(hours=11, minutes=15),
        location=Location(name="Paris CDG", iata_code="CDG"),
        cost=175.0,
    )

    action = PendingAction(
        action_id="act_rebook_01",
        action_type=ActionType.RESCHEDULE,
        description="Rebook Paris flight",
        payload={"impacted_segments": ["flight_leg_2"]},
        requires_explicit_approval=True,
        status=ActionStatus.APPROVED,
    )

    new_itinerary = apply_recovery_action(
        sample_multi_leg_itinerary,
        action,
        approved_replacement_segments=[replacement_flight],
    )

    segment_ids = [s.id for s in new_itinerary]
    assert "flight_leg_2" not in segment_ids
    assert "flight_leg_2_rebooked" in segment_ids
    assert len(new_itinerary) == 3


def test_apply_recovery_action_unapproved_raises(sample_multi_leg_itinerary):
    """Verify applying pending action without user approval raises ValueError."""
    action = PendingAction(
        action_id="act_pending_01",
        action_type=ActionType.RESCHEDULE,
        description="Rebook flight",
        payload={},
        status=ActionStatus.PENDING,
    )
    with pytest.raises(ValueError, match="Must be APPROVED"):
        apply_recovery_action(sample_multi_leg_itinerary, action)
