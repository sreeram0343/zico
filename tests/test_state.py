"""
Unit tests for ZICO LangGraph state models (ZicoGraphState, TripSegment, Location, etc.).
"""

from datetime import datetime, timedelta
import pytest
from app.graph.state import (
    ActionStatus,
    ActionType,
    Location,
    PendingAction,
    SegmentType,
    TripConstraints,
    TripSegment,
    ZicoGraphState,
)


def test_zico_graph_state_creation():
    """Verify initialization of ZicoGraphState with default values."""
    state = ZicoGraphState(
        messages=[],
        trip_id="trip_123",
        user_id="user_456"
    )
    assert state.trip_id == "trip_123"
    assert state.user_id == "user_456"
    assert len(state.messages) == 0
    assert len(state.itinerary) == 0
    assert state.constraints.min_connection_buffer_minutes == 90
    assert len(state.pending_actions) == 0
    assert len(state.active_disruptions) == 0
    assert state.next_node is None


def test_trip_segment_chronology_validation():
    """Verify that end_time cannot precede or equal start_time."""
    start = datetime.now()
    invalid_end = start - timedelta(hours=1)
    loc = Location(name="JFK Airport", iata_code="JFK")

    with pytest.raises(ValueError, match="end_time must be strictly after start_time"):
        TripSegment(
            id="seg_1",
            type=SegmentType.FLIGHT,
            title="Flight to London",
            start_time=start,
            end_time=invalid_end,
            location=loc,
            cost=500.0,
        )

    # Equal start and end time should also raise ValueError
    with pytest.raises(ValueError, match="end_time must be strictly after start_time"):
        TripSegment(
            id="seg_2",
            type=SegmentType.FLIGHT,
            title="Flight to London",
            start_time=start,
            end_time=start,
            location=loc,
            cost=500.0,
        )


def test_valid_trip_segment_addition():
    """Verify creating a valid segment and adding it to the itinerary."""
    start = datetime.now()
    end = start + timedelta(hours=7)
    loc = Location(name="London Heathrow", iata_code="LHR")

    segment = TripSegment(
        id="seg_flight_01",
        type=SegmentType.FLIGHT,
        title="Flight BA178",
        start_time=start,
        end_time=end,
        location=loc,
        cost=750.00,
        currency="USD",
        metadata={"flight_number": "BA178"},
    )

    state = ZicoGraphState(
        messages=[],
        trip_id="trip_123",
        user_id="user_456",
        itinerary=[segment],
    )

    assert len(state.itinerary) == 1
    assert state.itinerary[0].title == "Flight BA178"
    assert state.itinerary[0].location.iata_code == "LHR"
    assert state.itinerary[0].cost == 750.00
    assert state.itinerary[0].type == SegmentType.FLIGHT


def test_pending_action_creation():
    """Verify creation and defaults of PendingAction."""
    action = PendingAction(
        action_id="act_01",
        action_type=ActionType.BOOKING,
        description="Book hotel in Paris",
        payload={"hotel_id": "h_123", "nights": 3},
    )
    assert action.action_id == "act_01"
    assert action.action_type == ActionType.BOOKING
    assert action.requires_explicit_approval is True
    assert action.status == ActionStatus.PENDING