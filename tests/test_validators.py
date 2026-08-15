"""
Unit tests for deterministic itinerary validation engine.
"""

from datetime import datetime, timedelta
import pytest
from app.graph.state import Location, SegmentType, TripConstraints, TripSegment
from app.graph.validators import (
    ItineraryConflict,
    detect_itinerary_conflicts,
    validate_budget_cap,
)


@pytest.fixture
def sample_location():
    return Location(name="JFK Airport", iata_code="JFK")


def test_detect_itinerary_direct_overlap(sample_location):
    """Test detecting direct temporal overlap between two flight segments."""
    base_time = datetime(2026, 9, 1, 10, 0)
    seg1 = TripSegment(
        id="flight_1",
        type=SegmentType.FLIGHT,
        title="Flight NYC -> LON",
        start_time=base_time,
        end_time=base_time + timedelta(hours=6),  # Ends at 16:00
        location=sample_location,
        cost=400.0,
    )
    seg2 = TripSegment(
        id="flight_2",
        type=SegmentType.FLIGHT,
        title="Flight LON -> PAR",
        start_time=base_time + timedelta(hours=5),  # Starts at 15:00 (1 hour overlap)
        end_time=base_time + timedelta(hours=7),
        location=sample_location,
        cost=150.0,
    )

    constraints = TripConstraints(min_connection_buffer_minutes=90)
    conflicts = detect_itinerary_conflicts([seg1, seg2], constraints)

    assert len(conflicts) == 1
    assert conflicts[0].segment_a_id == "flight_1"
    assert conflicts[0].segment_b_id == "flight_2"
    assert conflicts[0].deficit_minutes == 60
    assert "Direct time overlap" in conflicts[0].reason


def test_detect_itinerary_insufficient_layover(sample_location):
    """Test detecting connection buffer layover below min_connection_buffer_minutes."""
    base_time = datetime(2026, 9, 1, 10, 0)
    seg1 = TripSegment(
        id="flight_1",
        type=SegmentType.FLIGHT,
        title="Flight NYC -> LON",
        start_time=base_time,
        end_time=base_time + timedelta(hours=6),  # Ends at 16:00
        location=sample_location,
        cost=400.0,
    )
    seg2 = TripSegment(
        id="flight_2",
        type=SegmentType.FLIGHT,
        title="Flight LON -> PAR",
        start_time=base_time + timedelta(hours=6, minutes=45),  # Starts at 16:45 (45 min layover)
        end_time=base_time + timedelta(hours=8),
        location=sample_location,
        cost=150.0,
    )

    constraints = TripConstraints(min_connection_buffer_minutes=90)
    conflicts = detect_itinerary_conflicts([seg1, seg2], constraints)

    assert len(conflicts) == 1
    assert conflicts[0].segment_a_id == "flight_1"
    assert conflicts[0].segment_b_id == "flight_2"
    assert conflicts[0].deficit_minutes == 45  # 90 - 45 = 45 min deficit
    assert "Insufficient connection buffer" in conflicts[0].reason


def test_detect_itinerary_valid_schedule(sample_location):
    """Test that a valid itinerary with adequate layover reports zero conflicts."""
    base_time = datetime(2026, 9, 1, 10, 0)
    seg1 = TripSegment(
        id="flight_1",
        type=SegmentType.FLIGHT,
        title="Flight NYC -> LON",
        start_time=base_time,
        end_time=base_time + timedelta(hours=6),  # Ends at 16:00
        location=sample_location,
        cost=400.0,
    )
    seg2 = TripSegment(
        id="flight_2",
        type=SegmentType.FLIGHT,
        title="Flight LON -> PAR",
        start_time=base_time + timedelta(hours=8),  # Starts at 18:00 (120 min layover >= 90)
        end_time=base_time + timedelta(hours=9, minutes=30),
        location=sample_location,
        cost=150.0,
    )

    constraints = TripConstraints(min_connection_buffer_minutes=90)
    conflicts = detect_itinerary_conflicts([seg1, seg2], constraints)

    assert len(conflicts) == 0


def test_validate_budget_cap(sample_location):
    """Test budget cap validation function."""
    base_time = datetime(2026, 9, 1, 10, 0)
    segments = [
        TripSegment(
            id="s1",
            type=SegmentType.FLIGHT,
            title="Flight",
            start_time=base_time,
            end_time=base_time + timedelta(hours=2),
            location=sample_location,
            cost=300.0,
        ),
        TripSegment(
            id="s2",
            type=SegmentType.HOTEL,
            title="Hotel",
            start_time=base_time + timedelta(hours=3),
            end_time=base_time + timedelta(days=2),
            location=sample_location,
            cost=450.0,
        ),
    ]

    # Total = 750.0
    assert validate_budget_cap(segments, max_budget=1000.0) is True
    assert validate_budget_cap(segments, max_budget=750.0) is True
    assert validate_budget_cap(segments, max_budget=700.0) is False
    assert validate_budget_cap(segments, max_budget=None) is True
