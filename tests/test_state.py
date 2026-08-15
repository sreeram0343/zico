"""
Unit tests for the ZICO TripState and TripSegment state models.

Tests state creation, segment manipulation, default preferences, and schema validation.
"""

from datetime import datetime, timedelta
import pytest
from app.graph.state import TripState, TripSegment, SegmentType, UserPreferences


def test_trip_state_creation():
    """Verify that a TripState instance initializes with correct defaults."""
    state = TripState(trip_id="123", user_id="user_1")
    
    # Assert primary identifiers
    assert state.trip_id == "123"
    assert state.user_id == "user_1"
    
    # Assert default empty state attributes
    assert len(state.segments) == 0
    assert state.preferences.currency == "USD"
    assert len(state.preferences.preferred_airlines) == 0
    assert len(state.constraints) == 0
    assert len(state.pending_actions) == 0


def test_add_segment():
    """Verify adding a TripSegment to TripState."""
    now = datetime.now()
    state = TripState(trip_id="123", user_id="user_1")
    
    # Construct a sample flight segment
    seg = TripSegment(
        id="s1",
        type=SegmentType.FLIGHT,
        start_time=now,
        end_time=now + timedelta(hours=5),
        location="NYC -> LHR",
        details={"flight_no": "AA123", "airline": "American Airlines"}
    )
    
    # Append segment to state
    state.segments.append(seg)
    
    # Assert segment properties
    assert len(state.segments) == 1
    assert state.segments[0].id == "s1"
    assert state.segments[0].type == SegmentType.FLIGHT
    assert state.segments[0].details["flight_no"] == "AA123"
    assert state.segments[0].is_confirmed is False


def test_user_preferences_customization():
    """Verify setting custom user preferences within TripState."""
    prefs = UserPreferences(
        currency="EUR",
        preferred_airlines=["BA", "LH"],
        dietary_restrictions=["Vegetarian"]
    )
    state = TripState(trip_id="456", user_id="user_2", preferences=prefs)
    
    assert state.preferences.currency == "EUR"
    assert "BA" in state.preferences.preferred_airlines
    assert "Vegetarian" in state.preferences.dietary_restrictions