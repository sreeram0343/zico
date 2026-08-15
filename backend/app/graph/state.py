"""
State Schema for ZICO Travel Operations System.

This module defines the core state data models used across LangGraph workflows,
FastAPI endpoints, and session management. ZICO treats travel as a state-based system
where every itinerary change, flight status update, or user preference is tracked in a
central source of truth (`TripState`).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SegmentType(str, Enum):
    """
    Enumeration of supported trip segment types.
    Inheriting from `str` allows for easy JSON serialization.
    """
    FLIGHT = "flight"
    HOTEL = "hotel"
    ACTIVITY = "activity"
    TRANSPORT = "transport"


class TripSegment(BaseModel):
    """
    Represents an individual component/reservation within a trip itinerary.
    
    Attributes:
        id: Unique identifier for the segment (e.g., "seg_flight_101").
        type: Category of the segment (FLIGHT, HOTEL, ACTIVITY, TRANSPORT).
        start_time: Scheduled start time (departure, check-in, event start).
        end_time: Scheduled end time (arrival, check-out, event end).
        location: Primary location or origin/destination label (e.g., "JFK -> LHR").
        details: Arbitrary metadata (flight numbers, confirmation codes, room type, etc.).
        is_confirmed: Confirmation status flag (default False until booking finalized).
    """
    id: str = Field(..., description="Unique identifier for the segment")
    type: SegmentType = Field(..., description="Type of segment (flight, hotel, activity, transport)")
    start_time: datetime = Field(..., description="Start date/time of the segment")
    end_time: datetime = Field(..., description="End date/time of the segment")
    location: str = Field(..., description="Location details (e.g., airport code, hotel address)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary (e.g., flight_no, confirmation_code)")
    is_confirmed: bool = Field(default=False, description="Booking confirmation status")


class UserPreferences(BaseModel):
    """
    Captures traveler preferences used by reasoning agents to tailor recommendations.
    
    Attributes:
        currency: Preferred currency code (default: "USD").
        preferred_airlines: List of preferred airline codes/names (e.g., ["AA", "DL"]).
        dietary_restrictions: List of dietary needs (e.g., ["Vegan", "Nut-free"]).
    """
    currency: str = Field(default="USD", description="ISO currency code for price display")
    preferred_airlines: List[str] = Field(default_factory=list, description="List of preferred airlines")
    dietary_restrictions: List[str] = Field(default_factory=list, description="List of dietary requirements")


class TripState(BaseModel):
    """
    Central State Object for ZICO Travel Operations System.
    
    Maintains full context for a traveler's journey and is passed through LangGraph nodes.
    
    Attributes:
        trip_id: Unique identifier for the overall trip session.
        user_id: Unique identifier for the user owning this trip.
        segments: Ordered list of trip segments (flights, hotels, activities).
        preferences: User preferences applied to route search and recommendations.
        constraints: Active constraints (e.g., budget limits, layover limits, time windows).
        pending_actions: High-impact actions awaiting Human-in-the-Loop verification.
    """
    trip_id: str = Field(..., description="Unique ID for the trip itinerary")
    user_id: str = Field(..., description="ID of the user associated with this trip")
    segments: List[TripSegment] = Field(default_factory=list, description="List of booked/planned trip segments")
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="User travel preferences")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Active reasoning constraints")
    pending_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Human-in-the-loop pending actions")
