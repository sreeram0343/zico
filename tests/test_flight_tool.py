"""
Unit tests for backend/app/tools/flight_tool.py
"""

from datetime import datetime, timedelta
import sys
import os
os.environ["QDRANT_URL"] = ":memory:"
from unittest.mock import patch, MagicMock
import pytest

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.graph.state import SegmentType, TripSegment
from app.tools.flight_tool import (
    FlightToolValidationError,
    get_airport_info,
    get_flight_status,
    lookup_airport,
    parse_flight_to_trip_segment,
    resolve_airport_code,
    resolve_country_code,
    search_airports,
    search_flights_aviationstack,
    track_flight_disruption,
    track_flight_status,
    aviationstack_flight_search,
)


def test_resolve_country_code():
    """Verify ISO alpha-2 country resolution using pycountry."""
    assert resolve_country_code("Bangladesh") == "BD"
    assert resolve_country_code("United States") == "US"
    assert resolve_country_code("Germany") == "DE"
    assert resolve_country_code("India") == "IN"
    assert resolve_country_code("GB") == "GB"
    assert resolve_country_code("USA") == "US"
    assert resolve_country_code("") is None


def test_resolve_airport_code_iata_and_icao():
    """Verify direct resolution of IATA and ICAO airport codes."""
    assert resolve_airport_code("DAC") == "DAC"
    assert resolve_airport_code("jfk") == "JFK"
    assert resolve_airport_code("LHR") == "LHR"
    assert resolve_airport_code("BOM") == "BOM"
    # ICAO to IATA
    assert resolve_airport_code("KJFK") == "JFK"
    assert resolve_airport_code("VGHS") == "DAC"
    assert resolve_airport_code("EGLL") == "LHR"


def test_resolve_airport_code_city_names():
    """Verify city name resolution to top international hub."""
    assert resolve_airport_code("Dhaka") == "DAC"
    assert resolve_airport_code("London") == "LHR"
    assert resolve_airport_code("New York") == "JFK"
    assert resolve_airport_code("Tokyo") == "HND"
    assert resolve_airport_code("Mumbai") == "BOM"
    assert resolve_airport_code("Paris") == "CDG"


def test_resolve_airport_code_city_and_country():
    """Verify resolution of 'City, Country' query strings."""
    assert resolve_airport_code("Dhaka, Bangladesh") == "DAC"
    assert resolve_airport_code("London, United Kingdom") == "LHR"


def test_resolve_airport_code_airport_name():
    """Verify resolution of airport names."""
    assert resolve_airport_code("Heathrow") == "LHR"
    assert resolve_airport_code("Hazrat Shahjalal") == "DAC"


def test_resolve_airport_code_fallback():
    """Verify fallback to default or DEFAULT_ORIGIN_IATA."""
    assert resolve_airport_code(None, default="DAC") == "DAC"
    assert resolve_airport_code("", default="SFO") == "SFO"


def test_get_airport_info():
    """Verify retrieval and enrichment of airport metadata."""
    info = get_airport_info("DAC")
    assert info is not None
    assert info["iata"] == "DAC"
    assert info["city"] == "Dhaka"
    assert info["country"] == "BD"
    assert info["country_name"] == "Bangladesh"
    assert info["lat"] is not None
    assert info["lon"] is not None
    assert info["tz"] == "Asia/Dhaka"


def test_search_airports():
    """Verify airport search functionality."""
    results = search_airports("Dhaka", limit=3)
    assert len(results) >= 1
    assert any(r["iata"] == "DAC" for r in results)


def test_parse_flight_to_trip_segment_success():
    """Verify transformation of raw AviationStack flight JSON into TripSegment."""
    mock_flight_entry = {
        "flight_date": "2026-09-08",
        "flight_status": "scheduled",
        "departure": {
            "airport": "Hazrat Shahjalal International Airport",
            "timezone": "Asia/Dhaka",
            "iata": "DAC",
            "icao": "VGHS",
            "terminal": "2",
            "gate": "4",
            "delay": 10,
            "scheduled": "2026-09-08T08:00:00+00:00",
            "estimated": "2026-09-08T08:10:00+00:00",
            "actual": None,
        },
        "arrival": {
            "airport": "Heathrow",
            "timezone": "Europe/London",
            "iata": "LHR",
            "icao": "EGLL",
            "terminal": "4",
            "gate": "12",
            "baggage": "6",
            "scheduled": "2026-09-08T18:30:00+00:00",
            "delay": None,
            "estimated": None,
            "actual": None,
        },
        "airline": {
            "name": "Biman Bangladesh Airlines",
            "iata": "BG",
            "icao": "BBC",
        },
        "flight": {
            "number": "201",
            "iata": "BG201",
            "icao": "BBC201",
        },
        "aircraft": {
            "registration": "S2-AHV",
            "iata": "B788",
        },
        "live": None,
    }

    seg = parse_flight_to_trip_segment(mock_flight_entry, default_cost=750.0, currency="USD")
    assert isinstance(seg, TripSegment)
    assert seg.type == SegmentType.FLIGHT
    assert "Biman Bangladesh Airlines" in seg.title
    assert "BG201" in seg.title
    assert seg.location.iata_code == "LHR"
    assert seg.location.lat is not None
    assert seg.location.lng is not None
    assert seg.cost == 750.0
    assert seg.currency == "USD"
    assert seg.start_time < seg.end_time
    assert seg.metadata["departure_iata"] == "DAC"
    assert seg.metadata["departure_terminal"] == "2"
    assert seg.metadata["arrival_gate"] == "12"
    assert seg.metadata["arrival_baggage"] == "6"
    assert seg.metadata["departure_delay_minutes"] == 10
    assert seg.metadata["flight_status"] == "scheduled"


def test_parse_flight_invalid_chronology_raises_error():
    """Verify that invalid chronology raises FlightToolValidationError."""
    invalid_flight_entry = {
        "departure": {
            "iata": "DAC",
            "scheduled": "2026-09-08T18:00:00+00:00",
        },
        "arrival": {
            "iata": "LHR",
            "scheduled": "2026-09-07T10:00:00+00:00",  # Prior day
        },
        "flight": {"iata": "BG201"},
        "airline": {"name": "Biman"},
    }
    with pytest.raises(FlightToolValidationError, match="arrival.*must be strictly after departure"):
        parse_flight_to_trip_segment(invalid_flight_entry)


@patch("app.tools.flight_tool.query_aviationstack")
def test_search_flights_aviationstack(mock_query):
    """Verify route search query and parsing."""
    mock_query.return_value = {
        "data": [
            {
                "flight_status": "scheduled",
                "departure": {
                    "airport": "Hazrat Shahjalal",
                    "iata": "DAC",
                    "scheduled": "2026-09-08T10:00:00+00:00",
                },
                "arrival": {
                    "airport": "Heathrow",
                    "iata": "LHR",
                    "scheduled": "2026-09-08T20:00:00+00:00",
                },
                "airline": {"name": "British Airways", "iata": "BA"},
                "flight": {"iata": "BA144"},
            }
        ]
    }

    results = search_flights_aviationstack(departure="Dhaka", arrival="London")
    assert len(results) == 1
    assert results[0].location.iata_code == "LHR"
    assert "BA144" in results[0].title


@patch("app.tools.flight_tool.query_aviationstack")
def test_get_flight_status(mock_query):
    """Verify real-time flight status query."""
    mock_query.return_value = {
        "data": [
            {
                "flight_status": "active",
                "flight_date": "2026-09-08",
                "departure": {
                    "airport": "JFK",
                    "iata": "JFK",
                    "terminal": "8",
                    "gate": "12",
                    "scheduled": "2026-09-08T12:00:00+00:00",
                    "delay": 25,
                },
                "arrival": {
                    "airport": "LHR",
                    "iata": "LHR",
                    "terminal": "3",
                    "gate": "40",
                    "baggage": "8",
                    "scheduled": "2026-09-08T23:45:00+00:00",
                    "delay": 20,
                },
                "airline": {"name": "American Airlines"},
                "flight": {"iata": "AA100"},
            }
        ]
    }

    status = get_flight_status("AA100")
    assert status["flight_number"] == "AA100"
    assert status["status"] == "active"
    assert status["departure"]["delay_minutes"] == 25
    assert status["arrival"]["terminal"] == "3"


@patch("app.tools.flight_tool.get_flight_status")
def test_track_flight_disruption_detected(mock_status):
    """Verify flight disruption detection for delay > 15m."""
    mock_status.return_value = {
        "flight_number": "AA100",
        "status": "active",
        "departure": {"iata": "JFK", "delay_minutes": 45},
        "arrival": {"iata": "LHR", "delay_minutes": 40},
    }

    disruption = track_flight_disruption("AA100")
    assert disruption is not None
    assert disruption["event_type"] == "DELAY"
    assert disruption["delay_minutes"] == 45
    assert "delayed by 45 minutes" in disruption["reason"]


@patch("app.tools.flight_tool.get_flight_status")
def test_track_flight_disruption_cancellation(mock_status):
    """Verify disruption detection for cancelled flight."""
    mock_status.return_value = {
        "flight_number": "UA200",
        "status": "cancelled",
        "departure": {"iata": "SFO", "delay_minutes": 0},
        "arrival": {"iata": "ORD", "delay_minutes": 0},
    }

    disruption = track_flight_disruption("UA200")
    assert disruption is not None
    assert disruption["event_type"] == "CANCELLATION"


def test_lookup_airport_tool():
    """Verify LangChain lookup_airport tool."""
    res = lookup_airport.invoke({"query": "Dhaka"})
    assert "DAC" in res
    assert "Bangladesh" in res
    assert "Asia/Dhaka" in res


@patch("app.tools.flight_tool.get_flight_status")
def test_track_flight_status_tool(mock_status):
    """Verify LangChain track_flight_status tool."""
    mock_status.return_value = {
        "flight_number": "BG201",
        "airline": "Biman Bangladesh",
        "status": "scheduled",
        "departure": {"airport": "Hazrat Shahjalal", "iata": "DAC", "terminal": "2", "gate": "4", "scheduled": "2026-09-08 08:00", "delay_minutes": 0},
        "arrival": {"airport": "Heathrow", "iata": "LHR", "terminal": "4", "gate": "12", "baggage": "6", "scheduled": "2026-09-08 18:30", "delay_minutes": 0},
    }
    res = track_flight_status.invoke({"flight_number": "BG201"})
    assert "Biman Bangladesh" in res
    assert "DAC" in res
    assert "LHR" in res


if __name__ == "__main__":
    print("Running test_resolve_country_code...")
    test_resolve_country_code()
    print("Running test_resolve_airport_code_iata_and_icao...")
    test_resolve_airport_code_iata_and_icao()
    print("Running test_resolve_airport_code_city_names...")
    test_resolve_airport_code_city_names()
    print("Running test_resolve_airport_code_city_and_country...")
    test_resolve_airport_code_city_and_country()
    print("Running test_resolve_airport_code_airport_name...")
    test_resolve_airport_code_airport_name()
    print("Running test_resolve_airport_code_fallback...")
    test_resolve_airport_code_fallback()
    print("Running test_get_airport_info...")
    test_get_airport_info()
    print("Running test_search_airports...")
    test_search_airports()
    print("Running test_parse_flight_to_trip_segment_success...")
    test_parse_flight_to_trip_segment_success()
    print("Running test_parse_flight_invalid_chronology_raises_error...")
    test_parse_flight_invalid_chronology_raises_error()
    print("Running test_search_flights_aviationstack...")
    test_search_flights_aviationstack()
    print("Running test_get_flight_status...")
    test_get_flight_status()
    print("Running test_track_flight_disruption_detected...")
    test_track_flight_disruption_detected()
    print("Running test_track_flight_disruption_cancellation...")
    test_track_flight_disruption_cancellation()
    print("Running test_lookup_airport_tool...")
    test_lookup_airport_tool()
    print("Running test_track_flight_status_tool...")
    test_track_flight_status_tool()
    print("\n==========================================")
    print("SUCCESS: ALL FLIGHT TOOL TESTS PASSED!")
    print("==========================================")
