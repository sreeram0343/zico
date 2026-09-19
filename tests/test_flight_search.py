from unittest.mock import patch

import pytest

from app.graph.state import SegmentType, TripSegment
from app.tools.flight_search import (
    FlightSearchValidationError,
    parse_flight_to_trip_segment,
    search_flights,
)


def test_parse_flight_to_trip_segment_success():
    """Verify parsing and schema validation of flight entry into TripSegment."""
    mock_flight_entry = {
        "flights": [
            {
                "airline": "Delta Air Lines",
                "flight_number": "DL 123",
                "departure_airport": {
                    "name": "John F. Kennedy Intl",
                    "id": "JFK",
                    "time": "2026-09-15 08:00",
                },
                "arrival_airport": {
                    "name": "Los Angeles Intl",
                    "id": "LAX",
                    "time": "2026-09-15 11:30",
                },
                "duration": 330,
            }
        ],
        "total_duration": 330,
        "price": 299.0,
    }

    segment = parse_flight_to_trip_segment(mock_flight_entry, currency="USD")
    assert isinstance(segment, TripSegment)
    assert segment.type == SegmentType.FLIGHT
    assert "Delta Air Lines" in segment.title
    assert "DL 123" in segment.title
    assert segment.location.name == "Los Angeles Intl"
    assert segment.location.iata_code == "LAX"
    assert segment.cost == 299.0
    assert segment.currency == "USD"
    assert segment.start_time < segment.end_time
    assert segment.metadata["airline"] == "Delta Air Lines"


def test_parse_aviationstack_native_flight_entry():
    """Verify parsing of native AviationStack response item into TripSegment."""
    aviationstack_entry = {
        "departure": {
            "airport": "John F. Kennedy Intl",
            "iata": "JFK",
            "scheduled": "2026-09-15T08:00:00+00:00",
        },
        "arrival": {
            "airport": "Los Angeles Intl",
            "iata": "LAX",
            "scheduled": "2026-09-15T11:30:00+00:00",
        },
        "airline": {
            "name": "Delta Air Lines",
            "iata": "DL",
        },
        "flight": {
            "iata": "DL123",
            "number": "123",
        },
        "price": 299.0,
    }

    segment = parse_flight_to_trip_segment(aviationstack_entry, currency="USD")
    assert isinstance(segment, TripSegment)
    assert segment.type == SegmentType.FLIGHT
    assert "Delta Air Lines" in segment.title
    assert "DL123" in segment.title
    assert segment.location.name == "Los Angeles Intl"
    assert segment.location.iata_code == "LAX"
    assert segment.cost == 299.0


def test_parse_flight_empty_legs_raises_validation_error():
    """Verify that missing or empty flights array raises FlightSearchValidationError."""
    with pytest.raises(FlightSearchValidationError, match="flights.*missing or empty"):
        parse_flight_to_trip_segment({}, currency="EUR")


def test_parse_flight_invalid_chronology_raises_validation_error():
    """Verify that invalid chronology (arrival before departure) raises FlightSearchValidationError."""
    invalid_entry = {
        "flights": [
            {
                "airline": "United",
                "flight_number": "UA 100",
                "departure_airport": {"name": "SFO", "id": "SFO", "time": "2026-09-15 15:00"},
                "arrival_airport": {
                    "name": "ORD",
                    "id": "ORD",
                    "time": "2026-09-15 10:00",
                },  # 5 hours earlier
            }
        ],
        "price": 200.0,
    }
    with pytest.raises(
        FlightSearchValidationError, match="arrival time.*must be strictly after departure"
    ):
        parse_flight_to_trip_segment(invalid_entry)


def test_parse_flight_missing_airport_data_raises_validation_error():
    """Verify that missing airport object raises FlightSearchValidationError."""
    malformed_entry = {
        "flights": [
            {
                "airline": "United",
                "flight_number": "UA 100",
                # missing departure_airport
                "arrival_airport": {"name": "ORD", "id": "ORD", "time": "2026-09-15 15:00"},
            }
        ]
    }
    with pytest.raises(FlightSearchValidationError, match="missing 'departure_airport'"):
        parse_flight_to_trip_segment(malformed_entry)


@patch("app.tools.flight_search.client.search")
def test_search_flights_tool_success(mock_search):
    """Verify search_flights tool execution returns strictly List[TripSegment]."""
    mock_search.return_value = {
        "best_flights": [
            {
                "flights": [
                    {
                        "airline": "United Airlines",
                        "flight_number": "UA 456",
                        "departure_airport": {
                            "name": "SFO",
                            "id": "SFO",
                            "time": "2026-09-15 09:00",
                        },
                        "arrival_airport": {"name": "ORD", "id": "ORD", "time": "2026-09-15 15:00"},
                        "duration": 240,
                    }
                ],
                "total_duration": 240,
                "price": 180.0,
            }
        ]
    }

    results = search_flights.invoke(
        {
            "departure_id": "SFO",
            "arrival_id": "ORD",
            "outbound_date": "2026-09-15",
            "currency": "USD",
        }
    )

    assert isinstance(results, list)
    assert len(results) == 1
    segment = results[0]
    assert isinstance(segment, TripSegment)
    assert segment.location.iata_code == "ORD"
    assert segment.cost == 180.0


@patch("app.tools.flight_search.client.search")
def test_search_flights_tool_api_error(mock_search):
    """Verify graceful handling and empty TripSegment list returned on API failure."""
    mock_search.side_effect = Exception("AviationStack network outage")

    result = search_flights.invoke(
        {
            "departure_id": "JFK",
            "arrival_id": "LHR",
            "outbound_date": "2026-09-15",
        }
    )

    assert isinstance(result, list)
    assert len(result) == 0


@patch("app.tools.flight_search.client.search")
def test_search_flights_tool_aviationstack_format_success(mock_search):
    """Verify search_flights tool parses native AviationStack response properly."""
    mock_search.return_value = {
        "data": [
            {
                "departure": {
                    "airport": "San Francisco International",
                    "iata": "SFO",
                    "scheduled": "2026-09-15T09:00:00+00:00",
                },
                "arrival": {
                    "airport": "Chicago O'Hare International",
                    "iata": "ORD",
                    "scheduled": "2026-09-15T15:00:00+00:00",
                },
                "airline": {
                    "name": "United Airlines",
                    "iata": "UA",
                },
                "flight": {
                    "iata": "UA456",
                    "number": "456",
                },
                "price": 180.0,
            }
        ]
    }

    results = search_flights.invoke(
        {
            "departure_id": "SFO",
            "arrival_id": "ORD",
            "outbound_date": "2026-09-15",
            "currency": "USD",
        }
    )

    assert isinstance(results, list)
    assert len(results) == 1
    segment = results[0]
    assert isinstance(segment, TripSegment)
    assert segment.location.iata_code == "ORD"
    assert "United Airlines" in segment.title
    assert segment.cost == 180.0


@patch("app.tools.flight_search.client.search")
def test_search_flights_city_resolution_and_defaults(mock_search):
    """Verify city name resolution to IATA codes, default tomorrow date, and typed list return."""
    mock_search.return_value = {"best_flights": []}

    # Test with city names and missing outbound_date
    result = search_flights.invoke(
        {
            "departure_id": "mumbai",
            "arrival_id": "pune",
        }
    )

    assert isinstance(result, list)
    assert len(result) == 0
    # Check that mock_search was called with resolved IATA codes BOM and PNQ
    mock_search.assert_called_once()
    call_args = mock_search.call_args[0][0]
    assert call_args["departure_id"] == "BOM"
    assert call_args["arrival_id"] == "PNQ"
    # Ensure outbound_date was automatically populated
    assert call_args["outbound_date"] is not None
    assert len(call_args["outbound_date"]) == 10
