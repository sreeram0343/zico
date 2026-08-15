from unittest.mock import MagicMock, patch
from app.tools.flight_search import search_flights, _parse_flight_entry


def test_parse_flight_entry():
    """Verify parsing of raw SerpApi Google Flights response item."""
    mock_flight_entry = {
        "flights": [
            {
                "airline": "Delta",
                "flight_number": "DL 123",
                "departure_airport": {"name": "John F. Kennedy Intl", "id": "JFK", "time": "2026-09-15 08:00"},
                "arrival_airport": {"name": "Los Angeles Intl", "id": "LAX", "time": "2026-09-15 11:30"},
                "duration": 330,
            }
        ],
        "total_duration": 330,
        "price": 299.0,
    }

    parsed = _parse_flight_entry(mock_flight_entry, currency="USD")
    assert parsed["airline"] == "Delta"
    assert parsed["flight_number"] == "DL 123"
    assert parsed["departure_airport"] == "John F. Kennedy Intl"
    assert parsed["departure_time"] == "2026-09-15 08:00"
    assert parsed["arrival_airport"] == "Los Angeles Intl"
    assert parsed["arrival_time"] == "2026-09-15 11:30"
    assert parsed["duration"] == 330
    assert parsed["price"] == 299.0
    assert parsed["currency"] == "USD"


def test_parse_flight_entry_empty():
    """Verify fallback for empty flight legs."""
    parsed = _parse_flight_entry({}, currency="EUR")
    assert parsed["airline"] == "Unknown"
    assert parsed["flight_number"] == "Unknown"
    assert parsed["currency"] == "EUR"


@patch("app.tools.flight_search.client.search")
def test_search_flights_tool_success(mock_search):
    """Verify search_flights tool execution with successful response."""
    mock_search.return_value = {
        "best_flights": [
            {
                "flights": [
                    {
                        "airline": "United Airlines",
                        "flight_number": "UA 456",
                        "departure_airport": {"name": "SFO", "id": "SFO", "time": "2026-09-15 09:00"},
                        "arrival_airport": {"name": "ORD", "id": "ORD", "time": "2026-09-15 15:00"},
                        "duration": 240,
                    }
                ],
                "total_duration": 240,
                "price": 180.0,
            }
        ]
    }

    results = search_flights.invoke({
        "departure_id": "SFO",
        "arrival_id": "ORD",
        "outbound_date": "2026-09-15",
        "currency": "USD",
    })

    assert len(results) == 1
    assert results[0]["airline"] == "United Airlines"
    assert results[0]["flight_number"] == "UA 456"
    assert results[0]["price"] == 180.0


@patch("app.tools.flight_search.client.search")
def test_search_flights_tool_error_handling(mock_search):
    """Verify graceful handling when SerpApi throws an error."""
    mock_search.side_effect = Exception("API connection failure")

    results = search_flights.invoke({
        "departure_id": "JFK",
        "arrival_id": "LHR",
        "outbound_date": "2026-09-15",
    })

    assert len(results) == 1
    assert "error" in results[0]
    assert "API connection failure" in results[0]["error"]
