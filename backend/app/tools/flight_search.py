from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import uuid
import dateutil.parser
from langchain_core.tools import tool
import serpapi

from app.core.config import settings
from app.graph.state import Location, SegmentType, TripSegment

logger = logging.getLogger(__name__)

# Initialize SerpApi client
client = serpapi.Client(api_key=settings.SERPAPI_API_KEY)


class FlightSearchValidationError(ValueError):
    """Raised when SerpApi flight search response contains malformed or unparseable data."""
    pass


def _parse_timestamp(time_str: str) -> datetime:
    """Parses date/time strings from SerpApi into a timezone-aware or UTC-standard datetime."""
    if not time_str or not isinstance(time_str, str):
        raise FlightSearchValidationError(f"Invalid or missing timestamp string: {time_str!r}")
    try:
        return dateutil.parser.parse(time_str)
    except Exception as exc:
        raise FlightSearchValidationError(f"Unable to parse timestamp '{time_str}': {exc}") from exc


def parse_flight_to_trip_segment(
    flight_entry: Dict[str, Any],
    currency: str = "USD",
) -> TripSegment:
    """
    Parses and strictly validates a raw SerpApi Google Flights entry into a TripSegment domain model.
    Raises FlightSearchValidationError on malformed or missing required fields.
    """
    if not isinstance(flight_entry, dict):
        raise FlightSearchValidationError(f"Expected dict for flight entry, received {type(flight_entry).__name__}")

    legs = flight_entry.get("flights")
    if not legs or not isinstance(legs, list):
        raise FlightSearchValidationError("Malformed SerpApi response: 'flights' list is missing or empty")

    first_leg = legs[0]
    last_leg = legs[-1]

    dep_airport = first_leg.get("departure_airport")
    if not dep_airport or not isinstance(dep_airport, dict):
        raise FlightSearchValidationError("Malformed flight leg: missing 'departure_airport' data")

    arr_airport = last_leg.get("arrival_airport")
    if not arr_airport or not isinstance(arr_airport, dict):
        raise FlightSearchValidationError("Malformed flight leg: missing 'arrival_airport' data")

    dep_time_str = dep_airport.get("time")
    if not dep_time_str:
        raise FlightSearchValidationError("Malformed departure_airport: missing 'time' field")

    arr_time_str = arr_airport.get("time")
    if not arr_time_str:
        raise FlightSearchValidationError("Malformed arrival_airport: missing 'time' field")

    start_time = _parse_timestamp(dep_time_str)
    end_time = _parse_timestamp(arr_time_str)

    if end_time <= start_time:
        raise FlightSearchValidationError(
            f"Invalid flight chronology: arrival time ({end_time}) must be strictly after departure ({start_time})"
        )

    airlines = [leg.get("airline") for leg in legs if leg.get("airline")]
    airline_str = ", ".join(dict.fromkeys(airlines)) if airlines else "Airline"

    flight_numbers = [leg.get("flight_number") for leg in legs if leg.get("flight_number")]
    flight_num_str = ", ".join(flight_numbers) if flight_numbers else f"FL-{uuid.uuid4().hex[:4].upper()}"

    arr_name = arr_airport.get("name") or arr_airport.get("id") or "Unknown Airport"
    arr_iata = arr_airport.get("id")

    location = Location(
        name=arr_name,
        iata_code=arr_iata,
    )

    raw_price = flight_entry.get("price")
    try:
        cost = float(raw_price) if raw_price is not None else 0.0
    except (ValueError, TypeError) as exc:
        raise FlightSearchValidationError(f"Invalid flight price value '{raw_price}': {exc}") from exc

    seg_id = f"flight_{first_leg.get('flight_number', uuid.uuid4().hex[:6])}".replace(" ", "_")

    return TripSegment(
        id=seg_id,
        type=SegmentType.FLIGHT,
        title=f"{airline_str} ({flight_num_str})",
        start_time=start_time,
        end_time=end_time,
        location=location,
        cost=cost,
        currency=currency,
        metadata={
            "airline": airline_str,
            "flight_number": flight_num_str,
            "departure_airport": dep_airport.get("name") or dep_airport.get("id", ""),
            "departure_iata": dep_airport.get("id", ""),
            "total_duration_minutes": flight_entry.get("total_duration", 0),
            "legs_count": len(legs),
        },
        is_confirmed=False,
    )


@tool
def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    currency: str = "USD",
    return_date: Optional[str] = None,
) -> List[TripSegment]:
    """Search for flights using SerpApi Google Flights engine and return strictly typed TripSegment list.

    Args:
        departure_id: Departure airport IATA code (e.g. 'JFK', 'SFO', 'LHR').
        arrival_id: Arrival airport IATA code (e.g. 'LAX', 'CDG', 'DXB').
        outbound_date: Outbound travel date in YYYY-MM-DD format (e.g. '2026-09-15').
        currency: Preferred 3-letter currency code (e.g. 'USD', 'EUR', 'GBP'). Defaults to 'USD'.
        return_date: Optional return date in YYYY-MM-DD format for round trips.

    Returns:
        List of strictly validated TripSegment domain models with departure, arrival,
        airline metadata, and pricing.

    Raises:
        FlightSearchValidationError: If the API response contains malformed or unparseable data.
    """
    if not settings.SERPAPI_API_KEY or settings.SERPAPI_API_KEY.startswith("test") or settings.SERPAPI_API_KEY == "":
        raise FlightSearchValidationError("SerpApi API key not configured in environment.")

    params: Dict[str, Any] = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "currency": currency,
        "hl": "en",
    }

    if return_date:
        params["return_date"] = return_date

    try:
        results = client.search(params)
        raw_results = results.as_dict() if hasattr(results, "as_dict") else dict(results)
    except Exception as exc:
        raise FlightSearchValidationError(f"SerpApi connection or query execution failed: {exc}") from exc

    if "error" in raw_results:
        raise FlightSearchValidationError(f"SerpApi returned an error: {raw_results['error']}")

    best_flights = raw_results.get("best_flights", [])
    flight_list = best_flights if best_flights else raw_results.get("other_flights", [])

    if not flight_list:
        return []

    parsed_segments: List[TripSegment] = []
    for f in flight_list:
        segment = parse_flight_to_trip_segment(f, currency=currency)
        parsed_segments.append(segment)

    return parsed_segments
