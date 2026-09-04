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


# Known City Name to Airport IATA Code Mapping
CITY_TO_IATA: Dict[str, str] = {
    "mumbai": "BOM",
    "bombay": "BOM",
    "pune": "PNQ",
    "delhi": "DEL",
    "new delhi": "DEL",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU",
    "kochi": "COK",
    "cochin": "COK",
    "goa": "GOI",
    "ahmedabad": "AMD",
    "jaipur": "JAI",
    "lucknow": "LKO",
    "amritsar": "ATQ",
    "srinagar": "SXR",
    "varanasi": "VNS",
    "indore": "IDR",
    "patna": "PAT",
    "chandigarh": "IXC",
    "bhubaneswar": "BBI",
    "guwahati": "GAU",
    "nagpur": "NAG",
    "coimbatore": "CJB",
    "mangalore": "IXE",
    "calicut": "CCJ",
    "london": "LHR",
    "paris": "CDG",
    "tokyo": "HND",
    "new york": "JFK",
    "nyc": "JFK",
    "san francisco": "SFO",
    "sfo": "SFO",
    "chicago": "ORD",
    "ord": "ORD",
    "los angeles": "LAX",
    "lax": "LAX",
    "dubai": "DXB",
    "singapore": "SIN",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
}


def resolve_airport_code(code_or_city: Optional[str], default: str = "BOM") -> str:
    """
    Resolves city name or lowercase IATA string to a validated 3-letter IATA code.
    E.g. 'mumbai' -> 'BOM', 'pune' -> 'PNQ', 'SFO' -> 'SFO'.
    """
    if not code_or_city or not isinstance(code_or_city, str):
        return default
    cleaned = code_or_city.strip()
    cleaned_lower = cleaned.lower()

    if cleaned_lower in CITY_TO_IATA:
        return CITY_TO_IATA[cleaned_lower]

    for city, iata in CITY_TO_IATA.items():
        if city in cleaned_lower:
            return iata

    if len(cleaned) == 3 and cleaned.isalpha():
        return cleaned.upper()

    return cleaned.upper()


@tool
def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: Optional[str] = None,
    currency: str = "USD",
    return_date: Optional[str] = None,
) -> List[TripSegment]:
    """Search for flights using SerpApi Google Flights engine and return strictly typed TripSegment list.

    Args:
        departure_id: Departure airport IATA code or city name (e.g. 'mumbai', 'JFK', 'SFO').
        arrival_id: Arrival airport IATA code or city name (e.g. 'pune', 'LAX', 'CDG').
        outbound_date: Outbound travel date in YYYY-MM-DD format. Defaults to tomorrow's date if not specified.
        currency: Preferred 3-letter currency code (e.g. 'USD', 'EUR', 'GBP'). Defaults to 'USD'.
        return_date: Optional return date in YYYY-MM-DD format for round trips.

    Returns:
        List of strictly validated TripSegment domain models with departure, arrival,
        airline metadata, and pricing.
    """
    from datetime import timedelta

    # 1. Enforce robust defaults for city names -> airport IATA codes
    dep_code = resolve_airport_code(departure_id, default="BOM")
    arr_code = resolve_airport_code(arrival_id, default="PNQ")

    # 2. Enforce robust default: if no departure date is specified, automatically inject tomorrow's date
    if not outbound_date or not isinstance(outbound_date, str) or not outbound_date.strip():
        tomorrow = datetime.now() + timedelta(days=1)
        outbound_date = tomorrow.strftime("%Y-%m-%d")
    else:
        outbound_date = outbound_date.strip()

    if not settings.SERPAPI_API_KEY or settings.SERPAPI_API_KEY.startswith("test") or settings.SERPAPI_API_KEY == "":
        logger.info("SerpApi API key not configured in environment.")
        return []

    params: Dict[str, Any] = {
        "engine": "google_flights",
        "departure_id": dep_code,
        "arrival_id": arr_code,
        "outbound_date": outbound_date,
        "currency": currency,
        "hl": "en",
    }

    if return_date:
        params["return_date"] = return_date

    # 3. Isolated try/except block for SerpApi calls returning typed TripSegment list
    try:
        results = client.search(params)
        raw_results = results.as_dict() if hasattr(results, "as_dict") else dict(results)
    except Exception as exc:
        logger.warning(f"SerpApi connection or query execution failed: {exc}")
        return []

    if "error" in raw_results:
        err_msg = raw_results.get("error", "Unknown SerpApi error")
        logger.warning(f"SerpApi returned error: {err_msg}")
        return []

    best_flights = raw_results.get("best_flights", [])
    flight_list = best_flights if best_flights else raw_results.get("other_flights", [])

    if not flight_list:
        return []

    parsed_segments: List[TripSegment] = []
    for f in flight_list:
        try:
            segment = parse_flight_to_trip_segment(f, currency=currency)
            parsed_segments.append(segment)
        except Exception as parse_err:
            logger.debug(f"Skipping malformed flight entry: {parse_err}")
            continue

    return parsed_segments


