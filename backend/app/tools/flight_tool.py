import os
import re
import time
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import certifi
import requests
import airportsdata
import pycountry
import dateutil.parser
from dotenv import load_dotenv
from langchain_core.tools import tool

from app.graph.state import Location, SegmentType, TripSegment

# Load environment variables
load_dotenv()

# Enforce secure SSL CA bundle path
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

logger = logging.getLogger(__name__)

# AviationStack API Configuration
API_KEY = os.getenv("AVIATIONSTACK_API_KEY", "")
BASE_URL = "http://api.aviationstack.com/v1"
DEFAULT_ORIGIN_IATA = os.getenv("DEFAULT_ORIGIN_IATA", "DAC").strip().strip('"').strip("'")

# In-memory TTL response cache: {cache_key: (timestamp, response_dict)}
_API_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes cache to prevent quota exhaustion and 429 rate limits


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FlightToolError(Exception):
    """Base exception for flight tool operations."""
    pass


class FlightToolValidationError(ValueError):
    """Raised when flight data fails validation or has malformed timestamps/chronology."""
    pass


# ---------------------------------------------------------------------------
# Airport & Location Resolution Engine (airportsdata + pycountry)
# ---------------------------------------------------------------------------

_IATA_AIRPORTS: Optional[Dict[str, Dict[str, Any]]] = None
_ICAO_AIRPORTS: Optional[Dict[str, Dict[str, Any]]] = None
_CITY_INDEX: Optional[Dict[str, List[Dict[str, Any]]]] = None
_COUNTRY_INDEX: Optional[Dict[str, List[Dict[str, Any]]]] = None

# Known major hub priority mapping for cities with multiple airports
MAJOR_CITY_HUBS: Dict[str, str] = {
    "dhaka": "DAC",
    "london": "LHR",
    "new york": "JFK",
    "nyc": "JFK",
    "paris": "CDG",
    "tokyo": "HND",
    "mumbai": "BOM",
    "bombay": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "kolkata": "CCU",
    "singapore": "SIN",
    "dubai": "DXB",
    "doha": "DOH",
    "bangkok": "BKK",
    "kuala lumpur": "KUL",
    "san francisco": "SFO",
    "los angeles": "LAX",
    "chicago": "ORD",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "rome": "FCO",
    "beijing": "PEK",
    "shanghai": "PVG",
    "sydney": "SYD",
    "melbourne": "MEL",
    "toronto": "YYZ",
    "hong kong": "HKG",
    "istanbul": "IST",
    "seoul": "ICN",
}


def _ensure_airport_databases_loaded() -> None:
    """Lazy-loads airportsdata databases and constructs secondary lookup indexes."""
    global _IATA_AIRPORTS, _ICAO_AIRPORTS, _CITY_INDEX, _COUNTRY_INDEX
    if _IATA_AIRPORTS is not None:
        return

    logger.debug("Loading global airport databases into memory...")
    _IATA_AIRPORTS = airportsdata.load("IATA")
    _ICAO_AIRPORTS = airportsdata.load("ICAO")

    city_idx: Dict[str, List[Dict[str, Any]]] = {}
    country_idx: Dict[str, List[Dict[str, Any]]] = {}

    for airport in _IATA_AIRPORTS.values():
        city = airport.get("city")
        if city:
            norm_city = city.strip().lower()
            city_idx.setdefault(norm_city, []).append(airport)

        country = airport.get("country")
        if country:
            norm_country = country.strip().upper()
            country_idx.setdefault(norm_country, []).append(airport)

    _CITY_INDEX = city_idx
    _COUNTRY_INDEX = country_idx


def resolve_country_code(country_name_or_code: str) -> Optional[str]:
    """
    Resolves country names (e.g. 'Bangladesh', 'United States', 'Germany')
    or alpha-3 codes ('USA', 'BGD') to an ISO-3166-1 alpha-2 2-letter code ('BD', 'US', 'DE').
    """
    if not country_name_or_code or not isinstance(country_name_or_code, str):
        return None

    cleaned = country_name_or_code.strip()
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned.upper()

    # Try exact pycountry lookups
    try:
        match = pycountry.countries.get(alpha_3=cleaned.upper())
        if match and hasattr(match, "alpha_2"):
            return match.alpha_2
    except Exception:
        pass

    try:
        match = pycountry.countries.get(name=cleaned)
        if match and hasattr(match, "alpha_2"):
            return match.alpha_2
    except Exception:
        pass

    # Try fuzzy search
    try:
        fuzzy_results = pycountry.countries.search_fuzzy(cleaned)
        if fuzzy_results:
            return fuzzy_results[0].alpha_2
    except Exception:
        pass

    return None


def resolve_airport_code(query: Optional[str], default: Optional[str] = None) -> str:
    """
    Intelligently resolves any airport input to a standard 3-letter IATA code.
    Supports:
      - 3-letter IATA codes (e.g. 'DAC', 'JFK', 'LHR', 'BOM')
      - 4-letter ICAO codes (e.g. 'VGHS', 'KJFK', 'EGLL')
      - City names (e.g. 'Dhaka', 'London', 'New York', 'Tokyo', 'Mumbai')
      - 'City, Country' strings (e.g. 'Dhaka, Bangladesh', 'London, UK')
      - Airport names (e.g. 'Heathrow', 'Hazrat Shahjalal', 'Changi', 'JFK')
    Falls back to `default` or `DEFAULT_ORIGIN_IATA` ('DAC').
    """
    fallback = (default or DEFAULT_ORIGIN_IATA or "DAC").upper()

    if not query or not isinstance(query, str):
        return fallback

    cleaned = query.strip()
    if not cleaned:
        return fallback

    _ensure_airport_databases_loaded()
    assert _IATA_AIRPORTS is not None
    assert _ICAO_AIRPORTS is not None
    assert _CITY_INDEX is not None
    assert _COUNTRY_INDEX is not None

    cleaned_upper = cleaned.upper()
    cleaned_lower = cleaned.lower()

    # 1. Direct IATA match
    if len(cleaned) == 3 and cleaned.isalpha() and cleaned_upper in _IATA_AIRPORTS:
        return cleaned_upper

    # 2. Direct ICAO match
    if len(cleaned) == 4 and cleaned.isalpha() and cleaned_upper in _ICAO_AIRPORTS:
        iata_code = _ICAO_AIRPORTS[cleaned_upper].get("iata")
        if iata_code:
            return iata_code

    # 3. Known major city hub override
    if cleaned_lower in MAJOR_CITY_HUBS:
        return MAJOR_CITY_HUBS[cleaned_lower]

    for city_name, hub_iata in MAJOR_CITY_HUBS.items():
        if city_name in cleaned_lower:
            return hub_iata

    # 4. Handle "City, Country" or "City, State" patterns
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        city_part = parts[0].lower()
        country_part = parts[1]
        c_code = resolve_country_code(country_part)

        # Check city index filtered by country
        if city_part in _CITY_INDEX:
            airports = _CITY_INDEX[city_part]
            if c_code:
                matching = [a for a in airports if a.get("country") == c_code]
                if matching:
                    # Pick international airport first
                    intl = [a for a in matching if "international" in a.get("name", "").lower()]
                    return (intl[0] if intl else matching[0])["iata"]
            return airports[0]["iata"]

    # 5. City index lookup
    if cleaned_lower in _CITY_INDEX:
        airports = _CITY_INDEX[cleaned_lower]
        if airports:
            intl = [a for a in airports if "international" in a.get("name", "").lower()]
            return (intl[0] if intl else airports[0])["iata"]

    # 6. Substring match in airport names
    for airport in _IATA_AIRPORTS.values():
        name_lower = airport.get("name", "").lower()
        if cleaned_lower in name_lower and airport.get("iata"):
            return airport["iata"]

    # 7. Substring match in city names
    for city_name, airports in _CITY_INDEX.items():
        if cleaned_lower in city_name and airports:
            intl = [a for a in airports if "international" in a.get("name", "").lower()]
            return (intl[0] if intl else airports[0])["iata"]

    # 8. If 3 letters alphabetic, assume it is an IATA code
    if len(cleaned) == 3 and cleaned.isalpha():
        return cleaned_upper

    return fallback


def get_airport_info(code_or_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns full metadata for an airport identified by IATA or ICAO code.
    Enriched with full country name via pycountry.
    """
    if not code_or_name or not isinstance(code_or_name, str):
        return None

    _ensure_airport_databases_loaded()
    assert _IATA_AIRPORTS is not None
    assert _ICAO_AIRPORTS is not None

    iata = resolve_airport_code(code_or_name)
    airport = _IATA_AIRPORTS.get(iata)
    if not airport:
        return None

    result = dict(airport)
    country_code = airport.get("country")
    if country_code:
        try:
            c = pycountry.countries.get(alpha_2=country_code)
            result["country_name"] = c.name if c else country_code
        except Exception:
            result["country_name"] = country_code

    return result


def search_airports(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Searches airports across global database matching query by name, city, or code."""
    if not query or not isinstance(query, str):
        return []

    _ensure_airport_databases_loaded()
    assert _IATA_AIRPORTS is not None

    q_lower = query.strip().lower()
    matches: List[Dict[str, Any]] = []

    for airport in _IATA_AIRPORTS.values():
        name = airport.get("name", "").lower()
        city = (airport.get("city") or "").lower()
        iata = airport.get("iata", "").lower()
        icao = airport.get("icao", "").lower()

        if q_lower in iata or q_lower in icao or q_lower in city or q_lower in name:
            matches.append(airport)
            if len(matches) >= limit:
                break

    return matches


# ---------------------------------------------------------------------------
# AviationStack API Client
# ---------------------------------------------------------------------------


def _get_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """Generates a stable cache key for request caching."""
    sorted_items = sorted((k, str(v)) for k, v in params.items() if k != "access_key")
    return f"{endpoint}:{sorted_items}"


def query_aviationstack(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Executes a resilient query against the AviationStack REST API.
    Handles rate-limiting with in-memory TTL caching, handles plan-restricted functions,
    and returns parsed JSON dictionary.
    """
    api_key = API_KEY or os.getenv("AVIATIONSTACK_API_KEY", "")
    if not api_key:
        logger.warning("AviationStack API key not set in environment (AVIATIONSTACK_API_KEY).")
        return {"data": [], "error": {"code": "missing_api_key", "message": "API key not configured"}}

    endpoint_clean = endpoint.lstrip("/")
    request_params = dict(params or {})
    request_params["access_key"] = api_key

    cache_key = _get_cache_key(endpoint_clean, request_params)
    now = time.time()

    if use_cache and cache_key in _API_CACHE:
        cached_time, cached_data = _API_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL_SECONDS:
            logger.debug(f"AviationStack cache hit for {cache_key}")
            return cached_data

    url = f"{BASE_URL}/{endpoint_clean}"

    try:
        response = requests.get(url, params=request_params, timeout=timeout)
        data = response.json()

        # Check for AviationStack API errors
        if isinstance(data, dict) and "error" in data:
            err = data.get("error", {})
            err_code = err.get("code", "unknown")
            err_msg = err.get("message", "Unknown AviationStack error")

            logger.warning(f"AviationStack API returned error [{err_code}]: {err_msg}")

            # If function_access_restricted (e.g. flight_date on free tier), allow caller to adapt
            if err_code == "function_access_restricted" and "flight_date" in request_params:
                logger.info("Retrying query without flight_date due to plan tier restriction...")
                retry_params = dict(request_params)
                del retry_params["flight_date"]
                return query_aviationstack(endpoint_clean, retry_params, use_cache=use_cache, timeout=timeout)

            # If rate limit exceeded, return cached data if available even if stale
            if err_code == "rate_limit_reached" and cache_key in _API_CACHE:
                logger.info("Serving stale cached flight data due to rate limit.")
                return _API_CACHE[cache_key][1]

            return data

        # Cache successful response
        if use_cache and response.status_code == 200:
            _API_CACHE[cache_key] = (now, data)

        return data

    except requests.exceptions.RequestException as exc:
        logger.error(f"Network error communicating with AviationStack: {exc}")
        if cache_key in _API_CACHE:
            logger.info("Serving stale cached flight data after network failure.")
            return _API_CACHE[cache_key][1]
        return {"data": [], "error": {"code": "network_error", "message": str(exc)}}
    except Exception as exc:
        logger.error(f"Unexpected error querying AviationStack: {exc}")
        return {"data": [], "error": {"code": "internal_error", "message": str(exc)}}


# ---------------------------------------------------------------------------
# Domain Model Conversion (AviationStack -> TripSegment)
# ---------------------------------------------------------------------------


def _parse_timestamp(time_str: Optional[str]) -> datetime:
    """Parses ISO-8601 timestamps into a standard datetime."""
    if not time_str or not isinstance(time_str, str):
        raise FlightToolValidationError(f"Invalid timestamp string: {time_str!r}")
    try:
        return dateutil.parser.parse(time_str)
    except Exception as exc:
        raise FlightToolValidationError(f"Unable to parse timestamp '{time_str}': {exc}") from exc


def parse_flight_to_trip_segment(
    flight_data: Dict[str, Any],
    default_cost: float = 0.0,
    currency: str = "USD",
) -> TripSegment:
    """
    Converts an AviationStack flight record into ZICO's strict TripSegment domain model.
    Enriches the arrival Location with exact coordinates (lat, lng) from airportsdata.
    """
    if not isinstance(flight_data, dict):
        raise FlightToolValidationError(f"Expected dict for flight entry, got {type(flight_data).__name__}")

    dep = flight_data.get("departure") or {}
    arr = flight_data.get("arrival") or {}
    flight_info = flight_data.get("flight") or {}
    airline_info = flight_data.get("airline") or {}

    dep_iata = (dep.get("iata") or "DEP").upper()
    arr_iata = (arr.get("iata") or "ARR").upper()

    dep_airport_name = dep.get("airport") or f"Airport ({dep_iata})"
    arr_airport_name = arr.get("airport") or f"Airport ({arr_iata})"

    dep_time_str = dep.get("scheduled") or dep.get("estimated") or dep.get("actual")
    arr_time_str = arr.get("scheduled") or arr.get("estimated") or arr.get("actual")

    if not dep_time_str or not arr_time_str:
        raise FlightToolValidationError("Flight entry missing required scheduled departure or arrival timestamps.")

    start_time = _parse_timestamp(dep_time_str)
    end_time = _parse_timestamp(arr_time_str)

    # Validate chronology: arrival must strictly follow departure
    if end_time <= start_time:
        # Check if arrival lacks rollover or is next-day crossing
        if end_time.date() == start_time.date() and end_time.time() <= start_time.time():
            end_time = end_time + timedelta(days=1)
        else:
            raise FlightToolValidationError(
                f"Invalid flight chronology: arrival ({end_time}) must be strictly after departure ({start_time})"
            )

    airline_name = airline_info.get("name") or "Commercial Airline"
    flight_iata = flight_info.get("iata") or flight_info.get("number") or f"FL-{uuid.uuid4().hex[:4].upper()}"

    # Enrich arrival Location with geographic coordinates from airportsdata
    _ensure_airport_databases_loaded()
    assert _IATA_AIRPORTS is not None
    arr_airport_meta = _IATA_AIRPORTS.get(arr_iata, {})
    arr_lat = arr_airport_meta.get("lat")
    arr_lng = arr_airport_meta.get("lon")

    location = Location(
        name=arr_airport_name,
        iata_code=arr_iata,
        lat=arr_lat,
        lng=arr_lng,
    )

    seg_id = f"flight_{flight_iata}_{start_time.strftime('%Y%m%d')}".replace(" ", "_")

    metadata: Dict[str, Any] = {
        "airline": airline_name,
        "airline_iata": airline_info.get("iata", ""),
        "airline_icao": airline_info.get("icao", ""),
        "flight_number": flight_iata,
        "flight_status": flight_data.get("flight_status", "unknown"),
        "departure_airport": dep_airport_name,
        "departure_iata": dep_iata,
        "departure_terminal": dep.get("terminal"),
        "departure_gate": dep.get("gate"),
        "departure_delay_minutes": dep.get("delay"),
        "arrival_airport": arr_airport_name,
        "arrival_iata": arr_iata,
        "arrival_terminal": arr.get("terminal"),
        "arrival_gate": arr.get("gate"),
        "arrival_baggage": arr.get("baggage"),
        "arrival_delay_minutes": arr.get("delay"),
        "aircraft_registration": (flight_data.get("aircraft") or {}).get("registration"),
        "live": flight_data.get("live"),
    }

    return TripSegment(
        id=seg_id,
        type=SegmentType.FLIGHT,
        title=f"{airline_name} ({flight_iata})",
        start_time=start_time,
        end_time=end_time,
        location=location,
        cost=default_cost,
        currency=currency,
        metadata=metadata,
        is_confirmed=False,
    )


# ---------------------------------------------------------------------------
# Core Flight Search & Tracking Operations
# ---------------------------------------------------------------------------


def search_flights_aviationstack(
    departure: str,
    arrival: str,
    outbound_date: Optional[str] = None,
    limit: int = 10,
    currency: str = "USD",
    default_cost: float = 0.0,
) -> List[TripSegment]:
    """
    Searches for flights between departure and arrival destinations using AviationStack.
    Resolves city names or airport codes to IATA codes, executes query, and converts to TripSegment models.
    """
    dep_iata = resolve_airport_code(departure, default="DAC")
    arr_iata = resolve_airport_code(arrival, default="LHR")

    params: Dict[str, Any] = {
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
        "limit": min(limit, 50),
    }

    if outbound_date:
        params["flight_date"] = outbound_date.strip()

    raw_response = query_aviationstack("flights", params=params)
    flights = raw_response.get("data", [])

    if not flights or not isinstance(flights, list):
        logger.info(f"No flight data returned for route {dep_iata} -> {arr_iata}")
        return []

    segments: List[TripSegment] = []
    for f in flights:
        try:
            seg = parse_flight_to_trip_segment(f, default_cost=default_cost, currency=currency)
            segments.append(seg)
        except Exception as exc:
            logger.debug(f"Skipping unparseable flight record: {exc}")
            continue

    return segments


def get_flight_status(flight_number: str) -> Dict[str, Any]:
    """
    Retrieves real-time tracking, status, delay, terminal, gate, and baggage details
    for a specific flight number (e.g. 'AA100', 'QF2019', 'BG001').
    """
    cleaned_num = re.sub(r"\s+", "", flight_number).upper()

    params: Dict[str, Any] = {
        "flight_iata": cleaned_num,
        "limit": 1,
    }

    raw = query_aviationstack("flights", params=params)
    data = raw.get("data", [])

    if not data:
        # Fallback to searching by flight number without airline prefix
        num_only = re.sub(r"^[A-Z]{2,3}", "", cleaned_num)
        if num_only:
            raw = query_aviationstack("flights", params={"flight_number": num_only, "limit": 1})
            data = raw.get("data", [])

    if not data:
        return {
            "flight_number": cleaned_num,
            "status": "NOT_FOUND",
            "message": f"No real-time flight records found for {cleaned_num}.",
        }

    flight_entry = data[0]
    dep = flight_entry.get("departure") or {}
    arr = flight_entry.get("arrival") or {}
    airline = flight_entry.get("airline") or {}
    aircraft = flight_entry.get("aircraft") or {}
    live = flight_entry.get("live") or {}

    return {
        "flight_number": cleaned_num,
        "airline": airline.get("name", "Unknown Airline"),
        "status": flight_entry.get("flight_status", "unknown"),
        "flight_date": flight_entry.get("flight_date"),
        "departure": {
            "airport": dep.get("airport"),
            "iata": dep.get("iata"),
            "terminal": dep.get("terminal"),
            "gate": dep.get("gate"),
            "scheduled": dep.get("scheduled"),
            "estimated": dep.get("estimated"),
            "actual": dep.get("actual"),
            "delay_minutes": dep.get("delay") or 0,
        },
        "arrival": {
            "airport": arr.get("airport"),
            "iata": arr.get("iata"),
            "terminal": arr.get("terminal"),
            "gate": arr.get("gate"),
            "baggage": arr.get("baggage"),
            "scheduled": arr.get("scheduled"),
            "estimated": arr.get("estimated"),
            "actual": arr.get("actual"),
            "delay_minutes": arr.get("delay") or 0,
        },
        "aircraft": {
            "registration": aircraft.get("registration"),
            "iata": aircraft.get("iata"),
            "icao": aircraft.get("icao"),
        },
        "live": live if live else None,
    }


def track_flight_disruption(flight_number: str) -> Optional[Dict[str, Any]]:
    """
    Checks if a flight is currently disrupted (cancelled, diverted, or delayed by > 15 minutes).
    Returns a structured disruption payload compatible with ZICO's DisruptionEvent.
    """
    status_data = get_flight_status(flight_number)
    if status_data.get("status") == "NOT_FOUND":
        return None

    status_str = str(status_data.get("status", "")).lower()
    dep_delay = (status_data.get("departure") or {}).get("delay_minutes", 0) or 0
    arr_delay = (status_data.get("arrival") or {}).get("delay_minutes", 0) or 0
    max_delay = max(dep_delay, arr_delay)

    is_cancelled = status_str in ["cancelled", "canceled"]
    is_diverted = status_str in ["diverted", "incident"]
    is_delayed = max_delay >= 15

    if not (is_cancelled or is_diverted or is_delayed):
        return None

    event_type = "CANCELLATION" if is_cancelled else ("WEATHER" if is_diverted else "DELAY")
    reason = (
        f"Flight {flight_number} is cancelled."
        if is_cancelled
        else (f"Flight {flight_number} is delayed by {max_delay} minutes." if is_delayed else f"Flight {flight_number} is diverted.")
    )

    return {
        "flight_number": flight_number,
        "event_type": event_type,
        "delay_minutes": max_delay,
        "reason": reason,
        "flight_status": status_str,
        "departure_iata": (status_data.get("departure") or {}).get("iata"),
        "arrival_iata": (status_data.get("arrival") or {}).get("iata"),
    }


# ---------------------------------------------------------------------------
# LangChain Agent Tools (@tool)
# ---------------------------------------------------------------------------


@tool
def aviationstack_flight_search(
    departure: str,
    arrival: str,
    outbound_date: Optional[str] = None,
    currency: str = "USD",
) -> List[TripSegment]:
    """Search for scheduled or active flights between two airports or cities using AviationStack.

    Args:
        departure: Origin airport IATA code or city name (e.g. 'DAC', 'Dhaka', 'JFK', 'London').
        arrival: Destination airport IATA code or city name (e.g. 'LHR', 'New York', 'BOM').
        outbound_date: Travel date in YYYY-MM-DD format (optional).
        currency: Preferred currency (e.g. 'USD').

    Returns:
        List of strictly validated TripSegment domain models with airport metadata and coordinates.
    """
    return search_flights_aviationstack(
        departure=departure,
        arrival=arrival,
        outbound_date=outbound_date,
        currency=currency,
    )


@tool
def track_flight_status(flight_number: str) -> str:
    """Track real-time flight status, delays, gates, terminals, and baggage carousel for a flight.

    Args:
        flight_number: Airline flight code (e.g. 'AA100', 'QF2019', 'BG001').

    Returns:
        Formatted summary of flight progress, scheduled vs estimated times, delay, terminal, and gate.
    """
    info = get_flight_status(flight_number)
    if info.get("status") == "NOT_FOUND":
        return f"Flight {flight_number}: No real-time status data found."

    dep = info.get("departure", {})
    arr = info.get("arrival", {})
    status = info.get("status", "unknown").upper()
    dep_delay = dep.get("delay_minutes", 0)
    arr_delay = arr.get("delay_minutes", 0)

    summary = (
        f"Flight: {info.get('airline')} ({info.get('flight_number')})\n"
        f"Status: {status}\n"
        f"Departure: {dep.get('airport')} ({dep.get('iata')}) | Scheduled: {dep.get('scheduled')} | Terminal: {dep.get('terminal')} | Gate: {dep.get('gate')} | Delay: {dep_delay} min\n"
        f"Arrival: {arr.get('airport')} ({arr.get('iata')}) | Scheduled: {arr.get('scheduled')} | Terminal: {arr.get('terminal')} | Gate: {arr.get('gate')} | Baggage: {arr.get('baggage')} | Delay: {arr_delay} min"
    )
    return summary


@tool
def lookup_airport(query: str) -> str:
    """Resolve an airport or city query to standard IATA/ICAO codes, coordinates, and timezone.

    Args:
        query: Airport code, city name, or location (e.g. 'Dhaka', 'Heathrow', 'JFK', 'Tokyo').

    Returns:
        Detailed airport metadata including name, IATA, ICAO, city, country, coordinates, and timezone.
    """
    iata = resolve_airport_code(query)
    info = get_airport_info(iata)
    if not info:
        return f"No airport information found for query '{query}'."

    return (
        f"Airport: {info.get('name')}\n"
        f"IATA: {info.get('iata')} | ICAO: {info.get('icao')}\n"
        f"City: {info.get('city')}, {info.get('country_name')} ({info.get('country')})\n"
        f"Coordinates: Lat {info.get('lat')}, Lon {info.get('lon')}\n"
        f"Timezone: {info.get('tz')}"
    )