from typing import Any, Dict, List, Optional
import serpapi
from langchain_core.tools import tool
from app.core.config import settings

# Initialize SerpApi client
client = serpapi.Client(api_key=settings.SERPAPI_API_KEY)


def _parse_flight_entry(flight_entry: Dict[str, Any], currency: str = "USD") -> Dict[str, Any]:
    """Parse a single flight option from SerpApi Google Flights response."""
    legs = flight_entry.get("flights", [])
    if not legs:
        return {
            "airline": "Unknown",
            "flight_number": "Unknown",
            "departure_airport": "",
            "departure_time": "",
            "arrival_airport": "",
            "arrival_time": "",
            "duration": flight_entry.get("total_duration", 0),
            "price": flight_entry.get("price"),
            "currency": currency,
        }

    airlines = [leg.get("airline") for leg in legs if leg.get("airline")]
    airline_str = ", ".join(dict.fromkeys(airlines)) if airlines else "Unknown"

    flight_numbers = [leg.get("flight_number") for leg in legs if leg.get("flight_number")]
    flight_number_str = ", ".join(flight_numbers) if flight_numbers else "Unknown"

    first_leg = legs[0]
    last_leg = legs[-1]

    dep_airport = first_leg.get("departure_airport", {})
    arr_airport = last_leg.get("arrival_airport", {})

    dep_time = dep_airport.get("time", "")
    arr_time = arr_airport.get("time", "")

    duration = flight_entry.get("total_duration") or sum(leg.get("duration", 0) for leg in legs)
    price = flight_entry.get("price")

    return {
        "airline": airline_str,
        "flight_number": flight_number_str,
        "departure_airport": dep_airport.get("name") or dep_airport.get("id", ""),
        "departure_time": dep_time,
        "arrival_airport": arr_airport.get("name") or arr_airport.get("id", ""),
        "arrival_time": arr_time,
        "duration": duration,
        "price": price,
        "currency": currency,
    }


@tool
def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    currency: str = "USD",
    return_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for flights using SerpApi Google Flights engine.

    Args:
        departure_id: Departure airport IATA code (e.g. 'JFK', 'SFO', 'LHR').
        arrival_id: Arrival airport IATA code (e.g. 'LAX', 'CDG', 'DXB').
        outbound_date: Outbound travel date in YYYY-MM-DD format (e.g. '2026-09-15').
        currency: Preferred 3-letter currency code (e.g. 'USD', 'EUR', 'GBP'). Defaults to 'USD'.
        return_date: Optional return date in YYYY-MM-DD format for round trips.

    Returns:
        List of structured flight information dictionaries including airline, flight number,
        departure time, arrival time, duration, and price.
    """
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

        best_flights = raw_results.get("best_flights", [])
        flight_list = best_flights if best_flights else raw_results.get("other_flights", [])

        parsed_flights: List[Dict[str, Any]] = [
            _parse_flight_entry(f, currency=currency) for f in flight_list
        ]
        return parsed_flights

    except Exception as exc:
        return [{
            "error": f"Failed to fetch flight data from SerpApi: {str(exc)}",
            "airline": "",
            "flight_number": "",
            "departure_airport": "",
            "departure_time": "",
            "arrival_airport": "",
            "arrival_time": "",
            "duration": 0,
            "price": None,
            "currency": currency,
        }]
