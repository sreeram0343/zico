from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import serpapi
from langchain_core.tools import tool
from app.core.config import settings

# Initialize SerpApi client
client = serpapi.Client(api_key=settings.SERPAPI_API_KEY)


class FlightOption(BaseModel):
    """Structured Pydantic schema for a parsed flight option from SerpApi."""

    airline: str = Field(default="Unknown", description="Operating airline name(s)")
    flight_number: str = Field(default="Unknown", description="Flight identification number(s)")
    departure_airport: str = Field(default="", description="Departure airport name or IATA code")
    departure_time: str = Field(default="", description="Departure timestamp string")
    arrival_airport: str = Field(default="", description="Arrival airport name or IATA code")
    arrival_time: str = Field(default="", description="Arrival timestamp string")
    duration: int = Field(default=0, description="Total flight duration in minutes")
    price: Optional[float] = Field(default=None, description="Ticket price in specified currency")
    currency: str = Field(default="USD", description="ISO currency code")
    error: Optional[str] = Field(default=None, description="Error message if search failed")

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backward compatibility."""
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        """Allow 'in' membership testing for backward compatibility."""
        return hasattr(self, item) and getattr(self, item) is not None

    def get(self, item: str, default: Any = None) -> Any:
        """Allow dict-like get access."""
        val = getattr(self, item, default)
        return default if val is None else val


def _parse_flight_entry(flight_entry: Dict[str, Any], currency: str = "USD") -> FlightOption:
    """Parse a single flight option from SerpApi Google Flights response into a Pydantic FlightOption."""
    legs = flight_entry.get("flights", [])
    if not legs:
        return FlightOption(
            airline="Unknown",
            flight_number="Unknown",
            departure_airport="",
            departure_time="",
            arrival_airport="",
            arrival_time="",
            duration=flight_entry.get("total_duration", 0),
            price=flight_entry.get("price"),
            currency=currency,
        )

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

    return FlightOption(
        airline=airline_str,
        flight_number=flight_number_str,
        departure_airport=dep_airport.get("name") or dep_airport.get("id", ""),
        departure_time=dep_time,
        arrival_airport=arr_airport.get("name") or arr_airport.get("id", ""),
        arrival_time=arr_time,
        duration=duration,
        price=price,
        currency=currency,
    )


@tool
def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    currency: str = "USD",
    return_date: Optional[str] = None,
) -> List[FlightOption]:
    """Search for flights using SerpApi Google Flights engine.

    Args:
        departure_id: Departure airport IATA code (e.g. 'JFK', 'SFO', 'LHR').
        arrival_id: Arrival airport IATA code (e.g. 'LAX', 'CDG', 'DXB').
        outbound_date: Outbound travel date in YYYY-MM-DD format (e.g. '2026-09-15').
        currency: Preferred 3-letter currency code (e.g. 'USD', 'EUR', 'GBP'). Defaults to 'USD'.
        return_date: Optional return date in YYYY-MM-DD format for round trips.

    Returns:
        List of structured FlightOption Pydantic models including airline, flight number,
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

        parsed_flights: List[FlightOption] = [
            _parse_flight_entry(f, currency=currency) for f in flight_list
        ]
        return parsed_flights

    except Exception as exc:
        return [
            FlightOption(
                error=f"Failed to fetch flight data from SerpApi: {str(exc)}",
                airline="",
                flight_number="",
                departure_airport="",
                departure_time="",
                arrival_airport="",
                arrival_time="",
                duration=0,
                price=None,
                currency=currency,
            )
        ]
