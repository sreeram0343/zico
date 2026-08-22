from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.tools.flight_search import FlightOption, search_flights

router = APIRouter()


class FlightSearchQuery(BaseModel):
    departure_id: str = Field(description="Departure airport code, e.g., JFK or SFO")
    arrival_id: str = Field(description="Arrival airport code, e.g., LHR or CDG")
    outbound_date: str = Field(description="Outbound date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(default=None, description="Optional return date in YYYY-MM-DD format")
    currency: str = Field(default="USD", description="Currency code (USD, EUR, GBP)")


@router.post("/search", response_model=List[FlightOption])
async def search_flights_endpoint(query: FlightSearchQuery) -> List[FlightOption]:
    """
    Direct flight search endpoint interfacing with SerpApi Google Flights.
    """
    results = search_flights.invoke({
        "departure_id": query.departure_id,
        "arrival_id": query.arrival_id,
        "outbound_date": query.outbound_date,
        "return_date": query.return_date,
        "currency": query.currency,
    })
    return results
