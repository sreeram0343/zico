from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.graph.state import TripSegment
from app.tools.flight_search import FlightSearchValidationError, search_flights

router = APIRouter()


class FlightSearchQuery(BaseModel):
    departure_id: str = Field(description="Departure airport code, e.g., JFK or SFO")
    arrival_id: str = Field(description="Arrival airport code, e.g., LHR or CDG")
    outbound_date: str = Field(description="Outbound date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(default=None, description="Optional return date in YYYY-MM-DD format")
    currency: str = Field(default="USD", description="Currency code (USD, EUR, GBP)")


@router.post("/search", response_model=List[TripSegment])
async def search_flights_endpoint(query: FlightSearchQuery) -> List[TripSegment]:
    """
    Direct flight search endpoint returning strictly typed List[TripSegment].
    """
    try:
        results = search_flights.invoke({
            "departure_id": query.departure_id,
            "arrival_id": query.arrival_id,
            "outbound_date": query.outbound_date,
            "return_date": query.return_date,
            "currency": query.currency,
        })
        return results
    except FlightSearchValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Flight validation error: {str(exc)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Flight search failed: {str(exc)}")
