from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Trip
from app.db.session import get_db
from app.graph.state import TripConstraints, TripSegment

router = APIRouter()


class TripCreateRequest(BaseModel):
    id: Optional[str] = None
    user_id: str
    itinerary: List[TripSegment] = Field(default_factory=list)
    constraints: TripConstraints = Field(default_factory=TripConstraints)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TripUpdateRequest(BaseModel):
    itinerary: Optional[List[TripSegment]] = None
    constraints: Optional[TripConstraints] = None
    metadata: Optional[Dict[str, Any]] = None


class TripResponse(BaseModel):
    id: str
    user_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    state_json: Dict[str, Any]


@router.post("", response_model=TripResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=TripResponse, status_code=status.HTTP_201_CREATED)
async def create_trip(
    payload: TripCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> TripResponse:
    """Creates and persists a new trip."""
    trip_id = payload.id or f"trip_{uuid.uuid4().hex[:10]}"
    state_data = {
        "trip_id": trip_id,
        "user_id": payload.user_id,
        "itinerary": [s.model_dump(mode="json") for s in payload.itinerary],
        "constraints": payload.constraints.model_dump(mode="json"),
        "metadata": payload.metadata,
    }

    trip = Trip(
        id=trip_id,
        user_id=payload.user_id,
        state_json=state_data,
    )
    db.add(trip)
    await db.commit()
    await db.refresh(trip)

    return TripResponse(
        id=trip.id,
        user_id=trip.user_id,
        created_at=str(trip.created_at) if trip.created_at else None,
        updated_at=str(trip.updated_at) if trip.updated_at else None,
        state_json=trip.state_json,
    )


@router.get("/{trip_id}", response_model=TripResponse)
async def get_trip(
    trip_id: str,
    db: AsyncSession = Depends(get_db),
) -> TripResponse:
    """Retrieves a trip by ID."""
    result = await db.execute(select(Trip).where(Trip.id == trip_id))
    trip = result.scalars().first()
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    return TripResponse(
        id=trip.id,
        user_id=trip.user_id,
        created_at=str(trip.created_at) if trip.created_at else None,
        updated_at=str(trip.updated_at) if trip.updated_at else None,
        state_json=trip.state_json,
    )


@router.get("", response_model=List[TripResponse])
@router.get("/", response_model=List[TripResponse])
async def list_trips(
    user_id: Optional[str] = Query(None, description="Filter trips by user ID"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> List[TripResponse]:
    """Lists saved trips with optional user_id filtering."""
    query = select(Trip)
    if user_id:
        query = query.where(Trip.user_id == user_id)
    query = query.order_by(Trip.updated_at.desc()).limit(limit)

    result = await db.execute(query)
    trips = result.scalars().all()

    return [
        TripResponse(
            id=t.id,
            user_id=t.user_id,
            created_at=str(t.created_at) if t.created_at else None,
            updated_at=str(t.updated_at) if t.updated_at else None,
            state_json=t.state_json,
        )
        for t in trips
    ]


@router.put("/{trip_id}", response_model=TripResponse)
async def update_trip(
    trip_id: str,
    payload: TripUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> TripResponse:
    """Updates an existing trip's itinerary and constraints."""
    result = await db.execute(select(Trip).where(Trip.id == trip_id))
    trip = result.scalars().first()
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    current_state = dict(trip.state_json or {})
    if payload.itinerary is not None:
        current_state["itinerary"] = [s.model_dump(mode="json") for s in payload.itinerary]
    if payload.constraints is not None:
        current_state["constraints"] = payload.constraints.model_dump(mode="json")
    if payload.metadata is not None:
        current_state["metadata"] = payload.metadata

    trip.state_json = current_state
    await db.commit()
    await db.refresh(trip)

    return TripResponse(
        id=trip.id,
        user_id=trip.user_id,
        created_at=str(trip.created_at) if trip.created_at else None,
        updated_at=str(trip.updated_at) if trip.updated_at else None,
        state_json=trip.state_json,
    )
