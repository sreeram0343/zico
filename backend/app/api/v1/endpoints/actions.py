from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import AuditLog, Trip
from app.db.session import get_db
from app.graph.disruption import apply_recovery_action
from app.graph.state import (
    ActionStatus,
    ActionType,
    PendingAction,
    TripSegment,
)

router = APIRouter()


class ActionDecisionRequest(BaseModel):
    trip_id: str
    action_type: ActionType = ActionType.RESCHEDULE
    description: Optional[str] = "Traveler confirmation"
    payload: Dict[str, Any] = Field(default_factory=dict)
    replacement_segments: Optional[List[TripSegment]] = None
    reason: Optional[str] = None


class ActionDecisionResponse(BaseModel):
    action_id: str
    trip_id: str
    status: ActionStatus
    message: str
    audit_log_id: str
    updated_itinerary: List[TripSegment] = Field(default_factory=list)


@router.post("/{action_id}/approve", response_model=ActionDecisionResponse)
async def approve_action(
    action_id: str,
    body: ActionDecisionRequest,
    db: AsyncSession = Depends(get_db),
) -> ActionDecisionResponse:
    """
    Approves a Human-in-the-Loop PendingAction, modifies the trip itinerary state,
    and appends an immutable entry into the audit trail.
    """
    # 1. Fetch Trip
    result = await db.execute(select(Trip).where(Trip.id == body.trip_id))
    trip = result.scalars().first()
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip '{body.trip_id}' not found.")

    current_state = dict(trip.state_json or {})
    raw_itinerary = current_state.get("itinerary", [])
    itinerary = [
        s if isinstance(s, TripSegment) else TripSegment.model_validate(s)
        for s in raw_itinerary
    ]

    action = PendingAction(
        action_id=action_id,
        action_type=body.action_type,
        description=body.description or "Action approval",
        payload=body.payload,
        requires_explicit_approval=True,
        status=ActionStatus.APPROVED,
    )

    # 2. Apply action to itinerary
    updated_itinerary = apply_recovery_action(
        itinerary,
        action,
        approved_replacement_segments=body.replacement_segments,
    )

    # 3. Update Trip State
    current_state["itinerary"] = [s.model_dump(mode="json") for s in updated_itinerary]
    trip.state_json = current_state

    # 4. Insert Audit Log
    audit = AuditLog(
        id=str(uuid.uuid4()),
        trip_id=trip.id,
        action_type=f"HITL_{action.action_type.value}_APPROVED",
        payload={
            "action_id": action_id,
            "decision": "APPROVED",
            "reason": body.reason,
            "action_payload": body.payload,
        },
    )
    db.add(audit)
    await db.commit()
    await db.refresh(audit)

    return ActionDecisionResponse(
        action_id=action_id,
        trip_id=trip.id,
        status=ActionStatus.APPROVED,
        message=f"Action '{action_id}' successfully approved and applied to itinerary.",
        audit_log_id=audit.id,
        updated_itinerary=updated_itinerary,
    )


@router.post("/{action_id}/reject", response_model=ActionDecisionResponse)
async def reject_action(
    action_id: str,
    body: ActionDecisionRequest,
    db: AsyncSession = Depends(get_db),
) -> ActionDecisionResponse:
    """
    Rejects a Human-in-the-Loop PendingAction, leaving itinerary unchanged,
    and logs the rejection into the audit trail.
    """
    result = await db.execute(select(Trip).where(Trip.id == body.trip_id))
    trip = result.scalars().first()
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip '{body.trip_id}' not found.")

    current_state = dict(trip.state_json or {})
    raw_itinerary = current_state.get("itinerary", [])
    itinerary = [
        s if isinstance(s, TripSegment) else TripSegment.model_validate(s)
        for s in raw_itinerary
    ]

    audit = AuditLog(
        id=str(uuid.uuid4()),
        trip_id=trip.id,
        action_type=f"HITL_{body.action_type.value}_REJECTED",
        payload={
            "action_id": action_id,
            "decision": "REJECTED",
            "reason": body.reason or "Traveler declined suggested modification",
        },
    )
    db.add(audit)
    await db.commit()
    await db.refresh(audit)

    return ActionDecisionResponse(
        action_id=action_id,
        trip_id=trip.id,
        status=ActionStatus.REJECTED,
        message=f"Action '{action_id}' was rejected. Itinerary preserved without modifications.",
        audit_log_id=audit.id,
        updated_itinerary=itinerary,
    )
