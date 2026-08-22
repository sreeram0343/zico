from fastapi import APIRouter
from app.api.v1.endpoints import (
    actions,
    chat,
    flights,
    health,
    rag,
    trips,
    voice,
)

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat & Orchestration"])
api_router.include_router(trips.router, prefix="/trips", tags=["Trips"])
api_router.include_router(actions.router, prefix="/actions", tags=["HITL Actions"])
api_router.include_router(flights.router, prefix="/flights", tags=["Flights"])
api_router.include_router(rag.router, prefix="/rag", tags=["Policy RAG"])
api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])
