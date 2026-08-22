from datetime import datetime, timedelta
import io
from unittest.mock import patch
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.graph.state import Location, SegmentType, TripConstraints, TripSegment
from app.main import app

# In-memory SQLite for isolated API tests
test_async_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture(autouse=True)
async def prepare_test_db():
    async with test_async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Verify root API returns system metadata and status."""
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "Phase 1" in data["phase"]


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Verify /api/v1/health endpoint execution."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "components" in data


@pytest.mark.asyncio
async def test_chat_endpoint_interaction(client: AsyncClient):
    """Verify /api/v1/chat invokes state engine and returns response."""
    payload = {
        "message": "Hello ZICO, can you find flights from JFK to London?",
        "user_id": "user_api_test_01",
        "trip_id": "trip_api_chat_01",
    }
    resp = await client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["trip_id"] == "trip_api_chat_01"
    assert "reply" in data
    assert len(data["reply"]) > 0


@pytest.mark.asyncio
async def test_trips_crud_lifecycle(client: AsyncClient):
    """Verify full CRUD lifecycle for trips."""
    base_time = datetime(2026, 10, 1, 9, 0).isoformat()
    end_time = (datetime(2026, 10, 1, 9, 0) + timedelta(hours=3)).isoformat()

    # 1. Create Trip
    create_payload = {
        "id": "trip_crud_01",
        "user_id": "user_crud_test",
        "itinerary": [
            {
                "id": "seg_fl_01",
                "type": "FLIGHT",
                "title": "Flight to London",
                "start_time": base_time,
                "end_time": end_time,
                "location": {"name": "Heathrow Airport", "iata_code": "LHR"},
                "cost": 450.0,
                "currency": "USD",
                "metadata": {},
                "is_confirmed": True,
            }
        ],
        "constraints": {"max_budget": 1200.0, "min_connection_buffer_minutes": 90},
        "metadata": {"destination": "London"},
    }
    resp = await client.post("/api/v1/trips", json=create_payload)
    assert resp.status_code == 201
    created_trip = resp.json()
    assert created_trip["id"] == "trip_crud_01"
    assert created_trip["user_id"] == "user_crud_test"

    # 2. Get Trip
    get_resp = await client.get("/api/v1/trips/trip_crud_01")
    assert get_resp.status_code == 200
    fetched_trip = get_resp.json()
    assert len(fetched_trip["state_json"]["itinerary"]) == 1

    # 3. List Trips
    list_resp = await client.get("/api/v1/trips?user_id=user_crud_test")
    assert list_resp.status_code == 200
    trips_list = list_resp.json()
    assert len(trips_list) >= 1

    # 4. Update Trip
    update_payload = {
        "constraints": {"max_budget": 1500.0, "min_connection_buffer_minutes": 120},
    }
    put_resp = await client.put("/api/v1/trips/trip_crud_01", json=update_payload)
    assert put_resp.status_code == 200
    updated_data = put_resp.json()
    assert updated_data["state_json"]["constraints"]["max_budget"] == 1500.0


@pytest.mark.asyncio
async def test_hitl_action_approve_and_reject_flow(client: AsyncClient):
    """Verify Human-in-the-Loop action approval, itinerary state update, and audit logging."""
    base_time = datetime(2026, 10, 5, 8, 0).isoformat()
    end_time = (datetime(2026, 10, 5, 8, 0) + timedelta(hours=2)).isoformat()

    # Seed Trip
    await client.post(
        "/api/v1/trips",
        json={
            "id": "trip_hitl_01",
            "user_id": "user_hitl",
            "itinerary": [
                {
                    "id": "flight_orig",
                    "type": "FLIGHT",
                    "title": "Morning Flight",
                    "start_time": base_time,
                    "end_time": end_time,
                    "location": {"name": "CDG Airport", "iata_code": "CDG"},
                    "cost": 200.0,
                }
            ],
        },
    )

    # 1. Approve Action with Replacement
    rebooked_start = (datetime(2026, 10, 5, 11, 0)).isoformat()
    rebooked_end = (datetime(2026, 10, 5, 13, 0)).isoformat()
    approve_payload = {
        "trip_id": "trip_hitl_01",
        "action_type": "RESCHEDULE",
        "description": "Rebooking delayed morning flight",
        "payload": {"impacted_segments": ["flight_orig"]},
        "replacement_segments": [
            {
                "id": "flight_rebooked",
                "type": "FLIGHT",
                "title": "Afternoon Rebooked Flight",
                "start_time": rebooked_start,
                "end_time": rebooked_end,
                "location": {"name": "CDG Airport", "iata_code": "CDG"},
                "cost": 220.0,
            }
        ],
    }
    approve_resp = await client.post("/api/v1/actions/act_test_01/approve", json=approve_payload)
    assert approve_resp.status_code == 200
    approve_data = approve_resp.json()
    assert approve_data["status"] == "APPROVED"
    assert "audit_log_id" in approve_data
    assert len(approve_data["updated_itinerary"]) == 1
    assert approve_data["updated_itinerary"][0]["id"] == "flight_rebooked"

    # 2. Reject Action
    reject_payload = {
        "trip_id": "trip_hitl_01",
        "action_type": "CANCELLATION",
        "reason": "Traveler prefers waiting at the terminal",
    }
    reject_resp = await client.post("/api/v1/actions/act_test_02/reject", json=reject_payload)
    assert reject_resp.status_code == 200
    reject_data = reject_resp.json()
    assert reject_data["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_flights_search_endpoint(client: AsyncClient):
    """Verify /api/v1/flights/search endpoint."""
    mock_results = {
        "best_flights": [
            {
                "flights": [
                    {
                        "airline": "Delta Air Lines",
                        "flight_number": "DL 89",
                        "departure_airport": {"name": "JFK", "id": "JFK", "time": "2026-10-10 19:00"},
                        "arrival_airport": {"name": "CDG", "id": "CDG", "time": "2026-10-11 08:30"},
                        "duration": 450,
                    }
                ],
                "total_duration": 450,
                "price": 680.0,
            }
        ]
    }
    with patch("app.tools.flight_search.client.search", return_value=mock_results):
        payload = {
            "departure_id": "JFK",
            "arrival_id": "CDG",
            "outbound_date": "2026-10-10",
        }
        resp = await client.post("/api/v1/flights/search", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["airline"] == "Delta Air Lines"


@pytest.mark.asyncio
async def test_rag_query_and_index_endpoints(client: AsyncClient):
    """Verify /api/v1/rag/query and /api/v1/rag/index endpoints."""
    # 1. Query
    query_resp = await client.post("/api/v1/rag/query", json={"query": "baggage weight limit"})
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert "context" in q_data
    assert len(q_data["results"]) >= 1

    # 2. Index
    doc_payload = {
        "id": "policy_api_test_01",
        "title": "Train Railpass Cancellation Policy",
        "category": "CANCELLATION",
        "content": "Eurostar tickets may be exchanged up to 1 hour before departure.",
        "metadata": {"source": "RailEurope"},
    }
    index_resp = await client.post("/api/v1/rag/index", json=doc_payload)
    assert index_resp.status_code == 200
    idx_data = index_resp.json()
    assert idx_data["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_voice_endpoints(client: AsyncClient):
    """Verify voice transcription and synthesis endpoints."""
    # 1. Synthesize
    synth_resp = await client.post(
        "/api/v1/voice/synthesize",
        json={"text": "Your flight has been booked successfully."},
    )
    assert synth_resp.status_code == 200
    assert len(synth_resp.content) > 0

    # 2. Transcribe
    audio_file = io.BytesIO(b"RIFF" + b"\x00" * 40)
    files = {"file": ("test_recording.wav", audio_file, "audio/wav")}
    trans_resp = await client.post("/api/v1/voice/transcribe", files=files)
    assert trans_resp.status_code == 200
    t_data = trans_resp.json()
    assert "transcript" in t_data
    assert len(t_data["transcript"]) > 0
