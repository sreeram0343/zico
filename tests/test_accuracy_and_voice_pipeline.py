"""
Comprehensive unit and integration tests verifying ZICO response accuracy and voice pipeline end-to-end.

Sections tested:
- Hotel routing
- Flight routing
- State isolation (no leakage of flight/hotel data across turns)
- Tool failure handling (controlled error instead of fabricated data)
- Empty retrieval handling
- Invalid tool results rejection by Validator
- Voice upload endpoint validation
- Invalid audio file rejection (empty audio & unsupported formats)
- Whisper failure propagation (no silent fallback to fake text)
- Successful transcription
- Transcript -> ZICO chat workflow
"""

import io
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.validator_agent import ValidatorAgent
from app.core.state import create_initial_state
from app.graph.engine import (
    _extract_airports,
    flight_search_worker_node,
    research_worker_node,
)
from app.graph.supervisor import _classify_intent_heuristically
from app.main import app
from app.services.voice import VoiceService, _generate_synthetic_wav_bytes


# ---------------------------------------------------------------------------
# 1. Routing Tests
# ---------------------------------------------------------------------------


def test_hotel_routing():
    """Verify hotel query routes to research_worker."""
    query = "Hotels in Dubai under 15K"
    route = _classify_intent_heuristically(query)
    assert route == "research_worker"


def test_flight_routing():
    """Verify flight query routes to flight_search_worker and extracts 3-letter IATA codes."""
    query = "Find flights from TRV to LHR"
    route = _classify_intent_heuristically(query)
    assert route == "flight_search_worker"

    origin, dest, origin_name, dest_name = _extract_airports(query)
    assert origin == "TRV"
    assert dest == "LHR"
    assert "TRV" in origin_name
    assert "LHR" in dest_name


def test_destination_research_routing():
    """Verify attraction query routes to research_worker."""
    query = "What are the best places to visit in Dubai?"
    route = _classify_intent_heuristically(query)
    assert route == "research_worker"


# ---------------------------------------------------------------------------
# 2. State Isolation Tests
# ---------------------------------------------------------------------------


def test_state_isolation_flight_to_hotel():
    """Verify flight results from a previous turn do not leak into a hotel search response."""
    # Simulate turn 1 state with flight results
    state_turn1 = {
        "messages": [HumanMessage(content="Find flights from TRV to LHR")],
        "flight_search_results": {
            "status": "RESULTS",
            "flights": [{"flightNumber": "AI-101", "airline": "Air India"}],
        },
        "itinerary": [],
    }

    # Now run research_worker_node for turn 2
    state_turn2 = {
        "messages": [
            HumanMessage(content="Find flights from TRV to LHR"),
            AIMessage(content="Here are flights"),
            HumanMessage(content="Hotels in Dubai under 15K"),
        ],
        "flight_search_results": state_turn1["flight_search_results"],
        "itinerary": [],
    }

    turn2_output = research_worker_node(state_turn2)
    # The output of research_worker must explicitly reset flight_search_results
    assert turn2_output.get("flight_search_results") == {}
    assert "flights" not in turn2_output.get("flight_search_results", {})


# ---------------------------------------------------------------------------
# 3. Tool Failure Handling Tests
# ---------------------------------------------------------------------------


def test_flight_tool_failure_returns_controlled_error():
    """Verify flight tool failure returns a controlled error message without fabricating data."""
    state = {
        "messages": [HumanMessage(content="Find flights from TRV to LHR")],
        "itinerary": [],
    }

    from app.tools.flight_search import search_flights
    with patch.object(type(search_flights), "invoke", side_effect=Exception("API connection timeout")):
        result = flight_search_worker_node(state)

    msg_content = result["messages"][0].content
    assert "Unable to retrieve live flight data" in msg_content or "flight provider service error" in msg_content
    # Ensure controlled error status and zero fabricated flights
    assert result.get("flight_search_results", {}).get("status") == "ERROR"
    assert result.get("flight_search_results", {}).get("flights") == []


def test_research_tool_failure_returns_controlled_error():
    """Verify research tool failure returns a controlled notice without fabricating hotels."""
    state = {
        "messages": [HumanMessage(content="Hotels in Dubai under 15K")],
        "itinerary": [],
    }

    with patch("app.tools.tavily_search.TavilySearchClient.search", side_effect=Exception("Tavily down")):
        result = research_worker_node(state)

    msg_content = result["messages"][0].content
    assert "No verified travel or accommodation options were found" in msg_content
    # Ensure no fabricated flight results
    assert result.get("flight_search_results") == {}


# ---------------------------------------------------------------------------
# 4. Empty Retrieval Handling Tests
# ---------------------------------------------------------------------------


def test_empty_flight_retrieval_handling():
    """Verify empty flight search returns a graceful no-results message without fabrication."""
    state = {
        "messages": [HumanMessage(content="Find flights from TRV to LHR on 2026-12-25")],
        "itinerary": [],
    }

    with patch("app.graph.engine.search_flights", return_value=[]):
        result = flight_search_worker_node(state)

    msg_content = result["messages"][0].content
    assert "No live flight options were found" in msg_content or "No available flights found" in msg_content
    assert result.get("flight_search_results", {}).get("status") == "NO_RESULTS"
    assert result.get("flight_search_results", {}).get("flights") == []


# ---------------------------------------------------------------------------
# 5. Validator Tests for Invalid Tool Results
# ---------------------------------------------------------------------------


def test_validator_rejects_route_mismatch():
    """Verify Validator rejects flight results where origin/destination mismatch the query."""
    validator = ValidatorAgent()
    state = create_initial_state(
        user_query="Find flights from TRV to LHR",
        session_id="sess-val-1",
        request_id="req-val-1",
    )
    state["intent"] = "FLIGHT_SEARCH"
    state["flight_status"] = "success"
    # Result has DXB instead of LHR as arrival
    state["flight_results"] = [
        {
            "flight_number": "EK523",
            "departure_airport": "TRV",
            "arrival_airport": "DXB",
            "price": 18000,
            "currency": "INR",
        }
    ]

    res = validator.validate(state)
    assert res["validation_status"] == "failed"
    assert any("Mismatched flight arrival" in err for err in res["validation_errors"])


def test_validator_rejects_same_origin_destination_contradiction():
    """Verify Validator rejects identical origin and destination."""
    validator = ValidatorAgent()
    state = create_initial_state(
        user_query="Find flights from DXB to DXB",
        session_id="sess-val-2",
        request_id="req-val-2",
    )
    state["intent"] = "FLIGHT_SEARCH"
    state["origin"] = "DXB"
    state["destination"] = "DXB"

    res = validator.validate(state)
    assert res["validation_status"] == "failed"
    assert any("contradiction" in err.lower() or "identical" in err.lower() for err in res["validation_errors"])


# ---------------------------------------------------------------------------
# 6. Voice API & Transcription Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voice_upload_valid_audio():
    """Verify voice upload with valid audio payload returns 200 and transcript."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        wav_content = _generate_synthetic_wav_bytes()
        files = {"file": ("query.wav", io.BytesIO(wav_content), "audio/wav")}

        with patch(
            "app.services.voice.VoiceService.transcribe_audio",
            new_callable=AsyncMock,
            return_value="Hotels in Dubai under 15K",
        ):
            resp = await client.post("/api/v1/voice/transcribe", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["transcript"] == "Hotels in Dubai under 15K"
            assert data["filename"] == "query.wav"


@pytest.mark.asyncio
async def test_voice_upload_empty_audio_rejected():
    """Verify voice upload with 0-byte audio payload is rejected with 400 Bad Request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
        resp = await client.post("/api/v1/voice/transcribe", files=files)
        assert resp.status_code == 400
        assert "cannot be empty" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_voice_upload_unsupported_extension_rejected():
    """Verify voice upload with unsupported file format is rejected with 400 Bad Request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("recording.txt", io.BytesIO(b"hello world"), "text/plain")}
        resp = await client.post("/api/v1/voice/transcribe", files=files)
        assert resp.status_code == 400
        assert "unsupported audio format" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_whisper_failure_propagated_visibly():
    """Verify Whisper STT failure raises an error visibly rather than silently returning fallback text."""
    svc = VoiceService()
    svc.openai_api_key = "sk-live-mock-key"
    svc.force_whisper = True

    dummy_audio = _generate_synthetic_wav_bytes()

    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("OpenAI quota exceeded")
        )
        mock_openai_cls.return_value = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            await svc.transcribe_audio(dummy_audio, filename="audio.wav")

        assert "OpenAI Whisper transcription failed" in str(exc_info.value)
        assert "quota exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_successful_transcription_service():
    """Verify VoiceService transcription returns transcript and logs lifecycle."""
    svc = VoiceService()
    dummy_audio = _generate_synthetic_wav_bytes()

    transcript = await svc.transcribe_audio(dummy_audio, filename="test.wav")
    assert isinstance(transcript, str)
    assert len(transcript) > 0


# ---------------------------------------------------------------------------
# 7. Transcript -> ZICO Chat Pipeline Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_to_chat_pipeline():
    """Verify that a transcribed voice string successfully feeds into the ZICO chat workflow."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Step 1: Voice transcription
        wav_content = _generate_synthetic_wav_bytes()
        files = {"file": ("query.wav", io.BytesIO(wav_content), "audio/wav")}

        with patch(
            "app.services.voice.VoiceService.transcribe_audio",
            new_callable=AsyncMock,
            return_value="What are the best places to visit in Dubai?",
        ):
            trans_resp = await client.post("/api/v1/voice/transcribe", files=files)
            assert trans_resp.status_code == 200
            transcribed_text = trans_resp.json()["transcript"]
            assert transcribed_text == "What are the best places to visit in Dubai?"

        # Step 2: Feed transcript into normal ZICO chat pipeline
        chat_resp = await client.post(
            "/api/v1/chat",
            json={
                "message": transcribed_text,
                "trip_id": "test_voice_trip_123",
                "user_id": "test_user_456",
                "is_voice": True,
            },
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert "response" in chat_data or "reply" in chat_data
        response_text = chat_data.get("response") or chat_data.get("reply")
        assert len(response_text) > 0
