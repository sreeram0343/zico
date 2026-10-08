import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from app.core.exceptions import ProviderError
from app.core.request_context import get_request_id, get_session_id
from app.graph.supervisor import supervisor_node
from app.main import app


def test_supervisor_node_raises_provider_error_on_openai_429():
    """Verify supervisor_node catches OpenAI 429 / insufficient_quota and raises ProviderError without bypassing."""
    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception(
        "Error code: 429 - {'error': {'message': 'You have no credits remaining.', 'type': 'insufficient_quota', 'code': 'credit_balance_exhausted'}}"
    )

    with patch("app.graph.supervisor.ChatOpenAI", return_value=mock_llm), \
         patch("app.graph.supervisor.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: mock_chain)):
        # Make is_mocked True so supervisor executes LLM branch
        mock_llm.assert_called = True

        state = {
            "messages": ["Find flights from Mumbai to London"],
            "request_id": "req_quota_unit_1",
            "session_id": "sess_quota_unit_1",
        }

        with pytest.raises(ProviderError) as exc_info:
            supervisor_node(state)

        err = exc_info.value
        assert "quota" in str(err.message).lower()
        assert "ZICO is temporarily unable to process this request because the AI service is unavailable." in err.safe_message
        assert get_request_id() == "req_quota_unit_1"
        assert get_session_id() == "sess_quota_unit_1"


def test_supervisor_node_raises_provider_error_on_openai_timeout():
    """Verify supervisor_node catches OpenAI timeout and raises ProviderError."""
    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = TimeoutError("Request timed out after 30.0 seconds")

    with patch("app.graph.supervisor.ChatOpenAI", return_value=mock_llm), \
         patch("app.graph.supervisor.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=lambda self, other: mock_chain)):
        mock_llm.assert_called = True

        state = {
            "messages": ["Check flight status"],
            "request_id": "req_timeout_unit_1",
            "session_id": "sess_timeout_unit_1",
        }

        with pytest.raises(ProviderError) as exc_info:
            supervisor_node(state)

        err = exc_info.value
        assert "timed out" in err.safe_message.lower()


def test_openai_success_produces_assistant_response():
    """Verify successful prompt via WebSocket produces assistant response and turn_complete."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream/trip_test_success_1") as websocket:
        payload = {
            "type": "prompt",
            "message": "Find flight policy rules for baggage limits",
            "user_id": "traveler_1",
            "request_id": "req_test_success_1",
            "session_id": "trip_test_success_1",
        }
        websocket.send_text(json.dumps(payload))

        received = []
        for _ in range(25):
            frame = websocket.receive_json()
            received.append(frame)
            if frame.get("type") == "turn_complete":
                break

        types = [f.get("type") for f in received]
        assert "turn_complete" in types

        # Must have received either streaming tokens or node_update with content
        assistant_contents = [
            f.get("content") or f.get("message")
            for f in received
            if f.get("type") in ("node_update", "token") and (f.get("content") or f.get("message"))
        ]
        assert len(assistant_contents) > 0


def test_websocket_stream_handles_openai_429_visibly():
    """
    Verify that when graph encounters an OpenAI 429 quota exhaustion,
    WebSocket stream returns a structured safe error and does NOT mark turn as SUCCESS.
    """
    client = TestClient(app)

    async def mock_stream_quota_exhausted(*args, **kwargs):
        raise ProviderError(
            "OpenAI quota exhausted: credit_balance_exhausted (429)",
            safe_message="ZICO is temporarily unable to process this request because the AI service is unavailable. Please try again later.",
            details={"provider": "openai", "error_type": "insufficient_quota"},
        )
        yield {}

    with patch("app.api.stream.graph_engine.astream_events", side_effect=mock_stream_quota_exhausted):
        with client.websocket_connect("/ws/stream/trip_test_quota_ws") as websocket:
            payload = {
                "type": "prompt",
                "message": "Find flights from Mumbai to London",
                "user_id": "traveler_quota",
                "request_id": "req_quota_ws_test",
                "session_id": "trip_test_quota_ws",
            }
            websocket.send_text(json.dumps(payload))

            received = []
            for _ in range(15):
                frame = websocket.receive_json()
                received.append(frame)
                if frame.get("type") == "error":
                    break

            types = [f.get("type") for f in received]
            assert "error" in types
            assert "turn_complete" not in types  # Turn must NOT be marked as SUCCESS!

            err_frame = next(f for f in received if f.get("type") == "error")
            assert err_frame.get("status_code") == 429
            assert err_frame.get("error_code") == "ZICO_PROVIDER_ERROR"
            assert "ZICO is temporarily unable to process this request because the AI service is unavailable." in err_frame.get("message", "")
            # Ensure no credentials or tracebacks are leaked
            assert "sk-" not in str(err_frame)
            assert "traceback" not in str(err_frame).lower()


def test_websocket_stream_handles_openai_timeout_visibly():
    """Verify that when graph encounters a timeout, WebSocket returns structured 504 error."""
    client = TestClient(app)

    async def mock_stream_timeout(*args, **kwargs):
        raise ProviderError(
            "OpenAI request timed out",
            safe_message="ZICO request timed out while contacting the AI service. Please try again later.",
            details={"provider": "openai", "error_type": "timeout"},
        )
        yield {}

    with patch("app.api.stream.graph_engine.astream_events", side_effect=mock_stream_timeout):
        with client.websocket_connect("/ws/stream/trip_test_timeout_ws") as websocket:
            payload = {
                "type": "prompt",
                "message": "Check flight delay status",
                "user_id": "traveler_timeout",
                "request_id": "req_timeout_ws_test",
                "session_id": "trip_test_timeout_ws",
            }
            websocket.send_text(json.dumps(payload))

            received = []
            for _ in range(15):
                frame = websocket.receive_json()
                received.append(frame)
                if frame.get("type") == "error":
                    break

            err_frame = next(f for f in received if f.get("type") == "error")
            assert err_frame.get("status_code") == 504
            assert err_frame.get("error_code") == "ZICO_TIMEOUT_ERROR"
            assert "timed out" in err_frame.get("message", "").lower()


def test_websocket_stream_handles_empty_response():
    """Verify that when graph finishes with empty response, WebSocket emits error frame instead of turn_complete."""
    client = TestClient(app)

    async def mock_stream_empty(*args, **kwargs):
        # Emits chain events but never emits tokens or AIMessages
        yield {"event": "on_chain_start", "name": "input_node", "data": {}, "metadata": {}}
        yield {"event": "on_chain_end", "name": "validator_node", "data": {"output": {}}, "metadata": {}}

    with patch("app.api.stream.graph_engine.astream_events", side_effect=mock_stream_empty):
        with client.websocket_connect("/ws/stream/trip_test_empty_ws") as websocket:
            payload = {
                "type": "prompt",
                "message": "test empty query",
                "user_id": "traveler_empty",
                "request_id": "req_empty_test",
                "session_id": "trip_test_empty_ws",
            }
            websocket.send_text(json.dumps(payload))

            received = []
            for _ in range(10):
                frame = websocket.receive_json()
                received.append(frame)
                if frame.get("type") == "error":
                    break

            types = [f.get("type") for f in received]
            assert "error" in types
            assert "turn_complete" not in types

            err_frame = next(f for f in received if f.get("type") == "error")
            assert err_frame.get("error_code") == "EMPTY_RESPONSE"


def test_request_id_and_session_id_propagation():
    """Verify custom request_id and session_id are propagated across all frames and context."""
    client = TestClient(app)
    custom_req_id = "req_custom_trace_987"
    custom_sess_id = "trip_custom_trace_654"

    with client.websocket_connect(f"/ws/stream/{custom_sess_id}") as websocket:
        payload = {
            "type": "prompt",
            "message": "Find flight policy rules",
            "user_id": "traveler_trace",
            "request_id": custom_req_id,
            "session_id": custom_sess_id,
        }
        websocket.send_text(json.dumps(payload))

        received = []
        for _ in range(25):
            frame = websocket.receive_json()
            received.append(frame)
            if frame.get("type") == "turn_complete":
                break

        # Check that frames include the custom request_id and session_id
        for frame in received:
            if frame.get("type") in ("status", "turn_complete", "node_update", "token"):
                assert frame.get("request_id") == custom_req_id
                assert frame.get("session_id") == custom_sess_id


def test_voice_transcription_quota_exhausted_handling():
    """Verify voice transcription returns clean 429 when quota is exhausted."""
    client = TestClient(app)

    simulated_whisper_quota = ProviderError(
        "OpenAI Whisper quota exhausted",
        safe_message="Voice transcription is temporarily unavailable because the AI service quota is exhausted. Please type your query or try again later.",
        details={"provider": "openai", "service": "whisper", "error_type": "insufficient_quota"},
    )

    with patch("app.services.voice.VoiceService.transcribe_audio", side_effect=simulated_whisper_quota):
        # 1-second dummy WAV payload
        wav_header = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        response = client.post(
            "/api/v1/voice/transcribe",
            files={"file": ("audio.wav", wav_header, "audio/wav")},
        )
        assert response.status_code == 429
        data = response.json()
        assert "quota is exhausted" in data.get("detail", "")
