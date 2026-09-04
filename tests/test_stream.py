import json
import pytest
from starlette.testclient import TestClient

from app.main import app


def test_websocket_streaming_prompt_flow():
    """Verify WebSocket connection, prompt transmission, node event streaming, and turn completion."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream/trip_stream_test_101") as websocket:
        # Send prompt
        payload = {
            "type": "prompt",
            "message": "Find flight policy rules for baggage limits",
            "user_id": "user_streamer",
        }
        websocket.send_text(json.dumps(payload))

        received_types = []
        # Receive streamed messages until turn_complete
        for _ in range(10):
            data = websocket.receive_json()
            received_types.append(data.get("type"))
            if data.get("type") == "turn_complete":
                break

        assert "node_update" in received_types or "turn_complete" in received_types


def test_websocket_invalid_json_handling():
    """Verify that malformed JSON payloads return error messages without killing the connection."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream/trip_stream_test_102") as websocket:
        websocket.send_text("THIS IS NOT JSON")
        data = websocket.receive_json()
        assert data["type"] == "error"
        assert "Invalid JSON" in data["message"]


def test_websocket_stream_normalized_event_structure():
    """Verify that streamed events contain normalized status, tool_call, or state_update frames."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream/trip_stream_test_103") as websocket:
        payload = {
            "type": "prompt",
            "message": "Search flights from Mumbai to Pune",
            "user_id": "test_normalizer",
        }
        websocket.send_text(json.dumps(payload))

        received = []
        for _ in range(40):
            frame = websocket.receive_json()
            received.append(frame)
            if frame.get("type") == "turn_complete":
                break

        types = [f.get("type") for f in received]
        assert "status" in types
        assert "turn_complete" in types
        # Check that status frame has node and content
        status_frame = next(f for f in received if f.get("type") == "status")
        assert "node" in status_frame
        assert "content" in status_frame

