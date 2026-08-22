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
