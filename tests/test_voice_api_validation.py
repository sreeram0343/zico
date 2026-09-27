"""
Unit and integration tests for Voice API endpoint validations and error handling.
"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.voice import _generate_synthetic_wav_bytes


@pytest.mark.asyncio
async def test_voice_transcribe_unsupported_format_rejected():
    """Verify non-audio files (e.g. .txt, .pdf) are rejected with 400 Bad Request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_file = io.BytesIO(b"This is a text file, not audio.")
        files = {"file": ("malicious.txt", fake_file, "text/plain")}

        resp = await client.post("/api/v1/voice/transcribe", files=files)
        assert resp.status_code == 400
        data = resp.json()
        assert "Unsupported audio format" in data["detail"]


@pytest.mark.asyncio
async def test_voice_transcribe_valid_wav_success():
    """Verify valid WAV audio upload transcribes cleanly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        wav_content = _generate_synthetic_wav_bytes()
        files = {"file": ("query.wav", io.BytesIO(wav_content), "audio/wav")}

        with patch(
            "app.services.voice.VoiceService.transcribe_audio",
            new_callable=AsyncMock,
            return_value="Find flights from Mumbai to London tomorrow.",
        ):
            resp = await client.post("/api/v1/voice/transcribe", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["transcript"] == "Find flights from Mumbai to London tomorrow."
            assert data["filename"] == "query.wav"


@pytest.mark.asyncio
async def test_voice_synthesize_empty_text_rejected():
    """Verify empty text for speech synthesis is rejected with 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/voice/synthesize", json={"text": "   "})
        assert resp.status_code == 400
        assert "cannot be empty" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_voice_synthesize_valid_text_returns_audio():
    """Verify synthesize endpoint returns audio payload with appropriate media type."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/voice/synthesize",
            json={"text": "Flight EK522 is boarding at Gate 14."},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] in ["audio/wav", "audio/mpeg"]
        assert len(resp.content) > 40
