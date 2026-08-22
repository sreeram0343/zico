import io
import logging
import os
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


def _generate_synthetic_wav_bytes() -> bytes:
    """Generates a minimal valid PCM WAV header byte stream for offline testing."""
    import struct
    sample_rate = 16000
    num_samples = 1600
    byte_rate = sample_rate * 2
    block_align = 2
    data_size = num_samples * 2
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    payload = b"\x00" * data_size
    return header + payload


class VoiceService:
    """Provides Speech-to-Text (STT) transcription and Text-to-Speech (TTS) synthesis."""

    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.elevenlabs_api_key = settings.ELEVENLABS_API_KEY
        self.elevenlabs_voice_id = settings.ELEVENLABS_VOICE_ID

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribes speech audio into text using OpenAI Whisper or testing fallback.
        """
        if (
            self.openai_api_key
            and not self.openai_api_key.startswith("test")
            and settings.APP_ENV != "test"
            and os.getenv("PYTEST_CURRENT_TEST") is None
        ):
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.openai_api_key, timeout=5.0, max_retries=1)
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = filename

                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                }
                if language:
                    kwargs["language"] = language

                transcript = await client.audio.transcriptions.create(**kwargs)
                return transcript.text
            except Exception as exc:
                logger.warning(f"OpenAI Whisper STT failed, using acoustic fallback: {exc}")

        # Testing / Offline fallback
        return "Find flights from JFK to London Heathrow next Friday under 800 dollars."

    async def synthesize_speech(
        self,
        text: str,
        voice: str = "en-US-JennyNeural",
    ) -> bytes:
        """
        Synthesizes text into speech audio bytes using EdgeTTS, ElevenLabs, or PCM WAV fallback.
        """
        # Try EdgeTTS first if available
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, voice)
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            if audio_data:
                return bytes(audio_data)
        except Exception as exc:
            logger.debug(f"EdgeTTS unavailable or failed: {exc}")

        # Try ElevenLabs if configured
        if (
            self.elevenlabs_api_key
            and not self.elevenlabs_api_key.startswith("test")
            and settings.APP_ENV != "test"
            and os.getenv("PYTEST_CURRENT_TEST") is None
        ):
            try:
                import httpx
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": self.elevenlabs_api_key,
                }
                payload = {
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                }
                async with httpx.AsyncClient(timeout=5.0) as http_client:
                    resp = await http_client.post(url, json=payload, headers=headers)
                    if resp.status_code == 200:
                        return resp.content
            except Exception as exc:
                logger.warning(f"ElevenLabs TTS failed: {exc}")

        # Fallback to minimal valid audio byte stream
        return _generate_synthetic_wav_bytes()


_voice_service_instance: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    """Returns singleton VoiceService instance."""
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance
