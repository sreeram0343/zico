import pytest
from app.services.voice import VoiceService, _generate_synthetic_wav_bytes, get_voice_service


@pytest.mark.asyncio
async def test_synthetic_wav_bytes_structure():
    """Verify synthetic WAV generator produces valid RIFF WAVE header."""
    wav_bytes = _generate_synthetic_wav_bytes()
    assert len(wav_bytes) > 44
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_voice_service_transcribe_audio():
    """Verify voice transcription returns text transcript."""
    service = VoiceService()
    dummy_audio = _generate_synthetic_wav_bytes()
    transcript = await service.transcribe_audio(dummy_audio, filename="test.wav")
    assert isinstance(transcript, str)
    assert len(transcript) > 0


@pytest.mark.asyncio
async def test_voice_service_synthesize_speech():
    """Verify speech synthesis returns audio stream bytes."""
    service = VoiceService()
    audio = await service.synthesize_speech("Flight BA178 has been rescheduled.")
    assert isinstance(audio, bytes)
    assert len(audio) > 0


@pytest.mark.asyncio
async def test_singleton_get_voice_service():
    """Verify singleton accessor."""
    svc = get_voice_service()
    assert isinstance(svc, VoiceService)
