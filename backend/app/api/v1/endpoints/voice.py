from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field
from app.services.voice import get_voice_service

router = APIRouter()


class SynthesizeRequest(BaseModel):
    text: str = Field(description="Text content to convert into speech")
    voice: Optional[str] = Field(default="en-US-JennyNeural", description="Voice profile identifier")


class TranscribeResponse(BaseModel):
    transcript: str
    filename: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
) -> TranscribeResponse:
    """
    Transcribes uploaded audio into text using OpenAI Whisper with audio processing.
    """
    try:
        content = await file.read()
        voice_service = get_voice_service()
        transcript = await voice_service.transcribe_audio(
            audio_bytes=content,
            filename=file.filename or "audio.wav",
            language=language,
        )
        return TranscribeResponse(
            transcript=transcript,
            filename=file.filename or "audio.wav",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio transcription failed: {str(exc)}",
        )


@router.post("/synthesize")
async def synthesize_speech_endpoint(payload: SynthesizeRequest) -> Response:
    """
    Synthesizes text into speech audio and streams back audio content.
    """
    try:
        voice_service = get_voice_service()
        audio_bytes = await voice_service.synthesize_speech(
            text=payload.text,
            voice=payload.voice or "en-US-JennyNeural",
        )
        # Return audio payload
        media_type = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
        return Response(content=audio_bytes, media_type=media_type)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speech synthesis failed: {str(exc)}",
        )
