from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.core.exceptions import sanitize_error_message
from app.services.voice import get_voice_service

router = APIRouter()

MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max limit
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac"}


class SynthesizeRequest(BaseModel):
    text: str = Field(description="Text content to convert into speech")
    voice: Optional[str] = Field(
        default="en-US-JennyNeural", description="Voice profile identifier"
    )


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
    Enforces format validation and file size limits.
    """
    filename = file.filename or "audio.wav"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Supported formats: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    try:
        content = await file.read()
        if len(content) > MAX_AUDIO_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio payload exceeds maximum permitted size of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)}MB.",
            )

        voice_service = get_voice_service()
        transcript = await voice_service.transcribe_audio(
            audio_bytes=content,
            filename=filename,
            language=language,
        )
        return TranscribeResponse(
            transcript=transcript,
            filename=filename,
        )
    except HTTPException:
        raise
    except Exception as exc:
        safe_msg = sanitize_error_message(str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio transcription failed: {safe_msg}",
        )


@router.post("/synthesize")
async def synthesize_speech_endpoint(payload: SynthesizeRequest) -> Response:
    """
    Synthesizes text into speech audio and streams back audio content.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text payload cannot be empty.",
        )

    try:
        voice_service = get_voice_service()
        audio_bytes = await voice_service.synthesize_speech(
            text=payload.text.strip(),
            voice=payload.voice or "en-US-JennyNeural",
        )
        media_type = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"
        return Response(content=audio_bytes, media_type=media_type)
    except Exception as exc:
        safe_msg = sanitize_error_message(str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {safe_msg}",
        )
