import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from app.core.exceptions import ProviderError, ZicoError, sanitize_error_message
from app.services.voice import get_voice_service

logger = logging.getLogger(__name__)

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
    logger.info("voice_upload_started")
    filename = file.filename or "audio.wav"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
        logger.error("voice_upload_failed: unsupported audio format")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Supported formats: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    try:
        content = await file.read()
        if not content or len(content) == 0:
            logger.error("voice_upload_failed: empty payload")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio content cannot be empty.",
            )
        if len(content) > MAX_AUDIO_SIZE_BYTES:
            logger.error("voice_upload_failed: payload too large")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio payload exceeds maximum permitted size of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)}MB.",
            )

        logger.info("voice_upload_completed")
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
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except ProviderError as prov_err:
        status_code = (
            status.HTTP_429_TOO_MANY_REQUESTS
            if "quota" in prov_err.message.lower()
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        raise HTTPException(
            status_code=status_code,
            detail=prov_err.safe_message,
        )
    except ZicoError as zico_err:
        raise HTTPException(
            status_code=zico_err.http_status_code,
            detail=zico_err.safe_message,
        )
    except Exception as exc:
        err_str = str(exc).lower()
        if any(
            k in err_str
            for k in ["quota", "429", "insufficient_quota", "credit_balance_exhausted"]
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Voice transcription is temporarily unavailable because the AI service quota is exhausted. Please type your query or try again later.",
            )
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
