"""TTS (Text-to-Speech) routes for the Stockpile API."""

import logging
import uuid
from dataclasses import asdict
from pathlib import Path

from api.dependencies import get_tts_service, get_voice_library
from api.schemas import PublicTTSRequest, TTSEndpointRequest
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from services.tts_service import TTSServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Text-to-Speech"])


@router.get("/api/tts/status", summary="TTS server status", description="Check connection status for both Colab and RunPod TTS backends.")
async def get_tts_status() -> dict:
    """Check TTS server connection status for both modes.

    Returns:
        Connection status including:
        - colab: status of custom Colab server connection
        - runpod: status of RunPod serverless endpoint
    """
    service = get_tts_service()

    # Get both statuses
    colab_status = await service.check_health()
    runpod_status = await service.check_runpod_health()

    return {
        "colab": colab_status,
        "runpod": runpod_status,
    }


@router.post("/api/tts/endpoint", summary="Set TTS server URL", description="Set or update the custom Colab TTS server URL.")
async def set_tts_endpoint(request: TTSEndpointRequest) -> dict:
    """Set or update the TTS server URL.

    Args:
        request: Request body containing the server URL

    Returns:
        Updated connection status
    """
    service = get_tts_service()
    service.set_server_url(request.url)

    # Check if we can connect to the new endpoint
    status = await service.check_health()

    if status.get("connected"):
        return {"message": "TTS server connected successfully", **status}
    else:
        return {"message": "URL saved but server not reachable", **status}


@router.post("/api/tts/generate", summary="Generate TTS audio", description="Generate speech audio from text using RunPod or Colab backend. Returns MP3 audio.", responses={400: {"description": "Invalid mode or missing config"}, 500: {"description": "TTS generation failed"}})
async def generate_tts(
    text: str = Form(...),
    mode: str = Form("runpod"),  # "runpod" or "colab"
    voice: UploadFile | None = File(None),
    voice_id: str | None = Form(None),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
) -> Response:
    """Generate TTS audio from text.

    Args:
        text: Text to convert to speech
        mode: TTS mode - "runpod" for RunPod Serverless, "colab" for custom Colab server
        voice: Optional voice reference audio file (5-10 seconds recommended)
        voice_id: Optional voice library ID (takes priority over uploaded voice file)
        exaggeration: Voice exaggeration level (0.0-1.0)
        cfg_weight: CFG weight for generation (0.0-1.0)
        temperature: Generation temperature (0.0-1.0)

    Returns:
        Audio file response (MP3)
    """
    service = get_tts_service()

    # Validate mode
    if mode not in ("runpod", "colab"):
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'runpod' or 'colab'.",
        )

    # Check appropriate service is configured
    if mode == "colab" and not service.server_url:
        raise HTTPException(
            status_code=400,
            detail="TTS server not configured. Set the server URL first.",
        )
    elif mode == "runpod" and not service.is_runpod_configured():
        raise HTTPException(
            status_code=400,
            detail="RunPod not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env",
        )

    # Validate text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    # Handle voice reference - voice_id takes priority over uploaded file
    voice_path: str | None = None
    is_library_voice = False

    if voice_id:
        # Use voice from library
        library = get_voice_library()
        library_voice = library.get_voice(voice_id)
        if not library_voice:
            raise HTTPException(
                status_code=404,
                detail=f"Voice {voice_id} not found",
            )
        # Presets without audio = no voice cloning (default voice)
        audio_path = library.get_audio_path(voice_id)
        if audio_path:
            voice_path = str(audio_path)
            is_library_voice = True
    elif voice and voice.filename:
        # Save uploaded voice file temporarily
        upload_dir = Path("uploads/tts_voices")
        upload_dir.mkdir(parents=True, exist_ok=True)
        voice_path = str(upload_dir / f"{uuid.uuid4()}_{voice.filename}")

        with open(voice_path, "wb") as f:
            content = await voice.read()
            f.write(content)

        logger.info(f"Saved voice reference: {voice_path}")

    try:
        # Generate based on mode
        if mode == "runpod":
            audio_bytes = await service.generate_runpod(
                text=text.strip(),
                voice_ref_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
        else:  # colab
            audio_bytes = await service.generate(
                text=text.strip(),
                voice_ref_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.mp3",
            },
        )

    except TTSServiceError as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up voice file only if it was a temp upload (not a library file)
        if voice_path and not is_library_voice and Path(voice_path).exists():
            try:
                Path(voice_path).unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up voice file: {cleanup_error}")


@router.get("/api/tts/public/status", summary="Public TTS status", description="Check if RunPod public Chatterbox Turbo endpoint is configured.")
async def get_public_tts_status() -> dict:
    """Check if public TTS endpoint is configured.

    Returns:
        Status dict with 'configured' and 'available'.
    """
    service = get_tts_service()
    return await service.check_public_health()


@router.post("/api/tts/public/generate", summary="Generate public TTS", description="Generate speech using RunPod's public Chatterbox Turbo endpoint. No voice cloning.", responses={400: {"description": "Missing config or empty text"}, 500: {"description": "Generation failed"}})
async def generate_public_tts(request: PublicTTSRequest) -> dict:
    """Generate TTS using RunPod's public Chatterbox Turbo endpoint.

    This uses the pre-deployed public endpoint - no custom Docker image needed.
    Note: Does NOT support voice cloning.

    Args:
        request: TTS parameters

    Returns:
        Dict with audio_url and cost
    """
    service = get_tts_service()

    # Check service is configured
    if not service.is_public_endpoint_configured():
        raise HTTPException(
            status_code=400,
            detail="RunPod not configured. Set RUNPOD_API_KEY in .env",
        )

    # Validate text
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        audio_url, cost = await service.generate_public(
            text=request.text.strip(),
            voice=request.voice,
        )

        return {
            "audio_url": audio_url,
            "cost": cost,
            "voice": request.voice,
        }

    except TTSServiceError as e:
        logger.error(f"Public TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Voice Library Endpoints
# =============================================================================


@router.get("/api/tts/voices", summary="List voices", description="List all voices in the voice library (presets + custom).")
async def list_voices() -> list[dict]:
    """List all voices (presets + custom)."""
    library = get_voice_library()
    voices = library.list_voices()
    return [asdict(v) for v in voices]


@router.post("/api/tts/voices", summary="Save voice", description="Upload and save a new custom voice reference for TTS cloning.")
async def save_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    """Upload and save a new voice."""
    library = get_voice_library()
    audio_bytes = await audio.read()
    voice = library.save_voice(name, audio_bytes, audio.filename or "voice.wav")
    return asdict(voice)


@router.delete("/api/tts/voices/{voice_id}", summary="Delete voice", description="Delete a custom voice from the library. Presets cannot be deleted.")
async def delete_voice(voice_id: str) -> dict:
    """Delete a custom voice."""
    library = get_voice_library()
    deleted = library.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice not found or is a preset")
    return {"message": "Voice deleted", "voice_id": voice_id}


@router.get("/api/tts/voices/{voice_id}/audio", summary="Get voice audio", description="Stream the audio file for a voice reference.")
async def get_voice_audio(voice_id: str):
    """Stream voice preview audio."""
    library = get_voice_library()
    audio_path = library.get_audio_path(voice_id)
    if not audio_path:
        raise HTTPException(status_code=404, detail="Voice audio not found")

    # Determine content type from extension
    ext = audio_path.suffix.lower()
    content_type = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
    }.get(ext, "audio/wav")

    return FileResponse(audio_path, media_type=content_type)
