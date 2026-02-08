"""TTS (Text-to-Speech) routes for the Stockpile API."""

import logging
import uuid
from pathlib import Path

from api.dependencies import get_tts_service
from api.schemas import PublicTTSRequest, TTSEndpointRequest
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
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
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
) -> Response:
    """Generate TTS audio from text.

    Args:
        text: Text to convert to speech
        mode: TTS mode - "runpod" for RunPod Serverless, "colab" for custom Colab server
        voice: Optional voice reference audio file (5-10 seconds recommended)
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

    # Handle voice reference file
    voice_path: str | None = None
    if voice and voice.filename:
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
        # Clean up voice file
        if voice_path and Path(voice_path).exists():
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
