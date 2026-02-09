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


@router.get("/api/tts/status", summary="TTS server status", description="Check connection status for all TTS backends.")
async def get_tts_status() -> dict:
    """Check TTS server connection status for all modes.

    Returns:
        Connection status for each backend:
        - colab: custom Colab server
        - runpod: original Chatterbox RunPod endpoint
        - qwen3: Qwen3-TTS RunPod endpoint
        - chatterbox_ext: Chatterbox Extended RunPod endpoint
    """
    service = get_tts_service()

    colab_status = await service.check_health()
    runpod_status = await service.check_runpod_health()
    qwen3_status = await service.check_qwen3_health()
    chatterbox_ext_status = await service.check_chatterbox_ext_health()

    return {
        "colab": colab_status,
        "runpod": runpod_status,
        "qwen3": qwen3_status,
        "chatterbox_ext": chatterbox_ext_status,
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


@router.post("/api/tts/generate", summary="Generate TTS audio", description="Generate speech audio from text using RunPod, Qwen3-TTS, Chatterbox Extended, or Colab backend.", responses={400: {"description": "Invalid mode or missing config"}, 500: {"description": "TTS generation failed"}})
async def generate_tts(
    text: str = Form(...),
    mode: str = Form("runpod"),  # "runpod" | "qwen3" | "chatterbox-ext" | "colab"
    voice: UploadFile | None = File(None),
    voice_id: str | None = Form(None),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
    # Qwen3-specific params
    language: str = Form("auto"),
    speaker_name: str | None = Form(None),
    instruction: str | None = Form(None),
    top_p: float = Form(0.9),
    voice_reference_transcript: str | None = Form(None),
    # Chatterbox Extended-specific params
    num_candidates: int = Form(1),
    enable_denoising: bool = Form(False),
    enable_whisper_validation: bool = Form(False),
) -> Response:
    """Generate TTS audio from text.

    Args:
        text: Text to convert to speech
        mode: TTS backend - "runpod", "qwen3", "chatterbox-ext", or "colab"
        voice: Optional voice reference audio file
        voice_id: Optional voice library ID (takes priority over uploaded voice file)
        exaggeration: Voice exaggeration (Chatterbox modes, 0.0-1.0)
        cfg_weight: CFG weight (Chatterbox modes, 0.0-1.0)
        temperature: Generation temperature (0.0-1.0)
        language: Language code for Qwen3 (default "auto")
        speaker_name: Qwen3 preset speaker name
        instruction: Qwen3 style instruction
        top_p: Qwen3 nucleus sampling top-p
        voice_reference_transcript: Transcript of voice reference (Qwen3)
        num_candidates: Chatterbox Extended candidate count
        enable_denoising: Chatterbox Extended denoising
        enable_whisper_validation: Chatterbox Extended Whisper validation

    Returns:
        Audio file response
    """
    service = get_tts_service()

    valid_modes = ("runpod", "qwen3", "chatterbox-ext", "colab")
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Use one of: {', '.join(valid_modes)}",
        )

    # Check appropriate service is configured
    if mode == "colab" and not service.server_url:
        raise HTTPException(
            status_code=400,
            detail="TTS server not configured. Set the server URL first.",
        )
    elif mode == "runpod" and not service.is_runpod_configured() and not service.is_public_endpoint_configured():
        raise HTTPException(
            status_code=400,
            detail="RunPod not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env",
        )
    elif mode == "qwen3" and not service.is_qwen3_configured():
        raise HTTPException(
            status_code=400,
            detail="Qwen3-TTS not configured. Set RUNPOD_API_KEY and RUNPOD_QWEN3_ENDPOINT_ID in .env",
        )
    elif mode == "chatterbox-ext" and not service.is_chatterbox_ext_configured():
        raise HTTPException(
            status_code=400,
            detail="Chatterbox Extended not configured. Set RUNPOD_API_KEY and RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID in .env",
        )

    # Validate text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    # Handle voice reference - voice_id takes priority over uploaded file
    voice_path: str | None = None
    is_library_voice = False

    if voice_id:
        library = get_voice_library()
        library_voice = library.get_voice(voice_id)
        if not library_voice:
            raise HTTPException(
                status_code=404,
                detail=f"Voice {voice_id} not found",
            )
        audio_path = library.get_audio_path(voice_id)
        if audio_path:
            voice_path = str(audio_path)
            is_library_voice = True
    elif voice and voice.filename:
        upload_dir = Path("uploads/tts_voices")
        upload_dir.mkdir(parents=True, exist_ok=True)
        voice_path = str(upload_dir / f"{uuid.uuid4()}_{voice.filename}")

        with open(voice_path, "wb") as f:
            content = await voice.read()
            f.write(content)

        logger.info(f"Saved voice reference: {voice_path}")

    try:
        if mode == "runpod":
            if service.is_runpod_configured():
                try:
                    audio_bytes = await service.generate_runpod(
                        text=text.strip(),
                        voice_ref_path=voice_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                except TTSServiceError as runpod_error:
                    should_fallback = (
                        not voice_path
                        and service.is_public_endpoint_configured()
                        and service._is_timeout_error(runpod_error)
                    )
                    if not should_fallback:
                        raise
                    logger.warning(
                        "Custom RunPod TTS timed out. Falling back to public endpoint."
                    )
                    audio_bytes, cost = await service.generate_public_audio(text=text.strip())
                    logger.info(f"Public TTS fallback cost: ${cost:.4f}")
            else:
                if voice_path:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Voice cloning requires RUNPOD_ENDPOINT_ID. "
                            "Configure custom RunPod endpoint in .env."
                        ),
                    )
                audio_bytes, cost = await service.generate_public_audio(text=text.strip())
                logger.info(f"Public TTS cost: ${cost:.4f}")

        elif mode == "qwen3":
            audio_bytes = await service.generate_qwen3(
                text=text.strip(),
                voice_ref_path=voice_path,
                voice_reference_transcript=voice_reference_transcript,
                speaker_name=speaker_name,
                instruction=instruction,
                language=language,
                temperature=temperature,
                top_p=top_p,
            )

        elif mode == "chatterbox-ext":
            audio_bytes = await service.generate_chatterbox_extended(
                text=text.strip(),
                voice_ref_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                num_candidates=num_candidates,
                enable_denoising=enable_denoising,
                enable_whisper_validation=enable_whisper_validation,
            )

        else:  # colab
            audio_bytes = await service.generate(
                text=text.strip(),
                voice_ref_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        audio_format = service.detect_audio_format(audio_bytes)
        media_type = service.media_type_for_audio_format(audio_format)
        extension = service.file_extension_for_audio_format(audio_format)
        filename = f"tts_output.{extension}"

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    except TTSServiceError as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
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
