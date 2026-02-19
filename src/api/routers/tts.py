"""TTS (Text-to-Speech) routes for the Stockpile API."""

import asyncio
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from api.dependencies import get_ai_service, get_tts_service, get_voice_library
from api.schemas import PublicTTSRequest, TTSEndpointRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from services.tts_service import TTSServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Text-to-Speech"])

# TTS job storage (in-memory)
tts_jobs: dict[str, dict] = {}

# WebSocket manager for TTS job updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()

# Jobs directory
JOBS_DIR = Path.home() / ".stockpile" / "jobs"


async def notify_tts_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for a TTS job."""
    await ws_manager.broadcast(job_id, message)


async def run_tts_generation(
    job_id: str,
    text: str,
    mode: str,
    voice_path: str | None,
    is_library_voice: bool,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    language: str,
    speaker_name: str | None,
    instruction: str | None,
    top_p: float,
    voice_reference_transcript: str | None,
    num_candidates: int,
    denoising_enabled: bool,
    whisper_enabled: bool,
    num_speakers: int = 1,
    moss_ttsd_mode: str = "generation",
    moss_ttsd_max_tokens: int = 2000,
) -> None:
    """Run TTS generation in the background.

    Args:
        job_id: Job ID
        text: Text to convert to speech
        mode: TTS backend mode
        voice_path: Path to voice reference file (or None)
        is_library_voice: Whether voice_path points to a library voice (don't delete)
        exaggeration: Voice exaggeration parameter
        cfg_weight: CFG weight parameter
        temperature: Generation temperature
        language: Language code for Qwen3
        speaker_name: Qwen3 preset speaker name
        instruction: Qwen3 style instruction
        top_p: Qwen3 nucleus sampling top-p
        voice_reference_transcript: Transcript of voice reference (Qwen3)
        num_candidates: Chatterbox Extended candidate count
        denoising_enabled: Chatterbox Extended denoising flag
        whisper_enabled: Chatterbox Extended Whisper validation flag
        num_speakers: MOSS-TTSD number of speakers (1-5)
        moss_ttsd_mode: MOSS-TTSD inference mode
        moss_ttsd_max_tokens: MOSS-TTSD max tokens to generate
    """
    if job_id not in tts_jobs:
        logger.error(f"TTS job {job_id} not found")
        return

    job = tts_jobs[job_id]
    service = get_tts_service()

    try:
        # Notify processing start
        await notify_tts_clients(job_id, {
            "type": "status",
            "status": "processing",
            "job_id": job_id,
        })

        # Generate audio based on mode
        audio_bytes: bytes
        if mode == "runpod":
            if service.is_runpod_configured():
                try:
                    audio_bytes = await service.generate_runpod(
                        text=text,
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
                    audio_bytes, cost = await service.generate_public_audio(text=text)
                    logger.info(f"Public TTS fallback cost: ${cost:.4f}")
            else:
                audio_bytes, cost = await service.generate_public_audio(text=text)
                logger.info(f"Public TTS cost: ${cost:.4f}")

        elif mode == "qwen3":
            audio_bytes = await service.generate_qwen3(
                text=text,
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
                text=text,
                voice_ref_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                num_candidates=num_candidates,
                enable_denoising=denoising_enabled,
                enable_whisper_validation=whisper_enabled,
            )

        elif mode == "moss-ttsd":
            # For single voice reference, map to S1
            voice_ref_paths = None
            if voice_path:
                voice_ref_paths = {"S1": voice_path}
            audio_bytes = await service.generate_moss_ttsd(
                text=text,
                voice_ref_paths=voice_ref_paths,
                language=language,
                temperature=temperature,
                max_tokens=moss_ttsd_max_tokens,
                inference_mode=moss_ttsd_mode,
                num_speakers=num_speakers,
            )

        else:  # colab
            audio_bytes = await service.generate(
                text=text,
                voice_ref_path=voice_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        # Detect format and save to jobs directory
        audio_format = service.detect_audio_format(audio_bytes)
        extension = service.file_extension_for_audio_format(audio_format)
        media_type = service.media_type_for_audio_format(audio_format)

        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = JOBS_DIR / f"{job_id}.{extension}"
        audio_path.write_bytes(audio_bytes)

        # Update job to completed
        job["status"] = "completed"
        job["audio_path"] = str(audio_path)
        job["audio_format"] = extension
        job["media_type"] = media_type
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"TTS job {job_id} completed: {audio_path}")

        await notify_tts_clients(job_id, {
            "type": "complete",
            "status": "completed",
            "job_id": job_id,
            "audio_format": extension,
        })

    except Exception as e:
        logger.error(f"TTS job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)

        await notify_tts_clients(job_id, {
            "type": "error",
            "status": "failed",
            "job_id": job_id,
            "message": str(e),
        })

    finally:
        # Clean up uploaded (non-library) voice file
        if voice_path and not is_library_voice and Path(voice_path).exists():
            try:
                Path(voice_path).unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up voice file: {cleanup_error}")


# =============================================================================
# Status & Config Endpoints (unchanged)
# =============================================================================


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
    moss_ttsd_status = await service.check_moss_ttsd_health()

    return {
        "colab": colab_status,
        "runpod": runpod_status,
        "qwen3": qwen3_status,
        "chatterbox_ext": chatterbox_ext_status,
        "moss_ttsd": moss_ttsd_status,
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


# =============================================================================
# Async TTS Generation (refactored to background job)
# =============================================================================


@router.post("/api/tts/generate", summary="Generate TTS audio", description="Start async TTS generation. Returns job ID immediately; poll or use WebSocket for status.", responses={202: {"description": "Job accepted"}, 400: {"description": "Invalid mode or missing config"}})
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
    enable_denoising: str = Form("false"),
    enable_whisper_validation: str = Form("false"),
    # MOSS-TTSD-specific params
    num_speakers: int = Form(1),
    moss_ttsd_mode: str = Form("generation"),
    moss_ttsd_max_tokens: int = Form(2000),
):
    """Start async TTS generation. Returns 202 with job_id immediately.

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
        202 response with job_id and status
    """
    service = get_tts_service()

    # Parse boolean form fields safely (bool("false") == True in Python!)
    denoising_enabled = enable_denoising.lower() in ("true", "1", "yes", "on")
    whisper_enabled = enable_whisper_validation.lower() in ("true", "1", "yes", "on")

    logger.info(
        f"TTS generate request: mode={mode}, exaggeration={exaggeration}, "
        f"cfg_weight={cfg_weight}, temperature={temperature}, "
        f"num_candidates={num_candidates}, denoising={denoising_enabled}, "
        f"whisper={whisper_enabled}, voice_id={voice_id}, "
        f"language={language}, speaker={speaker_name}, top_p={top_p}, "
        f"num_speakers={num_speakers}, moss_ttsd_mode={moss_ttsd_mode}, "
        f"moss_ttsd_max_tokens={moss_ttsd_max_tokens}"
    )

    valid_modes = ("runpod", "qwen3", "chatterbox-ext", "moss-ttsd", "colab")
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
    elif mode == "moss-ttsd" and not service.is_moss_ttsd_configured():
        raise HTTPException(
            status_code=400,
            detail="MOSS-TTSD not configured. Set RUNPOD_API_KEY and RUNPOD_MOSS_TTSD_ENDPOINT_ID in .env",
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
        # For uploaded voice files, save to a temp location that the background task can access
        upload_dir = Path("uploads/tts_voices")
        upload_dir.mkdir(parents=True, exist_ok=True)
        voice_path = str(upload_dir / f"{uuid.uuid4()}_{voice.filename}")

        with open(voice_path, "wb") as f:
            content = await voice.read()
            f.write(content)

        logger.info(f"Saved voice reference: {voice_path}")

    # Runpod mode: additional validation for voice cloning without custom endpoint
    if mode == "runpod" and not service.is_runpod_configured() and voice_path:
        # Clean up uploaded file before raising
        if voice_path and not is_library_voice and Path(voice_path).exists():
            try:
                Path(voice_path).unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=400,
            detail=(
                "Voice cloning requires RUNPOD_ENDPOINT_ID. "
                "Configure custom RunPod endpoint in .env."
            ),
        )

    # Create job
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "processing",
        "mode": mode,
        "text_preview": text.strip()[:50],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_path": None,
        "audio_format": None,
        "media_type": None,
        "completed_at": None,
        "error": None,
    }
    tts_jobs[job_id] = job

    # Start generation in background
    task = asyncio.create_task(
        run_tts_generation(
            job_id=job_id,
            text=text.strip(),
            mode=mode,
            voice_path=voice_path,
            is_library_voice=is_library_voice,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            language=language,
            speaker_name=speaker_name,
            instruction=instruction,
            top_p=top_p,
            voice_reference_transcript=voice_reference_transcript,
            num_candidates=num_candidates,
            denoising_enabled=denoising_enabled,
            whisper_enabled=whisper_enabled,
            num_speakers=num_speakers,
            moss_ttsd_mode=moss_ttsd_mode,
            moss_ttsd_max_tokens=moss_ttsd_max_tokens,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "processing"},
    )


# =============================================================================
# TTS Job Management Endpoints
# =============================================================================


@router.get("/api/tts/jobs", summary="List TTS jobs", description="List all recent TTS jobs.")
async def list_tts_jobs() -> list[dict]:
    """List all TTS jobs.

    Returns:
        List of job dicts (without audio_path for security).
    """
    jobs = []
    for job in tts_jobs.values():
        job_info = {k: v for k, v in job.items() if k != "audio_path"}
        jobs.append(job_info)
    return jobs


@router.get("/api/tts/jobs/{job_id}", summary="Get TTS job status", responses={404: {"description": "Job not found"}})
async def get_tts_job(job_id: str) -> dict:
    """Get a single TTS job status.

    Args:
        job_id: Job ID

    Returns:
        Job status dict.
    """
    if job_id not in tts_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = tts_jobs[job_id]
    return {k: v for k, v in job.items() if k != "audio_path"}


@router.get("/api/tts/jobs/{job_id}/audio", summary="Download TTS audio", description="Download the generated audio file for a completed TTS job.", responses={404: {"description": "Job or audio not found"}, 400: {"description": "Job not completed"}})
async def get_tts_job_audio(job_id: str):
    """Download the generated audio for a completed TTS job.

    Args:
        job_id: Job ID

    Returns:
        Audio file response.
    """
    if job_id not in tts_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = tts_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job['status']})",
        )

    audio_path = job.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    media_type = job.get("media_type", "audio/wav")
    extension = job.get("audio_format", "wav")

    return FileResponse(
        audio_path,
        media_type=media_type,
        filename=f"tts_{job_id}.{extension}",
    )


@router.delete("/api/tts/jobs/{job_id}", summary="Delete TTS job", description="Delete a TTS job and its audio file.", responses={404: {"description": "Job not found"}})
async def delete_tts_job(job_id: str) -> dict:
    """Delete a TTS job and clean up its audio file.

    Args:
        job_id: Job ID

    Returns:
        Confirmation message.
    """
    if job_id not in tts_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = tts_jobs[job_id]

    # Clean up audio file if it exists
    audio_path = job.get("audio_path")
    if audio_path and Path(audio_path).exists():
        try:
            Path(audio_path).unlink()
        except Exception as e:
            logger.warning(f"Failed to delete audio file for job {job_id}: {e}")

    del tts_jobs[job_id]
    return {"message": "Job deleted", "job_id": job_id}


# =============================================================================
# TTS WebSocket
# =============================================================================


@router.websocket("/ws/tts/{job_id}")
async def websocket_tts(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time TTS job updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    if job_id not in tts_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = tts_jobs[job_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "status": job["status"],
            "job_id": job_id,
            "error": job.get("error"),
        })

        # If job is already completed, send completion message
        if job["status"] == "completed":
            await websocket.send_json({
                "type": "complete",
                "status": "completed",
                "job_id": job_id,
                "audio_format": job.get("audio_format"),
            })

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for TTS job {job_id}: {e}")
    finally:
        ws_manager.disconnect(job_id, websocket)


# =============================================================================
# Public TTS Endpoints (unchanged)
# =============================================================================


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
# Voice Library Endpoints (unchanged)
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


@router.delete("/api/tts/voices/{voice_id}", summary="Delete voice", description="Delete a voice from the library.")
async def delete_voice(voice_id: str) -> dict:
    """Delete a voice."""
    library = get_voice_library()
    deleted = library.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"message": "Voice deleted", "voice_id": voice_id}


@router.patch("/api/tts/voices/{voice_id}/favorite", summary="Toggle favorite", description="Toggle the favorite status of a voice.")
async def toggle_voice_favorite(voice_id: str) -> dict:
    """Toggle favorite status for a voice."""
    library = get_voice_library()
    result = library.toggle_favorite(voice_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Voice not found")
    return {"voice_id": voice_id, "is_favorite": result}


@router.get("/api/tts/voices/{voice_id}/audio", summary="Get voice audio", description="Stream the audio file for a voice reference.")
async def get_voice_audio(voice_id: str):
    """Stream voice preview audio."""
    library = get_voice_library()

    # Try to get raw bytes first (works for both DB and filesystem backends)
    audio_bytes, audio_format = library.get_audio_bytes(voice_id)
    if audio_bytes:
        content_type = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
        }.get(audio_format or ".wav", "audio/wav")
        return Response(content=audio_bytes, media_type=content_type)

    raise HTTPException(status_code=404, detail="Voice audio not found")


@router.post("/api/tts/voices/{voice_id}/transcribe", summary="Transcribe voice audio", description="Transcribe a voice reference audio using Gemini (free). Returns the text spoken in the audio.")
async def transcribe_voice(voice_id: str) -> dict:
    """Auto-transcribe a voice reference audio file using Gemini."""
    library = get_voice_library()
    audio_path = library.get_audio_path(voice_id)
    if not audio_path:
        raise HTTPException(status_code=404, detail="Voice audio not found")

    ai_service = get_ai_service()
    if not ai_service.api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    try:
        audio_bytes = audio_path.read_bytes()
        ext = audio_path.suffix.lower()
        mime_type = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
        }.get(ext, "audio/wav")

        from google.genai import types

        response = ai_service.client.models.generate_content(
            model=ai_service.model_name,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        types.Part.from_text(text=
                            "Transcribe exactly what is said in this audio. "
                            "Return ONLY the transcript text, nothing else. "
                            "No quotes, no labels, no explanations."
                        ),
                    ]
                )
            ],
        )
        transcript = response.text.strip()
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Transcription failed for voice {voice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
