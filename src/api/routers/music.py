"""Music generation routes for the Stockpile API."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.dependencies import get_music_service
from api.schemas import MusicGenerateRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Music Generation"])

# Music job storage (in-memory)
music_jobs: dict[str, dict] = {}

# WebSocket manager for music job updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()

# Jobs directory for saving audio files
JOBS_DIR = Path.home() / ".stockpile" / "jobs"


async def _notify_music_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for a music job."""
    await ws_manager.broadcast(job_id, message)


async def _run_music_generate(job_id: str, genres: str, output_seconds: int, seed: int | None, steps: int, cfg: float) -> None:
    """Run music generation in the background."""
    if job_id not in music_jobs:
        logger.error(f"Music job {job_id} not found")
        return

    job = music_jobs[job_id]

    try:
        service = get_music_service()
        audio_bytes = await service.generate_music(
            genres=genres,
            output_seconds=output_seconds,
            seed=seed,
            steps=steps,
            cfg=cfg,
        )

        # Save audio to disk
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = JOBS_DIR / f"{job_id}.mp3"
        audio_path.write_bytes(audio_bytes)

        job["status"] = "completed"
        job["audio_path"] = str(audio_path)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_music_clients(job_id, {
            "type": "complete",
            "job_id": job_id,
            "status": "completed",
        })

    except Exception as e:
        logger.error(f"Music generation failed for job {job_id}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_music_clients(job_id, {
            "type": "error",
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        })


@router.get("/api/music/status", summary="Music generation status", description="Check Stable Audio 2.5 availability via Replicate.")
async def get_music_status() -> dict:
    """Check music generation service status."""
    service = get_music_service()
    return await service.check_health()


@router.post("/api/music/generate", summary="Generate music", description="Generate instrumental music. Returns 202 with job_id for async processing.", responses={202: {"description": "Job accepted"}, 400: {"description": "Invalid parameters or missing config"}}, status_code=202)
async def generate_music(request: MusicGenerateRequest) -> dict:
    """Generate instrumental music - returns 202 with job_id."""
    service = get_music_service()

    if not service.replicate_api_key:
        raise HTTPException(
            status_code=400,
            detail="REPLICATE_API_KEY not configured in .env",
        )

    if not request.genres or not request.genres.strip():
        raise HTTPException(status_code=400, detail="Genres description is required")

    logger.info(
        f"Music generate request: genres={request.genres}, "
        f"duration={request.output_seconds}s, steps={request.steps}, cfg={request.cfg}"
    )

    # Create job
    job_id = uuid.uuid4().hex[:12]
    music_jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "genres_preview": request.genres.strip()[:80],
        "duration": request.output_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "audio_path": None,
        "error": None,
    }

    # Ensure WS key exists before task starts
    ws_manager.ensure_key(job_id)

    # Start background task
    task = asyncio.create_task(_run_music_generate(
        job_id,
        genres=request.genres.strip(),
        output_seconds=request.output_seconds,
        seed=request.seed,
        steps=request.steps,
        cfg=request.cfg,
    ))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"job_id": job_id, "status": "processing"}


# =============================================================================
# Music Job Management Endpoints
# =============================================================================


@router.get("/api/music/jobs", summary="List music jobs", description="List all music generation jobs.")
async def list_music_jobs() -> list[dict]:
    """List all music jobs."""
    return list(music_jobs.values())


@router.get("/api/music/jobs/{job_id}", summary="Get music job", description="Get status of a specific music job.", responses={404: {"description": "Job not found"}})
async def get_music_job(job_id: str) -> dict:
    """Get music job status."""
    if job_id not in music_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return music_jobs[job_id]


@router.get("/api/music/jobs/{job_id}/audio", summary="Get music audio", description="Download the generated audio file.", responses={404: {"description": "Job not found or audio not ready"}, 400: {"description": "Job not completed"}})
async def get_music_job_audio(job_id: str) -> FileResponse:
    """Download the generated audio file for a completed music job."""
    if job_id not in music_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = music_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")

    audio_path = job.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="music_{job_id}.mp3"'},
    )


@router.delete("/api/music/jobs/{job_id}", summary="Delete music job", description="Remove a music job and its audio file.", responses={404: {"description": "Job not found"}})
async def delete_music_job(job_id: str) -> dict:
    """Delete a music job and clean up its audio file."""
    if job_id not in music_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = music_jobs[job_id]

    # Clean up audio file if it exists
    audio_path = job.get("audio_path")
    if audio_path:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up audio file for job {job_id}: {e}")

    del music_jobs[job_id]
    ws_manager.cleanup(job_id)
    return {"message": "Job deleted", "job_id": job_id}


# =============================================================================
# Music Job WebSocket
# =============================================================================


@router.websocket("/ws/music/{job_id}")
async def websocket_music_job(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time music job updates."""
    if job_id not in music_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = music_jobs[job_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "job_id": job_id,
            "status": job["status"],
            "error": job.get("error"),
        })

        # If already completed or failed, send final state
        if job["status"] in ("completed", "failed"):
            msg_type = "complete" if job["status"] == "completed" else "error"
            await websocket.send_json({
                "type": msg_type,
                "job_id": job_id,
                "status": job["status"],
                "error": job.get("error"),
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
        logger.error(f"WebSocket error for music job {job_id}: {e}")
    finally:
        ws_manager.disconnect(job_id, websocket)
