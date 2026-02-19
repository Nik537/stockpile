"""Video generation routes for the Stockpile API."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from api.dependencies import get_video_gen_service
from api.routers.storyboard import storyboard_jobs
from api.schemas import VideoGenerateRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Video Generation"])

# Video job storage (in-memory)
video_jobs: dict[str, dict] = {}

# WebSocket manager for video job updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()


async def _notify_video_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for a video job."""
    await ws_manager.broadcast(job_id, message)


async def _run_video_generate(job_id: str, **params) -> None:
    """Run video generation in the background."""
    if job_id not in video_jobs:
        logger.error(f"Video job {job_id} not found")
        return

    job = video_jobs[job_id]

    try:
        service = get_video_gen_service()
        result = await service.generate_video(**params)

        job["status"] = "completed"
        job["video_url"] = result["video_url"]
        job["seed"] = result.get("seed")
        job["generation_time_ms"] = result.get("generation_time_ms")
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_video_clients(job_id, {
            "type": "complete",
            "job_id": job_id,
            "status": "completed",
            "result": {
                "video_url": result["video_url"],
                "seed": result.get("seed"),
                "generation_time_ms": result.get("generation_time_ms"),
            },
        })

    except Exception as e:
        logger.error(f"Video generation failed for job {job_id}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_video_clients(job_id, {
            "type": "error",
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        })


@router.get("/api/video/status", summary="Video generation status")
async def get_video_status() -> dict:
    """Check LTX-Video 2 service status."""
    service = get_video_gen_service()
    return await service.check_health()


@router.post(
    "/api/video/generate",
    summary="Generate video",
    status_code=202,
)
async def generate_video(request: VideoGenerateRequest) -> dict:
    """Generate a video with LTX-Video 2. Returns 202 with job_id."""
    service = get_video_gen_service()

    if not service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="RUNPOD_API_KEY and RUNPOD_LTX_VIDEO_ENDPOINT_ID required in .env",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    logger.info(
        f"Video generate request: {request.width}x{request.height}, "
        f"{request.num_frames} frames, {len(request.conditioning_images or [])} ref images"
    )

    job_id = uuid.uuid4().hex[:12]
    video_jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "prompt_preview": request.prompt.strip()[:80],
        "width": request.width,
        "height": request.height,
        "num_frames": request.num_frames,
        "num_conditioning_images": len(request.conditioning_images or []),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "video_url": None,
        "seed": None,
        "generation_time_ms": None,
        "error": None,
    }

    ws_manager.ensure_key(job_id)

    task = asyncio.create_task(_run_video_generate(
        job_id,
        prompt=request.prompt.strip(),
        negative_prompt=request.negative_prompt.strip() if request.negative_prompt else "",
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        fps=request.fps,
        conditioning_images=request.conditioning_images,
        conditioning_strength=request.conditioning_strength,
    ))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"job_id": job_id, "status": "processing"}


@router.get("/api/video/jobs", summary="List video jobs")
async def list_video_jobs() -> list[dict]:
    """List all video generation jobs."""
    return list(video_jobs.values())


@router.get("/api/video/jobs/{job_id}", summary="Get video job")
async def get_video_job(job_id: str) -> dict:
    """Get video job status."""
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return video_jobs[job_id]


@router.delete("/api/video/jobs/{job_id}", summary="Delete video job")
async def delete_video_job(job_id: str) -> dict:
    """Delete a video job."""
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del video_jobs[job_id]
    ws_manager.cleanup(job_id)
    return {"message": "Job deleted", "job_id": job_id}


# =============================================================================
# Storyboard images for reference
# =============================================================================


@router.get("/api/video/storyboard-images", summary="Get available storyboard images")
async def get_storyboard_images() -> list[dict]:
    """Return completed storyboard jobs with their scene images for use as video references."""
    results = []
    for job_id, job in storyboard_jobs.items():
        if job.status not in ("completed",):
            continue
        scenes = []
        for scene in job.scene_images:
            if scene.get("status") == "completed" and scene.get("image_url"):
                scenes.append({
                    "scene_number": scene["scene_number"],
                    "image_url": scene["image_url"],
                })
        refs = {}
        for name, url in job.reference_images.items():
            if url:
                refs[name] = url
        if scenes or refs:
            results.append({
                "job_id": job_id,
                "title": job.plan.title if job.plan else "Untitled",
                "scenes": scenes,
                "reference_images": refs,
            })
    return results


# =============================================================================
# Video Job WebSocket
# =============================================================================


@router.websocket("/ws/video/{job_id}")
async def websocket_video_job(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time video job updates."""
    if job_id not in video_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = video_jobs[job_id]

        await websocket.send_json({
            "type": "status",
            "job_id": job_id,
            "status": job["status"],
            "error": job.get("error"),
        })

        if job["status"] in ("completed", "failed"):
            msg_type = "complete" if job["status"] == "completed" else "error"
            payload: dict = {
                "type": msg_type,
                "job_id": job_id,
                "status": job["status"],
            }
            if msg_type == "complete":
                payload["result"] = {
                    "video_url": job.get("video_url"),
                    "seed": job.get("seed"),
                }
            else:
                payload["error"] = job.get("error")
            await websocket.send_json(payload)

        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for video job {job_id}: {e}")
    finally:
        ws_manager.disconnect(job_id, websocket)
