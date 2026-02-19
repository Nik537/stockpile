"""Storyboard generation routes for the Stockpile API."""

import asyncio
import io
import json
import logging
import uuid
import zipfile

import httpx
from api.dependencies import get_ai_service, get_image_gen_service, get_storyboard_service
from api.schemas import StoryboardGenerateRequest, StoryboardPlanRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from models.storyboard import (
    CharacterProfile,
    StoryboardJob,
    StoryboardPlan,
    StoryboardScene,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Storyboard"])

# Storyboard job storage (in-memory)
storyboard_jobs: dict[str, StoryboardJob] = {}

# WebSocket manager for storyboard updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()


async def notify_storyboard_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for a storyboard job.

    Args:
        job_id: Job ID
        message: Message to send
    """
    await ws_manager.broadcast(job_id, message)


async def run_storyboard_generation(job_id: str) -> None:
    """Run storyboard image generation in the background.

    Args:
        job_id: Job ID
    """
    if job_id not in storyboard_jobs:
        logger.error(f"Storyboard job {job_id} not found")
        return

    job = storyboard_jobs[job_id]
    service = get_storyboard_service()

    try:
        job.status = "generating"

        await notify_storyboard_clients(job_id, {
            "type": "status",
            "status": "generating",
            "job_id": job_id,
        })

        async def on_progress(update):
            await notify_storyboard_clients(job_id, update)

        await service.run_storyboard_generation(job, on_progress=on_progress)

        # Determine final status
        failed_scenes = [s for s in job.scene_images if s["status"] == "failed"]
        if failed_scenes and len(failed_scenes) == len(job.scene_images):
            job.status = "failed"
            job.error = "All scene image generations failed"
        elif failed_scenes:
            job.status = "completed"
        else:
            job.status = "completed"

        await notify_storyboard_clients(job_id, {
            "type": "complete",
            "status": job.status,
            "job_id": job_id,
            "reference_images": job.reference_images,
            "scene_images": job.scene_images,
            "total_cost": job.total_cost,
        })

    except Exception as e:
        logger.error(f"Storyboard generation failed for job {job_id}: {e}")
        job.status = "failed"
        job.error = str(e)

        await notify_storyboard_clients(job_id, {
            "type": "error",
            "message": str(e),
        })


@router.post("/api/storyboard/plan", summary="Generate storyboard plan", description="Step 1: Generate a storyboard plan from a creative idea using AI.", responses={400: {"description": "Invalid idea"}, 500: {"description": "Plan generation failed"}})
async def generate_storyboard_plan(request: StoryboardPlanRequest) -> dict:
    """Generate a storyboard plan from a creative idea.

    Args:
        request: The storyboard plan request with idea, num_scenes, aspect_ratio.

    Returns:
        Job ID and generated plan for review.
    """
    if not request.idea or not request.idea.strip():
        raise HTTPException(status_code=400, detail="Idea is required")

    ai_service = get_ai_service()
    if not ai_service.api_key:
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY not configured. Set it in your .env file.",
        )

    service = get_storyboard_service()

    try:
        plan_data = service.generate_storyboard_plan(
            idea=request.idea.strip(),
            num_scenes=request.num_scenes,
            aspect_ratio=request.aspect_ratio,
        )

        job_id = str(uuid.uuid4())

        return {
            "job_id": job_id,
            "plan": plan_data,
        }

    except Exception as e:
        logger.error(f"Storyboard plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/storyboard/generate", summary="Start storyboard image generation", description="Step 2: Generate images from a storyboard plan. Track progress via WebSocket.", responses={400: {"description": "Invalid request"}, 404: {"description": "Job not found"}})
async def start_storyboard_generation(request: StoryboardGenerateRequest) -> dict:
    """Start generating storyboard images from a plan.

    Args:
        request: Job ID, plan, model settings, and dimensions.

    Returns:
        Job status with 202 accepted.
    """
    if not request.plan:
        raise HTTPException(status_code=400, detail="Plan is required")

    # Parse plan dict into StoryboardPlan
    try:
        characters = [
            CharacterProfile(
                name=c.get("name", ""),
                appearance=c.get("appearance", ""),
                clothing=c.get("clothing", ""),
                accessories=c.get("accessories", ""),
            )
            for c in request.plan.get("characters", [])
        ]
        scenes = [
            StoryboardScene(
                scene_number=s.get("scene_number", i + 1),
                description=s.get("description", ""),
                camera_angle=s.get("camera_angle", ""),
                character_action=s.get("character_action", ""),
                environment=s.get("environment", ""),
                image_prompt=s.get("image_prompt", ""),
            )
            for i, s in enumerate(request.plan.get("scenes", []))
        ]
        plan = StoryboardPlan(
            title=request.plan.get("title", "Untitled"),
            characters=characters,
            scenes=scenes,
            style_guide=request.plan.get("style_guide", ""),
            aspect_ratio=request.plan.get("aspect_ratio", "9:16"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid plan format: {e}")

    # Create or update job
    job_id = request.job_id
    if job_id in storyboard_jobs:
        job = storyboard_jobs[job_id]
        if job.status not in ("pending", "completed", "failed"):
            raise HTTPException(
                status_code=400,
                detail=f"Job is already {job.status}",
            )
        job.plan = plan
        job.status = "pending"
        job.scene_images = []
        job.reference_images = {}
        job.error = None
        job.total_cost = 0.0
    else:
        job = StoryboardJob(
            job_id=job_id,
            status="pending",
            plan=plan,
        )
        storyboard_jobs[job_id] = job

    # Store generation params on the job for the background task
    job._ref_model = request.ref_model  # type: ignore[attr-defined]
    job._scene_model = request.scene_model  # type: ignore[attr-defined]
    job._width = request.width  # type: ignore[attr-defined]
    job._height = request.height  # type: ignore[attr-defined]
    job._user_reference_images = request.user_reference_images or {}  # type: ignore[attr-defined]

    # Start generation in background
    task = asyncio.create_task(run_storyboard_generation(job_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "job_id": job_id,
        "status": "generating",
        "total_characters": len(plan.characters),
        "total_scenes": len(plan.scenes),
    }


@router.get("/api/storyboard/jobs/{job_id}", summary="Get storyboard job status", responses={404: {"description": "Job not found"}})
async def get_storyboard_job(job_id: str) -> dict:
    """Get storyboard job status and results.

    Args:
        job_id: Job ID

    Returns:
        Full job status with plan, reference images, and scene images.
    """
    if job_id not in storyboard_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return storyboard_jobs[job_id].to_dict()


@router.get("/api/storyboard/status", summary="Get storyboard service status")
async def get_storyboard_status() -> dict:
    """Return which image generation providers are configured.

    Returns:
        Provider availability status.
    """
    image_service = get_image_gen_service()
    ai_service = get_ai_service()

    return {
        "gemini": bool(ai_service.api_key),
        "runpod": image_service.is_runpod_configured(),
        "runware": image_service.is_runware_configured(),
    }


@router.websocket("/ws/storyboard/{job_id}")
async def websocket_storyboard(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time storyboard generation updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    if job_id not in storyboard_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = storyboard_jobs[job_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "status": job.status,
            "job_id": job_id,
            "reference_images": job.reference_images,
            "scene_images": job.scene_images,
            "error": job.error,
        })

        # If job is already completed, send results
        if job.status in ("completed", "failed"):
            await websocket.send_json({
                "type": "complete",
                "status": job.status,
                "job_id": job_id,
                "reference_images": job.reference_images,
                "scene_images": job.scene_images,
                "total_cost": job.total_cost,
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
        logger.error(f"WebSocket error for storyboard job {job_id}: {e}")
    finally:
        ws_manager.disconnect(job_id, websocket)


@router.get("/api/storyboard/{job_id}/download", summary="Download storyboard images", description="Download all scene images as a ZIP file.", responses={404: {"description": "Job or images not found"}, 400: {"description": "Job still in progress"}})
async def download_storyboard_images(job_id: str) -> Response:
    """Download all storyboard images as a ZIP file.

    Args:
        job_id: Job ID

    Returns:
        ZIP file containing all scene images and a manifest.
    """
    if job_id not in storyboard_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = storyboard_jobs[job_id]

    if job.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail="Job is still in progress",
        )

    # Collect all image URLs (references + scenes)
    all_images: list[tuple[str, str]] = []

    for name, url in job.reference_images.items():
        if url:
            all_images.append((f"references/{name}.png", url))

    for scene in job.scene_images:
        if scene.get("status") == "completed" and scene.get("image_url"):
            filename = f"scenes/scene_{scene['scene_number']:03d}.png"
            all_images.append((filename, scene["image_url"]))

    if not all_images:
        raise HTTPException(status_code=404, detail="No images to download")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    async with httpx.AsyncClient(timeout=60.0) as client:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, url in all_images:
                try:
                    # Skip data URLs (base64) - write directly
                    if url.startswith("data:"):
                        import base64
                        # Extract base64 data after the comma
                        b64_data = url.split(",", 1)[1] if "," in url else ""
                        zip_file.writestr(filename, base64.b64decode(b64_data))
                    else:
                        response = await client.get(url)
                        if response.status_code == 200:
                            zip_file.writestr(filename, response.content)
                except Exception as e:
                    logger.warning(f"Failed to download image {filename}: {e}")

            # Add manifest
            manifest = {
                "job_id": job.job_id,
                "plan": job.plan.to_dict() if job.plan else None,
                "reference_images": job.reference_images,
                "scene_images": job.scene_images,
                "total_cost": job.total_cost,
            }
            zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

    zip_buffer.seek(0)

    title_slug = "storyboard"
    if job.plan:
        title_slug = job.plan.title[:30].replace(" ", "_").lower()

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{title_slug}_{job_id[:8]}.zip"',
        },
    )
