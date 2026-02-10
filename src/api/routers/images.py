"""Image generation routes for the Stockpile API (Runware, Gemini, Nano Banana Pro)."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from api.dependencies import get_image_gen_service
from api.schemas import ImageEditRequestBody, ImageGenerateRequest, RunPodImageGenerateRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from models.image_generation import (
    ImageEditRequest,
    ImageGenerationModel,
    ImageGenerationRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Generation"])

# Image job storage (in-memory)
image_jobs: dict[str, dict] = {}

# WebSocket manager for image job updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()


async def _notify_image_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for an image job."""
    await ws_manager.broadcast(job_id, message)


async def _run_image_generate(job_id: str, gen_request: ImageGenerationRequest) -> None:
    """Run image generation in the background."""
    if job_id not in image_jobs:
        logger.error(f"Image job {job_id} not found")
        return

    job = image_jobs[job_id]

    try:
        service = get_image_gen_service()
        result = await service.generate_image(gen_request)
        result_dict = result.to_dict()

        job["status"] = "completed"
        job["result"] = result_dict
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_image_clients(job_id, {
            "type": "complete",
            "job_id": job_id,
            "status": "completed",
            "result": result_dict,
        })

    except Exception as e:
        logger.error(f"Image generation failed for job {job_id}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_image_clients(job_id, {
            "type": "error",
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        })


async def _run_image_edit(job_id: str, edit_request: ImageEditRequest) -> None:
    """Run image editing in the background."""
    if job_id not in image_jobs:
        logger.error(f"Image job {job_id} not found")
        return

    job = image_jobs[job_id]

    try:
        service = get_image_gen_service()
        result = await service.edit_image(edit_request)
        result_dict = result.to_dict()

        job["status"] = "completed"
        job["result"] = result_dict
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_image_clients(job_id, {
            "type": "complete",
            "job_id": job_id,
            "status": "completed",
            "result": result_dict,
        })

    except Exception as e:
        logger.error(f"Image editing failed for job {job_id}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        await _notify_image_clients(job_id, {
            "type": "error",
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        })


# =============================================================================
# Unified Image Generation Endpoints
# =============================================================================


@router.get("/api/image/status", summary="Image generation status", description="Check status of all configured image generation providers.")
async def get_image_status() -> dict:
    """Combined status for all image generation providers."""
    service = get_image_gen_service()
    runware = await service.check_runware_health()
    gemini = await service.check_gemini_health()
    runpod = await service.check_runpod_health()
    return {
        "runware": runware,
        "gemini": gemini,
        "runpod": runpod,
        "default_model": service.default_model,
    }


@router.post("/api/image/generate", summary="Generate image (unified)", description="Generate images using any supported model. Returns 202 with job_id for async processing.", responses={202: {"description": "Job accepted"}, 400: {"description": "Invalid parameters"}}, status_code=202)
async def unified_generate_image(request: ImageGenerateRequest) -> dict:
    """Unified image generation endpoint - returns 202 with job_id."""
    service = get_image_gen_service()

    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        valid = [m.value for m in ImageGenerationModel]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Valid models: {valid}",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    if request.width < 256 or request.width > 2048:
        raise HTTPException(status_code=400, detail="Width must be between 256 and 2048")
    if request.height < 256 or request.height > 2048:
        raise HTTPException(status_code=400, detail="Height must be between 256 and 2048")
    if request.num_images < 1 or request.num_images > 4:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 4")

    gen_request = ImageGenerationRequest(
        prompt=request.prompt.strip(),
        model=model,
        width=request.width,
        height=request.height,
        num_images=request.num_images,
        seed=request.seed,
        guidance_scale=request.guidance_scale,
    )

    # Create job
    job_id = uuid.uuid4().hex[:12]
    image_jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "type": "generate",
        "prompt_preview": request.prompt.strip()[:80],
        "model": request.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    # Ensure WS key exists before task starts
    ws_manager.ensure_key(job_id)

    # Start background task
    task = asyncio.create_task(_run_image_generate(job_id, gen_request))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"job_id": job_id, "status": "processing"}


@router.post("/api/image/edit", summary="Edit image (unified)", description="Edit/inpaint images. Returns 202 with job_id for async processing.", responses={202: {"description": "Job accepted"}, 400: {"description": "Invalid parameters"}}, status_code=202)
async def unified_edit_image(request: ImageEditRequestBody) -> dict:
    """Unified image editing endpoint - returns 202 with job_id."""
    service = get_image_gen_service()

    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'.",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not request.image_url or not request.image_url.strip():
        raise HTTPException(status_code=400, detail="Image URL is required")

    if request.strength < 0 or request.strength > 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1")

    edit_request = ImageEditRequest(
        prompt=request.prompt.strip(),
        input_image_url=request.image_url.strip(),
        model=model,
        strength=request.strength,
        seed=request.seed,
        guidance_scale=request.guidance_scale,
        mask_image=request.mask_image,
    )

    # Determine job type
    job_type = "inpaint" if request.mask_image else "edit"

    # Create job
    job_id = uuid.uuid4().hex[:12]
    image_jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "type": job_type,
        "prompt_preview": request.prompt.strip()[:80],
        "model": request.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    # Ensure WS key exists before task starts
    ws_manager.ensure_key(job_id)

    # Start background task
    task = asyncio.create_task(_run_image_edit(job_id, edit_request))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"job_id": job_id, "status": "processing"}


# =============================================================================
# Image Job Management Endpoints
# =============================================================================


@router.get("/api/image/jobs", summary="List image jobs", description="List all image generation/edit jobs.")
async def list_image_jobs() -> list[dict]:
    """List all image jobs."""
    return list(image_jobs.values())


@router.get("/api/image/jobs/{job_id}", summary="Get image job", description="Get status and result of a specific image job.", responses={404: {"description": "Job not found"}})
async def get_image_job(job_id: str) -> dict:
    """Get image job status and result."""
    if job_id not in image_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return image_jobs[job_id]


@router.delete("/api/image/jobs/{job_id}", summary="Delete image job", description="Remove an image job from memory.", responses={404: {"description": "Job not found"}})
async def delete_image_job(job_id: str) -> dict:
    """Delete an image job."""
    if job_id not in image_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del image_jobs[job_id]
    ws_manager.cleanup(job_id)
    return {"message": "Job deleted", "job_id": job_id}


# =============================================================================
# Image Job WebSocket
# =============================================================================


@router.websocket("/ws/image/{job_id}")
async def websocket_image_job(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time image job updates."""
    if job_id not in image_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = image_jobs[job_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "job_id": job_id,
            "status": job["status"],
            "result": job.get("result"),
            "error": job.get("error"),
        })

        # If already completed or failed, send final state
        if job["status"] in ("completed", "failed"):
            msg_type = "complete" if job["status"] == "completed" else "error"
            await websocket.send_json({
                "type": msg_type,
                "job_id": job_id,
                "status": job["status"],
                "result": job.get("result"),
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
        logger.error(f"WebSocket error for image job {job_id}: {e}")
    finally:
        ws_manager.disconnect(job_id, websocket)


# =============================================================================
# Backward-compatible endpoints (delegate to unified methods)
# =============================================================================


@router.get("/api/image-generation/status", summary="Legacy: fal.ai status", description="Legacy endpoint. Use /api/image/status instead.", include_in_schema=False)
async def get_image_generation_status() -> dict:
    """Legacy status endpoint - redirects to unified status."""
    return await get_image_status()


@router.post("/api/generate-image", summary="Legacy: Generate image", description="Legacy endpoint. Use /api/image/generate instead.", include_in_schema=False, status_code=202)
async def generate_image(request: ImageGenerateRequest) -> dict:
    """Legacy generate endpoint - delegates to unified."""
    return await unified_generate_image(request)


@router.post("/api/edit-image", summary="Legacy: Edit image", description="Legacy endpoint. Use /api/image/edit instead.", include_in_schema=False, status_code=202)
async def edit_image(request: ImageEditRequestBody) -> dict:
    """Legacy edit endpoint - delegates to unified."""
    return await unified_edit_image(request)


@router.get("/api/runpod-image/status", summary="Legacy: RunPod status", include_in_schema=False)
async def get_runpod_image_status() -> dict:
    """Legacy RunPod status endpoint."""
    service = get_image_gen_service()
    return await service.check_runpod_health()


@router.post("/api/runpod-image/generate", summary="Legacy: RunPod generate", include_in_schema=False, status_code=202)
async def generate_runpod_image(request: RunPodImageGenerateRequest) -> dict:
    """Legacy RunPod generate endpoint - delegates to unified."""
    gen_request = ImageGenerateRequest(
        prompt=request.prompt,
        model=request.model,
        width=request.width,
        height=request.height,
        seed=request.seed,
    )
    return await unified_generate_image(gen_request)


@router.get("/api/gemini-image/status", summary="Legacy: Gemini status", include_in_schema=False)
async def get_gemini_image_status() -> dict:
    """Legacy Gemini status endpoint."""
    service = get_image_gen_service()
    return await service.check_gemini_health()


@router.post("/api/gemini-image/generate", summary="Legacy: Gemini generate", include_in_schema=False, status_code=202)
async def generate_gemini_image(request: RunPodImageGenerateRequest) -> dict:
    """Legacy Gemini generate endpoint - delegates to unified."""
    gen_request = ImageGenerateRequest(
        prompt=request.prompt,
        model="gemini-flash",
        width=request.width,
        height=request.height,
        seed=request.seed,
    )
    return await unified_generate_image(gen_request)
