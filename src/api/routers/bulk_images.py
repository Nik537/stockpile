"""Bulk image generation routes for the Stockpile API."""

import asyncio
import io
import json
import logging
import zipfile

import httpx
from api.dependencies import get_ai_service, get_bulk_image_service, get_image_gen_service
from api.schemas import BulkImageGenerateRequest, BulkImagePromptsRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from models.bulk_image import BulkImageJob, BulkImagePrompt

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Bulk Image Generation"])

# Bulk image job storage (in-memory)
bulk_image_jobs: dict[str, BulkImageJob] = {}

# WebSocket manager for bulk image updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()


async def notify_bulk_image_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for a bulk image job.

    Args:
        job_id: Job ID
        message: Message to send
    """
    await ws_manager.broadcast(job_id, message)


async def run_bulk_image_generation(job_id: str) -> None:
    """Run bulk image generation in the background.

    Args:
        job_id: Job ID
    """
    if job_id not in bulk_image_jobs:
        logger.error(f"Bulk image job {job_id} not found")
        return

    job = bulk_image_jobs[job_id]
    service = get_bulk_image_service()

    try:
        job.status = "generating_images"

        # Notify start
        await notify_bulk_image_clients(job_id, {
            "type": "status",
            "status": "generating_images",
            "total_count": job.total_count,
            "completed_count": 0,
            "failed_count": 0,
        })

        # Define progress callback
        async def on_progress(result):
            message_type = "image_complete" if result.status == "completed" else "image_failed"
            await notify_bulk_image_clients(job_id, {
                "type": message_type,
                "index": result.index,
                "image_url": result.image_url,
                "prompt": result.prompt.to_dict(),
                "status": result.status,
                "error": result.error,
                "completed_count": job.completed_count,
                "failed_count": job.failed_count,
                "total_count": job.total_count,
            })

        # Generate all images
        await service.generate_all_images(job, on_progress=on_progress)

        # Notify completion
        await notify_bulk_image_clients(job_id, {
            "type": "complete",
            "status": job.status,
            "completed_count": job.completed_count,
            "failed_count": job.failed_count,
            "total_count": job.total_count,
            "total_cost": job.total_cost,
        })

    except Exception as e:
        logger.error(f"Bulk image generation failed for job {job_id}: {e}")
        job.status = "failed"
        job.error = str(e)

        await notify_bulk_image_clients(job_id, {
            "type": "error",
            "message": str(e),
        })


@router.post("/api/bulk-image/prompts", summary="Generate bulk image prompts", description="Step 1: Generate unique image prompts from a meta-prompt using AI. Review and edit before generating.", responses={400: {"description": "Invalid meta-prompt or count"}, 500: {"description": "Prompt generation failed"}})
async def generate_bulk_image_prompts(request: BulkImagePromptsRequest) -> dict:
    """Generate unique image prompts from a meta-prompt.

    Step 1 of bulk image generation workflow.

    Args:
        request: Meta-prompt and count

    Returns:
        Job ID and generated prompts for review
    """
    if not request.meta_prompt or not request.meta_prompt.strip():
        raise HTTPException(status_code=400, detail="Meta-prompt is required")

    if request.count < 10 or request.count > 200:
        raise HTTPException(status_code=400, detail="Count must be between 10 and 200")

    service = get_bulk_image_service()

    # Check if AI service is configured
    ai_service = get_ai_service()
    if not ai_service.api_key:
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY not configured. Set it in your .env file.",
        )

    try:
        # Generate prompts
        prompts = service.generate_prompts(request.meta_prompt.strip(), request.count)

        if not prompts:
            raise HTTPException(status_code=500, detail="Failed to generate prompts")

        # Create job
        job_id = service.generate_job_id()
        job = BulkImageJob(
            job_id=job_id,
            meta_prompt=request.meta_prompt.strip(),
            model="runware-flux-klein-4b",  # Default, will be set in generate
            width=1024,
            height=1024,
            total_count=len(prompts),
            prompts=prompts,
            status="pending",
        )

        # Calculate estimates
        estimated_cost = service.calculate_estimated_cost(
            len(prompts), "runware-flux-klein-4b", 1024, 1024
        )
        estimated_time = service.calculate_estimated_time(len(prompts), "runware-flux-klein-4b")
        job.estimated_cost = estimated_cost

        # Store job
        bulk_image_jobs[job_id] = job

        return {
            "job_id": job_id,
            "prompts": [p.to_dict() for p in prompts],
            "estimated_cost": estimated_cost,
            "estimated_time_seconds": estimated_time,
        }

    except Exception as e:
        logger.error(f"Bulk image prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/bulk-image/generate", summary="Start bulk image generation", description="Step 2: Generate images from reviewed prompts. Track progress via WebSocket.", responses={404: {"description": "Job not found"}, 400: {"description": "Invalid model or job already started"}})
async def start_bulk_image_generation(request: BulkImageGenerateRequest) -> dict:
    """Start generating images from (optionally edited) prompts.

    Step 2 of bulk image generation workflow.

    Args:
        request: Job ID, prompts, model, and dimensions

    Returns:
        Job status
    """
    if request.job_id not in bulk_image_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if not request.prompts:
        raise HTTPException(status_code=400, detail="Prompts are required")

    # Supported models by provider
    RUNWARE_MODELS = {"runware-flux-klein-4b", "runware-flux-klein-9b", "runware-z-image"}
    GEMINI_MODELS = {"gemini-flash"}
    RUNPOD_MODELS = {"nano-banana-pro"}
    ALL_MODELS = RUNWARE_MODELS | GEMINI_MODELS | RUNPOD_MODELS

    if request.model not in ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Supported: {', '.join(sorted(ALL_MODELS))}",
        )

    # Check the appropriate provider is configured
    image_service = get_image_gen_service()

    if request.model in RUNWARE_MODELS and not image_service.is_runware_configured():
        raise HTTPException(
            status_code=400,
            detail="RUNWARE_API_KEY not configured. Set it in your .env file.",
        )
    elif request.model in GEMINI_MODELS and not image_service.is_gemini_configured():
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY not configured. Set it in your .env file.",
        )
    elif request.model in RUNPOD_MODELS and not image_service.is_runpod_configured():
        raise HTTPException(
            status_code=400,
            detail="RUNPOD_API_KEY not configured. Set it in your .env file.",
        )

    job = bulk_image_jobs[request.job_id]

    if job.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Job is already {job.status}",
        )

    # Update job with user's edited prompts and settings
    job.prompts = [
        BulkImagePrompt(
            index=p.index,
            prompt=p.prompt,
            rendering_style=p.rendering_style,
            mood=p.mood,
            composition=p.composition,
            has_text_space=p.has_text_space,
        )
        for p in request.prompts
    ]
    job.model = request.model
    job.width = request.width
    job.height = request.height
    job.total_count = len(request.prompts)

    # Recalculate estimates
    service = get_bulk_image_service()
    job.estimated_cost = service.calculate_estimated_cost(
        len(request.prompts), request.model, request.width, request.height
    )

    # Start generation in background
    task = asyncio.create_task(run_bulk_image_generation(request.job_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "job_id": request.job_id,
        "status": "generating_images",
        "total_count": job.total_count,
        "estimated_cost": job.estimated_cost,
    }


@router.get("/api/bulk-image/{job_id}", summary="Get bulk image job status", responses={404: {"description": "Job not found"}})
async def get_bulk_image_job(job_id: str) -> dict:
    """Get bulk image job status and results.

    Args:
        job_id: Job ID

    Returns:
        Full job status with prompts and results
    """
    if job_id not in bulk_image_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return bulk_image_jobs[job_id].to_dict()


@router.get("/api/bulk-image/{job_id}/download", summary="Download bulk images", description="Download all completed images as a ZIP file with manifest.", responses={404: {"description": "Job or images not found"}, 400: {"description": "Job still in progress"}})
async def download_bulk_images(job_id: str) -> Response:
    """Download all completed images as a ZIP file.

    Args:
        job_id: Job ID

    Returns:
        ZIP file containing all images and a manifest
    """
    if job_id not in bulk_image_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = bulk_image_jobs[job_id]

    if job.status not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail="Job is still in progress",
        )

    # Filter successful results
    successful_results = [r for r in job.results if r.status == "completed" and r.image_url]

    if not successful_results:
        raise HTTPException(status_code=404, detail="No images to download")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    async with httpx.AsyncClient(timeout=60.0) as client:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Download and add each image
            for result in successful_results:
                try:
                    response = await client.get(result.image_url)
                    if response.status_code == 200:
                        # Determine file extension from content type
                        content_type = response.headers.get("content-type", "image/jpeg")
                        ext = "jpg" if "jpeg" in content_type else "png"
                        filename = f"image_{result.index:03d}.{ext}"
                        zip_file.writestr(filename, response.content)
                except Exception as e:
                    logger.warning(f"Failed to download image {result.index}: {e}")

            # Add manifest
            manifest = {
                "job_id": job.job_id,
                "meta_prompt": job.meta_prompt,
                "model": job.model,
                "width": job.width,
                "height": job.height,
                "total_count": job.total_count,
                "completed_count": job.completed_count,
                "failed_count": job.failed_count,
                "total_cost": job.total_cost,
                "prompts": [
                    {
                        "index": r.index,
                        "prompt": r.prompt.prompt,
                        "rendering_style": r.prompt.rendering_style,
                        "mood": r.prompt.mood,
                        "composition": r.prompt.composition,
                        "has_text_space": r.prompt.has_text_space,
                        "image_url": r.image_url,
                    }
                    for r in successful_results
                ],
            }
            zip_file.writestr("manifest.json", json.dumps(manifest, indent=2))

    zip_buffer.seek(0)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="bulk_images_{job_id[:8]}.zip"',
        },
    )


@router.delete("/api/bulk-image/{job_id}", summary="Cancel bulk image job", responses={404: {"description": "Job not found"}})
async def cancel_bulk_image_job(job_id: str) -> dict:
    """Cancel a bulk image job.

    Args:
        job_id: Job ID

    Returns:
        Confirmation message
    """
    if job_id not in bulk_image_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = bulk_image_jobs[job_id]
    job.status = "cancelled"

    # Note: Currently running tasks will complete, but no new ones will start
    # A more sophisticated implementation would track and cancel individual tasks

    return {"message": "Job cancelled", "job_id": job_id}


@router.websocket("/ws/bulk-image/{job_id}")
async def websocket_bulk_image(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time bulk image generation updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    if job_id not in bulk_image_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = bulk_image_jobs[job_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "status": job.status,
            "total_count": job.total_count,
            "completed_count": job.completed_count,
            "failed_count": job.failed_count,
            "error": job.error,
        })

        # If job is already completed, send results
        if job.status == "completed":
            await websocket.send_json({
                "type": "complete",
                "status": job.status,
                "completed_count": job.completed_count,
                "failed_count": job.failed_count,
                "total_count": job.total_count,
                "total_cost": job.total_cost,
                "results": [r.to_dict() for r in job.results],
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
        logger.error(f"WebSocket error for bulk image job {job_id}: {e}")
    finally:
        # Remove from active connections
        ws_manager.disconnect(job_id, websocket)
