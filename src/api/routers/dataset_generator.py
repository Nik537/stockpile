"""Dataset generator routes for the Stockpile API.

Provides endpoints for generating LoRA training datasets using cheap image providers.
Supports 4 modes: pair, single, reference, layered.
"""

import asyncio
import logging
from pathlib import Path

from api.dependencies import get_dataset_gen_service
from api.schemas import (
    DatasetGenerateRequest,
)
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from models.dataset_generator import (
    DatasetGenerationRequest,
    DatasetJob,
    DatasetMode,
    DatasetStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dataset Generator"])

# Dataset job storage (in-memory)
dataset_jobs: dict[str, DatasetJob] = {}

# WebSocket manager for dataset job updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()


async def notify_dataset_clients(job_id: str, message: dict) -> None:
    """Send WebSocket message to all connected clients for a dataset job."""
    await ws_manager.broadcast(job_id, message)


async def run_dataset_generation(job_id: str) -> None:
    """Run dataset generation in the background."""
    if job_id not in dataset_jobs:
        logger.error(f"Dataset job {job_id} not found")
        return

    job = dataset_jobs[job_id]
    service = get_dataset_gen_service()

    async def on_progress(message: dict) -> None:
        await notify_dataset_clients(job_id, message)

    try:
        await service.run_generation(job, progress_callback=on_progress)
    except Exception as e:
        logger.error(f"Dataset generation failed for job {job_id}: {e}")
        job.status = DatasetStatus.FAILED
        job.error = str(e)
        await notify_dataset_clients(job_id, {
            "type": "error",
            "message": str(e),
        })


@router.post(
    "/api/dataset/generate",
    summary="Start dataset generation",
    description="Start generating a LoRA training dataset. Returns job_id for tracking progress.",
    responses={400: {"description": "Invalid parameters"}},
    status_code=202,
)
async def start_dataset_generation(request: DatasetGenerateRequest) -> dict:
    """Start a dataset generation job."""
    # Validate mode
    try:
        mode = DatasetMode(request.mode)
    except ValueError:
        valid_modes = [m.value for m in DatasetMode]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Valid modes: {valid_modes}",
        )

    if not request.theme or not request.theme.strip():
        raise HTTPException(status_code=400, detail="Theme is required")

    if request.num_items < 1 or request.num_items > 200:
        raise HTTPException(status_code=400, detail="num_items must be between 1 and 200")

    if request.max_concurrent < 1 or request.max_concurrent > 10:
        raise HTTPException(status_code=400, detail="max_concurrent must be between 1 and 10")

    # Pair mode requires transformation
    if mode == DatasetMode.PAIR and not request.transformation:
        raise HTTPException(
            status_code=400,
            detail="Pair mode requires a transformation description",
        )

    # Reference mode requires reference image
    if mode == DatasetMode.REFERENCE and not request.reference_image_base64:
        raise HTTPException(
            status_code=400,
            detail="Reference mode requires a reference image (base64)",
        )

    # Build internal request
    gen_request = DatasetGenerationRequest(
        mode=mode,
        theme=request.theme.strip(),
        model=request.model,
        llm_model=request.llm_model,
        num_items=request.num_items,
        max_concurrent=request.max_concurrent,
        aspect_ratio=request.aspect_ratio,
        trigger_word=request.trigger_word.strip() if request.trigger_word else "",
        use_vision_caption=request.use_vision_caption,
        custom_system_prompt=request.custom_system_prompt.strip() if request.custom_system_prompt else "",
        transformation=request.transformation.strip() if request.transformation else "",
        action_name=request.action_name.strip() if request.action_name else "",
        reference_image_base64=request.reference_image_base64 or "",
        layered_use_case=request.layered_use_case or "character",
        elements_description=request.elements_description.strip() if request.elements_description else "",
        final_image_description=request.final_image_description.strip() if request.final_image_description else "",
        width=request.width,
        height=request.height,
    )

    # Create job
    service = get_dataset_gen_service()
    job_id = service.generate_job_id()

    job = DatasetJob(
        job_id=job_id,
        request=gen_request,
        status=DatasetStatus.PENDING,
        total_count=gen_request.num_items,
        estimated_cost=service.calculate_estimated_cost(gen_request),
    )

    dataset_jobs[job_id] = job

    # Ensure WS key exists before task starts
    ws_manager.ensure_key(job_id)

    # Start background task
    task = asyncio.create_task(run_dataset_generation(job_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {
        "job_id": job_id,
        "status": "pending",
        "estimated_cost": job.estimated_cost,
        "total_count": job.total_count,
    }


@router.get(
    "/api/dataset/{job_id}/status",
    summary="Get dataset job status",
    responses={404: {"description": "Job not found"}},
)
async def get_dataset_status(job_id: str) -> dict:
    """Get dataset job status and progress."""
    if job_id not in dataset_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return dataset_jobs[job_id].to_dict()


@router.get(
    "/api/dataset/{job_id}/download",
    summary="Download dataset ZIP",
    description="Download the completed dataset as a ZIP file.",
    responses={
        404: {"description": "Job or ZIP not found"},
        400: {"description": "Job not completed"},
    },
)
async def download_dataset(job_id: str) -> FileResponse:
    """Download dataset as a ZIP file."""
    if job_id not in dataset_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dataset_jobs[job_id]

    if job.status not in (DatasetStatus.COMPLETED, DatasetStatus.FAILED):
        raise HTTPException(status_code=400, detail="Job is still in progress")

    if not job.zip_path:
        raise HTTPException(status_code=404, detail="No ZIP file available")

    zip_path = Path(job.zip_path)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found on disk")

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"dataset_{job_id[:8]}.zip",
    )


@router.delete(
    "/api/dataset/{job_id}",
    summary="Cancel dataset job",
    responses={404: {"description": "Job not found"}},
)
async def cancel_dataset_job(job_id: str) -> dict:
    """Cancel a dataset generation job."""
    if job_id not in dataset_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dataset_jobs[job_id]

    if job.status in (DatasetStatus.COMPLETED, DatasetStatus.FAILED, DatasetStatus.CANCELLED):
        return {"message": f"Job already {job.status.value}", "job_id": job_id}

    job.status = DatasetStatus.CANCELLED

    await notify_dataset_clients(job_id, {
        "type": "cancelled",
        "message": "Job cancelled by user",
    })

    return {"message": "Job cancelled", "job_id": job_id}


@router.websocket("/ws/dataset/{job_id}")
async def websocket_dataset_job(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time dataset generation updates."""
    if job_id not in dataset_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = dataset_jobs[job_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "status": job.status.value,
            "total_count": job.total_count,
            "completed_count": job.completed_count,
            "failed_count": job.failed_count,
            "total_cost": job.total_cost,
            "error": job.error,
        })

        # If already completed, send final state
        if job.status in (DatasetStatus.COMPLETED, DatasetStatus.FAILED):
            await websocket.send_json({
                "type": "complete",
                "status": job.status.value,
                "completed_count": job.completed_count,
                "failed_count": job.failed_count,
                "total_count": job.total_count,
                "total_cost": job.total_cost,
                "zip_path": job.zip_path,
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
        logger.error(f"WebSocket error for dataset job {job_id}: {e}")
    finally:
        ws_manager.disconnect(job_id, websocket)
