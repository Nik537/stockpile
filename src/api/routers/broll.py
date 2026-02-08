"""B-roll processing routes for the Stockpile API."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from api.websocket_manager import WebSocketManager
from broll_processor import BRollProcessor
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from models.user_preferences import UserPreferences
from utils.config import load_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["B-Roll Processing"])

# Job storage (in-memory for MVP, would use Redis/DB in production)
jobs: dict[str, dict[str, Any]] = {}

# WebSocket manager for job status updates
ws_manager = WebSocketManager()


class JobStatus:
    """Job status constants."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def create_job(video_filename: str, preferences: dict | None = None) -> str:
    """Create a new processing job.

    Args:
        video_filename: Name of the uploaded video file
        preferences: User preferences for B-roll processing

    Returns:
        Job ID
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "video_filename": video_filename,
        "preferences": preferences or {},
        "status": JobStatus.QUEUED,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "progress": {"stage": "queued", "percent": 0, "message": "Job queued"},
        "error": None,
        "output_dir": None,
    }
    ws_manager.ensure_key(job_id)
    return job_id


async def update_job_status(
    job_id: str,
    status: str | None = None,
    progress: dict | None = None,
    error: str | None = None,
    output_dir: str | None = None,
) -> None:
    """Update job status and notify connected WebSocket clients.

    Args:
        job_id: Job ID
        status: New job status
        progress: Progress update dictionary
        error: Error message if failed
        output_dir: Output directory path if completed
    """
    if job_id not in jobs:
        return

    if status:
        jobs[job_id]["status"] = status
    if progress:
        jobs[job_id]["progress"] = progress
    if error:
        jobs[job_id]["error"] = error
    if output_dir:
        jobs[job_id]["output_dir"] = output_dir

    jobs[job_id]["updated_at"] = datetime.now().isoformat()

    # Notify all connected WebSocket clients for this job
    message = {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "progress": jobs[job_id]["progress"],
        "error": jobs[job_id].get("error"),
    }

    await ws_manager.broadcast(job_id, message)


async def process_job(job_id: str, video_path: str) -> None:
    """Process a video job in the background.

    Args:
        job_id: Job ID
        video_path: Path to uploaded video file
    """
    try:
        await update_job_status(
            job_id,
            status=JobStatus.PROCESSING,
            progress={"stage": "starting", "percent": 0, "message": "Starting processing"},
        )

        # Load configuration
        config = load_config()

        # Create processor
        processor = BRollProcessor(config)

        # Create status callback for progress updates
        def status_callback(stage: str, percent: int, message: str) -> None:
            """Callback for progress updates."""
            asyncio.create_task(
                update_job_status(
                    job_id, progress={"stage": stage, "percent": percent, "message": message}
                )
            )

        # Get user preferences
        prefs_dict = jobs[job_id].get("preferences", {})
        user_preferences = UserPreferences(**prefs_dict) if prefs_dict else None

        # Process video with status callback
        result = await processor.process_video(
            video_path, user_preferences, status_callback=status_callback
        )

        # Convert result to absolute path for reliable access
        output_dir_path = None
        if result:
            output_dir_path = Path(result).resolve()
            logger.info(f"Processing completed. Output directory: {output_dir_path}")

        # Update job with success
        await update_job_status(
            job_id,
            status=JobStatus.COMPLETED,
            progress={"stage": "completed", "percent": 100, "message": "Processing completed"},
            output_dir=str(output_dir_path) if output_dir_path else None,
        )

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await update_job_status(
            job_id,
            status=JobStatus.FAILED,
            progress={"stage": "failed", "percent": 0, "message": str(e)},
            error=str(e),
        )


@router.post(
    "/api/process",
    status_code=202,
    summary="Upload and process video",
    description="Upload a video file and start B-roll processing. Returns a job ID for tracking via WebSocket.",
    responses={400: {"description": "Invalid file or preferences JSON"}, 202: {"description": "Processing started"}},
)
async def process_video(
    file: UploadFile = File(...),
    preferences: str | None = None,
) -> JSONResponse:
    """Upload and process a video.

    Args:
        file: Video file upload
        preferences: JSON string of user preferences (optional)

    Returns:
        JSON response with job ID
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Parse preferences if provided
    prefs_dict = None
    if preferences:
        try:
            prefs_dict = json.loads(preferences)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid preferences JSON")

    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    with file_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    # Create job
    job_id = create_job(file.filename, prefs_dict)

    # Start processing in background
    asyncio.create_task(process_job(job_id, str(file_path)))

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "message": "Video uploaded successfully, processing started",
        },
    )


@router.get("/api/jobs", summary="List all jobs", description="Returns all processing jobs sorted by creation date (newest first).")
async def list_jobs() -> dict[str, list[dict]]:
    """List all jobs.

    Returns:
        Dictionary with jobs list
    """
    jobs_list = []
    for job in jobs.values():
        jobs_list.append(
            {
                "id": job["id"],
                "video_filename": job["video_filename"],
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "progress": job["progress"],
            }
        )

    # Sort by created_at descending (newest first)
    jobs_list.sort(key=lambda x: x["created_at"], reverse=True)

    return {"jobs": jobs_list}


@router.get("/api/jobs/{job_id}", summary="Get job details", responses={404: {"description": "Job not found"}})
async def get_job(job_id: str) -> dict:
    """Get job details.

    Args:
        job_id: Job ID

    Returns:
        Job details dictionary
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


@router.delete("/api/jobs/{job_id}", summary="Delete a job", responses={404: {"description": "Job not found"}})
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job.

    Args:
        job_id: Job ID

    Returns:
        Success message
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Remove job
    del jobs[job_id]

    # Clean up WebSocket connections
    ws_manager.cleanup(job_id)

    return {"message": "Job deleted successfully"}


@router.get("/api/jobs/{job_id}/download", summary="Download job results", responses={404: {"description": "Job or output not found"}, 400: {"description": "Job not completed"}})
async def download_results(job_id: str) -> FileResponse:
    """Download job results as a zip file.

    Args:
        job_id: Job ID

    Returns:
        Zip file with results
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_dir = job.get("output_dir")
    if not output_dir or not Path(output_dir).exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    # Create zip file of output directory
    import shutil

    zip_path = Path(output_dir).with_suffix(".zip")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir)

    return FileResponse(
        path=str(zip_path),
        filename=f"{job['video_filename']}_results.zip",
        media_type="application/zip",
    )


@router.websocket("/ws/status/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time job status updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    if job_id not in jobs:
        await websocket.accept()
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        # Send current status immediately
        await websocket.send_json(
            {
                "job_id": job_id,
                "status": jobs[job_id]["status"],
                "progress": jobs[job_id]["progress"],
                "error": jobs[job_id].get("error"),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong or other client messages if needed
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        # Remove from active connections
        ws_manager.disconnect(job_id, websocket)
