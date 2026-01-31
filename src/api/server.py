#!/usr/bin/env python
"""FastAPI server for stockpile web interface."""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src directory to Python path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from broll_processor import BRollProcessor
from models.user_preferences import UserPreferences
from models.outlier import OutlierVideo
from services.outlier_finder_service import OutlierFinderService
from utils.config import load_config

logger = logging.getLogger(__name__)

# Job storage (in-memory for MVP, would use Redis/DB in production)
jobs: dict[str, dict[str, Any]] = {}
active_websockets: dict[str, list[WebSocket]] = {}

# Outlier search storage
outlier_searches: dict[str, dict[str, Any]] = {}
outlier_websockets: dict[str, list[WebSocket]] = {}


class OutlierSearchParams(BaseModel):
    """Parameters for outlier search."""

    topic: str
    max_channels: int = 10
    min_score: float = 3.0
    days: int | None = None
    include_shorts: bool = False
    min_subs: int | None = None
    max_subs: int | None = None

app = FastAPI(title="Stockpile API", version="1.0.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    active_websockets[job_id] = []
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

    disconnected = []
    for ws in active_websockets.get(job_id, []):
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected WebSockets
    for ws in disconnected:
        active_websockets[job_id].remove(ws)


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


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Stockpile API", "version": "1.0.0"}


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/process")
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


@app.get("/api/jobs")
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


@app.get("/api/jobs/{job_id}")
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


@app.delete("/api/jobs/{job_id}")
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
    if job_id in active_websockets:
        del active_websockets[job_id]

    return {"message": "Job deleted successfully"}


@app.get("/api/jobs/{job_id}/download")
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


@app.websocket("/ws/status/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time job status updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    await websocket.accept()

    if job_id not in jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    # Add to active connections
    active_websockets[job_id].append(websocket)

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
        if websocket in active_websockets[job_id]:
            active_websockets[job_id].remove(websocket)


# =============================================================================
# Outlier Finder Endpoints
# =============================================================================


def create_outlier_search(params: OutlierSearchParams) -> str:
    """Create a new outlier search.

    Args:
        params: Search parameters

    Returns:
        Search ID
    """
    search_id = str(uuid.uuid4())
    outlier_searches[search_id] = {
        "id": search_id,
        "topic": params.topic,
        "max_channels": params.max_channels,
        "min_score": params.min_score,
        "days": params.days,
        "include_shorts": params.include_shorts,
        "min_subs": params.min_subs,
        "max_subs": params.max_subs,
        "status": "searching",
        "created_at": datetime.now().isoformat(),
        "channels_analyzed": 0,
        "total_channels": 0,
        "videos_scanned": 0,
        "outliers": [],
        "error": None,
    }
    outlier_websockets[search_id] = []
    return search_id


async def notify_outlier_clients(search_id: str, message: dict) -> None:
    """Send message to all connected WebSocket clients for a search.

    Args:
        search_id: Search ID
        message: Message to send
    """
    disconnected = []
    for ws in outlier_websockets.get(search_id, []):
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected WebSockets
    for ws in disconnected:
        if ws in outlier_websockets.get(search_id, []):
            outlier_websockets[search_id].remove(ws)


def outlier_to_dict(outlier: OutlierVideo) -> dict:
    """Convert OutlierVideo to dictionary.

    Args:
        outlier: OutlierVideo object

    Returns:
        Dictionary representation
    """
    return {
        "video_id": outlier.video_id,
        "title": outlier.title,
        "url": outlier.url,
        "thumbnail_url": outlier.thumbnail_url,
        "view_count": outlier.view_count,
        "outlier_score": round(outlier.outlier_score, 2),
        "channel_average_views": round(outlier.channel_average_views, 0),
        "channel_name": outlier.channel_name,
        "upload_date": outlier.upload_date,
        "outlier_tier": outlier.outlier_tier,
    }


async def run_outlier_search(search_id: str) -> None:
    """Run an outlier search in the background.

    Args:
        search_id: Search ID
    """
    if search_id not in outlier_searches:
        return

    search = outlier_searches[search_id]

    try:
        # Create service with parameters
        service = OutlierFinderService(
            min_score=search["min_score"],
            date_days=search["days"],
            exclude_shorts=not search["include_shorts"],
            min_subs=search.get("min_subs"),
            max_subs=search.get("max_subs"),
        )

        # Capture the event loop BEFORE entering the executor
        # This is critical because callbacks run in a thread pool without an event loop
        loop = asyncio.get_running_loop()

        # Define callbacks that will notify WebSocket clients
        # These run in a background thread, so we use call_soon_threadsafe
        def on_outlier_found(outlier: OutlierVideo) -> None:
            """Called when an outlier is found."""
            outlier_dict = outlier_to_dict(outlier)
            search["outliers"].append(outlier_dict)

            # Schedule async notification from worker thread to main event loop
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    notify_outlier_clients(
                        search_id,
                        {"type": "outlier", "outlier": outlier_dict}
                    )
                )
            )

        def on_channel_complete(channels_done: int, total: int, videos: int) -> None:
            """Called when a channel analysis is complete."""
            search["channels_analyzed"] = channels_done
            search["total_channels"] = total
            search["videos_scanned"] = videos

            # Schedule async notification from worker thread to main event loop
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    notify_outlier_clients(
                        search_id,
                        {
                            "type": "progress",
                            "channels_analyzed": channels_done,
                            "total_channels": total,
                            "videos_scanned": videos,
                        }
                    )
                )
            )

        # Run the search in a thread pool to avoid blocking
        result = await loop.run_in_executor(
            None,
            lambda: service.find_outliers_by_topic(
                topic=search["topic"],
                max_channels=search["max_channels"],
                on_outlier_found=on_outlier_found,
                on_channel_complete=on_channel_complete,
            )
        )

        # Update search with final results
        search["status"] = "completed"
        search["channels_analyzed"] = result.channels_analyzed
        search["videos_scanned"] = result.total_videos_scanned

        # Notify completion
        await notify_outlier_clients(
            search_id,
            {
                "type": "complete",
                "total_outliers": len(search["outliers"]),
                "channels_analyzed": result.channels_analyzed,
                "videos_scanned": result.total_videos_scanned,
            }
        )

    except Exception as e:
        logger.exception(f"Outlier search {search_id} failed: {e}")
        search["status"] = "failed"
        search["error"] = str(e)

        await notify_outlier_clients(
            search_id,
            {"type": "error", "message": str(e)}
        )


@app.post("/api/outliers/search")
async def start_outlier_search(params: OutlierSearchParams) -> JSONResponse:
    """Start a new outlier search.

    Args:
        params: Search parameters

    Returns:
        JSON response with search ID
    """
    if not params.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")

    if params.max_channels < 1 or params.max_channels > 100:
        raise HTTPException(status_code=400, detail="max_channels must be between 1 and 100")

    if params.min_score < 1.0:
        raise HTTPException(status_code=400, detail="min_score must be at least 1.0")

    # Create search
    search_id = create_outlier_search(params)

    # Start search in background
    asyncio.create_task(run_outlier_search(search_id))

    return JSONResponse(
        status_code=202,
        content={
            "search_id": search_id,
            "message": "Outlier search started",
        },
    )


@app.get("/api/outliers/{search_id}")
async def get_outlier_search(search_id: str) -> dict:
    """Get outlier search status and results.

    Args:
        search_id: Search ID

    Returns:
        Search status and results
    """
    if search_id not in outlier_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    return outlier_searches[search_id]


@app.delete("/api/outliers/{search_id}")
async def delete_outlier_search(search_id: str) -> dict[str, str]:
    """Delete an outlier search.

    Args:
        search_id: Search ID

    Returns:
        Success message
    """
    if search_id not in outlier_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    del outlier_searches[search_id]

    if search_id in outlier_websockets:
        del outlier_websockets[search_id]

    return {"message": "Search deleted successfully"}


@app.websocket("/ws/outliers/{search_id}")
async def websocket_outliers(websocket: WebSocket, search_id: str) -> None:
    """WebSocket endpoint for real-time outlier search updates.

    Args:
        websocket: WebSocket connection
        search_id: Search ID to monitor
    """
    await websocket.accept()

    if search_id not in outlier_searches:
        await websocket.send_json({"type": "error", "message": "Search not found"})
        await websocket.close()
        return

    # Add to active connections
    if search_id not in outlier_websockets:
        outlier_websockets[search_id] = []
    outlier_websockets[search_id].append(websocket)

    try:
        search = outlier_searches[search_id]

        # Send current status immediately
        await websocket.send_json({
            "type": "status",
            "status": search["status"],
            "channels_analyzed": search["channels_analyzed"],
            "total_channels": search["total_channels"],
            "videos_scanned": search["videos_scanned"],
            "outliers": search["outliers"],
            "error": search.get("error"),
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for outlier search {search_id}: {e}")
    finally:
        # Remove from active connections
        if search_id in outlier_websockets and websocket in outlier_websockets[search_id]:
            outlier_websockets[search_id].remove(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
