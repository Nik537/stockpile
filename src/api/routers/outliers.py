"""Outlier finder routes for the Stockpile API."""

import asyncio
import csv
import io
import json
import logging
import tempfile
import uuid
from datetime import datetime
from typing import Any

from api.schemas import OutlierSearchParams
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from models.outlier import OutlierVideo
from services.outlier_finder_service import OutlierFinderService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Outlier Finder"])

# Outlier search storage
outlier_searches: dict[str, dict[str, Any]] = {}

# WebSocket manager for outlier updates
ws_manager = WebSocketManager()

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()


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
        "min_views": params.min_views,
        "exclude_indian": params.exclude_indian,
        "status": "searching",
        "created_at": datetime.now().isoformat(),
        "channels_analyzed": 0,
        "total_channels": 0,
        "videos_scanned": 0,
        "outliers": [],
        "error": None,
    }
    ws_manager.ensure_key(search_id)
    return search_id


async def notify_outlier_clients(search_id: str, message: dict) -> None:
    """Send message to all connected WebSocket clients for a search.

    Args:
        search_id: Search ID
        message: Message to send
    """
    await ws_manager.broadcast(search_id, message)


def outlier_to_dict(outlier: OutlierVideo) -> dict:
    """Convert OutlierVideo to dictionary with all metrics.

    Args:
        outlier: OutlierVideo object

    Returns:
        Dictionary representation with full metrics
    """
    return {
        # Core fields
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
        # Engagement metrics
        "like_count": outlier.like_count,
        "comment_count": outlier.comment_count,
        "engagement_rate": (
            round(outlier.engagement_rate, 2) if outlier.engagement_rate else None
        ),
        # Velocity metrics
        "days_since_upload": outlier.days_since_upload,
        "views_per_day": (
            round(outlier.views_per_day, 0) if outlier.views_per_day else None
        ),
        "velocity_score": (
            round(outlier.velocity_score, 2) if outlier.velocity_score else None
        ),
        # Composite scoring
        "composite_score": (
            round(outlier.composite_score, 2) if outlier.composite_score else None
        ),
        "statistical_score": (
            round(outlier.statistical_score, 2) if outlier.statistical_score else None
        ),
        "engagement_score": (
            round(outlier.engagement_score, 2) if outlier.engagement_score else None
        ),
        # Reddit integration
        "found_on_reddit": outlier.found_on_reddit,
        "reddit_score": outlier.reddit_score,
        "reddit_subreddit": outlier.reddit_subreddit,
        # Momentum
        "momentum_score": (
            round(outlier.momentum_score, 2) if outlier.momentum_score else None
        ),
        "is_trending": outlier.is_trending,
    }


async def run_outlier_search(search_id: str) -> None:
    """Run an outlier search in the background.

    Args:
        search_id: Search ID
    """
    logger.info(f"run_outlier_search started for {search_id}")

    if search_id not in outlier_searches:
        logger.error(f"Search {search_id} not found in outlier_searches")
        return

    search = outlier_searches[search_id]

    try:
        logger.info(f"Creating OutlierFinderService for topic: {search['topic']}")
        # Create service with parameters
        service = OutlierFinderService(
            min_score=search["min_score"],
            date_days=search["days"],
            exclude_shorts=not search["include_shorts"],
            min_subs=search.get("min_subs"),
            max_subs=search.get("max_subs"),
            min_views=search.get("min_views", 5000),
            exclude_indian=search.get("exclude_indian", True),
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
        logger.info(f"Starting executor for search {search_id}")

        def run_search():
            logger.info(f"Executor thread started for {search_id}")
            try:
                result = service.find_outliers_by_topic(
                    topic=search["topic"],
                    max_channels=search["max_channels"],
                    on_outlier_found=on_outlier_found,
                    on_channel_complete=on_channel_complete,
                )
                logger.info(f"Executor thread completed for {search_id}: {len(result.outliers)} outliers")
                return result
            except Exception as e:
                logger.exception(f"Executor thread error for {search_id}: {e}")
                raise

        result = await loop.run_in_executor(None, run_search)

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


@router.post("/api/outliers/search", status_code=202, summary="Start outlier search", description="Start a new YouTube outlier search. Results stream via WebSocket.", responses={400: {"description": "Invalid parameters"}})
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

    # Start search in background - store task reference to prevent GC
    task = asyncio.create_task(run_outlier_search(search_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    logger.info(f"Started outlier search task for {search_id}")

    return JSONResponse(
        status_code=202,
        content={
            "search_id": search_id,
            "message": "Outlier search started",
        },
    )


@router.get("/api/outliers/{search_id}", summary="Get outlier search results", responses={404: {"description": "Search not found"}})
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


@router.get("/api/outliers/{search_id}/export", summary="Export outlier results", description="Export results as CSV or JSON file download.", responses={404: {"description": "Search not found"}})
async def export_outlier_search(search_id: str, format: str = "json") -> Any:
    """Export outlier search results as CSV or JSON.

    Args:
        search_id: Search ID
        format: Export format - "csv" or "json" (default: json)

    Returns:
        FileResponse with exported data
    """
    if search_id not in outlier_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    search = outlier_searches[search_id]
    outliers = search.get("outliers", [])

    if format.lower() == "csv":
        # Create CSV in memory
        output = io.StringIO()
        if outliers:
            # Get all unique keys from the outliers
            fieldnames = [
                "video_id",
                "title",
                "url",
                "channel_name",
                "view_count",
                "outlier_score",
                "outlier_tier",
                "channel_average_views",
                "upload_date",
                "like_count",
                "comment_count",
                "engagement_rate",
                "days_since_upload",
                "views_per_day",
                "velocity_score",
                "composite_score",
                "statistical_score",
                "engagement_score",
                "found_on_reddit",
                "reddit_score",
                "reddit_subreddit",
                "momentum_score",
                "is_trending",
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for outlier in outliers:
                writer.writerow(outlier)

        # Create temp file
        csv_content = output.getvalue()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_file:
            temp_file.write(csv_content)
            temp_path = temp_file.name

        return FileResponse(
            path=temp_path,
            filename=f"outliers_{search['topic'].replace(' ', '_')}_{search_id[:8]}.csv",
            media_type="text/csv",
        )

    else:
        # JSON format
        export_data = {
            "topic": search["topic"],
            "search_id": search_id,
            "created_at": search["created_at"],
            "channels_analyzed": search["channels_analyzed"],
            "videos_scanned": search["videos_scanned"],
            "total_outliers": len(outliers),
            "outliers": outliers,
        }

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            json.dump(export_data, temp_file, indent=2)
            temp_path = temp_file.name

        return FileResponse(
            path=temp_path,
            filename=f"outliers_{search['topic'].replace(' ', '_')}_{search_id[:8]}.json",
            media_type="application/json",
        )


@router.delete("/api/outliers/{search_id}", summary="Delete outlier search", responses={404: {"description": "Search not found"}})
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
    ws_manager.cleanup(search_id)

    return {"message": "Search deleted successfully"}


@router.websocket("/ws/outliers/{search_id}")
async def websocket_outliers(websocket: WebSocket, search_id: str) -> None:
    """WebSocket endpoint for real-time outlier search updates.

    Args:
        websocket: WebSocket connection
        search_id: Search ID to monitor
    """
    if search_id not in outlier_searches:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Search not found"})
        await websocket.close()
        return

    await ws_manager.connect(search_id, websocket)

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
        ws_manager.disconnect(search_id, websocket)
