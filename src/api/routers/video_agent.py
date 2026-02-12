"""Video Production Agent routes."""

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.dependencies import get_voice_library
from api.schemas import VideoProduceRequest
from api.websocket_manager import WebSocketManager
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Video Agent"])

# Video job storage (in-memory)
video_jobs: dict[str, dict] = {}

# WebSocket manager for video job updates
ws_manager = WebSocketManager()


def estimate_cost(
    target_duration: int,
    competitive_analysis_enabled: bool = True,
    previews_per_need: int = 2,
    use_processor_broll: bool = True,
) -> dict:
    """Estimate production cost breakdown for a given duration in minutes.

    Stock images are FREE (Pexels/Pixabay). Nano Banana Pro style enhancement
    costs $0.04/image when enabled (auto-enabled if RUNPOD_API_KEY is set).

    Args:
        target_duration: Target video duration in minutes.
        competitive_analysis_enabled: Whether competitive analysis is on.
        previews_per_need: Number of preview videos compared per scene.
        use_processor_broll: Whether enhanced B-roll processor pipeline is used.

    Returns:
        Cost breakdown dict with tts, images, music, broll, director, total.
    """
    import os

    scenes = target_duration * 6
    tts = round(target_duration * 0.045, 4)

    # Image cost: Google web search (SerpAPI ~$0.005/search) + optional Nano Banana Pro styling ($0.04/image)
    style_setting = os.getenv("IMAGE_STYLE_ENHANCEMENT", "auto")
    runpod_configured = bool(os.getenv("RUNPOD_API_KEY"))
    serpapi_configured = bool(os.getenv("SERPAPI_KEY"))
    style_enabled = (
        style_setting == "true"
        or (style_setting == "auto" and runpod_configured)
    )
    # SerpAPI ~$0.005/search; Nano Banana Pro $0.04/image if styling enabled
    search_cost = round(scenes * 0.005, 4) if serpapi_configured else 0.0
    style_cost = round(scenes * 0.04, 4) if style_enabled else 0.0
    images = round(search_cost + style_cost, 4)

    music = 0.03

    # B-roll cost: evaluation + download + clip extraction per scene
    # Base: ~$0.01/scene for AI evaluation + extraction
    # Competitive analysis multiplies evaluation cost by previews_per_need
    if use_processor_broll:
        broll_scenes = scenes // 2  # roughly half scenes get B-roll video
        base_broll_cost = round(broll_scenes * 0.01, 4)
        if competitive_analysis_enabled:
            # Competitive analysis downloads previews_per_need candidates per scene
            broll = round(base_broll_cost * previews_per_need, 4)
        else:
            broll = base_broll_cost
    else:
        broll = 0.0

    director = round(0.01 * 2, 4)  # ~2 review loops
    total = round(tts + images + music + broll + director, 4)

    if style_enabled and serpapi_configured:
        img_note = "Google web search + Nano Banana Pro styling"
    elif serpapi_configured:
        img_note = "Google web search (primary) + Pexels/Pixabay"
    elif style_enabled:
        img_note = "stock photos + Nano Banana Pro styling"
    else:
        img_note = "stock photos (Pexels/Pixabay, free)"

    result = {
        "tts": tts,
        "images": images,
        "images_note": img_note,
        "music": music,
        "broll": broll,
        "director": director,
        "total": total,
    }

    if use_processor_broll:
        result["broll_note"] = (
            f"Enhanced pipeline, competitive={'on' if competitive_analysis_enabled else 'off'}, "
            f"{previews_per_need} previews/scene"
        )

    return result

# Keep references to background tasks to prevent garbage collection
_background_tasks: set = set()

# Jobs directory
VIDEO_JOBS_DIR = Path.home() / ".stockpile" / "video_jobs"


async def _run_video_production(
    job_id: str,
    topic: str,
    style: str,
    target_duration: int,
    subtitle_style: str,
    voice_ref: str | None,
    *,
    competitive_analysis_enabled: bool = True,
    previews_per_need: int = 2,
    clips_per_need_target: int = 1,
    use_processor_broll: bool = True,
    semantic_verification_enabled: bool = True,
    style_detection_enabled: bool = True,
    video_only: bool = False,
) -> None:
    """Run video production pipeline in the background.

    Args:
        job_id: Job ID
        topic: Video topic
        style: Video style
        target_duration: Target duration in minutes
        subtitle_style: Subtitle style preset
        voice_ref: Optional path to voice reference file
        competitive_analysis_enabled: Enable multi-candidate B-roll comparison
        previews_per_need: Number of preview videos to compare per scene
        clips_per_need_target: Final clips per scene
        use_processor_broll: Use enhanced B-roll processor pipeline
        semantic_verification_enabled: Verify clips match scene context
        style_detection_enabled: Detect content style for visual coherence
    """
    if job_id not in video_jobs:
        logger.error(f"Video job {job_id} not found")
        return

    job = video_jobs[job_id]

    try:
        from utils.config import load_config
        from video_agent.agent import VideoProductionAgent
        from video_agent.video_composer import VideoComposer

        # Create fresh agent per job with B-roll pipeline config overrides
        config = load_config()
        config["competitive_analysis_enabled"] = competitive_analysis_enabled
        config["previews_per_need"] = previews_per_need
        config["clips_per_need_target"] = clips_per_need_target
        config["use_processor_broll"] = use_processor_broll
        config["semantic_verification_enabled"] = semantic_verification_enabled
        config["style_detection_enabled"] = style_detection_enabled
        config["video_only"] = video_only
        agent = VideoProductionAgent(config)

        # Set up project directory
        project_dir = VIDEO_JOBS_DIR / job_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Set up composer to output into project directory
        agent.composer = VideoComposer(output_dir=project_dir)

        try:
            # -- Stage 1: Script Generation (0-10%) --
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "script_generation",
                "percent": 5,
                "message": "Generating script...",
            })
            job["stage"] = "script_generation"
            job["percent"] = 5

            script = await asyncio.to_thread(
                agent.script_gen.generate, topic, style, target_duration
            )

            # Save script JSON for debugging
            agent._save_script_json(script, project_dir)

            # Broadcast script data
            await ws_manager.broadcast(job_id, {
                "type": "script",
                "title": script.title,
                "hook_voiceover": script.hook.voiceover,
                "scenes": [
                    {
                        "id": s.id,
                        "voiceover": s.voiceover,
                        "visual_type": s.visual_type.value,
                        "visual_keywords": s.visual_keywords,
                        "duration_est": s.duration_est,
                    }
                    for s in script.scenes
                ],
            })

            # Store script in job for REST polling
            job["script"] = {
                "title": script.title,
                "hook_voiceover": script.hook.voiceover,
                "scenes": [
                    {
                        "id": s.id,
                        "voiceover": s.voiceover,
                        "visual_type": s.visual_type.value,
                        "visual_keywords": s.visual_keywords,
                        "duration_est": s.duration_est,
                    }
                    for s in script.scenes
                ],
            }

            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "script_generation",
                "percent": 10,
                "message": "Script generated.",
            })
            job["percent"] = 10

            # -- Stage 2: Narration (10-30%) --
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "narration",
                "percent": 12,
                "message": "Generating narration...",
            })
            job["stage"] = "narration"
            job["percent"] = 12

            scene_audios = await agent._generate_narration(script, project_dir, voice_ref)

            master_audio = project_dir / "master_audio.wav"
            agent._merge_audio_files(scene_audios, master_audio)

            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "narration",
                "percent": 30,
                "message": "Narration complete.",
            })
            job["percent"] = 30

            # Broadcast cost update after TTS
            await ws_manager.broadcast(job_id, {
                "type": "cost_update",
                "cost": job["cost"],
            })

            # -- Stage 3: Word Timestamps (30-35%) --
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "word_timestamps",
                "percent": 32,
                "message": "Generating word timestamps...",
            })
            job["stage"] = "word_timestamps"
            job["percent"] = 32

            word_timings = await agent._generate_word_timestamps(master_audio)

            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "word_timestamps",
                "percent": 35,
                "message": "Word timestamps generated.",
            })
            job["percent"] = 35

            # -- Stage 4: Asset Acquisition (35-70%) --
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "asset_acquisition",
                "percent": 40,
                "message": "Acquiring visuals and generating music...",
            })
            job["stage"] = "asset_acquisition"
            job["percent"] = 40

            async def asset_progress(percent: int, message: str) -> None:
                """Broadcast scene-level asset acquisition progress via WebSocket."""
                job["percent"] = percent
                await ws_manager.broadcast(job_id, {
                    "type": "progress",
                    "stage": "asset_acquisition",
                    "percent": percent,
                    "message": message,
                })

            visual_paths, music_path = await asyncio.gather(
                agent._acquire_visuals(
                    script, project_dir, progress_callback=asset_progress
                ),
                agent._generate_music(script, project_dir),
            )

            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "asset_acquisition",
                "percent": 70,
                "message": f"Assets acquired: {len(visual_paths)} visuals.",
            })
            job["percent"] = 70

            # Broadcast cost update after assets
            await ws_manager.broadcast(job_id, {
                "type": "cost_update",
                "cost": job["cost"],
            })

            # -- Stage 5: Subtitle Generation (70-80%) --
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "subtitle_generation",
                "percent": 72,
                "message": "Building timeline and generating subtitles...",
            })
            job["stage"] = "subtitle_generation"
            job["percent"] = 72

            timeline = agent._build_timeline(
                script, scene_audios, visual_paths,
                master_audio, music_path, word_timings,
                subtitle_style, project_dir,
            )

            ass_path = await agent._generate_subtitles(
                word_timings, subtitle_style, project_dir
            )
            if ass_path:
                timeline.subtitle_path = ass_path

            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "subtitle_generation",
                "percent": 80,
                "message": "Subtitles generated.",
            })
            job["percent"] = 80

            # -- Stage 6: Director Review (80-90%) --
            if agent.director:
                await ws_manager.broadcast(job_id, {
                    "type": "progress",
                    "stage": "director_review",
                    "percent": 82,
                    "message": "Director reviewing draft...",
                })
                job["stage"] = "director_review"
                job["percent"] = 82

                timeline = await agent._director_review_loop(
                    script, timeline, project_dir
                )

                await ws_manager.broadcast(job_id, {
                    "type": "progress",
                    "stage": "director_review",
                    "percent": 90,
                    "message": "Director review complete.",
                })
                job["percent"] = 90

            # -- Stage 7: Final Video Composition (90-100%) --
            await ws_manager.broadcast(job_id, {
                "type": "progress",
                "stage": "video_composition",
                "percent": 92,
                "message": "Composing final video (1080p)...",
            })
            job["stage"] = "video_composition"
            job["percent"] = 92

            final_video = await agent.composer.compose(timeline, draft=False)

            # -- Complete --
            job["status"] = "completed"
            job["percent"] = 100
            job["stage"] = "complete"
            job["video_path"] = str(final_video)
            job["completed_at"] = datetime.now(timezone.utc).isoformat()

            logger.info(f"Video job {job_id} completed: {final_video}")

            await ws_manager.broadcast(job_id, {
                "type": "complete",
                "status": "completed",
                "percent": 100,
                "message": "Video production complete!",
                "video_path": str(final_video),
            })

        except Exception as e:
            logger.error(f"Video job {job_id} failed: {e}")
            job["status"] = "failed"
            job["error"] = str(e)

            await ws_manager.broadcast(job_id, {
                "type": "error",
                "status": "failed",
                "message": str(e),
            })

        finally:
            await agent.close()

    except Exception as e:
        logger.error(f"Video job {job_id} initialization failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)

        await ws_manager.broadcast(job_id, {
            "type": "error",
            "status": "failed",
            "message": str(e),
        })


# =============================================================================
# Video Production Endpoints
# =============================================================================


@router.post(
    "/api/video/estimate-cost",
    summary="Estimate video production cost",
    description="Return an approximate cost breakdown for a given target duration.",
)
async def estimate_video_cost(request: VideoProduceRequest) -> dict:
    """Return estimated cost breakdown based on target duration and B-roll config.

    Args:
        request: Video production parameters.

    Returns:
        Cost breakdown dict.
    """
    return estimate_cost(
        request.target_duration,
        competitive_analysis_enabled=request.competitive_analysis_enabled,
        previews_per_need=request.previews_per_need,
        use_processor_broll=request.use_processor_broll,
    )


@router.post("/api/video/produce", summary="Start video production", description="Start an autonomous video production pipeline. Returns job ID immediately.", responses={202: {"description": "Job accepted"}, 404: {"description": "Voice not found"}})
async def produce_video(request: VideoProduceRequest):
    """Start async video production. Returns 202 with job_id immediately.

    Args:
        request: Video production parameters

    Returns:
        202 response with job_id and status
    """
    # Resolve voice_id to voice_ref path if provided
    voice_ref: str | None = None
    if request.voice_id:
        library = get_voice_library()
        voice = library.get_voice(request.voice_id)
        if not voice:
            raise HTTPException(status_code=404, detail=f"Voice {request.voice_id} not found")
        audio_path = library.get_audio_path(request.voice_id)
        voice_ref = str(audio_path) if audio_path else None

    # Create job
    job_id = uuid.uuid4().hex[:12]
    cost_estimate = estimate_cost(
        request.target_duration,
        competitive_analysis_enabled=request.competitive_analysis_enabled,
        previews_per_need=request.previews_per_need,
        use_processor_broll=request.use_processor_broll,
    )
    job = {
        "id": job_id,
        "status": "processing",
        "topic": request.topic,
        "style": request.style,
        "target_duration": request.target_duration,
        "subtitle_style": request.subtitle_style,
        "voice_id": request.voice_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "stage": "queued",
        "percent": 0,
        "script": None,
        "video_path": None,
        "error": None,
        "cost": cost_estimate,
        # B-roll pipeline config
        "competitive_analysis_enabled": request.competitive_analysis_enabled,
        "previews_per_need": request.previews_per_need,
        "clips_per_need_target": request.clips_per_need_target,
        "use_processor_broll": request.use_processor_broll,
        "semantic_verification_enabled": request.semantic_verification_enabled,
        "style_detection_enabled": request.style_detection_enabled,
        "video_only": request.video_only,
    }
    video_jobs[job_id] = job

    # Ensure WebSocket key exists before task starts broadcasting
    ws_manager.ensure_key(job_id)

    # Start production in background
    task = asyncio.create_task(
        _run_video_production(
            job_id=job_id,
            topic=request.topic,
            style=request.style,
            target_duration=request.target_duration,
            subtitle_style=request.subtitle_style,
            voice_ref=voice_ref,
            competitive_analysis_enabled=request.competitive_analysis_enabled,
            previews_per_need=request.previews_per_need,
            clips_per_need_target=request.clips_per_need_target,
            use_processor_broll=request.use_processor_broll,
            semantic_verification_enabled=request.semantic_verification_enabled,
            style_detection_enabled=request.style_detection_enabled,
            video_only=request.video_only,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "processing"},
    )


@router.get("/api/video/jobs", summary="List video jobs", description="List all video production jobs.")
async def list_video_jobs() -> list[dict]:
    """List all video production jobs.

    Returns:
        List of job dicts (without internal paths).
    """
    jobs = []
    for job in video_jobs.values():
        job_info = {k: v for k, v in job.items() if k != "video_path"}
        jobs.append(job_info)
    return jobs


@router.get("/api/video/jobs/{job_id}", summary="Get video job status", responses={404: {"description": "Job not found"}})
async def get_video_job(job_id: str) -> dict:
    """Get a single video job status.

    Args:
        job_id: Job ID

    Returns:
        Job status dict with progress and script.
    """
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = video_jobs[job_id]
    return {k: v for k, v in job.items() if k != "video_path"}


@router.get("/api/video/jobs/{job_id}/download", summary="Download video", description="Download the completed video file.", responses={404: {"description": "Job or video not found"}, 400: {"description": "Job not completed"}})
async def download_video(job_id: str):
    """Download the final video for a completed job.

    Args:
        job_id: Job ID

    Returns:
        Video file response.
    """
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = video_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job['status']})",
        )

    # Find video file - check explicit path first, then scan project dir
    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"video_{job_id}.mp4",
        )

    # Fallback: scan project directory for .mp4 files
    project_dir = VIDEO_JOBS_DIR / job_id
    if project_dir.exists():
        mp4_files = list(project_dir.glob("*.mp4"))
        if mp4_files:
            return FileResponse(
                str(mp4_files[0]),
                media_type="video/mp4",
                filename=f"video_{job_id}.mp4",
            )

    raise HTTPException(status_code=404, detail="Video file not found")


def _resolve_video_path(job_id: str) -> Path:
    """Resolve the video file path for a completed job.

    Args:
        job_id: Job ID

    Returns:
        Path to the video file.

    Raises:
        HTTPException: If job not found, not completed, or video file missing.
    """
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = video_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job['status']})",
        )

    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        return Path(video_path)

    project_dir = VIDEO_JOBS_DIR / job_id
    if project_dir.exists():
        mp4_files = list(project_dir.glob("*.mp4"))
        if mp4_files:
            return mp4_files[0]

    raise HTTPException(status_code=404, detail="Video file not found")


@router.get(
    "/api/video/jobs/{job_id}/stream",
    summary="Stream video",
    description="Stream the completed video with HTTP Range support for seeking.",
    responses={
        200: {"description": "Full video"},
        206: {"description": "Partial content"},
        404: {"description": "Job or video not found"},
        400: {"description": "Job not completed"},
    },
)
async def stream_video(job_id: str, request: Request):
    """Stream the final video for a completed job with Range request support.

    Args:
        job_id: Job ID
        request: FastAPI request (for Range header)

    Returns:
        StreamingResponse with appropriate range headers.
    """
    video_path = _resolve_video_path(job_id)
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range: bytes=start-end
        range_spec = range_header.strip().lower()
        if not range_spec.startswith("bytes="):
            raise HTTPException(status_code=416, detail="Invalid range header")

        range_val = range_spec[6:]
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(
                status_code=416,
                detail="Range not satisfiable",
                headers={"Content-Range": f"bytes */{file_size}"},
            )

        content_length = end - start + 1

        def iter_range():
            chunk_size = 64 * 1024  # 64KB chunks
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    # No range header - serve full file
    def iter_full():
        chunk_size = 64 * 1024
        with open(video_path, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

    return StreamingResponse(
        iter_full(),
        status_code=200,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


@router.delete("/api/video/jobs/{job_id}", summary="Delete video job", description="Delete a video job and its project directory.", responses={404: {"description": "Job not found"}})
async def delete_video_job(job_id: str) -> dict:
    """Delete a video job and clean up its project directory.

    Args:
        job_id: Job ID

    Returns:
        Confirmation message.
    """
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Clean up project directory
    project_dir = VIDEO_JOBS_DIR / job_id
    if project_dir.exists():
        try:
            shutil.rmtree(project_dir)
        except Exception as e:
            logger.warning(f"Failed to delete project dir for job {job_id}: {e}")

    del video_jobs[job_id]
    return {"message": "Job deleted", "job_id": job_id}


# =============================================================================
# Video WebSocket
# =============================================================================


@router.websocket("/ws/video/{job_id}")
async def websocket_video(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time video production updates.

    Args:
        websocket: WebSocket connection
        job_id: Job ID to monitor
    """
    if job_id not in video_jobs:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    await ws_manager.connect(job_id, websocket)

    try:
        job = video_jobs[job_id]

        # Send current status immediately
        status_msg = {
            "type": "status",
            "status": job["status"],
            "job_id": job_id,
            "stage": job.get("stage"),
            "percent": job.get("percent", 0),
            "error": job.get("error"),
            "cost": job.get("cost"),
        }
        await websocket.send_json(status_msg)

        # If script is already available, send it
        if job.get("script"):
            await websocket.send_json({
                "type": "script",
                **job["script"],
            })

        # If job is already completed, send completion message
        if job["status"] == "completed":
            await websocket.send_json({
                "type": "complete",
                "status": "completed",
                "percent": 100,
                "message": "Video production complete!",
            })

        # Keep connection alive with ping/pong
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
