"""Video generation service - LTX-Video 2 via RunPod custom endpoint."""

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class VideoGenServiceError(Exception):
    """Raised when video generation fails."""


class VideoGenService:
    """Generates videos using LTX-Video 2 on a custom RunPod endpoint."""

    def __init__(self) -> None:
        self.runpod_api_key = os.getenv("RUNPOD_API_KEY", "")
        self.endpoint_id = os.getenv("RUNPOD_LTX_VIDEO_ENDPOINT_ID", "")
        self.client = httpx.AsyncClient(timeout=300.0)

    def is_configured(self) -> bool:
        return bool(self.runpod_api_key and self.endpoint_id)

    async def check_health(self) -> dict:
        if not self.runpod_api_key:
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_API_KEY not configured",
            }
        if not self.endpoint_id:
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_LTX_VIDEO_ENDPOINT_ID not configured",
            }

        # Check endpoint health
        try:
            url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/health"
            headers = {"Authorization": f"Bearer {self.runpod_api_key}"}
            response = await self.client.get(url, headers=headers, timeout=10.0)
            data = response.json()
            workers = data.get("workers", {})
            ready = workers.get("ready", 0)
            return {
                "configured": True,
                "available": ready > 0 or workers.get("initializing", 0) > 0,
                "workers_ready": ready,
                "workers_total": workers.get("ready", 0) + workers.get("initializing", 0) + workers.get("idle", 0),
            }
        except Exception as e:
            return {
                "configured": True,
                "available": False,
                "error": f"Health check failed: {e}",
            }

    async def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.0,
        seed: int | None = None,
        fps: int = 24,
        conditioning_images: list[str] | None = None,
        conditioning_strength: float = 1.0,
    ) -> dict:
        """Generate a video using LTX-Video 2.

        Args:
            prompt: Text description of the video.
            negative_prompt: What to avoid.
            width: Video width (must be divisible by 32).
            height: Video height (must be divisible by 32).
            num_frames: Total frames (must be 8n+1, e.g. 25, 49, 97, 121).
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale.
            seed: Random seed for reproducibility.
            fps: Frames per second for export.
            conditioning_images: List of base64 data URLs for image-to-video.
            conditioning_strength: How strongly to condition on images (0-1).

        Returns:
            Dict with video_url, seed, generation_time_ms.
        """
        if not self.is_configured():
            raise VideoGenServiceError(
                "Video generation not configured. Set RUNPOD_API_KEY and "
                "RUNPOD_LTX_VIDEO_ENDPOINT_ID in your .env file."
            )

        url = f"{RUNPOD_API_BASE}/{self.endpoint_id}/runsync"
        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "input": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "fps": fps,
            }
        }

        if seed is not None:
            payload["input"]["seed"] = seed

        # Image-to-video conditioning
        if conditioning_images:
            payload["input"]["conditioning_images"] = conditioning_images
            payload["input"]["conditioning_strength"] = conditioning_strength

            # Spread images evenly across frames
            n = len(conditioning_images)
            if n == 1:
                frame_indices = [0]
            else:
                step = (num_frames - 1) / (n - 1)
                frame_indices = [round(i * step) for i in range(n)]
            payload["input"]["conditioning_frame_indices"] = frame_indices

        duration_sec = num_frames / fps
        logger.info(
            f"Generating video with LTX-Video 2 ({width}x{height}, "
            f"{num_frames} frames, ~{duration_sec:.1f}s, "
            f"{len(conditioning_images or [])} conditioning images)"
        )

        start_time = time.time()

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            if result_data.get("status") == "FAILED":
                error_msg = result_data.get("error", "Unknown error")
                raise VideoGenServiceError(f"LTX-Video 2 generation failed: {error_msg}")

            output = result_data.get("output", {})

            # Extract video URL from various possible response shapes
            video_url = (
                output.get("video_url")
                or output.get("video")
                or output.get("result")
            )

            if not video_url:
                raise VideoGenServiceError("LTX-Video 2 returned no video URL")

            logger.info(f"LTX-Video 2 generated video in {generation_time_ms}ms")

            return {
                "video_url": video_url,
                "seed": output.get("seed", seed),
                "generation_time_ms": generation_time_ms,
            }

        except httpx.TimeoutException:
            raise VideoGenServiceError(
                "LTX-Video 2 request timed out. Video generation can take 1-5 minutes."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise VideoGenServiceError(f"LTX-Video 2 API error: {error_detail}")
        except VideoGenServiceError:
            raise
        except Exception as e:
            raise VideoGenServiceError(f"LTX-Video 2 generation failed: {e}")

    async def close(self) -> None:
        await self.client.aclose()
