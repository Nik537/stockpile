"""Music Service - HTTP client for music generation via Stable Audio 2.5."""

import asyncio
import logging
import os
import random

import httpx

logger = logging.getLogger(__name__)

REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
STABLE_AUDIO_MODEL_VERSION = "a61ac8edbb27cd2eda1b2eff2bbc03dcff1131f5560836ff77a052df05b77491"


class MusicServiceError(Exception):
    """Error from Music service."""

    pass


class MusicService:
    """HTTP client for music generation via Stable Audio 2.5 on Replicate."""

    def __init__(self):
        """Initialize Music service."""
        self.replicate_api_key = os.getenv("REPLICATE_API_KEY", "")
        # Long timeout - music generation can take a while
        self.client = httpx.AsyncClient(timeout=300.0)

    def is_configured(self) -> bool:
        """Check if the service is configured."""
        return bool(self.replicate_api_key)

    async def check_health(self) -> dict:
        """Check health status of the music generation backend.

        Returns:
            Health status dict.
        """
        return {
            "configured": bool(self.replicate_api_key),
            "available": bool(self.replicate_api_key),
            "error": None if self.replicate_api_key else "REPLICATE_API_KEY not configured",
            "model": "stable-audio-2.5",
        }

    async def generate_music(self, **kwargs) -> bytes:
        """Generate music using Stable Audio 2.5.

        Args:
            **kwargs: Generation parameters (genres, output_seconds, seed, steps, cfg)

        Returns:
            Audio bytes

        Raises:
            MusicServiceError: If generation fails
        """
        # Pop unused params that may be sent by older clients
        kwargs.pop("backend", None)
        kwargs.pop("lyrics", None)
        kwargs.pop("lyrics_strength", None)
        kwargs.pop("instrumental", None)
        return await self.generate_stable_audio(**kwargs)

    async def generate_stable_audio(
        self,
        genres: str,
        output_seconds: int = 60,
        seed: int | None = None,
        steps: int = 8,
        cfg: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Generate music using Stable Audio 2.5 via Replicate API.

        Stable Audio 2.5 is instrumental only -- lyrics are ignored.

        Args:
            genres: Genre/style description used as the prompt
            output_seconds: Duration in seconds (1-190)
            seed: Seed for reproducibility
            steps: Inference steps (4-8, default 8)
            cfg: CFG scale (1-25, default 1)

        Returns:
            Audio bytes

        Raises:
            MusicServiceError: If generation fails
        """
        if not self.replicate_api_key:
            raise MusicServiceError("REPLICATE_API_KEY not configured")

        # Clamp duration to Stable Audio's max of 190s
        output_seconds = min(output_seconds, 190)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        headers = {
            "Authorization": f"Bearer {self.replicate_api_key}",
            "Content-Type": "application/json",
        }

        input_params = {
            "prompt": genres,
            "duration": output_seconds,
            "steps": steps,
            "cfg_scale": cfg,
        }
        if seed is not None:
            input_params["seed"] = seed

        payload = {
            "version": STABLE_AUDIO_MODEL_VERSION,
            "input": input_params,
        }

        logger.info(
            f"Generating music via Stable Audio 2.5: prompt={genres}, "
            f"duration={output_seconds}s, steps={steps}, cfg_scale={cfg}, seed={seed}"
        )

        try:
            # Create prediction
            response = await self.client.post(REPLICATE_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            prediction = response.json()

            prediction_url = prediction.get("urls", {}).get("get") or f"{REPLICATE_API_URL}/{prediction['id']}"

            # Poll for completion
            max_wait = 300
            elapsed = 0
            poll_interval = 3

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status_response = await self.client.get(prediction_url, headers=headers)
                status_response.raise_for_status()
                result = status_response.json()

                status = result.get("status")
                if status == "succeeded":
                    # Stable Audio returns a single URL string (not an array)
                    output_url = result.get("output")
                    if not output_url:
                        raise MusicServiceError("Stable Audio returned no output URL")

                    # Download the audio file
                    audio_response = await self.client.get(output_url)
                    audio_response.raise_for_status()
                    audio_bytes = audio_response.content

                    if not audio_bytes or len(audio_bytes) < 100:
                        raise MusicServiceError("Stable Audio returned empty audio")

                    logger.info(f"Stable Audio generation complete: {len(audio_bytes)} bytes")
                    return audio_bytes

                elif status == "failed":
                    error = result.get("error", "Unknown error")
                    raise MusicServiceError(f"Stable Audio generation failed: {error}")

                elif status in ("starting", "processing"):
                    continue

            raise MusicServiceError(f"Stable Audio generation timed out after {max_wait}s")

        except httpx.TimeoutException:
            raise MusicServiceError("Stable Audio request timed out")
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise MusicServiceError(f"Stable Audio API error: {error_detail}")
        except MusicServiceError:
            raise
        except Exception as e:
            raise MusicServiceError(f"Stable Audio music generation failed: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
