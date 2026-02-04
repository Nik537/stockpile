"""Image Generation Service - HTTP client for fal.ai image generation APIs."""

import base64
import json
import logging
import os
import time
from pathlib import Path

import httpx

from models.image_generation import (
    GeneratedImage,
    ImageEditRequest,
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
)

logger = logging.getLogger(__name__)

# Settings file for persistence
SETTINGS_FILE = Path.home() / ".stockpile" / "image_gen_settings.json"

# fal.ai API configuration
FAL_API_KEY = os.getenv("FAL_API_KEY", "")
FAL_API_BASE = "https://fal.run"

# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# Model endpoints mapping (fal.ai)
MODEL_ENDPOINTS = {
    ImageGenerationModel.FLUX_KLEIN: {
        "generate": "fal-ai/flux-2/klein/4b",
        "edit": "fal-ai/flux-2/klein/4b/edit",
    },
    ImageGenerationModel.Z_IMAGE: {
        "generate": "fal-ai/z-image/turbo",
        "edit": "fal-ai/z-image/turbo/image-to-image",
    },
}

# RunPod public endpoint slugs
RUNPOD_ENDPOINTS = {
    ImageGenerationModel.RUNPOD_FLUX_DEV: "black-forest-labs-flux-1-dev",
    ImageGenerationModel.RUNPOD_FLUX_SCHNELL: "black-forest-labs-flux-1-schnell",
    ImageGenerationModel.RUNPOD_FLUX_KONTEXT: "black-forest-labs-flux-1-kontext-dev",
}

# Pricing per megapixel (approximate)
PRICING_PER_MP = {
    ImageGenerationModel.FLUX_KLEIN: 0.012,  # ~$0.009-0.014/MP
    ImageGenerationModel.Z_IMAGE: 0.005,  # ~$0.005/MP
    ImageGenerationModel.RUNPOD_FLUX_DEV: 0.02,  # $0.02/MP
    ImageGenerationModel.RUNPOD_FLUX_SCHNELL: 0.0024,  # $0.0024/MP
    ImageGenerationModel.RUNPOD_FLUX_KONTEXT: 0.02,  # ~$0.02/MP
}


class ImageGenerationServiceError(Exception):
    """Error from image generation service."""

    pass


class ImageGenerationService:
    """HTTP client for fal.ai image generation APIs."""

    def __init__(self, api_key: str = "", runpod_api_key: str = ""):
        """Initialize the image generation service.

        Args:
            api_key: fal.ai API key. If not provided, uses FAL_API_KEY env var.
            runpod_api_key: RunPod API key. If not provided, uses RUNPOD_API_KEY env var.
        """
        self.api_key = api_key or FAL_API_KEY
        self.runpod_api_key = runpod_api_key or RUNPOD_API_KEY
        self.default_model = self._load_settings().get(
            "default_model", ImageGenerationModel.FLUX_KLEIN.value
        )
        # Long timeout for image generation (can take a while)
        self.client = httpx.AsyncClient(timeout=120.0)

    def _load_settings(self) -> dict:
        """Load settings from file."""
        try:
            if SETTINGS_FILE.exists():
                return json.loads(SETTINGS_FILE.read_text())
        except Exception as e:
            logger.warning(f"Failed to load image generation settings: {e}")
        return {}

    def _save_settings(self) -> None:
        """Persist settings to file."""
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            settings = {"default_model": self.default_model}
            SETTINGS_FILE.write_text(json.dumps(settings))
            logger.debug(f"Image generation settings saved to {SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save image generation settings: {e}")

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def _get_headers(self) -> dict:
        """Get HTTP headers for fal.ai API requests."""
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def _calculate_cost(
        self, model: ImageGenerationModel, width: int, height: int, num_images: int
    ) -> float:
        """Calculate estimated cost for image generation.

        Args:
            model: The model being used
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images to generate

        Returns:
            Estimated cost in USD
        """
        megapixels = (width * height) / 1_000_000
        price_per_mp = PRICING_PER_MP.get(model, 0.01)
        return megapixels * num_images * price_per_mp

    async def check_health(self) -> dict:
        """Check if the service is configured and accessible.

        Returns:
            Health status dict with 'configured', 'available', and optional 'error'.
        """
        if not self.is_configured():
            return {
                "configured": False,
                "available": False,
                "error": "FAL_API_KEY not configured",
            }

        # fal.ai serverless endpoints are always "available" when configured
        # The actual availability is determined at request time
        return {
            "configured": True,
            "available": True,
            "default_model": self.default_model,
        }

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate images from a text prompt.

        Args:
            request: The generation request parameters

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationServiceError: If generation fails
        """
        if not self.is_configured():
            raise ImageGenerationServiceError(
                "FAL_API_KEY not configured. Set it in your .env file."
            )

        model = request.model
        endpoint = MODEL_ENDPOINTS[model]["generate"]
        url = f"{FAL_API_BASE}/{endpoint}"

        # Build request payload
        payload = {
            "prompt": request.prompt,
            "image_size": {"width": request.width, "height": request.height},
            "num_images": request.num_images,
            "guidance_scale": request.guidance_scale,
        }

        if request.seed is not None:
            payload["seed"] = request.seed

        logger.info(
            f"Generating {request.num_images} image(s) with {model.value} "
            f"({request.width}x{request.height})"
        )

        start_time = time.time()

        try:
            response = await self.client.post(
                url, headers=self._get_headers(), json=payload
            )
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            # Parse response
            images = []
            for img_data in result_data.get("images", []):
                images.append(
                    GeneratedImage(
                        url=img_data.get("url", ""),
                        width=img_data.get("width", request.width),
                        height=img_data.get("height", request.height),
                        content_type=img_data.get("content_type", "image/png"),
                        seed=result_data.get("seed"),
                    )
                )

            cost = self._calculate_cost(
                model, request.width, request.height, len(images)
            )

            logger.info(
                f"Generated {len(images)} image(s) in {generation_time_ms}ms "
                f"(est. cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=model,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=cost,
            )

        except httpx.TimeoutException:
            raise ImageGenerationServiceError(
                "Image generation timed out. The server may be overloaded. "
                "Try again or use a smaller image size."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"fal.ai API error: {error_detail}")
        except Exception as e:
            raise ImageGenerationServiceError(f"Image generation failed: {e}")

    async def edit(self, request: ImageEditRequest) -> ImageGenerationResult:
        """Edit an image using a text prompt.

        Args:
            request: The edit request parameters including input image

        Returns:
            ImageGenerationResult with edited images

        Raises:
            ImageGenerationServiceError: If editing fails
        """
        if not self.is_configured():
            raise ImageGenerationServiceError(
                "FAL_API_KEY not configured. Set it in your .env file."
            )

        model = request.model
        endpoint = MODEL_ENDPOINTS[model]["edit"]
        url = f"{FAL_API_BASE}/{endpoint}"

        # Build request payload
        payload = {
            "prompt": request.prompt,
            "image_url": request.input_image_url,
            "strength": request.strength,
            "guidance_scale": request.guidance_scale,
        }

        if request.seed is not None:
            payload["seed"] = request.seed

        logger.info(f"Editing image with {model.value} (strength={request.strength})")

        start_time = time.time()

        try:
            response = await self.client.post(
                url, headers=self._get_headers(), json=payload
            )
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            # Parse response
            images = []
            for img_data in result_data.get("images", []):
                images.append(
                    GeneratedImage(
                        url=img_data.get("url", ""),
                        width=img_data.get("width", 1024),
                        height=img_data.get("height", 1024),
                        content_type=img_data.get("content_type", "image/png"),
                        seed=result_data.get("seed"),
                    )
                )

            # Estimate cost based on output image size
            width = images[0].width if images else 1024
            height = images[0].height if images else 1024
            cost = self._calculate_cost(model, width, height, len(images))

            logger.info(
                f"Edited {len(images)} image(s) in {generation_time_ms}ms "
                f"(est. cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=model,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=cost,
            )

        except httpx.TimeoutException:
            raise ImageGenerationServiceError(
                "Image editing timed out. The server may be overloaded. "
                "Try again or use a smaller image."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"fal.ai API error: {error_detail}")
        except Exception as e:
            raise ImageGenerationServiceError(f"Image editing failed: {e}")

    def is_runpod_configured(self) -> bool:
        """Check if RunPod API key is configured."""
        return bool(self.runpod_api_key)

    async def check_runpod_health(self) -> dict:
        """Check if RunPod is configured.

        Returns:
            Health status dict with 'configured' and 'available'.
        """
        if not self.is_runpod_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_API_KEY not configured",
            }

        return {
            "configured": True,
            "available": True,
            "models": list(RUNPOD_ENDPOINTS.keys()),
        }

    async def generate_runpod(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using RunPod public endpoints.

        Args:
            request: The generation request (must use a RUNPOD_* model)

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationServiceError: If generation fails
        """
        if not self.is_runpod_configured():
            raise ImageGenerationServiceError(
                "RUNPOD_API_KEY not configured. Set it in your .env file."
            )

        model = request.model
        if model not in RUNPOD_ENDPOINTS:
            raise ImageGenerationServiceError(
                f"Model {model.value} is not a RunPod model. "
                f"Use one of: {list(RUNPOD_ENDPOINTS.keys())}"
            )

        endpoint_slug = RUNPOD_ENDPOINTS[model]
        url = f"{RUNPOD_API_BASE}/{endpoint_slug}/runsync"

        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        # Build RunPod input payload
        # Note: RunPod public Flux endpoints only accept prompt (and optionally seed)
        payload = {
            "input": {
                "prompt": request.prompt,
            }
        }

        if request.seed is not None:
            payload["input"]["seed"] = request.seed

        logger.info(
            f"Generating {request.num_images} image(s) with RunPod {model.value} "
            f"({request.width}x{request.height})"
        )

        start_time = time.time()

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            # Check for RunPod execution errors
            if result_data.get("status") == "FAILED":
                error_msg = result_data.get("error", "Unknown error")
                raise ImageGenerationServiceError(f"RunPod execution failed: {error_msg}")

            # Parse RunPod response
            output = result_data.get("output", {})
            if "error" in output:
                raise ImageGenerationServiceError(f"Generation error: {output['error']}")

            images = []

            # Handle different response formats
            # Format 1: images array
            for img_data in output.get("images", []):
                url = img_data.get("url", "") if isinstance(img_data, dict) else img_data
                images.append(
                    GeneratedImage(
                        url=url,
                        width=request.width,
                        height=request.height,
                        content_type="image/jpeg",
                        seed=output.get("seed"),
                    )
                )

            # Format 2: single image_url (RunPod public endpoints)
            if not images and output.get("image_url"):
                images.append(
                    GeneratedImage(
                        url=output.get("image_url"),
                        width=request.width,
                        height=request.height,
                        content_type="image/jpeg",
                        seed=output.get("seed"),
                    )
                )

            # Format 3: single image object
            if not images and output.get("image"):
                img = output.get("image")
                url = img.get("url", img) if isinstance(img, dict) else img
                images.append(
                    GeneratedImage(
                        url=url,
                        width=request.width,
                        height=request.height,
                        content_type="image/jpeg",
                        seed=output.get("seed"),
                    )
                )

            # Use actual cost from response if available, otherwise estimate
            cost = output.get("cost", 0.0)
            if not cost:
                cost = self._calculate_cost(
                    model, request.width, request.height, len(images) or 1
                )

            logger.info(
                f"RunPod generated {len(images)} image(s) in {generation_time_ms}ms "
                f"(cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=model,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=cost,
            )

        except httpx.TimeoutException:
            raise ImageGenerationServiceError(
                "RunPod request timed out. The endpoint may be experiencing a cold start. "
                "Try again in a few seconds."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"RunPod API error: {error_detail}")
        except ImageGenerationServiceError:
            raise
        except Exception as e:
            raise ImageGenerationServiceError(f"RunPod image generation failed: {e}")

    async def edit_runpod(self, request: ImageEditRequest) -> ImageGenerationResult:
        """Edit an image using RunPod Flux Kontext.

        Args:
            request: The edit request (model should be RUNPOD_FLUX_KONTEXT)

        Returns:
            ImageGenerationResult with edited images

        Raises:
            ImageGenerationServiceError: If editing fails
        """
        if not self.is_runpod_configured():
            raise ImageGenerationServiceError(
                "RUNPOD_API_KEY not configured. Set it in your .env file."
            )

        # Use Flux Kontext for editing
        endpoint_slug = RUNPOD_ENDPOINTS[ImageGenerationModel.RUNPOD_FLUX_KONTEXT]
        url = f"{RUNPOD_API_BASE}/{endpoint_slug}/runsync"

        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": {
                "prompt": request.prompt,
                "image_url": request.input_image_url,
            }
        }

        if request.seed is not None:
            payload["input"]["seed"] = request.seed

        logger.info(f"Editing image with RunPod Flux Kontext")

        start_time = time.time()

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            if result_data.get("status") == "FAILED":
                error_msg = result_data.get("error", "Unknown error")
                raise ImageGenerationServiceError(f"RunPod execution failed: {error_msg}")

            output = result_data.get("output", {})
            if "error" in output:
                raise ImageGenerationServiceError(f"Edit error: {output['error']}")

            images = []
            for img_data in output.get("images", []):
                images.append(
                    GeneratedImage(
                        url=img_data.get("url", ""),
                        width=1024,
                        height=1024,
                        content_type="image/png",
                        seed=output.get("seed"),
                    )
                )

            cost = self._calculate_cost(
                ImageGenerationModel.RUNPOD_FLUX_KONTEXT, 1024, 1024, len(images) or 1
            )

            logger.info(
                f"RunPod edited {len(images)} image(s) in {generation_time_ms}ms "
                f"(est. cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=ImageGenerationModel.RUNPOD_FLUX_KONTEXT,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=cost,
            )

        except httpx.TimeoutException:
            raise ImageGenerationServiceError(
                "RunPod request timed out. Try again in a few seconds."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"RunPod API error: {error_detail}")
        except ImageGenerationServiceError:
            raise
        except Exception as e:
            raise ImageGenerationServiceError(f"RunPod image editing failed: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
