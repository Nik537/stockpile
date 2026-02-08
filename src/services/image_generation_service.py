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

# Gemini (Google) API configuration - FREE 500/day
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Replicate API configuration
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "")
REPLICATE_API_BASE = "https://api.replicate.com/v1"

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
    # Qwen models (best text rendering in images)
    ImageGenerationModel.RUNPOD_QWEN_IMAGE: "qwen-image-t2i",
    ImageGenerationModel.RUNPOD_QWEN_IMAGE_LORA: "qwen-image-t2i-lora",
    ImageGenerationModel.RUNPOD_QWEN_IMAGE_EDIT: "qwen-image-edit",
    # Seedream models
    ImageGenerationModel.RUNPOD_SEEDREAM_3: "seedream-3-0-t2i",
    ImageGenerationModel.RUNPOD_SEEDREAM_4: "seedream-v4-t2i",
}

# Pricing per megapixel (approximate)
PRICING_PER_MP = {
    ImageGenerationModel.FLUX_KLEIN: 0.012,  # ~$0.009-0.014/MP
    ImageGenerationModel.Z_IMAGE: 0.005,  # ~$0.005/MP (CHEAPEST fal.ai)
    ImageGenerationModel.RUNPOD_FLUX_DEV: 0.02,  # $0.02/MP
    ImageGenerationModel.RUNPOD_FLUX_SCHNELL: 0.0024,  # $0.0024/MP (CHEAPEST overall)
    ImageGenerationModel.RUNPOD_FLUX_KONTEXT: 0.02,  # ~$0.02/MP
    # Qwen models - best for text in images
    ImageGenerationModel.RUNPOD_QWEN_IMAGE: 0.02,  # $0.02/MP
    ImageGenerationModel.RUNPOD_QWEN_IMAGE_LORA: 0.02,  # $0.02/MP
    ImageGenerationModel.RUNPOD_QWEN_IMAGE_EDIT: 0.02,  # $0.02/MP
    # Seedream models
    ImageGenerationModel.RUNPOD_SEEDREAM_3: 0.03,  # $0.03/MP
    ImageGenerationModel.RUNPOD_SEEDREAM_4: 0.027,  # $0.027/MP
    # Gemini - FREE 500/day, then $0.039/image
    ImageGenerationModel.GEMINI_FLASH: 0.0,  # FREE (500/day quota)
    # Replicate Flux Klein - fast, ~$0.003/image
    ImageGenerationModel.REPLICATE_FLUX_KLEIN: 0.003,  # ~$0.003/image flat
}

# Aspect ratio mapping for Gemini/Replicate (uses ratios, not pixels)
ASPECT_RATIO_MAP = {
    (1, 1): "1:1",
    (16, 9): "16:9",
    (9, 16): "9:16",
    (4, 3): "4:3",
    (3, 4): "3:4",
    (3, 2): "3:2",
    (2, 3): "2:3",
    (21, 9): "21:9",
    (9, 21): "9:21",
    (5, 4): "5:4",
    (4, 5): "4:5",
    (7, 3): "21:9",  # Alternate form of 21:9
    (3, 7): "9:21",  # Alternate form of 9:21
}


def get_aspect_ratio(width: int, height: int) -> str:
    """Convert pixel dimensions to aspect ratio string for Gemini/Replicate."""
    from math import gcd

    # First: try exact match via GCD reduction
    divisor = gcd(width, height)
    ratio = (width // divisor, height // divisor)

    if ratio in ASPECT_RATIO_MAP:
        return ASPECT_RATIO_MAP[ratio]

    # Second: find closest matching ratio
    actual_ratio = width / height
    best_match = None
    best_diff = float('inf')

    for (w, h), ar in ASPECT_RATIO_MAP.items():
        diff = abs(actual_ratio - w / h)
        if diff < best_diff:
            best_diff = diff
            best_match = ar

    # Accept if within 5% tolerance
    if best_diff < 0.05 and best_match:
        return best_match

    logger.warning(
        f"No matching aspect ratio for {width}x{height}. Defaulting to 1:1"
    )
    return "1:1"


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
        # RunPod Flux endpoints accept: prompt, width, height, seed, num_inference_steps, guidance
        # Width/height must be in range 256-1536
        width = max(256, min(1536, request.width))
        height = max(256, min(1536, request.height))

        payload = {
            "input": {
                "prompt": request.prompt,
                "width": width,
                "height": height,
            }
        }

        if request.seed is not None:
            payload["input"]["seed"] = request.seed

        if request.guidance_scale is not None:
            payload["input"]["guidance"] = request.guidance_scale

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

            # Format 3: single image object or base64 string (output.image)
            if not images and output.get("image"):
                img = output.get("image")
                if isinstance(img, dict):
                    url = img.get("url", "")
                elif isinstance(img, str):
                    # Check if it's already a URL or base64 data
                    if img.startswith("http"):
                        url = img
                    else:
                        # Base64 encoded image data - convert to data URL
                        url = f"data:image/png;base64,{img}"
                else:
                    url = str(img)
                images.append(
                    GeneratedImage(
                        url=url,
                        width=request.width,
                        height=request.height,
                        content_type="image/png",
                        seed=output.get("seed"),
                    )
                )

            # Format 4: result field (Qwen endpoints use this)
            if not images and output.get("result"):
                result_url = output.get("result")
                if isinstance(result_url, str) and result_url.startswith("http"):
                    images.append(
                        GeneratedImage(
                            url=result_url,
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

    # =========================================================================
    # Gemini (Google) - FREE 500/day
    # =========================================================================

    def is_gemini_configured(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(GEMINI_API_KEY)

    async def check_gemini_health(self) -> dict:
        """Check if Gemini is configured."""
        if not self.is_gemini_configured():
            return {
                "configured": False,
                "available": False,
                "error": "GEMINI_API_KEY not configured",
            }
        return {
            "configured": True,
            "available": True,
            "free_quota": "500 images/day",
        }

    async def generate_gemini(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Gemini 2.5 Flash (FREE 500/day).

        Args:
            request: The generation request

        Returns:
            ImageGenerationResult with generated images
        """
        if not self.is_gemini_configured():
            raise ImageGenerationServiceError(
                "GEMINI_API_KEY not configured. Set it in your .env file."
            )

        url = f"{GEMINI_API_BASE}/models/gemini-2.5-flash-image:generateContent"

        headers = {
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        }

        # Convert dimensions to aspect ratio
        aspect_ratio = get_aspect_ratio(request.width, request.height)

        payload = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio},
            },
        }

        logger.info(
            f"Generating image with Gemini Flash (aspect={aspect_ratio})"
        )

        start_time = time.time()

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            images = []
            candidates = result_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    inline_data = part.get("inlineData", {})
                    if inline_data.get("data"):
                        # Convert base64 to data URL
                        mime_type = inline_data.get("mimeType", "image/jpeg")
                        data_url = f"data:{mime_type};base64,{inline_data['data']}"
                        images.append(
                            GeneratedImage(
                                url=data_url,
                                width=request.width,
                                height=request.height,
                                content_type=mime_type,
                                seed=None,
                            )
                        )

            logger.info(
                f"Gemini generated {len(images)} image(s) in {generation_time_ms}ms (FREE)"
            )

            return ImageGenerationResult(
                images=images,
                model=ImageGenerationModel.GEMINI_FLASH,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=0.0,  # FREE
            )

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"Gemini API error: {error_detail}")
        except Exception as e:
            raise ImageGenerationServiceError(f"Gemini image generation failed: {e}")

    # =========================================================================
    # Replicate - Flux Klein (fast, ~$0.003/image)
    # =========================================================================

    def is_replicate_configured(self) -> bool:
        """Check if Replicate API key is configured."""
        return bool(REPLICATE_API_KEY)

    async def check_replicate_health(self) -> dict:
        """Check if Replicate is configured."""
        if not self.is_replicate_configured():
            return {
                "configured": False,
                "available": False,
                "error": "REPLICATE_API_KEY not configured",
            }
        return {
            "configured": True,
            "available": True,
            "models": ["flux-klein"],
        }

    async def generate_replicate(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Replicate Flux Klein.

        Args:
            request: The generation request

        Returns:
            ImageGenerationResult with generated images
        """
        if not self.is_replicate_configured():
            raise ImageGenerationServiceError(
                "REPLICATE_API_KEY not configured. Set it in your .env file."
            )

        url = f"{REPLICATE_API_BASE}/predictions"

        headers = {
            "Authorization": f"Bearer {REPLICATE_API_KEY}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # Synchronous mode
        }

        # Convert dimensions to aspect ratio
        aspect_ratio = get_aspect_ratio(request.width, request.height)

        # Determine megapixels from dimensions
        total_pixels = request.width * request.height
        if total_pixels <= 250000:
            output_mp = "0.25"
        elif total_pixels <= 500000:
            output_mp = "0.5"
        elif total_pixels <= 1000000:
            output_mp = "1"
        elif total_pixels <= 2000000:
            output_mp = "2"
        else:
            output_mp = "4"

        payload = {
            "version": "black-forest-labs/flux-2-klein-4b",
            "input": {
                "prompt": request.prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "jpg",
                "output_megapixels": output_mp,
                "output_quality": 95,
            },
        }

        if request.seed is not None:
            payload["input"]["seed"] = request.seed

        logger.info(
            f"Generating image with Replicate Flux Klein (aspect={aspect_ratio}, mp={output_mp})"
        )

        start_time = time.time()

        try:
            response = await self.client.post(
                url, headers=headers, json=payload, timeout=120.0
            )
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            # Check for errors
            if result_data.get("status") == "failed":
                error_msg = result_data.get("error", "Unknown error")
                raise ImageGenerationServiceError(f"Replicate failed: {error_msg}")

            images = []
            output = result_data.get("output", [])
            if isinstance(output, list):
                for img_url in output:
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        images.append(
                            GeneratedImage(
                                url=img_url,
                                width=request.width,
                                height=request.height,
                                content_type="image/jpeg",
                                seed=None,
                            )
                        )

            # Estimate cost based on actual predict_time if available
            predict_time = result_data.get("metrics", {}).get("predict_time", 2.0)
            cost = predict_time * 0.001525  # $0.001525/second

            logger.info(
                f"Replicate generated {len(images)} image(s) in {generation_time_ms}ms "
                f"(cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=ImageGenerationModel.REPLICATE_FLUX_KLEIN,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=cost,
            )

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"Replicate API error: {error_detail}")
        except Exception as e:
            raise ImageGenerationServiceError(f"Replicate image generation failed: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
