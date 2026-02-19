"""Image Generation Service - Runware, Gemini, Nano Banana Pro, and Qwen Image providers."""

import json
import logging
import os
import time
import uuid
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

# Runware API configuration
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY", "")
RUNWARE_API_BASE = "https://api.runware.ai/v1"

# RunPod API configuration (for Nano Banana Pro)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# Gemini (Google) API configuration - FREE 500/day
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# RunPod Qwen Image endpoint (custom endpoint overrides public)
RUNPOD_QWEN_IMAGE_ENDPOINT_ID = os.getenv("RUNPOD_QWEN_IMAGE_ENDPOINT_ID", "")
RUNPOD_QWEN_IMAGE_PUBLIC_SLUG = "qwen-image-t2i"

# RunPod public endpoint slugs
RUNPOD_ENDPOINTS = {
    ImageGenerationModel.NANO_BANANA_PRO: "nano-banana-pro-edit",
    ImageGenerationModel.QWEN_IMAGE: RUNPOD_QWEN_IMAGE_ENDPOINT_ID or RUNPOD_QWEN_IMAGE_PUBLIC_SLUG,
}

# Runware model ID mapping
RUNWARE_MODEL_IDS = {
    ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B: "runware:100@1",
    ImageGenerationModel.RUNWARE_FLUX_KLEIN_9B: "runware:101@1",
    ImageGenerationModel.RUNWARE_Z_IMAGE: "civitai:981927@1105492",
}

# Flat pricing per image (not per-megapixel for Runware)
PRICING_PER_MP = {
    ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B: 0.0006,
    ImageGenerationModel.RUNWARE_FLUX_KLEIN_9B: 0.00078,
    ImageGenerationModel.RUNWARE_Z_IMAGE: 0.0006,
    ImageGenerationModel.GEMINI_FLASH: 0.0,
    ImageGenerationModel.NANO_BANANA_PRO: 0.04,
    ImageGenerationModel.QWEN_IMAGE: 0.02,
}

# Aspect ratio mapping for Gemini (uses ratios, not pixels)
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
    """Convert pixel dimensions to aspect ratio string for Gemini."""
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
    """Image generation service supporting Runware, Gemini, and Nano Banana Pro."""

    def __init__(
        self,
        api_key: str = "",
        runpod_api_key: str = "",
        runware_api_key: str = "",
    ):
        """Initialize the image generation service.

        Args:
            api_key: Legacy parameter (unused, kept for backward compatibility).
            runpod_api_key: RunPod API key for Nano Banana Pro.
            runware_api_key: Runware API key for Flux Klein and Z-Image models.
        """
        self.runpod_api_key = runpod_api_key or RUNPOD_API_KEY
        self.runware_api_key = runware_api_key or RUNWARE_API_KEY
        self.default_model = self._load_settings().get(
            "default_model", ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B.value
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

    def _calculate_cost(
        self, model: ImageGenerationModel, width: int, height: int, num_images: int
    ) -> float:
        """Calculate estimated cost for image generation.

        For Runware models, pricing is flat per image (not per-megapixel).
        For other models, uses the stored rate.

        Args:
            model: The model being used
            width: Image width in pixels
            height: Image height in pixels
            num_images: Number of images to generate

        Returns:
            Estimated cost in USD
        """
        price = PRICING_PER_MP.get(model, 0.001)
        return num_images * price

    # =========================================================================
    # Runware API (Flux Klein 4B/9B, Z-Image)
    # =========================================================================

    def is_runware_configured(self) -> bool:
        """Check if Runware API key is configured."""
        return bool(self.runware_api_key)

    async def check_runware_health(self) -> dict:
        """Check if Runware is configured."""
        if not self.is_runware_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RUNWARE_API_KEY not configured",
            }
        return {
            "configured": True,
            "available": True,
            "models": [
                ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B.value,
                ImageGenerationModel.RUNWARE_FLUX_KLEIN_9B.value,
                ImageGenerationModel.RUNWARE_Z_IMAGE.value,
            ],
        }

    async def generate_runware(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Runware API.

        Args:
            request: The generation request (must use a Runware model)

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationServiceError: If generation fails
        """
        if not self.is_runware_configured():
            raise ImageGenerationServiceError(
                "RUNWARE_API_KEY not configured. Set it in your .env file."
            )

        model = request.model
        if model not in RUNWARE_MODEL_IDS:
            raise ImageGenerationServiceError(
                f"Model {model.value} is not a Runware model. "
                f"Use one of: {list(RUNWARE_MODEL_IDS.keys())}"
            )

        model_id = RUNWARE_MODEL_IDS[model]
        task_uuid = str(uuid.uuid4())

        headers = {
            "Authorization": f"Bearer {self.runware_api_key}",
            "Content-Type": "application/json",
        }

        payload = [
            {
                "taskType": "imageInference",
                "taskUUID": task_uuid,
                "positivePrompt": request.prompt,
                "model": model_id,
                "width": request.width,
                "height": request.height,
                "numberResults": request.num_images,
            }
        ]

        if request.seed is not None:
            payload[0]["seed"] = request.seed

        logger.info(
            f"Generating {request.num_images} image(s) with Runware {model.value} "
            f"({request.width}x{request.height})"
        )

        start_time = time.time()

        try:
            response = await self.client.post(
                RUNWARE_API_BASE, headers=headers, json=payload
            )
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            images = []
            data_items = result_data.get("data", [])
            for item in data_items:
                image_url = item.get("imageURL", "")
                if image_url:
                    images.append(
                        GeneratedImage(
                            url=image_url,
                            width=request.width,
                            height=request.height,
                            content_type="image/png",
                            seed=item.get("seed"),
                        )
                    )

            cost = self._calculate_cost(
                model, request.width, request.height, len(images) or 1
            )

            logger.info(
                f"Runware generated {len(images)} image(s) in {generation_time_ms}ms "
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
                "Runware request timed out. Try again in a few seconds."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = json.dumps(error_data)
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(f"Runware API error: {error_detail}")
        except ImageGenerationServiceError:
            raise
        except Exception as e:
            raise ImageGenerationServiceError(f"Runware image generation failed: {e}")

    # =========================================================================
    # RunPod - Nano Banana Pro (generation and editing)
    # =========================================================================

    def is_runpod_configured(self) -> bool:
        """Check if RunPod API key is configured."""
        return bool(self.runpod_api_key)

    async def check_runpod_health(self) -> dict:
        """Check if RunPod is configured."""
        if not self.is_runpod_configured():
            return {
                "configured": False,
                "available": False,
                "error": "RUNPOD_API_KEY not configured",
            }
        return {
            "configured": True,
            "available": True,
            "models": [
                ImageGenerationModel.NANO_BANANA_PRO.value,
                ImageGenerationModel.QWEN_IMAGE.value,
            ],
        }

    @staticmethod
    def _resolution_from_dimensions(width: int, height: int) -> str:
        """Convert pixel dimensions to RunPod resolution string (1k/2k/4k)."""
        max_dim = max(width, height)
        if max_dim <= 1024:
            return "1k"
        elif max_dim <= 2048:
            return "2k"
        else:
            return "4k"

    async def generate_runpod(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using RunPod Nano Banana Pro.

        Note: nano-banana-pro-edit is an editing endpoint and does not support
        pure text-to-image generation. This will raise an error.

        Args:
            request: The generation request

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationServiceError: If generation fails
        """
        raise ImageGenerationServiceError(
            "Nano Banana Pro only supports image editing, not text-to-image generation. "
            "Use a Runware or Gemini model for text-to-image."
        )

    async def edit_runpod(self, request: ImageEditRequest) -> ImageGenerationResult:
        """Edit an image using RunPod Nano Banana Pro.

        API format:
          input.images: [url_string] - array of source image URLs
          input.prompt: str - edit description
          input.resolution: "1k"|"2k"|"4k"
          output.result: url_string - edited image URL
          output.cost: float

        Args:
            request: The edit request

        Returns:
            ImageGenerationResult with edited images

        Raises:
            ImageGenerationServiceError: If editing fails
        """
        if not self.is_runpod_configured():
            raise ImageGenerationServiceError(
                "RUNPOD_API_KEY not configured. Set it in your .env file."
            )

        endpoint_slug = RUNPOD_ENDPOINTS[ImageGenerationModel.NANO_BANANA_PRO]
        url = f"{RUNPOD_API_BASE}/{endpoint_slug}/runsync"

        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        resolution = self._resolution_from_dimensions(1024, 1024)

        payload = {
            "input": {
                "prompt": request.prompt,
                "images": [request.input_image_url],
                "resolution": resolution,
            }
        }

        if request.seed is not None:
            payload["input"]["seed"] = request.seed

        if request.mask_image:
            payload["input"]["mask_image"] = request.mask_image

        if request.strength is not None:
            payload["input"]["strength"] = request.strength

        logger.info("Editing image with RunPod Nano Banana Pro")

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
            if isinstance(output, dict) and output.get("status") == "error":
                raise ImageGenerationServiceError(f"Edit error: {output}")

            images = []

            # Primary response format: output.result is a URL string
            if output.get("result"):
                result_url = output["result"]
                if isinstance(result_url, str) and result_url.startswith("http"):
                    images.append(
                        GeneratedImage(
                            url=result_url,
                            width=1024,
                            height=1024,
                            content_type="image/jpeg",
                            seed=output.get("seed"),
                        )
                    )

            # Fallback: check images array
            if not images:
                for img_data in output.get("images", []):
                    img_url = img_data.get("url", "") if isinstance(img_data, dict) else img_data
                    if img_url:
                        images.append(
                            GeneratedImage(
                                url=img_url,
                                width=1024,
                                height=1024,
                                content_type="image/jpeg",
                                seed=output.get("seed"),
                            )
                        )

            # Fallback: image_url
            if not images and output.get("image_url"):
                images.append(
                    GeneratedImage(
                        url=output["image_url"],
                        width=1024,
                        height=1024,
                        content_type="image/jpeg",
                        seed=output.get("seed"),
                    )
                )

            cost = output.get("cost", 0.0)
            if not cost:
                cost = self._calculate_cost(
                    ImageGenerationModel.NANO_BANANA_PRO, 1024, 1024, len(images) or 1
                )

            logger.info(
                f"RunPod edited {len(images)} image(s) in {generation_time_ms}ms "
                f"(cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=ImageGenerationModel.NANO_BANANA_PRO,
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
    # RunPod - Qwen Image (text-to-image generation)
    # =========================================================================

    @staticmethod
    def _qwen_size_from_dimensions(width: int, height: int) -> str:
        """Map pixel dimensions to Qwen Image public endpoint size presets.

        The public qwen-image-t2i endpoint accepts three preset sizes:
        1328x1328 (square), 1472x1140 (landscape), 1140x1472 (portrait).
        """
        ratio = width / height
        if ratio > 1.15:
            return "1472x1140"
        elif ratio < 0.87:
            return "1140x1472"
        return "1328x1328"

    async def generate_qwen_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Generate images using Qwen Image via RunPod.

        Supports both the public qwen-image-t2i endpoint and custom serverless
        endpoints deployed from the Qwen-Image worker.

        Args:
            request: The generation request

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationServiceError: If generation fails
        """
        if not self.is_runpod_configured():
            raise ImageGenerationServiceError(
                "RUNPOD_API_KEY not configured. Set it in your .env file."
            )

        endpoint_slug = RUNPOD_ENDPOINTS[ImageGenerationModel.QWEN_IMAGE]
        is_custom = bool(RUNPOD_QWEN_IMAGE_ENDPOINT_ID)
        url = f"{RUNPOD_API_BASE}/{endpoint_slug}/runsync"

        headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json",
        }

        # Build payload based on endpoint type
        input_payload: dict = {"prompt": request.prompt}

        if is_custom:
            # Custom endpoint accepts width/height and extra params
            input_payload["width"] = request.width
            input_payload["height"] = request.height
            input_payload["negative_prompt"] = ""
            input_payload["true_cfg_scale"] = request.guidance_scale
        else:
            # Public endpoint uses preset size strings
            input_payload["size"] = self._qwen_size_from_dimensions(
                request.width, request.height
            )

        if request.seed is not None:
            input_payload["seed"] = request.seed

        payload = {"input": input_payload}

        logger.info(
            f"Generating image with Qwen Image ({'custom' if is_custom else 'public'} "
            f"endpoint, {request.width}x{request.height})"
        )

        start_time = time.time()

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result_data = response.json()
            generation_time_ms = int((time.time() - start_time) * 1000)

            if result_data.get("status") == "FAILED":
                error_msg = result_data.get("error", "Unknown error")
                raise ImageGenerationServiceError(
                    f"Qwen Image generation failed: {error_msg}"
                )

            output = result_data.get("output", {})
            images = []

            # Custom endpoint returns base64 image in output.image
            if output.get("image"):
                data_url = f"data:image/png;base64,{output['image']}"
                images.append(
                    GeneratedImage(
                        url=data_url,
                        width=request.width,
                        height=request.height,
                        content_type="image/png",
                        seed=output.get("seed"),
                    )
                )

            # Public endpoint may return image_url or images array
            if not images and output.get("image_url"):
                images.append(
                    GeneratedImage(
                        url=output["image_url"],
                        width=request.width,
                        height=request.height,
                        content_type="image/png",
                        seed=output.get("seed"),
                    )
                )

            if not images:
                for img_data in output.get("images", []):
                    img_url = (
                        img_data.get("url", "")
                        if isinstance(img_data, dict)
                        else img_data
                    )
                    if img_url:
                        images.append(
                            GeneratedImage(
                                url=img_url,
                                width=request.width,
                                height=request.height,
                                content_type="image/png",
                                seed=output.get("seed"),
                            )
                        )

            # Fallback: output.result (some RunPod endpoints use this)
            if not images and output.get("result"):
                result_val = output["result"]
                if isinstance(result_val, str) and result_val.startswith("http"):
                    images.append(
                        GeneratedImage(
                            url=result_val,
                            width=request.width,
                            height=request.height,
                            content_type="image/png",
                            seed=output.get("seed"),
                        )
                    )

            cost = output.get("cost", 0.0)
            if not cost:
                cost = self._calculate_cost(
                    ImageGenerationModel.QWEN_IMAGE,
                    request.width,
                    request.height,
                    len(images) or 1,
                )

            logger.info(
                f"Qwen Image generated {len(images)} image(s) in {generation_time_ms}ms "
                f"(cost: ${cost:.4f})"
            )

            return ImageGenerationResult(
                images=images,
                model=ImageGenerationModel.QWEN_IMAGE,
                prompt=request.prompt,
                generation_time_ms=generation_time_ms,
                cost_estimate=cost,
            )

        except httpx.TimeoutException:
            raise ImageGenerationServiceError(
                "Qwen Image request timed out. The model may need longer "
                "for high-resolution images — try again."
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise ImageGenerationServiceError(
                f"Qwen Image API error: {error_detail}"
            )
        except ImageGenerationServiceError:
            raise
        except Exception as e:
            raise ImageGenerationServiceError(
                f"Qwen Image generation failed: {e}"
            )

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
                cost_estimate=0.0,
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
    # Unified routing methods
    # =========================================================================

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResult:
        """Unified generation - routes to correct provider based on model."""
        model = request.model
        if model in (
            ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B,
            ImageGenerationModel.RUNWARE_FLUX_KLEIN_9B,
            ImageGenerationModel.RUNWARE_Z_IMAGE,
        ):
            return await self.generate_runware(request)
        elif model == ImageGenerationModel.GEMINI_FLASH:
            return await self.generate_gemini(request)
        elif model == ImageGenerationModel.NANO_BANANA_PRO:
            return await self.generate_runpod(request)
        elif model == ImageGenerationModel.QWEN_IMAGE:
            return await self.generate_qwen_image(request)
        else:
            raise ImageGenerationServiceError(f"Unknown model: {model.value}")

    async def edit_image(self, request: ImageEditRequest) -> ImageGenerationResult:
        """Unified edit - routes to correct provider based on model."""
        return await self.edit_runpod(request)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
