"""Bulk Image Generation Service - Orchestrates parallel image generation."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Callable, Optional

from models.bulk_image import BulkImageJob, BulkImagePrompt, BulkImageResult
from models.image_generation import ImageGenerationModel, ImageGenerationRequest
from services.ai_service import AIService
from services.image_generation_service import (
    ImageGenerationService,
    ImageGenerationServiceError,
    PRICING_PER_MP,
)

logger = logging.getLogger(__name__)

# Maximum concurrent image generations by provider
MAX_CONCURRENT_DEFAULT = 10
MAX_CONCURRENT_REPLICATE = 1  # Replicate has strict rate limits for low-credit accounts
MAX_CONCURRENT_GEMINI = 5  # Gemini has 15 RPM limit
REPLICATE_MIN_REQUEST_INTERVAL = 11  # 6 RPM = 10s between requests, add 1s buffer


class BulkImageService:
    """Service for bulk image generation with parallel execution."""

    def __init__(
        self,
        ai_service: AIService,
        image_gen_service: ImageGenerationService,
    ):
        """Initialize the bulk image service.

        Args:
            ai_service: AI service for prompt generation
            image_gen_service: Image generation service for image creation
        """
        self.ai_service = ai_service
        self.image_gen_service = image_gen_service
        # Different semaphores for different providers (rate limit handling)
        self.semaphore_default = asyncio.Semaphore(MAX_CONCURRENT_DEFAULT)
        self.semaphore_replicate = asyncio.Semaphore(MAX_CONCURRENT_REPLICATE)
        self.semaphore_gemini = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)
        # Track last Replicate request time for rate limiting
        self._last_replicate_request_time: float = 0.0

    def generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())

    # Model string to enum mapping
    MODEL_MAP = {
        "runpod-flux-schnell": ImageGenerationModel.RUNPOD_FLUX_SCHNELL,
        "runpod-flux-dev": ImageGenerationModel.RUNPOD_FLUX_DEV,
        "flux-klein": ImageGenerationModel.FLUX_KLEIN,
        "z-image": ImageGenerationModel.Z_IMAGE,
        "runpod-qwen-image": ImageGenerationModel.RUNPOD_QWEN_IMAGE,
        "runpod-qwen-image-lora": ImageGenerationModel.RUNPOD_QWEN_IMAGE_LORA,
        "runpod-qwen-image-edit": ImageGenerationModel.RUNPOD_QWEN_IMAGE_EDIT,
        "runpod-seedream-3": ImageGenerationModel.RUNPOD_SEEDREAM_3,
        "runpod-seedream-4": ImageGenerationModel.RUNPOD_SEEDREAM_4,
        # Gemini (FREE 500/day)
        "gemini-flash": ImageGenerationModel.GEMINI_FLASH,
        # Replicate
        "replicate-flux-klein": ImageGenerationModel.REPLICATE_FLUX_KLEIN,
    }

    # Provider routing
    RUNPOD_MODELS = {
        "runpod-flux-schnell",
        "runpod-flux-dev",
        "runpod-qwen-image",
        "runpod-qwen-image-lora",
        "runpod-qwen-image-edit",
        "runpod-seedream-3",
        "runpod-seedream-4",
    }

    GEMINI_MODELS = {"gemini-flash"}

    REPLICATE_MODELS = {"replicate-flux-klein"}

    def calculate_estimated_cost(
        self,
        count: int,
        model: str,
        width: int,
        height: int,
    ) -> float:
        """Calculate estimated cost for bulk generation.

        Args:
            count: Number of images to generate
            model: Model name (e.g., "runpod-flux-schnell")
            width: Image width
            height: Image height

        Returns:
            Estimated cost in USD
        """
        megapixels = (width * height) / 1_000_000

        # Map model string to enum for pricing lookup
        model_enum = self.MODEL_MAP.get(model, ImageGenerationModel.RUNPOD_FLUX_SCHNELL)
        price_per_mp = PRICING_PER_MP.get(model_enum, 0.0024)
        return megapixels * count * price_per_mp

    def calculate_estimated_time(
        self,
        count: int,
        model: str,
    ) -> int:
        """Calculate estimated time in seconds for bulk generation.

        Args:
            count: Number of images to generate
            model: Model name

        Returns:
            Estimated time in seconds
        """
        # Fast models: schnell, z-image, klein
        fast_models = {"runpod-flux-schnell", "z-image", "flux-klein"}
        time_per_image = 3 if model in fast_models else 6

        # Calculate batches needed
        batches = (count + MAX_CONCURRENT_DEFAULT - 1) // MAX_CONCURRENT_DEFAULT
        return batches * time_per_image

    def generate_prompts(
        self,
        meta_prompt: str,
        count: int,
    ) -> list[BulkImagePrompt]:
        """Generate unique image prompts from a meta-prompt.

        Args:
            meta_prompt: High-level creative concept
            count: Number of prompts to generate

        Returns:
            List of BulkImagePrompt objects
        """
        # Call AI service to generate prompts
        raw_prompts = self.ai_service.generate_bulk_image_prompts(meta_prompt, count)

        # Convert to BulkImagePrompt objects
        prompts = []
        for i, raw in enumerate(raw_prompts):
            prompts.append(
                BulkImagePrompt(
                    index=i,
                    prompt=raw.get("prompt", ""),
                    rendering_style=raw.get("rendering_style", "3d-render"),
                    mood=raw.get("mood", "neutral"),
                    composition=raw.get("composition", "scene"),
                    has_text_space=raw.get("has_text_space", False),
                )
            )

        return prompts

    async def generate_single_image(
        self,
        prompt: BulkImagePrompt,
        model: str,
        width: int,
        height: int,
    ) -> BulkImageResult:
        """Generate a single image with semaphore rate limiting.

        Args:
            prompt: The prompt to generate
            model: Model name
            width: Image width
            height: Image height

        Returns:
            BulkImageResult with success or failure
        """
        # Select appropriate semaphore based on provider rate limits
        if model in self.REPLICATE_MODELS:
            semaphore = self.semaphore_replicate
        elif model in self.GEMINI_MODELS:
            semaphore = self.semaphore_gemini
        else:
            semaphore = self.semaphore_default

        async with semaphore:
            # Enforce rate limiting for Replicate BEFORE the request
            if model in self.REPLICATE_MODELS:
                elapsed = time.time() - self._last_replicate_request_time
                if elapsed < REPLICATE_MIN_REQUEST_INTERVAL:
                    wait_time = REPLICATE_MIN_REQUEST_INTERVAL - elapsed
                    logger.debug(f"Rate limiting: waiting {wait_time:.1f}s before Replicate request")
                    await asyncio.sleep(wait_time)
                # Update timestamp BEFORE request (so failures also count)
                self._last_replicate_request_time = time.time()

            start_time = time.time()

            try:
                # Map model string to enum
                model_enum = self.MODEL_MAP.get(
                    model, ImageGenerationModel.RUNPOD_FLUX_SCHNELL
                )

                request = ImageGenerationRequest(
                    prompt=prompt.prompt,
                    model=model_enum,
                    width=width,
                    height=height,
                    num_images=1,
                )

                # Route to correct provider
                if model in self.RUNPOD_MODELS:
                    result = await self.image_gen_service.generate_runpod(request)
                elif model in self.GEMINI_MODELS:
                    result = await self.image_gen_service.generate_gemini(request)
                elif model in self.REPLICATE_MODELS:
                    result = await self.image_gen_service.generate_replicate(request)
                else:
                    # fal.ai models (flux-klein, z-image)
                    result = await self.image_gen_service.generate(request)

                generation_time_ms = int((time.time() - start_time) * 1000)

                # Get the image URL
                image_url = None
                if result.images:
                    image_url = result.images[0].url

                return BulkImageResult(
                    index=prompt.index,
                    prompt=prompt,
                    image_url=image_url,
                    width=width,
                    height=height,
                    generation_time_ms=generation_time_ms,
                    status="completed",
                )

            except ImageGenerationServiceError as e:
                generation_time_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Image generation failed for prompt {prompt.index}: {e}")
                return BulkImageResult(
                    index=prompt.index,
                    prompt=prompt,
                    image_url=None,
                    width=width,
                    height=height,
                    generation_time_ms=generation_time_ms,
                    status="failed",
                    error=str(e),
                )
            except Exception as e:
                generation_time_ms = int((time.time() - start_time) * 1000)
                logger.error(f"Unexpected error generating image {prompt.index}: {e}")
                return BulkImageResult(
                    index=prompt.index,
                    prompt=prompt,
                    image_url=None,
                    width=width,
                    height=height,
                    generation_time_ms=generation_time_ms,
                    status="failed",
                    error=str(e),
                )

    async def generate_all_images(
        self,
        job: BulkImageJob,
        on_progress: Optional[Callable] = None,
    ) -> BulkImageJob:
        """Generate all images for a bulk job with parallel execution.

        Args:
            job: The bulk image job with prompts
            on_progress: Optional callback for progress updates

        Returns:
            Updated job with results
        """
        job.status = "generating_images"

        # Create tasks for all prompts
        tasks = []
        for prompt in job.prompts:
            task = asyncio.create_task(
                self.generate_single_image(
                    prompt=prompt,
                    model=job.model,
                    width=job.width,
                    height=job.height,
                )
            )
            tasks.append(task)

        # Process results as they complete
        total_cost = 0.0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            job.results.append(result)

            if result.status == "completed":
                job.completed_count += 1
                # Calculate cost for this image
                megapixels = (result.width * result.height) / 1_000_000
                model_enum = self.MODEL_MAP.get(
                    job.model, ImageGenerationModel.RUNPOD_FLUX_SCHNELL
                )
                price_per_mp = PRICING_PER_MP.get(model_enum, 0.0024)
                total_cost += megapixels * price_per_mp
            else:
                job.failed_count += 1

            job.total_cost = total_cost

            # Call progress callback if provided
            if on_progress:
                try:
                    await on_progress(result)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        # Sort results by index for consistent ordering
        job.results.sort(key=lambda r: r.index)

        # Update final status
        if job.failed_count == job.total_count:
            job.status = "failed"
            job.error = "All image generations failed"
        else:
            job.status = "completed"

        job.completed_at = datetime.now()

        logger.info(
            f"Bulk image job {job.job_id} completed: "
            f"{job.completed_count}/{job.total_count} succeeded, "
            f"total cost: ${job.total_cost:.4f}"
        )

        return job
