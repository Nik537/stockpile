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
MAX_CONCURRENT_GEMINI = 5  # Gemini has 15 RPM limit


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
        self.semaphore_default = asyncio.Semaphore(MAX_CONCURRENT_DEFAULT)
        self.semaphore_gemini = asyncio.Semaphore(MAX_CONCURRENT_GEMINI)

    def generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())

    # Model string to enum mapping
    MODEL_MAP = {
        "runware-flux-klein-4b": ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B,
        "runware-flux-klein-9b": ImageGenerationModel.RUNWARE_FLUX_KLEIN_9B,
        "runware-z-image": ImageGenerationModel.RUNWARE_Z_IMAGE,
        "gemini-flash": ImageGenerationModel.GEMINI_FLASH,
        "nano-banana-pro": ImageGenerationModel.NANO_BANANA_PRO,
    }

    # Provider routing
    RUNWARE_MODELS = {"runware-flux-klein-4b", "runware-flux-klein-9b", "runware-z-image"}
    GEMINI_MODELS = {"gemini-flash"}
    RUNPOD_MODELS = {"nano-banana-pro"}

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
            model: Model name (e.g., "runware-flux-klein-4b")
            width: Image width
            height: Image height

        Returns:
            Estimated cost in USD
        """
        model_enum = self.MODEL_MAP.get(model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B)
        price = PRICING_PER_MP.get(model_enum, 0.0006)
        return count * price

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
        fast_models = {"runware-flux-klein-4b", "runware-z-image"}
        time_per_image = 3 if model in fast_models else 6

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
        raw_prompts = self.ai_service.generate_bulk_image_prompts(meta_prompt, count)

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
        if model in self.GEMINI_MODELS:
            semaphore = self.semaphore_gemini
        else:
            semaphore = self.semaphore_default

        async with semaphore:
            start_time = time.time()

            try:
                model_enum = self.MODEL_MAP.get(
                    model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B
                )

                request = ImageGenerationRequest(
                    prompt=prompt.prompt,
                    model=model_enum,
                    width=width,
                    height=height,
                    num_images=1,
                )

                # Route to correct provider
                if model in self.RUNWARE_MODELS:
                    result = await self.image_gen_service.generate_runware(request)
                elif model in self.GEMINI_MODELS:
                    result = await self.image_gen_service.generate_gemini(request)
                elif model in self.RUNPOD_MODELS:
                    result = await self.image_gen_service.generate_runpod(request)
                else:
                    result = await self.image_gen_service.generate_image(request)

                generation_time_ms = int((time.time() - start_time) * 1000)

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

        total_cost = 0.0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            job.results.append(result)

            if result.status == "completed":
                job.completed_count += 1
                model_enum = self.MODEL_MAP.get(
                    job.model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B
                )
                price = PRICING_PER_MP.get(model_enum, 0.0006)
                total_cost += price
            else:
                job.failed_count += 1

            job.total_cost = total_cost

            if on_progress:
                try:
                    await on_progress(result)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        job.results.sort(key=lambda r: r.index)

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
