"""Dataset Generator Service - Generates LoRA training datasets using cheap image providers.

Supports 4 modes:
- Pair: Generate start image, edit to create end image (transformation pairs)
- Single: Generate individual images with AI captions
- Reference: Upload reference image, generate variations via edit endpoint
- Layered: Generate 2x2 grid, split cells, remove backgrounds, composite
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

import httpx

from models.dataset_generator import (
    DatasetGenerationRequest,
    DatasetItem,
    DatasetJob,
    DatasetMode,
    DatasetStatus,
)
from models.image_generation import (
    ImageEditRequest,
    ImageGenerationModel,
    ImageGenerationRequest,
)
from services.ai_service import AIService
from services.image_generation_service import (
    ImageGenerationService,
    ImageGenerationServiceError,
    PRICING_PER_MP,
)

logger = logging.getLogger(__name__)

# Storage directory for datasets
DATASETS_DIR = Path.home() / ".stockpile" / "datasets"

# Gemini API config (for vision captioning)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Model string to enum mapping
MODEL_MAP = {
    "runware-flux-klein-4b": ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B,
    "runware-flux-klein-9b": ImageGenerationModel.RUNWARE_FLUX_KLEIN_9B,
    "runware-z-image": ImageGenerationModel.RUNWARE_Z_IMAGE,
    "gemini-flash": ImageGenerationModel.GEMINI_FLASH,
    "nano-banana-pro": ImageGenerationModel.NANO_BANANA_PRO,
}

# Default system prompts for each mode (derived from original NanoBananaLoraDatasetGenerator)
DEFAULT_SYSTEM_PROMPTS = {
    DatasetMode.PAIR: """You are a creative AI assistant generating image transformation pairs for LoRA training.
Given a theme, generate creative and diverse prompt pairs where:
- "start_prompt" describes the BEFORE image
- "end_prompt" describes the AFTER image (transformed version)
- The transformation should be: {transformation}
- Action name: {action_name}

Each pair should be visually distinct but clearly related through the transformation.
Be creative and varied - don't repeat similar concepts.

Return a JSON array of objects with keys: start_prompt, end_prompt
Example: [{{"start_prompt": "a calm lake at sunrise", "end_prompt": "a frozen lake with ice crystals at sunrise"}}]""",

    DatasetMode.SINGLE: """You are a creative AI assistant generating diverse image prompts for LoRA training.
Given a theme, generate creative and varied image prompts suitable for training a style/aesthetic LoRA.

Each prompt should:
- Be detailed and descriptive (30-60 words)
- Vary in subject matter, composition, lighting, and mood
- Stay within the given theme
- Be suitable for AI image generation

Return a JSON array of strings (just the prompts).
Example: ["a serene mountain landscape with golden hour lighting, dramatic clouds above snow-capped peaks, photorealistic", "abstract geometric patterns in warm earth tones, flowing organic shapes intertwined with sharp angles, modern art style"]""",

    DatasetMode.REFERENCE: """You are a creative AI assistant generating variation prompts for reference-based LoRA training.
Given a theme and a reference image concept, generate creative edit prompts that describe
interesting variations of the reference image.

Each prompt should:
- Describe a variation or modification of the original
- Keep the core subject/style but change details
- Be suitable for image editing models

Return a JSON array of strings (just the edit prompts).
Example: ["change the background to a sunset beach scene", "add dramatic storm clouds and rain", "convert to watercolor painting style"]""",

    DatasetMode.LAYERED: """You are a creative AI assistant generating 2x2 grid image prompts for layered LoRA training.
Given a use case ({layered_use_case}) and elements description, generate prompts for
2x2 grid images where each cell contains a distinct element that will be separated.

Elements: {elements_description}
Final composite description: {final_image_description}

Each prompt should describe a 2x2 grid where:
- Top-left: first element
- Top-right: second element
- Bottom-left: third element
- Bottom-right: fourth element

Return a JSON array of strings (grid description prompts).
Example: ["2x2 grid: top-left a red car, top-right a blue bicycle, bottom-left a green truck, bottom-right a yellow motorcycle, all on white backgrounds"]""",
}


class DatasetGeneratorService:
    """Service for generating LoRA training datasets."""

    def __init__(
        self,
        ai_service: AIService,
        image_gen_service: ImageGenerationService,
    ):
        self.ai_service = ai_service
        self.image_gen_service = image_gen_service
        self.client = httpx.AsyncClient(timeout=120.0)

    def generate_job_id(self) -> str:
        """Generate a unique job ID."""
        return str(uuid.uuid4())

    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a dataset job."""
        job_dir = DATASETS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _build_system_prompt(self, request: DatasetGenerationRequest) -> str:
        """Build the system prompt for LLM prompt generation."""
        if request.custom_system_prompt:
            return request.custom_system_prompt

        template = DEFAULT_SYSTEM_PROMPTS.get(request.mode, DEFAULT_SYSTEM_PROMPTS[DatasetMode.SINGLE])

        if request.mode == DatasetMode.PAIR:
            return template.format(
                transformation=request.transformation or "generic transformation",
                action_name=request.action_name or "transform",
            )
        elif request.mode == DatasetMode.LAYERED:
            return template.format(
                layered_use_case=request.layered_use_case,
                elements_description=request.elements_description or "various elements",
                final_image_description=request.final_image_description or "composite image",
            )
        return template

    def calculate_estimated_cost(self, request: DatasetGenerationRequest) -> float:
        """Calculate estimated cost for dataset generation."""
        model_enum = MODEL_MAP.get(request.model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B)
        price_per_image = PRICING_PER_MP.get(model_enum, 0.0006)

        if request.mode == DatasetMode.PAIR:
            # 2 images per item (start + end via edit)
            edit_price = PRICING_PER_MP.get(ImageGenerationModel.NANO_BANANA_PRO, 0.04)
            return request.num_items * (price_per_image + edit_price)
        elif request.mode == DatasetMode.LAYERED:
            # 1 grid image per item
            return request.num_items * price_per_image
        else:
            # 1 image per item
            return request.num_items * price_per_image

    # =========================================================================
    # LLM Prompt Generation
    # =========================================================================

    async def generate_prompts(self, request: DatasetGenerationRequest) -> list[dict]:
        """Use Gemini to generate creative prompts based on theme and mode.

        Returns list of prompt dicts appropriate for the mode.
        """
        system_prompt = self._build_system_prompt(request)

        user_message = f"""Theme: {request.theme}
Number of prompts needed: {request.num_items}

{f'Trigger word to include: {request.trigger_word}' if request.trigger_word else ''}

Generate exactly {request.num_items} unique and creative prompts following the system instructions.
Return ONLY valid JSON - no markdown, no explanation."""

        full_prompt = f"{system_prompt}\n\n{user_message}"

        try:
            from google.genai import types
            response = self.ai_service.client.models.generate_content(
                model=self.ai_service.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.8,
                ),
            )

            if not response.text:
                logger.error("Empty response from LLM for prompt generation")
                return []

            # Strip markdown code blocks if present
            from services.prompts import strip_markdown_code_blocks
            text = strip_markdown_code_blocks(response.text)
            prompts_data = json.loads(text)

            if not isinstance(prompts_data, list):
                logger.error("LLM response is not a list")
                return []

            # Normalize to list of dicts
            result = []
            for i, item in enumerate(prompts_data):
                if request.mode == DatasetMode.PAIR:
                    if isinstance(item, dict) and "start_prompt" in item:
                        result.append({
                            "index": i,
                            "start_prompt": item["start_prompt"],
                            "end_prompt": item["end_prompt"],
                        })
                else:
                    prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
                    # Prepend trigger word if specified
                    if request.trigger_word:
                        prompt_text = f"{request.trigger_word} {prompt_text}"
                    result.append({
                        "index": i,
                        "prompt": prompt_text,
                    })

            logger.info(f"Generated {len(result)} prompts for {request.mode.value} mode")
            return result[:request.num_items]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM prompt response: {e}")
            return []
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            return []

    # =========================================================================
    # Image Generation (per mode)
    # =========================================================================

    async def generate_pair_item(
        self,
        prompt_data: dict,
        request: DatasetGenerationRequest,
        job_dir: Path,
    ) -> DatasetItem:
        """Generate a pair (start + end) for pair mode.

        1. Generate start image from start_prompt
        2. Edit start image with end_prompt to create end image
        """
        item = DatasetItem(
            index=prompt_data["index"],
            start_prompt=prompt_data["start_prompt"],
            end_prompt=prompt_data["end_prompt"],
            status="generating",
        )
        start_time = time.time()
        total_cost = 0.0

        try:
            # Step 1: Generate start image
            model_enum = MODEL_MAP.get(request.model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B)
            gen_request = ImageGenerationRequest(
                prompt=prompt_data["start_prompt"],
                model=model_enum,
                width=request.width,
                height=request.height,
                num_images=1,
            )
            start_result = await self.image_gen_service.generate_image(gen_request)

            if not start_result.images:
                raise ImageGenerationServiceError("No start image generated")

            start_url = start_result.images[0].url
            item.start_url = start_url
            total_cost += start_result.cost_estimate

            # Save start image locally
            await self._save_image_from_url(
                start_url, job_dir / f"pair_{item.index:03d}_start.png"
            )

            # Step 2: Edit start image to create end image
            edit_prompt = prompt_data["end_prompt"]
            if request.action_name:
                edit_prompt = f"{request.action_name}: {edit_prompt}"

            edit_request = ImageEditRequest(
                prompt=edit_prompt,
                input_image_url=start_url,
                model=ImageGenerationModel.NANO_BANANA_PRO,
                strength=0.75,
            )
            end_result = await self.image_gen_service.edit_image(edit_request)

            if not end_result.images:
                raise ImageGenerationServiceError("No end image generated")

            end_url = end_result.images[0].url
            item.end_url = end_url
            total_cost += end_result.cost_estimate

            # Save end image locally
            await self._save_image_from_url(
                end_url, job_dir / f"pair_{item.index:03d}_end.png"
            )

            # Step 3: Generate caption for the pair
            if request.use_vision_caption:
                caption = await self.caption_image(start_url)
                if caption:
                    transformation_desc = request.transformation or "transformation"
                    item.caption = f"{caption} [transformation: {transformation_desc}]"
                    # Save caption file
                    caption_path = job_dir / f"pair_{item.index:03d}.txt"
                    caption_path.write_text(item.caption)

            item.status = "completed"
            item.cost = total_cost
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        except Exception as e:
            logger.error(f"Pair generation failed for index {item.index}: {e}")
            item.status = "failed"
            item.error = str(e)
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        return item

    async def generate_single_item(
        self,
        prompt_data: dict,
        request: DatasetGenerationRequest,
        job_dir: Path,
    ) -> DatasetItem:
        """Generate a single image for single mode."""
        item = DatasetItem(
            index=prompt_data["index"],
            prompt=prompt_data["prompt"],
            status="generating",
        )
        start_time = time.time()

        try:
            model_enum = MODEL_MAP.get(request.model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B)
            gen_request = ImageGenerationRequest(
                prompt=prompt_data["prompt"],
                model=model_enum,
                width=request.width,
                height=request.height,
                num_images=1,
            )
            result = await self.image_gen_service.generate_image(gen_request)

            if not result.images:
                raise ImageGenerationServiceError("No image generated")

            image_url = result.images[0].url
            item.image_url = image_url
            item.cost = result.cost_estimate

            # Save image locally
            await self._save_image_from_url(
                image_url, job_dir / f"image_{item.index:03d}.png"
            )

            # Caption with vision model
            if request.use_vision_caption:
                item.status = "captioning"
                caption = await self.caption_image(image_url)
                if caption:
                    if request.trigger_word:
                        item.caption = f"{request.trigger_word}, {caption}"
                    else:
                        item.caption = caption
                else:
                    # Fallback to the generation prompt
                    item.caption = prompt_data["prompt"]
            else:
                item.caption = prompt_data["prompt"]

            # Save caption
            caption_path = job_dir / f"image_{item.index:03d}.txt"
            caption_path.write_text(item.caption)

            item.status = "completed"
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        except Exception as e:
            logger.error(f"Single image generation failed for index {item.index}: {e}")
            item.status = "failed"
            item.error = str(e)
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        return item

    async def generate_reference_item(
        self,
        prompt_data: dict,
        request: DatasetGenerationRequest,
        job_dir: Path,
    ) -> DatasetItem:
        """Generate a variation of the reference image."""
        item = DatasetItem(
            index=prompt_data["index"],
            prompt=prompt_data["prompt"],
            status="generating",
        )
        start_time = time.time()

        try:
            edit_request = ImageEditRequest(
                prompt=prompt_data["prompt"],
                input_image_url=request.reference_image_base64,
                model=ImageGenerationModel.NANO_BANANA_PRO,
                strength=0.65,
            )
            result = await self.image_gen_service.edit_image(edit_request)

            if not result.images:
                raise ImageGenerationServiceError("No image generated from reference")

            image_url = result.images[0].url
            item.image_url = image_url
            item.cost = result.cost_estimate

            # Save image locally
            await self._save_image_from_url(
                image_url, job_dir / f"ref_{item.index:03d}.png"
            )

            # Caption with vision
            if request.use_vision_caption:
                item.status = "captioning"
                caption = await self.caption_image(image_url)
                if caption:
                    if request.trigger_word:
                        item.caption = f"{request.trigger_word}, {caption}"
                    else:
                        item.caption = caption
                else:
                    item.caption = prompt_data["prompt"]
            else:
                item.caption = prompt_data["prompt"]

            caption_path = job_dir / f"ref_{item.index:03d}.txt"
            caption_path.write_text(item.caption)

            item.status = "completed"
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        except Exception as e:
            logger.error(f"Reference variation failed for index {item.index}: {e}")
            item.status = "failed"
            item.error = str(e)
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        return item

    async def generate_layered_item(
        self,
        prompt_data: dict,
        request: DatasetGenerationRequest,
        job_dir: Path,
    ) -> DatasetItem:
        """Generate a 2x2 grid image for layered mode.

        Generates a 2x2 grid, saves it as the full image, and saves the caption.
        Background removal and cell splitting can be done client-side or as a post-step.
        """
        item = DatasetItem(
            index=prompt_data["index"],
            prompt=prompt_data["prompt"],
            status="generating",
        )
        start_time = time.time()

        try:
            model_enum = MODEL_MAP.get(request.model, ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B)
            # Double dimensions for 2x2 grid
            gen_request = ImageGenerationRequest(
                prompt=prompt_data["prompt"],
                model=model_enum,
                width=request.width,
                height=request.height,
                num_images=1,
            )
            result = await self.image_gen_service.generate_image(gen_request)

            if not result.images:
                raise ImageGenerationServiceError("No grid image generated")

            image_url = result.images[0].url
            item.image_url = image_url
            item.cost = result.cost_estimate

            # Save full grid image
            await self._save_image_from_url(
                image_url, job_dir / f"grid_{item.index:03d}.png"
            )

            # Caption
            if request.use_vision_caption:
                item.status = "captioning"
                caption = await self.caption_image(image_url)
                item.caption = caption or prompt_data["prompt"]
            else:
                item.caption = prompt_data["prompt"]

            caption_path = job_dir / f"grid_{item.index:03d}.txt"
            caption_path.write_text(item.caption)

            item.status = "completed"
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        except Exception as e:
            logger.error(f"Layered grid generation failed for index {item.index}: {e}")
            item.status = "failed"
            item.error = str(e)
            item.generation_time_ms = int((time.time() - start_time) * 1000)

        return item

    # =========================================================================
    # Vision Captioning (Gemini Flash - FREE)
    # =========================================================================

    async def caption_image(self, image_url: str) -> Optional[str]:
        """Use Gemini Flash vision to describe an image.

        Args:
            image_url: URL or data:image/... base64 string

        Returns:
            Description string, or None on failure
        """
        api_key = GEMINI_API_KEY or self.ai_service.api_key
        if not api_key:
            logger.warning("No Gemini API key for captioning")
            return None

        url = f"{GEMINI_API_BASE}/models/gemini-2.5-flash:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        # Build image part
        if image_url.startswith("data:"):
            # Base64 data URL
            parts = image_url.split(",", 1)
            mime_match = parts[0].split(":")[1].split(";")[0] if ":" in parts[0] else "image/png"
            image_data = parts[1] if len(parts) > 1 else ""
            image_part = {
                "inlineData": {
                    "mimeType": mime_match,
                    "data": image_data,
                }
            }
        else:
            # HTTP URL - download and convert to base64
            try:
                response = await self.client.get(image_url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "image/png")
                b64_data = base64.b64encode(response.content).decode("utf-8")
                image_part = {
                    "inlineData": {
                        "mimeType": content_type,
                        "data": b64_data,
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to download image for captioning: {e}")
                return None

        payload = {
            "contents": [{
                "parts": [
                    image_part,
                    {"text": "Describe this image in detail for use as a training caption. "
                             "Focus on the subject, style, composition, lighting, colors, and mood. "
                             "Be specific and descriptive in 1-3 sentences. "
                             "Do not start with 'This image shows' or 'The image depicts'."},
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 300,
            },
        }

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result_data = response.json()
            candidates = result_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if part.get("text"):
                        caption = part["text"].strip()
                        logger.debug(f"Caption generated: {caption[:80]}...")
                        return caption

            logger.warning("No caption text in Gemini response")
            return None

        except Exception as e:
            logger.warning(f"Vision captioning failed: {e}")
            return None

    # =========================================================================
    # Image Download Helper
    # =========================================================================

    async def _save_image_from_url(self, image_url: str, save_path: Path) -> bool:
        """Download an image from URL and save to disk.

        Handles both HTTP URLs and data: base64 URLs.
        """
        try:
            if image_url.startswith("data:"):
                # Base64 data URL
                parts = image_url.split(",", 1)
                if len(parts) > 1:
                    image_bytes = base64.b64decode(parts[1])
                    save_path.write_bytes(image_bytes)
                    return True
                return False
            else:
                response = await self.client.get(image_url)
                response.raise_for_status()
                save_path.write_bytes(response.content)
                return True
        except Exception as e:
            logger.warning(f"Failed to save image to {save_path}: {e}")
            return False

    # =========================================================================
    # ZIP Packaging
    # =========================================================================

    async def create_zip(self, job: DatasetJob) -> Optional[str]:
        """Package dataset job into a ZIP file.

        Returns the path to the ZIP file on success, None on failure.
        """
        job_dir = self._get_job_dir(job.job_id)
        zip_path = job_dir / f"dataset_{job.job_id[:8]}.zip"

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add all image and caption files from job directory
                for file_path in sorted(job_dir.iterdir()):
                    if file_path.suffix in (".png", ".jpg", ".jpeg", ".txt") and file_path != zip_path:
                        zf.write(file_path, file_path.name)

                # Add manifest
                manifest = {
                    "job_id": job.job_id,
                    "mode": job.request.mode.value,
                    "theme": job.request.theme,
                    "model": job.request.model,
                    "total_items": job.total_count,
                    "completed": job.completed_count,
                    "failed": job.failed_count,
                    "total_cost": job.total_cost,
                    "items": [item.to_dict() for item in job.items],
                }
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            logger.info(f"Dataset ZIP created: {zip_path} ({zip_path.stat().st_size} bytes)")
            return str(zip_path)

        except Exception as e:
            logger.error(f"Failed to create dataset ZIP: {e}")
            return None

    # =========================================================================
    # Main Orchestration
    # =========================================================================

    async def run_generation(
        self,
        job: DatasetJob,
        progress_callback: Optional[Callable] = None,
    ) -> DatasetJob:
        """Main orchestration method - runs the full dataset generation pipeline.

        1. Generate prompts via LLM
        2. Generate images (with concurrency control)
        3. Caption images (if enabled)
        4. Package as ZIP
        """
        request = job.request
        job_dir = self._get_job_dir(job.job_id)
        semaphore = asyncio.Semaphore(request.max_concurrent)

        try:
            # Phase 1: Generate prompts
            job.status = DatasetStatus.GENERATING_PROMPTS
            if progress_callback:
                await progress_callback({
                    "type": "status",
                    "status": job.status.value,
                    "message": "Generating creative prompts...",
                })

            prompts = await self.generate_prompts(request)

            if not prompts:
                raise Exception("Failed to generate prompts - empty response from LLM")

            job.total_count = len(prompts)
            job.estimated_cost = self.calculate_estimated_cost(request)

            if progress_callback:
                await progress_callback({
                    "type": "prompts_ready",
                    "count": len(prompts),
                    "estimated_cost": job.estimated_cost,
                })

            # Phase 2: Generate images
            job.status = DatasetStatus.GENERATING_IMAGES
            if progress_callback:
                await progress_callback({
                    "type": "status",
                    "status": job.status.value,
                    "message": f"Generating {len(prompts)} images...",
                    "total_count": job.total_count,
                })

            # Select generation function based on mode
            mode_generators = {
                DatasetMode.PAIR: self.generate_pair_item,
                DatasetMode.SINGLE: self.generate_single_item,
                DatasetMode.REFERENCE: self.generate_reference_item,
                DatasetMode.LAYERED: self.generate_layered_item,
            }
            generator_fn = mode_generators[request.mode]

            # Launch all tasks with semaphore control
            async def limited_generate(prompt_data):
                async with semaphore:
                    return await generator_fn(prompt_data, request, job_dir)

            tasks = [
                asyncio.create_task(limited_generate(prompt_data))
                for prompt_data in prompts
            ]

            # Collect results as they complete
            for coro in asyncio.as_completed(tasks):
                # Check for cancellation
                if job.status == DatasetStatus.CANCELLED:
                    for t in tasks:
                        t.cancel()
                    break

                item = await coro
                job.items.append(item)

                if item.status == "completed":
                    job.completed_count += 1
                    job.total_cost += item.cost
                else:
                    job.failed_count += 1

                if progress_callback:
                    await progress_callback({
                        "type": "item_complete",
                        "item": item.to_dict(),
                        "completed_count": job.completed_count,
                        "failed_count": job.failed_count,
                        "total_count": job.total_count,
                        "total_cost": job.total_cost,
                    })

            # Sort items by index
            job.items.sort(key=lambda x: x.index)

            # Phase 3: Package as ZIP
            job.status = DatasetStatus.PACKAGING
            if progress_callback:
                await progress_callback({
                    "type": "status",
                    "status": job.status.value,
                    "message": "Packaging dataset...",
                })

            zip_path = await self.create_zip(job)
            job.zip_path = zip_path

            # Final status
            if job.failed_count == job.total_count:
                job.status = DatasetStatus.FAILED
                job.error = "All image generations failed"
            else:
                job.status = DatasetStatus.COMPLETED

            from datetime import datetime
            job.completed_at = datetime.now()

            if progress_callback:
                await progress_callback({
                    "type": "complete",
                    "status": job.status.value,
                    "completed_count": job.completed_count,
                    "failed_count": job.failed_count,
                    "total_count": job.total_count,
                    "total_cost": job.total_cost,
                    "zip_path": job.zip_path,
                })

            logger.info(
                f"Dataset job {job.job_id} completed: "
                f"{job.completed_count}/{job.total_count} succeeded, "
                f"total cost: ${job.total_cost:.4f}"
            )

        except Exception as e:
            logger.error(f"Dataset generation failed for job {job.job_id}: {e}")
            job.status = DatasetStatus.FAILED
            job.error = str(e)

            if progress_callback:
                await progress_callback({
                    "type": "error",
                    "message": str(e),
                })

        return job

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
