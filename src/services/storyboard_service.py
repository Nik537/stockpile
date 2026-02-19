"""Storyboard generation service - AI planning and image generation orchestration."""

import json
import logging
from typing import Callable

from google.genai import types

from models.image_generation import ImageGenerationModel, ImageGenerationRequest
from models.storyboard import (
    CharacterProfile,
    StoryboardJob,
    StoryboardPlan,
    StoryboardScene,
)
from services.ai_service import AIService
from services.image_generation_service import ImageGenerationService

logger = logging.getLogger(__name__)

# RunPod Flux models have a max dimension constraint (typically 1440px).
# Exceeding it causes "height/width does not meet the constraints" errors.
RUNPOD_FLUX_MAX_DIM = 1440

RUNPOD_FLUX_MODELS = {"flux-dev", "flux-schnell", "flux-kontext"}


def clamp_dimensions_for_model(model_id: str, width: int, height: int) -> tuple[int, int]:
    """Clamp dimensions for models with size constraints, preserving aspect ratio.

    Args:
        model_id: The image generation model ID.
        width: Desired width.
        height: Desired height.

    Returns:
        (width, height) tuple, possibly scaled down.
    """
    if model_id not in RUNPOD_FLUX_MODELS:
        return width, height

    max_dim = max(width, height)
    if max_dim <= RUNPOD_FLUX_MAX_DIM:
        return width, height

    scale = RUNPOD_FLUX_MAX_DIM / max_dim
    new_w = int(width * scale) // 8 * 8  # Round to nearest 8 for model compatibility
    new_h = int(height * scale) // 8 * 8
    logger.info(f"Clamped dimensions from {width}x{height} to {new_w}x{new_h} for {model_id}")
    return new_w, new_h


class StoryboardService:
    """Orchestrates storyboard planning via Gemini and image generation."""

    def __init__(
        self,
        ai_service: AIService,
        image_gen_service: ImageGenerationService,
    ):
        """Initialize the storyboard service.

        Args:
            ai_service: AIService for Gemini API calls.
            image_gen_service: ImageGenerationService for image generation.
        """
        self.ai_service = ai_service
        self.image_gen_service = image_gen_service

    def generate_storyboard_plan(
        self,
        idea: str,
        num_scenes: int = 6,
        aspect_ratio: str = "9:16",
    ) -> dict:
        """Generate a storyboard plan from a creative idea using Gemini.

        Args:
            idea: The creative concept or story idea.
            num_scenes: Number of scenes to generate.
            aspect_ratio: Aspect ratio for the storyboard images.

        Returns:
            Dictionary with title, characters, scenes, and style_guide.
        """
        prompt = f"""You are a professional storyboard artist and creative director.

Generate a detailed storyboard plan for the following idea:

"{idea}"

Requirements:
- Create exactly {num_scenes} scenes
- Aspect ratio: {aspect_ratio}
- Each scene must have a clear visual description suitable for AI image generation
- Characters must have consistent, detailed appearance descriptions
- Include camera angles, character actions, and environment details

Return a JSON object with this exact structure:
{{
  "title": "Short title for the storyboard",
  "characters": [
    {{
      "name": "Character name",
      "appearance": "Detailed physical appearance (age, build, face, hair color/style, skin tone, distinguishing features)",
      "clothing": "Detailed clothing description (specific garments, colors, patterns, fit)",
      "accessories": "Any accessories (glasses, jewelry, hats, props they carry)"
    }}
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "description": "What is happening in this scene",
      "camera_angle": "e.g. wide shot, close-up, over-the-shoulder, bird's eye, low angle",
      "character_action": "What the character(s) are doing, their pose and expression",
      "environment": "Detailed setting/background description",
      "image_prompt": "Complete, self-contained image generation prompt that includes character appearance, action, environment, camera angle, lighting, and style - do NOT reference other scenes"
    }}
  ],
  "style_guide": "Overall visual style guide (e.g. cinematic, anime, watercolor, photorealistic, comic book) with lighting and color palette notes"
}}

CRITICAL: Each scene's image_prompt must be FULLY SELF-CONTAINED. It must include the complete character appearance description so it can generate a consistent image without any other context. Never say "same character as scene 1" - instead, repeat the full appearance description."""

        response = self.ai_service.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)
        return result

    async def generate_reference_image(
        self,
        character: CharacterProfile,
        style: str,
        model_id: str,
        width: int,
        height: int,
    ) -> str:
        """Generate a reference image for a character.

        Args:
            character: The character profile to visualize.
            style: Style guide for the image.
            model_id: Image generation model to use.
            width: Image width.
            height: Image height.

        Returns:
            URL of the generated reference image.
        """
        prompt = (
            f"Character reference sheet, {style}. "
            f"{character.name}: {character.appearance}. "
            f"Wearing {character.clothing}. "
            f"Accessories: {character.accessories}. "
            f"Full body portrait, neutral pose, clean background, "
            f"high detail, consistent lighting."
        )

        clamped_w, clamped_h = clamp_dimensions_for_model(model_id, width, height)

        request = ImageGenerationRequest(
            prompt=prompt,
            width=clamped_w,
            height=clamped_h,
        )

        if model_id == "flux-dev":
            request.model = ImageGenerationModel.FLUX_DEV
            result = await self.image_gen_service.generate_flux_dev(request)
        elif model_id == "flux-schnell":
            request.model = ImageGenerationModel.FLUX_SCHNELL
            result = await self.image_gen_service.generate_flux_schnell(request)
        elif model_id == "qwen-image":
            request.model = ImageGenerationModel.QWEN_IMAGE
            result = await self.image_gen_service.generate_qwen_image(request)
        elif model_id == "gemini-flash":
            request.model = ImageGenerationModel.GEMINI_FLASH
            result = await self.image_gen_service.generate_gemini(request)
        else:
            # Default to Runware models
            model_enum = ImageGenerationModel(model_id)
            request.model = model_enum
            result = await self.image_gen_service.generate_runware(request)

        if result.images:
            return result.images[0].url
        raise RuntimeError(f"No image generated for character {character.name}")

    async def generate_scene_image(
        self,
        scene: StoryboardScene,
        reference_url: str | None,
        style: str,
        model_id: str,
        width: int,
        height: int,
    ) -> str:
        """Generate an image for a storyboard scene.

        Args:
            scene: The scene to visualize.
            reference_url: URL of character reference image (for Kontext).
            style: Style guide for the image.
            model_id: Image generation model to use.
            width: Image width.
            height: Image height.

        Returns:
            URL of the generated scene image.
        """
        full_prompt = f"{style}. {scene.image_prompt}"
        clamped_w, clamped_h = clamp_dimensions_for_model(model_id, width, height)

        if model_id == "flux-kontext" and reference_url:
            return await self.image_gen_service.generate_flux_kontext(
                prompt=full_prompt,
                reference_image_url=reference_url,
                width=clamped_w,
                height=clamped_h,
            )

        # If flux-kontext was selected but no reference URL, fall back to flux-dev
        effective_model = model_id
        if model_id == "flux-kontext" and not reference_url:
            logger.warning(
                f"Scene {scene.scene_number}: No reference image for Kontext, "
                f"falling back to flux-dev"
            )
            effective_model = "flux-dev"
            clamped_w, clamped_h = clamp_dimensions_for_model(effective_model, width, height)

        # Standard text-to-image
        request = ImageGenerationRequest(
            prompt=full_prompt,
            width=clamped_w,
            height=clamped_h,
        )

        if effective_model == "flux-dev":
            request.model = ImageGenerationModel.FLUX_DEV
            result = await self.image_gen_service.generate_flux_dev(request)
        elif effective_model == "flux-schnell":
            request.model = ImageGenerationModel.FLUX_SCHNELL
            result = await self.image_gen_service.generate_flux_schnell(request)
        elif effective_model == "qwen-image":
            request.model = ImageGenerationModel.QWEN_IMAGE
            result = await self.image_gen_service.generate_qwen_image(request)
        elif effective_model == "gemini-flash":
            request.model = ImageGenerationModel.GEMINI_FLASH
            result = await self.image_gen_service.generate_gemini(request)
        else:
            model_enum = ImageGenerationModel(effective_model)
            request.model = model_enum
            result = await self.image_gen_service.generate_runware(request)

        if result.images:
            return result.images[0].url
        raise RuntimeError(f"No image generated for scene {scene.scene_number}")

    async def run_storyboard_generation(
        self,
        job: StoryboardJob,
        on_progress: Callable,
    ) -> None:
        """Orchestrate full storyboard image generation.

        Generates reference images for each character, then scene images.

        Args:
            job: The storyboard job to process.
            on_progress: Async callback for progress updates.
        """
        if not job.plan:
            raise RuntimeError("No plan set on job")

        plan = job.plan
        ref_model = getattr(job, "_ref_model", "flux-dev")
        scene_model = getattr(job, "_scene_model", "flux-kontext")
        width = getattr(job, "_width", 1080)
        height = getattr(job, "_height", 1920)
        user_reference_images: dict[str, str] = getattr(job, "_user_reference_images", {})

        # Count characters that need AI generation (exclude user-provided references)
        chars_needing_generation = [
            c for c in plan.characters if c.name not in user_reference_images
        ]
        total_steps = len(chars_needing_generation) + len(plan.scenes)
        current_step = 0

        # Phase 1: Handle reference images
        job.status = "generating_references"

        # First, store user-provided reference images immediately
        for character in plan.characters:
            if character.name in user_reference_images:
                url = user_reference_images[character.name]
                job.reference_images[character.name] = url
                await on_progress({
                    "type": "reference_complete",
                    "character": character.name,
                    "image_url": url,
                    "step": current_step,
                    "total_steps": total_steps,
                })

        # Then generate references for remaining characters
        for character in chars_needing_generation:
            try:
                await on_progress({
                    "type": "reference_start",
                    "character": character.name,
                    "step": current_step,
                    "total_steps": total_steps,
                })

                url = await self.generate_reference_image(
                    character=character,
                    style=plan.style_guide,
                    model_id=ref_model,
                    width=width,
                    height=height,
                )
                job.reference_images[character.name] = url
                current_step += 1

                await on_progress({
                    "type": "reference_complete",
                    "character": character.name,
                    "image_url": url,
                    "step": current_step,
                    "total_steps": total_steps,
                })

            except Exception as e:
                logger.error(f"Failed to generate reference for {character.name}: {e}")
                current_step += 1
                await on_progress({
                    "type": "reference_failed",
                    "character": character.name,
                    "error": str(e),
                    "step": current_step,
                    "total_steps": total_steps,
                })

        # Phase 2: Generate scene images
        job.status = "generating_scenes"
        # Pick the first character's reference image for Kontext (if available)
        first_ref_url = None
        if job.reference_images:
            first_ref_url = next(iter(job.reference_images.values()))

        for scene in plan.scenes:
            scene_entry = {
                "scene_number": scene.scene_number,
                "image_url": None,
                "status": "pending",
                "error": None,
            }
            job.scene_images.append(scene_entry)

            try:
                await on_progress({
                    "type": "scene_start",
                    "scene_number": scene.scene_number,
                    "step": current_step,
                    "total_steps": total_steps,
                })

                url = await self.generate_scene_image(
                    scene=scene,
                    reference_url=first_ref_url,
                    style=plan.style_guide,
                    model_id=scene_model,
                    width=width,
                    height=height,
                )
                scene_entry["image_url"] = url
                scene_entry["status"] = "completed"
                current_step += 1

                await on_progress({
                    "type": "scene_complete",
                    "scene_number": scene.scene_number,
                    "image_url": url,
                    "step": current_step,
                    "total_steps": total_steps,
                })

            except Exception as e:
                logger.error(
                    f"Failed to generate scene {scene.scene_number}: {e}"
                )
                scene_entry["status"] = "failed"
                scene_entry["error"] = str(e)
                current_step += 1

                await on_progress({
                    "type": "scene_failed",
                    "scene_number": scene.scene_number,
                    "error": str(e),
                    "step": current_step,
                    "total_steps": total_steps,
                })
