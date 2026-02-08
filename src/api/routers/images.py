"""Image generation routes for the Stockpile API (fal.ai, RunPod, Gemini, Replicate)."""

import logging

from api.dependencies import get_image_gen_service
from api.schemas import ImageEditRequestBody, ImageGenerateRequest, RunPodImageGenerateRequest
from fastapi import APIRouter, HTTPException
from models.image_generation import (
    ImageEditRequest,
    ImageGenerationModel,
    ImageGenerationRequest,
)
from services.image_generation_service import ImageGenerationServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Generation"])


# =============================================================================
# fal.ai Image Generation Endpoints (Flux 2 Klein, Z-Image Turbo)
# =============================================================================


@router.get("/api/image-generation/status", summary="fal.ai image gen status", description="Check if fal.ai image generation (Flux Klein, Z-Image) is configured.")
async def get_image_generation_status() -> dict:
    """Check if image generation is configured.

    Returns:
        Status dict with 'configured', 'available', and optional 'error'.
    """
    service = get_image_gen_service()
    return await service.check_health()


@router.post("/api/generate-image", summary="Generate image (fal.ai)", description="Generate images via fal.ai using Flux Klein or Z-Image models.", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Generation failed"}})
async def generate_image(request: ImageGenerateRequest) -> dict:
    """Generate images from a text prompt.

    Args:
        request: Generation parameters

    Returns:
        Generation result with image URLs
    """
    service = get_image_gen_service()

    # Validate model
    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid model. Use 'flux-klein' or 'z-image'.",
        )

    # Check service is configured
    if not service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Image generation not configured. Set FAL_API_KEY in .env",
        )

    # Validate prompt
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Validate dimensions
    if request.width < 256 or request.width > 2048:
        raise HTTPException(
            status_code=400, detail="Width must be between 256 and 2048"
        )
    if request.height < 256 or request.height > 2048:
        raise HTTPException(
            status_code=400, detail="Height must be between 256 and 2048"
        )
    if request.num_images < 1 or request.num_images > 4:
        raise HTTPException(
            status_code=400, detail="num_images must be between 1 and 4"
        )

    try:
        gen_request = ImageGenerationRequest(
            prompt=request.prompt.strip(),
            model=model,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
        )

        result = await service.generate(gen_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/edit-image", summary="Edit image (fal.ai)", description="Edit an existing image using a text prompt via fal.ai.", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Editing failed"}})
async def edit_image(request: ImageEditRequestBody) -> dict:
    """Edit an image using a text prompt.

    Args:
        request: Edit parameters including input image

    Returns:
        Generation result with edited image URLs
    """
    service = get_image_gen_service()

    # Validate model
    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid model. Use 'flux-klein' or 'z-image'.",
        )

    # Check service is configured
    if not service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Image generation not configured. Set FAL_API_KEY in .env",
        )

    # Validate prompt
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Validate image URL
    if not request.image_url or not request.image_url.strip():
        raise HTTPException(status_code=400, detail="Image URL is required")

    # Validate strength
    if request.strength < 0 or request.strength > 1:
        raise HTTPException(
            status_code=400, detail="Strength must be between 0 and 1"
        )

    try:
        edit_request = ImageEditRequest(
            prompt=request.prompt.strip(),
            input_image_url=request.image_url.strip(),
            model=model,
            strength=request.strength,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
        )

        result = await service.edit(edit_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"Image editing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RunPod Image Generation Endpoints (Flux Dev, Flux Schnell)
# =============================================================================


@router.get("/api/runpod-image/status", summary="RunPod image gen status", description="Check if RunPod image generation (Flux Dev/Schnell) is configured.")
async def get_runpod_image_status() -> dict:
    """Check if RunPod image generation is configured.

    Returns:
        Status dict with 'configured', 'available', and available models.
    """
    service = get_image_gen_service()
    return await service.check_runpod_health()


@router.post("/api/runpod-image/generate", summary="Generate image (RunPod)", description="Generate images using RunPod public Flux Dev or Schnell endpoints.", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Generation failed"}})
async def generate_runpod_image(request: RunPodImageGenerateRequest) -> dict:
    """Generate images using RunPod public endpoints (Flux Dev/Schnell).

    Args:
        request: Generation parameters

    Returns:
        Generation result with image URLs
    """
    service = get_image_gen_service()

    # Validate model
    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid model. Use 'runpod-flux-dev' or 'runpod-flux-schnell'.",
        )

    # Check service is configured
    if not service.is_runpod_configured():
        raise HTTPException(
            status_code=400,
            detail="RunPod not configured. Set RUNPOD_API_KEY in .env",
        )

    # Validate prompt
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        gen_request = ImageGenerationRequest(
            prompt=request.prompt.strip(),
            model=model,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )

        result = await service.generate_runpod(gen_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"RunPod image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Gemini Image Generation (FREE 500/day)
# =============================================================================


@router.get("/api/gemini-image/status", summary="Gemini image gen status", description="Check if Gemini image generation is configured (FREE 500 images/day).")
async def get_gemini_image_status() -> dict:
    """Check if Gemini image generation is configured.

    Returns:
        Status dict with 'configured', 'available', and free quota info.
    """
    service = get_image_gen_service()
    return await service.check_gemini_health()


@router.post("/api/gemini-image/generate", summary="Generate image (Gemini)", description="Generate images using Gemini 2.5 Flash. FREE tier: 500 images/day.", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Generation failed"}})
async def generate_gemini_image(request: RunPodImageGenerateRequest) -> dict:
    """Generate images using Gemini 2.5 Flash (FREE 500/day).

    Args:
        request: Generation parameters

    Returns:
        Generation result with image data
    """
    service = get_image_gen_service()

    if not service.is_gemini_configured():
        raise HTTPException(
            status_code=400,
            detail="Gemini not configured. Set GEMINI_API_KEY in .env",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        gen_request = ImageGenerationRequest(
            prompt=request.prompt.strip(),
            model=ImageGenerationModel.GEMINI_FLASH,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )

        result = await service.generate_gemini(gen_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"Gemini image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Replicate Image Generation (Flux Klein - fastest)
# =============================================================================


@router.get("/api/replicate-image/status", summary="Replicate image gen status", description="Check if Replicate image generation (Flux Klein) is configured.")
async def get_replicate_image_status() -> dict:
    """Check if Replicate image generation is configured.

    Returns:
        Status dict with 'configured', 'available', and available models.
    """
    service = get_image_gen_service()
    return await service.check_replicate_health()


@router.post("/api/replicate-image/generate", summary="Generate image (Replicate)", description="Generate images using Replicate Flux Klein. Fast and cheap (~$0.003/image).", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Generation failed"}})
async def generate_replicate_image(request: RunPodImageGenerateRequest) -> dict:
    """Generate images using Replicate Flux Klein (fast, ~$0.003/image).

    Args:
        request: Generation parameters

    Returns:
        Generation result with image URLs
    """
    service = get_image_gen_service()

    if not service.is_replicate_configured():
        raise HTTPException(
            status_code=400,
            detail="Replicate not configured. Set REPLICATE_API_KEY in .env",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        gen_request = ImageGenerationRequest(
            prompt=request.prompt.strip(),
            model=ImageGenerationModel.REPLICATE_FLUX_KLEIN,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )

        result = await service.generate_replicate(gen_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"Replicate image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
