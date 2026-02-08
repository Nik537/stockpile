"""Image generation routes for the Stockpile API (Runware, Gemini, Nano Banana Pro)."""

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
# Unified Image Generation Endpoints
# =============================================================================


@router.get("/api/image/status", summary="Image generation status", description="Check status of all configured image generation providers.")
async def get_image_status() -> dict:
    """Combined status for all image generation providers."""
    service = get_image_gen_service()
    runware = await service.check_runware_health()
    gemini = await service.check_gemini_health()
    runpod = await service.check_runpod_health()
    return {
        "runware": runware,
        "gemini": gemini,
        "runpod": runpod,
        "default_model": service.default_model,
    }


@router.post("/api/image/generate", summary="Generate image (unified)", description="Generate images using any supported model. Routes to the correct provider automatically.", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Generation failed"}})
async def unified_generate_image(request: ImageGenerateRequest) -> dict:
    """Unified image generation endpoint - routes to correct provider based on model."""
    service = get_image_gen_service()

    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        valid = [m.value for m in ImageGenerationModel]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Valid models: {valid}",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    if request.width < 256 or request.width > 2048:
        raise HTTPException(status_code=400, detail="Width must be between 256 and 2048")
    if request.height < 256 or request.height > 2048:
        raise HTTPException(status_code=400, detail="Height must be between 256 and 2048")
    if request.num_images < 1 or request.num_images > 4:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 4")

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

        result = await service.generate_image(gen_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/image/edit", summary="Edit image (unified)", description="Edit/inpaint images. Currently routes to Nano Banana Pro.", responses={400: {"description": "Invalid parameters"}, 500: {"description": "Editing failed"}})
async def unified_edit_image(request: ImageEditRequestBody) -> dict:
    """Unified image editing endpoint."""
    service = get_image_gen_service()

    try:
        model = ImageGenerationModel(request.model)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'.",
        )

    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    if not request.image_url or not request.image_url.strip():
        raise HTTPException(status_code=400, detail="Image URL is required")

    if request.strength < 0 or request.strength > 1:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 1")

    try:
        edit_request = ImageEditRequest(
            prompt=request.prompt.strip(),
            input_image_url=request.image_url.strip(),
            model=model,
            strength=request.strength,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            mask_image=request.mask_image,
        )

        result = await service.edit_image(edit_request)
        return result.to_dict()

    except ImageGenerationServiceError as e:
        logger.error(f"Image editing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Backward-compatible endpoints (delegate to unified methods)
# =============================================================================


@router.get("/api/image-generation/status", summary="Legacy: fal.ai status", description="Legacy endpoint. Use /api/image/status instead.", include_in_schema=False)
async def get_image_generation_status() -> dict:
    """Legacy status endpoint - redirects to unified status."""
    return await get_image_status()


@router.post("/api/generate-image", summary="Legacy: Generate image", description="Legacy endpoint. Use /api/image/generate instead.", include_in_schema=False)
async def generate_image(request: ImageGenerateRequest) -> dict:
    """Legacy generate endpoint - delegates to unified."""
    return await unified_generate_image(request)


@router.post("/api/edit-image", summary="Legacy: Edit image", description="Legacy endpoint. Use /api/image/edit instead.", include_in_schema=False)
async def edit_image(request: ImageEditRequestBody) -> dict:
    """Legacy edit endpoint - delegates to unified."""
    return await unified_edit_image(request)


@router.get("/api/runpod-image/status", summary="Legacy: RunPod status", include_in_schema=False)
async def get_runpod_image_status() -> dict:
    """Legacy RunPod status endpoint."""
    service = get_image_gen_service()
    return await service.check_runpod_health()


@router.post("/api/runpod-image/generate", summary="Legacy: RunPod generate", include_in_schema=False)
async def generate_runpod_image(request: RunPodImageGenerateRequest) -> dict:
    """Legacy RunPod generate endpoint - delegates to unified."""
    gen_request = ImageGenerateRequest(
        prompt=request.prompt,
        model=request.model,
        width=request.width,
        height=request.height,
        seed=request.seed,
    )
    return await unified_generate_image(gen_request)


@router.get("/api/gemini-image/status", summary="Legacy: Gemini status", include_in_schema=False)
async def get_gemini_image_status() -> dict:
    """Legacy Gemini status endpoint."""
    service = get_image_gen_service()
    return await service.check_gemini_health()


@router.post("/api/gemini-image/generate", summary="Legacy: Gemini generate", include_in_schema=False)
async def generate_gemini_image(request: RunPodImageGenerateRequest) -> dict:
    """Legacy Gemini generate endpoint - delegates to unified."""
    gen_request = ImageGenerateRequest(
        prompt=request.prompt,
        model="gemini-flash",
        width=request.width,
        height=request.height,
        seed=request.seed,
    )
    return await unified_generate_image(gen_request)
