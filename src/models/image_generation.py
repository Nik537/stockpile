"""Models for AI image generation using fal.ai (Flux 2 Klein, Z-Image Turbo)."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ImageGenerationModel(str, Enum):
    """Supported image generation models."""

    # fal.ai models
    FLUX_KLEIN = "flux-klein"
    Z_IMAGE = "z-image"

    # RunPod public endpoint models
    RUNPOD_FLUX_DEV = "runpod-flux-dev"
    RUNPOD_FLUX_SCHNELL = "runpod-flux-schnell"
    RUNPOD_FLUX_KONTEXT = "runpod-flux-kontext"  # Image editing

    # RunPod Qwen models (cheap, good text rendering)
    RUNPOD_QWEN_IMAGE = "runpod-qwen-image"
    RUNPOD_QWEN_IMAGE_LORA = "runpod-qwen-image-lora"
    RUNPOD_QWEN_IMAGE_EDIT = "runpod-qwen-image-edit"

    # RunPod Seedream models
    RUNPOD_SEEDREAM_3 = "runpod-seedream-3"
    RUNPOD_SEEDREAM_4 = "runpod-seedream-4"

    # Gemini (Google) - FREE 500/day
    GEMINI_FLASH = "gemini-flash"

    # Replicate models
    REPLICATE_FLUX_KLEIN = "replicate-flux-klein"


class ImageGenerationStatus(str, Enum):
    """Status of an image generation job."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ImageGenerationRequest:
    """Request for text-to-image generation."""

    prompt: str
    model: ImageGenerationModel = ImageGenerationModel.FLUX_KLEIN
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: Optional[int] = None
    guidance_scale: float = 7.5

    def to_dict(self) -> dict:
        """Convert to dictionary for API requests."""
        return {
            "prompt": self.prompt,
            "model": self.model.value,
            "width": self.width,
            "height": self.height,
            "num_images": self.num_images,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }


@dataclass
class ImageEditRequest:
    """Request for image editing (image-to-image)."""

    prompt: str
    input_image_url: str  # URL or data:image/... base64
    model: ImageGenerationModel = ImageGenerationModel.FLUX_KLEIN
    strength: float = 0.75  # How much to transform (0=none, 1=complete)
    seed: Optional[int] = None
    guidance_scale: float = 7.5

    def to_dict(self) -> dict:
        """Convert to dictionary for API requests."""
        return {
            "prompt": self.prompt,
            "input_image_url": self.input_image_url,
            "model": self.model.value,
            "strength": self.strength,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }


@dataclass
class GeneratedImage:
    """A single generated image result."""

    url: str
    width: int
    height: int
    content_type: str = "image/png"
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "url": self.url,
            "width": self.width,
            "height": self.height,
            "content_type": self.content_type,
            "seed": self.seed,
        }


@dataclass
class ImageGenerationResult:
    """Result of an image generation request."""

    images: list[GeneratedImage]
    model: ImageGenerationModel
    prompt: str
    generation_time_ms: int
    cost_estimate: float  # Estimated cost in USD

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "images": [img.to_dict() for img in self.images],
            "model": self.model.value,
            "prompt": self.prompt,
            "generation_time_ms": self.generation_time_ms,
            "cost_estimate": self.cost_estimate,
        }


@dataclass
class ImageGenerationJob:
    """Job for tracking image generation requests."""

    id: str
    request_type: str  # "generate" or "edit"
    prompt: str
    model: ImageGenerationModel
    status: ImageGenerationStatus = ImageGenerationStatus.PENDING
    result: Optional[ImageGenerationResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "request_type": self.request_type,
            "prompt": self.prompt,
            "model": self.model.value,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
