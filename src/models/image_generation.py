"""Models for AI image generation (Runware, Gemini, Nano Banana Pro)."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ImageGenerationModel(str, Enum):
    """Supported image generation models."""

    RUNWARE_FLUX_KLEIN_4B = "runware-flux-klein-4b"     # $0.0006/img
    RUNWARE_FLUX_KLEIN_9B = "runware-flux-klein-9b"     # $0.00078/img
    RUNWARE_Z_IMAGE = "runware-z-image"                  # $0.0006/img
    GEMINI_FLASH = "gemini-flash"                        # FREE (500/day)
    NANO_BANANA_PRO = "nano-banana-pro"                  # RunPod public endpoint


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
    model: ImageGenerationModel = ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B
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
    model: ImageGenerationModel = ImageGenerationModel.NANO_BANANA_PRO
    strength: float = 0.75  # How much to transform (0=none, 1=complete)
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    mask_image: Optional[str] = None  # base64 data URL for inpainting mask

    def to_dict(self) -> dict:
        """Convert to dictionary for API requests."""
        result = {
            "prompt": self.prompt,
            "input_image_url": self.input_image_url,
            "model": self.model.value,
            "strength": self.strength,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }
        if self.mask_image:
            result["mask_image"] = self.mask_image
        return result


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
