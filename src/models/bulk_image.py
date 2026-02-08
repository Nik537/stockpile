"""Models for bulk image generation feature."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass
class BulkImagePrompt:
    """A single prompt for bulk image generation."""

    index: int
    prompt: str
    rendering_style: str  # cartoon, claymation, isometric-3d, watercolor, etc.
    mood: str  # energetic, calm, dramatic, playful, sophisticated, whimsical, luxurious, cheerful, bold
    composition: str  # centered-character, scene, product-hero, poster, pattern, vignette, banner, macro-detail
    has_text_space: bool = False  # Whether prompt includes space for text/slogans

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "rendering_style": self.rendering_style,
            "mood": self.mood,
            "composition": self.composition,
            "has_text_space": self.has_text_space,
        }


@dataclass
class BulkImageResult:
    """Result of a single image generation within a bulk job."""

    index: int
    prompt: BulkImagePrompt
    image_url: Optional[str]
    width: int
    height: int
    generation_time_ms: int
    status: Literal["completed", "failed"]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "index": self.index,
            "prompt": self.prompt.to_dict(),
            "image_url": self.image_url,
            "width": self.width,
            "height": self.height,
            "generation_time_ms": self.generation_time_ms,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class BulkImageJob:
    """Job for tracking bulk image generation requests."""

    job_id: str
    meta_prompt: str
    model: str  # runpod-flux-schnell or runpod-flux-dev
    width: int
    height: int
    total_count: int
    completed_count: int = 0
    failed_count: int = 0
    status: str = "pending"  # pending, generating_prompts, generating_images, completed, failed
    prompts: list[BulkImagePrompt] = field(default_factory=list)
    results: list[BulkImageResult] = field(default_factory=list)
    total_cost: float = 0.0
    estimated_cost: float = 0.0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "meta_prompt": self.meta_prompt,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "total_count": self.total_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "status": self.status,
            "prompts": [p.to_dict() for p in self.prompts],
            "results": [r.to_dict() for r in self.results],
            "total_cost": self.total_cost,
            "estimated_cost": self.estimated_cost,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
