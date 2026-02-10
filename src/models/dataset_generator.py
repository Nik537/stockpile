"""Models for LoRA dataset generation feature."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DatasetMode(str, Enum):
    """Supported dataset generation modes."""

    PAIR = "pair"
    SINGLE = "single"
    REFERENCE = "reference"
    LAYERED = "layered"


class DatasetStatus(str, Enum):
    """Status of a dataset generation job."""

    PENDING = "pending"
    GENERATING_PROMPTS = "generating_prompts"
    GENERATING_IMAGES = "generating_images"
    CAPTIONING = "captioning"
    PACKAGING = "packaging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DatasetGenerationRequest:
    """Request parameters for dataset generation."""

    mode: DatasetMode
    theme: str
    model: str = "runware-flux-klein-4b"
    llm_model: str = "gemini-flash"
    num_items: int = 10
    max_concurrent: int = 3
    aspect_ratio: str = "1:1"
    trigger_word: str = ""
    use_vision_caption: bool = True
    custom_system_prompt: str = ""
    # Pair mode specific
    transformation: str = ""
    action_name: str = ""
    # Reference mode specific
    reference_image_base64: str = ""
    # Layered mode specific
    layered_use_case: str = "character"
    elements_description: str = ""
    final_image_description: str = ""
    width: int = 1024
    height: int = 1024

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "mode": self.mode.value,
            "theme": self.theme,
            "model": self.model,
            "llm_model": self.llm_model,
            "num_items": self.num_items,
            "max_concurrent": self.max_concurrent,
            "aspect_ratio": self.aspect_ratio,
            "trigger_word": self.trigger_word,
            "use_vision_caption": self.use_vision_caption,
            "custom_system_prompt": self.custom_system_prompt,
            "transformation": self.transformation,
            "action_name": self.action_name,
            "reference_image_base64": bool(self.reference_image_base64),
            "layered_use_case": self.layered_use_case,
            "elements_description": self.elements_description,
            "final_image_description": self.final_image_description,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class DatasetItem:
    """A single item in a dataset generation job."""

    index: int
    status: str = "pending"  # pending, generating, captioning, completed, failed
    # Pair mode fields
    start_url: str = ""
    end_url: str = ""
    start_prompt: str = ""
    end_prompt: str = ""
    # Single / reference mode fields
    image_url: str = ""
    prompt: str = ""
    # Caption (all modes)
    caption: str = ""
    # Error
    error: str = ""
    generation_time_ms: int = 0
    cost: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "index": self.index,
            "status": self.status,
            "start_url": self.start_url,
            "end_url": self.end_url,
            "start_prompt": self.start_prompt,
            "end_prompt": self.end_prompt,
            "image_url": self.image_url,
            "prompt": self.prompt,
            "caption": self.caption,
            "error": self.error,
            "generation_time_ms": self.generation_time_ms,
            "cost": self.cost,
        }


@dataclass
class DatasetJob:
    """Job for tracking dataset generation requests."""

    job_id: str
    request: DatasetGenerationRequest
    status: DatasetStatus = DatasetStatus.PENDING
    items: list[DatasetItem] = field(default_factory=list)
    total_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    total_cost: float = 0.0
    estimated_cost: float = 0.0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    zip_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "job_id": self.job_id,
            "request": self.request.to_dict(),
            "status": self.status.value,
            "items": [item.to_dict() for item in self.items],
            "total_count": self.total_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "total_cost": self.total_cost,
            "estimated_cost": self.estimated_cost,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "zip_path": self.zip_path,
        }
