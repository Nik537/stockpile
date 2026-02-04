"""TTS (Text-to-Speech) data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TTSJobStatus(str, Enum):
    """Status of a TTS generation job."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TTSJob:
    """Represents a TTS generation job."""

    id: str
    text: str
    status: TTSJobStatus = TTSJobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    voice_reference_path: str | None = None
    output_path: str | None = None
    error: str | None = None
    # Generation parameters
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "text_length": len(self.text),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "voice_reference": self.voice_reference_path is not None,
            "output_path": self.output_path,
            "error": self.error,
        }
