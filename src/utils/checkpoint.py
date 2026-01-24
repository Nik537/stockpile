"""Checkpoint system for resuming processing after failures."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for resuming video processing."""

    video_path: str
    project_dir: str | None = None
    stage: str = "initialized"  # initialized, transcribed, planned, processing, completed
    transcript_file: str | None = None
    broll_plan_file: str | None = None
    completed_needs: list[str] = field(default_factory=list)  # List of completed need folder names
    total_needs: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            "updated_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingCheckpoint":
        """Create checkpoint from dictionary."""
        return cls(**data)

    def save(self, checkpoint_path: Path) -> None:
        """Save checkpoint to file.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with checkpoint_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.debug(f"Checkpoint saved: {checkpoint_path} (stage: {self.stage})")

    @classmethod
    def load(cls, checkpoint_path: Path) -> "ProcessingCheckpoint | None":
        """Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            ProcessingCheckpoint object or None if file doesn't exist
        """
        if not checkpoint_path.exists():
            return None

        try:
            with checkpoint_path.open() as f:
                data = json.load(f)

            logger.info(f"Loaded checkpoint: {checkpoint_path} (stage: {data.get('stage')})")
            return cls.from_dict(data)

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None


def get_checkpoint_path(video_path: str, output_dir: str) -> Path:
    """Get the checkpoint file path for a video.

    Args:
        video_path: Path to the video being processed
        output_dir: Output directory

    Returns:
        Path to checkpoint file
    """
    video_name = Path(video_path).stem
    checkpoint_dir = Path(output_dir) / ".checkpoints"
    return checkpoint_dir / f"{video_name}.checkpoint.json"


def cleanup_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint file after successful completion.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    try:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.debug(f"Checkpoint cleaned up: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup checkpoint {checkpoint_path}: {e}")
