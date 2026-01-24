"""Progress tracking and status reporting for B-roll processing."""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStage:
    """Represents a stage in the processing pipeline."""

    name: str
    total_items: int
    completed_items: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimate time remaining in seconds."""
        if self.completed_items == 0 or self.start_time is None:
            return None

        elapsed = self.elapsed_time
        rate = self.completed_items / elapsed  # items per second
        remaining_items = self.total_items - self.completed_items

        if rate > 0:
            return remaining_items / rate
        return None


class ProcessingStatus:
    """Tracks overall processing status and progress.

    Manages multiple processing stages and provides real-time updates
    through callbacks and JSON file exports.

    Example usage:
        status = ProcessingStatus(video_path="input.mp4")

        # Register stages
        status.register_stage("transcribe", total_items=1)
        status.register_stage("plan", total_items=1)
        status.register_stage("download", total_items=10)

        # Update progress
        status.start_stage("transcribe")
        status.update_stage("transcribe", completed=1)
        status.complete_stage("transcribe")

        # Get overall progress
        print(f"Overall: {status.overall_progress:.1f}%")
    """

    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        status_file: str = ".processing_status.json",
        update_callback: Optional[Callable] = None,
    ):
        """Initialize processing status tracker.

        Args:
            video_path: Path to input video being processed
            output_dir: Directory for output files
            status_file: Path to JSON status file for external monitoring
            update_callback: Optional callback function called on updates
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.status_file = Path(status_file)
        self.update_callback = update_callback

        self.stages: dict[str, ProcessingStage] = {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self.current_stage: Optional[str] = None

        logger.info(f"Initialized progress tracking for: {video_path}")

    def register_stage(self, stage_name: str, total_items: int):
        """Register a processing stage.

        Args:
            stage_name: Name of the stage (e.g., "transcribe", "download")
            total_items: Total number of items to process in this stage
        """
        self.stages[stage_name] = ProcessingStage(
            name=stage_name, total_items=total_items
        )
        logger.debug(f"Registered stage: {stage_name} ({total_items} items)")
        self._notify_update()

    def start_stage(self, stage_name: str):
        """Mark a stage as started.

        Args:
            stage_name: Name of the stage to start
        """
        if stage_name not in self.stages:
            logger.warning(f"Stage {stage_name} not registered, auto-registering")
            self.register_stage(stage_name, total_items=1)

        stage = self.stages[stage_name]
        stage.status = "in_progress"
        stage.start_time = time.time()
        self.current_stage = stage_name

        logger.info(f"Started stage: {stage_name}")
        self._notify_update()

    def update_stage(
        self,
        stage_name: str,
        completed: Optional[int] = None,
        increment: int = 0,
    ):
        """Update progress for a stage.

        Args:
            stage_name: Name of the stage to update
            completed: Set completed items to this value (absolute)
            increment: Increment completed items by this amount (relative)
        """
        if stage_name not in self.stages:
            logger.warning(f"Cannot update unregistered stage: {stage_name}")
            return

        stage = self.stages[stage_name]

        if completed is not None:
            stage.completed_items = min(completed, stage.total_items)
        else:
            stage.completed_items = min(
                stage.completed_items + increment, stage.total_items
            )

        logger.debug(
            f"Updated stage {stage_name}: {stage.completed_items}/{stage.total_items}"
        )
        self._notify_update()

    def complete_stage(self, stage_name: str):
        """Mark a stage as completed.

        Args:
            stage_name: Name of the stage to complete
        """
        if stage_name not in self.stages:
            logger.warning(f"Cannot complete unregistered stage: {stage_name}")
            return

        stage = self.stages[stage_name]
        stage.status = "completed"
        stage.completed_items = stage.total_items
        stage.end_time = time.time()

        logger.info(
            f"Completed stage: {stage_name} ({stage.elapsed_time:.1f}s)"
        )
        self._notify_update()

    def fail_stage(self, stage_name: str, error: str):
        """Mark a stage as failed.

        Args:
            stage_name: Name of the stage that failed
            error: Error message
        """
        if stage_name not in self.stages:
            logger.warning(f"Cannot fail unregistered stage: {stage_name}")
            return

        stage = self.stages[stage_name]
        stage.status = "failed"
        stage.end_time = time.time()
        self.error_message = error

        logger.error(f"Failed stage: {stage_name} - {error}")
        self._notify_update()

    def complete_processing(self):
        """Mark entire processing as complete."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        logger.info(f"Processing complete in {elapsed:.1f}s")
        self._notify_update()

    def fail_processing(self, error: str):
        """Mark entire processing as failed.

        Args:
            error: Error message
        """
        self.end_time = time.time()
        self.error_message = error

        logger.error(f"Processing failed: {error}")
        self._notify_update()

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage across all stages."""
        if not self.stages:
            return 0.0

        total_items = sum(stage.total_items for stage in self.stages.values())
        completed_items = sum(
            stage.completed_items for stage in self.stages.values()
        )

        if total_items == 0:
            return 0.0

        return (completed_items / total_items) * 100

    @property
    def overall_eta_seconds(self) -> Optional[float]:
        """Estimate overall time remaining in seconds."""
        if not self.stages:
            return None

        total_items = sum(stage.total_items for stage in self.stages.values())
        completed_items = sum(
            stage.completed_items for stage in self.stages.values()
        )

        if completed_items == 0:
            return None

        elapsed = time.time() - self.start_time
        rate = completed_items / elapsed  # items per second
        remaining_items = total_items - completed_items

        if rate > 0:
            return remaining_items / rate
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all stages are complete."""
        return all(
            stage.status == "completed" for stage in self.stages.values()
        )

    @property
    def has_failed(self) -> bool:
        """Check if any stage has failed."""
        return any(
            stage.status == "failed" for stage in self.stages.values()
        ) or self.error_message is not None

    def to_dict(self) -> dict:
        """Convert status to dictionary for JSON export."""
        return {
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": time.time() - self.start_time,
            "current_stage": self.current_stage,
            "overall_progress": round(self.overall_progress, 2),
            "overall_eta_seconds": self.overall_eta_seconds,
            "is_complete": self.is_complete,
            "has_failed": self.has_failed,
            "error_message": self.error_message,
            "stages": {
                name: {
                    "name": stage.name,
                    "status": stage.status,
                    "total_items": stage.total_items,
                    "completed_items": stage.completed_items,
                    "progress_percent": round(stage.progress_percent, 2),
                    "elapsed_seconds": stage.elapsed_time,
                    "eta_seconds": stage.eta_seconds,
                }
                for name, stage in self.stages.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

    def write_status_file(self):
        """Write current status to JSON file for external monitoring."""
        try:
            status_data = self.to_dict()
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2)
            logger.debug(f"Wrote status to {self.status_file}")
        except Exception as e:
            logger.warning(f"Failed to write status file: {e}")

    def _notify_update(self):
        """Notify callback and update status file."""
        # Write status file
        self.write_status_file()

        # Call update callback if registered
        if self.update_callback:
            try:
                self.update_callback(self)
            except Exception as e:
                logger.warning(f"Update callback failed: {e}")


def format_eta(seconds: Optional[float]) -> str:
    """Format ETA in human-readable format.

    Args:
        seconds: ETA in seconds

    Returns:
        Formatted string (e.g., "2m 30s", "45s", "1h 5m")
    """
    if seconds is None:
        return "calculating..."

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
