"""Clip-related data models for B-roll extraction."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ClipSegment:
    """Represents a segment of video identified for extraction."""

    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    relevance_score: int  # 1-10 score for how relevant the segment is
    description: str  # AI-generated description of the segment content

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class ClipResult:
    """Represents an extracted clip from a source video."""

    source_video_path: str  # Path to original downloaded video
    clip_path: str  # Path to extracted clip file
    segment: ClipSegment  # Segment info that was extracted
    search_phrase: str  # The search phrase this clip relates to
    source_video_id: str  # YouTube video ID
    extraction_success: bool = True  # Whether extraction succeeded
    error_message: Optional[str] = None


@dataclass
class VideoAnalysisResult:
    """Result of AI video analysis for B-roll segments."""

    video_path: str  # Path to analyzed video
    video_id: str  # YouTube video ID
    search_phrase: str  # Context for what we're looking for
    segments: List[ClipSegment] = field(default_factory=list)
    analysis_success: bool = True
    error_message: Optional[str] = None
    total_duration: Optional[float] = None  # Video duration in seconds
