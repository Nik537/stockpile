"""Data models for timeline-aware B-roll planning."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BRollNeed:
    """A specific B-roll need at a point in the source video."""

    timestamp: float  # When in source video this B-roll is needed (seconds)
    search_phrase: str  # YouTube search query
    description: str  # What this B-roll should show
    context: str  # Surrounding transcript text for reference
    suggested_duration: float = 5.0  # How long the B-roll should be (4-15s)

    def __post_init__(self):
        """Validate and clamp values."""
        self.suggested_duration = max(4.0, min(15.0, self.suggested_duration))
        self.timestamp = max(0.0, self.timestamp)

    @property
    def folder_name(self) -> str:
        """Generate timestamp-prefixed folder name for this B-roll need."""
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        # Sanitize description for folder name
        safe_desc = self._sanitize_for_folder(self.description)
        return f"{minutes}m{seconds:02d}s_{safe_desc}"

    def _sanitize_for_folder(self, text: str) -> str:
        """Sanitize text for use in folder names."""
        import re

        # Remove special characters, keep alphanumeric and spaces
        sanitized = re.sub(r"[^\w\s-]", "", text)
        # Replace spaces with underscores
        sanitized = re.sub(r"\s+", "_", sanitized)
        # Limit length
        return sanitized[:40].strip("_").lower()


@dataclass
class BRollPlan:
    """Complete B-roll plan for a source video."""

    source_duration: float  # Total source video length in seconds
    needs: List[BRollNeed] = field(default_factory=list)
    clips_per_minute: float = 2.0  # Target density
    source_file: Optional[str] = None  # Original input file path

    @property
    def expected_clip_count(self) -> int:
        """Calculate expected number of clips based on duration and density."""
        return max(1, int(self.source_duration / 60 * self.clips_per_minute))

    @property
    def actual_clip_count(self) -> int:
        """Return actual number of B-roll needs identified."""
        return len(self.needs)

    def get_needs_sorted_by_timestamp(self) -> List[BRollNeed]:
        """Return needs sorted by timestamp in the source video."""
        return sorted(self.needs, key=lambda n: n.timestamp)


@dataclass
class TranscriptSegment:
    """A segment from Whisper transcription with timing info."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str  # Transcribed text for this segment


@dataclass
class TranscriptResult:
    """Full transcription result with timing data."""

    text: str  # Full transcript text
    segments: List[TranscriptSegment] = field(default_factory=list)
    duration: float = 0.0  # Total audio duration in seconds
    language: Optional[str] = None  # Detected language

    def get_text_around_timestamp(
        self, timestamp: float, context_seconds: float = 10.0
    ) -> str:
        """Get transcript text around a specific timestamp."""
        start_time = max(0, timestamp - context_seconds / 2)
        end_time = timestamp + context_seconds / 2

        relevant_segments = [
            seg
            for seg in self.segments
            if seg.end >= start_time and seg.start <= end_time
        ]

        return " ".join(seg.text for seg in relevant_segments).strip()

    def format_with_timestamps(self) -> str:
        """Format transcript with timestamps for AI analysis."""
        lines = []
        for seg in self.segments:
            minutes = int(seg.start // 60)
            seconds = int(seg.start % 60)
            lines.append(f"[{minutes}:{seconds:02d}] {seg.text}")
        return "\n".join(lines)
