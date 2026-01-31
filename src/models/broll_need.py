"""Data models for timeline-aware B-roll planning."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BRollNeed:
    """A specific B-roll need at a point in the source video.

    Q2 Enhancement: Extended with metadata fields for smarter search and evaluation:
    - primary_search: Main search phrase (uses search_phrase for backwards compatibility)
    - alternate_searches: 2-3 synonym/related search phrases for fallback
    - negative_keywords: Terms to exclude from search results
    - visual_style: Preferred style (cinematic, documentary, raw, vlog)
    - time_of_day: Preferred lighting (golden hour, night, day)
    - movement: Camera movement preference (static, pan, drone, handheld)

    Feature 2: Extended with ±10 second context window fields:
    - context_before: Text from ~10 seconds before timestamp
    - context_after: Text from ~10 seconds after timestamp
    - themes: Key themes extracted from full context
    - emotional_tone: Detected emotional tone at this point
    """

    timestamp: float  # When in source video this B-roll is needed (seconds)
    search_phrase: str  # YouTube search query (primary search)
    description: str  # What this B-roll should show
    context: str  # Surrounding transcript text for reference
    suggested_duration: float = 5.0  # How long the B-roll should be (4-15s)

    # Fix 4: Semantic context preservation fields
    # These fields preserve the original transcript context throughout the entire pipeline,
    # ensuring AI evaluators and clip extractors understand exactly what visual is needed.
    original_context: str = ""
    """Full transcript segment (100+ characters) that this B-roll supports.

    Contains the exact words from the transcript that explain what visual is needed.
    This context is passed to all downstream AI services (evaluation, extraction)
    to ensure selected clips truly match the narrative meaning, not just keywords.

    Example: "Today I want to talk about how remote work has transformed the way
    we collaborate. More and more people are working from coffee shops and co-working
    spaces, using their laptops to stay connected with teams across the globe."
    """

    required_elements: List[str] = field(default_factory=list)
    """List of visual elements that MUST appear in the selected clip.

    These are extracted from the transcript context by AI during planning,
    representing concrete visual requirements that any matching clip must contain.
    Used during video evaluation and clip extraction to filter out clips that
    don't show the necessary visual elements.

    Examples: ["people", "laptops", "morning light", "coffee shop interior"]
    """

    # Q2 Enhanced search metadata fields
    alternate_searches: List[str] = field(default_factory=list)  # 2-3 synonym phrases
    negative_keywords: List[str] = field(default_factory=list)  # Terms to exclude
    visual_style: Optional[str] = None  # cinematic, documentary, raw, vlog
    time_of_day: Optional[str] = None  # golden hour, night, day, doesn't matter
    movement: Optional[str] = None  # static, pan, drone, handheld, tracking

    # Feature 2: Extended context window (±10 seconds)
    context_before: str = ""
    """Transcript text from ~10 seconds before this timestamp."""

    context_after: str = ""
    """Transcript text from ~10 seconds after this timestamp."""

    themes: List[str] = field(default_factory=list)
    """Key themes extracted from the extended context window."""

    entities: List[str] = field(default_factory=list)
    """Named entities (people, places, things) from context."""

    emotional_tone: Optional[str] = None
    """Detected emotional tone at this point (e.g., 'serious', 'excited', 'contemplative')."""

    def __post_init__(self):
        """Validate and clamp values."""
        self.suggested_duration = max(4.0, min(15.0, self.suggested_duration))
        self.timestamp = max(0.0, self.timestamp)

    @property
    def primary_search(self) -> str:
        """Return primary search phrase (same as search_phrase for compatibility)."""
        return self.search_phrase

    @property
    def all_search_phrases(self) -> List[str]:
        """Return all search phrases including alternates."""
        return [self.search_phrase] + list(self.alternate_searches)

    def has_enhanced_metadata(self) -> bool:
        """Check if this need has enhanced Q2 metadata (alternates/negatives)."""
        return bool(self.alternate_searches or self.negative_keywords or self.visual_style)

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
        return sanitized[:55].strip("_").lower()


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

    def get_context_window(
        self, timestamp: float, window_seconds: float = 10.0
    ) -> tuple[str, str, str]:
        """Get ±window_seconds of context around a timestamp.

        Feature 2: Context Window support.

        Args:
            timestamp: The target timestamp in seconds
            window_seconds: Seconds before and after to include (default 10.0)

        Returns:
            Tuple of (context_before, context_at, context_after)
            - context_before: Text from (timestamp - window_seconds) to timestamp
            - context_at: Text at the timestamp (±2 seconds)
            - context_after: Text from timestamp to (timestamp + window_seconds)
        """
        # Context before: window_seconds before timestamp
        before_start = max(0, timestamp - window_seconds)
        before_end = timestamp
        context_before = " ".join(
            seg.text for seg in self.segments
            if seg.start >= before_start and seg.end <= before_end
        ).strip()

        # Context at: ±2 seconds around timestamp
        at_start = max(0, timestamp - 2)
        at_end = timestamp + 2
        context_at = " ".join(
            seg.text for seg in self.segments
            if seg.end >= at_start and seg.start <= at_end
        ).strip()

        # Context after: timestamp to window_seconds after
        after_start = timestamp
        after_end = min(self.duration, timestamp + window_seconds)
        context_after = " ".join(
            seg.text for seg in self.segments
            if seg.start >= after_start and seg.start <= after_end
        ).strip()

        return (context_before, context_at, context_after)

    def get_full_context_window(
        self, timestamp: float, window_seconds: float = 10.0
    ) -> str:
        """Get the full combined context window as a single string.

        Args:
            timestamp: The target timestamp in seconds
            window_seconds: Seconds before and after to include

        Returns:
            Combined context string
        """
        before, at, after = self.get_context_window(timestamp, window_seconds)
        parts = [p for p in [before, at, after] if p]
        return " ".join(parts)

    def format_with_timestamps(self) -> str:
        """Format transcript with timestamps for AI analysis."""
        lines = []
        for seg in self.segments:
            minutes = int(seg.start // 60)
            seconds = int(seg.start % 60)
            lines.append(f"[{minutes}:{seconds:02d}] {seg.text}")
        return "\n".join(lines)
