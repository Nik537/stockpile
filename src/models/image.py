"""Data models for image acquisition."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ImageResult:
    """Represents an image search result from any source.

    Supports Pexels Photos, Pixabay Images, Google Images, and other sources.
    Similar to VideoResult but optimized for still images.
    """

    image_id: str
    title: str
    url: str  # Page URL where image is hosted
    download_url: str  # Direct download URL for the image
    width: int
    height: int
    source: str  # pexels, pixabay, google, etc.
    description: Optional[str] = None
    license: Optional[str] = None  # CC0, Pexels License, etc.
    photographer: Optional[str] = None  # Credit for the image
    thumbnail_url: Optional[str] = None  # Small preview for AI evaluation

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def is_landscape(self) -> bool:
        """Check if image is landscape orientation."""
        return self.aspect_ratio > 1.0

    @property
    def resolution(self) -> str:
        """Return resolution string."""
        return f"{self.width}x{self.height}"


@dataclass
class ImageNeed:
    """A specific image need at a point in the source video.

    Generated every N seconds (default: 5) to provide visual aids
    throughout the video timeline.

    Feature 2: Includes ±10 second context window for better semantic matching.
    """

    timestamp: float  # When in source video this image is needed (seconds)
    search_phrase: str  # Image search query
    context: str  # Transcript text at this timestamp for reference

    # Enhanced metadata for better image selection
    required_elements: List[str] = field(default_factory=list)
    """Visual elements that should appear in the image."""

    visual_style: Optional[str] = None
    """Preferred style: cinematic, documentary, professional, etc."""

    # Feature 2: Extended context window (±10 seconds)
    context_before: str = ""
    """Transcript text from ~10 seconds before this timestamp."""

    context_after: str = ""
    """Transcript text from ~10 seconds after this timestamp."""

    full_context: str = ""
    """Combined context (before + current + after)."""

    themes: List[str] = field(default_factory=list)
    """Key themes extracted from the extended context."""

    entities: List[str] = field(default_factory=list)
    """Named entities (people, places, things) from context."""

    emotional_tone: Optional[str] = None
    """Detected emotional tone at this point (e.g., 'serious', 'excited', 'contemplative')."""

    def __post_init__(self):
        """Validate values."""
        self.timestamp = max(0.0, self.timestamp)

    def get_enhanced_context(self) -> str:
        """Get the best available context for AI prompts.

        Returns full context if available, otherwise falls back to basic context.
        """
        if self.full_context:
            return self.full_context
        return self.context

    @property
    def folder_name(self) -> str:
        """Generate timestamp-prefixed folder name for this image need.

        Format: {timestamp}_{sanitized_search_phrase}
        Example: 0m30s_mma_fighter_training
        """
        minutes = int(self.timestamp // 60)
        seconds = int(self.timestamp % 60)
        timestamp = f"{minutes}m{seconds:02d}s"

        # Include sanitized search phrase for easy identification
        phrase_slug = self._sanitize_for_folder(self.search_phrase)
        if phrase_slug:
            return f"{timestamp}_{phrase_slug}"
        return timestamp

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
class ImagePlan:
    """Complete image acquisition plan for a source video."""

    source_duration: float  # Total source video length in seconds
    needs: List[ImageNeed] = field(default_factory=list)
    interval_seconds: float = 5.0  # Target interval between images
    source_file: Optional[str] = None  # Original input file path

    @property
    def expected_image_count(self) -> int:
        """Calculate expected number of images based on duration and interval."""
        return max(1, int(self.source_duration / self.interval_seconds))

    @property
    def actual_image_count(self) -> int:
        """Return actual number of image needs identified."""
        return len(self.needs)

    def get_needs_sorted_by_timestamp(self) -> List[ImageNeed]:
        """Return needs sorted by timestamp in the source video."""
        return sorted(self.needs, key=lambda n: n.timestamp)


@dataclass
class ScoredImage:
    """Represents an image with AI evaluation score."""

    image_id: str
    score: int  # 1-10 rating from AI evaluator
    image_result: ImageResult
    reason: str = ""  # Why this image was selected/rejected
