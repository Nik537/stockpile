"""Base abstraction for video sources."""

from abc import ABC, abstractmethod

from models.video import VideoResult


class VideoSource(ABC):
    """Abstract base class for video sources (YouTube, Pexels, Vimeo, etc.)."""

    @abstractmethod
    def search_videos(self, phrase: str) -> list[VideoResult]:
        """Search for videos matching the search phrase.

        Args:
            phrase: Search query string

        Returns:
            List of VideoResult objects matching the search criteria
        """

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this video source.

        Returns:
            Source name (e.g., "youtube", "pexels", "vimeo")
        """

    @abstractmethod
    def supports_section_downloads(self) -> bool:
        """Check if this source supports downloading specific time sections.

        Returns:
            True if source supports downloading clip sections, False otherwise
        """

    def is_configured(self) -> bool:
        """Check if this source has required configuration (API keys, etc.).

        Default implementation returns True (no config required).
        Override in subclasses that require API keys.

        Returns:
            True if source is properly configured and ready to use
        """
        return True
