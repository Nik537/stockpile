"""Pixabay video source for professional CC0-licensed stock footage."""

import logging
import os
from typing import Optional

import aiohttp

from models.video import VideoResult
from services.video_sources.base import VideoSource
from utils.retry import retry_api_call, NetworkError, TemporaryServiceError

logger = logging.getLogger(__name__)


class PixabayVideoSource(VideoSource):
    """Pixabay video source for CC0-licensed stock footage.

    Pixabay provides high-quality, royalty-free videos under the Pixabay License
    (similar to CC0 - no attribution required for most uses).

    API Documentation: https://pixabay.com/api/docs/

    To get an API key:
    1. Create a free account at https://pixabay.com
    2. API key is shown at https://pixabay.com/api/docs/ when logged in
    """

    BASE_URL = "https://pixabay.com/api/videos/"

    def __init__(self, max_results: int = 10):
        """Initialize Pixabay video source.

        Args:
            max_results: Maximum number of search results to return (max 200 per page)
        """
        self.max_results = min(max_results, 200)  # Pixabay API limit
        self.api_key = os.getenv("PIXABAY_API_KEY", "")

        if not self.api_key:
            logger.warning(
                "[Pixabay] No API key configured. Set PIXABAY_API_KEY to enable Pixabay search."
            )

    def get_source_name(self) -> str:
        """Get the name of this video source."""
        return "pixabay"

    def supports_section_downloads(self) -> bool:
        """Check if this source supports downloading specific time sections.

        Pixabay provides direct download links, so we download the full video
        and extract sections locally with FFmpeg.
        """
        return False  # Need to download full video, then extract with FFmpeg

    def is_configured(self) -> bool:
        """Check if this source has required configuration."""
        return bool(self.api_key)

    @retry_api_call(max_retries=3, base_delay=1.0)
    def search_videos(self, phrase: str) -> list[VideoResult]:
        """Search Pixabay for videos matching the search phrase.

        Uses synchronous requests via aiohttp to maintain compatibility
        with the VideoSource interface. For better performance in async
        contexts, use search_videos_async().

        Args:
            phrase: Search query string

        Returns:
            List of VideoResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        if not self.api_key:
            logger.debug("[Pixabay] Skipping search - no API key configured")
            return []

        logger.info(f"[Pixabay] Searching for: '{phrase}'")

        import asyncio

        # Run async search in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self._search_async(phrase))
        except Exception as e:
            logger.error(f"[Pixabay] Search failed for '{phrase}': {e}")
            self._handle_error(e)
            return []

    async def search_videos_async(self, phrase: str) -> list[VideoResult]:
        """Async version of search for better performance in async contexts.

        Args:
            phrase: Search query string

        Returns:
            List of VideoResult objects
        """
        if not phrase.strip() or not self.api_key:
            return []

        return await self._search_async(phrase)

    async def _search_async(self, phrase: str) -> list[VideoResult]:
        """Internal async search implementation.

        Args:
            phrase: Search query string

        Returns:
            List of VideoResult objects
        """
        params = {
            "key": self.api_key,
            "q": phrase,
            "per_page": self.max_results,
            "video_type": "all",  # film, animation, or all
            "safesearch": "true",  # Safe content only
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, params=params, timeout=30
                ) as response:
                    if response.status == 401:
                        logger.error("[Pixabay] Invalid API key")
                        return []

                    if response.status == 429:
                        logger.warning("[Pixabay] Rate limit exceeded")
                        raise TemporaryServiceError("Pixabay rate limit exceeded")

                    if response.status != 200:
                        logger.warning(
                            f"[Pixabay] API returned status {response.status}"
                        )
                        return []

                    data = await response.json()
                    videos = data.get("hits", [])

                    results = []
                    for video in videos:
                        result = self._parse_video(video)
                        if result:
                            results.append(result)

                    logger.info(f"[Pixabay] Found {len(results)} videos")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Pixabay] Network error: {e}")
            raise NetworkError(f"Pixabay network error: {e}") from e

    def _parse_video(self, video: dict) -> Optional[VideoResult]:
        """Parse Pixabay API video response into VideoResult.

        Args:
            video: Video dict from Pixabay API

        Returns:
            VideoResult or None if parsing fails
        """
        try:
            video_id = str(video.get("id", ""))
            if not video_id:
                return None

            # Get video files - Pixabay provides different sizes
            videos_dict = video.get("videos", {})
            if not videos_dict:
                return None

            # Quality preference order: large (1920x1080) > medium (1280x720) > small > tiny
            quality_order = ["large", "medium", "small", "tiny"]
            download_url = ""
            best_quality = None

            for quality in quality_order:
                if quality in videos_dict and videos_dict[quality].get("url"):
                    download_url = videos_dict[quality]["url"]
                    best_quality = quality
                    break

            if not download_url:
                return None

            # Get video page URL
            video_url = video.get("pageURL", f"https://pixabay.com/videos/id-{video_id}/")

            # Get tags for title (Pixabay doesn't provide titles)
            tags = video.get("tags", "")
            title = tags.replace(",", " ").strip() if tags else f"pixabay_video_{video_id}"
            # Clean up title
            title = " ".join(title.split()[:5])  # First 5 words
            title = title.title()

            # Calculate duration from video info if available
            duration = video.get("duration", 0)

            return VideoResult(
                video_id=f"pixabay_{video_id}",
                title=title,
                url=video_url,
                duration=duration,
                description=f"Pixabay video by {video.get('user', 'Unknown')}. Tags: {tags}",
                source="pixabay",
                license="Pixabay License (CC0-like)",
                download_url=download_url,
            )

        except Exception as e:
            logger.warning(f"[Pixabay] Failed to parse video: {e}")
            return None

    def _handle_error(self, error: Exception) -> None:
        """Convert errors to retryable types where appropriate.

        Args:
            error: The exception to handle
        """
        error_msg = str(error).lower()
        if "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            raise NetworkError(f"Pixabay network error: {error}") from error
        if "rate" in error_msg or "429" in error_msg:
            raise TemporaryServiceError(f"Pixabay rate limit: {error}") from error
