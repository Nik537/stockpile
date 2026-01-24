"""Pexels video source for professional CC0-licensed stock footage."""

import logging
import os
from typing import Optional

import aiohttp

from models.video import VideoResult
from services.video_sources.base import VideoSource
from utils.retry import retry_api_call, NetworkError, TemporaryServiceError

logger = logging.getLogger(__name__)


class PexelsVideoSource(VideoSource):
    """Pexels video source for CC0-licensed stock footage.

    Pexels provides high-quality, royalty-free videos under the Pexels license
    (similar to CC0 - no attribution required for most uses).

    API Documentation: https://www.pexels.com/api/documentation/

    To get an API key:
    1. Create a free account at https://www.pexels.com
    2. Go to https://www.pexels.com/api/new/ to generate an API key
    """

    BASE_URL = "https://api.pexels.com/videos/search"

    def __init__(self, max_results: int = 10):
        """Initialize Pexels video source.

        Args:
            max_results: Maximum number of search results to return (max 80 per page)
        """
        self.max_results = min(max_results, 80)  # Pexels API limit
        self.api_key = os.getenv("PEXELS_API_KEY", "")

        if not self.api_key:
            logger.warning(
                "[Pexels] No API key configured. Set PEXELS_API_KEY to enable Pexels search."
            )

    def get_source_name(self) -> str:
        """Get the name of this video source."""
        return "pexels"

    def supports_section_downloads(self) -> bool:
        """Check if this source supports downloading specific time sections.

        Pexels provides direct download links, so we download the full video
        and extract sections locally with FFmpeg.
        """
        return False  # Need to download full video, then extract with FFmpeg

    def is_configured(self) -> bool:
        """Check if this source has required configuration."""
        return bool(self.api_key)

    @retry_api_call(max_retries=3, base_delay=1.0)
    def search_videos(self, phrase: str) -> list[VideoResult]:
        """Search Pexels for videos matching the search phrase.

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
            logger.debug("[Pexels] Skipping search - no API key configured")
            return []

        logger.info(f"[Pexels] Searching for: '{phrase}'")

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
            logger.error(f"[Pexels] Search failed for '{phrase}': {e}")
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
        headers = {"Authorization": self.api_key}
        params = {
            "query": phrase,
            "per_page": self.max_results,
            "orientation": "landscape",  # Prefer landscape for B-roll
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, headers=headers, params=params, timeout=30
                ) as response:
                    if response.status == 401:
                        logger.error("[Pexels] Invalid API key")
                        return []

                    if response.status == 429:
                        logger.warning("[Pexels] Rate limit exceeded")
                        raise TemporaryServiceError("Pexels rate limit exceeded")

                    if response.status != 200:
                        logger.warning(
                            f"[Pexels] API returned status {response.status}"
                        )
                        return []

                    data = await response.json()
                    videos = data.get("videos", [])

                    results = []
                    for video in videos:
                        result = self._parse_video(video)
                        if result:
                            results.append(result)

                    logger.info(f"[Pexels] Found {len(results)} videos")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Pexels] Network error: {e}")
            raise NetworkError(f"Pexels network error: {e}") from e

    def _parse_video(self, video: dict) -> Optional[VideoResult]:
        """Parse Pexels API video response into VideoResult.

        Args:
            video: Video dict from Pexels API

        Returns:
            VideoResult or None if parsing fails
        """
        try:
            video_id = str(video.get("id", ""))
            if not video_id:
                return None

            # Get video files - prefer HD quality
            video_files = video.get("video_files", [])
            if not video_files:
                return None

            # Sort by quality preference: HD > SD > others
            # Higher height is generally better quality
            video_files_sorted = sorted(
                video_files,
                key=lambda f: (
                    f.get("quality") == "hd",  # Prefer HD
                    f.get("height", 0),  # Then by height
                ),
                reverse=True,
            )

            best_file = video_files_sorted[0]
            download_url = best_file.get("link", "")

            # Get video page URL
            video_url = video.get("url", f"https://www.pexels.com/video/{video_id}/")

            # Extract title from URL or use fallback
            # Pexels URLs are like: https://www.pexels.com/video/title-here-12345/
            url_parts = video_url.rstrip("/").split("/")
            title = url_parts[-1] if url_parts else f"pexels_video_{video_id}"
            # Clean up title - remove video ID suffix if present
            if title.endswith(f"-{video_id}"):
                title = title[: -len(f"-{video_id}")]
            title = title.replace("-", " ").title()

            return VideoResult(
                video_id=f"pexels_{video_id}",
                title=title,
                url=video_url,
                duration=video.get("duration", 0),
                description=f"Pexels video by {video.get('user', {}).get('name', 'Unknown')}",
                source="pexels",
                license="Pexels License (CC0-like)",
                download_url=download_url,
            )

        except Exception as e:
            logger.warning(f"[Pexels] Failed to parse video: {e}")
            return None

    def _handle_error(self, error: Exception) -> None:
        """Convert errors to retryable types where appropriate.

        Args:
            error: The exception to handle
        """
        error_msg = str(error).lower()
        if "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            raise NetworkError(f"Pexels network error: {error}") from error
        if "rate" in error_msg or "429" in error_msg:
            raise TemporaryServiceError(f"Pexels rate limit: {error}") from error
