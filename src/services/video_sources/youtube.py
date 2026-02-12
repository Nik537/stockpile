"""YouTube video source implementation using yt-dlp."""

import asyncio
import logging

import yt_dlp
from models.video import VideoResult
from services.video_sources.base import VideoSource
from utils.config import load_config
from utils.retry import NetworkError, TemporaryServiceError, retry_api_call

logger = logging.getLogger(__name__)

# Configure yt-dlp logging to be silent
logging.getLogger("yt_dlp").setLevel(logging.CRITICAL)
logging.getLogger("yt_dlp.extractor").setLevel(logging.CRITICAL)
logging.getLogger("yt_dlp.downloader").setLevel(logging.CRITICAL)


def video_filter(info: dict, max_duration: int | None = None) -> str | None:
    """Filter videos based on duration and other criteria.

    Args:
        info: Video information dictionary from yt-dlp
        max_duration: Override max duration in seconds (None = use config)

    Returns:
        String describing why video was filtered, or None if it passes
    """
    if max_duration is None:
        config = load_config()
        max_duration = config.get("max_video_duration_seconds", 600)
    max_size = 100 * 1024 * 1024  # 100MB

    # Check duration
    duration = info.get("duration")
    if duration is not None and duration > max_duration:
        return f"Duration {duration}s exceeds maximum {max_duration}s"

    size = info.get("filesize")
    if size is not None and size > max_size:
        return f"Size {size} exceeds maximum {max_size} bytes"

    return None


class YouTubeVideoSource(VideoSource):
    """YouTube video source using yt-dlp for search and download."""

    def __init__(self, max_results: int = 20, max_duration: int | None = None):
        """Initialize YouTube video source.

        Args:
            max_results: Maximum number of search results to return
            max_duration: Max video duration in seconds for filtering (None = use config)
        """
        self.max_results = max_results
        self.max_duration = max_duration

        self.ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "ignoreerrors": True,
        }

    def get_source_name(self) -> str:
        """Get the name of this video source."""
        return "youtube"

    def supports_section_downloads(self) -> bool:
        """Check if this source supports downloading specific time sections."""
        return True  # YouTube supports section downloads via yt-dlp

    def is_configured(self) -> bool:
        """YouTube doesn't require an API key (uses yt-dlp scraping).

        Check if yt-dlp is importable and if YouTube B-roll is enabled in config.
        """
        try:
            config = load_config()
            return config.get("youtube_broll_enabled", True)
        except Exception:
            return True  # Default to enabled if config can't be loaded

    @retry_api_call(max_retries=3, base_delay=2.0)
    def search_videos(self, phrase: str) -> list[VideoResult]:
        """Search YouTube for videos matching the search phrase.

        Args:
            phrase: Search query string

        Returns:
            List of VideoResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        logger.info(f"[YouTube] Searching for: '{phrase}'")

        try:
            search_query = f"ytsearch{self.max_results}:{phrase}"

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                search_results = ydl.extract_info(search_query, download=False)

                if not search_results or "entries" not in search_results:
                    return []

                video_results = []
                filtered_count = 0
                for entry in search_results["entries"]:
                    if entry is None:
                        continue

                    # Filter video before parsing
                    filter_result = video_filter(entry, max_duration=self.max_duration)
                    if filter_result:
                        filtered_count += 1
                        logger.debug(f"Video {entry.get('id', 'unknown')} filtered: {filter_result}")
                        continue

                    video_result = self._parse_video_entry(entry)
                    if video_result:
                        video_results.append(video_result)

                if filtered_count > 0:
                    logger.info(f"[YouTube] Filtered out {filtered_count} videos that didn't meet criteria")

                logger.info(f"[YouTube] Found {len(video_results)} videos")
                return video_results

        except Exception as e:
            logger.error(f"[YouTube] Search failed for '{phrase}': {e}")
            if "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(f"Network error: {e}") from e
            if "unavailable" in str(e).lower():
                raise TemporaryServiceError(f"YouTube unavailable: {e}") from e
            raise

    async def search_videos_async(self, phrase: str) -> list[VideoResult]:
        """Async version of search_videos - runs blocking yt-dlp in a thread pool.

        Args:
            phrase: Search query string

        Returns:
            List of VideoResult objects
        """
        if not phrase.strip():
            return []

        return await asyncio.to_thread(self.search_videos, phrase)

    def search_broll(
        self,
        keywords: list[str],
        visual_style: str = "",
        max_results: int = 5,
    ) -> list[VideoResult]:
        """Search YouTube specifically for B-roll footage.

        Enhances the query with B-roll-relevant terms and the visual style.

        Args:
            keywords: Visual keywords to search for
            visual_style: Style hint (e.g., "aerial", "cinematic", "close-up")
            max_results: Max results to return

        Returns:
            List of VideoResult objects suitable for B-roll
        """
        base_query = " ".join(keywords[:3])

        # Enhance query with style and stock footage terms
        query_parts = [base_query]
        if visual_style:
            query_parts.append(visual_style)
        query_parts.append("stock footage b-roll")

        enhanced_query = " ".join(query_parts)

        # Temporarily override max_results for this search
        original_max = self.max_results
        self.max_results = max_results
        try:
            results = self.search_videos(enhanced_query)
        finally:
            self.max_results = original_max

        return results

    async def search_broll_async(
        self,
        keywords: list[str],
        visual_style: str = "",
        max_results: int = 5,
    ) -> list[VideoResult]:
        """Async version of search_broll.

        Args:
            keywords: Visual keywords to search for
            visual_style: Style hint (e.g., "aerial", "cinematic", "close-up")
            max_results: Max results to return

        Returns:
            List of VideoResult objects suitable for B-roll
        """
        return await asyncio.to_thread(
            self.search_broll, keywords, visual_style, max_results
        )

    def _parse_video_entry(self, entry: dict) -> VideoResult | None:
        """Parse a yt-dlp video entry into a VideoResult object.

        Args:
            entry: Dictionary from yt-dlp with video information

        Returns:
            VideoResult object or None if parsing failed
        """
        try:
            video_id = entry.get("id")
            if not video_id:
                return None

            return VideoResult(
                video_id=video_id,
                title=entry.get("title", "Unknown Title"),
                url=f"https://www.youtube.com/watch?v={video_id}",
                duration=entry.get("duration", 0) or 0,
                description=entry.get("description", ""),
                source="youtube",
                view_count=entry.get("view_count"),
                channel=entry.get("channel") or entry.get("uploader"),
            )

        except Exception as e:
            logger.warning(f"Failed to parse video entry: {e}")
            return None
