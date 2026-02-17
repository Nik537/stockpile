"""DuckDuckGo image search as emergency fallback source."""

import hashlib
import logging
from typing import Optional

from models.image import ImageResult
from services.image_sources.base import ImageSource

logger = logging.getLogger(__name__)


class DuckDuckGoImageSource(ImageSource):
    """DuckDuckGo image search as emergency fallback.

    Uses the duckduckgo-search library for completely free image search
    with no API key required.

    Install: pip install duckduckgo-search

    Limitations:
    - Rate limits are aggressive and undocumented
    - No retry logic (retrying makes rate limiting worse)
    - Best used as last-resort fallback when other sources fail

    This source always returns True for is_configured() since no API key is needed.
    """

    def __init__(self, max_results: int = 10):
        """Initialize DuckDuckGo image source.

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self._available: Optional[bool] = None

    def _check_available(self) -> bool:
        """Check if duckduckgo-search library is installed."""
        if self._available is not None:
            return self._available
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning(
                "[DuckDuckGo] duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )
            self._available = False
        return self._available

    def get_source_name(self) -> str:
        """Get the name of this image source."""
        return "duckduckgo"

    def is_configured(self) -> bool:
        """Always True if library is available - no API key needed."""
        return self._check_available()

    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search DuckDuckGo for images.

        No retry decorator - DDG rate limits are aggressive and undocumented.
        Single attempt, fail gracefully.

        Args:
            phrase: Search query string
            per_page: Maximum number of results (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        if not self._check_available():
            return []

        logger.debug(f"[DuckDuckGo] Searching for: '{phrase}'")

        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            raw_results = ddgs.images(
                phrase,
                max_results=min(per_page, self.max_results),
                safesearch="moderate",
                size="Large",
            )

            results = []
            for item in raw_results:
                result = self._parse_image(item)
                if result:
                    results.append(result)

            logger.debug(f"[DuckDuckGo] Found {len(results)} images")
            return results

        except ImportError:
            logger.error("[DuckDuckGo] duckduckgo-search not installed")
            return []
        except Exception as e:
            # Catch all DDG exceptions (RatelimitException, DuckDuckGoSearchException, etc.)
            error_name = type(e).__name__
            logger.warning(f"[DuckDuckGo] Search failed ({error_name}): {e}")
            return []

    def _parse_image(self, item: dict) -> Optional[ImageResult]:
        """Parse DuckDuckGo image result into ImageResult.

        Args:
            item: Result dict from duckduckgo-search

        Returns:
            ImageResult or None if parsing fails
        """
        try:
            download_url = item.get("image", "")
            if not download_url:
                return None

            # Generate unique ID from URL hash
            url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]
            image_id = f"ddg_{url_hash}"

            title = item.get("title", "")[:100] if item.get("title") else "DuckDuckGo Image"
            thumbnail = item.get("thumbnail", "")
            width = item.get("width", 0) or 0
            height = item.get("height", 0) or 0
            source_url = item.get("source", "")

            return ImageResult(
                image_id=image_id,
                title=title,
                url=source_url or download_url,
                download_url=download_url,
                width=width,
                height=height,
                source="duckduckgo",
                description=f"Via DuckDuckGo",
                license="Unknown",
                photographer=source_url,
                thumbnail_url=thumbnail,
            )

        except Exception as e:
            logger.warning(f"[DuckDuckGo] Failed to parse image: {e}")
            return None
