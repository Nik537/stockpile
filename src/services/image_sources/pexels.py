"""Pexels image source for professional CC0-licensed stock photos."""

import logging
import os
from typing import Optional

import aiohttp

from models.image import ImageResult
from services.image_sources.base import ImageSource
from utils.retry import NetworkError, TemporaryServiceError, retry_api_call

logger = logging.getLogger(__name__)


class PexelsImageSource(ImageSource):
    """Pexels image source for CC0-licensed stock photos.

    Pexels provides high-quality, royalty-free images under the Pexels license
    (similar to CC0 - no attribution required for most uses).

    API Documentation: https://www.pexels.com/api/documentation/

    To get an API key:
    1. Create a free account at https://www.pexels.com
    2. Go to https://www.pexels.com/api/new/ to generate an API key

    Rate limits: 200 requests per hour, 20,000 requests per month
    """

    BASE_URL = "https://api.pexels.com/v1/search"

    def __init__(self, max_results: int = 10):
        """Initialize Pexels image source.

        Args:
            max_results: Maximum number of search results to return (max 80 per page)
        """
        self.max_results = min(max_results, 80)  # Pexels API limit
        self.api_key = os.getenv("PEXELS_API_KEY", "")

        if not self.api_key:
            logger.warning(
                "[Pexels Images] No API key configured. Set PEXELS_API_KEY to enable Pexels search."
            )

    def get_source_name(self) -> str:
        """Get the name of this image source."""
        return "pexels"

    def is_configured(self) -> bool:
        """Check if this source has required configuration."""
        return bool(self.api_key)

    @retry_api_call(max_retries=3, base_delay=1.0)
    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search Pexels for images matching the search phrase.

        Args:
            phrase: Search query string
            per_page: Maximum number of results (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        if not self.api_key:
            logger.debug("[Pexels Images] Skipping search - no API key configured")
            return []

        logger.debug(f"[Pexels Images] Searching for: '{phrase}'")

        headers = {"Authorization": self.api_key}
        params = {
            "query": phrase,
            "per_page": min(per_page, self.max_results),
            "orientation": "landscape",  # Prefer landscape for video supplements
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, headers=headers, params=params, timeout=30
                ) as response:
                    if response.status == 401:
                        logger.error("[Pexels Images] Invalid API key")
                        return []

                    if response.status == 429:
                        logger.warning("[Pexels Images] Rate limit exceeded")
                        raise TemporaryServiceError("Pexels rate limit exceeded")

                    if response.status != 200:
                        logger.warning(
                            f"[Pexels Images] API returned status {response.status}"
                        )
                        return []

                    data = await response.json()
                    photos = data.get("photos", [])

                    results = []
                    for photo in photos:
                        result = self._parse_photo(photo)
                        if result:
                            results.append(result)

                    logger.debug(f"[Pexels Images] Found {len(results)} images")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Pexels Images] Network error: {e}")
            raise NetworkError(f"Pexels network error: {e}") from e

    def _parse_photo(self, photo: dict) -> Optional[ImageResult]:
        """Parse Pexels API photo response into ImageResult.

        Args:
            photo: Photo dict from Pexels API

        Returns:
            ImageResult or None if parsing fails
        """
        try:
            photo_id = str(photo.get("id", ""))
            if not photo_id:
                return None

            # Get image URLs - prefer large size for quality
            src = photo.get("src", {})
            if not src:
                return None

            # Use 'large' for downloads, 'medium' for thumbnails
            download_url = src.get("large2x") or src.get("large") or src.get("original", "")
            thumbnail_url = src.get("medium") or src.get("small", "")

            if not download_url:
                return None

            # Get dimensions
            width = photo.get("width", 0)
            height = photo.get("height", 0)

            # Get page URL
            photo_url = photo.get("url", f"https://www.pexels.com/photo/{photo_id}/")

            # Get photographer
            photographer = photo.get("photographer", "Unknown")

            # Extract title from alt text or generate from URL
            alt_text = photo.get("alt", "")
            if alt_text:
                title = alt_text[:100]  # Limit title length
            else:
                title = f"Pexels Photo {photo_id}"

            return ImageResult(
                image_id=f"pexels_{photo_id}",
                title=title,
                url=photo_url,
                download_url=download_url,
                width=width,
                height=height,
                source="pexels",
                description=alt_text or None,
                license="Pexels License (CC0-like)",
                photographer=photographer,
                thumbnail_url=thumbnail_url,
            )

        except Exception as e:
            logger.warning(f"[Pexels Images] Failed to parse photo: {e}")
            return None
