"""Pixabay image source for CC0-licensed stock photos."""

import logging
import os
from typing import Optional

import aiohttp

from models.image import ImageResult
from services.image_sources.base import ImageSource
from utils.retry import NetworkError, TemporaryServiceError, retry_api_call

logger = logging.getLogger(__name__)


class PixabayImageSource(ImageSource):
    """Pixabay image source for CC0-licensed stock photos.

    Pixabay provides royalty-free images under the Pixabay Content License
    (similar to CC0 - free for commercial and noncommercial use).

    API Documentation: https://pixabay.com/api/docs/

    To get an API key:
    1. Create a free account at https://pixabay.com
    2. Go to https://pixabay.com/api/docs/ to view your API key

    Rate limits: 100 requests per minute
    """

    BASE_URL = "https://pixabay.com/api/"

    def __init__(self, max_results: int = 10):
        """Initialize Pixabay image source.

        Args:
            max_results: Maximum number of search results to return (max 200 per page)
        """
        self.max_results = min(max_results, 200)  # Pixabay API limit
        self.api_key = os.getenv("PIXABAY_API_KEY", "")

        if not self.api_key:
            logger.warning(
                "[Pixabay Images] No API key configured. Set PIXABAY_API_KEY to enable Pixabay search."
            )

    def get_source_name(self) -> str:
        """Get the name of this image source."""
        return "pixabay"

    def is_configured(self) -> bool:
        """Check if this source has required configuration."""
        return bool(self.api_key)

    @retry_api_call(max_retries=3, base_delay=1.0)
    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search Pixabay for images matching the search phrase.

        Args:
            phrase: Search query string
            per_page: Maximum number of results (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        if not self.api_key:
            logger.debug("[Pixabay Images] Skipping search - no API key configured")
            return []

        logger.debug(f"[Pixabay Images] Searching for: '{phrase}'")

        # Pixabay requires per_page between 3-200, so we request min 3 and slice results
        actual_per_page = max(3, min(per_page, self.max_results))
        params = {
            "key": self.api_key,
            "q": phrase,
            "per_page": actual_per_page,
            "image_type": "photo",  # Photos only, not illustrations
            "safesearch": "true",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, params=params, timeout=30
                ) as response:
                    if response.status == 401:
                        logger.error("[Pixabay Images] Invalid API key")
                        return []

                    if response.status == 429:
                        logger.warning("[Pixabay Images] Rate limit exceeded")
                        raise TemporaryServiceError("Pixabay rate limit exceeded")

                    if response.status != 200:
                        text = await response.text()
                        logger.warning(
                            f"[Pixabay Images] API returned status {response.status}: {text[:200]}"
                        )
                        return []

                    data = await response.json()
                    hits = data.get("hits", [])

                    results = []
                    for hit in hits:
                        result = self._parse_hit(hit)
                        if result:
                            results.append(result)

                    # Slice to originally requested per_page (we may have fetched more due to Pixabay's min=3)
                    results = results[:per_page]
                    logger.debug(f"[Pixabay Images] Found {len(results)} images")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Pixabay Images] Network error: {e}")
            raise NetworkError(f"Pixabay network error: {e}") from e

    def _parse_hit(self, hit: dict) -> Optional[ImageResult]:
        """Parse Pixabay API hit response into ImageResult.

        Args:
            hit: Hit dict from Pixabay API

        Returns:
            ImageResult or None if parsing fails
        """
        try:
            image_id = str(hit.get("id", ""))
            if not image_id:
                return None

            # Get image URLs - prefer largeImageURL for quality
            # Note: Full resolution requires editorial/commercial API access
            download_url = hit.get("largeImageURL") or hit.get("webformatURL", "")
            thumbnail_url = hit.get("previewURL", "")

            if not download_url:
                return None

            # Get dimensions (from webformat image)
            width = hit.get("imageWidth", hit.get("webformatWidth", 0))
            height = hit.get("imageHeight", hit.get("webformatHeight", 0))

            # Get page URL
            page_url = hit.get("pageURL", f"https://pixabay.com/photos/{image_id}/")

            # Get user info
            user = hit.get("user", "Unknown")

            # Generate title from tags
            tags = hit.get("tags", "")
            if tags:
                # Clean up tags to make a title
                title = tags.replace(",", " -").title()[:100]
            else:
                title = f"Pixabay Image {image_id}"

            return ImageResult(
                image_id=f"pixabay_{image_id}",
                title=title,
                url=page_url,
                download_url=download_url,
                width=width,
                height=height,
                source="pixabay",
                description=tags or None,
                license="Pixabay Content License (CC0-like)",
                photographer=user,
                thumbnail_url=thumbnail_url,
            )

        except Exception as e:
            logger.warning(f"[Pixabay Images] Failed to parse hit: {e}")
            return None
