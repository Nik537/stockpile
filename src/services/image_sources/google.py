"""Google Images source via Google Custom Search Engine (CSE) JSON API."""

import hashlib
import logging
import os
from typing import Optional

import aiohttp

from models.image import ImageResult
from services.image_sources.base import ImageSource
from utils.retry import NetworkError, TemporaryServiceError, retry_api_call

logger = logging.getLogger(__name__)


class GoogleImageSource(ImageSource):
    """Google Images source via Custom Search Engine (CSE) JSON API.

    Uses Google's official Custom Search JSON API for image search.
    Provides 100 free queries/day (3,000/month).

    API Documentation: https://developers.google.com/custom-search/v1/overview

    Setup:
    1. Create a Programmable Search Engine at https://programmablesearchengine.google.com/
       - Enable "Search the entire web" and "Image search"
       - Copy the Search Engine ID (cx)
    2. Get an API key at https://console.cloud.google.com/
       - Enable "Custom Search API"
       - Create credentials -> API key
    3. Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX in .env

    Rate limits: 100 queries/day free, then $5 per 1000 queries
    """

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, max_results: int = 10):
        """Initialize Google image source via CSE.

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = min(max_results, 10)  # CSE returns max 10 per request
        self.api_key = os.getenv("GOOGLE_CSE_API_KEY", "")
        self.cx = os.getenv("GOOGLE_CSE_CX", "")

        if not self.api_key or not self.cx:
            logger.warning(
                "[Google Images] Not configured. Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX to enable."
            )

    def get_source_name(self) -> str:
        """Get the name of this image source."""
        return "google"

    def is_configured(self) -> bool:
        """Check if this source has required configuration."""
        return bool(self.api_key and self.cx)

    @retry_api_call(max_retries=3, base_delay=1.0)
    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search Google Images via Custom Search Engine JSON API.

        Args:
            phrase: Search query string
            per_page: Maximum number of results (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        if not self.api_key or not self.cx:
            logger.debug("[Google Images] Skipping search - not configured")
            return []

        logger.debug(f"[Google Images] Searching for: '{phrase}'")

        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": phrase,
            "searchType": "image",
            "num": min(per_page, self.max_results),
            "imgSize": "large",
            "safe": "active",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, params=params, timeout=30
                ) as response:
                    if response.status == 401:
                        logger.error("[Google Images] Invalid API key")
                        return []

                    if response.status == 429:
                        logger.warning("[Google Images] Daily quota exceeded")
                        raise TemporaryServiceError("Google CSE daily quota exceeded")

                    if response.status == 403:
                        data = await response.json()
                        error_msg = data.get("error", {}).get("message", "")
                        if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                            logger.warning(f"[Google Images] Quota exceeded: {error_msg}")
                            raise TemporaryServiceError("Google CSE quota exceeded")
                        logger.error(f"[Google Images] Forbidden: {error_msg}")
                        return []

                    if response.status != 200:
                        logger.warning(
                            f"[Google Images] API returned status {response.status}"
                        )
                        return []

                    data = await response.json()

                    if "error" in data:
                        logger.error(f"[Google Images] API error: {data['error']}")
                        return []

                    # Log remaining quota info if available
                    total_results = data.get("searchInformation", {}).get("totalResults", "?")
                    logger.debug(f"[Google Images] Total results available: {total_results}")

                    items = data.get("items", [])

                    results = []
                    for idx, item in enumerate(items):
                        if idx >= per_page:
                            break
                        result = self._parse_image(item, idx)
                        if result:
                            results.append(result)

                    logger.debug(f"[Google Images] Found {len(results)} images")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Google Images] Network error: {e}")
            raise NetworkError(f"Google Images network error: {e}") from e

    def _parse_image(self, item: dict, idx: int) -> Optional[ImageResult]:
        """Parse Google CSE response item into ImageResult.

        Args:
            item: Item dict from CSE response
            idx: Index for generating unique ID

        Returns:
            ImageResult or None if parsing fails
        """
        try:
            # Direct image URL
            download_url = item.get("link", "")
            if not download_url:
                return None

            image_info = item.get("image", {})

            # Thumbnail URL
            thumbnail = image_info.get("thumbnailLink", "")

            # Dimensions
            width = image_info.get("width", 0)
            height = image_info.get("height", 0)

            # Title
            title = item.get("title", "")[:100] if item.get("title") else f"Google Image {idx + 1}"

            # Source domain
            source_domain = item.get("displayLink", "")

            # Page URL where image is hosted
            page_url = image_info.get("contextLink", download_url)

            # Generate unique ID from URL hash
            url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]
            image_id = f"google_{url_hash}"

            return ImageResult(
                image_id=image_id,
                title=title,
                url=page_url,
                download_url=download_url,
                width=width,
                height=height,
                source="google",
                description=f"From {source_domain}" if source_domain else None,
                license="Unknown (Google Images)",
                photographer=source_domain,
                thumbnail_url=thumbnail,
            )

        except Exception as e:
            logger.warning(f"[Google Images] Failed to parse image: {e}")
            return None

    async def download_image(self, image: ImageResult, output_path: str) -> Optional[str]:
        """Download an image from Google Images.

        Override base implementation to add custom headers for
        potentially problematic URLs from various sources.

        Args:
            image: ImageResult with download_url
            output_path: Full path where image should be saved

        Returns:
            Path to downloaded file, or None if download failed
        """
        if not image.download_url:
            logger.warning(f"[Google Images] No download URL for image {image.image_id}")
            return None

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/*,*/*;q=0.8",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    image.download_url, headers=headers, timeout=60, allow_redirects=True
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"[Google Images] Download failed with status {response.status} for {image.image_id}"
                        )
                        return None

                    content = await response.read()

                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"[Google Images] Unexpected content type: {content_type}"
                        )
                        return None

                    ext = self._get_extension(content_type, image.download_url)

                    if not output_path.endswith(ext):
                        output_path = f"{output_path}{ext}"

                    with open(output_path, "wb") as f:
                        f.write(content)

                    logger.debug(f"[Google Images] Downloaded {image.image_id} to {output_path}")
                    return output_path

        except aiohttp.ClientError as e:
            logger.warning(f"[Google Images] Network error downloading {image.image_id}: {e}")
            return None
        except OSError as e:
            logger.warning(f"[Google Images] File error saving {image.image_id}: {e}")
            return None
