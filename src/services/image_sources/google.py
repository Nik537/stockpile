"""Google Images source via SerpAPI for web image search."""

import logging
import os
from typing import Optional

import aiohttp

from models.image import ImageResult
from services.image_sources.base import ImageSource
from utils.retry import NetworkError, TemporaryServiceError, retry_api_call

logger = logging.getLogger(__name__)


class GoogleImageSource(ImageSource):
    """Google Images source via SerpAPI.

    Uses SerpAPI to access Google Images search results programmatically.
    Provides access to a wider variety of images than stock photo sites.

    API Documentation: https://serpapi.com/google-images-api

    To get an API key:
    1. Create an account at https://serpapi.com
    2. Copy your API key from the dashboard

    Rate limits: Depends on plan (100-5000 searches/month)

    Note: Images from Google may have various licensing. This source
    filters for images with creative commons licenses when possible.
    """

    BASE_URL = "https://serpapi.com/search"

    def __init__(self, max_results: int = 10):
        """Initialize Google image source via SerpAPI.

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = min(max_results, 100)  # SerpAPI limit
        self.api_key = os.getenv("SERPAPI_KEY", "")

        if not self.api_key:
            logger.warning(
                "[Google Images] No API key configured. Set SERPAPI_KEY to enable Google Images search."
            )

    def get_source_name(self) -> str:
        """Get the name of this image source."""
        return "google"

    def is_configured(self) -> bool:
        """Check if this source has required configuration."""
        return bool(self.api_key)

    @retry_api_call(max_retries=3, base_delay=1.0)
    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search Google Images for matching images via SerpAPI.

        Args:
            phrase: Search query string
            per_page: Maximum number of results (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        if not self.api_key:
            logger.debug("[Google Images] Skipping search - no API key configured")
            return []

        logger.debug(f"[Google Images] Searching for: '{phrase}'")

        params = {
            "api_key": self.api_key,
            "engine": "google_images",
            "q": phrase,
            "num": min(per_page, self.max_results),
            # Filter for images that can be reused
            "tbs": "itp:photo,isz:l",  # Large photos only
            "safe": "active",  # Safe search enabled
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
                        logger.warning("[Google Images] Rate limit exceeded")
                        raise TemporaryServiceError("SerpAPI rate limit exceeded")

                    if response.status != 200:
                        logger.warning(
                            f"[Google Images] API returned status {response.status}"
                        )
                        return []

                    data = await response.json()

                    # Check for API errors
                    if "error" in data:
                        logger.error(f"[Google Images] API error: {data['error']}")
                        return []

                    images_results = data.get("images_results", [])

                    results = []
                    for idx, image in enumerate(images_results):
                        if idx >= per_page:
                            break
                        result = self._parse_image(image, idx)
                        if result:
                            results.append(result)

                    logger.debug(f"[Google Images] Found {len(results)} images")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Google Images] Network error: {e}")
            raise NetworkError(f"Google Images network error: {e}") from e

    def _parse_image(self, image: dict, idx: int) -> Optional[ImageResult]:
        """Parse SerpAPI Google Images response into ImageResult.

        Args:
            image: Image dict from SerpAPI response
            idx: Index for generating unique ID

        Returns:
            ImageResult or None if parsing fails
        """
        try:
            # Get the direct image URL
            original_url = image.get("original", "")
            if not original_url:
                return None

            # Get thumbnail for AI evaluation
            thumbnail = image.get("thumbnail", "")

            # Get page URL where image is hosted
            link = image.get("link", "")

            # Get dimensions if available
            width = image.get("original_width", 0)
            height = image.get("original_height", 0)

            # Get title
            title = image.get("title", "")[:100] if image.get("title") else f"Google Image {idx + 1}"

            # Get source domain
            source_domain = image.get("source", "")

            # Generate a unique ID from URL hash
            import hashlib

            url_hash = hashlib.md5(original_url.encode()).hexdigest()[:8]
            image_id = f"google_{url_hash}"

            return ImageResult(
                image_id=image_id,
                title=title,
                url=link or original_url,
                download_url=original_url,
                width=width,
                height=height,
                source="google",
                description=f"From {source_domain}" if source_domain else None,
                license="Unknown (Google Images)",  # License varies
                photographer=source_domain,  # Use source as attribution
                thumbnail_url=thumbnail,
            )

        except Exception as e:
            logger.warning(f"[Google Images] Failed to parse image: {e}")
            return None

    async def download_image(self, image: ImageResult, output_path: str) -> Optional[str]:
        """Download an image from Google Images.

        Override base implementation to add better error handling for
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

                    # Verify it's actually an image
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
