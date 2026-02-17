"""Openverse image source for CC-licensed images."""

import logging
import os
from typing import Optional

import aiohttp

from models.image import ImageResult
from services.image_sources.base import ImageSource
from utils.retry import NetworkError, TemporaryServiceError, retry_api_call

logger = logging.getLogger(__name__)


class OpenverseImageSource(ImageSource):
    """Openverse (WordPress/CC Search) image source for Creative Commons images.

    Openverse provides access to over 800 million CC-licensed images from
    multiple sources (Flickr, Wikimedia, etc.).

    API Documentation: https://api.openverse.org/v1/

    Authentication is optional:
    - Anonymous: 100 requests/day (5 requests/hr burst)
    - Authenticated: 10,000 requests/day (100 requests/min burst)

    To get higher rate limits:
    1. Register at https://api.openverse.org/v1/auth_tokens/register/
    2. Set OPENVERSE_CLIENT_ID and OPENVERSE_CLIENT_SECRET in .env

    All images are Creative Commons licensed.
    """

    BASE_URL = "https://api.openverse.org/v1/images/"
    TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"

    def __init__(self, max_results: int = 10):
        """Initialize Openverse image source.

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = min(max_results, 20)  # Openverse max page_size
        self.client_id = os.getenv("OPENVERSE_CLIENT_ID", "")
        self.client_secret = os.getenv("OPENVERSE_CLIENT_SECRET", "")
        self._access_token: Optional[str] = None

    def get_source_name(self) -> str:
        """Get the name of this image source."""
        return "openverse"

    def is_configured(self) -> bool:
        """Always returns True - Openverse works without auth (lower limits)."""
        return True

    async def _get_access_token(self) -> Optional[str]:
        """Get OAuth2 access token for higher rate limits.

        Returns:
            Access token string or None if auth not configured/failed
        """
        if self._access_token:
            return self._access_token

        if not self.client_id or not self.client_secret:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.TOKEN_URL,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "grant_type": "client_credentials",
                    },
                    timeout=10,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._access_token = data.get("access_token")
                        logger.debug("[Openverse] Authenticated for higher rate limits")
                        return self._access_token
                    else:
                        logger.warning(
                            f"[Openverse] Auth failed with status {response.status}, "
                            "falling back to anonymous access"
                        )
                        return None
        except Exception as e:
            logger.warning(f"[Openverse] Auth error: {e}, falling back to anonymous access")
            return None

    @retry_api_call(max_retries=3, base_delay=1.0)
    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search Openverse for CC-licensed images.

        Args:
            phrase: Search query string
            per_page: Maximum number of results (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """
        if not phrase.strip():
            return []

        logger.debug(f"[Openverse] Searching for: '{phrase}'")

        params = {
            "q": phrase,
            "page_size": min(per_page, self.max_results),
            "license_type": "commercial",
            "mature": "false",
        }

        headers = {}
        token = await self._get_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL, params=params, headers=headers, timeout=30
                ) as response:
                    if response.status == 401:
                        # Token expired, clear it and retry without auth
                        self._access_token = None
                        logger.warning("[Openverse] Token expired, retrying without auth")
                        headers.pop("Authorization", None)
                        async with session.get(
                            self.BASE_URL, params=params, headers=headers, timeout=30
                        ) as retry_resp:
                            if retry_resp.status != 200:
                                return []
                            data = await retry_resp.json()
                    elif response.status == 429:
                        logger.warning("[Openverse] Rate limit exceeded")
                        raise TemporaryServiceError("Openverse rate limit exceeded")
                    elif response.status != 200:
                        logger.warning(
                            f"[Openverse] API returned status {response.status}"
                        )
                        return []
                    else:
                        data = await response.json()

                    results_list = data.get("results", [])

                    results = []
                    for idx, item in enumerate(results_list):
                        if idx >= per_page:
                            break
                        result = self._parse_image(item)
                        if result:
                            results.append(result)

                    logger.debug(f"[Openverse] Found {len(results)} CC-licensed images")
                    return results

        except aiohttp.ClientError as e:
            logger.error(f"[Openverse] Network error: {e}")
            raise NetworkError(f"Openverse network error: {e}") from e

    def _parse_image(self, item: dict) -> Optional[ImageResult]:
        """Parse Openverse API response item into ImageResult.

        Args:
            item: Result dict from Openverse API

        Returns:
            ImageResult or None if parsing fails
        """
        try:
            download_url = item.get("url", "")
            if not download_url:
                return None

            openverse_id = item.get("id", "")
            image_id = f"openverse_{openverse_id}" if openverse_id else f"openverse_unknown"

            title = item.get("title", "")[:100] if item.get("title") else "Openverse Image"
            thumbnail = item.get("thumbnail", "")
            width = item.get("width", 0) or 0
            height = item.get("height", 0) or 0
            source_platform = item.get("source", "")
            creator = item.get("creator", "")
            license_type = item.get("license", "CC")

            return ImageResult(
                image_id=image_id,
                title=title,
                url=download_url,
                download_url=download_url,
                width=width,
                height=height,
                source="openverse",
                description=f"CC-{license_type} from {source_platform}" if source_platform else f"CC-{license_type}",
                license=f"CC-{license_type}",
                photographer=creator or source_platform,
                thumbnail_url=thumbnail,
            )

        except Exception as e:
            logger.warning(f"[Openverse] Failed to parse image: {e}")
            return None
