"""Base abstraction for image sources."""

from abc import ABC, abstractmethod
from typing import Optional

from models.image import ImageResult


class ImageSource(ABC):
    """Abstract base class for image sources (Pexels, Pixabay, Google, etc.)."""

    @abstractmethod
    async def search_images(self, phrase: str, per_page: int = 1) -> list[ImageResult]:
        """Search for images matching the search phrase.

        Args:
            phrase: Search query string
            per_page: Maximum number of results to return (default 1 for efficiency)

        Returns:
            List of ImageResult objects matching the search criteria
        """

    @abstractmethod
    def get_source_name(self) -> str:
        """Get the name of this image source.

        Returns:
            Source name (e.g., "pexels", "pixabay", "google")
        """

    def is_configured(self) -> bool:
        """Check if this source has required configuration (API keys, etc.).

        Default implementation returns True (no config required).
        Override in subclasses that require API keys.

        Returns:
            True if source is properly configured and ready to use
        """
        return True

    async def download_image(self, image: ImageResult, output_path: str) -> Optional[str]:
        """Download an image to the specified path.

        Default implementation uses aiohttp to download from download_url.
        Override in subclasses that need special download handling.

        Args:
            image: ImageResult with download_url
            output_path: Full path where image should be saved

        Returns:
            Path to downloaded file, or None if download failed
        """
        import logging

        import aiohttp

        logger = logging.getLogger(__name__)

        if not image.download_url:
            logger.warning(f"[{self.get_source_name()}] No download URL for image {image.image_id}")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image.download_url, timeout=60) as response:
                    if response.status != 200:
                        logger.warning(
                            f"[{self.get_source_name()}] Download failed with status {response.status}"
                        )
                        return None

                    content = await response.read()

                    # Determine file extension from content type or URL
                    content_type = response.headers.get("content-type", "")
                    ext = self._get_extension(content_type, image.download_url)

                    # Ensure output_path has correct extension
                    if not output_path.endswith(ext):
                        output_path = f"{output_path}{ext}"

                    with open(output_path, "wb") as f:
                        f.write(content)

                    logger.debug(f"[{self.get_source_name()}] Downloaded {image.image_id} to {output_path}")
                    return output_path

        except aiohttp.ClientError as e:
            logger.error(f"[{self.get_source_name()}] Network error downloading {image.image_id}: {e}")
            return None
        except OSError as e:
            logger.error(f"[{self.get_source_name()}] File error saving {image.image_id}: {e}")
            return None

    def _get_extension(self, content_type: str, url: str) -> str:
        """Determine file extension from content type or URL."""
        # Try content type first
        type_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        for mime, ext in type_map.items():
            if mime in content_type:
                return ext

        # Fall back to URL extension
        url_lower = url.lower()
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
            if ext in url_lower:
                return ext if ext != ".jpeg" else ".jpg"

        # Default to jpg
        return ".jpg"
