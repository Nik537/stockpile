"""Image download service with parallel acquisition and AI selection.

Feature 1: Style/Mood Detection - Passes ContentStyle to AI selection
Feature 3: Feedback Loop - Applies feedback filtering to candidates
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from models.image import ImageNeed, ImageResult, ScoredImage
from models.style import ContentStyle
from services.image_sources.base import ImageSource

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Service for downloading images with concurrency control.

    Handles parallel downloading of images from multiple sources
    with rate limiting and error handling.
    """

    def __init__(self, output_dir: str, max_concurrent: int = 10):
        """Initialize the image downloader.

        Args:
            output_dir: Base directory for downloaded images
            max_concurrent: Maximum concurrent downloads (default 10)
        """
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def download_image(
        self,
        image: ImageResult,
        source: ImageSource,
        output_folder: Path,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Download a single image using the appropriate source.

        Args:
            image: ImageResult to download
            source: ImageSource that provided this image (for download method)
            output_folder: Folder to save the image
            filename: Optional custom filename (without extension)

        Returns:
            Path to downloaded file, or None if failed
        """
        async with self.semaphore:
            try:
                # Create output folder if needed
                output_folder.mkdir(parents=True, exist_ok=True)

                # Use custom filename if provided, otherwise generate from source/ID
                if filename:
                    base_filename = filename
                else:
                    base_filename = f"{image.source}_{image.image_id.split('_')[-1]}"

                output_path = str(output_folder / base_filename)

                # Download using source-specific method
                result = await source.download_image(image, output_path)

                if result:
                    logger.info(f"[ImageDownloader] Downloaded: {Path(result).name}")
                return result

            except Exception as e:
                logger.error(f"[ImageDownloader] Failed to download {image.image_id}: {e}")
                return None

    async def download_images_parallel(
        self,
        images: list[tuple[ImageNeed, ImageResult, ImageSource]],
        project_dir: str,
    ) -> dict[str, str]:
        """Download multiple images in parallel.

        Args:
            images: List of (ImageNeed, ImageResult, ImageSource) tuples
            project_dir: Base project directory for output

        Returns:
            Dictionary mapping timestamp folder names to downloaded file paths
        """
        results: dict[str, str] = {}
        images_base = Path(project_dir) / "images"

        async def download_one(need: ImageNeed, image: ImageResult, source: ImageSource):
            """Download a single image and track result."""
            folder = images_base / need.folder_name
            path = await self.download_image(image, source, folder)
            if path:
                return (need.folder_name, path)
            return None

        # Create all download tasks
        tasks = [
            download_one(need, image, source)
            for need, image, source in images
        ]

        # Execute in parallel with semaphore limiting
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in completed:
            if isinstance(result, Exception):
                logger.error(f"[ImageDownloader] Download task failed: {result}")
            elif result:
                folder_name, path = result
                results[folder_name] = path

        logger.info(f"[ImageDownloader] Downloaded {len(results)}/{len(images)} images")
        return results


class ImageAcquisitionService:
    """High-level service for image acquisition with AI-powered selection.

    Coordinates searching multiple sources, AI evaluation, and downloading
    the best image for each need.

    Feature 1: Passes ContentStyle to AI for style-aware selection
    Feature 3: Applies FeedbackService filtering to candidates
    """

    def __init__(
        self,
        image_sources: list[ImageSource],
        ai_service,  # AIService type, avoid circular import
        output_dir: str,
        max_concurrent_downloads: int = 10,
        feedback_service=None,  # FeedbackService type, avoid circular import
    ):
        """Initialize the image acquisition service.

        Args:
            image_sources: List of configured image sources
            ai_service: AIService for image selection
            output_dir: Base output directory
            max_concurrent_downloads: Max parallel downloads
            feedback_service: Optional FeedbackService for learning from rejections
        """
        self.image_sources = image_sources
        self.ai_service = ai_service
        self.downloader = ImageDownloader(output_dir, max_concurrent_downloads)
        self.source_map = {s.get_source_name(): s for s in image_sources}
        self.feedback_service = feedback_service
        self.content_style: Optional[ContentStyle] = None  # Set per-video

    def set_content_style(self, style: Optional[ContentStyle]) -> None:
        """Set the content style for the current video being processed.

        Args:
            style: ContentStyle detected from the source video
        """
        self.content_style = style
        if style:
            logger.debug(
                f"[ImageAcquisition] Content style set: visual={style.visual_style.value}, "
                f"topic='{style.topic[:40]}'"
            )

    async def process_single_image_need(
        self,
        need: ImageNeed,
        project_dir: str,
    ) -> Optional[str]:
        """Process a single image need: search sources, AI picks best, download winner.

        Args:
            need: ImageNeed specifying what image to find
            project_dir: Directory for output

        Returns:
            Path to downloaded image, or None if failed
        """
        logger.debug(f"[ImageAcquisition] Processing need at {need.timestamp}s: {need.search_phrase}")

        # Step 1: Search all configured sources in parallel
        search_tasks = [
            source.search_images(need.search_phrase, per_page=1)
            for source in self.image_sources
        ]

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect one image from each source (up to 3 candidates)
        candidates: list[tuple[ImageResult, ImageSource]] = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.debug(f"[ImageAcquisition] Source {self.image_sources[i].get_source_name()} failed: {result}")
                continue
            if result and len(result) > 0:
                candidates.append((result[0], self.image_sources[i]))
                logger.debug(
                    f"[ImageAcquisition] Got candidate from {self.image_sources[i].get_source_name()}: "
                    f"{result[0].title[:50]}"
                )

        if not candidates:
            logger.warning(f"[ImageAcquisition] No candidates found for: {need.search_phrase}")
            return None

        # Step 2: If only one candidate, use it directly
        if len(candidates) == 1:
            best_image, best_source = candidates[0]
            logger.debug(f"[ImageAcquisition] Single candidate, using: {best_image.source}")
        else:
            # Multiple candidates - AI picks the best one
            best_image, best_source = await self._evaluate_and_select(candidates, need)

        # Step 3: Download the winning image
        # All images go in a single 'images/' folder with descriptive filenames
        images_folder = Path(project_dir) / "images"
        images_folder.mkdir(parents=True, exist_ok=True)

        # Filename format: {timestamp}_{search_phrase}.jpg
        # e.g., 0m30s_mma_fighter_training.jpg
        descriptive_filename = need.folder_name  # Already has timestamp + search phrase

        downloaded_path = await self.downloader.download_image(
            best_image, best_source, images_folder, filename=descriptive_filename
        )

        if downloaded_path:
            logger.info(
                f"[ImageAcquisition] Downloaded: {Path(downloaded_path).name}"
            )

        return downloaded_path

    async def _evaluate_and_select(
        self,
        candidates: list[tuple[ImageResult, ImageSource]],
        need: ImageNeed,
    ) -> tuple[ImageResult, ImageSource]:
        """Use AI to select the best image from candidates.

        Feature 1: Passes ContentStyle to AI for style-aware selection
        Feature 2: Uses extended context window from ImageNeed
        Feature 3: Applies feedback filtering before AI selection

        Args:
            candidates: List of (ImageResult, ImageSource) tuples
            need: The ImageNeed for context

        Returns:
            Best (ImageResult, ImageSource) tuple
        """
        # Feature 3: Apply feedback filtering to candidates
        if self.feedback_service:
            candidate_images = [c[0] for c in candidates]
            filtered_images = self.feedback_service.apply_to_image_candidates(
                candidate_images, need.search_phrase
            )
            # Rebuild candidates list with filtered images
            filtered_ids = {img.image_id for img in filtered_images}
            candidates = [c for c in candidates if c[0].image_id in filtered_ids]

            if not candidates:
                logger.warning(
                    f"[ImageAcquisition] All candidates filtered by feedback, "
                    f"using original list"
                )
                candidates = [(c[0], c[1]) for c in candidates]

        try:
            # Prepare candidate info for AI
            candidate_info = [
                {
                    "index": i,
                    "title": c[0].title,
                    "source": c[0].source,
                    "resolution": c[0].resolution,
                    "description": c[0].description or "",
                }
                for i, c in enumerate(candidates)
            ]

            # Feature 2: Build extended context from ImageNeed
            extended_context = need.get_enhanced_context() if hasattr(need, 'get_enhanced_context') else need.context

            # Feature 3: Get feedback-based prompt additions
            feedback_context = ""
            if self.feedback_service:
                feedback_context = self.feedback_service.get_prompt_additions(need.search_phrase)

            # Use AI service to pick best (with style and feedback context)
            best_index = await self.ai_service.select_best_image(
                search_phrase=need.search_phrase,
                context=extended_context,
                candidates=candidate_info,
                content_style=self.content_style,
                feedback_context=feedback_context,
            )

            # Validate index
            if 0 <= best_index < len(candidates):
                selected = candidates[best_index]
                logger.debug(
                    f"[ImageAcquisition] AI selected index {best_index}: "
                    f"{selected[0].source} - {selected[0].title[:40]}"
                )
                return selected
            else:
                logger.warning(f"[ImageAcquisition] Invalid AI selection index: {best_index}")

        except Exception as e:
            logger.warning(f"[ImageAcquisition] AI selection failed: {e}")

        # Fallback to first candidate (prefer google for variety)
        # Sort by source priority: google > pexels > pixabay
        priority = {"google": 0, "pexels": 1, "pixabay": 2}
        candidates.sort(key=lambda c: priority.get(c[0].source, 99))
        return candidates[0]

    async def process_all_image_needs(
        self,
        needs: list[ImageNeed],
        project_dir: str,
    ) -> dict[str, str]:
        """Process all image needs in parallel.

        Args:
            needs: List of ImageNeed objects
            project_dir: Base project directory

        Returns:
            Dictionary mapping folder names to downloaded file paths
        """
        logger.info(f"[ImageAcquisition] Processing {len(needs)} image needs")

        results: dict[str, str] = {}

        # Process needs in parallel batches to avoid overwhelming APIs
        batch_size = min(5, len(self.image_sources) * 3)  # Limit parallel requests

        for batch_start in range(0, len(needs), batch_size):
            batch_end = min(batch_start + batch_size, len(needs))
            batch = needs[batch_start:batch_end]

            tasks = [
                self.process_single_image_need(need, project_dir)
                for need in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for need, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"[ImageAcquisition] Failed for {need.folder_name}: {result}")
                elif result:
                    results[need.folder_name] = result

        logger.info(
            f"[ImageAcquisition] Completed: {len(results)}/{len(needs)} images acquired"
        )
        return results
