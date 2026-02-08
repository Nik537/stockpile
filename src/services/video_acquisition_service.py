"""Video acquisition service for downloading and processing B-roll footage.

This module extracts video acquisition functionality from BRollProcessor into a
dedicated service that handles:
- Processing individual B-roll needs (search, evaluate, download, extract)
- Two-pass download optimization (preview + high-res clips)
- Competitive analysis mode for selecting best clips
- Clip extraction with semaphore-based concurrency limiting
- Semantic verification of extracted clips

This is part of the BRollProcessor decomposition into focused service classes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models.broll_need import BRollNeed, TranscriptResult
    from models.user_preferences import UserPreferences
    from models.video import ScoredVideo
    from services.ai_service import AIService
    from services.clip_extractor import ClipExtractor
    from services.drive_service import DriveService
    from services.feedback_service import FeedbackService
    from services.file_organizer import FileOrganizer
    from services.semantic_verifier import SemanticVerifier
    from services.video_downloader import VideoDownloader
    from services.video_search_service import VideoSearchService

logger = logging.getLogger(__name__)


class VideoAcquisitionService:
    """Service for video acquisition and clip extraction.

    This service encapsulates video acquisition logic that was previously
    embedded in BRollProcessor, providing a clean interface for:
    - Processing individual B-roll needs end-to-end
    - Two-pass download optimization (preview + high-res)
    - Competitive analysis for best clip selection
    - Parallel clip extraction with concurrency limiting
    - Semantic verification of extracted clips
    """

    def __init__(
        self,
        video_downloader: "VideoDownloader",
        clip_extractor: Optional["ClipExtractor"],
        semantic_verifier: Optional["SemanticVerifier"],
        file_organizer: "FileOrganizer",
        video_search_service: "VideoSearchService",
        ai_service: "AIService",
        feedback_service: Optional["FeedbackService"] = None,
        drive_service: Optional["DriveService"] = None,
        download_semaphore: Optional[asyncio.Semaphore] = None,
        extraction_semaphore: Optional[asyncio.Semaphore] = None,
        ai_semaphore: Optional[asyncio.Semaphore] = None,
        # Configuration values
        clips_per_need_target: int = 5,
        use_two_pass_download: bool = True,
        competitive_analysis_enabled: bool = True,
        previews_per_need: int = 2,
        content_filter: Optional[str] = None,
        delete_original_after_extraction: bool = True,
        max_clip_duration: float = 15.0,
        min_clip_duration: float = 4.0,
        max_videos_per_phrase: int = 3,
        preview_max_height: int = 360,
        clip_download_format: str = "bestvideo[height<=1080]+bestaudio/best",
        reject_below_threshold: bool = True,
        protected_input_files: Optional[set[str]] = None,
    ):
        """Initialize the video acquisition service.

        Args:
            video_downloader: VideoDownloader instance for video downloads
            clip_extractor: Optional ClipExtractor instance for clip extraction
            semantic_verifier: Optional SemanticVerifier for clip verification
            file_organizer: FileOrganizer instance for folder management
            video_search_service: VideoSearchService instance for search and evaluation
            ai_service: AIService instance for AI operations
            feedback_service: Optional FeedbackService for learning from rejections
            drive_service: Optional DriveService for cloud uploads
            download_semaphore: Semaphore for limiting concurrent downloads
            extraction_semaphore: Semaphore for limiting concurrent extractions
            ai_semaphore: Semaphore for limiting concurrent AI calls
            clips_per_need_target: Target clips per B-roll need (default: 5)
            use_two_pass_download: Enable two-pass download optimization (default: True)
            competitive_analysis_enabled: Enable competitive analysis mode (default: True)
            previews_per_need: Number of previews for competitive analysis (default: 2)
            content_filter: Optional content filter string
            delete_original_after_extraction: Delete originals after extraction (default: True)
            max_clip_duration: Maximum clip duration in seconds (default: 15.0)
            min_clip_duration: Minimum clip duration in seconds (default: 4.0)
            max_videos_per_phrase: Maximum videos per search (default: 3)
            preview_max_height: Preview video height in pixels (default: 360)
            clip_download_format: yt-dlp format string for final downloads
            reject_below_threshold: Delete clips below semantic threshold (default: True)
            protected_input_files: Set of protected file paths that must not be deleted
        """
        self.video_downloader = video_downloader
        self.clip_extractor = clip_extractor
        self.semantic_verifier = semantic_verifier
        self.file_organizer = file_organizer
        self.video_search_service = video_search_service
        self.ai_service = ai_service
        self.feedback_service = feedback_service
        self.drive_service = drive_service

        # Semaphores for concurrency limiting
        self.download_semaphore = download_semaphore or asyncio.Semaphore(3)
        self.extraction_semaphore = extraction_semaphore or asyncio.Semaphore(2)
        self.ai_semaphore = ai_semaphore or asyncio.Semaphore(5)

        # Configuration values
        self.clips_per_need_target = clips_per_need_target
        self.use_two_pass_download = use_two_pass_download
        self.competitive_analysis_enabled = competitive_analysis_enabled
        self.previews_per_need = previews_per_need
        self.content_filter = content_filter
        self.delete_original_after_extraction = delete_original_after_extraction
        self.max_clip_duration = max_clip_duration
        self.min_clip_duration = min_clip_duration
        self.max_videos_per_phrase = max_videos_per_phrase
        self.preview_max_height = preview_max_height
        self.clip_download_format = clip_download_format
        self.reject_below_threshold = reject_below_threshold
        self.protected_input_files = protected_input_files or set()

        logger.info(
            f"[VideoAcquisitionService] Initialized with "
            f"clips_target={clips_per_need_target}, "
            f"two_pass={use_two_pass_download}, "
            f"competitive={competitive_analysis_enabled}"
        )

    async def process_single_need(
        self,
        need: "BRollNeed",
        need_index: int,
        total_needs: int,
        project_dir: str,
        drive_project_folder_id: Optional[str] = None,
        transcript_result: Optional["TranscriptResult"] = None,
        user_preferences: Optional["UserPreferences"] = None,
        video_prefilter: Optional[object] = None,
    ) -> Optional[tuple[str, list[str]]]:
        """Process a single B-roll need: search, evaluate, download, extract clips.

        Q4 IMPROVEMENT: Now accepts transcript_result and user_preferences
        for context-aware video evaluation.

        Args:
            need: BRollNeed object with search phrase and timestamp
            need_index: Index of this need (1-based)
            total_needs: Total number of needs being processed
            project_dir: Local project directory
            drive_project_folder_id: Google Drive folder ID for this project
            transcript_result: Optional TranscriptResult for context-aware evaluation
            user_preferences: Optional UserPreferences for style customization
            video_prefilter: Optional VideoPreFilter for pre-filtering videos

        Returns:
            Tuple of (folder_name, list of clip files) or None if failed
        """
        logger.info(
            f"[{need_index}/{total_needs}] Processing: {need.search_phrase} "
            f"at {need.timestamp:.1f}s"
        )

        try:
            # Q2 IMPROVEMENT: Use enhanced search with fallback and negative keyword filtering
            if need.has_enhanced_metadata():
                logger.info(
                    f"  [{need_index}/{total_needs}] Using enhanced search "
                    f"(alternates: {len(need.alternate_searches)}, "
                    f"negatives: {len(need.negative_keywords)}, "
                    f"style: {need.visual_style or 'any'})"
                )
                video_results = await self.video_search_service.search_with_fallback(
                    need, min_results=5
                )
            else:
                # Fallback to simple search for non-enhanced B-roll needs
                video_results = await self.video_search_service.search_youtube_videos(
                    need.search_phrase
                )
            logger.info(f"  [{need_index}/{total_needs}] Found {len(video_results)} videos")

            # S5 IMPROVEMENT: Pre-filter videos before AI evaluation
            if video_results and video_prefilter:
                original_count = len(video_results)
                video_results = video_prefilter.filter(video_results)
                filter_diff = original_count - len(video_results)
                if filter_diff > 0:
                    logger.info(
                        f"  [{need_index}/{total_needs}] Pre-filtered: {original_count} -> "
                        f"{len(video_results)} videos ({filter_diff} removed)"
                    )

            if not video_results:
                logger.info(
                    f"  [{need_index}/{total_needs}] All videos filtered for: {need.search_phrase}"
                )
                return None

            # Q2/Q4 IMPROVEMENT: Use context-aware evaluation with enhanced metadata
            scored_videos = await self.video_search_service.evaluate_videos_enhanced(
                need=need,
                videos=video_results,
                transcript_result=transcript_result,
                user_preferences=user_preferences,
            )
            logger.info(f"  [{need_index}/{total_needs}] Selected {len(scored_videos)} top videos")

            if not scored_videos:
                logger.info(
                    f"  [{need_index}/{total_needs}] No suitable videos found for: {need.search_phrase}"
                )
                return None

            # Create timestamp-prefixed folder for this need
            need_folder = self.file_organizer.create_need_folder(project_dir, need)
            logger.info(f"  [{need_index}/{total_needs}] Output folder: {Path(need_folder).name}")

            # Google Drive folder for this need
            drive_need_folder_id = None
            if self.drive_service and drive_project_folder_id:
                loop = asyncio.get_event_loop()
                drive_need_folder_id = await loop.run_in_executor(
                    None,
                    self.drive_service.create_phrase_folder,
                    need.folder_name,
                    drive_project_folder_id,
                )

            # Process videos - stop early if we hit target clip count
            final_files: list[str] = []

            loop = asyncio.get_event_loop()

            if (
                self.competitive_analysis_enabled
                and self.use_two_pass_download
                and self.clip_extractor
            ):
                # COMPETITIVE ANALYSIS MODE
                final_files = await self._process_competitive_mode(
                    need=need,
                    need_index=need_index,
                    total_needs=total_needs,
                    scored_videos=scored_videos,
                    need_folder=need_folder,
                    drive_need_folder_id=drive_need_folder_id,
                )
            else:
                # SEQUENTIAL PROCESSING MODE (Original behavior)
                final_files = await self._process_sequential_mode(
                    need=need,
                    need_index=need_index,
                    total_needs=total_needs,
                    scored_videos=scored_videos,
                    need_folder=need_folder,
                    drive_need_folder_id=drive_need_folder_id,
                )

            logger.info(f"  [{need_index}/{total_needs}] Completed: {len(final_files)} clips total")

            # S6 IMPROVEMENT: Semantic verification for extracted clips
            if self.semantic_verifier and final_files:
                verified_files = await self._verify_clips_semantically(
                    clip_paths=final_files,
                    broll_need=need,
                    need_index=need_index,
                    total_needs=total_needs,
                )

                if verified_files:
                    removed_count = len(final_files) - len(verified_files)
                    if removed_count > 0:
                        logger.info(
                            f"  [{need_index}/{total_needs}] Semantic verification: "
                            f"{len(verified_files)} passed, {removed_count} filtered out"
                        )
                    final_files = verified_files
                else:
                    logger.warning(
                        f"  [{need_index}/{total_needs}] All clips failed semantic verification, "
                        "keeping original files"
                    )

            # Clean up intermediate files
            await loop.run_in_executor(
                None,
                self.video_downloader.cleanup_intermediate_files,
                Path(need_folder),
            )

            return (need.folder_name, final_files)

        except Exception as e:
            logger.error(
                f"[{need_index}/{total_needs}] Failed to process need '{need.search_phrase}': {e}"
            )
            return None

    async def _process_competitive_mode(
        self,
        need: "BRollNeed",
        need_index: int,
        total_needs: int,
        scored_videos: list["ScoredVideo"],
        need_folder: str,
        drive_need_folder_id: Optional[str],
    ) -> list[str]:
        """Process videos using competitive analysis mode.

        Downloads multiple previews in parallel, analyzes them together,
        and downloads only the winning clip in high resolution.

        Args:
            need: BRollNeed object being processed
            need_index: Index of this need (1-based)
            total_needs: Total number of needs
            scored_videos: List of scored videos to process
            need_folder: Local output folder
            drive_need_folder_id: Optional Drive folder ID

        Returns:
            List of final clip file paths
        """
        final_files: list[str] = []
        loop = asyncio.get_event_loop()

        logger.info(
            f"  [{need_index}/{total_needs}] Competitive analysis: "
            f"downloading {self.previews_per_need} previews in parallel"
        )

        # Select top N videos for comparison
        preview_videos = scored_videos[:self.previews_per_need]

        # Download all previews in parallel
        download_tasks = [
            self._download_preview_with_limit(
                video=video,
                need_folder=str(need_folder),
                need_index=need_index,
                video_idx=idx,
                total_videos=len(preview_videos),
            )
            for idx, video in enumerate(preview_videos, 1)
        ]

        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Filter out failures and exceptions
        preview_files = [
            result
            for result in download_results
            if result is not None and not isinstance(result, Exception)
        ]

        if not preview_files:
            logger.warning(
                f"  [{need_index}/{total_needs}] No previews downloaded, skipping need"
            )
            return final_files

        # Analyze all previews together to find best clip
        logger.info(
            f"  [{need_index}/{total_needs}] Analyzing {len(preview_files)} previews..."
        )

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.clip_extractor.analyze_videos_competitive(
                    preview_files,
                    need.search_phrase,
                    broll_need=need,
                ),
            )

            if result:
                source_video_path, best_segment = result

                # Find which video the winning segment came from
                source_video = None
                for video, preview_file in preview_files:
                    if Path(preview_file) == source_video_path:
                        source_video = video
                        break

                if source_video:
                    logger.info(
                        f"  [{need_index}/{total_needs}] Best clip: "
                        f"{best_segment.start_time:.1f}s-{best_segment.end_time:.1f}s "
                        f"from {source_video.video_id} (score: {best_segment.relevance_score}/10)"
                    )

                    # Download winner in high res
                    clip_files = await loop.run_in_executor(
                        None,
                        self.video_downloader.download_clip_sections,
                        source_video,
                        [best_segment],
                        str(need_folder),
                        self.clip_download_format,
                    )

                    if clip_files:
                        final_files.extend(clip_files)
                        logger.info(
                            f"  [{need_index}/{total_needs}] Downloaded winning clip in high quality"
                        )

                        # Upload to Drive if configured
                        if self.drive_service and drive_need_folder_id:
                            for clip_file in clip_files:
                                await loop.run_in_executor(
                                    None,
                                    self.drive_service.upload_file,
                                    clip_file,
                                    drive_need_folder_id,
                                )
                    else:
                        logger.warning(
                            f"  [{need_index}/{total_needs}] Failed to download winning clip"
                        )
                else:
                    logger.warning(
                        f"  [{need_index}/{total_needs}] Could not identify source video for best clip"
                    )
            else:
                logger.warning(
                    f"  [{need_index}/{total_needs}] No good clips found in any preview"
                )

        except Exception as e:
            logger.error(
                f"  [{need_index}/{total_needs}] Competitive analysis failed: {e}"
            )

        # Cleanup all previews
        for video, preview_file in preview_files:
            try:
                Path(preview_file).unlink()
            except Exception as e:
                logger.warning(
                    f"  [{need_index}/{total_needs}] Could not delete preview: {e}"
                )

        return final_files

    async def _process_sequential_mode(
        self,
        need: "BRollNeed",
        need_index: int,
        total_needs: int,
        scored_videos: list["ScoredVideo"],
        need_folder: str,
        drive_need_folder_id: Optional[str],
    ) -> list[str]:
        """Process videos sequentially, one at a time.

        Args:
            need: BRollNeed object being processed
            need_index: Index of this need (1-based)
            total_needs: Total number of needs
            scored_videos: List of scored videos to process
            need_folder: Local output folder
            drive_need_folder_id: Optional Drive folder ID

        Returns:
            List of final clip file paths
        """
        final_files: list[str] = []
        loop = asyncio.get_event_loop()

        logger.info(
            f"  [{need_index}/{total_needs}] Sequential processing: "
            "analyzing videos one by one"
        )

        for video_idx, video in enumerate(scored_videos, 1):
            # Early stopping if we have enough clips
            if len(final_files) >= self.clips_per_need_target:
                logger.info(
                    f"  [{need_index}/{total_needs}] Target of {self.clips_per_need_target} "
                    "clips reached, stopping early"
                )
                break

            logger.info(
                f"  [{need_index}/{total_needs}] Video {video_idx}/{len(scored_videos)}: "
                f"{video.video_id}"
            )

            try:
                # Check if two-pass download is enabled and supported
                use_two_pass = (
                    self.use_two_pass_download
                    and self.video_downloader._supports_section_downloads()
                    and self.clip_extractor is not None
                )

                if use_two_pass:
                    clips = await self._process_two_pass(
                        video=video,
                        need=need,
                        need_index=need_index,
                        total_needs=total_needs,
                        need_folder=need_folder,
                        drive_need_folder_id=drive_need_folder_id,
                    )
                    if clips:
                        final_files.extend(clips)
                        continue

                # TRADITIONAL WORKFLOW (FALLBACK)
                clips = await self._process_traditional(
                    video=video,
                    need=need,
                    need_index=need_index,
                    total_needs=total_needs,
                    need_folder=need_folder,
                    drive_need_folder_id=drive_need_folder_id,
                )
                if clips:
                    final_files.extend(clips)

            except Exception as e:
                logger.error(
                    f"    [{need_index}/{total_needs}] Failed to process video "
                    f"{video.video_id}: {e}"
                )
                continue

        return final_files

    async def _process_two_pass(
        self,
        video: "ScoredVideo",
        need: "BRollNeed",
        need_index: int,
        total_needs: int,
        need_folder: str,
        drive_need_folder_id: Optional[str],
    ) -> list[str]:
        """Process a video using two-pass download (preview + high-res clips).

        Args:
            video: Video to process
            need: BRollNeed object
            need_index: Index of current need
            total_needs: Total number of needs
            need_folder: Output folder
            drive_need_folder_id: Optional Drive folder ID

        Returns:
            List of clip file paths, or empty list if failed
        """
        loop = asyncio.get_event_loop()
        final_files: list[str] = []

        logger.info(
            f"    [{need_index}/{total_needs}] Using two-pass download (preview + clips)"
        )

        # Pass 1: Download low-quality preview
        preview_file = await loop.run_in_executor(
            None,
            self.video_downloader.download_preview,
            video,
            str(need_folder),
            self.preview_max_height,
        )

        if not preview_file:
            logger.warning(
                f"    [{need_index}/{total_needs}] Preview download failed"
            )
            return []

        logger.info(
            f"    [{need_index}/{total_needs}] Preview downloaded, analyzing..."
        )

        # Analyze preview to get clip segments
        segments = await loop.run_in_executor(
            None,
            lambda: self.clip_extractor.analyze_video(
                preview_file,
                need.search_phrase,
                video.video_id,
                broll_need=need,
            ),
        )

        if segments and segments.analysis_success and segments.segments:
            logger.info(
                f"    [{need_index}/{total_needs}] Found {len(segments.segments)} clips, "
                "downloading in high quality..."
            )

            # Pass 2: Download only the identified clips in high quality
            clip_files = await loop.run_in_executor(
                None,
                self.video_downloader.download_clip_sections,
                video,
                segments.segments,
                str(need_folder),
                self.clip_download_format,
            )

            if clip_files:
                final_files.extend(clip_files)
                logger.info(
                    f"    [{need_index}/{total_needs}] Downloaded {len(clip_files)} clips "
                    "in high quality"
                )

                # Clean up preview
                try:
                    Path(preview_file).unlink()
                    logger.debug(f"    [{need_index}/{total_needs}] Deleted preview")
                except Exception as e:
                    logger.warning(
                        f"    [{need_index}/{total_needs}] Could not delete preview: {e}"
                    )

                # Upload clips to Drive if configured
                if self.drive_service and drive_need_folder_id:
                    for clip_file in clip_files:
                        await loop.run_in_executor(
                            None,
                            self.drive_service.upload_file,
                            clip_file,
                            drive_need_folder_id,
                        )

                return final_files
            else:
                logger.warning(
                    f"    [{need_index}/{total_needs}] No clips downloaded"
                )
        else:
            logger.warning(
                f"    [{need_index}/{total_needs}] No clips found in preview"
            )

        # Clean up preview on failure
        try:
            Path(preview_file).unlink()
        except Exception:
            pass

        return []

    async def _process_traditional(
        self,
        video: "ScoredVideo",
        need: "BRollNeed",
        need_index: int,
        total_needs: int,
        need_folder: str,
        drive_need_folder_id: Optional[str],
    ) -> list[str]:
        """Process a video using traditional download (full video + extraction).

        Args:
            video: Video to process
            need: BRollNeed object
            need_index: Index of current need
            total_needs: Total number of needs
            need_folder: Output folder
            drive_need_folder_id: Optional Drive folder ID

        Returns:
            List of clip file paths, or empty list if failed
        """
        loop = asyncio.get_event_loop()
        final_files: list[str] = []

        logger.info(
            f"    [{need_index}/{total_needs}] Using traditional download "
            "(full video + extraction)"
        )

        # Download full video
        downloaded_file = await loop.run_in_executor(
            None,
            self.video_downloader.download_single_video_to_folder,
            video,
            str(need_folder),
            self.drive_service,
            drive_need_folder_id,
        )

        if not downloaded_file:
            logger.warning(
                f"    [{need_index}/{total_needs}] Failed to download {video.video_id}"
            )
            return []

        logger.info(
            f"    [{need_index}/{total_needs}] Downloaded: {Path(downloaded_file).name}"
        )

        # Extract clips from full video
        if self.clip_extractor:
            clips, should_delete = await loop.run_in_executor(
                None,
                lambda: self.clip_extractor.process_downloaded_video(
                    downloaded_file,
                    need.search_phrase,
                    video.video_id,
                    output_dir=None,
                    broll_need=need,
                ),
            )

            if clips:
                final_files.extend([c.clip_path for c in clips])
                logger.info(
                    f"    [{need_index}/{total_needs}] Extracted {len(clips)} clips"
                )

                # Delete original if clips extracted and setting enabled
                if should_delete and self.delete_original_after_extraction:
                    self.clip_extractor.cleanup_original_video(downloaded_file)
                    logger.info(
                        f"    [{need_index}/{total_needs}] Deleted original: "
                        f"{Path(downloaded_file).name}"
                    )
            else:
                # Keep original if no clips extracted
                final_files.append(downloaded_file)
                logger.info(
                    f"    [{need_index}/{total_needs}] No clips extracted, keeping original"
                )
        else:
            final_files.append(downloaded_file)

        return final_files

    async def _download_preview_with_limit(
        self,
        video: "ScoredVideo",
        need_folder: str,
        need_index: int,
        video_idx: int,
        total_videos: int,
    ) -> Optional[tuple["ScoredVideo", str]]:
        """Download a single preview video with semaphore limiting.

        Args:
            video: Video to download
            need_folder: Output folder
            need_index: Index of current need
            video_idx: Index of this video (1-based)
            total_videos: Total number of videos being downloaded

        Returns:
            Tuple of (video, preview_file_path) or None if download failed
        """
        async with self.download_semaphore:
            try:
                logger.info(
                    f"    [{need_index}] Downloading preview {video_idx}/{total_videos}: "
                    f"{video.video_id}"
                )

                loop = asyncio.get_event_loop()
                preview_file = await loop.run_in_executor(
                    None,
                    self.video_downloader.download_preview,
                    video,
                    str(need_folder),
                    self.preview_max_height,
                )

                if preview_file:
                    logger.info(
                        f"    [{need_index}] Preview {video_idx} downloaded: "
                        f"{Path(preview_file).name}"
                    )
                    return (video, preview_file)
                else:
                    logger.warning(f"    [{need_index}] Preview {video_idx} download failed")
                    return None

            except Exception as e:
                logger.error(f"    [{need_index}] Failed to download preview {video_idx}: {e}")
                return None

    async def extract_clips_from_videos(
        self,
        downloaded_files: list[str],
        phrase: str,
        scored_videos: list["ScoredVideo"],
    ) -> list[str]:
        """Extract clips from downloaded videos using AI analysis.

        Args:
            downloaded_files: List of downloaded video file paths
            phrase: Search phrase for context
            scored_videos: List of scored video objects for video ID lookup

        Returns:
            List of final file paths (clips if extraction succeeded, originals otherwise)
        """
        if not self.clip_extractor:
            return downloaded_files

        # Build a lookup from video path to video ID
        video_id_lookup: dict[str, str] = {}
        for video_path in downloaded_files:
            path_lower = video_path.lower()
            for sv in scored_videos:
                if sv.video_id.lower() in path_lower:
                    video_id_lookup[video_path] = sv.video_id
                    break
            if video_path not in video_id_lookup:
                video_id_lookup[video_path] = "unknown"

        # Extract clips from all videos in parallel
        extraction_tasks = [
            self._extract_clips_from_single_video(
                video_path=video_path,
                video_id=video_id_lookup.get(video_path, "unknown"),
                phrase=phrase,
            )
            for video_path in downloaded_files
        ]

        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Collect results
        final_files: list[str] = []
        originals_to_delete: list[str] = []

        for result in extraction_results:
            if isinstance(result, Exception):
                logger.error(f"Clip extraction task failed: {result}")
                continue

            clip_paths, original_to_delete = result
            final_files.extend(clip_paths)
            if original_to_delete:
                originals_to_delete.append(original_to_delete)

        # Clean up original videos (with input protection check)
        for original_path in originals_to_delete:
            if self._is_protected_file(original_path):
                logger.warning(f"BLOCKED deletion of protected input file: {original_path}")
                continue

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.clip_extractor.cleanup_original_video, original_path
            )

        return final_files

    async def _extract_clips_from_single_video(
        self,
        video_path: str,
        video_id: str,
        phrase: str,
        broll_need: Optional["BRollNeed"] = None,
    ) -> tuple[list[str], Optional[str]]:
        """Extract clips from a single video with semaphore limiting.

        Args:
            video_path: Path to video file
            video_id: Video ID for identification
            phrase: Search phrase for context
            broll_need: Optional BRollNeed for semantic context matching

        Returns:
            Tuple of (list of clip paths, original path to delete or None)
        """
        async with self.extraction_semaphore:
            try:
                logger.info(f"Extracting clips from: {Path(video_path).name}")

                loop = asyncio.get_event_loop()
                clips, should_delete = await loop.run_in_executor(
                    None,
                    lambda: self.clip_extractor.process_downloaded_video(
                        video_path,
                        phrase,
                        video_id,
                        output_dir=None,
                        broll_need=broll_need,
                    ),
                )

                if clips:
                    clip_paths = [c.clip_path for c in clips]
                    logger.info(f"Extracted {len(clips)} clips from {Path(video_path).name}")
                    return (
                        clip_paths,
                        video_path
                        if should_delete and self.delete_original_after_extraction
                        else None,
                    )
                else:
                    logger.info(f"No clips extracted, keeping original: {Path(video_path).name}")
                    return ([video_path], None)

            except Exception as e:
                logger.error(f"Failed to extract clips from {Path(video_path).name}: {e}")
                return ([video_path], None)

    def _is_protected_file(self, file_path: str) -> bool:
        """Check if a file path is a protected input file that must NOT be deleted.

        Args:
            file_path: Path to check

        Returns:
            True if the file is protected and should not be deleted
        """
        resolved_path = str(Path(file_path).resolve())
        return resolved_path in self.protected_input_files

    async def _verify_clips_semantically(
        self,
        clip_paths: list[str],
        broll_need: "BRollNeed",
        need_index: int,
        total_needs: int,
    ) -> list[str]:
        """Verify clips semantically match the original transcript context.

        S6 IMPROVEMENT: Uses SemanticVerifier to analyze extracted clips and verify
        they match the original transcript context and contain required visual elements.

        Args:
            clip_paths: List of clip file paths to verify
            broll_need: BRollNeed with original_context and required_elements
            need_index: Index of current need (for logging)
            total_needs: Total number of needs (for logging)

        Returns:
            List of verified clip paths that pass semantic verification threshold
        """
        if not self.semantic_verifier:
            return clip_paths

        if not clip_paths:
            return clip_paths

        logger.info(
            f"  [{need_index}/{total_needs}] Running semantic verification on "
            f"{len(clip_paths)} clips"
        )

        verified_files: list[str] = []
        rejected_files: list[tuple[str, object]] = []

        for clip_path in clip_paths:
            clip_file = Path(clip_path)
            if not clip_file.exists():
                logger.warning(f"  [{need_index}/{total_needs}] Clip not found: {clip_path}")
                continue

            try:
                result = await self.semantic_verifier.verify_clip(
                    clip_path=clip_file,
                    broll_need=broll_need,
                )

                if result.passed:
                    verified_files.append(clip_path)
                    logger.info(
                        f"    [{need_index}/{total_needs}] Verified {clip_file.name}: "
                        f"score={result.similarity_score:.2%}, "
                        f"matched={len(result.matched_elements)}/"
                        f"{len(result.matched_elements) + len(result.missing_elements)}"
                    )
                else:
                    rejected_files.append((clip_path, result))
                    logger.warning(
                        f"    [{need_index}/{total_needs}] Rejected {clip_file.name}: "
                        f"score={result.similarity_score:.2%} < threshold, "
                        f"missing={result.missing_elements}"
                    )

            except Exception as e:
                logger.error(
                    f"    [{need_index}/{total_needs}] Verification failed for "
                    f"{clip_file.name}: {e}"
                )
                verified_files.append(clip_path)

        # If all clips were rejected, keep the best scoring one with a warning
        if not verified_files and rejected_files:
            rejected_files.sort(key=lambda x: x[1].similarity_score, reverse=True)
            best_path, best_result = rejected_files[0]
            verified_files.append(best_path)
            logger.warning(
                f"  [{need_index}/{total_needs}] All clips below threshold, keeping best: "
                f"{Path(best_path).name} (score={best_result.similarity_score:.2%})"
            )

            # Clean up rejected clips (except the one we're keeping)
            if self.reject_below_threshold:
                for path, result in rejected_files[1:]:
                    try:
                        Path(path).unlink()
                        logger.debug(
                            f"    [{need_index}/{total_needs}] Deleted rejected clip: {path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"    [{need_index}/{total_needs}] Could not delete rejected clip: {e}"
                        )
        elif rejected_files:
            # Clean up rejected clips
            if self.reject_below_threshold:
                for path, result in rejected_files:
                    try:
                        Path(path).unlink()
                        logger.debug(
                            f"    [{need_index}/{total_needs}] Deleted rejected clip: {path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"    [{need_index}/{total_needs}] Could not delete rejected clip: {e}"
                        )

        return verified_files
