"""Main stockpile class for orchestrating the entire workflow."""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
import asyncio

from models.video import VideoResult, ScoredVideo
from models.broll_need import BRollNeed, BRollPlan, TranscriptResult
from models.user_preferences import UserPreferences
from utils.config import load_config, validate_config
from utils.progress import ProcessingStatus
from services.transcription import TranscriptionService
from services.ai_service import AIService
from services.youtube_service import YouTubeService
from services.video_downloader import VideoDownloader
from services.file_organizer import FileOrganizer
from services.notification import NotificationService
from services.drive_service import DriveService
from services.file_monitor import FileMonitor
from services.clip_extractor import ClipExtractor

logger = logging.getLogger(__name__)


class BRollProcessor:
    """Central orchestrator for stockpile."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the stockpile with configuration."""
        self.config = config or load_config()
        self.processing_files: Set[str] = set()
        self.protected_input_files: Set[str] = set()  # Input videos that must NEVER be deleted
        self.event_loop = None

        config_errors = validate_config(self.config)
        if config_errors:
            error_msg = "Configuration errors: " + "; ".join(config_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        gemini_api_key = self.config.get("gemini_api_key")
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")
        gemini_model = self.config.get("gemini_model", "gemini-3-flash-preview")
        self.ai_service = AIService(gemini_api_key, gemini_model)

        client_id = self.config.get("google_client_id")
        client_secret = self.config.get("google_client_secret")
        if client_id and client_secret:
            notification_email = self.config.get("notification_email")
            self.notification_service = NotificationService(
                client_id, client_secret, notification_email
            )
        else:
            self.notification_service = None

        max_videos_per_phrase = self.config.get("max_videos_per_phrase", 3)
        self.youtube_service = YouTubeService(max_results=max_videos_per_phrase * 3)

        whisper_model = self.config.get("whisper_model", "base")
        self.transcription_service = TranscriptionService(whisper_model)

        output_dir = self.config.get("local_output_folder", "../output")
        self.video_downloader = VideoDownloader(output_dir)
        self.file_organizer = FileOrganizer(output_dir)

        output_folder_id = self.config.get("google_drive_output_folder_id")
        if output_folder_id:
            if not client_id:
                raise ValueError(
                    "Google Drive requires GOOGLE_CLIENT_ID environment variable"
                )
            if not client_secret:
                raise ValueError(
                    "Google Drive requires GOOGLE_CLIENT_SECRET environment variable"
                )
            self.drive_service = DriveService(
                client_id, client_secret, output_folder_id
            )
        else:
            self.drive_service = None

        self.file_monitor = FileMonitor(self.config, self._handle_new_file)

        # Initialize clip extractor if enabled
        clip_extraction_enabled = self.config.get("clip_extraction_enabled", True)
        if clip_extraction_enabled:
            self.clip_extractor = ClipExtractor(
                api_key=gemini_api_key,
                model_name=gemini_model,
                min_clip_duration=self.config.get("min_clip_duration", 4.0),
                max_clip_duration=self.config.get("max_clip_duration", 15.0),
                max_clips_per_video=self.config.get("max_clips_per_video", 3),
            )
            self.delete_original_after_extraction = self.config.get(
                "delete_original_after_extraction", True
            )
            logger.info("Clip extraction enabled")
        else:
            self.clip_extractor = None
            self.delete_original_after_extraction = False
            logger.info("Clip extraction disabled")

        # Timeline-aware B-roll planning configuration
        self.clips_per_minute = self.config.get("clips_per_minute", 2.0)
        logger.info(f"B-roll density: {self.clips_per_minute} clips per minute")

        # Content filter for B-roll search (e.g., "men only, no women")
        self.content_filter = self.config.get("content_filter")
        if self.content_filter:
            logger.info(f"Content filter active: {self.content_filter}")

        logger.info("stockpile initialized successfully")

    def start(self) -> None:
        """Start the processor."""
        logger.info("Starting stockpile...")

        self.event_loop = asyncio.get_running_loop()

        self.file_monitor.start_monitoring()

        logger.info("Processor started successfully")

    def _handle_new_file(self, file_path: str, source: str) -> None:
        """Handle new file detected by file monitor."""
        logger.info(f"New file detected from {source}: {file_path}")

        if file_path in self.processing_files:
            return

        if self.event_loop and not self.event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.process_video(file_path), self.event_loop
            )
        else:
            logger.error("No event loop available to schedule file processing")

    async def process_video(self, file_path: str, user_preferences: UserPreferences = None, status_callback=None) -> None:
        """Process a video file through the complete B-roll pipeline.

        Args:
            file_path: Path to the video file to process
            user_preferences: Optional user preferences for B-roll style customization
            status_callback: Optional callback function for progress updates
        """
        if file_path in self.processing_files:
            logger.info(f"File already being processed: {file_path}")
            return

        self.processing_files.add(file_path)

        # CRITICAL: Protect the input video from any accidental deletion
        # This resolves the absolute path to ensure protection works
        input_path = Path(file_path).resolve()
        self.protected_input_files.add(str(input_path))
        logger.info(f"Protected input video: {input_path}")

        start_time = time.time()

        # Initialize progress tracking
        status = ProcessingStatus(
            video_path=file_path,
            output_dir=self.config.get("local_output_folder", "../output"),
            update_callback=status_callback,
        )

        try:
            await self._execute_pipeline(file_path, start_time, user_preferences, status)
            status.complete_processing()
            logger.info(f"Processing completed successfully: {file_path}")

        except Exception as e:
            logger.error(f"Processing failed for {file_path}: {e}")
            status.fail_processing(str(e))
            processing_time = self._format_processing_time(time.time() - start_time)
            await self._send_notification(
                "failed", str(e), processing_time=processing_time, input_file=file_path
            )
            raise

        finally:
            self.processing_files.discard(file_path)

    async def _execute_pipeline(self, file_path: str, start_time: float, user_preferences: UserPreferences = None, status: ProcessingStatus = None) -> None:
        """Execute the complete timeline-aware B-roll processing pipeline.

        Args:
            file_path: Path to the video file to process
            start_time: Start time for processing metrics
            user_preferences: Optional user preferences for B-roll customization
            status: Optional ProcessingStatus for progress tracking

        Pipeline steps:
        1. Transcribe audio with timestamps
        2. Plan B-roll needs (2+ per minute based on clips_per_minute setting)
        3. For each B-roll need:
           - Search YouTube for matching videos
           - Evaluate and score videos
           - Download top videos to timestamp-prefixed folder
           - Extract clips from downloaded videos
           - Delete original videos after extraction
        """
        logger.info(f"Starting pipeline for: {file_path}")

        # Register processing stages
        if status:
            status.register_stage("transcribe", total_items=1)
            status.register_stage("plan", total_items=1)
            status.register_stage("project_setup", total_items=1)
            # B-roll processing will be registered after we know the count

        # Step 1: Transcribe with timestamps
        if status:
            status.start_stage("transcribe")

        transcript_result = await self.transcribe_audio(file_path)

        if status:
            status.update_stage("transcribe", completed=1)
            status.complete_stage("transcribe")
        duration_minutes = transcript_result.duration / 60.0
        logger.info(
            f"Transcription completed: {len(transcript_result.text)} chars, "
            f"{len(transcript_result.segments)} segments, {duration_minutes:.1f} min duration"
        )

        # Step 2: Plan B-roll needs (timeline-aware)
        if status:
            status.start_stage("plan")

        broll_plan = await self.plan_broll_needs(transcript_result, file_path, user_preferences)

        if status:
            status.update_stage("plan", completed=1)
            status.complete_stage("plan")
        logger.info(
            f"B-roll planning complete: {len(broll_plan.needs)} needs identified "
            f"(target: {broll_plan.expected_clip_count} at {self.clips_per_minute}/min)"
        )

        if not broll_plan.needs:
            logger.warning("No B-roll needs identified, nothing to process")
            return

        # Log the planned needs for debugging
        for i, need in enumerate(broll_plan.needs, 1):
            logger.info(
                f"  [{i}] {need.timestamp:.1f}s - {need.search_phrase} "
                f"({need.description})"
            )

        # Step 3: Create project structure
        if status:
            status.start_stage("project_setup")

        source_filename = Path(file_path).name
        project_dir = await self._create_project_structure(file_path, source_filename)
        logger.info(f"Project structure created: {project_dir}")

        if status:
            status.update_stage("project_setup", completed=1)
            status.complete_stage("project_setup")

        # Google Drive setup if configured
        drive_project_folder_id = None
        drive_folder_url = None
        if self.drive_service and source_filename:
            project_name = self._generate_project_name(file_path, source_filename)
            loop = asyncio.get_event_loop()
            drive_project_folder_id = await loop.run_in_executor(
                None,
                self.drive_service.create_project_structure,
                project_name,
            )
            drive_folder_url = (
                f"https://drive.google.com/drive/folders/{drive_project_folder_id}"
            )
            logger.info(f"Google Drive structure created: {drive_folder_url}")

        # Step 4: Process B-roll needs in parallel batches
        logger.info(f"Processing {len(broll_plan.needs)} B-roll needs")
        need_downloads: Dict[str, List[str]] = {}
        total_clips = 0

        # Register B-roll processing stage
        if status:
            status.register_stage("process_needs", total_items=len(broll_plan.needs))
            status.start_stage("process_needs")

        # Get parallel processing config
        max_concurrent = self.config.get("max_concurrent_needs", 5)

        # Process needs in batches
        batch_size = max_concurrent
        for batch_start in range(0, len(broll_plan.needs), batch_size):
            batch_end = min(batch_start + batch_size, len(broll_plan.needs))
            batch = broll_plan.needs[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(broll_plan.needs) + batch_size - 1)//batch_size} ({len(batch)} needs in parallel)")

            # Process this batch in parallel
            tasks = []
            for i, need in enumerate(batch, batch_start + 1):
                task = self._process_single_need(
                    need=need,
                    need_index=i,
                    total_needs=len(broll_plan.needs),
                    project_dir=project_dir,
                    drive_project_folder_id=drive_project_folder_id,
                )
                tasks.append(task)

            # Wait for all tasks in this batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch task failed: {result}")
                    # Still update progress even for failures
                    if status:
                        status.update_stage("process_needs", increment=1)
                    continue
                if result:
                    folder_name, files = result
                    need_downloads[folder_name] = files
                    total_clips += len(files)
                # Update progress
                if status:
                    status.update_stage("process_needs", increment=1)

        # Original sequential processing (keeping for reference, but replaced above)
        """
        for i, need in enumerate(broll_plan.needs, 1):
            logger.info(
                f"[{i}/{len(broll_plan.needs)}] Processing: {need.search_phrase} "
                f"at {need.timestamp:.1f}s"
            )

            # Search YouTube
            video_results = await self.search_youtube_videos(need.search_phrase)
            logger.info(f"  Found {len(video_results)} videos")

            # Evaluate and score videos
            scored_videos = await self.evaluate_videos(need.search_phrase, video_results)
            logger.info(f"  Selected {len(scored_videos)} top videos")

            if not scored_videos:
                logger.info(f"  No suitable videos found for: {need.search_phrase}")
                continue

            # Create timestamp-prefixed folder for this need
            need_folder = self.file_organizer.create_need_folder(project_dir, need)
            logger.info(f"  Output folder: {Path(need_folder).name}")

            # Google Drive folder for this need
            drive_need_folder_id = None
            if self.drive_service and drive_project_folder_id:
                loop = asyncio.get_event_loop()
                drive_need_folder_id = await loop.run_in_executor(
                    None,
                    self.drive_service.create_phrase_folder,
                    need.folder_name,  # Use timestamp-prefixed name
                    drive_project_folder_id,
                )

            # Process videos sequentially: download one → extract clips → delete original → next
            final_files = []
            for video_idx, video in enumerate(scored_videos, 1):
                logger.info(f"  Video {video_idx}/{len(scored_videos)}: {video.video_id}")

                loop = asyncio.get_event_loop()

                try:
                    # Check if two-pass download is enabled and supported
                    use_two_pass = (
                        self.config.get("use_two_pass_download", True)
                        and self.video_downloader._supports_section_downloads()
                        and self.clip_extractor is not None
                    )

                    if use_two_pass:
                        # TWO-PASS WORKFLOW (OPTIMIZED)
                        logger.info(f"    Using two-pass download (preview + clips)")

                        # Pass 1: Download low-quality preview
                        preview_file = await loop.run_in_executor(
                            None,
                            self.video_downloader.download_preview,
                            video,
                            str(need_folder),
                            self.config.get("preview_max_height", 360),
                        )

                        if not preview_file:
                            logger.warning(f"    Preview download failed, trying traditional download")
                            use_two_pass = False
                        else:
                            logger.info(f"    Preview downloaded, analyzing...")

                            # Analyze preview to get clip segments
                            segments = await loop.run_in_executor(
                                None,
                                self.clip_extractor.analyze_video,
                                preview_file,
                                need.search_phrase,
                                video.video_id,
                            )

                            if segments and segments.analysis_success and segments.segments:
                                logger.info(f"    Found {len(segments.segments)} clips, downloading in high quality...")

                                # Pass 2: Download only the identified clips in high quality
                                clip_files = await loop.run_in_executor(
                                    None,
                                    self.video_downloader.download_clip_sections,
                                    video,
                                    segments.segments,
                                    str(need_folder),
                                    self.config.get(
                                        "clip_download_format",
                                        "bestvideo[height<=1080]+bestaudio/best",
                                    ),
                                )

                                if clip_files:
                                    final_files.extend(clip_files)
                                    logger.info(f"    Downloaded {len(clip_files)} clips in high quality")

                                    # Clean up preview
                                    try:
                                        Path(preview_file).unlink()
                                        logger.debug(f"    Deleted preview")
                                    except Exception as e:
                                        logger.warning(f"    Could not delete preview: {e}")

                                    # Upload clips to Drive if configured
                                    if self.drive_service and drive_need_folder_id:
                                        for clip_file in clip_files:
                                            await loop.run_in_executor(
                                                None,
                                                self.drive_service.upload_file,
                                                clip_file,
                                                drive_need_folder_id,
                                            )

                                    continue  # Skip traditional workflow
                                else:
                                    logger.warning(f"    No clips downloaded, trying traditional download")
                                    use_two_pass = False
                            else:
                                logger.warning(f"    No clips found in preview, trying traditional download")
                                use_two_pass = False

                            # Clean up preview if falling back
                            if not use_two_pass and preview_file:
                                try:
                                    Path(preview_file).unlink()
                                except Exception:
                                    pass

                    if not use_two_pass:
                        # TRADITIONAL WORKFLOW (FALLBACK)
                        logger.info(f"    Using traditional download (full video + extraction)")

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
                            logger.warning(f"    Failed to download {video.video_id}")
                            continue

                        logger.info(f"    Downloaded: {Path(downloaded_file).name}")

                        # Extract clips from full video
                        if self.clip_extractor:
                            clips, should_delete = await loop.run_in_executor(
                                None,
                                self.clip_extractor.process_downloaded_video,
                                downloaded_file,
                                need.search_phrase,
                                video.video_id,
                                None,  # Use same directory as source
                            )

                            if clips:
                                final_files.extend([c.clip_path for c in clips])
                                logger.info(f"    Extracted {len(clips)} clips")

                                # Delete original immediately if clips extracted and setting enabled
                                if should_delete and self.delete_original_after_extraction:
                                    self.clip_extractor.cleanup_original_video(downloaded_file)
                                    logger.info(f"    Deleted original: {Path(downloaded_file).name}")
                            else:
                                # Keep original if no clips extracted
                                final_files.append(downloaded_file)
                                logger.info(f"    No clips extracted, keeping original")
                        else:
                            final_files.append(downloaded_file)

                except Exception as e:
                    logger.error(f"    Failed to process video {video.video_id}: {e}")
                    continue

            need_downloads[need.folder_name] = final_files
            total_clips += len(final_files)
            logger.info(f"  Completed: {len(final_files)} clips total")

            # Clean up intermediate files (previews, webm, failed downloads) to save disk space
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.video_downloader.cleanup_intermediate_files,
                Path(need_folder),
            )
        """

        # Complete the processing stage
        if status:
            status.complete_stage("process_needs")

        # Clean up empty directories
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.file_organizer._cleanup_empty_directories,
        )

        # Send completion notification
        processing_time = self._format_processing_time(time.time() - start_time)
        await self._send_notification(
            "completed",
            project_dir,
            drive_folder_url,
            processing_time,
            file_path,
            total_clips,
        )

        logger.info(
            f"Pipeline complete: {total_clips} clips from {len(broll_plan.needs)} B-roll needs"
        )

        # Clean up local output if Google Drive is configured
        if self.drive_service and project_dir:
            await self._cleanup_local_output(project_dir)

    async def _process_single_need(
        self,
        need: 'BRollNeed',
        need_index: int,
        total_needs: int,
        project_dir: str,
        drive_project_folder_id: Optional[str],
    ) -> Optional[tuple[str, List[str]]]:
        """Process a single B-roll need: search, evaluate, download, extract clips.

        Args:
            need: BRollNeed object with search phrase and timestamp
            need_index: Index of this need (1-based)
            total_needs: Total number of needs being processed
            project_dir: Local project directory
            drive_project_folder_id: Google Drive folder ID for this project

        Returns:
            Tuple of (folder_name, list of clip files) or None if failed
        """
        logger.info(
            f"[{need_index}/{total_needs}] Processing: {need.search_phrase} "
            f"at {need.timestamp:.1f}s"
        )

        try:
            # Search YouTube
            video_results = await self.search_youtube_videos(need.search_phrase)
            logger.info(f"  [{need_index}/{total_needs}] Found {len(video_results)} videos")

            # Evaluate and score videos
            scored_videos = await self.evaluate_videos(need.search_phrase, video_results)
            logger.info(f"  [{need_index}/{total_needs}] Selected {len(scored_videos)} top videos")

            if not scored_videos:
                logger.info(f"  [{need_index}/{total_needs}] No suitable videos found for: {need.search_phrase}")
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
                    need.folder_name,  # Use timestamp-prefixed name
                    drive_project_folder_id,
                )

            # Process videos - stop early if we hit target clip count
            final_files = []
            clips_target = self.config.get("clips_per_need_target", 5)

            # Check if competitive analysis is enabled
            competitive_mode = self.config.get("competitive_analysis_enabled", True)
            previews_per_need = self.config.get("previews_per_need", 2)

            loop = asyncio.get_event_loop()

            if competitive_mode and self.config.get("use_two_pass_download", True) and self.clip_extractor:
                # COMPETITIVE ANALYSIS MODE
                logger.info(f"  [{need_index}/{total_needs}] Competitive analysis: downloading {previews_per_need} previews")

                # Select top N videos for comparison
                preview_videos = scored_videos[:previews_per_need]
                preview_files = []

                # Download all previews
                for video_idx, video in enumerate(preview_videos, 1):
                    try:
                        logger.info(f"    [{need_index}/{total_needs}] Downloading preview {video_idx}/{len(preview_videos)}: {video.video_id}")

                        preview_file = await loop.run_in_executor(
                            None,
                            self.video_downloader.download_preview,
                            video,
                            str(need_folder),
                            self.config.get("preview_max_height", 360),
                        )

                        if preview_file:
                            preview_files.append((video, preview_file))
                            logger.info(f"    [{need_index}/{total_needs}] Preview {video_idx} downloaded: {Path(preview_file).name}")
                        else:
                            logger.warning(f"    [{need_index}/{total_needs}] Preview {video_idx} download failed")
                    except Exception as e:
                        logger.error(f"    [{need_index}/{total_needs}] Failed to download preview {video_idx}: {e}")
                        continue

                if not preview_files:
                    logger.warning(f"  [{need_index}/{total_needs}] No previews downloaded, skipping need")
                else:
                    # Analyze all previews together to find best clip
                    logger.info(f"  [{need_index}/{total_needs}] Analyzing {len(preview_files)} previews...")

                    try:
                        result = await loop.run_in_executor(
                            None,
                            self.clip_extractor.analyze_videos_competitive,
                            preview_files,
                            need.search_phrase,
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
                                logger.info(f"  [{need_index}/{total_needs}] Best clip: {best_segment.start_time:.1f}s-{best_segment.end_time:.1f}s from {source_video.video_id} (score: {best_segment.relevance_score}/10)")

                                # Download winner in high res
                                clip_files = await loop.run_in_executor(
                                    None,
                                    self.video_downloader.download_clip_sections,
                                    source_video,
                                    [best_segment],
                                    str(need_folder),
                                    self.config.get("clip_download_format", "bestvideo[height<=1080]+bestaudio/best"),
                                )

                                if clip_files:
                                    final_files.extend(clip_files)
                                    logger.info(f"  [{need_index}/{total_needs}] Downloaded winning clip in high quality")

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
                                    logger.warning(f"  [{need_index}/{total_needs}] Failed to download winning clip")
                            else:
                                logger.warning(f"  [{need_index}/{total_needs}] Could not identify source video for best clip")
                        else:
                            logger.warning(f"  [{need_index}/{total_needs}] No good clips found in any preview")

                    except Exception as e:
                        logger.error(f"  [{need_index}/{total_needs}] Competitive analysis failed: {e}")

                    # Cleanup all previews
                    for video, preview_file in preview_files:
                        try:
                            Path(preview_file).unlink()
                        except Exception as e:
                            logger.warning(f"  [{need_index}/{total_needs}] Could not delete preview: {e}")

            else:
                # SEQUENTIAL PROCESSING MODE (Original behavior)
                logger.info(f"  [{need_index}/{total_needs}] Sequential processing: analyzing videos one by one")

                for video_idx, video in enumerate(scored_videos, 1):
                    # Early stopping if we have enough clips
                    if len(final_files) >= clips_target:
                        logger.info(f"  [{need_index}/{total_needs}] Target of {clips_target} clips reached, stopping early")
                        break

                    logger.info(f"  [{need_index}/{total_needs}] Video {video_idx}/{len(scored_videos)}: {video.video_id}")

                    try:
                        # Check if two-pass download is enabled and supported
                        use_two_pass = (
                            self.config.get("use_two_pass_download", True)
                            and self.video_downloader._supports_section_downloads()
                            and self.clip_extractor is not None
                        )

                        if use_two_pass:
                            # TWO-PASS WORKFLOW (OPTIMIZED)
                            logger.info(f"    [{need_index}/{total_needs}] Using two-pass download (preview + clips)")

                            # Pass 1: Download low-quality preview
                            preview_file = await loop.run_in_executor(
                                None,
                                self.video_downloader.download_preview,
                                video,
                                str(need_folder),
                                self.config.get("preview_max_height", 360),
                            )

                            if not preview_file:
                                logger.warning(f"    [{need_index}/{total_needs}] Preview download failed, trying traditional download")
                                use_two_pass = False
                            else:
                                logger.info(f"    [{need_index}/{total_needs}] Preview downloaded, analyzing...")

                                # Analyze preview to get clip segments
                                segments = await loop.run_in_executor(
                                    None,
                                    self.clip_extractor.analyze_video,
                                    preview_file,
                                    need.search_phrase,
                                    video.video_id,
                                )

                                if segments and segments.analysis_success and segments.segments:
                                    logger.info(f"    [{need_index}/{total_needs}] Found {len(segments.segments)} clips, downloading in high quality...")

                                    # Pass 2: Download only the identified clips in high quality
                                    clip_files = await loop.run_in_executor(
                                        None,
                                        self.video_downloader.download_clip_sections,
                                        video,
                                        segments.segments,
                                        str(need_folder),
                                        self.config.get(
                                            "clip_download_format",
                                            "bestvideo[height<=1080]+bestaudio/best",
                                        ),
                                    )

                                    if clip_files:
                                        final_files.extend(clip_files)
                                        logger.info(f"    [{need_index}/{total_needs}] Downloaded {len(clip_files)} clips in high quality")

                                        # Clean up preview
                                        try:
                                            Path(preview_file).unlink()
                                            logger.debug(f"    [{need_index}/{total_needs}] Deleted preview")
                                        except Exception as e:
                                            logger.warning(f"    [{need_index}/{total_needs}] Could not delete preview: {e}")

                                        # Upload clips to Drive if configured
                                        if self.drive_service and drive_need_folder_id:
                                            for clip_file in clip_files:
                                                await loop.run_in_executor(
                                                    None,
                                                    self.drive_service.upload_file,
                                                    clip_file,
                                                    drive_need_folder_id,
                                                )

                                        continue  # Skip traditional workflow
                                    else:
                                        logger.warning(f"    [{need_index}/{total_needs}] No clips downloaded, trying traditional download")
                                        use_two_pass = False
                                else:
                                    logger.warning(f"    [{need_index}/{total_needs}] No clips found in preview, trying traditional download")
                                    use_two_pass = False

                                # Clean up preview if falling back
                                if not use_two_pass and preview_file:
                                    try:
                                        Path(preview_file).unlink()
                                    except Exception:
                                        pass

                        if not use_two_pass:
                            # TRADITIONAL WORKFLOW (FALLBACK)
                            logger.info(f"    [{need_index}/{total_needs}] Using traditional download (full video + extraction)")

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
                                logger.warning(f"    [{need_index}/{total_needs}] Failed to download {video.video_id}")
                                continue

                            logger.info(f"    [{need_index}/{total_needs}] Downloaded: {Path(downloaded_file).name}")

                            # Extract clips from full video
                            if self.clip_extractor:
                                clips, should_delete = await loop.run_in_executor(
                                    None,
                                    self.clip_extractor.process_downloaded_video,
                                    downloaded_file,
                                    need.search_phrase,
                                    video.video_id,
                                    None,  # Use same directory as source
                                )

                                if clips:
                                    final_files.extend([c.clip_path for c in clips])
                                    logger.info(f"    [{need_index}/{total_needs}] Extracted {len(clips)} clips")

                                    # Delete original immediately if clips extracted and setting enabled
                                    if should_delete and self.delete_original_after_extraction:
                                        self.clip_extractor.cleanup_original_video(downloaded_file)
                                        logger.info(f"    [{need_index}/{total_needs}] Deleted original: {Path(downloaded_file).name}")
                                else:
                                    # Keep original if no clips extracted
                                    final_files.append(downloaded_file)
                                    logger.info(f"    [{need_index}/{total_needs}] No clips extracted, keeping original")
                            else:
                                final_files.append(downloaded_file)

                    except Exception as e:
                        logger.error(f"    [{need_index}/{total_needs}] Failed to process video {video.video_id}: {e}")
                        continue

            logger.info(f"  [{need_index}/{total_needs}] Completed: {len(final_files)} clips total")

            # Clean up intermediate files (previews, webm, failed downloads) to save disk space
            await loop.run_in_executor(
                None,
                self.video_downloader.cleanup_intermediate_files,
                Path(need_folder),
            )

            return (need.folder_name, final_files)

        except Exception as e:
            logger.error(f"[{need_index}/{total_needs}] Failed to process need '{need.search_phrase}': {e}")
            return None

    async def transcribe_audio(self, file_path: str) -> TranscriptResult:
        """Transcribe audio content using Whisper with timestamps.

        Returns:
            TranscriptResult with text, segments (with timestamps), and duration
        """
        if not self.transcription_service.is_supported_file(file_path):
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")

        # Get full transcript with timestamps
        transcript_result = await self.transcription_service.transcribe_audio(
            file_path, with_timestamps=True
        )
        return transcript_result

    async def extract_search_phrases(self, transcript: str) -> List[str]:
        """Extract relevant search phrases using Gemini AI (legacy method)."""
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided for phrase extraction")
            return []

        loop = asyncio.get_event_loop()
        search_phrases = await loop.run_in_executor(
            None, self.ai_service.extract_search_phrases, transcript
        )
        return search_phrases

    async def plan_broll_needs(
        self, transcript_result: TranscriptResult, source_file: str, user_preferences: UserPreferences = None
    ) -> BRollPlan:
        """Plan timeline-aware B-roll needs from transcript.

        Uses AI to identify specific moments in the source video that need
        B-roll footage, with target density of clips_per_minute.

        Args:
            transcript_result: TranscriptResult with segments and timestamps
            source_file: Path to source file for reference
            user_preferences: Optional user preferences for B-roll customization

        Returns:
            BRollPlan with list of BRollNeed objects
        """
        if not transcript_result.text or not transcript_result.text.strip():
            logger.warning("Empty transcript provided for B-roll planning")
            return BRollPlan(
                source_duration=transcript_result.duration,
                needs=[],
                clips_per_minute=self.clips_per_minute,
                source_file=source_file,
            )

        loop = asyncio.get_event_loop()
        broll_plan = await loop.run_in_executor(
            None,
            lambda: self.ai_service.plan_broll_needs(
                transcript_result,
                self.clips_per_minute,
                source_file,
                self.content_filter,
                user_preferences,
            ),
        )
        return broll_plan

    async def transcribe_and_prompt(self, file_path: str) -> TranscriptResult:
        """Transcribe video and return result for interactive prompting.

        Used by interactive mode to get transcript before asking questions.
        This allows the UI to show transcript preview and generate context-aware
        questions before proceeding with B-roll planning.

        Args:
            file_path: Path to video file to transcribe

        Returns:
            TranscriptResult with text, segments, and duration
        """
        return await self.transcribe_audio(file_path)

    async def search_youtube_videos(self, phrase: str) -> List[VideoResult]:
        """Search YouTube for videos matching the phrase."""
        if not phrase or not phrase.strip():
            logger.warning("Empty search phrase provided")
            return []

        loop = asyncio.get_event_loop()
        video_results = await loop.run_in_executor(
            None, self.youtube_service.search_videos, phrase
        )
        return video_results

    async def evaluate_videos(
        self, phrase: str, videos: List[VideoResult]
    ) -> List[ScoredVideo]:
        """Evaluate videos using Gemini AI."""
        if not videos:
            logger.info(f"No videos to evaluate for phrase: {phrase}")
            return []

        loop = asyncio.get_event_loop()
        scored_videos = await loop.run_in_executor(
            None,
            lambda: self.ai_service.evaluate_videos(phrase, videos, self.content_filter)
        )

        max_videos = self.config.get("max_videos_per_phrase", 3)
        limited_videos = scored_videos[:max_videos]
        return limited_videos

    async def _create_project_structure(
        self, file_path: str, source_filename: str
    ) -> str:
        """Create the project folder structure upfront."""
        loop = asyncio.get_event_loop()
        project_path = await loop.run_in_executor(
            None,
            self.file_organizer.create_project_structure,
            file_path,
            source_filename,
        )
        return project_path

    def _generate_project_name(self, file_path: str, source_filename: str) -> str:
        """Generate a consistent project name for both local and Drive folders."""
        from datetime import datetime
        import hashlib

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

        if source_filename:
            source_base = Path(source_filename).stem
            source_base = self.file_organizer._sanitize_folder_name(source_base)[:30]
            return f"{source_base}_{file_hash}_{timestamp}"
        else:
            return f"{file_hash}_{timestamp}"

    def _format_processing_time(self, seconds: float) -> str:
        """Format processing time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"

    async def _send_notification(
        self,
        status: str,
        output_path: Optional[str] = None,
        drive_folder_url: Optional[str] = None,
        processing_time: Optional[str] = None,
        input_file: Optional[str] = None,
        video_count: Optional[int] = None,
    ) -> None:
        """Send email notification about processing completion."""
        if not self.notification_service:
            logger.debug("Email notifications not configured, skipping notification")
            return

        logger.info(f"Sending notification: {status}")

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self.notification_service.send_notification,
                status,
                output_path,
                drive_folder_url,
                processing_time,
                input_file,
                video_count,
            )
            logger.info(f"Email notification sent: {status}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _cleanup_local_output(self, project_dir: str) -> None:
        """Clean up local output directory after successful Drive upload."""
        import shutil

        try:
            if Path(project_dir).exists():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, shutil.rmtree, project_dir)
                logger.info(f"Cleaned up local output directory: {project_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up local directory {project_dir}: {e}")

    async def _extract_clips_from_videos(
        self,
        downloaded_files: List[str],
        phrase: str,
        scored_videos: List[ScoredVideo],
    ) -> List[str]:
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
        video_id_lookup = {}
        for video_path in downloaded_files:
            path_lower = video_path.lower()
            for sv in scored_videos:
                if sv.video_id.lower() in path_lower:
                    video_id_lookup[video_path] = sv.video_id
                    break
            if video_path not in video_id_lookup:
                # Fallback: extract from filename pattern
                video_id_lookup[video_path] = "unknown"

        final_files = []
        originals_to_delete = []

        for video_path in downloaded_files:
            video_id = video_id_lookup.get(video_path, "unknown")

            logger.info(f"Extracting clips from: {Path(video_path).name}")

            loop = asyncio.get_event_loop()
            clips, should_delete = await loop.run_in_executor(
                None,
                self.clip_extractor.process_downloaded_video,
                video_path,
                phrase,
                video_id,
                None,  # Use same directory as source
            )

            if clips:
                # Add clip paths to final files
                final_files.extend([c.clip_path for c in clips])
                if should_delete and self.delete_original_after_extraction:
                    originals_to_delete.append(video_path)
                logger.info(
                    f"Extracted {len(clips)} clips from {Path(video_path).name}"
                )
            else:
                # Keep original if no clips extracted
                final_files.append(video_path)
                logger.info(
                    f"No clips extracted, keeping original: {Path(video_path).name}"
                )

        # Clean up original videos (with input protection check)
        for original_path in originals_to_delete:
            # CRITICAL: Never delete protected input files
            if self._is_protected_file(original_path):
                logger.warning(
                    f"BLOCKED deletion of protected input file: {original_path}"
                )
                continue

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.clip_extractor.cleanup_original_video, original_path
            )

        return final_files

    def _is_protected_file(self, file_path: str) -> bool:
        """Check if a file path is a protected input file that must NOT be deleted.

        Args:
            file_path: Path to check

        Returns:
            True if the file is protected and should not be deleted
        """
        resolved_path = str(Path(file_path).resolve())
        return resolved_path in self.protected_input_files
