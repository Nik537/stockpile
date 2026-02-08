"""Main stockpile class for orchestrating the entire workflow.

S2 IMPROVEMENT: AI Response Caching
- Injects AIResponseCache into AIService and ClipExtractor
- 100% cost savings on re-runs with cached AI responses
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from models.broll_need import BRollNeed, BRollPlan, TranscriptResult
from models.image import ImageNeed, ImagePlan
from models.style import ContentStyle
from models.user_preferences import UserPreferences
from models.video import ScoredVideo, VideoResult
from services.ai_service import AIService
from services.feedback_service import FeedbackService
from services.clip_extractor import ClipExtractor
from services.drive_service import DriveService
from services.file_monitor import FileMonitor
from services.file_organizer import FileOrganizer
from services.notification import NotificationService
from services.semantic_verifier import SemanticVerifier
from services.transcription import TranscriptionService
from services.video_downloader import VideoDownloader
from services.video_filter import VideoPreFilter
from services.video_sources import (
    PexelsVideoSource,
    PixabayVideoSource,
    VideoSource,
    YouTubeVideoSource,
)
from services.image_sources import (
    ImageSource,
    PexelsImageSource,
    PixabayImageSource,
    GoogleImageSource,
)
from services.image_downloader import ImageAcquisitionService
from services.transcription_planning_service import TranscriptionPlanningService
from services.video_acquisition_service import VideoAcquisitionService
from services.video_search_service import VideoSearchService
from utils.cache import AIResponseCache, load_cache_from_config
from utils.checkpoint import (
    ProcessingCheckpoint,
    cleanup_checkpoint,
    get_checkpoint_path,
)
from utils.config import load_config, validate_config
from utils.cost_tracker import CostTracker
from utils.logging import clear_job_context, set_job_context
from utils.progress import ProcessingStatus

logger = logging.getLogger(__name__)


class BRollProcessor:
    """Central orchestrator for stockpile."""

    def __init__(
        self,
        config: Optional[dict] = None,
        # Dependency injection - all optional for backward compatibility
        ai_service: Optional["AIService"] = None,
        notification_service: Optional["NotificationService"] = None,
        video_sources: Optional[list] = None,
        image_sources: Optional[list] = None,
        image_acquisition_service: Optional["ImageAcquisitionService"] = None,
        transcription_service: Optional["TranscriptionService"] = None,
        video_downloader: Optional["VideoDownloader"] = None,
        file_organizer: Optional["FileOrganizer"] = None,
        drive_service: Optional["DriveService"] = None,
        file_monitor: Optional["FileMonitor"] = None,
        clip_extractor: Optional["ClipExtractor"] = None,
        cost_tracker: Optional["CostTracker"] = None,
        video_prefilter: Optional["VideoPreFilter"] = None,
        semantic_verifier: Optional["SemanticVerifier"] = None,
        feedback_service: Optional["FeedbackService"] = None,
        ai_cache: Optional["AIResponseCache"] = None,
    ):
        """Initialize the stockpile with configuration.

        All service parameters are optional for dependency injection.
        When not provided, services are created internally using config.
        This enables testing with mock services while maintaining
        backward compatibility with existing code.
        """
        self.config = config or load_config()
        self.processing_files: set[str] = set()
        self.protected_input_files: set[str] = set()  # Input videos that must NEVER be deleted
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

        # S2 IMPROVEMENT: Initialize AI response cache
        # Provides 100% cost savings on re-runs with same content
        if ai_cache is not None:
            self.ai_cache = ai_cache
            if self.ai_cache and getattr(self.ai_cache, 'enabled', False):
                logger.info("[S2] AI response caching ENABLED (injected)")
            else:
                logger.info("[S2] AI response caching DISABLED (injected)")
        else:
            cache_config = {
                "cache_enabled": self.config.get("ai_cache_enabled", True),
                "cache_dir": self.config.get("ai_cache_dir", ".cache/ai_responses"),
                "cache_ttl_days": self.config.get("ai_cache_ttl_days", 30),
                "cache_max_size_gb": self.config.get("ai_cache_max_size_gb", 1.0),
            }
            self.ai_cache = load_cache_from_config(cache_config)
            if self.ai_cache and self.ai_cache.enabled:
                logger.info(
                    f"[S2] AI response caching ENABLED: dir={cache_config['cache_dir']}, "
                    f"TTL={cache_config['cache_ttl_days']}d, max_size={cache_config['cache_max_size_gb']}GB"
                )
            else:
                logger.info("[S2] AI response caching DISABLED")

        if ai_service is not None:
            self.ai_service = ai_service
        else:
            self.ai_service = AIService(gemini_api_key, gemini_model, cache=self.ai_cache)

        client_id = self.config.get("google_client_id")
        client_secret = self.config.get("google_client_secret")
        if notification_service is not None:
            self.notification_service = notification_service
        else:
            if client_id and client_secret:
                notification_email = self.config.get("notification_email")
                self.notification_service = NotificationService(
                    client_id, client_secret, notification_email
                )
            else:
                self.notification_service = None

        # PHASE 3 FEATURES: Video source abstraction for multi-platform support
        # Q3 IMPROVEMENT: Multi-source search (YouTube, Pexels, Pixabay)
        max_videos_per_phrase = self.config.get("max_videos_per_phrase", 3)

        if video_sources is not None:
            self.video_sources: list[VideoSource] = video_sources
            logger.info(
                f"[VideoSources] Active sources (injected): {[s.get_source_name() for s in self.video_sources]}"
            )
        else:
            self.video_sources: list[VideoSource] = []

            # Get enabled search sources from config
            search_sources = self.config.get("search_sources", ["youtube", "pexels", "pixabay"])
            prefer_stock = self.config.get("prefer_stock_footage", True)

            # Initialize video sources based on configuration order
            # If prefer_stock_footage is True, reorder to put stock sources first
            source_order = list(search_sources)
            if prefer_stock:
                stock_sources = [s for s in source_order if s in ("pexels", "pixabay")]
                other_sources = [s for s in source_order if s not in ("pexels", "pixabay")]
                source_order = stock_sources + other_sources

            for source_name in source_order:
                source_name = source_name.strip().lower()

                if source_name == "youtube":
                    youtube_source = YouTubeVideoSource(max_results=max_videos_per_phrase * 3)
                    self.video_sources.append(youtube_source)
                    logger.info("[VideoSources] Enabled: YouTube")

                elif source_name == "pexels":
                    pexels_source = PexelsVideoSource(max_results=max_videos_per_phrase * 3)
                    if pexels_source.is_configured():
                        self.video_sources.append(pexels_source)
                        logger.info("[VideoSources] Enabled: Pexels (CC0 stock footage)")
                    else:
                        logger.debug("[VideoSources] Pexels skipped - no API key configured")

                elif source_name == "pixabay":
                    pixabay_source = PixabayVideoSource(max_results=max_videos_per_phrase * 3)
                    if pixabay_source.is_configured():
                        self.video_sources.append(pixabay_source)
                        logger.info("[VideoSources] Enabled: Pixabay (CC0 stock footage)")
                    else:
                        logger.debug("[VideoSources] Pixabay skipped - no API key configured")

                else:
                    logger.warning(f"[VideoSources] Unknown source '{source_name}' - skipping")

            if not self.video_sources:
                # Fallback to YouTube if no sources configured
                logger.warning("[VideoSources] No valid sources configured, falling back to YouTube")
                self.video_sources.append(YouTubeVideoSource(max_results=max_videos_per_phrase * 3))

            logger.info(
                f"[VideoSources] Active sources: {[s.get_source_name() for s in self.video_sources]}"
            )

        # IMAGE ACQUISITION: Initialize image sources for parallel image downloads
        if image_sources is not None:
            self.image_sources: list[ImageSource] = image_sources
            logger.info(
                f"[ImageSources] Image sources (injected): {[s.get_source_name() for s in self.image_sources]}"
            )
        else:
            self.image_sources: list[ImageSource] = []

            image_acquisition_enabled = self.config.get("image_acquisition_enabled", True)
            if image_acquisition_enabled:
                image_source_names = self.config.get("image_sources", ["pexels", "pixabay", "google"])

                for source_name in image_source_names:
                    source_name = source_name.strip().lower()

                    if source_name == "pexels":
                        pexels_img_source = PexelsImageSource(max_results=5)
                        if pexels_img_source.is_configured():
                            self.image_sources.append(pexels_img_source)
                            logger.debug("[ImageSources] Enabled: Pexels Photos")

                    elif source_name == "pixabay":
                        pixabay_img_source = PixabayImageSource(max_results=5)
                        if pixabay_img_source.is_configured():
                            self.image_sources.append(pixabay_img_source)
                            logger.debug("[ImageSources] Enabled: Pixabay Images")

                    elif source_name == "google":
                        google_img_source = GoogleImageSource(max_results=5)
                        if google_img_source.is_configured():
                            self.image_sources.append(google_img_source)
                            logger.debug("[ImageSources] Enabled: Google Images")

        if image_acquisition_service is not None:
            self.image_acquisition_service: Optional[ImageAcquisitionService] = image_acquisition_service
            logger.info("[ImageSources] Image acquisition service (injected)")
        else:
            self.image_acquisition_service: Optional[ImageAcquisitionService] = None

            if self.image_sources:
                self.image_acquisition_service = ImageAcquisitionService(
                    image_sources=self.image_sources,
                    ai_service=self.ai_service,
                    output_dir=self.config.get("local_output_folder", "../output"),
                    max_concurrent_downloads=self.config.get("parallel_image_downloads", 10),
                )
                logger.info(
                    f"[ImageSources] Image acquisition ENABLED: "
                    f"{[s.get_source_name() for s in self.image_sources]}, "
                    f"interval={self.config.get('image_interval_seconds', 5.0)}s"
                )
            else:
                image_acquisition_enabled = self.config.get("image_acquisition_enabled", True)
                if image_acquisition_enabled:
                    logger.info("[ImageSources] Image acquisition DISABLED (no sources configured)")
                else:
                    logger.info("[ImageSources] Image acquisition DISABLED (config)")

        whisper_model = self.config.get("whisper_model", "base")
        if transcription_service is not None:
            self.transcription_service = transcription_service
        else:
            self.transcription_service = TranscriptionService(whisper_model)

        output_dir = self.config.get("local_output_folder", "../output")
        if video_downloader is not None:
            self.video_downloader = video_downloader
        else:
            self.video_downloader = VideoDownloader(output_dir)

        if file_organizer is not None:
            self.file_organizer = file_organizer
        else:
            self.file_organizer = FileOrganizer(output_dir)

        if drive_service is not None:
            self.drive_service = drive_service
        else:
            output_folder_id = self.config.get("google_drive_output_folder_id")
            if output_folder_id:
                if not client_id:
                    raise ValueError("Google Drive requires GOOGLE_CLIENT_ID environment variable")
                if not client_secret:
                    raise ValueError("Google Drive requires GOOGLE_CLIENT_SECRET environment variable")
                self.drive_service = DriveService(client_id, client_secret, output_folder_id)
            else:
                self.drive_service = None

        if file_monitor is not None:
            self.file_monitor = file_monitor
        else:
            self.file_monitor = FileMonitor(self.config, self._handle_new_file)

        # Initialize clip extractor if enabled
        # Always read delete_original_after_extraction from config, even when clip_extractor is injected
        self.delete_original_after_extraction = self.config.get(
            "delete_original_after_extraction", True
        )

        if clip_extractor is not None:
            self.clip_extractor = clip_extractor
            logger.info("Clip extraction enabled (injected)")
        else:
            clip_extraction_enabled = self.config.get("clip_extraction_enabled", True)
            if clip_extraction_enabled:
                self.clip_extractor = ClipExtractor(
                    api_key=gemini_api_key,
                    model_name=gemini_model,
                    min_clip_duration=self.config.get("min_clip_duration", 4.0),
                    max_clip_duration=self.config.get("max_clip_duration", 15.0),
                    max_clips_per_video=self.config.get("max_clips_per_video", 3),
                    cache=self.ai_cache,  # S2 IMPROVEMENT: Inject cache for video analysis
                )
                logger.info("Clip extraction enabled (with AI cache)")
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

        # PHASE 2 PERFORMANCE: Resource limiting with semaphores
        self.download_semaphore = asyncio.Semaphore(self.config.get("parallel_downloads", 3))
        self.extraction_semaphore = asyncio.Semaphore(self.config.get("parallel_extractions", 2))
        self.ai_semaphore = asyncio.Semaphore(self.config.get("parallel_ai_calls", 5))
        logger.info(
            f"Parallel processing limits: downloads={self.config.get('parallel_downloads', 3)}, "
            f"extractions={self.config.get('parallel_extractions', 2)}, "
            f"AI calls={self.config.get('parallel_ai_calls', 5)}"
        )

        # PHASE 3 FEATURES: Cost tracking
        if cost_tracker is not None:
            self.cost_tracker = cost_tracker
            logger.info("Cost tracking enabled (injected)")
        else:
            budget_limit = self.config.get("budget_limit_usd", 0.0)
            self.cost_tracker = CostTracker(
                budget_limit_usd=budget_limit if budget_limit > 0 else None
            )
            if budget_limit > 0:
                logger.info(f"Cost tracking enabled with budget limit: ${budget_limit:.2f}")
            else:
                logger.info("Cost tracking enabled (no budget limit)")

        # S5 IMPROVEMENT: Video pre-filtering before downloads
        # Filters videos based on metadata (views, duration, keywords) to avoid wasted bandwidth
        if video_prefilter is not None:
            self.video_prefilter = video_prefilter
            logger.info("Video pre-filter enabled (injected)")
        else:
            self.video_prefilter = VideoPreFilter(self.config)
            logger.info(
                f"Video pre-filter enabled: min_views={self.video_prefilter.filter_config.min_view_count}, "
                f"max_duration={self.video_prefilter.filter_config.max_prefilter_duration}s, "
                f"blocked_keywords={len(self.video_prefilter.filter_config.blocked_title_keywords)}"
            )

        # S6 IMPROVEMENT: Semantic verification for clip quality assurance
        # Verifies extracted clips semantically match original transcript context
        if semantic_verifier is not None:
            self.semantic_verifier = semantic_verifier
            logger.info("Semantic verification ENABLED (injected)")
        else:
            semantic_verification_enabled = self.config.get("semantic_verification_enabled", False)
            if semantic_verification_enabled:
                self.semantic_verifier = SemanticVerifier(
                    model_name=gemini_model,
                    api_key=gemini_api_key,
                )
                logger.info(
                    f"Semantic verification ENABLED: threshold={self.config.get('semantic_match_threshold', 0.9)}, "
                    f"reject_below_threshold={self.config.get('reject_below_threshold', True)}"
                )
            else:
                self.semantic_verifier = None
                logger.info("Semantic verification DISABLED")

        # Feature 3: Feedback Loop - learns from user rejections to improve selection
        if feedback_service is not None:
            self.feedback_service = feedback_service
            logger.info("Feedback system ENABLED (injected)")
            # Connect feedback service to image acquisition even when injected
            if self.image_acquisition_service:
                self.image_acquisition_service.feedback_service = self.feedback_service
        else:
            feedback_enabled = self.config.get("feedback_enabled", True)
            if feedback_enabled:
                feedback_dir = self.config.get("feedback_dir", ".stockpile")
                self.feedback_service = FeedbackService(feedback_dir)
                stats = self.feedback_service.get_statistics()
                logger.info(
                    f"Feedback system ENABLED: {stats['total_rejections']} rejections, "
                    f"{stats['total_approvals']} approvals stored in {feedback_dir}"
                )
                # Connect feedback service to image acquisition
                if self.image_acquisition_service:
                    self.image_acquisition_service.feedback_service = self.feedback_service
            else:
                self.feedback_service = None
                logger.info("Feedback system DISABLED")

        # Feature 1: Style detection - will be populated per video during processing
        self.content_style: Optional[ContentStyle] = None

        # Store config values needed by decomposed services
        self.max_videos_per_phrase = self.config.get("max_videos_per_phrase", 3)
        self.evaluation_context_seconds = self.config.get("evaluation_context_seconds", 30.0)

        # DECOMPOSITION: Create decomposed services for cleaner architecture
        # VideoSearchService: handles multi-source search, fallback, and evaluation
        self.video_search_service = VideoSearchService(
            video_sources=self.video_sources,
            video_prefilter=self.video_prefilter,
            ai_service=self.ai_service,
            feedback_service=self.feedback_service,
            content_style=self.content_style,
            content_filter=self.content_filter,
            max_videos_per_phrase=self.max_videos_per_phrase,
            evaluation_context_seconds=self.evaluation_context_seconds,
        )

        # TranscriptionPlanningService: handles transcription and B-roll planning
        self.transcription_planning_service = TranscriptionPlanningService(
            transcription_service=self.transcription_service,
            ai_service=self.ai_service,
            clips_per_minute=self.clips_per_minute,
            content_filter=self.content_filter,
        )

        # VideoAcquisitionService: handles download, extraction, and verification
        self.video_acquisition_service = VideoAcquisitionService(
            video_downloader=self.video_downloader,
            clip_extractor=self.clip_extractor,
            semantic_verifier=self.semantic_verifier,
            file_organizer=self.file_organizer,
            video_search_service=self.video_search_service,
            ai_service=self.ai_service,
            feedback_service=self.feedback_service,
            drive_service=self.drive_service,
            download_semaphore=self.download_semaphore,
            extraction_semaphore=self.extraction_semaphore,
            ai_semaphore=self.ai_semaphore,
            clips_per_need_target=self.config.get("clips_per_need_target", 5),
            use_two_pass_download=self.config.get("use_two_pass_download", True),
            competitive_analysis_enabled=self.config.get("competitive_analysis_enabled", True),
            previews_per_need=self.config.get("previews_per_need", 2),
            content_filter=self.content_filter,
            delete_original_after_extraction=self.delete_original_after_extraction,
            max_clip_duration=self.config.get("max_clip_duration", 15.0),
            min_clip_duration=self.config.get("min_clip_duration", 4.0),
            max_videos_per_phrase=self.max_videos_per_phrase,
            preview_max_height=self.config.get("preview_max_height", 360),
            clip_download_format=self.config.get(
                "clip_download_format", "bestvideo[height<=1080]+bestaudio/best"
            ),
            reject_below_threshold=self.config.get("reject_below_threshold", True),
            protected_input_files=self.protected_input_files,
        )

        logger.info("stockpile initialized successfully (with decomposed services)")

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
            asyncio.run_coroutine_threadsafe(self.process_video(file_path), self.event_loop)
        else:
            logger.error("No event loop available to schedule file processing")

    async def process_video(
        self,
        file_path: str,
        user_preferences: UserPreferences = None,
        status_callback=None,
        resume: bool = True,
    ) -> str | None:
        """Process a video file through the complete B-roll pipeline.

        Args:
            file_path: Path to the video file to process
            user_preferences: Optional user preferences for B-roll style customization
            status_callback: Optional callback function for progress updates
            resume: Whether to resume from checkpoint if available (default: True)

        Returns:
            Path to the output project directory, or None if processing failed
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

        # PHASE 3A: Set job context for correlated logging
        # Generate a simple job ID from the file path hash
        job_id = str(abs(hash(file_path)))[:8]
        set_job_context(job_id)
        logger.info(f"Job context set: {job_id}")

        # PHASE 3 FEATURES: Load checkpoint if resuming
        output_dir = self.config.get("local_output_folder", "../output")
        checkpoint_path = get_checkpoint_path(file_path, output_dir)
        checkpoint = None

        if resume and checkpoint_path.exists():
            checkpoint = ProcessingCheckpoint.load(checkpoint_path)
            if checkpoint:
                logger.info(
                    f"📍 Resuming from checkpoint (stage: {checkpoint.stage}, "
                    f"completed: {len(checkpoint.completed_needs)}/{checkpoint.total_needs} needs)"
                )
        else:
            # Initialize new checkpoint
            checkpoint = ProcessingCheckpoint(
                video_path=file_path,
                stage="initialized",
            )
            checkpoint.save(checkpoint_path)

        start_time = time.time()

        # Initialize progress tracking
        status = ProcessingStatus(
            video_path=file_path,
            output_dir=output_dir,
            update_callback=status_callback,
        )

        try:
            project_dir = await self._execute_pipeline(
                file_path, start_time, user_preferences, status, checkpoint
            )
            status.complete_processing()

            # PHASE 3 FEATURES: Clean up checkpoint after successful completion
            cleanup_checkpoint(checkpoint_path)

            # S2 IMPROVEMENT: Log cache statistics after processing
            if self.ai_cache and self.ai_cache.enabled:
                self.ai_cache.log_stats()

            logger.info(f"Processing completed successfully: {file_path}")
            return project_dir

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
            # PHASE 3A: Clear job context after processing completes
            clear_job_context()

    async def _execute_pipeline(
        self,
        file_path: str,
        start_time: float,
        user_preferences: UserPreferences = None,
        status: ProcessingStatus = None,
        checkpoint: ProcessingCheckpoint | None = None,
    ) -> str | None:
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

        # PHASE 3 FEATURES: Track transcription cost
        if self.cost_tracker:
            duration_minutes = transcript_result.duration / 60.0
            self.cost_tracker.track_whisper_call(duration_minutes)

        # PHASE 3 FEATURES: Save checkpoint after transcription
        if checkpoint:
            checkpoint.stage = "transcribed"
            checkpoint.save(get_checkpoint_path(file_path, self.config.get("local_output_folder", "../output")))

        if status:
            status.update_stage("transcribe", completed=1)
            status.complete_stage("transcribe")
        duration_minutes = transcript_result.duration / 60.0
        logger.info(
            f"Transcription completed: {len(transcript_result.text)} chars, "
            f"{len(transcript_result.segments)} segments, {duration_minutes:.1f} min duration"
        )

        # Feature 1: Detect content style (topic, audience, visual style)
        # This is done once per video and passed to all evaluations
        style_detection_enabled = self.config.get("style_detection_enabled", True)
        if style_detection_enabled:
            logger.info("Detecting content style from transcript...")
            self.content_style = self.ai_service.detect_content_style(
                transcript_result=transcript_result,
                user_preferences=user_preferences,
            )
            logger.info(
                f"Content style detected: topic='{self.content_style.topic}', "
                f"audience='{self.content_style.target_audience}', "
                f"visual_style={self.content_style.visual_style.value}"
            )
            # Pass content style to image acquisition service and video search service
            if self.image_acquisition_service:
                self.image_acquisition_service.set_content_style(self.content_style)
            self.video_search_service.set_content_style(self.content_style)
        else:
            self.content_style = None
            logger.info("Style detection DISABLED (config)")
            if self.image_acquisition_service:
                self.image_acquisition_service.set_content_style(None)
            self.video_search_service.set_content_style(None)

        # Step 2: Plan B-roll needs (timeline-aware)
        if status:
            status.start_stage("plan")

        broll_plan = await self.plan_broll_needs(transcript_result, file_path, user_preferences)

        # PHASE 3 FEATURES: Track planning cost (estimated tokens)
        # TODO: Get actual token counts from AI service
        if self.cost_tracker:
            # Estimate: planning uses ~1000 input tokens + ~500 output tokens per minute of video
            estimated_input = int(duration_minutes * 1000)
            estimated_output = int(duration_minutes * 500)
            self.cost_tracker.track_gemini_call("plan_broll_needs", estimated_input, estimated_output)

        # PHASE 3 FEATURES: Save checkpoint after planning
        if checkpoint:
            checkpoint.stage = "planned"
            checkpoint.total_needs = len(broll_plan.needs)
            checkpoint.save(get_checkpoint_path(file_path, self.config.get("local_output_folder", "../output")))

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
                f"  [{i}] {need.timestamp:.1f}s - {need.search_phrase} " f"({need.description})"
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
            drive_folder_url = f"https://drive.google.com/drive/folders/{drive_project_folder_id}"
            logger.info(f"Google Drive structure created: {drive_folder_url}")

        # Step 4: Generate image plan (if image acquisition is enabled)
        # Images are acquired in parallel with video clips
        image_plan: Optional[ImagePlan] = None
        if self.image_acquisition_service:
            logger.info("Generating image acquisition plan...")
            image_interval = self.config.get("image_interval_seconds", 5.0)
            image_plan = self.ai_service.generate_image_queries(
                transcript_result, interval_seconds=image_interval
            )
            logger.info(
                f"Image plan generated: {len(image_plan.needs)} images "
                f"(1 per {image_interval}s for {transcript_result.duration:.1f}s video)"
            )

        # Step 5: Process B-roll needs AND images in PARALLEL
        logger.info(f"Processing {len(broll_plan.needs)} B-roll needs")
        need_downloads: dict[str, list[str]] = {}
        total_clips = 0

        # Register B-roll processing stage
        if status:
            status.register_stage("process_needs", total_items=len(broll_plan.needs))
            if image_plan:
                status.register_stage("process_images", total_items=len(image_plan.needs))
            status.start_stage("process_needs")

        # Create the video processing coroutine
        async def process_all_broll_needs():
            """Process all B-roll video needs."""
            nonlocal need_downloads, total_clips
            max_concurrent = self.config.get("max_concurrent_needs", 5)
            batch_size = max_concurrent

            for batch_start in range(0, len(broll_plan.needs), batch_size):
                batch_end = min(batch_start + batch_size, len(broll_plan.needs))
                batch = broll_plan.needs[batch_start:batch_end]

                logger.info(
                    f"Processing video batch {batch_start//batch_size + 1}/"
                    f"{(len(broll_plan.needs) + batch_size - 1)//batch_size} "
                    f"({len(batch)} needs in parallel)"
                )

                tasks = []
                for i, need in enumerate(batch, batch_start + 1):
                    task = self._process_single_need(
                        need=need,
                        need_index=i,
                        total_needs=len(broll_plan.needs),
                        project_dir=project_dir,
                        drive_project_folder_id=drive_project_folder_id,
                        transcript_result=transcript_result,
                        user_preferences=user_preferences,
                    )
                    tasks.append(task)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch task failed: {result}")
                        if status:
                            status.update_stage("process_needs", increment=1)
                        continue
                    if result:
                        folder_name, files = result
                        need_downloads[folder_name] = files
                        total_clips += len(files)
                    if status:
                        status.update_stage("process_needs", increment=1)

            return need_downloads

        # Create the image processing coroutine (if enabled)
        async def process_all_images():
            """Process all image needs."""
            if not self.image_acquisition_service or not image_plan or not image_plan.needs:
                return {}

            if status:
                status.start_stage("process_images")

            logger.info(f"[ImageAcquisition] Starting parallel image acquisition for {len(image_plan.needs)} images...")
            image_results = await self.image_acquisition_service.process_all_image_needs(
                image_plan.needs, project_dir
            )

            if status:
                status.update_stage("process_images", completed=len(image_results))
                status.complete_stage("process_images")

            logger.info(f"[ImageAcquisition] Completed: {len(image_results)} images acquired")
            return image_results

        # Run video and image acquisition in PARALLEL using asyncio.gather
        if self.image_acquisition_service and image_plan and image_plan.needs:
            logger.info(
                f"Starting PARALLEL acquisition: {len(broll_plan.needs)} video needs + "
                f"{len(image_plan.needs)} image needs"
            )
            video_result, image_result = await asyncio.gather(
                process_all_broll_needs(),
                process_all_images(),
                return_exceptions=True,
            )

            # Handle any exceptions
            if isinstance(video_result, Exception):
                logger.error(f"Video acquisition failed: {video_result}")
                need_downloads = {}
            if isinstance(image_result, Exception):
                logger.error(f"Image acquisition failed: {image_result}")
                image_result = {}

            # Log image results
            if image_result and not isinstance(image_result, Exception):
                logger.info(f"Image acquisition complete: {len(image_result)} images in {project_dir}/images/")
        else:
            # No image acquisition - just process videos
            # Get parallel processing config
            max_concurrent = self.config.get("max_concurrent_needs", 5)

            # Process needs in batches (original sequential batch processing)
            batch_size = max_concurrent
            for batch_start in range(0, len(broll_plan.needs), batch_size):
                batch_end = min(batch_start + batch_size, len(broll_plan.needs))
                batch = broll_plan.needs[batch_start:batch_end]

                logger.info(
                    f"Processing batch {batch_start//batch_size + 1}/{(len(broll_plan.needs) + batch_size - 1)//batch_size} ({len(batch)} needs in parallel)"
                )

                # Process this batch in parallel
                # Q4 IMPROVEMENT: Pass transcript and user preferences for context-aware evaluation
                tasks = []
                for i, need in enumerate(batch, batch_start + 1):
                    task = self._process_single_need(
                        need=need,
                        need_index=i,
                        total_needs=len(broll_plan.needs),
                        project_dir=project_dir,
                        drive_project_folder_id=drive_project_folder_id,
                        transcript_result=transcript_result,
                        user_preferences=user_preferences,
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
                            # Fix 4: Pass broll_need for semantic context matching
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
                        # Fix 4: Pass broll_need for semantic context matching
                        if self.clip_extractor:
                            clips, should_delete = await loop.run_in_executor(
                                None,
                                lambda: self.clip_extractor.process_downloaded_video(
                                    downloaded_file,
                                    need.search_phrase,
                                    video.video_id,
                                    output_dir=None,  # Use same directory as source
                                    broll_need=need,
                                ),
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

        # PHASE 3 FEATURES: Mark checkpoint as completed
        if checkpoint:
            checkpoint.stage = "completed"
            checkpoint.project_dir = project_dir
            checkpoint.save(get_checkpoint_path(file_path, self.config.get("local_output_folder", "../output")))

        # PHASE 3 FEATURES: Save cost report BEFORE cleanup
        if project_dir and self.cost_tracker:
            try:
                self.cost_tracker.save_report(Path(project_dir), file_path)
                logger.info(
                    f"💰 Total cost: ${self.cost_tracker.total_cost:.4f} "
                    f"({len(self.cost_tracker.api_calls)} API calls)"
                )
            except Exception as e:
                logger.warning(f"Failed to save cost report: {e}")

        # S5 IMPROVEMENT: Log pre-filter statistics and save report
        filter_stats = self.video_prefilter.get_stats()
        if filter_stats.total_input > 0:
            logger.info(
                f"🔍 Pre-filter stats: {filter_stats.total_passed}/{filter_stats.total_input} videos passed "
                f"({filter_stats.total_filtered} filtered, {filter_stats.filter_rate:.1f}% rejection rate)"
            )
            # Save filter report to project directory
            if project_dir:
                try:
                    filter_report_path = Path(project_dir) / "filter_report.txt"
                    filter_report_path.write_text(self.video_prefilter.get_report())
                    logger.debug(f"Saved filter report to {filter_report_path}")
                except Exception as e:
                    logger.warning(f"Failed to save filter report: {e}")
            # Reset stats for next video processing
            self.video_prefilter.reset_stats()

        # Clean up local output if Google Drive is configured
        if self.drive_service and project_dir:
            await self._cleanup_local_output(project_dir)

        return project_dir

    async def _process_single_need(
        self,
        need: "BRollNeed",
        need_index: int,
        total_needs: int,
        project_dir: str,
        drive_project_folder_id: Optional[str],
        transcript_result: TranscriptResult = None,
        user_preferences: UserPreferences = None,
    ) -> Optional[tuple[str, list[str]]]:
        """Process a single B-roll need: search, evaluate, download, extract clips.

        DELEGATED to VideoAcquisitionService for cleaner architecture.

        Args:
            need: BRollNeed object with search phrase and timestamp
            need_index: Index of this need (1-based)
            total_needs: Total number of needs being processed
            project_dir: Local project directory
            drive_project_folder_id: Google Drive folder ID for this project
            transcript_result: Optional TranscriptResult for context-aware evaluation
            user_preferences: Optional UserPreferences for style customization

        Returns:
            Tuple of (folder_name, list of clip files) or None if failed
        """
        return await self.video_acquisition_service.process_single_need(
            need=need,
            need_index=need_index,
            total_needs=total_needs,
            project_dir=project_dir,
            drive_project_folder_id=drive_project_folder_id,
            transcript_result=transcript_result,
            user_preferences=user_preferences,
            video_prefilter=self.video_prefilter,
        )

    async def transcribe_audio(self, file_path: str) -> TranscriptResult:
        """Transcribe audio content using Whisper with timestamps.

        DELEGATED to TranscriptionPlanningService for cleaner architecture.

        Returns:
            TranscriptResult with text, segments (with timestamps), and duration
        """
        return await self.transcription_planning_service.transcribe_audio(file_path)

    async def extract_search_phrases(self, transcript: str) -> list[str]:
        """Extract relevant search phrases using Gemini AI (legacy method).

        DELEGATED to TranscriptionPlanningService for cleaner architecture.
        """
        return await self.transcription_planning_service.extract_search_phrases(transcript)

    async def plan_broll_needs(
        self,
        transcript_result: TranscriptResult,
        source_file: str,
        user_preferences: UserPreferences = None,
    ) -> BRollPlan:
        """Plan timeline-aware B-roll needs from transcript.

        DELEGATED to TranscriptionPlanningService for cleaner architecture.

        Args:
            transcript_result: TranscriptResult with segments and timestamps
            source_file: Path to source file for reference
            user_preferences: Optional user preferences for B-roll customization

        Returns:
            BRollPlan with list of BRollNeed objects
        """
        return await self.transcription_planning_service.plan_broll_needs(
            transcript_result, source_file, user_preferences
        )

    async def transcribe_and_prompt(self, file_path: str) -> TranscriptResult:
        """Transcribe video and return result for interactive prompting.

        DELEGATED to TranscriptionPlanningService for cleaner architecture.

        Args:
            file_path: Path to video file to transcribe

        Returns:
            TranscriptResult with text, segments, and duration
        """
        return await self.transcription_planning_service.transcribe_and_prompt(file_path)

    async def search_youtube_videos(self, phrase: str) -> list[VideoResult]:
        """Search video sources for videos matching the phrase.

        DELEGATED to VideoSearchService for cleaner architecture.
        """
        return await self.video_search_service.search_youtube_videos(phrase)

    async def search_with_fallback(
        self, need: BRollNeed, min_results: int = 5
    ) -> list[VideoResult]:
        """Search for videos with fallback to alternate search phrases.

        DELEGATED to VideoSearchService for cleaner architecture.

        Args:
            need: BRollNeed with primary search and optional alternates
            min_results: Minimum results before trying alternates (default: 5)

        Returns:
            List of VideoResult objects, filtered by negative keywords
        """
        return await self.video_search_service.search_with_fallback(need, min_results)

    async def evaluate_videos(self, phrase: str, videos: list[VideoResult]) -> list[ScoredVideo]:
        """Evaluate videos using Gemini AI (legacy method).

        DELEGATED to VideoSearchService for cleaner architecture.

        NOTE: For Q4 context-aware evaluation, use evaluate_videos_enhanced instead.
        """
        return await self.video_search_service.evaluate_videos(phrase, videos)

    async def evaluate_videos_enhanced(
        self,
        need: BRollNeed,
        videos: list[VideoResult],
        transcript_result: TranscriptResult = None,
        user_preferences: UserPreferences = None,
    ) -> list[ScoredVideo]:
        """Evaluate videos using context-aware AI scoring (Q4 improvement).

        DELEGATED to VideoSearchService for cleaner architecture.

        Args:
            need: BRollNeed with search phrase, timestamp, and enhanced metadata
            videos: List of video results to evaluate
            transcript_result: Optional TranscriptResult to extract context segment
            user_preferences: Optional UserPreferences for style customization

        Returns:
            List of scored videos, limited to max_videos_per_phrase
        """
        return await self.video_search_service.evaluate_videos_enhanced(
            need, videos, transcript_result, user_preferences
        )

    async def _create_project_structure(self, file_path: str, source_filename: str) -> str:
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
        import hashlib
        from datetime import datetime

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
        downloaded_files: list[str],
        phrase: str,
        scored_videos: list[ScoredVideo],
    ) -> list[str]:
        """Extract clips from downloaded videos using AI analysis.

        DELEGATED to VideoAcquisitionService for cleaner architecture.

        Args:
            downloaded_files: List of downloaded video file paths
            phrase: Search phrase for context
            scored_videos: List of scored video objects for video ID lookup

        Returns:
            List of final file paths (clips if extraction succeeded, originals otherwise)
        """
        return await self.video_acquisition_service.extract_clips_from_videos(
            downloaded_files, phrase, scored_videos
        )

    async def _download_preview_with_limit(
        self,
        video: "ScoredVideo",
        need_folder: str,
        need_index: int,
        video_idx: int,
        total_videos: int,
    ) -> Optional[tuple["ScoredVideo", str]]:
        """Download a single preview video with semaphore limiting.

        DELEGATED to VideoAcquisitionService for cleaner architecture.

        Args:
            video: Video to download
            need_folder: Output folder
            need_index: Index of current need
            video_idx: Index of this video (1-based)
            total_videos: Total number of videos being downloaded

        Returns:
            Tuple of (video, preview_file_path) or None if download failed
        """
        return await self.video_acquisition_service._download_preview_with_limit(
            video, need_folder, need_index, video_idx, total_videos
        )

    async def _extract_clips_from_single_video(
        self,
        video_path: str,
        video_id: str,
        phrase: str,
        broll_need: Optional[BRollNeed] = None,
    ) -> tuple[list[str], Optional[str]]:
        """Extract clips from a single video with semaphore limiting.

        DELEGATED to VideoAcquisitionService for cleaner architecture.

        Args:
            video_path: Path to video file
            video_id: Video ID for identification
            phrase: Search phrase for context
            broll_need: Optional BRollNeed for semantic context matching

        Returns:
            Tuple of (list of clip paths, original path to delete or None)
        """
        return await self.video_acquisition_service._extract_clips_from_single_video(
            video_path, video_id, phrase, broll_need
        )

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

        DELEGATED to VideoAcquisitionService for cleaner architecture.

        Args:
            clip_paths: List of clip file paths to verify
            broll_need: BRollNeed with original_context and required_elements
            need_index: Index of current need (for logging)
            total_needs: Total number of needs (for logging)

        Returns:
            List of verified clip paths that pass semantic verification threshold
        """
        return await self.video_acquisition_service._verify_clips_semantically(
            clip_paths, broll_need, need_index, total_needs
        )
