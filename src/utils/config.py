"""Configuration loading and validation for stockpile."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

# Get the project root directory (parent of src)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict:
    """Load configuration from environment variables."""

    # Helper function to resolve paths relative to project root
    def resolve_path(path: str | None, default_relative: str) -> str:
        if not path:
            return str(PROJECT_ROOT / default_relative)
        if Path(path).is_absolute():
            return path
        return str(PROJECT_ROOT / path)

    config = {
        # Required API keys
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        # YouTube Data API (optional but recommended for outlier finder)
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY"),
        "use_youtube_api": os.getenv("USE_YOUTUBE_API", "false").lower() == "true",
        # Turso Cloud Database (optional - for cloud channel index)
        "turso_database_url": os.getenv("TURSO_DATABASE_URL"),
        "turso_auth_token": os.getenv("TURSO_AUTH_TOKEN"),
        "use_cloud_cache": os.getenv("USE_CLOUD_CACHE", "false").lower() == "true",
        # Cloudflare R2 Storage (optional - for exports)
        "r2_account_id": os.getenv("R2_ACCOUNT_ID"),
        "r2_access_key_id": os.getenv("R2_ACCESS_KEY_ID"),
        "r2_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY"),
        "r2_bucket_name": os.getenv("R2_BUCKET_NAME", "stockpile-exports"),
        "r2_public_url": os.getenv("R2_PUBLIC_URL"),
        # Reddit Integration (optional - enhances outlier finding)
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID"),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "enable_reddit_discovery": os.getenv("ENABLE_REDDIT_DISCOVERY", "true").lower() == "true",
        "reddit_subreddits": [s.strip() for s in os.getenv("REDDIT_SUBREDDITS", "videos,mealtimevideos,Documentaries").split(",") if s.strip()],
        # Input sources (at least one required)
        "local_input_folder": resolve_path(os.getenv("LOCAL_INPUT_FOLDER"), "input"),
        "google_drive_input_folder_id": os.getenv("GOOGLE_DRIVE_INPUT_FOLDER_ID"),
        # Output destinations (at least one required)
        "local_output_folder": resolve_path(os.getenv("LOCAL_OUTPUT_FOLDER"), "output"),
        "google_drive_output_folder_id": os.getenv("GOOGLE_DRIVE_OUTPUT_FOLDER_ID"),
        # Model configurations
        "whisper_model": os.getenv("WHISPER_MODEL", "base"),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        # Google Drive OAuth
        "google_client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "google_client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        # Email notifications
        "notification_email": os.getenv(
            "NOTIFICATION_EMAIL"
        ),  # Optional: if not set, uses authenticated user's email
        # Processing settings
        "max_videos_per_phrase": int(os.getenv("MAX_VIDEOS_PER_PHRASE", "3")),
        "max_video_duration_seconds": int(
            os.getenv("MAX_VIDEO_DURATION_SECONDS", "600")
        ),
        "max_video_size_mb": int(os.getenv("MAX_VIDEO_SIZE_MB", "100")),
        # Timeline-aware B-roll planning
        "clips_per_minute": float(os.getenv("CLIPS_PER_MINUTE", "2")),
        # Clip extraction settings
        "clip_extraction_enabled": os.getenv("CLIP_EXTRACTION_ENABLED", "true").lower()
        == "true",
        "min_clip_duration": float(os.getenv("MIN_CLIP_DURATION", "4")),
        "max_clip_duration": float(os.getenv("MAX_CLIP_DURATION", "15")),
        "max_clips_per_video": int(os.getenv("MAX_CLIPS_PER_VIDEO", "3")),
        "delete_original_after_extraction": os.getenv(
            "DELETE_ORIGINAL_AFTER_EXTRACTION", "true"
        ).lower()
        == "true",
        # Content filter for B-roll search (e.g., "men only, no women")
        "content_filter": os.getenv("CONTENT_FILTER"),
        # Interactive mode settings
        "interactive_max_questions": int(os.getenv("INTERACTIVE_MAX_QUESTIONS", "3")),
        # Two-pass download optimization
        "use_two_pass_download": os.getenv("USE_TWO_PASS_DOWNLOAD", "true").lower() == "true",
        "preview_max_height": int(os.getenv("PREVIEW_MAX_HEIGHT", "360")),
        "clip_download_format": os.getenv("CLIP_DOWNLOAD_FORMAT",
                                           "bestvideo[height<=1080]+bestaudio/best"),
        # PHASE 1 OPTIMIZATIONS: Parallel processing
        "max_concurrent_needs": int(os.getenv("MAX_CONCURRENT_NEEDS", "5")),
        # PHASE 2 PERFORMANCE: Granular parallelization
        "parallel_downloads": int(os.getenv("PARALLEL_DOWNLOADS", "3")),
        "parallel_extractions": int(os.getenv("PARALLEL_EXTRACTIONS", "2")),
        "parallel_ai_calls": int(os.getenv("PARALLEL_AI_CALLS", "5")),
        # S2 IMPROVEMENT: AI Response Caching
        # Caches Gemini API responses keyed by content hash for 100% savings on re-runs
        "ai_cache_enabled": os.getenv("AI_CACHE_ENABLED", "true").lower() == "true",
        "ai_cache_ttl_days": int(os.getenv("AI_CACHE_TTL_DAYS", "30")),
        "ai_cache_max_size_gb": float(os.getenv("AI_CACHE_MAX_SIZE_GB", "1.0")),
        "ai_cache_dir": resolve_path(os.getenv("AI_CACHE_DIR"), ".cache/ai_responses"),
        # Competitive analysis: Compare multiple videos per B-roll need
        "competitive_analysis_enabled": os.getenv("COMPETITIVE_ANALYSIS_ENABLED", "true").lower() == "true",
        "previews_per_need": int(os.getenv("PREVIEWS_PER_NEED", "2")),
        "clips_per_need_target": int(os.getenv("CLIPS_PER_NEED_TARGET", "1")),
        # PHASE 2 OPTIMIZATIONS: YouTube stability & rate limiting
        "ytdlp_rate_limit": int(os.getenv("YTDLP_RATE_LIMIT", "2000000")),  # 2MB/s default
        "ytdlp_sleep_interval": int(os.getenv("YTDLP_SLEEP_INTERVAL", "2")),
        "ytdlp_max_sleep_interval": int(os.getenv("YTDLP_MAX_SLEEP_INTERVAL", "5")),
        "ytdlp_retries": int(os.getenv("YTDLP_RETRIES", "5")),
        "ytdlp_cookies_file": os.getenv("YTDLP_COOKIES_FILE"),  # Optional cookie file path
        # PHASE 3 FEATURES: Cost tracking
        "budget_limit_usd": float(os.getenv("BUDGET_LIMIT_USD", "0.0")),
        # Q4 IMPROVEMENT: Context-aware evaluation settings
        # Seconds of transcript context to extract around each B-roll timestamp
        "evaluation_context_seconds": float(os.getenv("EVALUATION_CONTEXT_SECONDS", "30.0")),
        # S3 IMPROVEMENT: faster-whisper configuration
        "whisper_device": os.getenv("WHISPER_DEVICE", "auto"),  # "auto", "cpu", "cuda"
        "whisper_compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "auto"),  # "auto", "int8", "float16", "float32"
        # S5 IMPROVEMENT: Video pre-filtering settings
        "min_view_count": int(os.getenv("MIN_VIEW_COUNT", "0")),  # Minimum views for a video
        "max_prefilter_duration": int(os.getenv("MAX_PREFILTER_DURATION", "600")),  # Max video duration in seconds
        "blocked_title_keywords": os.getenv("BLOCKED_TITLE_KEYWORDS", "").split(",") if os.getenv("BLOCKED_TITLE_KEYWORDS") else [],
        # Semantic matching settings
        "semantic_match_threshold": float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.9")),
        "semantic_verification_enabled": os.getenv("SEMANTIC_VERIFICATION_ENABLED", "true").lower() == "true",
        "reject_below_threshold": os.getenv("REJECT_BELOW_THRESHOLD", "true").lower() == "true",
        "min_required_elements_match": float(os.getenv("MIN_REQUIRED_ELEMENTS_MATCH", "0.8")),
        # IMAGE ACQUISITION SETTINGS
        # Parallel image acquisition - one image per N seconds of video
        "image_acquisition_enabled": os.getenv("IMAGE_ACQUISITION_ENABLED", "true").lower() == "true",
        "image_interval_seconds": float(os.getenv("IMAGE_INTERVAL_SECONDS", "5.0")),
        "image_sources": [s.strip() for s in os.getenv("IMAGE_SOURCES", "pexels,pixabay,google").split(",") if s.strip()],
        "parallel_image_downloads": int(os.getenv("PARALLEL_IMAGE_DOWNLOADS", "10")),
        # SerpAPI key for Google Images search
        "serpapi_key": os.getenv("SERPAPI_KEY"),
        # Feature 1: Style/Mood Detection
        # Analyzes source video to detect visual style, topic, and audience
        "style_detection_enabled": os.getenv("STYLE_DETECTION_ENABLED", "true").lower() == "true",
        # Feature 2: Context Window
        # Uses ±N seconds of transcript around each B-roll timestamp
        "context_window_seconds": float(os.getenv("CONTEXT_WINDOW_SECONDS", "10.0")),
        # Feature 3: Feedback Loop
        # Learns from user rejections to improve future selections
        "feedback_enabled": os.getenv("FEEDBACK_ENABLED", "true").lower() == "true",
        "feedback_dir": resolve_path(os.getenv("FEEDBACK_DIR"), ".stockpile"),
        # Source preferences (soft bias, not hard filter)
        # For video clips: "youtube", "pexels", "pixabay"
        # For images: "google", "pexels", "pixabay"
        "video_preferred_source": os.getenv("VIDEO_PREFERRED_SOURCE", "youtube"),
        "image_preferred_source": os.getenv("IMAGE_PREFERRED_SOURCE", "google"),
        # TTS (Text-to-Speech) settings
        # URL of Chatterbox-TTS-Server running on Colab
        "tts_server_url": os.getenv("TTS_SERVER_URL", ""),
        # AI Image Generation settings
        "fal_api_key": os.getenv("FAL_API_KEY", ""),
        "runpod_api_key": os.getenv("RUNPOD_API_KEY", ""),
        "runpod_qwen3_endpoint_id": os.getenv("RUNPOD_QWEN3_ENDPOINT_ID", ""),
        "runpod_chatterbox_ext_endpoint_id": os.getenv("RUNPOD_CHATTERBOX_EXT_ENDPOINT_ID", ""),
        "runware_api_key": os.getenv("RUNWARE_API_KEY", ""),
        "default_image_gen_model": os.getenv("DEFAULT_IMAGE_GEN_MODEL", "runware-flux-klein-4b"),
    }

    return config


logger = logging.getLogger(__name__)


def validate_config_with_warnings(config: dict) -> tuple[list[str], list[str]]:
    """Validate configuration and return tuple of (errors, warnings).

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Tuple of (errors, warnings) where:
        - errors: List of critical configuration errors that prevent operation
        - warnings: List of non-critical warnings about missing optional features
    """
    errors = []
    warnings = []

    # Check required API key
    if not config.get("gemini_api_key"):
        errors.append("GEMINI_API_KEY is required")

    # Check input sources (at least one required)
    has_local_input = bool(config.get("local_input_folder"))
    has_drive_input = bool(config.get("google_drive_input_folder_id"))

    if not (has_local_input or has_drive_input):
        errors.append(
            "At least one input source required: LOCAL_INPUT_FOLDER or GOOGLE_DRIVE_INPUT_FOLDER_ID"
        )

    # Check output destinations (at least one required)
    has_local_output = bool(config.get("local_output_folder"))
    has_drive_output = bool(config.get("google_drive_output_folder_id"))

    if not (has_local_output or has_drive_output):
        errors.append(
            "At least one output destination required: LOCAL_OUTPUT_FOLDER or GOOGLE_DRIVE_OUTPUT_FOLDER_ID"
        )

    # Validate local paths exist
    if has_local_input:
        input_path = Path(config["local_input_folder"])
        try:
            input_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create local input folder: {e}")

    if has_local_output:
        output_path = Path(config["local_output_folder"])
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create local output folder: {e}")

    # Validate Google Drive credentials if using Google Drive
    if has_drive_output or has_drive_input:
        if not config.get("google_client_id") or not config.get("google_client_secret"):
            errors.append(
                "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET required for Google Drive integration"
            )

    # ==========================================================================
    # Numeric range validation
    # ==========================================================================

    # clips_per_minute must be > 0 (controls B-roll density)
    clips_per_minute = config.get("clips_per_minute", 2.0)
    if clips_per_minute <= 0:
        errors.append(f"CLIPS_PER_MINUTE must be > 0 (got {clips_per_minute})")

    # min_clip_duration must be > 0
    min_clip_duration = config.get("min_clip_duration", 4.0)
    if min_clip_duration <= 0:
        errors.append(f"MIN_CLIP_DURATION must be > 0 (got {min_clip_duration})")

    # max_clip_duration must be > min_clip_duration
    max_clip_duration = config.get("max_clip_duration", 15.0)
    if max_clip_duration <= min_clip_duration:
        errors.append(
            f"MAX_CLIP_DURATION ({max_clip_duration}) must be > MIN_CLIP_DURATION ({min_clip_duration})"
        )

    # max_clips_per_video must be > 0
    max_clips_per_video = config.get("max_clips_per_video", 3)
    if max_clips_per_video <= 0:
        errors.append(f"MAX_CLIPS_PER_VIDEO must be > 0 (got {max_clips_per_video})")

    # max_videos_per_phrase must be > 0
    max_videos_per_phrase = config.get("max_videos_per_phrase", 3)
    if max_videos_per_phrase <= 0:
        errors.append(f"MAX_VIDEOS_PER_PHRASE must be > 0 (got {max_videos_per_phrase})")

    # parallel_downloads must be > 0
    parallel_downloads = config.get("parallel_downloads", 3)
    if parallel_downloads <= 0:
        errors.append(f"PARALLEL_DOWNLOADS must be > 0 (got {parallel_downloads})")

    # parallel_extractions must be > 0
    parallel_extractions = config.get("parallel_extractions", 2)
    if parallel_extractions <= 0:
        errors.append(f"PARALLEL_EXTRACTIONS must be > 0 (got {parallel_extractions})")

    # parallel_ai_calls must be > 0
    parallel_ai_calls = config.get("parallel_ai_calls", 5)
    if parallel_ai_calls <= 0:
        errors.append(f"PARALLEL_AI_CALLS must be > 0 (got {parallel_ai_calls})")

    # ==========================================================================
    # Warnings for missing optional-but-recommended API keys
    # ==========================================================================

    # YOUTUBE_API_KEY - recommended for outlier finder feature
    if not config.get("youtube_api_key"):
        warnings.append(
            "YOUTUBE_API_KEY not set - outlier finder feature will have limited functionality"
        )

    # RUNWARE_API_KEY - recommended for cheapest image generation
    if not config.get("runware_api_key"):
        warnings.append(
            "RUNWARE_API_KEY not set - Runware image generation (Flux Klein, Z-Image) will be unavailable"
        )

    # RUNPOD_API_KEY - recommended for TTS and Nano Banana Pro
    if not config.get("runpod_api_key"):
        warnings.append(
            "RUNPOD_API_KEY not set - RunPod TTS and Nano Banana Pro will be unavailable"
        )

    # Log warnings (they don't prevent operation but user should be aware)
    for warning in warnings:
        logger.warning(warning)

    return errors, warnings


def validate_config(config: dict) -> list[str]:
    """Validate configuration and return list of errors.

    This is the backward-compatible version that returns only errors.
    For errors and warnings, use validate_config_with_warnings() instead.

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        List of critical configuration errors
    """
    errors, _ = validate_config_with_warnings(config)
    return errors


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Uses structlog for structured logging. Falls back to basic logging
    if structlog is not installed.
    """
    try:
        from utils.logging import setup_logging as _setup_structlog
        _setup_structlog(log_level=log_level, json_output=False)
    except ImportError:
        # Fallback to basic logging if structlog not installed
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )


def get_supported_video_formats() -> list[str]:
    """Return list of supported video file extensions."""
    return [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]


def get_supported_audio_formats() -> list[str]:
    """Return list of supported audio file extensions."""
    return [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"]
