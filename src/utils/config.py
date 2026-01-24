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
        # AI Response Caching
        "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        "cache_ttl_days": int(os.getenv("CACHE_TTL_DAYS", "30")),
        "cache_max_size_gb": float(os.getenv("CACHE_MAX_SIZE_GB", "1.0")),
        "cache_dir": resolve_path(os.getenv("CACHE_DIR"), ".cache/ai_responses"),
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
    }

    return config


def validate_config(config: dict) -> list[str]:
    """Validate configuration and return list of errors."""
    errors = []

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

    # YouTube search is handled by yt-dlp directly, no API key validation needed

    return errors


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration with Rich for beautiful terminal output."""
    # Clear any existing handlers
    logging.root.handlers.clear()

    # Rich handler for beautiful console output
    rich_handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,  # Disable markup to avoid conflicts
    )

    # File handler for plain text logging (always in src directory)
    log_file = PROJECT_ROOT / "src" / "broll_processor.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[rich_handler, file_handler],
        format="%(message)s",
    )

    # Suppress noisy third-party loggers
    noisy_loggers = [
        "httpx",
        "google_genai",
        "google_genai.models",
        "googleapiclient.discovery_cache",
        "google_auth_oauthlib.flow",
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_supported_video_formats() -> list[str]:
    """Return list of supported video file extensions."""
    return [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]


def get_supported_audio_formats() -> list[str]:
    """Return list of supported audio file extensions."""
    return [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"]
