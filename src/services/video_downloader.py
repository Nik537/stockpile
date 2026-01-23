"""Video download service using yt-dlp."""

import logging
import re
import random
from pathlib import Path
from typing import List, Dict, Optional
import yt_dlp

from models.video import ScoredVideo
from utils.retry import retry_download, NetworkError, TemporaryServiceError, YouTubeRateLimitError
from utils.config import load_config

logger = logging.getLogger(__name__)

# User agent rotation to avoid rate limiting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# RELIABILITY IMPROVEMENT: Format fallback strategies
# Try multiple formats in sequence if initial format fails
CLIP_FORMAT_FALLBACKS = [
    "bestvideo[height<=1080]+bestaudio/best",  # Preferred: 1080p with audio merge
    "best[height<=1080]",  # Fallback 1: Pre-merged 1080p
    "best"  # Fallback 2: Any available format
]

PREVIEW_FORMAT_FALLBACKS = [
    "worst[height<=360]/worst",  # Preferred: Lowest quality for speed
    "worst",  # Fallback 1: Absolute worst quality
    "best[height<=360]"  # Fallback 2: 360p if worst unavailable
]


def get_random_user_agent() -> str:
    """Get a random user agent to avoid detection."""
    return random.choice(USER_AGENTS)

logging.getLogger("yt_dlp").setLevel(logging.CRITICAL)
logging.getLogger("yt_dlp.extractor").setLevel(logging.CRITICAL)
logging.getLogger("yt_dlp.downloader").setLevel(logging.CRITICAL)
logging.getLogger("yt_dlp.postprocessor").setLevel(logging.CRITICAL)


class VideoDownloader:
    """Service for downloading videos using yt-dlp with custom options."""

    def __init__(self, output_dir: str):
        """Initialize video downloader with output directory.

        Args:
            output_dir: Base directory for downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized video downloader with output dir: {self.output_dir}")

    @retry_download(max_retries=3, base_delay=2.0)
    def download_videos(self, videos: List[ScoredVideo], phrase: str) -> List[str]:
        """Download videos using yt-dlp with custom options.

        Args:
            videos: List of scored videos to download
            phrase: Search phrase for organizing downloads

        Returns:
            List of paths to downloaded files
        """
        if not videos:
            logger.info("No videos to download")
            return []

        # Create phrase-specific directory
        phrase_dir = self.output_dir / self.sanitize_filename(phrase)
        phrase_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {len(videos)} videos for phrase: '{phrase}'")

        downloaded_files = []

        for video in videos:
            try:
                downloaded_file = self._download_single_video(video, phrase_dir)
                if downloaded_file:
                    downloaded_files.append(downloaded_file)
                    logger.info(f"Downloaded: {Path(downloaded_file).name}")
                else:
                    logger.warning(f"Failed to download video: {video.video_id}")

            except Exception as e:
                logger.error(f"Error downloading video {video.video_id}: {e}")
                continue

        logger.info(
            f"Download completed: {len(downloaded_files)} videos for phrase: '{phrase}'"
        )
        return downloaded_files

    def download_single_video_to_folder(
        self,
        video: ScoredVideo,
        target_folder: str,
        drive_service=None,
        drive_folder_id: str | None = None,
    ) -> Optional[str]:
        """Download a single video to a specific target folder.

        Args:
            video: ScoredVideo to download
            target_folder: Exact target folder path
            drive_service: Optional DriveService instance for immediate upload
            drive_folder_id: Drive folder ID to upload to

        Returns:
            Path to downloaded file or None if failed
        """
        target_path = Path(target_folder)
        target_path.mkdir(parents=True, exist_ok=True)

        try:
            downloaded_file = self._download_single_video(video, target_path)
            if downloaded_file:
                logger.info(f"Downloaded: {Path(downloaded_file).name}")

                # Upload to Drive if configured
                if drive_service and drive_folder_id:
                    try:
                        drive_service.upload_file(downloaded_file, drive_folder_id)
                        logger.info(f"Uploaded to Drive: {Path(downloaded_file).name}")
                    except Exception as e:
                        logger.error(f"Drive upload failed for {Path(downloaded_file).name}: {e}")

                return downloaded_file
            else:
                logger.warning(f"Failed to download video: {video.video_id}")
                return None
        except Exception as e:
            logger.error(f"Error downloading video {video.video_id}: {e}")
            return None

    def download_videos_to_folder(
        self,
        videos: List[ScoredVideo],
        phrase: str,
        target_folder: str,
        drive_service=None,
        drive_folder_id: str | None = None,
    ) -> List[str]:
        """Download videos directly to a specific target folder with optional Drive upload.

        Args:
            videos: List of scored videos to download
            phrase: Search phrase (for logging)
            target_folder: Exact target folder path
            drive_service: Optional DriveService instance for immediate upload
            drive_folder_id: Drive folder ID to upload to

        Returns:
            List of paths to downloaded files
        """
        if not videos:
            logger.info(f"No videos to download for phrase: {phrase}")
            return []

        target_path = Path(target_folder)
        target_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloading {len(videos)} videos for phrase '{phrase}' to: {target_path}"
        )

        downloaded_files = []
        for video in videos:
            try:
                downloaded_file = self._download_single_video(video, target_path)
                if downloaded_file:
                    downloaded_files.append(downloaded_file)
                    logger.info(f"Downloaded: {Path(downloaded_file).name}")

                    # Start upload to Drive immediately (non-blocking)
                    if drive_service and drive_folder_id:
                        from concurrent.futures import ThreadPoolExecutor
                        import threading

                        def upload_task():
                            try:
                                drive_service.upload_file(downloaded_file, drive_folder_id)
                                logger.info(f"Uploaded to Drive: {Path(downloaded_file).name}")
                            except Exception as e:
                                logger.error(f"Drive upload failed for {Path(downloaded_file).name}: {e}")

                        # Use dedicated thread pool for uploads
                        if not hasattr(self, '_upload_executor'):
                            self._upload_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="upload")
                        self._upload_executor.submit(upload_task)
                else:
                    logger.warning(f"Failed to download video: {video.video_id}")
            except Exception as e:
                logger.error(f"Error downloading video {video.video_id}: {e}")
                continue

        logger.info(
            f"Successfully downloaded {len(downloaded_files)} videos for phrase: '{phrase}'"
        )
        return downloaded_files

    def _download_single_video(
        self, video: ScoredVideo, output_dir: Path
    ) -> Optional[str]:
        """Download a single video with yt-dlp.

        Args:
            video: ScoredVideo object to download
            output_dir: Directory to save the video

        Returns:
            Path to downloaded file or None if failed
        """

        # Get config for file size limit and rate limiting
        config = load_config()
        max_size = config.get("max_video_size_mb", 100) * 1024 * 1024

        # PHASE 2: Rate limiting and stability options
        rate_limit = config.get("ytdlp_rate_limit", 2000000)
        sleep_interval = config.get("ytdlp_sleep_interval", 2)
        max_sleep_interval = config.get("ytdlp_max_sleep_interval", 5)
        retries = config.get("ytdlp_retries", 5)
        cookies_file = config.get("ytdlp_cookies_file")

        # Configure yt-dlp options
        ydl_opts = {
            # Output template with score prefix for easy identification
            "outtmpl": str(output_dir / f"score{video.score:02d}_%(title)s.%(ext)s"),
            # Metadata options
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "writethumbnail": False,
            # Error handling
            "ignoreerrors": True,
            "no_warnings": True,  # Suppress warnings
            "retries": retries,
            # PHASE 2: Rate limiting to avoid 403 errors
            "ratelimit": rate_limit,  # 2MB/s default - prevents throttling
            "sleep_interval": sleep_interval,  # 2 seconds between downloads
            "max_sleep_interval": max_sleep_interval,  # Max 5 seconds
            # PHASE 2: User agent rotation
            "http_headers": {
                "User-Agent": get_random_user_agent(),
            },
            # Audio options (keep audio for B-roll)
            "extractaudio": False,
            # Post-processing with ffmpeg suppression
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            # Suppress all ffmpeg output completely
            "postprocessor_args": {
                "FFmpeg": ["-v", "quiet", "-nostats", "-loglevel", "error"],
                "Merger+ffmpeg": ["-v", "quiet", "-nostats", "-loglevel", "error"],
                "VideoConvertor+ffmpeg": [
                    "-v",
                    "quiet",
                    "-nostats",
                    "-loglevel",
                    "error",
                ],
                "VideoRemuxer+ffmpeg": [
                    "-v",
                    "quiet",
                    "-nostats",
                    "-loglevel",
                    "error",
                ],
                "ExtractAudio+ffmpeg": [
                    "-v",
                    "quiet",
                    "-nostats",
                    "-loglevel",
                    "error",
                ],
            },
            "max_filesize": max_size,  # Use configured max size
            # Logging options - complete suppression
            "quiet": True,  # Suppress most output
            "no_progress": True,  # Disable progress bar
        }

        # PHASE 2: Add cookie support if configured
        if cookies_file and Path(cookies_file).exists():
            ydl_opts["cookiefile"] = cookies_file
            logger.debug(f"Using cookies from: {cookies_file}")

        try:
            # Get list of files before download
            files_before = set(output_dir.glob("*"))

            # Download the video using the full options
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video.video_result.url])

                # Find newly created files with the score prefix
                files_after = set(output_dir.glob("*"))
                new_files = files_after - files_before

                score_prefix = f"score{video.score:02d}_"
                for file_path in new_files:
                    if file_path.is_file() and file_path.name.startswith(score_prefix):
                        return str(file_path)

                logger.warning(
                    f"No downloaded file found with score prefix: {score_prefix}"
                )
                return None

        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp download error for {video.video_id}: {e}")
            # Convert to retryable errors where appropriate
            error_msg = str(e).lower()
            # PHASE 2: Detect YouTube rate limiting (HTTP 403)
            if "403" in error_msg or "forbidden" in error_msg:
                raise YouTubeRateLimitError(f"YouTube rate limit (403) for {video.video_id}: {e}")
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error downloading {video.video_id}: {e}")
            elif "unavailable" in error_msg or "private" in error_msg:
                raise TemporaryServiceError(f"Video unavailable {video.video_id}: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error downloading {video.video_id}: {e}")
            raise

    def get_video_info(self, url: str) -> Optional[Dict]:
        """Extract video information without downloading.

        Args:
            url: Video URL

        Returns:
            Video information dictionary or None if extraction fails
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            # PHASE 2: User agent rotation
            "http_headers": {
                "User-Agent": get_random_user_agent(),
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                result = ydl.sanitize_info(info) if info else None
                return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Failed to extract info for {url}: {e}")
            return None

    def download_preview(
        self,
        video: ScoredVideo,
        output_folder: str,
        max_height: int = 360,
    ) -> Optional[str]:
        """Download low-quality preview for AI analysis.

        Downloads worst available quality up to max_height (default 360p).
        Much faster than full quality - typically 5-10MB vs 50-100MB.

        Args:
            video: Video to download
            output_folder: Where to save preview
            max_height: Maximum video height (default: 360)

        Returns:
            Path to downloaded preview file, or None if failed
        """
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # PHASE 2: Load rate limiting config
        config = load_config()
        rate_limit = config.get("ytdlp_rate_limit", 2000000)
        sleep_interval = config.get("ytdlp_sleep_interval", 2)
        max_sleep_interval = config.get("ytdlp_max_sleep_interval", 5)
        retries = config.get("ytdlp_retries", 5)
        cookies_file = config.get("ytdlp_cookies_file")

        # Use "worst" format with height constraint for fastest download
        format_selector = f"worst[height<={max_height}]/worst"

        ydl_opts = {
            "format": format_selector,
            "outtmpl": str(output_dir / f"preview_{video.video_id}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            "quiet": True,
            "no_progress": True,
            "retries": retries,
            "ignoreerrors": True,
            # PHASE 2: Rate limiting and user agent
            "ratelimit": rate_limit,
            "sleep_interval": sleep_interval,
            "max_sleep_interval": max_sleep_interval,
            "http_headers": {
                "User-Agent": get_random_user_agent(),
            },
        }

        # PHASE 2: Add cookie support if configured
        if cookies_file and Path(cookies_file).exists():
            ydl_opts["cookiefile"] = cookies_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video.video_result.url])

            # Find downloaded preview file
            preview_files = list(output_dir.glob(f"preview_{video.video_id}.*"))
            if preview_files:
                preview_path = str(preview_files[0])
                # RELIABILITY IMPROVEMENT: Validate file size before returning
                if self.validate_downloaded_file(preview_path):
                    logger.info(f"Downloaded preview: {preview_files[0].name}")
                    return preview_path
                else:
                    logger.warning(f"Preview file invalid (empty or too small): {video.video_id}")
                    return None
            else:
                logger.warning(f"Preview file not found after download: {video.video_id}")
                return None

        except Exception as e:
            logger.error(f"Preview download failed for {video.video_id}: {e}")
            # RELIABILITY IMPROVEMENT: Check for SABR errors
            if self.detect_sabr_error(str(e)):
                logger.warning(f"SABR streaming error detected for {video.video_id}")
            return None

    def download_clip_sections(
        self,
        video: ScoredVideo,
        segments: List,
        output_folder: str,
        format_selector: str = "bestvideo[height<=1080]+bestaudio/best",
    ) -> List[str]:
        """Download specific time ranges in high quality.

        Uses yt-dlp --download-sections to download only the needed parts.
        This is much faster than downloading full video + FFmpeg extraction.

        Args:
            video: Video to download from
            segments: List of clip segments with start/end times
            output_folder: Where to save clips
            format_selector: Quality format (default: 1080p max)

        Returns:
            List of successfully downloaded clip paths
        """
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # PHASE 2: Load rate limiting config once
        config = load_config()
        rate_limit = config.get("ytdlp_rate_limit", 2000000)
        sleep_interval = config.get("ytdlp_sleep_interval", 2)
        max_sleep_interval = config.get("ytdlp_max_sleep_interval", 5)
        retries = config.get("ytdlp_retries", 5)
        cookies_file = config.get("ytdlp_cookies_file")

        downloaded_clips = []

        # RELIABILITY IMPROVEMENT: Use format fallback if custom format fails
        format_list = [format_selector] if format_selector not in CLIP_FORMAT_FALLBACKS else CLIP_FORMAT_FALLBACKS

        for idx, segment in enumerate(segments, 1):
            clip_basename = (
                f"clip{idx}_{segment.start_time:.1f}s-{segment.end_time:.1f}s_{video.video_id}"
            )

            # Capture segment times in the closure (avoid late binding bug)
            start_time = segment.start_time
            end_time = segment.end_time

            # Define download_ranges callback with correct signature: (info_dict, ydl)
            def make_download_ranges(start, end):
                def download_ranges_func(info_dict, ydl):
                    return [{"start_time": start, "end_time": end}]
                return download_ranges_func

            # RELIABILITY IMPROVEMENT: Try multiple formats on failure
            clip_downloaded = False
            last_error = None

            for format_idx, fmt in enumerate(format_list):
                if format_idx > 0:
                    logger.info(f"Trying fallback format {format_idx}: {fmt}")

                ydl_opts = {
                    "format": fmt,
                    "download_ranges": make_download_ranges(start_time, end_time),
                    "force_keyframes_at_cuts": True,  # Required for precise time range cuts
                    "outtmpl": str(output_dir / f"{clip_basename}.%(ext)s"),
                    "postprocessors": [
                        {
                            "key": "FFmpegVideoConvertor",
                            "preferedformat": "mp4",
                        }
                    ],
                    "quiet": True,
                    "no_progress": True,
                    "retries": retries,
                    "ignoreerrors": True,
                    # PHASE 2: Rate limiting and user agent
                    "ratelimit": rate_limit,
                    "sleep_interval": sleep_interval,
                    "max_sleep_interval": max_sleep_interval,
                    "http_headers": {
                        "User-Agent": get_random_user_agent(),
                    },
                }

                # PHASE 2: Add cookie support if configured
                if cookies_file and Path(cookies_file).exists():
                    ydl_opts["cookiefile"] = cookies_file

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video.video_result.url])

                    # Find the downloaded clip (extension may vary)
                    clip_files = list(output_dir.glob(f"{clip_basename}.*"))
                    if clip_files:
                        clip_path = str(clip_files[0])
                        # RELIABILITY IMPROVEMENT: Validate file before accepting
                        if self.validate_downloaded_file(clip_path):
                            downloaded_clips.append(clip_path)
                            logger.info(f"Downloaded clip: {Path(clip_path).name}")
                            clip_downloaded = True
                            break  # Success, stop trying formats
                        else:
                            logger.warning(f"Clip file invalid, trying next format")
                            Path(clip_path).unlink()  # Delete invalid file
                    else:
                        logger.warning(f"Clip not found after download: {clip_basename}")

                except Exception as e:
                    last_error = e
                    # RELIABILITY IMPROVEMENT: Check for SABR errors
                    if self.detect_sabr_error(str(e)):
                        logger.warning(f"SABR streaming error detected, skipping remaining formats")
                        break  # SABR errors won't be fixed by format changes

                    # Try next format if available
                    if format_idx < len(format_list) - 1:
                        logger.info(f"Format {fmt} failed, trying next format")
                        continue
                    else:
                        logger.error(f"Failed to download clip {idx} with all formats: {e}")

            if not clip_downloaded and last_error:
                logger.error(f"All formats failed for clip {idx}: {last_error}")

        return downloaded_clips

    def test_clip_extraction_on_preview(
        self,
        preview_path: str,
        start_time: float,
        end_time: float,
        video_id: str,
    ) -> bool:
        """RELIABILITY IMPROVEMENT: Test if clip extraction will work before downloading high-res.

        Downloads a small test segment from the preview to verify FFmpeg can process it.
        Prevents wasting bandwidth on high-res downloads that will fail.

        Args:
            preview_path: Path to preview file
            start_time: Clip start time in seconds
            end_time: Clip end time in seconds
            video_id: Video ID for logging

        Returns:
            True if extraction succeeded, False if it will likely fail
        """
        try:
            import subprocess

            # Create a temporary test clip (don't save it)
            test_output = Path(preview_path).parent / f"_test_extraction_{video_id}.mp4"

            # Try to extract the segment using FFmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", preview_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "copy",  # Fast copy without re-encoding
                "-c:a", "copy",
                "-y",  # Overwrite if exists
                "-loglevel", "error",
                str(test_output)
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check if extraction succeeded
            if result.returncode == 0 and test_output.exists():
                file_valid = self.validate_downloaded_file(str(test_output))
                test_output.unlink()  # Clean up test file
                if file_valid:
                    logger.debug(f"Preview extraction test passed for {video_id}")
                    return True
                else:
                    logger.warning(f"Preview extraction created empty file for {video_id}")
                    return False
            else:
                logger.warning(f"Preview extraction test failed for {video_id}: {result.stderr}")
                if test_output.exists():
                    test_output.unlink()
                return False

        except Exception as e:
            logger.warning(f"Could not test preview extraction for {video_id}: {e}")
            return True  # Don't block if test fails, proceed with download

    def _supports_section_downloads(self) -> bool:
        """Check if yt-dlp version supports --download-sections.

        Feature was added in yt-dlp 2023.03.04.

        Returns:
            True if supported, False otherwise
        """
        try:
            import yt_dlp.version

            version = yt_dlp.version.__version__

            # Parse version string (format: YYYY.MM.DD or YYYY.MM.DD.post0)
            parts = version.split(".")
            if len(parts) >= 3:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2].split("post")[0])

                # Check if >= 2023.03.04
                if year > 2023:
                    return True
                elif year == 2023:
                    if month > 3:
                        return True
                    elif month == 3 and day >= 4:
                        return True

            logger.warning(f"yt-dlp version {version} does not support --download-sections")
            return False

        except Exception as e:
            logger.warning(f"Could not determine yt-dlp version: {e}")
            return False  # Assume not supported if can't determine

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        filename = re.sub(r"\s+", "_", filename)
        filename = filename.strip("._")
        filename = filename[:50] if filename else "unnamed"
        return filename

    @staticmethod
    def validate_downloaded_file(file_path: str, min_size_kb: int = 1) -> bool:
        """RELIABILITY IMPROVEMENT: Validate downloaded file exists and meets minimum size.

        Prevents processing empty files that yt-dlp reports as successful.

        Args:
            file_path: Path to file to validate
            min_size_kb: Minimum file size in KB (default: 1KB)

        Returns:
            True if file is valid, False otherwise
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False

        size_kb = path.stat().st_size / 1024
        if size_kb < min_size_kb:
            logger.warning(f"File too small ({size_kb:.2f} KB): {file_path}")
            return False

        return True

    @staticmethod
    def detect_sabr_error(error_message: str) -> bool:
        """RELIABILITY IMPROVEMENT: Detect if error is a YouTube SABR streaming issue.

        SABR (Server-Assisted Bitrate Reduction) errors indicate YouTube is blocking
        the download attempt with format restrictions.

        Args:
            error_message: Error message to check

        Returns:
            True if SABR error detected, False otherwise
        """
        sabr_indicators = [
            "sabr",
            "forcing sabr",
            "web client https formats",
            "formats have been skipped"
        ]
        error_lower = str(error_message).lower()
        return any(indicator in error_lower for indicator in sabr_indicators)

    def cleanup_failed_downloads(self, phrase_dir: Path) -> None:
        """Clean up any partial or failed download files.

        Args:
            phrase_dir: Directory to clean up
        """
        try:
            patterns = ["*.part", "*.tmp", "*.ytdl", "*.f*"]

            for pattern in patterns:
                for file_path in phrase_dir.glob(pattern):
                    try:
                        file_path.unlink()
                        logger.debug(f"Cleaned up partial download: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not clean up {file_path}: {e}")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def cleanup_intermediate_files(self, phrase_dir: Path) -> None:
        """Clean up intermediate files (previews, webm, failed downloads) to save disk space.

        Args:
            phrase_dir: Directory to clean up
        """
        try:
            # Patterns for intermediate files that should be deleted
            patterns = [
                "preview_*.mp4",  # Preview files from two-pass download
                "preview_*.webm",  # Preview webm files
                "*.webm",  # Intermediate webm files (before mp4 conversion)
                "*.part",  # Partial downloads
                "*.tmp",   # Temporary files
                "*.ytdl",  # yt-dlp metadata
                "*.f*",    # Fragment files
            ]

            cleaned_count = 0
            cleaned_size = 0

            for pattern in patterns:
                for file_path in phrase_dir.glob(pattern):
                    try:
                        # Get size before deletion for logging
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        file_path.unlink()
                        cleaned_count += 1
                        cleaned_size += size_mb
                        logger.info(f"Cleaned up: {file_path.name} ({size_mb:.1f} MB)")
                    except Exception as e:
                        logger.warning(f"Could not clean up {file_path}: {e}")

            if cleaned_count > 0:
                logger.info(
                    f"Cleanup complete: {cleaned_count} files, {cleaned_size:.1f} MB freed"
                )

        except Exception as e:
            logger.warning(f"Error during intermediate file cleanup: {e}")

    def get_download_stats(self, phrase_dir: Path) -> Dict[str, int]:
        """Get statistics about downloaded files in a phrase directory.

        Args:
            phrase_dir: Directory to analyze

        Returns:
            Dictionary with download statistics
        """
        if not phrase_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}

        total_files = 0
        total_size = 0

        for file_path in phrase_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in [
                ".mp4",
                ".webm",
                ".mkv",
                ".avi",
            ]:
                total_files += 1
                total_size += file_path.stat().st_size

        return {
            "total_files": int(total_files),
            "total_size_mb": int(round(total_size / (1024 * 1024), 2)),
        }
