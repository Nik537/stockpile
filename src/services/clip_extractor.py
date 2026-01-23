"""Clip extraction service using Gemini video analysis and FFmpeg."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

from google.genai import Client, types

from models.clip import ClipSegment, ClipResult, VideoAnalysisResult
from utils.retry import retry_api_call, APIRateLimitError, NetworkError

logger = logging.getLogger(__name__)


class ClipExtractor:
    """Service for intelligent B-roll clip extraction using Gemini multimodal analysis."""

    # Configuration constants
    MIN_CLIP_DURATION = 4.0  # Minimum clip length in seconds
    MAX_CLIP_DURATION = 15.0  # Maximum clip length in seconds
    FILE_POLL_INTERVAL = 5.0  # Seconds between file status checks
    FILE_POLL_TIMEOUT = 300.0  # Max seconds to wait for file processing

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-flash-preview",
        min_clip_duration: float = 4.0,
        max_clip_duration: float = 15.0,
        max_clips_per_video: int = 3,
    ):
        """Initialize clip extractor.

        Args:
            api_key: Google GenAI API key
            model_name: Gemini model for video analysis
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds
            max_clips_per_video: Maximum number of clips to extract per video
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Client(api_key=api_key)

        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.max_clips_per_video = max_clips_per_video

        logger.info(f"Initialized ClipExtractor with model: {model_name}")

    def process_downloaded_video(
        self,
        video_path: str,
        search_phrase: str,
        video_id: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[ClipResult], bool]:
        """Process a downloaded video: analyze, extract clips, cleanup.

        Args:
            video_path: Path to downloaded video file
            search_phrase: The search phrase context for B-roll relevance
            video_id: YouTube video ID
            output_dir: Directory for extracted clips (defaults to same as source)

        Returns:
            Tuple of (list of ClipResult objects, whether original should be deleted)
        """
        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            return [], False

        output_path = Path(output_dir) if output_dir else video_file.parent

        # Step 1: Analyze video with Gemini
        logger.info(f"Analyzing video for B-roll segments: {video_file.name}")
        analysis = self.analyze_video(video_path, search_phrase, video_id)

        if not analysis.analysis_success or not analysis.segments:
            logger.warning(f"No segments found in video: {video_file.name}")
            return [], False  # Keep original if analysis failed

        # Step 2: Extract clips for each segment
        clips = []
        for i, segment in enumerate(analysis.segments[: self.max_clips_per_video]):
            clip_result = self.extract_clip(
                video_path=video_path,
                segment=segment,
                search_phrase=search_phrase,
                video_id=video_id,
                output_dir=str(output_path),
                clip_index=i + 1,
            )
            if clip_result.extraction_success:
                clips.append(clip_result)
                logger.info(f"Extracted clip {i+1}: {Path(clip_result.clip_path).name}")
            else:
                logger.warning(
                    f"Failed to extract clip {i+1}: {clip_result.error_message}"
                )

        # Determine if original should be deleted (only if we got at least one clip)
        should_delete_original = len(clips) > 0

        return clips, should_delete_original

    @retry_api_call(max_retries=3, base_delay=2.0)
    def analyze_video(
        self, video_path: str, search_phrase: str, video_id: str
    ) -> VideoAnalysisResult:
        """Analyze video using Gemini to identify relevant B-roll segments.

        Args:
            video_path: Path to video file
            search_phrase: Context for what B-roll content to look for
            video_id: YouTube video ID

        Returns:
            VideoAnalysisResult with identified segments
        """
        temp_file_path = None
        try:
            # Get video duration for validation
            video_duration = self._get_video_duration(video_path)

            # Upload video to Gemini File API
            # Handle Unicode filenames by copying to temp file with ASCII-safe name
            logger.info(f"Uploading video to Gemini File API: {Path(video_path).name}")
            upload_path = video_path
            try:
                # Test if path can be encoded as ASCII
                video_path.encode("ascii")
            except UnicodeEncodeError:
                # Contains non-ASCII characters, copy to temp file
                logger.info("Filename contains Unicode characters, using temp file for upload")
                suffix = Path(video_path).suffix
                temp_fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
                os.close(temp_fd)
                shutil.copy2(video_path, temp_file_path)
                upload_path = temp_file_path

            uploaded_file = self.client.files.upload(file=upload_path)

            # Wait for file processing to complete
            file_ready = self._wait_for_file_processing(uploaded_file.name)
            if not file_ready:
                return VideoAnalysisResult(
                    video_path=video_path,
                    video_id=video_id,
                    search_phrase=search_phrase,
                    analysis_success=False,
                    error_message="File processing timed out or failed",
                )

            # Construct analysis prompt
            prompt = self._build_analysis_prompt(search_phrase, video_duration)

            # Generate content with video analysis
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Low temperature for consistent timestamps
                ),
            )

            # Parse response into segments
            segments = self._parse_segments_response(response.text, video_duration)

            # Clean up uploaded file
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file: {e}")

            return VideoAnalysisResult(
                video_path=video_path,
                video_id=video_id,
                search_phrase=search_phrase,
                segments=segments,
                analysis_success=True,
                total_duration=video_duration,
            )

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit hit: {e}")
            elif "network" in str(e).lower():
                raise NetworkError(f"Network error: {e}")

            return VideoAnalysisResult(
                video_path=video_path,
                video_id=video_id,
                search_phrase=search_phrase,
                analysis_success=False,
                error_message=str(e),
            )

        finally:
            # Clean up temp file if we created one
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def _wait_for_file_processing(self, file_name: str) -> bool:
        """Wait for uploaded file to finish processing.

        Args:
            file_name: Name of the uploaded file

        Returns:
            True if file is ready, False if failed/timeout
        """
        start_time = time.time()

        while time.time() - start_time < self.FILE_POLL_TIMEOUT:
            file_info = self.client.files.get(name=file_name)

            if file_info.state.name == "ACTIVE":
                logger.info(f"File ready for analysis: {file_name}")
                return True
            elif file_info.state.name == "FAILED":
                logger.error(f"File processing failed: {file_name}")
                return False

            logger.debug(f"File still processing: {file_info.state.name}")
            time.sleep(self.FILE_POLL_INTERVAL)

        logger.error(f"File processing timeout: {file_name}")
        return False

    def _build_analysis_prompt(self, search_phrase: str, video_duration: float) -> str:
        """Build the prompt for video segment analysis."""
        return f"""You are a B-Roll Clip Analyzer. Your task is to identify the best segments in this video that would work as B-roll footage for the topic: "{search_phrase}"

GOAL:
Find {self.max_clips_per_video} or fewer segments that contain high-quality, visually relevant B-roll footage matching the search phrase.

VIDEO DURATION: {video_duration:.1f} seconds

REQUIREMENTS:
- Each segment should be {self.min_clip_duration}-{self.max_clip_duration} seconds long
- Focus on visually compelling shots (cinematic, clear subject, good lighting)
- Avoid segments with: talking heads, text overlays, logos, watermarks, transitions
- Prefer: establishing shots, action shots, close-ups of relevant subjects, atmospheric footage
- Segments must not overlap

OUTPUT FORMAT:
Return ONLY a JSON array with this exact structure:
[
  {{
    "start_time": 12.5,
    "end_time": 20.0,
    "relevance_score": 9,
    "description": "Wide shot of subject in natural setting"
  }}
]

RULES:
- start_time and end_time are in seconds (decimals allowed)
- relevance_score is 1-10 (only include segments scoring 7+)
- description should be brief (under 50 characters)
- Return empty array [] if no suitable segments found
- No markdown, no extra text, just the JSON array"""

    def _parse_segments_response(
        self, response_text: str, video_duration: float
    ) -> List[ClipSegment]:
        """Parse AI response into ClipSegment objects."""
        if not response_text:
            return []

        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            segments_data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse segments JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return []

        if not isinstance(segments_data, list):
            logger.warning("Segments response is not a list")
            return []

        segments = []
        for item in segments_data:
            try:
                start = float(item.get("start_time", 0))
                end = float(item.get("end_time", 0))
                score = int(item.get("relevance_score", 0))
                desc = str(item.get("description", ""))

                # Validate segment
                if start < 0 or end <= start:
                    continue
                if end > video_duration:
                    end = video_duration
                if score < 7:
                    continue

                duration = end - start
                if duration < self.min_clip_duration:
                    continue
                if duration > self.max_clip_duration:
                    # Trim to max duration, keeping start point
                    end = start + self.max_clip_duration

                segments.append(
                    ClipSegment(
                        start_time=start,
                        end_time=end,
                        relevance_score=score,
                        description=desc[:100],
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid segment data: {item}, error: {e}")
                continue

        # Sort by relevance score (highest first)
        segments.sort(key=lambda s: s.relevance_score, reverse=True)

        return segments

    def extract_clip(
        self,
        video_path: str,
        segment: ClipSegment,
        search_phrase: str,
        video_id: str,
        output_dir: str,
        clip_index: int = 1,
    ) -> ClipResult:
        """Extract a clip from video using FFmpeg.

        Args:
            video_path: Source video path
            segment: ClipSegment with timing info
            search_phrase: For naming context
            video_id: YouTube video ID
            output_dir: Where to save the clip
            clip_index: Index for naming multiple clips

        Returns:
            ClipResult with extraction details
        """
        source_path = Path(video_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate clip filename
        # Format: clip{index}_{start}s-{end}s_{original_name}
        start_str = f"{segment.start_time:.1f}".replace(".", "_")
        end_str = f"{segment.end_time:.1f}".replace(".", "_")
        clip_name = f"clip{clip_index}_{start_str}s-{end_str}s_{source_path.stem}.mp4"
        clip_path = output_path / clip_name

        try:
            duration = segment.end_time - segment.start_time

            # FFmpeg command for clip extraction
            # -ss before -i for fast seeking
            # -t for duration
            # Re-encoding for precise cuts
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-ss",
                str(segment.start_time),  # Seek to start
                "-i",
                str(video_path),
                "-t",
                str(duration),  # Duration
                "-c:v",
                "libx264",  # Re-encode video for precise cuts
                "-c:a",
                "aac",  # Re-encode audio
                "-preset",
                "fast",  # Fast encoding
                "-crf",
                "23",  # Good quality
                "-movflags",
                "+faststart",  # Web-optimized
                "-v",
                "quiet",
                str(clip_path),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120  # 2 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            if not clip_path.exists():
                raise RuntimeError("Clip file was not created")

            logger.info(f"Extracted clip: {clip_name} ({duration:.1f}s)")

            return ClipResult(
                source_video_path=video_path,
                clip_path=str(clip_path),
                segment=segment,
                search_phrase=search_phrase,
                source_video_id=video_id,
                extraction_success=True,
            )

        except subprocess.TimeoutExpired:
            error_msg = "FFmpeg extraction timed out"
            logger.error(error_msg)
            return ClipResult(
                source_video_path=video_path,
                clip_path="",
                segment=segment,
                search_phrase=search_phrase,
                source_video_id=video_id,
                extraction_success=False,
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = f"Clip extraction failed: {e}"
            logger.error(error_msg)
            return ClipResult(
                source_video_path=video_path,
                clip_path="",
                segment=segment,
                search_phrase=search_phrase,
                source_video_id=video_id,
                extraction_success=False,
                error_message=error_msg,
            )

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using FFprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return 600.0  # Default to 10 minutes if unknown

    def cleanup_original_video(self, video_path: str) -> bool:
        """Delete the original video file after successful clip extraction.

        SAFETY: This method has built-in protection to never delete files
        from 'input' directories, which contain the user's source videos.

        Args:
            video_path: Path to original video

        Returns:
            True if deletion succeeded
        """
        try:
            video_file = Path(video_path).resolve()

            # SAFETY: Never delete files from input directories
            path_parts = [p.lower() for p in video_file.parts]
            if "input" in path_parts:
                logger.warning(
                    f"BLOCKED deletion of input file (safety check): {video_file}"
                )
                return False

            if video_file.exists():
                video_file.unlink()
                logger.info(f"Deleted original video: {video_file.name}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete original video {video_path}: {e}")
            return False
