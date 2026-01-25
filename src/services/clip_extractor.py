"""Clip extraction service using Gemini video analysis, CLIP embeddings, and FFmpeg.

S2 IMPROVEMENT: AI Response Caching for video analysis
- Caches Gemini video analysis results by file hash + search phrase
- 100% cost savings on re-analyzing the same video with same search phrase
- File hash computed incrementally for large video files

CLIP Integration (Q1 improvement - SINGLE BIGGEST quality improvement):
- Uses OpenAI CLIP to compare video frames against text descriptions
- Provides visual relevance scoring independent of metadata
- Can be used as pre-filter or combined with Gemini scores

Scene Detection Integration (Q5 improvement):
- Uses PySceneDetect to identify natural scene boundaries
- Adjusts clip timestamps to avoid cutting mid-action or mid-sentence
- Produces more coherent B-roll clips with natural transitions
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from google.genai import Client, types

from models.broll_need import BRollNeed
from models.clip import ClipResult, ClipSegment, VideoAnalysisResult
from utils.cache import AIResponseCache, compute_content_hash, compute_file_hash
from utils.retry import APIRateLimitError, NetworkError, retry_api_call

if TYPE_CHECKING:
    from services.scene_extractor import SceneAwareExtractor
    from services.visual_matcher import VisualMatcher

logger = logging.getLogger(__name__)


# S2 IMPROVEMENT: Prompt version for cache invalidation
# v3: Updated to use original_context and required_elements from BRollNeed
CLIP_ANALYSIS_PROMPT_VERSION = "v3"


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
        # S2 IMPROVEMENT: AI response caching
        cache: Optional[AIResponseCache] = None,
        # Q1 IMPROVEMENT: CLIP visual matching configuration
        clip_enabled: bool = True,
        clip_model: str = "openai/clip-vit-base-patch32",
        clip_sample_rate: int = 1,
        clip_min_score_threshold: float = 0.3,
        clip_weight_in_score: float = 0.4,
        # Q5 IMPROVEMENT: Scene detection configuration
        scene_detection_enabled: bool = True,
        scene_detection_threshold: int = 27,
        prefer_complete_scenes: bool = True,
    ):
        """Initialize clip extractor.

        Args:
            api_key: Google GenAI API key
            model_name: Gemini model for video analysis
            min_clip_duration: Minimum clip duration in seconds
            max_clip_duration: Maximum clip duration in seconds
            max_clips_per_video: Maximum number of clips to extract per video
            cache: Optional AIResponseCache for caching video analysis results (S2)
            clip_enabled: Whether to use CLIP for visual matching
            clip_model: HuggingFace CLIP model name
            clip_sample_rate: Frames per second for CLIP analysis
            clip_min_score_threshold: Minimum CLIP score to consider relevant
            clip_weight_in_score: Weight of CLIP score in combined score (0-1)
            scene_detection_enabled: Whether to use scene detection (Q5)
            scene_detection_threshold: PySceneDetect threshold (1-100)
            prefer_complete_scenes: Adjust clips to scene boundaries when possible
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Client(api_key=api_key)

        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration
        self.max_clips_per_video = max_clips_per_video

        # S2 IMPROVEMENT: AI response caching
        self.cache = cache

        # Q1 IMPROVEMENT: CLIP visual matching
        self.clip_enabled = clip_enabled
        self.clip_model = clip_model
        self.clip_sample_rate = clip_sample_rate
        self.clip_min_score_threshold = clip_min_score_threshold
        self.clip_weight_in_score = clip_weight_in_score

        # Lazy-loaded visual matcher
        self._visual_matcher: Optional["VisualMatcher"] = None

        # Q5 IMPROVEMENT: Scene detection
        self.scene_detection_enabled = scene_detection_enabled
        self.scene_detection_threshold = scene_detection_threshold
        self.prefer_complete_scenes = prefer_complete_scenes

        # Lazy-loaded scene extractor
        self._scene_extractor: Optional["SceneAwareExtractor"] = None

        # Cache for scene detection results per video (avoid re-detecting)
        self._scene_cache: Dict[str, List[Tuple[float, float]]] = {}

        cache_status = "enabled" if cache else "disabled"
        logger.info(
            f"Initialized ClipExtractor with model: {model_name} (cache: {cache_status})"
        )
        if clip_enabled:
            logger.info(
                f"CLIP visual matching enabled: {clip_model} "
                f"(weight: {clip_weight_in_score}, threshold: {clip_min_score_threshold})"
            )
        if scene_detection_enabled:
            logger.info(
                f"Scene detection enabled: threshold={scene_detection_threshold}, "
                f"prefer_complete={prefer_complete_scenes}"
            )

    def process_downloaded_video(
        self,
        video_path: str,
        search_phrase: str,
        video_id: str,
        output_dir: Optional[str] = None,
        broll_need: Optional[BRollNeed] = None,
    ) -> Tuple[List[ClipResult], bool]:
        """Process a downloaded video: analyze, extract clips, cleanup.

        Args:
            video_path: Path to downloaded video file
            search_phrase: The search phrase context for B-roll relevance
            video_id: YouTube video ID
            output_dir: Directory for extracted clips (defaults to same as source)
            broll_need: Optional BRollNeed with original_context and required_elements
                        for enhanced semantic matching

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
        analysis = self.analyze_video(video_path, search_phrase, video_id, broll_need)

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
        self,
        video_path: str,
        search_phrase: str,
        video_id: str,
        broll_need: Optional[BRollNeed] = None,
    ) -> VideoAnalysisResult:
        """Analyze video using Gemini to identify relevant B-roll segments.

        S2 IMPROVEMENT: Results are cached by video file hash + search phrase.
        Cache key includes prompt version, file hash, and search phrase.

        Fix 4 Enhancement: When broll_need is provided, uses original_context and
        required_elements for semantic matching instead of just search_phrase.
        This produces more contextually relevant B-roll selections.

        Args:
            video_path: Path to video file
            search_phrase: Context for what B-roll content to look for
            video_id: YouTube video ID
            broll_need: Optional BRollNeed with original_context and required_elements
                        for enhanced semantic matching. When provided, the analysis
                        prioritizes segments showing required visual elements.

        Returns:
            VideoAnalysisResult with identified segments
        """
        temp_file_path = None
        try:
            # Get video duration for validation
            video_duration = self._get_video_duration(video_path)

            # S2 IMPROVEMENT: Generate cache key from video file hash + search phrase
            # File hash is computed incrementally for large files
            # Fix 4: Include original_context in cache key when available
            if self.cache:
                file_hash = compute_file_hash(video_path)
                context_for_cache = (
                    broll_need.original_context if broll_need and broll_need.original_context
                    else search_phrase
                )
                cache_key_content = (
                    f"{CLIP_ANALYSIS_PROMPT_VERSION}|"
                    f"{file_hash[:32]}|"
                    f"{compute_content_hash(context_for_cache)[:16]}"
                )

                cached_response = self.cache.get(cache_key_content, self.model_name)
                if cached_response:
                    try:
                        cached_data = json.loads(cached_response)
                        segments = self._parse_cached_segments(
                            cached_data, video_duration
                        )
                        logger.info(
                            f"[CACHE HIT] Video analysis: {len(segments)} segments "
                            f"for {video_id} (saved ~$0.06 API cost)"
                        )
                        return VideoAnalysisResult(
                            video_path=video_path,
                            video_id=video_id,
                            search_phrase=search_phrase,
                            analysis_success=True,
                            segments=segments,
                            total_duration=video_duration,
                        )
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to parse cached video analysis: {e}")
                        # Continue to make fresh API call

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
            # Fix 4: Pass broll_need for semantic context-aware prompting
            prompt = self._build_analysis_prompt(search_phrase, video_duration, broll_need)

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

            # S2 IMPROVEMENT: Cache the successful response
            if self.cache and segments:
                # Cache segments as JSON for later retrieval
                cache_data = [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "score": seg.relevance_score,
                        "rationale": seg.description,
                    }
                    for seg in segments
                ]
                self.cache.set(
                    cache_key_content, self.model_name, json.dumps(cache_data)
                )
                logger.debug(
                    f"[CACHE SAVE] Video analysis cached ({len(segments)} segments)"
                )

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

    def _build_analysis_prompt(
        self,
        search_phrase: str,
        video_duration: float,
        broll_need: Optional[BRollNeed] = None,
    ) -> str:
        """Build the prompt for video segment analysis.

        Fix 4 Enhancement: When broll_need is provided with original_context and
        required_elements, builds a semantic context-aware prompt that prioritizes
        segments matching the full narrative context, not just keywords.

        Args:
            search_phrase: Basic search phrase for fallback matching
            video_duration: Video duration in seconds
            broll_need: Optional BRollNeed with semantic context

        Returns:
            Prompt string for Gemini video analysis
        """
        # Fix 4: Use original_context and required_elements when available
        if broll_need and broll_need.original_context:
            return self._build_semantic_analysis_prompt(
                broll_need, search_phrase, video_duration
            )

        # Fallback to basic search phrase-only prompt
        return self._build_basic_analysis_prompt(search_phrase, video_duration)

    def _build_basic_analysis_prompt(
        self, search_phrase: str, video_duration: float
    ) -> str:
        """Build basic analysis prompt using only search phrase (legacy behavior)."""
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

SCORING (only include segments scoring 6 or higher):
- 10: Perfectly matches topic with excellent visual quality
- 8: Strong match with good quality
- 6: Adequate match, usable as B-roll
- 4: Loosely related (DO NOT INCLUDE)
- 2: Barely related (DO NOT INCLUDE)
- 0: Unrelated (DO NOT INCLUDE)

RULES:
- start_time and end_time are in seconds (decimals allowed)
- relevance_score is 1-10 (ONLY include segments scoring 6+)
- description should be brief (under 50 characters)
- Return empty array [] if no suitable segments found
- No markdown, no extra text, just the JSON array"""

    def _build_semantic_analysis_prompt(
        self,
        broll_need: BRollNeed,
        search_phrase: str,
        video_duration: float,
    ) -> str:
        """Build semantic context-aware analysis prompt (Fix 4 enhancement).

        This prompt uses original_context and required_elements from the BRollNeed
        to find segments that truly match the narrative meaning, not just keywords.

        Args:
            broll_need: BRollNeed with original_context and required_elements
            search_phrase: Fallback search phrase
            video_duration: Video duration in seconds

        Returns:
            Semantic context-aware prompt for Gemini video analysis
        """
        # Format required elements as a bulleted list
        required_elements_str = ""
        if broll_need.required_elements:
            elements_list = "\n".join(
                f"  - {element}" for element in broll_need.required_elements
            )
            required_elements_str = f"""
REQUIRED VISUAL ELEMENTS (must be visible in selected segments):
{elements_list}
"""

        return f"""You are a B-Roll Clip Analyzer specializing in semantic context matching.
Your task is to find segments in this video that match the ORIGINAL CONTEXT, not just keywords.

ORIGINAL CONTEXT (this is what the video narrator is discussing):
"{broll_need.original_context}"

SEARCH PHRASE: "{search_phrase}"
{required_elements_str}
GOAL:
Find {self.max_clips_per_video} or fewer segments ({self.min_clip_duration}-{self.max_clip_duration} seconds) where MOST required elements are visible.
Prefer fewer high-quality matches over many mediocre ones.

VIDEO DURATION: {video_duration:.1f} seconds

SCORING CRITERIA (ONLY return segments scoring 6 or higher):
- 10: ALL required elements clearly visible, perfectly matches original context meaning
- 8: MOST required elements visible, strong contextual match with minor gaps
- 6: SOME required elements visible, adequate contextual match
- 4: Matches search phrase keywords only (not the context meaning) - DO NOT INCLUDE
- 2: Loosely related to topic - DO NOT INCLUDE
- 0: Unrelated content - DO NOT INCLUDE

VISUAL QUALITY REQUIREMENTS:
- Focus on visually compelling shots (cinematic, clear subject, good lighting)
- Avoid: talking heads, text overlays, logos, watermarks, transitions
- Prefer: establishing shots, action shots, close-ups of relevant subjects

OUTPUT FORMAT:
Return ONLY a JSON array with this exact structure:
[
  {{
    "start_time": 12.5,
    "end_time": 20.0,
    "relevance_score": 9,
    "description": "Wide shot showing [visible elements]"
  }}
]

CRITICAL RULES:
- ONLY return segments scoring 6 or higher
- start_time and end_time are in seconds (decimals allowed)
- description should mention which required elements are visible
- Segments must not overlap
- Return empty array [] if no segments score 6+
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
                # Fix 4: Updated minimum score from 7 to 6 to match new scoring criteria
                if score < 6:
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

    def _parse_cached_segments(
        self, cached_data: list, video_duration: float
    ) -> List[ClipSegment]:
        """Parse segments from cached JSON data.

        S2 IMPROVEMENT: Helper method for reconstructing segments from cache.

        Args:
            cached_data: List of dicts with segment data
            video_duration: Video duration for validation

        Returns:
            List of ClipSegment objects
        """
        segments = []
        for item in cached_data:
            try:
                start = float(item.get("start_time", 0))
                end = float(item.get("end_time", 0))
                score = int(item.get("score", 0))
                rationale = str(item.get("rationale", ""))

                # Validate segment
                if start < 0 or end <= start:
                    continue
                if end > video_duration:
                    end = video_duration

                duration = end - start
                if duration < self.min_clip_duration:
                    continue
                if duration > self.max_clip_duration:
                    end = start + self.max_clip_duration

                segments.append(
                    ClipSegment(
                        start_time=start,
                        end_time=end,
                        relevance_score=score,
                        description=rationale[:100],
                    )
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid cached segment data: {item}, error: {e}")
                continue

        # Sort by relevance score (highest first)
        segments.sort(key=lambda s: s.relevance_score, reverse=True)
        return segments

    def analyze_videos_competitive(
        self,
        video_data,  # List[Tuple[VideoSearchResult, str]]
        search_phrase: str,
        broll_need: Optional[BRollNeed] = None,
    ) -> Optional[Tuple[Path, ClipSegment]]:
        """Analyze multiple videos and return single best clip across all.

        This method enables competitive analysis: download multiple preview videos,
        analyze all of them, and select the single best clip based on relevance scores.

        Fix 4 Enhancement: When broll_need is provided, uses original_context and
        required_elements for semantic matching to select the best clip.

        Args:
            video_data: List of (video_object, video_file_path) tuples
            search_phrase: Search phrase for relevance scoring
            broll_need: Optional BRollNeed with original_context and required_elements
                        for enhanced semantic matching

        Returns:
            Tuple of (source_video_path, best_segment) or None if no good clips found
        """
        all_segments = []

        for video, video_path in video_data:
            try:
                analysis = self.analyze_video(
                    video_path=str(video_path),
                    search_phrase=search_phrase,
                    video_id=video.video_id,
                    broll_need=broll_need,
                )

                if analysis.analysis_success and analysis.segments:
                    for segment in analysis.segments:
                        all_segments.append((Path(video_path), segment))
            except Exception as e:
                logger.warning(f"Failed to analyze {Path(video_path).name}: {e}")
                continue

        if not all_segments:
            return None

        # Sort by relevance score (highest first)
        all_segments.sort(key=lambda x: x[1].relevance_score, reverse=True)

        return all_segments[0]  # Return best clip across all videos

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

    # =========================================================================
    # Q1 IMPROVEMENT: CLIP Visual Matching Methods
    # =========================================================================

    def _get_visual_matcher(self) -> Optional["VisualMatcher"]:
        """Get or create the visual matcher instance (lazy loading).

        Returns:
            VisualMatcher instance if CLIP is enabled and available, None otherwise.
        """
        if not self.clip_enabled:
            return None

        if self._visual_matcher is not None:
            return self._visual_matcher

        try:
            from services.visual_matcher import VisualMatcher

            self._visual_matcher = VisualMatcher(
                model_name=self.clip_model,
                sample_rate=self.clip_sample_rate,
                min_score_threshold=self.clip_min_score_threshold,
            )

            if not self._visual_matcher.is_available():
                logger.warning(
                    "CLIP dependencies not available. "
                    "Install with: pip install transformers torch torchvision"
                )
                self.clip_enabled = False
                self._visual_matcher = None
                return None

            logger.info("CLIP visual matcher loaded successfully")
            return self._visual_matcher

        except ImportError as e:
            logger.warning(f"Failed to load CLIP visual matcher: {e}")
            self.clip_enabled = False
            return None

    def score_video_with_clip(
        self,
        video_path: str,
        description: str,
    ) -> Optional[float]:
        """Score a video's visual relevance using CLIP embeddings.

        Args:
            video_path: Path to video file
            description: Text description to match against

        Returns:
            Relevance score (0-1) or None if CLIP is disabled/unavailable
        """
        matcher = self._get_visual_matcher()
        if matcher is None:
            return None

        try:
            score = matcher.score_video_relevance(
                video_path,
                description,
                sample_rate=self.clip_sample_rate,
            )
            logger.debug(f"CLIP score for {Path(video_path).name}: {score:.3f}")
            return score
        except Exception as e:
            logger.warning(f"CLIP scoring failed for {video_path}: {e}")
            return None

    def find_best_segment_with_clip(
        self,
        video_path: str,
        description: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Find the best video segment using CLIP visual matching.

        Args:
            video_path: Path to video file
            description: Text description to match against

        Returns:
            Tuple of (start_time, end_time, score) or None if unavailable
        """
        matcher = self._get_visual_matcher()
        if matcher is None:
            return None

        try:
            result = matcher.find_best_segment(
                video_path,
                description,
                sample_rate=self.clip_sample_rate,
                min_duration=self.min_clip_duration,
                max_duration=self.max_clip_duration,
            )
            logger.info(
                f"CLIP found best segment: {result.best_start_time:.1f}s - "
                f"{result.best_end_time:.1f}s (score: {result.average_score:.3f})"
            )
            return (result.best_start_time, result.best_end_time, result.average_score)
        except Exception as e:
            logger.warning(f"CLIP segment finding failed for {video_path}: {e}")
            return None

    def enhance_segments_with_clip(
        self,
        video_path: str,
        segments: List[ClipSegment],
        description: str,
    ) -> List[ClipSegment]:
        """Enhance segment scores using CLIP visual matching.

        Combines Gemini relevance scores with CLIP visual similarity scores
        to produce more accurate final scores.

        Args:
            video_path: Path to video file
            segments: List of segments from Gemini analysis
            description: Text description to match against

        Returns:
            Enhanced segments with CLIP scores included
        """
        matcher = self._get_visual_matcher()
        if matcher is None or not segments:
            return segments

        try:
            # Extract frames for the entire video once
            frames, timestamps = matcher.extract_frames(
                video_path,
                sample_rate=self.clip_sample_rate,
            )

            if not frames:
                logger.warning("No frames extracted for CLIP enhancement")
                return segments

            # Score all frames against the description
            all_scores = matcher.score_frames(frames, description)

            # Map timestamps to indices for quick lookup
            timestamp_to_idx = {t: i for i, t in enumerate(timestamps)}

            enhanced_segments = []
            for segment in segments:
                # Find frames within this segment's time range
                segment_indices = [
                    i for i, t in enumerate(timestamps)
                    if segment.start_time <= t <= segment.end_time
                ]

                if segment_indices:
                    # Calculate average CLIP score for this segment
                    segment_scores = [all_scores[i] for i in segment_indices]
                    clip_score = sum(segment_scores) / len(segment_scores)
                else:
                    clip_score = 0.5  # Neutral if no frames in range

                # Store original Gemini score
                gemini_score = segment.relevance_score

                # Combine scores: CLIP weight determines blend
                # CLIP score is 0-1, Gemini is 1-10
                clip_score_scaled = clip_score * 10
                combined_score = int(
                    gemini_score * (1 - self.clip_weight_in_score) +
                    clip_score_scaled * self.clip_weight_in_score
                )

                # Clamp to 1-10 range
                combined_score = max(1, min(10, combined_score))

                # Create enhanced segment with CLIP metadata
                enhanced_segment = ClipSegment(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    relevance_score=combined_score,
                    description=segment.description,
                    clip_score=clip_score,
                    gemini_score=gemini_score,
                )
                enhanced_segments.append(enhanced_segment)

                logger.debug(
                    f"Segment {segment.start_time:.1f}s-{segment.end_time:.1f}s: "
                    f"Gemini={gemini_score}, CLIP={clip_score:.2f}, "
                    f"Combined={combined_score}"
                )

            # Re-sort by combined score
            enhanced_segments.sort(key=lambda s: s.relevance_score, reverse=True)

            logger.info(
                f"Enhanced {len(enhanced_segments)} segments with CLIP scores "
                f"(weight: {self.clip_weight_in_score})"
            )

            return enhanced_segments

        except Exception as e:
            logger.warning(f"CLIP enhancement failed: {e}")
            return segments

    def analyze_video_with_clip(
        self,
        video_path: str,
        search_phrase: str,
        video_id: str,
        use_clip_enhancement: bool = True,
        broll_need: Optional[BRollNeed] = None,
    ) -> VideoAnalysisResult:
        """Analyze video using both Gemini and CLIP for improved accuracy.

        This combines Gemini's contextual understanding with CLIP's visual
        matching to produce more accurate segment identification.

        Fix 4 Enhancement: When broll_need is provided, uses original_context and
        required_elements for semantic matching.

        Args:
            video_path: Path to video file
            search_phrase: Context for what B-roll content to look for
            video_id: Video identifier
            use_clip_enhancement: Whether to enhance with CLIP scores
            broll_need: Optional BRollNeed with original_context and required_elements

        Returns:
            VideoAnalysisResult with CLIP-enhanced segments
        """
        # First, run standard Gemini analysis
        result = self.analyze_video(video_path, search_phrase, video_id, broll_need)

        if not result.analysis_success or not result.segments:
            return result

        # Enhance with CLIP if enabled
        if self.clip_enabled and use_clip_enhancement:
            result.segments = self.enhance_segments_with_clip(
                video_path,
                result.segments,
                search_phrase,
            )

            # Also compute overall CLIP score for the video
            overall_clip_score = self.score_video_with_clip(video_path, search_phrase)
            if overall_clip_score is not None:
                result.clip_overall_score = overall_clip_score

        return result

    # =========================================================================
    # Q5 IMPROVEMENT: Scene Detection Methods
    # =========================================================================

    def _get_scene_extractor(self) -> Optional["SceneAwareExtractor"]:
        """Lazy-load the scene extractor.

        Returns:
            SceneAwareExtractor instance or None if scene detection is disabled
        """
        if not self.scene_detection_enabled:
            return None

        if self._scene_extractor is None:
            try:
                from services.scene_extractor import SceneAwareExtractor

                self._scene_extractor = SceneAwareExtractor(
                    threshold=self.scene_detection_threshold,
                    enabled=self.scene_detection_enabled,
                    prefer_complete_scenes=self.prefer_complete_scenes,
                )
                logger.info("Scene extractor loaded successfully")
            except ImportError as e:
                logger.warning(f"Scene detection dependencies not available: {e}")
                self.scene_detection_enabled = False
                return None
            except Exception as e:
                logger.error(f"Failed to initialize scene extractor: {e}")
                self.scene_detection_enabled = False
                return None

        return self._scene_extractor

    def _get_scenes_for_video(self, video_path: str) -> List[Tuple[float, float]]:
        """Get cached scene boundaries for a video, detecting if needed.

        Args:
            video_path: Path to video file

        Returns:
            List of (start_time, end_time) tuples for detected scenes
        """
        if not self.scene_detection_enabled:
            return []

        # Check cache first
        if video_path in self._scene_cache:
            logger.debug(f"Using cached scenes for {Path(video_path).name}")
            return self._scene_cache[video_path]

        # Detect scenes
        scene_extractor = self._get_scene_extractor()
        if scene_extractor is None:
            return []

        scenes = scene_extractor.detect_scenes(video_path)
        self._scene_cache[video_path] = scenes

        return scenes

    def _adjust_to_scene_boundaries(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        tolerance: float = 2.0,
    ) -> Tuple[float, float]:
        """Adjust clip boundaries to align with nearby scene boundaries.

        If scene detection is enabled and prefer_complete_scenes is True,
        this method will adjust the start and end times to snap to nearby
        scene boundaries, producing more coherent clips.

        Args:
            video_path: Path to video file
            start_time: Original start time in seconds
            end_time: Original end time in seconds
            tolerance: Maximum adjustment in seconds (default: 2.0)

        Returns:
            Tuple of (adjusted_start, adjusted_end) in seconds
        """
        if not self.scene_detection_enabled or not self.prefer_complete_scenes:
            return (start_time, end_time)

        scene_extractor = self._get_scene_extractor()
        if scene_extractor is None:
            return (start_time, end_time)

        scenes = self._get_scenes_for_video(video_path)
        if not scenes:
            return (start_time, end_time)

        adjusted_start = start_time
        adjusted_end = end_time

        # Find nearest scene boundaries
        for scene_start, scene_end in scenes:
            # Check if scene_start is close to our start_time
            if abs(scene_start - start_time) <= tolerance:
                adjusted_start = scene_start
                logger.debug(
                    f"Adjusted start from {start_time:.1f}s to scene boundary "
                    f"{scene_start:.1f}s"
                )

            # Check if scene_end is close to our end_time
            if abs(scene_end - end_time) <= tolerance:
                adjusted_end = scene_end
                logger.debug(
                    f"Adjusted end from {end_time:.1f}s to scene boundary "
                    f"{scene_end:.1f}s"
                )

        # Validate adjusted duration is within limits
        adjusted_duration = adjusted_end - adjusted_start
        if (
            adjusted_duration < self.min_clip_duration
            or adjusted_duration > self.max_clip_duration
        ):
            logger.debug(
                f"Adjusted duration {adjusted_duration:.1f}s outside limits "
                f"[{self.min_clip_duration}, {self.max_clip_duration}], "
                f"using original timestamps"
            )
            return (start_time, end_time)

        if (adjusted_start, adjusted_end) != (start_time, end_time):
            logger.info(
                f"Scene-adjusted timestamps: [{start_time:.1f}s-{end_time:.1f}s] -> "
                f"[{adjusted_start:.1f}s-{adjusted_end:.1f}s]"
            )

        return (adjusted_start, adjusted_end)

    def _find_best_complete_scene(
        self,
        video_path: str,
        target_start: float,
        target_end: float,
        search_phrase: str,
    ) -> Optional[Tuple[float, float]]:
        """Find the best complete scene that overlaps with target timestamps.

        Instead of using arbitrary timestamps, this finds a complete scene
        that has significant overlap with the target range, resulting in
        more coherent clips.

        Args:
            video_path: Path to video file
            target_start: Target start time in seconds
            target_end: Target end time in seconds
            search_phrase: Description for scoring scenes (if CLIP enabled)

        Returns:
            Tuple of (scene_start, scene_end) or None if no suitable scene
        """
        if not self.scene_detection_enabled or not self.prefer_complete_scenes:
            return None

        scene_extractor = self._get_scene_extractor()
        if scene_extractor is None:
            return None

        scenes = self._get_scenes_for_video(video_path)
        if not scenes:
            return None

        # Filter scenes by duration
        valid_scenes = scene_extractor.filter_scenes_by_duration(
            scenes, self.min_clip_duration, self.max_clip_duration
        )
        if not valid_scenes:
            return None

        # Find scenes that overlap with target range
        target_duration = target_end - target_start
        best_scene = None
        best_overlap_ratio = 0.0

        for scene_start, scene_end in valid_scenes:
            # Calculate overlap
            overlap_start = max(scene_start, target_start)
            overlap_end = min(scene_end, target_end)
            overlap = max(0, overlap_end - overlap_start)

            # Calculate overlap ratio (how much of target is covered)
            overlap_ratio = overlap / target_duration if target_duration > 0 else 0

            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_scene = (scene_start, scene_end)

        # Only use complete scene if significant overlap (>50%)
        if best_overlap_ratio > 0.5 and best_scene:
            logger.info(
                f"Using complete scene [{best_scene[0]:.1f}s-{best_scene[1]:.1f}s] "
                f"(overlap: {best_overlap_ratio * 100:.0f}%) instead of "
                f"[{target_start:.1f}s-{target_end:.1f}s]"
            )
            return best_scene

        return None

    def clear_scene_cache(self, video_path: Optional[str] = None) -> None:
        """Clear the scene detection cache.

        Args:
            video_path: If provided, clear cache only for this video.
                       If None, clear entire cache.
        """
        if video_path:
            if video_path in self._scene_cache:
                del self._scene_cache[video_path]
                logger.debug(f"Cleared scene cache for {Path(video_path).name}")
        else:
            self._scene_cache.clear()
            logger.debug("Cleared all scene cache")

    def extract_clip_with_scene_detection(
        self,
        video_path: str,
        segment: ClipSegment,
        search_phrase: str,
        video_id: str,
        output_dir: str,
        clip_index: int = 1,
    ) -> ClipResult:
        """Extract a clip with scene detection adjustment.

        This is an enhanced version of extract_clip that uses scene detection
        to align clip boundaries with natural scene transitions.

        Args:
            video_path: Source video path
            segment: ClipSegment with timing info
            search_phrase: For naming context and scene scoring
            video_id: Video identifier
            output_dir: Where to save the clip
            clip_index: Index for naming multiple clips

        Returns:
            ClipResult with extraction details
        """
        # Apply scene detection adjustment
        start_time = segment.start_time
        end_time = segment.end_time

        if self.scene_detection_enabled:
            # First, try to find a complete scene that overlaps with our target
            complete_scene = self._find_best_complete_scene(
                video_path, start_time, end_time, search_phrase
            )

            if complete_scene:
                start_time, end_time = complete_scene
            else:
                # Fall back to adjusting to nearby scene boundaries
                start_time, end_time = self._adjust_to_scene_boundaries(
                    video_path, start_time, end_time, tolerance=2.0
                )

        # Create an updated segment with adjusted times
        adjusted_segment = ClipSegment(
            start_time=start_time,
            end_time=end_time,
            relevance_score=segment.relevance_score,
            description=segment.description,
        )

        # Use the base extract_clip with adjusted segment
        result = self.extract_clip(
            video_path=video_path,
            segment=adjusted_segment,
            search_phrase=search_phrase,
            video_id=video_id,
            output_dir=output_dir,
            clip_index=clip_index,
        )

        # Log if we adjusted timestamps
        if (start_time, end_time) != (segment.start_time, segment.end_time):
            logger.info(
                f"Scene-aligned extraction: [{segment.start_time:.1f}s-"
                f"{segment.end_time:.1f}s] -> [{start_time:.1f}s-{end_time:.1f}s]"
            )

        return result
