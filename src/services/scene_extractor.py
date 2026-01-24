"""Scene-aware clip extraction using PySceneDetect.

This module provides scene detection capabilities for more coherent B-roll clips.
Instead of cutting mid-action or mid-sentence, this identifies natural scene
boundaries and selects complete scenes that best match the B-roll requirement.

Q5 improvement from improvements-plan-24-jan.md
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Any, Protocol

logger = logging.getLogger(__name__)


class FrameScorer(Protocol):
    """Protocol for frame scoring implementations (e.g., CLIP scorer from Q1)."""

    def score_frames(self, frames: List[Any], description: str) -> List[float]:
        """Score how well each frame matches the text description.

        Args:
            frames: List of PIL Image frames
            description: Text description to match against

        Returns:
            List of similarity scores (0-1) for each frame
        """
        ...


class SceneAwareExtractor:
    """Service for scene-based clip extraction using PySceneDetect.

    This class detects scene boundaries in videos and selects the best
    complete scene that matches a given description. This produces more
    coherent B-roll clips that don't cut mid-action.

    Configuration via environment variables:
        SCENE_DETECTION_ENABLED: Enable/disable scene detection (default: true)
        SCENE_DETECTION_THRESHOLD: ContentDetector threshold (default: 27)
        PREFER_COMPLETE_SCENES: Prefer complete scenes over timestamp cuts (default: true)
    """

    def __init__(
        self,
        threshold: Optional[int] = None,
        enabled: Optional[bool] = None,
        prefer_complete_scenes: Optional[bool] = None,
    ):
        """Initialize scene extractor.

        Args:
            threshold: Scene detection threshold (1-100, lower = more sensitive)
            enabled: Override for SCENE_DETECTION_ENABLED env var
            prefer_complete_scenes: Override for PREFER_COMPLETE_SCENES env var
        """
        self.threshold = threshold if threshold is not None else int(
            os.getenv("SCENE_DETECTION_THRESHOLD", "27")
        )
        self.enabled = enabled if enabled is not None else (
            os.getenv("SCENE_DETECTION_ENABLED", "true").lower() == "true"
        )
        self.prefer_complete_scenes = prefer_complete_scenes if prefer_complete_scenes is not None else (
            os.getenv("PREFER_COMPLETE_SCENES", "true").lower() == "true"
        )

        # Lazy-load scenedetect to handle import errors gracefully
        self._scenedetect_available: Optional[bool] = None

        logger.info(
            f"SceneAwareExtractor initialized: enabled={self.enabled}, "
            f"threshold={self.threshold}, prefer_complete_scenes={self.prefer_complete_scenes}"
        )

    def _check_scenedetect_available(self) -> bool:
        """Check if scenedetect is available and cache the result."""
        if self._scenedetect_available is not None:
            return self._scenedetect_available

        try:
            from scenedetect import detect, ContentDetector
            self._scenedetect_available = True
            logger.debug("PySceneDetect is available")
        except ImportError as e:
            self._scenedetect_available = False
            logger.warning(f"PySceneDetect not available: {e}")

        return self._scenedetect_available

    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Detect scene boundaries in video.

        Uses PySceneDetect's ContentDetector to find scene changes based on
        content changes between frames. The threshold controls sensitivity:
        lower values detect more subtle changes.

        Args:
            video_path: Path to video file

        Returns:
            List of (start_time, end_time) tuples in seconds for each scene.
            Returns empty list if scene detection is disabled or unavailable.
        """
        if not self.enabled:
            logger.debug("Scene detection disabled, returning empty list")
            return []

        if not self._check_scenedetect_available():
            logger.warning("PySceneDetect not available, returning empty list")
            return []

        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            return []

        try:
            from scenedetect import detect, ContentDetector

            logger.info(f"Detecting scenes in: {video_file.name} (threshold={self.threshold})")

            scene_list = detect(video_path, ContentDetector(threshold=self.threshold))

            scenes = [
                (scene[0].get_seconds(), scene[1].get_seconds())
                for scene in scene_list
            ]

            logger.info(f"Detected {len(scenes)} scenes in {video_file.name}")

            return scenes

        except Exception as e:
            logger.error(f"Scene detection failed for {video_path}: {e}")
            return []

    def filter_scenes_by_duration(
        self,
        scenes: List[Tuple[float, float]],
        min_duration: float = 4.0,
        max_duration: float = 15.0,
    ) -> List[Tuple[float, float]]:
        """Filter scenes by duration constraints.

        Args:
            scenes: List of (start_time, end_time) tuples
            min_duration: Minimum scene duration in seconds (default: 4.0)
            max_duration: Maximum scene duration in seconds (default: 15.0)

        Returns:
            Filtered list of scenes that meet duration constraints
        """
        valid_scenes = []

        for start, end in scenes:
            duration = end - start
            if min_duration <= duration <= max_duration:
                valid_scenes.append((start, end))
            else:
                logger.debug(
                    f"Filtered out scene {start:.1f}s-{end:.1f}s "
                    f"(duration {duration:.1f}s not in [{min_duration}, {max_duration}])"
                )

        logger.info(f"Filtered to {len(valid_scenes)} scenes meeting duration constraints")
        return valid_scenes

    def find_best_scene(
        self,
        video_path: str,
        description: str,
        min_duration: float = 4.0,
        max_duration: float = 15.0,
        clip_scorer: Optional[FrameScorer] = None,
    ) -> Optional[Tuple[float, float]]:
        """Find the scene that best matches the description.

        This method combines scene detection with optional CLIP-based scoring
        to find the best complete scene for B-roll usage.

        Args:
            video_path: Path to video file
            description: Text description of desired B-roll content
            min_duration: Minimum scene duration (default: 4.0s)
            max_duration: Maximum scene duration (default: 15.0s)
            clip_scorer: Optional frame scorer (e.g., CLIP model from Q1)

        Returns:
            Tuple of (start_time, end_time) for best scene, or None if no valid scenes
        """
        if not self.enabled:
            logger.debug("Scene detection disabled, returning None")
            return None

        # Detect all scenes
        scenes = self.detect_scenes(video_path)
        if not scenes:
            logger.info(f"No scenes detected in {video_path}")
            return None

        # Filter by duration
        valid_scenes = self.filter_scenes_by_duration(scenes, min_duration, max_duration)
        if not valid_scenes:
            logger.info(f"No scenes meet duration constraints in {video_path}")
            return None

        # If no scorer provided, return the longest valid scene
        # (longer scenes typically have more content and context)
        if clip_scorer is None:
            best_scene = max(valid_scenes, key=lambda s: s[1] - s[0])
            logger.info(
                f"No scorer provided, selecting longest scene: "
                f"{best_scene[0]:.1f}s-{best_scene[1]:.1f}s "
                f"(duration: {best_scene[1] - best_scene[0]:.1f}s)"
            )
            return best_scene

        # Score each scene with CLIP
        best_score = 0.0
        best_scene = None

        for start, end in valid_scenes:
            try:
                frames = self._extract_scene_frames(video_path, start, end)
                if not frames:
                    continue

                scores = clip_scorer.score_frames(frames, description)

                # Handle both list and single score returns
                if isinstance(scores, list):
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                else:
                    avg_score = float(scores)

                logger.debug(
                    f"Scene {start:.1f}s-{end:.1f}s scored {avg_score:.3f} "
                    f"for '{description[:30]}...'"
                )

                if avg_score > best_score:
                    best_score = avg_score
                    best_scene = (start, end)

            except Exception as e:
                logger.warning(f"Failed to score scene {start:.1f}s-{end:.1f}s: {e}")
                continue

        if best_scene:
            logger.info(
                f"Best scene for '{description[:30]}...': "
                f"{best_scene[0]:.1f}s-{best_scene[1]:.1f}s (score: {best_score:.3f})"
            )
        else:
            logger.warning(f"No scorable scenes found for '{description[:30]}...'")

        return best_scene

    def _extract_scene_frames(
        self,
        video_path: str,
        start: float,
        end: float,
        num_frames: int = 5,
    ) -> List[Any]:
        """Extract representative frames from a scene.

        Extracts evenly-spaced frames from the scene for scoring.

        Args:
            video_path: Path to video file
            start: Scene start time in seconds
            end: Scene end time in seconds
            num_frames: Number of frames to extract (default: 5)

        Returns:
            List of PIL Image frames, or empty list on error
        """
        try:
            import cv2
            from PIL import Image

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Default assumption
                logger.warning(f"Could not get FPS, assuming {fps}")

            frames = []
            duration = end - start

            # Calculate evenly-spaced timestamps
            if num_frames > 1:
                interval = duration / (num_frames - 1)
            else:
                interval = 0

            for i in range(num_frames):
                timestamp = start + (i * interval)
                frame_num = int(timestamp * fps)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                else:
                    logger.debug(f"Could not read frame at {timestamp:.2f}s")

            cap.release()

            logger.debug(f"Extracted {len(frames)} frames from scene {start:.1f}s-{end:.1f}s")
            return frames

        except ImportError as e:
            logger.error(f"Required library not available for frame extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []

    def get_scene_aware_segments(
        self,
        video_path: str,
        min_duration: float = 4.0,
        max_duration: float = 15.0,
        max_segments: int = 3,
        clip_scorer: Optional[FrameScorer] = None,
        description: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        """Get multiple scene-aware segments from a video.

        This method finds the top N scenes that best match the description.
        Useful when you want multiple clips from a single video.

        Args:
            video_path: Path to video file
            min_duration: Minimum scene duration (default: 4.0s)
            max_duration: Maximum scene duration (default: 15.0s)
            max_segments: Maximum number of segments to return (default: 3)
            clip_scorer: Optional frame scorer (e.g., CLIP model)
            description: Optional text description for scoring

        Returns:
            List of (start_time, end_time) tuples, sorted by quality score
        """
        if not self.enabled:
            return []

        scenes = self.detect_scenes(video_path)
        if not scenes:
            return []

        valid_scenes = self.filter_scenes_by_duration(scenes, min_duration, max_duration)
        if not valid_scenes:
            return []

        # If no scorer, return longest scenes
        if clip_scorer is None or description is None:
            sorted_scenes = sorted(
                valid_scenes,
                key=lambda s: s[1] - s[0],
                reverse=True
            )
            return sorted_scenes[:max_segments]

        # Score and rank scenes
        scored_scenes = []
        for start, end in valid_scenes:
            try:
                frames = self._extract_scene_frames(video_path, start, end)
                if not frames:
                    continue

                scores = clip_scorer.score_frames(frames, description)
                avg_score = sum(scores) / len(scores) if isinstance(scores, list) else float(scores)
                scored_scenes.append((start, end, avg_score))

            except Exception as e:
                logger.warning(f"Failed to score scene {start:.1f}s-{end:.1f}s: {e}")
                continue

        # Sort by score (highest first)
        scored_scenes.sort(key=lambda x: x[2], reverse=True)

        # Return top N scenes (without scores)
        return [(s[0], s[1]) for s in scored_scenes[:max_segments]]

    def adjust_segment_to_scene_boundary(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        tolerance: float = 2.0,
    ) -> Tuple[float, float]:
        """Adjust a segment's boundaries to align with nearby scene boundaries.

        If the start or end of a segment is close to a scene boundary,
        adjust it to use the natural scene boundary instead.

        Args:
            video_path: Path to video file
            start_time: Original start time in seconds
            end_time: Original end time in seconds
            tolerance: Maximum adjustment in seconds (default: 2.0)

        Returns:
            Tuple of (adjusted_start, adjusted_end) in seconds
        """
        if not self.enabled or not self.prefer_complete_scenes:
            return (start_time, end_time)

        scenes = self.detect_scenes(video_path)
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
                    f"Adjusted start from {start_time:.1f}s to scene boundary {scene_start:.1f}s"
                )

            # Check if scene_end is close to our end_time
            if abs(scene_end - end_time) <= tolerance:
                adjusted_end = scene_end
                logger.debug(
                    f"Adjusted end from {end_time:.1f}s to scene boundary {scene_end:.1f}s"
                )

        return (adjusted_start, adjusted_end)
