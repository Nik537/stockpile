"""Visual matching service using CLIP embeddings for text-to-image similarity scoring."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VisualMatchResult:
    """Result of visual matching analysis."""

    video_path: str
    description: str
    best_start_time: float
    best_end_time: float
    peak_score: float
    average_score: float
    frame_scores: list[float]
    timestamps: list[float]


class VisualMatcher:
    """Service for scoring video frames against text descriptions using CLIP embeddings.

    CLIP (Contrastive Language-Image Pre-training) enables comparing images to text
    descriptions, allowing us to find video segments that best match a search phrase
    without relying solely on metadata.

    This is the SINGLE BIGGEST quality improvement for B-roll relevance.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        sample_rate: int = 1,
        min_score_threshold: float = 0.3,
    ):
        """Initialize the visual matcher with CLIP model.

        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to use (None for auto-detection)
            sample_rate: Frames per second to analyze
            min_score_threshold: Minimum similarity score to consider relevant
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.min_score_threshold = min_score_threshold

        # Lazy loading - only import when actually used
        self._model = None
        self._processor = None
        self._device = device
        self._torch = None

        logger.info(f"VisualMatcher initialized with model: {model_name}")

    def _ensure_loaded(self) -> None:
        """Lazy load the CLIP model and processor."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self._torch = torch

            # Auto-detect device: CUDA > MPS (Apple Silicon) > CPU
            if self._device is None:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

            logger.info(f"Loading CLIP model on device: {self._device}")

            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)

            # Move model to device
            self._model = self._model.to(self._device)
            self._model.eval()  # Set to evaluation mode

            logger.info(f"CLIP model loaded successfully on {self._device}")

        except ImportError as e:
            logger.error(
                f"Failed to import CLIP dependencies. Install with: "
                f"pip install transformers torch torchvision"
            )
            raise ImportError(
                "CLIP dependencies not installed. Install with: "
                "pip install transformers torch torchvision"
            ) from e

    def extract_frames(
        self,
        video_path: str,
        sample_rate: Optional[int] = None,
        max_frames: int = 300,
    ) -> tuple[list, list[float]]:
        """Extract frames from video at given sample rate.

        Uses OpenCV for efficient frame extraction without decoding entire video.

        Args:
            video_path: Path to video file
            sample_rate: Frames per second to extract (None uses instance default)
            max_frames: Maximum frames to extract (prevents memory issues)

        Returns:
            Tuple of (list of PIL Images, list of timestamps in seconds)
        """
        try:
            import cv2
            from PIL import Image
        except ImportError as e:
            logger.error("OpenCV/Pillow not installed. Install with: pip install opencv-python pillow")
            raise ImportError(
                "OpenCV and Pillow required for frame extraction. "
                "Install with: pip install opencv-python pillow"
            ) from e

        if sample_rate is None:
            sample_rate = self.sample_rate

        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = []
        timestamps = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                logger.warning(f"Could not get FPS, defaulting to 30")
                fps = 30.0

            # Calculate frame interval based on sample rate
            # If sample_rate=1, extract 1 frame per second
            # If sample_rate=2, extract 2 frames per second, etc.
            frame_interval = max(1, int(fps / sample_rate))

            logger.debug(
                f"Video: {fps:.1f} FPS, {total_frames} total frames, "
                f"extracting every {frame_interval}th frame"
            )

            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    frames.append(pil_image)
                    timestamps.append(frame_count / fps)
                    extracted_count += 1

                    if extracted_count >= max_frames:
                        logger.warning(
                            f"Reached max frames limit ({max_frames}), stopping extraction"
                        )
                        break

                frame_count += 1

            logger.info(f"Extracted {len(frames)} frames from {video_path}")

        finally:
            cap.release()

        return frames, timestamps

    def score_frames(
        self,
        frames: list,  # List of PIL Images
        description: str,
        batch_size: int = 32,
    ) -> list[float]:
        """Score how well each frame matches the text description.

        Uses CLIP to compute cosine similarity between frame embeddings
        and text embedding.

        Args:
            frames: List of PIL Image objects
            description: Text description to match against
            batch_size: Number of frames to process at once (memory optimization)

        Returns:
            List of similarity scores (0-1) for each frame
        """
        if not frames:
            return []

        self._ensure_loaded()

        all_scores = []

        # Process in batches to avoid memory issues
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]

            # Prepare inputs
            inputs = self._processor(
                text=[description],
                images=batch_frames,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self._model(**inputs)

                # Get image-text similarity scores
                # logits_per_image: (batch_size, 1) - each image vs the text
                logits_per_image = outputs.logits_per_image

                # Apply sigmoid to get scores in 0-1 range
                # Note: logits are already scaled, so we use sigmoid not softmax
                scores = self._torch.sigmoid(logits_per_image / 100.0)

                # Alternative: normalize scores using softmax across all frames
                # But for per-frame scoring, sigmoid is more interpretable

                batch_scores = scores.squeeze().cpu().tolist()

                # Handle single frame case
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                all_scores.extend(batch_scores)

        return all_scores

    def find_peak_region(
        self,
        scores: list[float],
        timestamps: list[float],
        min_duration: float = 4.0,
        max_duration: float = 15.0,
        smoothing_window: int = 3,
    ) -> tuple[float, float, float]:
        """Find the contiguous region with highest average score.

        Uses a sliding window approach to find the best segment that meets
        duration requirements.

        Args:
            scores: List of frame scores
            timestamps: List of frame timestamps in seconds
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration
            smoothing_window: Window size for score smoothing

        Returns:
            Tuple of (start_time, end_time, average_score)
        """
        if len(scores) < 2 or len(timestamps) < 2:
            return (0.0, min_duration, 0.0)

        # Apply simple moving average smoothing
        smoothed_scores = []
        for i in range(len(scores)):
            start_idx = max(0, i - smoothing_window // 2)
            end_idx = min(len(scores), i + smoothing_window // 2 + 1)
            smoothed_scores.append(sum(scores[start_idx:end_idx]) / (end_idx - start_idx))

        best_start = 0.0
        best_end = min(timestamps[-1], min_duration)
        best_avg_score = 0.0

        # Calculate frame interval from timestamps
        if len(timestamps) >= 2:
            frame_interval = timestamps[1] - timestamps[0]
        else:
            frame_interval = 1.0

        # Sliding window to find best segment
        for start_idx in range(len(timestamps)):
            start_time = timestamps[start_idx]

            # Try different end points
            for end_idx in range(start_idx + 1, len(timestamps)):
                end_time = timestamps[end_idx]
                duration = end_time - start_time

                if duration < min_duration:
                    continue
                if duration > max_duration:
                    break

                # Calculate average score for this segment
                segment_scores = smoothed_scores[start_idx:end_idx + 1]
                avg_score = sum(segment_scores) / len(segment_scores)

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_start = start_time
                    best_end = end_time

        return best_start, best_end, best_avg_score

    def find_best_segment(
        self,
        video_path: str,
        description: str,
        sample_rate: Optional[int] = None,
        min_duration: float = 4.0,
        max_duration: float = 15.0,
    ) -> VisualMatchResult:
        """Find the video segment that best matches the description.

        Complete workflow: extract frames, score them, find peak region.

        Args:
            video_path: Path to video file
            description: Text description to match against
            sample_rate: Frames per second to analyze (None uses instance default)
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration

        Returns:
            VisualMatchResult with segment timing and scores
        """
        logger.info(f"Analyzing video for: '{description}'")

        # Extract frames
        frames, timestamps = self.extract_frames(video_path, sample_rate)

        if not frames:
            logger.warning(f"No frames extracted from {video_path}")
            return VisualMatchResult(
                video_path=video_path,
                description=description,
                best_start_time=0.0,
                best_end_time=min_duration,
                peak_score=0.0,
                average_score=0.0,
                frame_scores=[],
                timestamps=[],
            )

        # Score frames against description
        scores = self.score_frames(frames, description)

        # Find best segment
        best_start, best_end, best_avg = self.find_peak_region(
            scores, timestamps, min_duration, max_duration
        )

        # Calculate peak score
        peak_score = max(scores) if scores else 0.0

        logger.info(
            f"Best segment: {best_start:.1f}s - {best_end:.1f}s "
            f"(avg score: {best_avg:.3f}, peak: {peak_score:.3f})"
        )

        return VisualMatchResult(
            video_path=video_path,
            description=description,
            best_start_time=best_start,
            best_end_time=best_end,
            peak_score=peak_score,
            average_score=best_avg,
            frame_scores=scores,
            timestamps=timestamps,
        )

    def score_video_relevance(
        self,
        video_path: str,
        description: str,
        sample_rate: Optional[int] = None,
    ) -> float:
        """Get overall relevance score for a video against a description.

        Quick scoring method that returns a single 0-1 score.
        Useful for pre-filtering videos before detailed analysis.

        Args:
            video_path: Path to video file
            description: Text description to match against
            sample_rate: Frames per second to analyze

        Returns:
            Float score 0-1 indicating video-description relevance
        """
        result = self.find_best_segment(
            video_path, description, sample_rate
        )

        # Return the peak score as the overall relevance metric
        return result.peak_score

    def is_available(self) -> bool:
        """Check if CLIP is available (dependencies installed).

        Returns:
            True if CLIP can be used, False otherwise
        """
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            return True
        except ImportError:
            return False


# Module-level singleton for convenience
_visual_matcher_instance: Optional[VisualMatcher] = None


def get_visual_matcher(
    model_name: str = "openai/clip-vit-base-patch32",
    sample_rate: int = 1,
    min_score_threshold: float = 0.3,
) -> VisualMatcher:
    """Get or create a singleton VisualMatcher instance.

    This avoids loading the model multiple times.

    Args:
        model_name: HuggingFace model name for CLIP
        sample_rate: Frames per second to analyze
        min_score_threshold: Minimum similarity score threshold

    Returns:
        VisualMatcher instance
    """
    global _visual_matcher_instance

    if _visual_matcher_instance is None:
        _visual_matcher_instance = VisualMatcher(
            model_name=model_name,
            sample_rate=sample_rate,
            min_score_threshold=min_score_threshold,
        )

    return _visual_matcher_instance
