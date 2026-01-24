"""Unit tests for VisualMatcher CLIP-based visual matching service."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class TestVisualMatcherInit:
    """Tests for VisualMatcher initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from services.visual_matcher import VisualMatcher

        matcher = VisualMatcher()

        assert matcher.model_name == "openai/clip-vit-base-patch32"
        assert matcher.sample_rate == 1
        assert matcher.min_score_threshold == 0.3
        assert matcher._model is None  # Lazy loading
        assert matcher._processor is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from services.visual_matcher import VisualMatcher

        matcher = VisualMatcher(
            model_name="openai/clip-vit-large-patch14",
            sample_rate=2,
            min_score_threshold=0.5,
            device="cpu",
        )

        assert matcher.model_name == "openai/clip-vit-large-patch14"
        assert matcher.sample_rate == 2
        assert matcher.min_score_threshold == 0.5
        assert matcher._device == "cpu"


class TestVisualMatchResult:
    """Tests for VisualMatchResult dataclass."""

    def test_visual_match_result_creation(self):
        """Test creating a VisualMatchResult."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from services.visual_matcher import VisualMatchResult

        result = VisualMatchResult(
            video_path="/path/to/video.mp4",
            description="city skyline",
            best_start_time=5.0,
            best_end_time=12.0,
            peak_score=0.95,
            average_score=0.85,
            frame_scores=[0.7, 0.8, 0.9, 0.95, 0.85],
            timestamps=[5.0, 6.0, 7.0, 8.0, 9.0],
        )

        assert result.video_path == "/path/to/video.mp4"
        assert result.description == "city skyline"
        assert result.best_start_time == 5.0
        assert result.best_end_time == 12.0
        assert result.peak_score == 0.95
        assert result.average_score == 0.85
        assert len(result.frame_scores) == 5


class TestPeakRegionFinding:
    """Tests for finding peak scoring regions."""

    @pytest.fixture
    def matcher(self):
        """Create a basic matcher for peak region testing."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from services.visual_matcher import VisualMatcher

        return VisualMatcher()

    def test_find_peak_region_empty_scores(self, matcher):
        """Test with empty scores list."""
        start, end, score = matcher.find_peak_region([], [])
        assert start == 0.0
        assert end == 4.0  # min_duration default
        assert score == 0.0

    def test_find_peak_region_single_score(self, matcher):
        """Test with single score."""
        start, end, score = matcher.find_peak_region([0.8], [0.0])
        assert start == 0.0
        assert end == 4.0

    def test_find_peak_region_finds_peak(self, matcher):
        """Test that peak region is correctly identified."""
        # Create scores with a clear peak in the middle
        scores = [0.1, 0.2, 0.3, 0.9, 0.95, 0.9, 0.3, 0.2, 0.1]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        start, end, avg_score = matcher.find_peak_region(
            scores, timestamps, min_duration=4.0, max_duration=6.0
        )

        # Peak should be found somewhere in the 3-7 second range
        assert 2.0 <= start <= 4.0
        assert end >= start + 4.0
        assert avg_score > 0.5

    def test_find_peak_region_respects_min_duration(self, matcher):
        """Test that min_duration is respected."""
        scores = [0.9, 0.9, 0.9, 0.1, 0.1]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]

        start, end, _ = matcher.find_peak_region(
            scores, timestamps, min_duration=2.0, max_duration=5.0
        )

        assert (end - start) >= 2.0

    def test_find_peak_region_respects_max_duration(self, matcher):
        """Test that max_duration is respected."""
        scores = [0.9] * 20
        timestamps = list(range(20))

        start, end, _ = matcher.find_peak_region(
            scores, timestamps, min_duration=4.0, max_duration=8.0
        )

        assert (end - start) <= 8.0


class TestSingletonGetter:
    """Tests for the module-level singleton getter."""

    def test_get_visual_matcher_returns_singleton(self):
        """Test that get_visual_matcher returns the same instance."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        # Reset the singleton
        import services.visual_matcher as vm

        vm._visual_matcher_instance = None

        from services.visual_matcher import get_visual_matcher

        matcher1 = get_visual_matcher()
        matcher2 = get_visual_matcher()

        assert matcher1 is matcher2

    def test_get_visual_matcher_with_params(self):
        """Test singleton creation with custom parameters."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        # Reset the singleton
        import services.visual_matcher as vm

        vm._visual_matcher_instance = None

        from services.visual_matcher import get_visual_matcher

        matcher = get_visual_matcher(
            model_name="openai/clip-vit-large-patch14",
            sample_rate=2,
        )

        assert matcher.model_name == "openai/clip-vit-large-patch14"
        assert matcher.sample_rate == 2


class TestClipSegmentModel:
    """Tests for the ClipSegment model with CLIP metadata."""

    def test_clip_segment_with_clip_metadata(self):
        """Test ClipSegment with CLIP score metadata."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from models.clip import ClipSegment

        segment = ClipSegment(
            start_time=5.0,
            end_time=12.0,
            relevance_score=8,
            description="City skyline at sunset",
            clip_score=0.85,
            gemini_score=9,
        )

        assert segment.start_time == 5.0
        assert segment.end_time == 12.0
        assert segment.duration == 7.0
        assert segment.relevance_score == 8
        assert segment.clip_score == 0.85
        assert segment.gemini_score == 9

    def test_clip_segment_metadata_dict(self):
        """Test ClipSegment to_metadata_dict includes CLIP scores."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from models.clip import ClipSegment

        segment = ClipSegment(
            start_time=5.0,
            end_time=12.0,
            relevance_score=8,
            description="City skyline",
            clip_score=0.85,
            gemini_score=9,
        )

        metadata = segment.to_metadata_dict()

        assert metadata["start_time"] == 5.0
        assert metadata["end_time"] == 12.0
        assert metadata["duration"] == 7.0
        assert metadata["clip_score"] == 0.85
        assert metadata["gemini_score"] == 9


class TestClipExtractorCLIPIntegration:
    """Tests for CLIP integration in ClipExtractor."""

    def test_clip_extractor_init_with_clip_params(self):
        """Test ClipExtractor initializes with CLIP parameters."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        with patch("services.clip_extractor.Client"):
            from services.clip_extractor import ClipExtractor

            extractor = ClipExtractor(
                api_key="test_key",
                clip_enabled=True,
                clip_model="openai/clip-vit-base-patch32",
                clip_sample_rate=2,
                clip_min_score_threshold=0.4,
                clip_weight_in_score=0.5,
            )

            assert extractor.clip_enabled is True
            assert extractor.clip_model == "openai/clip-vit-base-patch32"
            assert extractor.clip_sample_rate == 2
            assert extractor.clip_min_score_threshold == 0.4
            assert extractor.clip_weight_in_score == 0.5
            assert extractor._visual_matcher is None  # Lazy loaded

    def test_clip_extractor_disabled(self):
        """Test ClipExtractor with CLIP disabled."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        with patch("services.clip_extractor.Client"):
            from services.clip_extractor import ClipExtractor

            extractor = ClipExtractor(
                api_key="test_key",
                clip_enabled=False,
            )

            assert extractor.clip_enabled is False

            # These should return None when CLIP is disabled
            result = extractor.score_video_with_clip("/fake/video.mp4", "test")
            assert result is None

    def test_score_video_with_clip_returns_none_when_disabled(self):
        """Test that scoring returns None when CLIP is disabled."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        with patch("services.clip_extractor.Client"):
            from services.clip_extractor import ClipExtractor

            extractor = ClipExtractor(api_key="test_key", clip_enabled=False)
            score = extractor.score_video_with_clip("/path/to/video.mp4", "test query")

            assert score is None


class TestVideoAnalysisResultCLIPMetadata:
    """Tests for VideoAnalysisResult with CLIP metadata."""

    def test_video_analysis_result_clip_score(self):
        """Test VideoAnalysisResult includes CLIP overall score."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from models.clip import VideoAnalysisResult, ClipSegment

        segment = ClipSegment(
            start_time=5.0,
            end_time=12.0,
            relevance_score=8,
            description="Test segment",
            clip_score=0.8,
            gemini_score=9,
        )

        result = VideoAnalysisResult(
            video_path="/path/to/video.mp4",
            video_id="test_id",
            search_phrase="test query",
            segments=[segment],
            analysis_success=True,
            clip_overall_score=0.75,
        )

        assert result.clip_overall_score == 0.75
        assert len(result.segments) == 1
        assert result.segments[0].clip_score == 0.8
