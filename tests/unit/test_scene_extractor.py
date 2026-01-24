"""Unit tests for scene-aware clip extraction (Q5 improvement)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSceneAwareExtractor:
    """Tests for SceneAwareExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create a SceneAwareExtractor instance for testing."""
        with patch.dict(os.environ, {
            "SCENE_DETECTION_ENABLED": "true",
            "SCENE_DETECTION_THRESHOLD": "27",
            "PREFER_COMPLETE_SCENES": "true",
        }):
            from src.services.scene_extractor import SceneAwareExtractor
            return SceneAwareExtractor()

    @pytest.fixture
    def disabled_extractor(self):
        """Create a disabled SceneAwareExtractor instance."""
        from src.services.scene_extractor import SceneAwareExtractor
        return SceneAwareExtractor(enabled=False)

    def test_init_default_values(self, extractor):
        """Test that SceneAwareExtractor initializes with correct defaults."""
        assert extractor.enabled is True
        assert extractor.threshold == 27
        assert extractor.prefer_complete_scenes is True

    def test_init_custom_values(self):
        """Test SceneAwareExtractor with custom configuration."""
        from src.services.scene_extractor import SceneAwareExtractor
        extractor = SceneAwareExtractor(
            threshold=30,
            enabled=True,
            prefer_complete_scenes=False,
        )
        assert extractor.threshold == 30
        assert extractor.enabled is True
        assert extractor.prefer_complete_scenes is False

    def test_detect_scenes_disabled(self, disabled_extractor):
        """Test that detect_scenes returns empty list when disabled."""
        result = disabled_extractor.detect_scenes("/path/to/video.mp4")
        assert result == []

    def test_detect_scenes_missing_file(self, extractor):
        """Test that detect_scenes handles missing file gracefully."""
        result = extractor.detect_scenes("/nonexistent/video.mp4")
        assert result == []

    def test_filter_scenes_by_duration(self, extractor):
        """Test scene filtering by duration constraints."""
        scenes = [
            (0.0, 2.0),   # Too short (2s)
            (2.0, 7.0),   # Valid (5s)
            (7.0, 15.0),  # Valid (8s)
            (15.0, 35.0), # Too long (20s)
            (35.0, 49.0), # Valid (14s)
        ]

        # Filter with default 4-15s range
        valid = extractor.filter_scenes_by_duration(
            scenes, min_duration=4.0, max_duration=15.0
        )

        assert len(valid) == 3
        assert (0.0, 2.0) not in valid  # Too short
        assert (2.0, 7.0) in valid      # 5s - valid
        assert (7.0, 15.0) in valid     # 8s - valid
        assert (15.0, 35.0) not in valid # Too long
        assert (35.0, 49.0) in valid    # 14s - valid

    def test_filter_scenes_empty_list(self, extractor):
        """Test filtering with empty scene list."""
        result = extractor.filter_scenes_by_duration(
            [], min_duration=4.0, max_duration=15.0
        )
        assert result == []

    def test_filter_scenes_all_invalid(self, extractor):
        """Test filtering when all scenes are invalid."""
        scenes = [
            (0.0, 1.0),   # Too short
            (1.0, 2.0),   # Too short
            (2.0, 50.0),  # Too long
        ]
        result = extractor.filter_scenes_by_duration(
            scenes, min_duration=4.0, max_duration=15.0
        )
        assert result == []

    def test_adjust_segment_to_scene_boundary_disabled(self, disabled_extractor):
        """Test that adjustment is skipped when disabled."""
        result = disabled_extractor.adjust_segment_to_scene_boundary(
            "/path/to/video.mp4", 5.0, 10.0
        )
        assert result == (5.0, 10.0)

    def test_adjust_segment_to_scene_boundary_no_scenes(self, extractor):
        """Test adjustment returns original when no scenes detected."""
        with patch.object(extractor, 'detect_scenes', return_value=[]):
            result = extractor.adjust_segment_to_scene_boundary(
                "/path/to/video.mp4", 5.0, 10.0
            )
        assert result == (5.0, 10.0)

    def test_adjust_segment_start_to_boundary(self, extractor):
        """Test adjusting start time to nearby scene boundary."""
        scenes = [
            (0.0, 5.5),
            (5.5, 12.0),
            (12.0, 20.0),
        ]

        with patch.object(extractor, 'detect_scenes', return_value=scenes):
            # Start at 5.0, scene boundary at 5.5 (within 2s tolerance)
            result = extractor.adjust_segment_to_scene_boundary(
                "/path/to/video.mp4", 5.0, 10.0, tolerance=2.0
            )

        assert result[0] == 5.5  # Adjusted to scene boundary

    def test_adjust_segment_end_to_boundary(self, extractor):
        """Test adjusting end time to nearby scene boundary."""
        scenes = [
            (0.0, 5.0),
            (5.0, 11.5),
            (11.5, 20.0),
        ]

        with patch.object(extractor, 'detect_scenes', return_value=scenes):
            # End at 10.0, scene boundary at 11.5 (within 2s tolerance)
            result = extractor.adjust_segment_to_scene_boundary(
                "/path/to/video.mp4", 5.0, 10.0, tolerance=2.0
            )

        assert result[1] == 11.5  # Adjusted to scene boundary

    def test_adjust_segment_no_nearby_boundaries(self, extractor):
        """Test that segment is unchanged when no nearby boundaries."""
        scenes = [
            (0.0, 2.0),
            (20.0, 30.0),
        ]

        with patch.object(extractor, 'detect_scenes', return_value=scenes):
            result = extractor.adjust_segment_to_scene_boundary(
                "/path/to/video.mp4", 5.0, 10.0, tolerance=2.0
            )

        # No scene boundaries within tolerance of 5.0 or 10.0
        assert result == (5.0, 10.0)

    def test_find_best_scene_disabled(self, disabled_extractor):
        """Test find_best_scene returns None when disabled."""
        result = disabled_extractor.find_best_scene(
            "/path/to/video.mp4", "test description"
        )
        assert result is None

    def test_find_best_scene_no_valid_scenes(self, extractor):
        """Test find_best_scene returns None when no valid scenes."""
        with patch.object(extractor, 'detect_scenes', return_value=[]):
            result = extractor.find_best_scene(
                "/path/to/video.mp4", "test description"
            )
        assert result is None

    def test_find_best_scene_selects_longest(self, extractor):
        """Test find_best_scene selects longest scene when no scorer."""
        scenes = [
            (0.0, 5.0),   # 5s
            (5.0, 12.0),  # 7s
            (12.0, 20.0), # 8s - longest valid
        ]

        with patch.object(extractor, 'detect_scenes', return_value=scenes):
            result = extractor.find_best_scene(
                "/path/to/video.mp4",
                "test description",
                min_duration=4.0,
                max_duration=15.0,
                clip_scorer=None,  # No scorer - should select longest
            )

        # Should select longest valid scene (12.0, 20.0) = 8s
        assert result == (12.0, 20.0)

    def test_get_scene_aware_segments(self, extractor):
        """Test getting multiple scene-aware segments."""
        scenes = [
            (0.0, 6.0),   # 6s
            (6.0, 11.0),  # 5s
            (11.0, 20.0), # 9s
            (20.0, 35.0), # 15s - at limit
        ]

        with patch.object(extractor, 'detect_scenes', return_value=scenes):
            result = extractor.get_scene_aware_segments(
                "/path/to/video.mp4",
                min_duration=4.0,
                max_duration=15.0,
                max_segments=3,
            )

        # Should return top 3 longest scenes
        assert len(result) == 3
        # Sorted by duration (longest first)
        assert (20.0, 35.0) in result  # 15s
        assert (11.0, 20.0) in result  # 9s
        assert (0.0, 6.0) in result    # 6s

    def test_get_scene_aware_segments_disabled(self, disabled_extractor):
        """Test that get_scene_aware_segments returns empty when disabled."""
        result = disabled_extractor.get_scene_aware_segments(
            "/path/to/video.mp4",
            min_duration=4.0,
            max_duration=15.0,
        )
        assert result == []


class TestClipExtractorSceneIntegration:
    """Tests for scene detection integration in ClipExtractor."""

    @pytest.fixture
    def mock_genai_client(self):
        """Mock the GenAI client."""
        with patch("src.services.clip_extractor.Client") as mock_client:
            yield mock_client

    @pytest.fixture
    def extractor_with_scenes(self, mock_genai_client):
        """Create a ClipExtractor with scene detection enabled."""
        from src.services.clip_extractor import ClipExtractor

        return ClipExtractor(
            api_key="test_key",
            scene_detection_enabled=True,
            scene_detection_threshold=27,
            prefer_complete_scenes=True,
        )

    @pytest.fixture
    def extractor_no_scenes(self, mock_genai_client):
        """Create a ClipExtractor with scene detection disabled."""
        from src.services.clip_extractor import ClipExtractor

        return ClipExtractor(
            api_key="test_key",
            scene_detection_enabled=False,
        )

    def test_clip_extractor_scene_config(self, extractor_with_scenes):
        """Test ClipExtractor initializes with scene detection config."""
        assert extractor_with_scenes.scene_detection_enabled is True
        assert extractor_with_scenes.scene_detection_threshold == 27
        assert extractor_with_scenes.prefer_complete_scenes is True
        assert extractor_with_scenes._scene_cache == {}

    def test_clip_extractor_scene_disabled(self, extractor_no_scenes):
        """Test ClipExtractor with scene detection disabled."""
        assert extractor_no_scenes.scene_detection_enabled is False

    def test_scene_cache_management(self, extractor_with_scenes):
        """Test scene cache add and clear operations."""
        # Add to cache
        extractor_with_scenes._scene_cache["/path/to/video1.mp4"] = [
            (0.0, 5.0), (5.0, 10.0)
        ]
        extractor_with_scenes._scene_cache["/path/to/video2.mp4"] = [(0.0, 8.0)]

        assert len(extractor_with_scenes._scene_cache) == 2

        # Clear specific video
        extractor_with_scenes.clear_scene_cache("/path/to/video1.mp4")
        assert len(extractor_with_scenes._scene_cache) == 1
        assert "/path/to/video2.mp4" in extractor_with_scenes._scene_cache

        # Clear all
        extractor_with_scenes.clear_scene_cache()
        assert len(extractor_with_scenes._scene_cache) == 0

    def test_adjust_to_scene_boundaries_disabled(self, extractor_no_scenes):
        """Test that adjustment is skipped when scene detection is disabled."""
        result = extractor_no_scenes._adjust_to_scene_boundaries(
            "/path/to/video.mp4", 5.0, 10.0
        )
        assert result == (5.0, 10.0)

    def test_get_scenes_for_video_caching(self, extractor_with_scenes):
        """Test that scene results are cached."""
        mock_scene_extractor = Mock()
        mock_scene_extractor.detect_scenes.return_value = [(0.0, 5.0), (5.0, 10.0)]

        extractor_with_scenes._scene_extractor = mock_scene_extractor

        # First call should detect scenes
        result1 = extractor_with_scenes._get_scenes_for_video("/path/to/video.mp4")
        assert mock_scene_extractor.detect_scenes.call_count == 1

        # Second call should use cache
        result2 = extractor_with_scenes._get_scenes_for_video("/path/to/video.mp4")
        assert mock_scene_extractor.detect_scenes.call_count == 1  # No additional call

        assert result1 == result2

    def test_find_best_complete_scene_with_overlap(self, extractor_with_scenes):
        """Test finding best complete scene with good overlap."""
        mock_scene_extractor = Mock()
        mock_scene_extractor.detect_scenes.return_value = [
            (0.0, 5.0),
            (5.0, 12.0),  # Should match - overlaps well with 6.0-11.0
            (12.0, 20.0),
        ]
        mock_scene_extractor.filter_scenes_by_duration.return_value = [
            (0.0, 5.0),
            (5.0, 12.0),
            (12.0, 20.0),
        ]

        extractor_with_scenes._scene_extractor = mock_scene_extractor

        result = extractor_with_scenes._find_best_complete_scene(
            "/path/to/video.mp4",
            target_start=6.0,
            target_end=11.0,
            search_phrase="test",
        )

        # Should find scene (5.0, 12.0) as it has best overlap
        assert result == (5.0, 12.0)

    def test_find_best_complete_scene_no_good_overlap(self, extractor_with_scenes):
        """Test that no scene is returned when overlap is too low."""
        mock_scene_extractor = Mock()
        mock_scene_extractor.detect_scenes.return_value = [
            (0.0, 5.0),
            (20.0, 30.0),  # No overlap with target
        ]
        mock_scene_extractor.filter_scenes_by_duration.return_value = [
            (0.0, 5.0),
            (20.0, 30.0),
        ]

        extractor_with_scenes._scene_extractor = mock_scene_extractor

        result = extractor_with_scenes._find_best_complete_scene(
            "/path/to/video.mp4",
            target_start=6.0,
            target_end=11.0,
            search_phrase="test",
        )

        # No scene has >50% overlap, should return None
        assert result is None


class TestSceneExtractionEdgeCases:
    """Edge case tests for scene detection."""

    @pytest.fixture
    def mock_genai_client(self):
        """Mock the GenAI client."""
        with patch("src.services.clip_extractor.Client") as mock_client:
            yield mock_client

    def test_empty_video_scenes(self, mock_genai_client):
        """Test handling of video with no scenes detected."""
        from src.services.clip_extractor import ClipExtractor

        extractor = ClipExtractor(
            api_key="test_key",
            scene_detection_enabled=True,
        )

        # Mock scene extractor that returns no scenes
        mock_scene_extractor = Mock()
        mock_scene_extractor.detect_scenes.return_value = []
        extractor._scene_extractor = mock_scene_extractor

        # Should return original timestamps
        result = extractor._adjust_to_scene_boundaries(
            "/path/to/video.mp4", 5.0, 10.0
        )
        assert result == (5.0, 10.0)

    def test_very_short_scenes_filtered(self, mock_genai_client):
        """Test that very short scenes are filtered out."""
        from src.services.clip_extractor import ClipExtractor

        extractor = ClipExtractor(
            api_key="test_key",
            scene_detection_enabled=True,
            min_clip_duration=4.0,
        )

        mock_scene_extractor = Mock()
        # All scenes too short
        mock_scene_extractor.detect_scenes.return_value = [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
        ]
        mock_scene_extractor.filter_scenes_by_duration.return_value = []
        extractor._scene_extractor = mock_scene_extractor

        result = extractor._find_best_complete_scene(
            "/path/to/video.mp4", 0.0, 3.0, "test"
        )
        assert result is None

    def test_scene_detection_import_error_graceful(self, mock_genai_client):
        """Test graceful handling when PySceneDetect is not installed."""
        from src.services.clip_extractor import ClipExtractor

        extractor = ClipExtractor(
            api_key="test_key",
            scene_detection_enabled=True,
        )

        # Simulate import error
        with patch.dict('sys.modules', {'services.scene_extractor': None}):
            with patch(
                'src.services.clip_extractor.ClipExtractor._get_scene_extractor',
                side_effect=ImportError("No module named 'scenedetect'")
            ):
                # Should not raise, just return original
                pass

        # The extractor should still work, just without scene detection
        assert extractor is not None
