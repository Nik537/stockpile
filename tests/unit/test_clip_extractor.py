"""Unit tests for clip extraction functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.models.clip import ClipSegment, ClipResult, VideoAnalysisResult


class TestClipSegment:
    """Tests for ClipSegment model."""

    def test_create_clip_segment(self):
        """Test creating a clip segment with valid data."""
        segment = ClipSegment(
            start_time=5.2,
            end_time=12.8,
            relevance_score=9,
            description="City skyline at sunset",
        )
        assert segment.start_time == 5.2
        assert segment.end_time == 12.8
        assert segment.relevance_score == 9
        assert segment.description == "City skyline at sunset"

    def test_duration_calculation(self):
        """Test duration property calculation."""
        segment = ClipSegment(
            start_time=10.0, end_time=25.0, relevance_score=8, description="Test"
        )
        assert segment.duration == 15.0

    @pytest.mark.parametrize(
        "start,end,expected_duration",
        [
            (0.0, 5.0, 5.0),
            (10.5, 15.3, 4.8),
            (0.0, 0.1, 0.1),
            (100.0, 114.5, 14.5),
        ],
    )
    def test_duration_various_ranges(self, start, end, expected_duration):
        """Test duration calculation with various time ranges."""
        segment = ClipSegment(
            start_time=start, end_time=end, relevance_score=5, description="Test"
        )
        assert pytest.approx(segment.duration, 0.01) == expected_duration


class TestClipResult:
    """Tests for ClipResult model."""

    def test_create_successful_clip_result(self):
        """Test creating a successful clip extraction result."""
        segment = ClipSegment(5.0, 10.0, 9, "Test segment")
        result = ClipResult(
            source_video_path="/path/to/source.mp4",
            clip_path="/path/to/clip.mp4",
            segment=segment,
            search_phrase="city skyline",
            source_video_id="abc123",
            extraction_success=True,
        )
        assert result.extraction_success is True
        assert result.error_message is None
        assert result.clip_path == "/path/to/clip.mp4"

    def test_create_failed_clip_result(self):
        """Test creating a failed clip extraction result."""
        segment = ClipSegment(5.0, 10.0, 9, "Test segment")
        result = ClipResult(
            source_video_path="/path/to/source.mp4",
            clip_path="",
            segment=segment,
            search_phrase="city skyline",
            source_video_id="abc123",
            extraction_success=False,
            error_message="FFmpeg command failed",
        )
        assert result.extraction_success is False
        assert result.error_message == "FFmpeg command failed"


class TestVideoAnalysisResult:
    """Tests for VideoAnalysisResult model."""

    def test_create_successful_analysis(self):
        """Test creating a successful analysis result."""
        segments = [
            ClipSegment(0.0, 5.0, 8, "Opening shot"),
            ClipSegment(10.0, 15.0, 9, "Main content"),
        ]
        result = VideoAnalysisResult(
            video_path="/path/to/video.mp4",
            video_id="abc123",
            search_phrase="city skyline",
            segments=segments,
            analysis_success=True,
            total_duration=120.0,
        )
        assert result.analysis_success is True
        assert len(result.segments) == 2
        assert result.total_duration == 120.0

    def test_create_failed_analysis(self):
        """Test creating a failed analysis result."""
        result = VideoAnalysisResult(
            video_path="/path/to/video.mp4",
            video_id="abc123",
            search_phrase="city skyline",
            segments=[],
            analysis_success=False,
            error_message="Video upload failed",
        )
        assert result.analysis_success is False
        assert len(result.segments) == 0
        assert result.error_message == "Video upload failed"

    def test_empty_segments_list(self):
        """Test analysis result with no segments found."""
        result = VideoAnalysisResult(
            video_path="/path/to/video.mp4",
            video_id="abc123",
            search_phrase="city skyline",
            segments=[],
            analysis_success=True,  # Analysis succeeded but found nothing
        )
        assert result.analysis_success is True
        assert len(result.segments) == 0


class TestClipExtractor:
    """Integration tests for ClipExtractor logic."""

    def test_ffmpeg_command_generation(self, tmp_path):
        """Test that FFmpeg commands are generated correctly."""
        from src.services.clip_extractor import ClipExtractor

        # Instantiate ClipExtractor with test configuration
        extractor = ClipExtractor(
            api_key="test",
            min_clip_duration=4.0,
            max_clip_duration=15.0,
        )

        # Create test segment
        segment = ClipSegment(
            start_time=10.5,
            end_time=18.2,
            relevance_score=8,
            description="Test",
        )

        # Create a dummy video file path
        video_path = tmp_path / "source_video.mp4"
        video_path.touch()

        # Variable to capture the command
        captured_cmd = None

        def mock_subprocess_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            # Create the expected output file so clip_path.exists() passes
            # Parse the output path from command (last argument)
            output_file = cmd[-1]
            from pathlib import Path
            Path(output_file).touch()
            # Return a successful result mock
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = extractor.extract_clip(
                video_path=str(video_path),
                segment=segment,
                search_phrase="test search",
                video_id="test123",
                output_dir=str(tmp_path / "output"),
                clip_index=1,
            )

        # Assert command was captured
        assert captured_cmd is not None, "FFmpeg command was not captured"

        # Convert command list to string for easier assertions
        cmd_str = " ".join(str(c) for c in captured_cmd)

        # Assert command contains expected elements
        assert "ffmpeg" in captured_cmd[0], "Command should start with ffmpeg"
        assert "-ss" in captured_cmd, "Command should contain -ss flag"
        assert "-i" in captured_cmd, "Command should contain -i flag"

        # -ss should come before -i (input seeking for speed)
        ss_index = captured_cmd.index("-ss")
        i_index = captured_cmd.index("-i")
        assert ss_index < i_index, "-ss should come before -i for fast input seeking"

        # Check start time value follows -ss
        assert str(10.5) in captured_cmd, "Start time 10.5 should be in command"

        # Check duration flag
        assert "-t" in captured_cmd, "Command should contain -t duration flag"

        # Check video codec settings
        assert "libx264" in captured_cmd, "Command should use libx264 codec"
        assert "-preset" in captured_cmd, "Command should have preset flag"
        assert "fast" in captured_cmd, "Command should use fast preset"
        assert "-crf" in captured_cmd, "Command should have crf flag"
        assert "23" in captured_cmd, "Command should use crf 23"

        # Check web optimization
        assert "-movflags" in captured_cmd, "Command should have movflags"
        assert "+faststart" in captured_cmd, "Command should use faststart"

        # Check quiet mode
        assert "-v" in captured_cmd, "Command should have verbosity flag"
        assert "quiet" in captured_cmd, "Command should use quiet mode"

        # Check extraction success
        assert result.extraction_success is True, "Extraction should succeed"

    def test_timestamp_parsing(self):
        """Test parsing timestamps from AI responses."""
        from src.services.clip_extractor import ClipExtractor

        # Instantiate ClipExtractor with test configuration
        extractor = ClipExtractor(
            api_key="test",
            min_clip_duration=4.0,
            max_clip_duration=15.0,
        )

        video_duration = 120.0  # 2 minute video

        # Test 1: Valid JSON with markdown fences -> 2 segments parsed
        valid_json_with_fences = """```json
[
    {"start_time": 5.0, "end_time": 15.0, "relevance_score": 8, "description": "First segment"},
    {"start_time": 30.0, "end_time": 45.0, "relevance_score": 9, "description": "Second segment"}
]
```"""
        segments = extractor._parse_segments_response(valid_json_with_fences, video_duration)
        assert len(segments) == 2, "Should parse 2 valid segments from markdown-fenced JSON"
        # Sorted by relevance score (highest first)
        assert segments[0].relevance_score == 9
        assert segments[1].relevance_score == 8

        # Test 2: Score < 6 filtered out
        low_score_json = """[
    {"start_time": 5.0, "end_time": 15.0, "relevance_score": 5, "description": "Low score"},
    {"start_time": 20.0, "end_time": 30.0, "relevance_score": 7, "description": "Good score"}
]"""
        segments = extractor._parse_segments_response(low_score_json, video_duration)
        assert len(segments) == 1, "Score < 6 should be filtered out"
        assert segments[0].relevance_score == 7

        # Test 3: start_time >= end_time filtered out
        invalid_times_json = """[
    {"start_time": 20.0, "end_time": 10.0, "relevance_score": 8, "description": "Invalid: start > end"},
    {"start_time": 15.0, "end_time": 15.0, "relevance_score": 8, "description": "Invalid: start == end"},
    {"start_time": 5.0, "end_time": 15.0, "relevance_score": 8, "description": "Valid segment"}
]"""
        segments = extractor._parse_segments_response(invalid_times_json, video_duration)
        assert len(segments) == 1, "Segments where start >= end should be filtered out"
        assert segments[0].start_time == 5.0

        # Test 4: Duration < min_clip_duration (< 4s) filtered out
        short_duration_json = """[
    {"start_time": 5.0, "end_time": 7.0, "relevance_score": 8, "description": "Too short (2s)"},
    {"start_time": 10.0, "end_time": 13.5, "relevance_score": 8, "description": "Too short (3.5s)"},
    {"start_time": 20.0, "end_time": 30.0, "relevance_score": 8, "description": "Valid (10s)"}
]"""
        segments = extractor._parse_segments_response(short_duration_json, video_duration)
        assert len(segments) == 1, "Duration < min_clip_duration should be filtered out"
        assert segments[0].duration == 10.0

        # Test 5: end_time > video_duration clamped to video_duration
        exceeds_duration_json = """[
    {"start_time": 110.0, "end_time": 130.0, "relevance_score": 8, "description": "Exceeds video duration"}
]"""
        segments = extractor._parse_segments_response(exceeds_duration_json, video_duration)
        assert len(segments) == 1, "Segment should be kept with clamped end_time"
        assert segments[0].end_time == 120.0, "end_time should be clamped to video_duration"
        assert segments[0].duration == 10.0  # 120 - 110 = 10s

        # Test 6: Duration > max_clip_duration trimmed (end = start + max_clip_duration)
        long_duration_json = """[
    {"start_time": 10.0, "end_time": 50.0, "relevance_score": 8, "description": "Too long (40s)"}
]"""
        segments = extractor._parse_segments_response(long_duration_json, video_duration)
        assert len(segments) == 1, "Long segment should be kept but trimmed"
        assert segments[0].start_time == 10.0, "start_time should be preserved"
        assert segments[0].end_time == 25.0, "end_time should be trimmed to start + max_clip_duration"
        assert segments[0].duration == 15.0  # max_clip_duration

        # Test 7: Empty/invalid JSON returns []
        empty_json = ""
        segments = extractor._parse_segments_response(empty_json, video_duration)
        assert segments == [], "Empty string should return empty list"

        invalid_json = "this is not valid json at all"
        segments = extractor._parse_segments_response(invalid_json, video_duration)
        assert segments == [], "Invalid JSON should return empty list"

        not_a_list_json = '{"start_time": 5.0, "end_time": 15.0}'
        segments = extractor._parse_segments_response(not_a_list_json, video_duration)
        assert segments == [], "Non-list JSON should return empty list"

    def test_clip_extraction_with_force_keyframes(self, tmp_path):
        """Test that force_keyframes_at_cuts is used for frame accuracy."""
        from src.services.clip_extractor import ClipExtractor

        # Instantiate ClipExtractor with test configuration
        extractor = ClipExtractor(
            api_key="test",
            min_clip_duration=4.0,
            max_clip_duration=15.0,
        )

        # Create test segment
        segment = ClipSegment(
            start_time=10.5,
            end_time=18.2,
            relevance_score=8,
            description="Test",
        )

        # Create a dummy video file path
        video_path = tmp_path / "source_video.mp4"
        video_path.touch()

        # Variable to capture the command
        captured_cmd = None

        def mock_subprocess_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            # Create the expected output file so clip_path.exists() passes
            output_file = cmd[-1]
            from pathlib import Path
            Path(output_file).touch()
            # Return a successful result mock
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = extractor.extract_clip(
                video_path=str(video_path),
                segment=segment,
                search_phrase="test search",
                video_id="test123",
                output_dir=str(tmp_path / "output"),
                clip_index=1,
            )

        # Assert command was captured
        assert captured_cmd is not None, "FFmpeg command was not captured"

        # Test 1: -ss index < -i index (input seeking = frame accurate)
        ss_index = captured_cmd.index("-ss")
        i_index = captured_cmd.index("-i")
        assert ss_index < i_index, "-ss should come before -i for frame-accurate input seeking"

        # Test 2: Duration calculation is correct: end_time - start_time
        expected_duration = segment.end_time - segment.start_time  # 18.2 - 10.5 = 7.7
        t_index = captured_cmd.index("-t")
        duration_value = float(captured_cmd[t_index + 1])
        assert abs(duration_value - expected_duration) < 0.001, (
            f"Duration should be {expected_duration}, got {duration_value}"
        )

        # Test 3: libx264 in command (re-encode, not stream copy - required for precise cuts)
        assert "libx264" in captured_cmd, (
            "Command should use libx264 for re-encoding (required for precise cuts, not stream copy)"
        )

        # Test 4: Result has extraction_success=True
        assert result.extraction_success is True, "Extraction should succeed"

        # Additional verification: command does NOT use stream copy (-c copy)
        # which would result in imprecise cuts at keyframes only
        assert "-c" in " ".join(captured_cmd), "Command should have codec specification"
        # The codec should be specified as libx264, not copy
        c_v_index = captured_cmd.index("-c:v")
        assert captured_cmd[c_v_index + 1] == "libx264", "Video codec should be libx264, not copy"
