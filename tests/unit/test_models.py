"""Unit tests for data models."""

import pytest

from src.models.broll_need import BRollNeed, BRollPlan, TranscriptResult, TranscriptSegment


class TestBRollNeed:
    """Tests for BRollNeed model."""

    def test_create_broll_need(self):
        """Test creating a B-roll need with valid data."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline aerial",
            description="Aerial shot of city skyline at sunset",
            context="discussing urban development in modern cities",
        )
        assert need.timestamp == 30.0
        assert need.search_phrase == "city skyline aerial"
        assert need.suggested_duration == 5.0  # Default value

    def test_timestamp_clamping(self):
        """Test that negative timestamps are clamped to zero."""
        need = BRollNeed(
            timestamp=-10.0,
            search_phrase="test",
            description="test",
            context="test",
        )
        assert need.timestamp == 0.0

    def test_duration_clamping_min(self):
        """Test that duration below 4s is clamped to 4s."""
        need = BRollNeed(
            timestamp=0.0,
            search_phrase="test",
            description="test",
            context="test",
            suggested_duration=2.0,
        )
        assert need.suggested_duration == 4.0

    def test_duration_clamping_max(self):
        """Test that duration above 15s is clamped to 15s."""
        need = BRollNeed(
            timestamp=0.0,
            search_phrase="test",
            description="test",
            context="test",
            suggested_duration=20.0,
        )
        assert need.suggested_duration == 15.0

    def test_folder_name_generation(self):
        """Test timestamp-prefixed folder name generation."""
        need = BRollNeed(
            timestamp=90.5,  # 1m30s
            search_phrase="test",
            description="City Skyline Aerial!",
            context="test",
        )
        folder_name = need.folder_name
        assert folder_name.startswith("1m30s_")
        assert "city_skyline_aerial" in folder_name
        assert "!" not in folder_name  # Special chars removed

    def test_folder_name_sanitization(self):
        """Test that special characters are removed from folder names."""
        need = BRollNeed(
            timestamp=0.0,
            search_phrase="test",
            description="Test @#$% Description! With Symbols",
            context="test",
        )
        folder_name = need.folder_name
        # Should not contain special characters
        assert "@" not in folder_name
        assert "#" not in folder_name
        assert "$" not in folder_name
        assert "%" not in folder_name
        assert "!" not in folder_name

    def test_folder_name_length_limit(self):
        """Test that folder names are limited to reasonable length."""
        long_desc = "This is a very long description " * 10
        need = BRollNeed(
            timestamp=0.0,
            search_phrase="test",
            description=long_desc,
            context="test",
        )
        folder_name = need.folder_name
        # Should be trimmed (prefix + 55 chars + underscores)
        assert len(folder_name) < 75


class TestBRollPlan:
    """Tests for BRollPlan model."""

    def test_create_empty_plan(self):
        """Test creating an empty B-roll plan."""
        plan = BRollPlan(source_duration=300.0)  # 5 minutes
        assert plan.source_duration == 300.0
        assert plan.actual_clip_count == 0
        assert plan.clips_per_minute == 2.0

    def test_expected_clip_count_calculation(self):
        """Test calculation of expected clip count based on duration."""
        plan = BRollPlan(source_duration=300.0, clips_per_minute=2.0)  # 5 minutes
        assert plan.expected_clip_count == 10  # 5 min * 2 clips/min

    def test_expected_clip_count_minimum(self):
        """Test that expected clip count is at least 1."""
        plan = BRollPlan(source_duration=10.0, clips_per_minute=0.5)
        assert plan.expected_clip_count >= 1

    def test_actual_clip_count(self):
        """Test actual clip count matches number of needs."""
        plan = BRollPlan(source_duration=300.0)
        plan.needs = [
            BRollNeed(0.0, "test1", "desc1", "ctx1"),
            BRollNeed(30.0, "test2", "desc2", "ctx2"),
            BRollNeed(60.0, "test3", "desc3", "ctx3"),
        ]
        assert plan.actual_clip_count == 3

    def test_needs_sorting(self):
        """Test that needs can be sorted by timestamp."""
        plan = BRollPlan(source_duration=300.0)
        plan.needs = [
            BRollNeed(60.0, "test3", "desc3", "ctx3"),
            BRollNeed(0.0, "test1", "desc1", "ctx1"),
            BRollNeed(30.0, "test2", "desc2", "ctx2"),
        ]
        sorted_needs = plan.get_needs_sorted_by_timestamp()
        assert sorted_needs[0].timestamp == 0.0
        assert sorted_needs[1].timestamp == 30.0
        assert sorted_needs[2].timestamp == 60.0


class TestTranscriptSegment:
    """Tests for TranscriptSegment model."""

    def test_create_segment(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(
            start=0.0, end=5.0, text="Welcome to this video."
        )
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Welcome to this video."


class TestTranscriptResult:
    """Tests for TranscriptResult model."""

    def test_create_transcript_result(self):
        """Test creating a transcript result."""
        segments = [
            TranscriptSegment(0.0, 5.0, "Welcome to this video."),
            TranscriptSegment(5.0, 10.0, "Today we'll discuss AI."),
        ]
        result = TranscriptResult(
            text="Welcome to this video. Today we'll discuss AI.",
            segments=segments,
            duration=10.0,
            language="en",
        )
        assert result.duration == 10.0
        assert result.language == "en"
        assert len(result.segments) == 2

    def test_get_text_around_timestamp(self):
        """Test extracting text around a specific timestamp."""
        segments = [
            TranscriptSegment(0.0, 5.0, "First segment."),
            TranscriptSegment(5.0, 10.0, "Second segment."),
            TranscriptSegment(10.0, 15.0, "Third segment."),
            TranscriptSegment(15.0, 20.0, "Fourth segment."),
        ]
        result = TranscriptResult(text="", segments=segments, duration=20.0)

        # Get text around 10s with 10s context window
        text = result.get_text_around_timestamp(10.0, context_seconds=10.0)

        # Should include segments that overlap [5s, 15s]
        assert "Second segment." in text
        assert "Third segment." in text

    def test_get_text_around_timestamp_edge_cases(self):
        """Test edge cases for timestamp context extraction."""
        segments = [
            TranscriptSegment(0.0, 5.0, "First segment."),
            TranscriptSegment(5.0, 10.0, "Second segment."),
        ]
        result = TranscriptResult(text="", segments=segments, duration=10.0)

        # Test timestamp before video starts (should clamp to 0)
        text = result.get_text_around_timestamp(0.0, context_seconds=10.0)
        assert "First segment." in text

        # Test timestamp at end of video
        text = result.get_text_around_timestamp(10.0, context_seconds=10.0)
        assert "Second segment." in text

    def test_format_with_timestamps(self):
        """Test formatting transcript with timestamps."""
        segments = [
            TranscriptSegment(0.0, 5.0, "Welcome to this video."),
            TranscriptSegment(65.0, 70.0, "This is one minute in."),
        ]
        result = TranscriptResult(text="", segments=segments)

        formatted = result.format_with_timestamps()

        # Should have timestamp format [M:SS]
        assert "[0:00] Welcome to this video." in formatted
        assert "[1:05] This is one minute in." in formatted

    def test_empty_transcript(self):
        """Test handling of empty transcript."""
        result = TranscriptResult(text="", segments=[], duration=0.0)
        assert len(result.segments) == 0
        assert result.format_with_timestamps() == ""

    @pytest.mark.parametrize(
        "timestamp,expected_minutes,expected_seconds",
        [
            (0.0, 0, 0),
            (30.0, 0, 30),
            (65.0, 1, 5),
            (125.0, 2, 5),
            (3665.0, 61, 5),  # 1 hour, 1 minute, 5 seconds
        ],
    )
    def test_timestamp_formatting(
        self, timestamp, expected_minutes, expected_seconds
    ):
        """Test various timestamp formats."""
        segments = [TranscriptSegment(timestamp, timestamp + 5.0, "Test")]
        result = TranscriptResult(text="", segments=segments)
        formatted = result.format_with_timestamps()

        expected_format = f"[{expected_minutes}:{expected_seconds:02d}]"
        assert expected_format in formatted
