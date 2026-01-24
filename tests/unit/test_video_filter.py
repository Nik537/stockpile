"""Unit tests for video pre-filtering service (S5 improvement)."""

import pytest
from unittest.mock import patch

from src.models.video import VideoResult
from src.services.video_filter import (
    FilterConfig,
    FilterStats,
    VideoPreFilter,
    filter_videos,
    should_download,
)


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""

    def test_default_values(self):
        """Test default filter configuration values."""
        config = FilterConfig()
        assert config.min_view_count == 1000
        assert config.max_prefilter_duration == 600
        assert config.prefer_creative_commons is True
        assert "compilation" in config.blocked_title_keywords
        assert "top 10" in config.blocked_title_keywords

    def test_from_config_with_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "min_view_count": 5000,
            "max_prefilter_duration": 300,
            "blocked_title_keywords": "test,keywords",
            "prefer_creative_commons": False,
        }
        config = FilterConfig.from_config(config_dict)
        assert config.min_view_count == 5000
        assert config.max_prefilter_duration == 300
        assert config.blocked_title_keywords == ["test", "keywords"]
        assert config.prefer_creative_commons is False

    def test_from_config_with_list_keywords(self):
        """Test creating config with list of blocked keywords."""
        config_dict = {
            "blocked_title_keywords": ["one", "two", "three"],
        }
        config = FilterConfig.from_config(config_dict)
        assert config.blocked_title_keywords == ["one", "two", "three"]

    @patch.dict(
        "os.environ",
        {
            "MIN_VIEW_COUNT": "2000",
            "MAX_PREFILTER_DURATION": "900",
            "BLOCKED_TITLE_KEYWORDS": "custom,blocked",
            "PREFER_CREATIVE_COMMONS": "false",
        },
    )
    def test_from_config_with_env_vars(self):
        """Test creating config from environment variables."""
        config = FilterConfig.from_config({})
        assert config.min_view_count == 2000
        assert config.max_prefilter_duration == 900
        assert config.blocked_title_keywords == ["custom", "blocked"]
        assert config.prefer_creative_commons is False


class TestFilterStats:
    """Tests for FilterStats dataclass."""

    def test_filter_rate_calculation(self):
        """Test filter rate percentage calculation."""
        stats = FilterStats(total_input=100, total_passed=75, total_filtered=25)
        assert stats.filter_rate == 25.0

    def test_filter_rate_with_zero_input(self):
        """Test filter rate is 0 when no input."""
        stats = FilterStats()
        assert stats.filter_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = FilterStats(
            total_input=10,
            total_passed=7,
            total_filtered=3,
            reasons={"Too long: 700s > 600s": 2, "Low view count: 500 < 1000": 1},
        )
        result = stats.to_dict()
        assert result["total_input"] == 10
        assert result["total_passed"] == 7
        assert result["total_filtered"] == 3
        assert result["filter_rate_percent"] == 30.0
        assert "Too long: 700s > 600s" in result["reasons"]

    def test_str_representation(self):
        """Test string representation."""
        stats = FilterStats(total_input=10, total_passed=7, total_filtered=3)
        assert "Filtered 3/10" in str(stats)
        assert "30.0%" in str(stats)
        assert "7 passed" in str(stats)


class TestShouldDownload:
    """Tests for should_download function."""

    @pytest.fixture
    def default_config(self):
        """Default filter configuration for tests."""
        return FilterConfig(
            min_view_count=1000,
            max_prefilter_duration=600,
            blocked_title_keywords=["compilation", "top 10", "reaction"],
            prefer_creative_commons=True,
        )

    @pytest.fixture
    def good_video(self):
        """Video that should pass all filters."""
        return VideoResult(
            video_id="good123",
            title="Beautiful City Skyline Footage",
            url="https://youtube.com/watch?v=good123",
            duration=120,
            view_count=10000,
        )

    def test_passes_with_good_video(self, good_video, default_config):
        """Test that a good video passes all filters."""
        passed, reason = should_download(good_video, default_config)
        assert passed is True
        assert reason == "Passed all filters"

    def test_rejects_low_view_count(self, default_config):
        """Test rejection of videos with low view count."""
        video = VideoResult(
            video_id="lowviews",
            title="Some Video",
            url="https://youtube.com/watch?v=lowviews",
            duration=60,
            view_count=500,
        )
        passed, reason = should_download(video, default_config)
        assert passed is False
        assert "Low view count" in reason
        assert "500" in reason

    def test_passes_without_view_count(self, default_config):
        """Test that videos without view count info are not rejected."""
        video = VideoResult(
            video_id="noviews",
            title="Some Video",
            url="https://youtube.com/watch?v=noviews",
            duration=60,
            view_count=None,
        )
        passed, reason = should_download(video, default_config)
        assert passed is True

    def test_rejects_too_long_video(self, default_config):
        """Test rejection of videos exceeding max duration."""
        video = VideoResult(
            video_id="toolong",
            title="Long Video",
            url="https://youtube.com/watch?v=toolong",
            duration=700,
            view_count=5000,
        )
        passed, reason = should_download(video, default_config)
        assert passed is False
        assert "Too long" in reason
        assert "700s" in reason

    def test_rejects_blocked_keyword_compilation(self, default_config):
        """Test rejection of videos with 'compilation' in title."""
        video = VideoResult(
            video_id="comp",
            title="Epic City Compilation 2024",
            url="https://youtube.com/watch?v=comp",
            duration=60,
            view_count=5000,
        )
        passed, reason = should_download(video, default_config)
        assert passed is False
        assert "Blocked keyword" in reason
        assert "compilation" in reason

    def test_rejects_blocked_keyword_top10(self, default_config):
        """Test rejection of videos with 'top 10' in title."""
        video = VideoResult(
            video_id="top",
            title="Top 10 Best City Views",
            url="https://youtube.com/watch?v=top",
            duration=60,
            view_count=5000,
        )
        passed, reason = should_download(video, default_config)
        assert passed is False
        assert "Blocked keyword" in reason
        assert "top 10" in reason

    def test_rejects_blocked_keyword_reaction(self, default_config):
        """Test rejection of videos with 'reaction' in title."""
        video = VideoResult(
            video_id="react",
            title="My Reaction to City Footage",
            url="https://youtube.com/watch?v=react",
            duration=60,
            view_count=5000,
        )
        passed, reason = should_download(video, default_config)
        assert passed is False
        assert "Blocked keyword" in reason
        assert "reaction" in reason

    def test_blocked_keyword_case_insensitive(self, default_config):
        """Test that keyword matching is case-insensitive."""
        video = VideoResult(
            video_id="caps",
            title="COMPILATION of Best Shots",
            url="https://youtube.com/watch?v=caps",
            duration=60,
            view_count=5000,
        )
        passed, reason = should_download(video, default_config)
        assert passed is False

    def test_exact_duration_limit(self, default_config):
        """Test video at exact duration limit passes."""
        video = VideoResult(
            video_id="exact",
            title="Video",
            url="https://youtube.com/watch?v=exact",
            duration=600,  # Exactly at limit
            view_count=5000,
        )
        passed, reason = should_download(video, default_config)
        assert passed is True

    def test_exact_view_count_limit(self, default_config):
        """Test video at exact view count limit passes."""
        video = VideoResult(
            video_id="exact",
            title="Video",
            url="https://youtube.com/watch?v=exact",
            duration=60,
            view_count=1000,  # Exactly at limit
        )
        passed, reason = should_download(video, default_config)
        assert passed is True


class TestFilterVideos:
    """Tests for filter_videos function."""

    @pytest.fixture
    def sample_videos(self):
        """Sample list of videos for batch filtering tests."""
        return [
            VideoResult(
                video_id="good1",
                title="Beautiful Sunset",
                url="https://youtube.com/watch?v=good1",
                duration=60,
                view_count=10000,
            ),
            VideoResult(
                video_id="good2",
                title="City Skyline HD",
                url="https://youtube.com/watch?v=good2",
                duration=90,
                view_count=5000,
            ),
            VideoResult(
                video_id="low_views",
                title="Random Video",
                url="https://youtube.com/watch?v=low_views",
                duration=60,
                view_count=100,
            ),
            VideoResult(
                video_id="too_long",
                title="Super Long Video",
                url="https://youtube.com/watch?v=too_long",
                duration=1200,
                view_count=20000,
            ),
            VideoResult(
                video_id="compilation",
                title="Best Compilation 2024",
                url="https://youtube.com/watch?v=compilation",
                duration=60,
                view_count=50000,
            ),
        ]

    def test_filters_batch_correctly(self, sample_videos):
        """Test filtering a batch of videos."""
        config = FilterConfig(
            min_view_count=1000,
            max_prefilter_duration=600,
            blocked_title_keywords=["compilation"],
        )
        filtered, stats = filter_videos(sample_videos, config)

        # Should keep good1 and good2, reject low_views, too_long, and compilation
        assert len(filtered) == 2
        assert stats.total_input == 5
        assert stats.total_passed == 2
        assert stats.total_filtered == 3

    def test_empty_input(self):
        """Test filtering empty list returns empty list."""
        filtered, stats = filter_videos([])
        assert filtered == []
        assert stats.total_input == 0
        assert stats.total_passed == 0
        assert stats.total_filtered == 0

    def test_tracks_reasons(self, sample_videos):
        """Test that filter reasons are tracked."""
        config = FilterConfig(
            min_view_count=1000,
            max_prefilter_duration=600,
            blocked_title_keywords=["compilation"],
        )
        filtered, stats = filter_videos(sample_videos, config)

        assert len(stats.reasons) > 0
        # Should have at least one reason for each filter type triggered

    def test_log_callback(self, sample_videos):
        """Test that log callback is called for filtered videos."""
        logged_messages = []

        def capture_log(msg):
            logged_messages.append(msg)

        config = FilterConfig(
            min_view_count=1000,
            max_prefilter_duration=600,
            blocked_title_keywords=["compilation"],
        )
        filter_videos(sample_videos, config, log_callback=capture_log)

        # Should have logged 3 rejected videos
        assert len(logged_messages) == 3


class TestVideoPreFilter:
    """Tests for VideoPreFilter service class."""

    def test_initialization(self):
        """Test service initialization."""
        prefilter = VideoPreFilter()
        assert prefilter.filter_config is not None
        assert prefilter.cumulative_stats.total_input == 0

    def test_filter_accumulates_stats(self):
        """Test that multiple filter calls accumulate statistics."""
        prefilter = VideoPreFilter({"min_view_count": 1000, "max_prefilter_duration": 600})

        batch1 = [
            VideoResult(
                video_id="v1", title="Video 1", url="url", duration=60, view_count=5000
            ),
            VideoResult(
                video_id="v2", title="Video 2", url="url", duration=60, view_count=100
            ),
        ]

        batch2 = [
            VideoResult(
                video_id="v3", title="Video 3", url="url", duration=60, view_count=5000
            ),
        ]

        prefilter.filter(batch1)
        prefilter.filter(batch2)

        stats = prefilter.get_stats()
        assert stats.total_input == 3  # 2 + 1
        assert stats.total_passed == 2  # 1 + 1
        assert stats.total_filtered == 1  # 1 + 0

    def test_reset_stats(self):
        """Test resetting accumulated statistics."""
        prefilter = VideoPreFilter()
        prefilter.cumulative_stats.total_input = 10
        prefilter.cumulative_stats.total_filtered = 5

        prefilter.reset_stats()

        stats = prefilter.get_stats()
        assert stats.total_input == 0
        assert stats.total_filtered == 0

    def test_get_report_no_videos(self):
        """Test report generation with no videos processed."""
        prefilter = VideoPreFilter()
        report = prefilter.get_report()
        assert "No videos processed" in report

    def test_get_report_with_videos(self):
        """Test report generation after processing videos."""
        prefilter = VideoPreFilter({"min_view_count": 1000, "max_prefilter_duration": 600})

        videos = [
            VideoResult(
                video_id="v1", title="Good Video", url="url", duration=60, view_count=5000
            ),
            VideoResult(
                video_id="v2", title="Low Views", url="url", duration=60, view_count=100
            ),
            VideoResult(
                video_id="v3", title="Too Long", url="url", duration=1000, view_count=5000
            ),
        ]

        prefilter.filter(videos)
        report = prefilter.get_report()

        assert "Pre-Filter Report" in report
        assert "Total videos processed: 3" in report
        assert "Videos passed: 1" in report
        assert "Videos filtered: 2" in report
        assert "Filter reasons:" in report
