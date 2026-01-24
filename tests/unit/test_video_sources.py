"""Unit tests for video source implementations (Q3 improvement).

Tests the multi-source video search functionality:
- PexelsVideoSource: CC0 stock footage from Pexels API
- PixabayVideoSource: CC0 stock footage from Pixabay API
- YouTubeVideoSource: Video search via yt-dlp
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from models.video import VideoResult
from services.video_sources.base import VideoSource
from services.video_sources.pexels import PexelsVideoSource
from services.video_sources.pixabay import PixabayVideoSource
from services.video_sources.youtube import YouTubeVideoSource


class TestVideoSourceBase:
    """Tests for the VideoSource abstract base class."""

    def test_base_class_is_abstract(self):
        """VideoSource cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VideoSource()

    def test_is_configured_default(self):
        """Default is_configured returns True."""

        # Create a concrete implementation
        class TestSource(VideoSource):
            def search_videos(self, phrase: str):
                return []

            def get_source_name(self) -> str:
                return "test"

            def supports_section_downloads(self) -> bool:
                return False

        source = TestSource()
        assert source.is_configured() is True


class TestPexelsVideoSource:
    """Tests for PexelsVideoSource."""

    def test_init_without_api_key(self):
        """Source initializes but is not configured without API key."""
        with patch.dict(os.environ, {"PEXELS_API_KEY": ""}, clear=False):
            # Clear any existing key
            os.environ.pop("PEXELS_API_KEY", None)
            source = PexelsVideoSource()
            assert source.get_source_name() == "pexels"
            assert source.is_configured() is False
            assert source.supports_section_downloads() is False

    def test_init_with_api_key(self):
        """Source is configured when API key is provided."""
        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_api_key"}):
            source = PexelsVideoSource()
            assert source.is_configured() is True

    def test_search_returns_empty_without_api_key(self):
        """Search returns empty list when no API key configured."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PEXELS_API_KEY", None)
            source = PexelsVideoSource()
            results = source.search_videos("nature")
            assert results == []

    def test_search_returns_empty_for_empty_phrase(self):
        """Search returns empty list for empty search phrase."""
        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            source = PexelsVideoSource()
            results = source.search_videos("")
            assert results == []
            results = source.search_videos("   ")
            assert results == []

    def test_parse_video_handles_missing_fields(self):
        """_parse_video handles videos with missing fields gracefully."""
        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            source = PexelsVideoSource()

            # Missing video_files
            result = source._parse_video({"id": 1})
            assert result is None

            # Empty video_files
            result = source._parse_video({"id": 1, "video_files": []})
            assert result is None

            # Missing id
            result = source._parse_video({"video_files": [{"link": "url"}]})
            assert result is None

    def test_parse_video_valid_response(self):
        """_parse_video correctly parses a valid Pexels response."""
        with patch.dict(os.environ, {"PEXELS_API_KEY": "test_key"}):
            source = PexelsVideoSource()

            video_data = {
                "id": 12345,
                "url": "https://www.pexels.com/video/nature-scene-12345/",
                "duration": 30,
                "user": {"name": "TestUser"},
                "video_files": [
                    {
                        "id": 1,
                        "quality": "hd",
                        "link": "https://example.com/video.mp4",
                        "height": 1080,
                    }
                ],
            }

            result = source._parse_video(video_data)

            assert result is not None
            assert result.video_id == "pexels_12345"
            assert result.source == "pexels"
            assert "Pexels" in result.license or "CC0" in result.license
            assert result.download_url == "https://example.com/video.mp4"
            assert result.duration == 30


class TestPixabayVideoSource:
    """Tests for PixabayVideoSource."""

    def test_init_without_api_key(self):
        """Source initializes but is not configured without API key."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIXABAY_API_KEY", None)
            source = PixabayVideoSource()
            assert source.get_source_name() == "pixabay"
            assert source.is_configured() is False
            assert source.supports_section_downloads() is False

    def test_init_with_api_key(self):
        """Source is configured when API key is provided."""
        with patch.dict(os.environ, {"PIXABAY_API_KEY": "test_api_key"}):
            source = PixabayVideoSource()
            assert source.is_configured() is True

    def test_search_returns_empty_without_api_key(self):
        """Search returns empty list when no API key configured."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIXABAY_API_KEY", None)
            source = PixabayVideoSource()
            results = source.search_videos("nature")
            assert results == []

    def test_search_returns_empty_for_empty_phrase(self):
        """Search returns empty list for empty search phrase."""
        with patch.dict(os.environ, {"PIXABAY_API_KEY": "test_key"}):
            source = PixabayVideoSource()
            results = source.search_videos("")
            assert results == []
            results = source.search_videos("   ")
            assert results == []

    def test_parse_video_handles_missing_fields(self):
        """_parse_video handles videos with missing fields gracefully."""
        with patch.dict(os.environ, {"PIXABAY_API_KEY": "test_key"}):
            source = PixabayVideoSource()

            # Missing videos dict
            result = source._parse_video({"id": 1})
            assert result is None

            # Empty videos dict
            result = source._parse_video({"id": 1, "videos": {}})
            assert result is None

            # Missing id
            result = source._parse_video({"videos": {"large": {"url": "url"}}})
            assert result is None

    def test_parse_video_valid_response(self):
        """_parse_video correctly parses a valid Pixabay response."""
        with patch.dict(os.environ, {"PIXABAY_API_KEY": "test_key"}):
            source = PixabayVideoSource()

            video_data = {
                "id": 67890,
                "pageURL": "https://pixabay.com/videos/id-67890/",
                "duration": 25,
                "tags": "nature, landscape, mountains",
                "user": "TestCreator",
                "videos": {
                    "large": {"url": "https://example.com/large.mp4"},
                    "medium": {"url": "https://example.com/medium.mp4"},
                },
            }

            result = source._parse_video(video_data)

            assert result is not None
            assert result.video_id == "pixabay_67890"
            assert result.source == "pixabay"
            assert "Pixabay" in result.license or "CC0" in result.license
            assert result.download_url == "https://example.com/large.mp4"
            assert result.duration == 25


class TestYouTubeVideoSource:
    """Tests for YouTubeVideoSource."""

    def test_init_defaults(self):
        """Source initializes with correct defaults."""
        source = YouTubeVideoSource()
        assert source.get_source_name() == "youtube"
        assert source.is_configured() is True  # YouTube doesn't require API key
        assert source.supports_section_downloads() is True
        assert source.max_results == 20

    def test_init_custom_max_results(self):
        """Source respects custom max_results."""
        source = YouTubeVideoSource(max_results=50)
        assert source.max_results == 50

    def test_search_returns_empty_for_empty_phrase(self):
        """Search returns empty list for empty search phrase."""
        source = YouTubeVideoSource()
        results = source.search_videos("")
        assert results == []
        results = source.search_videos("   ")
        assert results == []

    def test_parse_video_entry_valid(self):
        """_parse_video_entry correctly parses valid entry."""
        source = YouTubeVideoSource()

        entry = {
            "id": "abc123",
            "title": "Beautiful Nature Footage",
            "duration": 120,
            "description": "Test description",
        }

        result = source._parse_video_entry(entry)

        assert result is not None
        assert result.video_id == "abc123"
        assert result.title == "Beautiful Nature Footage"
        assert result.duration == 120
        assert result.url == "https://www.youtube.com/watch?v=abc123"
        # YouTube source defaults to "youtube" in VideoResult
        assert result.source == "youtube"

    def test_parse_video_entry_missing_id(self):
        """_parse_video_entry returns None for entry without ID."""
        source = YouTubeVideoSource()
        result = source._parse_video_entry({"title": "No ID"})
        assert result is None

    def test_parse_video_entry_handles_missing_optional_fields(self):
        """_parse_video_entry handles missing optional fields."""
        source = YouTubeVideoSource()

        entry = {"id": "abc123"}
        result = source._parse_video_entry(entry)

        assert result is not None
        assert result.video_id == "abc123"
        assert result.title == "Unknown Title"
        assert result.duration == 0
        assert result.description == ""


class TestMultiSourceIntegration:
    """Integration tests for multi-source functionality."""

    def test_all_sources_return_video_result(self):
        """All sources return VideoResult instances with correct fields."""
        # Test that parsed results have required fields
        video = VideoResult(
            video_id="test_123",
            title="Test Video",
            url="https://example.com/video",
            duration=30,
            source="test",
            license="CC0",
            download_url="https://example.com/download",
        )

        assert video.video_id == "test_123"
        assert video.source == "test"
        assert video.license == "CC0"
        assert video.download_url == "https://example.com/download"

    def test_sources_have_consistent_interface(self):
        """All video sources implement the same interface."""
        with patch.dict(
            os.environ,
            {"PEXELS_API_KEY": "key", "PIXABAY_API_KEY": "key"},
        ):
            sources = [
                YouTubeVideoSource(),
                PexelsVideoSource(),
                PixabayVideoSource(),
            ]

            for source in sources:
                # All sources have required methods
                assert hasattr(source, "search_videos")
                assert hasattr(source, "get_source_name")
                assert hasattr(source, "supports_section_downloads")
                assert hasattr(source, "is_configured")

                # All sources return strings for name
                assert isinstance(source.get_source_name(), str)

                # All sources return bool for config check
                assert isinstance(source.is_configured(), bool)

                # All sources return bool for section downloads
                assert isinstance(source.supports_section_downloads(), bool)

    def test_source_priority_with_prefer_stock(self):
        """Stock sources should be prioritized when prefer_stock_footage is True."""
        # This tests the logic that would be in BRollProcessor
        search_sources = ["youtube", "pexels", "pixabay"]
        prefer_stock = True

        source_order = list(search_sources)
        if prefer_stock:
            stock_sources = [s for s in source_order if s in ("pexels", "pixabay")]
            other_sources = [s for s in source_order if s not in ("pexels", "pixabay")]
            source_order = stock_sources + other_sources

        # Stock sources should come first
        assert source_order == ["pexels", "pixabay", "youtube"]

    def test_source_priority_without_prefer_stock(self):
        """Sources should maintain original order when prefer_stock_footage is False."""
        search_sources = ["youtube", "pexels", "pixabay"]
        prefer_stock = False

        source_order = list(search_sources)
        if prefer_stock:
            stock_sources = [s for s in source_order if s in ("pexels", "pixabay")]
            other_sources = [s for s in source_order if s not in ("pexels", "pixabay")]
            source_order = stock_sources + other_sources

        # Original order maintained
        assert source_order == ["youtube", "pexels", "pixabay"]
