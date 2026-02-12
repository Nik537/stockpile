"""Unit tests for YouTube B-roll integration in the video agent.

Tests the YouTube-first B-roll acquisition pipeline:
- YouTubeVideoSource enhancements (search_broll, search_videos_async, is_configured)
- Video agent source priority (YouTube -> Pexels -> Pixabay)
- YouTube download via yt-dlp at 720p
- Search query enhancement with visual style and "stock footage" suffix
- Max duration filtering
- Graceful fallback when YouTube fails
- Config options (YOUTUBE_BROLL_ENABLED, YOUTUBE_BROLL_MAX_DURATION, BROLL_SOURCE_PRIORITY)
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from models.video import VideoResult
from services.video_sources.youtube import YouTubeVideoSource, video_filter


# ---------------------------------------------------------------------------
# YouTubeVideoSource unit tests
# ---------------------------------------------------------------------------


class TestYouTubeVideoSourceBroll:
    """Tests for YouTubeVideoSource B-roll search enhancements."""

    def test_search_broll_enhances_query_with_style(self):
        """search_broll adds visual_style and stock footage terms to the query."""
        source = YouTubeVideoSource(max_results=5)

        with patch.object(source, "search_videos") as mock_search:
            mock_search.return_value = []
            source.search_broll(
                keywords=["city", "skyline"],
                visual_style="aerial",
                max_results=5,
            )

            # Should have called search_videos with enhanced query
            call_args = mock_search.call_args[0][0]
            assert "city skyline" in call_args
            assert "aerial" in call_args
            assert "stock footage" in call_args
            assert "b-roll" in call_args

    def test_search_broll_without_style(self):
        """search_broll works without visual_style."""
        source = YouTubeVideoSource(max_results=5)

        with patch.object(source, "search_videos") as mock_search:
            mock_search.return_value = []
            source.search_broll(
                keywords=["nature", "forest"],
                visual_style="",
                max_results=3,
            )

            call_args = mock_search.call_args[0][0]
            assert "nature forest" in call_args
            assert "stock footage b-roll" in call_args

    def test_search_broll_limits_keywords_to_three(self):
        """search_broll only uses first 3 keywords."""
        source = YouTubeVideoSource(max_results=5)

        with patch.object(source, "search_videos") as mock_search:
            mock_search.return_value = []
            source.search_broll(
                keywords=["alpha", "beta", "gamma", "delta", "epsilon"],
                max_results=3,
            )

            call_args = mock_search.call_args[0][0]
            assert "alpha beta gamma" in call_args
            assert "delta" not in call_args
            assert "epsilon" not in call_args

    def test_search_broll_respects_max_results(self):
        """search_broll temporarily overrides max_results and restores it."""
        source = YouTubeVideoSource(max_results=20)

        with patch.object(source, "search_videos") as mock_search:
            mock_search.return_value = []
            source.search_broll(keywords=["test"], max_results=3)

        # max_results should be restored to original
        assert source.max_results == 20

    @pytest.mark.asyncio
    async def test_search_broll_async(self):
        """search_broll_async runs search_broll in thread pool."""
        source = YouTubeVideoSource(max_results=5)
        expected = [
            VideoResult(
                video_id="yt1", title="Test", url="https://youtube.com/watch?v=yt1",
                duration=30, source="youtube"
            )
        ]

        with patch.object(source, "search_broll", return_value=expected):
            results = await source.search_broll_async(
                keywords=["test"], visual_style="cinematic", max_results=3
            )
            assert results == expected

    @pytest.mark.asyncio
    async def test_search_videos_async(self):
        """search_videos_async runs search_videos in thread pool."""
        source = YouTubeVideoSource(max_results=5)
        expected = [
            VideoResult(
                video_id="yt2", title="Test2", url="https://youtube.com/watch?v=yt2",
                duration=45, source="youtube"
            )
        ]

        with patch.object(source, "search_videos", return_value=expected):
            results = await source.search_videos_async("city skyline")
            assert results == expected

    @pytest.mark.asyncio
    async def test_search_videos_async_empty_phrase(self):
        """search_videos_async returns empty for empty phrase."""
        source = YouTubeVideoSource()
        results = await source.search_videos_async("")
        assert results == []
        results = await source.search_videos_async("   ")
        assert results == []


class TestYouTubeIsConfigured:
    """Tests for YouTubeVideoSource.is_configured()."""

    def test_is_configured_when_enabled(self):
        """Returns True when YOUTUBE_BROLL_ENABLED is true."""
        with patch("services.video_sources.youtube.load_config") as mock_cfg:
            mock_cfg.return_value = {"youtube_broll_enabled": True}
            source = YouTubeVideoSource()
            assert source.is_configured() is True

    def test_not_configured_when_disabled(self):
        """Returns False when YOUTUBE_BROLL_ENABLED is false."""
        with patch("services.video_sources.youtube.load_config") as mock_cfg:
            mock_cfg.return_value = {"youtube_broll_enabled": False}
            source = YouTubeVideoSource()
            assert source.is_configured() is False

    def test_defaults_to_enabled_on_config_error(self):
        """Returns True if config loading fails."""
        with patch("services.video_sources.youtube.load_config", side_effect=Exception("fail")):
            source = YouTubeVideoSource()
            assert source.is_configured() is True


class TestVideoFilterDuration:
    """Tests for the video_filter function with max_duration override."""

    def test_filter_allows_short_videos(self):
        """Videos under max_duration pass the filter."""
        result = video_filter({"duration": 30}, max_duration=60)
        assert result is None  # None means passes

    def test_filter_rejects_long_videos(self):
        """Videos over max_duration are filtered."""
        result = video_filter({"duration": 120}, max_duration=60)
        assert result is not None
        assert "exceeds" in result

    def test_filter_allows_exactly_max_duration(self):
        """Videos exactly at max_duration pass."""
        result = video_filter({"duration": 60}, max_duration=60)
        assert result is None

    def test_filter_handles_none_duration(self):
        """Videos without duration info pass."""
        result = video_filter({}, max_duration=60)
        assert result is None

    def test_filter_rejects_large_files(self):
        """Videos exceeding size limit are filtered."""
        result = video_filter({"filesize": 200 * 1024 * 1024}, max_duration=600)
        assert result is not None
        assert "Size" in result


class TestYouTubeMaxDuration:
    """Tests for max_duration constructor parameter."""

    def test_max_duration_passed_to_filter(self):
        """YouTubeVideoSource passes max_duration to video_filter."""
        source = YouTubeVideoSource(max_results=5, max_duration=30)
        assert source.max_duration == 30

    def test_max_duration_defaults_to_none(self):
        """max_duration defaults to None (uses config)."""
        source = YouTubeVideoSource(max_results=5)
        assert source.max_duration is None


class TestYouTubeParseVideoEntry:
    """Tests for _parse_video_entry with new fields."""

    def test_includes_source_field(self):
        """Parsed results include source='youtube'."""
        source = YouTubeVideoSource()
        result = source._parse_video_entry({
            "id": "abc123",
            "title": "Test Video",
            "duration": 30,
        })
        assert result is not None
        assert result.source == "youtube"

    def test_includes_view_count(self):
        """Parsed results include view_count from yt-dlp data."""
        source = YouTubeVideoSource()
        result = source._parse_video_entry({
            "id": "abc123",
            "title": "Test Video",
            "duration": 30,
            "view_count": 50000,
        })
        assert result is not None
        assert result.view_count == 50000

    def test_includes_channel(self):
        """Parsed results include channel name."""
        source = YouTubeVideoSource()
        result = source._parse_video_entry({
            "id": "abc123",
            "title": "Test Video",
            "duration": 30,
            "channel": "TestChannel",
        })
        assert result is not None
        assert result.channel == "TestChannel"

    def test_falls_back_to_uploader(self):
        """Uses uploader field when channel is not available."""
        source = YouTubeVideoSource()
        result = source._parse_video_entry({
            "id": "abc123",
            "title": "Test Video",
            "duration": 30,
            "uploader": "UploaderName",
        })
        assert result is not None
        assert result.channel == "UploaderName"


# ---------------------------------------------------------------------------
# Video Agent source priority tests
# ---------------------------------------------------------------------------


class TestAgentSourcePriority:
    """Tests for VideoProductionAgent source priority configuration."""

    def _make_config(self, **overrides):
        """Create a minimal config for agent testing."""
        base = {
            "gemini_api_key": "test_key",
            "gemini_model": "gemini-3-flash-preview",
            "local_output_folder": "/tmp/test_output",
            "youtube_broll_enabled": True,
            "youtube_broll_max_duration": 60,
            "broll_source_priority": ["youtube", "pexels", "pixabay"],
        }
        base.update(overrides)
        return base

    @patch("video_agent.agent.TTSService")
    @patch("video_agent.agent.AIService")
    @patch("video_agent.agent.MusicService")
    @patch("video_agent.agent.ImageGenerationService")
    @patch("video_agent.agent.ScriptGenerator")
    def test_default_priority_youtube_first(self, *mocks):
        """Default priority places YouTube as first source."""
        from video_agent.agent import VideoProductionAgent

        config = self._make_config()
        agent = VideoProductionAgent(config=config)

        source_names = [s.get_source_name() for s in agent.video_sources]
        assert source_names == ["youtube", "pexels", "pixabay"]

    @patch("video_agent.agent.TTSService")
    @patch("video_agent.agent.AIService")
    @patch("video_agent.agent.MusicService")
    @patch("video_agent.agent.ImageGenerationService")
    @patch("video_agent.agent.ScriptGenerator")
    def test_custom_priority_pexels_first(self, *mocks):
        """Custom priority can place Pexels first."""
        from video_agent.agent import VideoProductionAgent

        config = self._make_config(broll_source_priority=["pexels", "youtube", "pixabay"])
        agent = VideoProductionAgent(config=config)

        source_names = [s.get_source_name() for s in agent.video_sources]
        assert source_names == ["pexels", "youtube", "pixabay"]

    @patch("video_agent.agent.TTSService")
    @patch("video_agent.agent.AIService")
    @patch("video_agent.agent.MusicService")
    @patch("video_agent.agent.ImageGenerationService")
    @patch("video_agent.agent.ScriptGenerator")
    def test_empty_priority_fallback(self, *mocks):
        """Empty priority list falls back to Pexels + Pixabay."""
        from video_agent.agent import VideoProductionAgent

        config = self._make_config(broll_source_priority=[])
        agent = VideoProductionAgent(config=config)

        source_names = [s.get_source_name() for s in agent.video_sources]
        assert source_names == ["pexels", "pixabay"]

    @patch("video_agent.agent.TTSService")
    @patch("video_agent.agent.AIService")
    @patch("video_agent.agent.MusicService")
    @patch("video_agent.agent.ImageGenerationService")
    @patch("video_agent.agent.ScriptGenerator")
    def test_youtube_max_duration_from_config(self, *mocks):
        """YouTube source gets max_duration from config."""
        from video_agent.agent import VideoProductionAgent

        config = self._make_config(youtube_broll_max_duration=30)
        agent = VideoProductionAgent(config=config)

        youtube_sources = [s for s in agent.video_sources if s.get_source_name() == "youtube"]
        assert len(youtube_sources) == 1
        assert youtube_sources[0].max_duration == 30


# ---------------------------------------------------------------------------
# Video Agent download and fallback tests
# ---------------------------------------------------------------------------


class TestAgentBrollDownload:
    """Tests for _search_and_download_broll with YouTube integration."""

    def _make_agent(self, config_overrides=None):
        """Create an agent with mocked services."""
        config = {
            "gemini_api_key": "test_key",
            "gemini_model": "gemini-3-flash-preview",
            "local_output_folder": "/tmp/test_output",
            "youtube_broll_enabled": True,
            "youtube_broll_max_duration": 60,
            "broll_source_priority": ["youtube", "pexels", "pixabay"],
        }
        if config_overrides:
            config.update(config_overrides)

        with patch("video_agent.agent.TTSService"), \
             patch("video_agent.agent.AIService"), \
             patch("video_agent.agent.MusicService"), \
             patch("video_agent.agent.ImageGenerationService"), \
             patch("video_agent.agent.ScriptGenerator"):
            from video_agent.agent import VideoProductionAgent
            return VideoProductionAgent(config=config)

    @pytest.mark.asyncio
    async def test_youtube_searched_first(self):
        """YouTube is searched before Pexels/Pixabay."""
        agent = self._make_agent()

        search_order = []

        # Mock YouTube source
        yt_source = agent.video_sources[0]  # YouTube is first
        assert yt_source.get_source_name() == "youtube"

        async def mock_yt_search(*args, **kwargs):
            search_order.append("youtube")
            return []  # Return empty to trigger fallback

        # Mock Pexels source
        pexels_source = agent.video_sources[1]
        async def mock_pexels_search(*args, **kwargs):
            search_order.append("pexels")
            return []

        # Mock Pixabay source
        pixabay_source = agent.video_sources[2]
        async def mock_pixabay_search(*args, **kwargs):
            search_order.append("pixabay")
            return []

        with patch.object(yt_source, "search_broll_async", side_effect=mock_yt_search), \
             patch.object(yt_source, "is_configured", return_value=True), \
             patch.object(pexels_source, "search_videos_async", side_effect=mock_pexels_search), \
             patch.object(pexels_source, "is_configured", return_value=True), \
             patch.object(pixabay_source, "search_videos_async", side_effect=mock_pixabay_search), \
             patch.object(pixabay_source, "is_configured", return_value=True):

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "scene_001.mp4"
                await agent._search_and_download_broll(
                    keywords=["city", "skyline"],
                    output_path=output_path,
                    visual_style="aerial",
                )

        # YouTube should have been searched first
        assert search_order[0] == "youtube"

    @pytest.mark.asyncio
    async def test_fallback_to_pexels_on_youtube_failure(self):
        """Falls back to Pexels when YouTube search fails."""
        agent = self._make_agent()

        yt_source = agent.video_sources[0]
        pexels_source = agent.video_sources[1]

        pexels_result = VideoResult(
            video_id="pexels_123", title="City", url="https://pexels.com/123",
            duration=15, source="pexels",
            download_url="https://example.com/video.mp4",
        )

        with patch.object(yt_source, "search_broll_async", side_effect=Exception("YouTube down")), \
             patch.object(yt_source, "is_configured", return_value=True), \
             patch.object(pexels_source, "search_videos_async", return_value=[pexels_result]), \
             patch.object(pexels_source, "is_configured", return_value=True):

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "scene_001.mp4"

                # Mock httpx at the module level (it's imported inside the method)
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b"x" * 2048  # Fake video data
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)

                with patch("httpx.AsyncClient", return_value=mock_client):
                    result = await agent._search_and_download_broll(
                        keywords=["city", "skyline"],
                        output_path=output_path,
                    )

                    assert result == output_path

    @pytest.mark.asyncio
    async def test_skips_unconfigured_sources(self):
        """Skips sources that are not configured."""
        agent = self._make_agent()

        yt_source = agent.video_sources[0]
        pexels_source = agent.video_sources[1]
        pixabay_source = agent.video_sources[2]

        with patch.object(yt_source, "is_configured", return_value=False), \
             patch.object(pexels_source, "is_configured", return_value=False), \
             patch.object(pixabay_source, "is_configured", return_value=False):

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "scene_001.mp4"
                result = await agent._search_and_download_broll(
                    keywords=["test"], output_path=output_path,
                )
                assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_sources_empty(self):
        """Returns None when no source has results."""
        agent = self._make_agent()

        for source in agent.video_sources:
            if hasattr(source, "search_broll_async"):
                with patch.object(source, "search_broll_async", return_value=[]):
                    pass
            if hasattr(source, "search_videos_async"):
                with patch.object(source, "search_videos_async", return_value=[]):
                    pass

        # Patch all sources to return empty
        patches = []
        for source in agent.video_sources:
            patches.append(patch.object(source, "is_configured", return_value=True))
            if source.get_source_name() == "youtube":
                patches.append(patch.object(source, "search_broll_async", return_value=[]))
            else:
                patches.append(patch.object(source, "search_videos_async", return_value=[]))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene_001.mp4"
            for p in patches:
                p.start()
            try:
                result = await agent._search_and_download_broll(
                    keywords=["nonexistent"], output_path=output_path,
                )
                assert result is None
            finally:
                for p in patches:
                    p.stop()


class TestAgentYouTubeDownload:
    """Tests for _download_youtube_broll method."""

    def _make_agent(self):
        """Create an agent with mocked services."""
        config = {
            "gemini_api_key": "test_key",
            "gemini_model": "gemini-3-flash-preview",
            "local_output_folder": "/tmp/test_output",
            "youtube_broll_enabled": True,
            "youtube_broll_max_duration": 60,
            "broll_source_priority": ["youtube", "pexels", "pixabay"],
        }
        with patch("video_agent.agent.TTSService"), \
             patch("video_agent.agent.AIService"), \
             patch("video_agent.agent.MusicService"), \
             patch("video_agent.agent.ImageGenerationService"), \
             patch("video_agent.agent.ScriptGenerator"):
            from video_agent.agent import VideoProductionAgent
            return VideoProductionAgent(config=config)

    @pytest.mark.asyncio
    async def test_download_youtube_broll_success(self):
        """Successful YouTube download returns True."""
        agent = self._make_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene_001.mp4"

            # Mock _ytdlp_download to "succeed" and create the file
            def mock_download(url, opts):
                # Simulate yt-dlp creating the file
                output_path.write_bytes(b"x" * 2048)
                return True

            with patch.object(agent, "_ytdlp_download", side_effect=mock_download):
                result = await agent._download_youtube_broll(
                    url="https://youtube.com/watch?v=test123",
                    output_path=output_path,
                )
                assert result is True
                assert output_path.exists()

    @pytest.mark.asyncio
    async def test_download_youtube_broll_failure(self):
        """Failed YouTube download returns False."""
        agent = self._make_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene_001.mp4"

            with patch.object(agent, "_ytdlp_download", return_value=False):
                result = await agent._download_youtube_broll(
                    url="https://youtube.com/watch?v=test123",
                    output_path=output_path,
                )
                assert result is False

    @pytest.mark.asyncio
    async def test_download_youtube_broll_exception(self):
        """Exception in YouTube download returns False (no crash)."""
        agent = self._make_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene_001.mp4"

            with patch.object(agent, "_ytdlp_download", side_effect=Exception("network error")):
                result = await agent._download_youtube_broll(
                    url="https://youtube.com/watch?v=test123",
                    output_path=output_path,
                )
                assert result is False

    @pytest.mark.asyncio
    async def test_download_youtube_broll_empty_file(self):
        """YouTube download that produces empty file returns False."""
        agent = self._make_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scene_001.mp4"

            # Mock to create a tiny file (< 1KB threshold)
            def mock_download(url, opts):
                output_path.write_bytes(b"x" * 100)
                return True

            with patch.object(agent, "_ytdlp_download", side_effect=mock_download):
                result = await agent._download_youtube_broll(
                    url="https://youtube.com/watch?v=test123",
                    output_path=output_path,
                )
                assert result is False

    def test_ytdlp_download_static_method(self):
        """_ytdlp_download calls yt-dlp correctly."""
        from video_agent.agent import VideoProductionAgent

        mock_ydl = MagicMock()
        mock_ydl_ctx = MagicMock()
        mock_ydl_ctx.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_ctx.__exit__ = MagicMock(return_value=False)

        with patch("yt_dlp.YoutubeDL", return_value=mock_ydl_ctx):
            result = VideoProductionAgent._ytdlp_download(
                "https://youtube.com/watch?v=test",
                {"format": "best", "quiet": True},
            )

            assert result is True
            mock_ydl.download.assert_called_once_with(["https://youtube.com/watch?v=test"])

    def test_ytdlp_download_handles_exception(self):
        """_ytdlp_download returns False on exception."""
        from video_agent.agent import VideoProductionAgent

        with patch("yt_dlp.YoutubeDL", side_effect=Exception("yt-dlp crash")):
            result = VideoProductionAgent._ytdlp_download(
                "https://youtube.com/watch?v=test",
                {"format": "best"},
            )

            assert result is False


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestBrollConfig:
    """Tests for YouTube B-roll configuration options."""

    def test_config_youtube_broll_enabled_default(self):
        """YOUTUBE_BROLL_ENABLED defaults to True."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOUTUBE_BROLL_ENABLED", None)
            from utils.config import load_config
            config = load_config()
            assert config["youtube_broll_enabled"] is True

    def test_config_youtube_broll_disabled(self):
        """YOUTUBE_BROLL_ENABLED=false disables YouTube."""
        with patch.dict(os.environ, {"YOUTUBE_BROLL_ENABLED": "false"}):
            from utils.config import load_config
            config = load_config()
            assert config["youtube_broll_enabled"] is False

    def test_config_youtube_broll_max_duration_default(self):
        """YOUTUBE_BROLL_MAX_DURATION defaults to 60."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YOUTUBE_BROLL_MAX_DURATION", None)
            from utils.config import load_config
            config = load_config()
            assert config["youtube_broll_max_duration"] == 60

    def test_config_youtube_broll_max_duration_custom(self):
        """YOUTUBE_BROLL_MAX_DURATION can be overridden."""
        with patch.dict(os.environ, {"YOUTUBE_BROLL_MAX_DURATION": "30"}):
            from utils.config import load_config
            config = load_config()
            assert config["youtube_broll_max_duration"] == 30

    def test_config_broll_source_priority_default(self):
        """BROLL_SOURCE_PRIORITY defaults to youtube,pexels,pixabay."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BROLL_SOURCE_PRIORITY", None)
            from utils.config import load_config
            config = load_config()
            assert config["broll_source_priority"] == ["youtube", "pexels", "pixabay"]

    def test_config_broll_source_priority_custom(self):
        """BROLL_SOURCE_PRIORITY can be reordered."""
        with patch.dict(os.environ, {"BROLL_SOURCE_PRIORITY": "pexels,pixabay"}):
            from utils.config import load_config
            config = load_config()
            assert config["broll_source_priority"] == ["pexels", "pixabay"]
