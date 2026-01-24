"""Unit tests for ParallelDownloader class."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from services.video_downloader import (
    DownloadProgress,
    ParallelDownloader,
    VideoDownloader,
)


class TestDownloadProgress:
    """Test DownloadProgress dataclass."""

    def test_initial_state(self):
        """Test initial progress state."""
        progress = DownloadProgress(total=5)
        assert progress.total == 5
        assert progress.completed == 0
        assert progress.failed == 0
        assert progress.in_progress == 0
        assert progress.success_rate == 0.0

    def test_start_download(self):
        """Test starting a download."""
        progress = DownloadProgress(total=5)
        progress.start_download()
        assert progress.in_progress == 1

    def test_complete_download(self):
        """Test completing a download."""
        progress = DownloadProgress(total=5)
        progress.start_download()
        progress.complete_download("/path/to/file.mp4")

        assert progress.in_progress == 0
        assert progress.completed == 1
        assert "/path/to/file.mp4" in progress.completed_files
        assert progress.success_rate == 100.0

    def test_fail_download(self):
        """Test failing a download."""
        progress = DownloadProgress(total=5)
        progress.start_download()
        progress.fail_download("https://youtube.com/watch?v=abc123")

        assert progress.in_progress == 0
        assert progress.failed == 1
        assert "https://youtube.com/watch?v=abc123" in progress.failed_urls
        assert progress.success_rate == 0.0

    def test_success_rate_mixed(self):
        """Test success rate with mixed results."""
        progress = DownloadProgress(total=5)

        # Complete 3, fail 2
        for i in range(3):
            progress.start_download()
            progress.complete_download(f"/path/file{i}.mp4")

        for i in range(2):
            progress.start_download()
            progress.fail_download(f"https://url{i}")

        assert progress.success_rate == 60.0

    def test_status_message(self):
        """Test status message generation."""
        progress = DownloadProgress(total=5)
        progress.start_download()
        progress.complete_download("/path/file.mp4")

        status = progress.get_status_message()
        assert "1/5 completed" in status
        assert "0 in progress" in status
        assert "0 failed" in status

    def test_elapsed_time(self):
        """Test elapsed time tracking."""
        progress = DownloadProgress(total=1)
        # Should have some small elapsed time
        assert progress.elapsed_time >= 0


class TestParallelDownloader:
    """Test ParallelDownloader class."""

    @pytest.fixture
    def mock_video_downloader(self):
        """Create a mock VideoDownloader."""
        mock = Mock(spec=VideoDownloader)
        mock.download_preview = Mock(return_value="/path/preview.mp4")
        mock.download_clip_sections = Mock(return_value=["/path/clip1.mp4"])
        mock.download_single_video_to_folder = Mock(return_value="/path/full.mp4")
        return mock

    @pytest.fixture
    def mock_scored_video(self):
        """Create a mock ScoredVideo."""
        mock = Mock()
        mock.video_id = "abc123"
        mock.video_result = Mock()
        mock.video_result.url = "https://youtube.com/watch?v=abc123"
        mock.score = 8
        return mock

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {
                "max_concurrent_downloads": 5,
                "download_timeout_seconds": 120,
                "download_stagger_delay": 0.5,
            }
            downloader = ParallelDownloader()

            assert downloader.max_concurrent == 5
            assert downloader.timeout == 120
            assert downloader.stagger_delay == 0.5

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=3,
                timeout=60,
                stagger_delay=1.0,
            )

            assert downloader.max_concurrent == 3
            assert downloader.timeout == 60
            assert downloader.stagger_delay == 1.0

    @pytest.mark.asyncio
    async def test_download_many_previews_empty_list(self):
        """Test downloading with empty video list."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(max_concurrent=5)
            results = await downloader.download_many_previews([], Path("/output"))

            assert results == []

    @pytest.mark.asyncio
    async def test_download_many_previews_success(
        self, mock_video_downloader, mock_scored_video, temp_dir
    ):
        """Test successful parallel preview downloads."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=2,
                timeout=10,
                stagger_delay=0.01,  # Minimal delay for tests
                video_downloader=mock_video_downloader,
            )

            videos = [mock_scored_video, mock_scored_video]
            results = await downloader.download_many_previews(videos, temp_dir)

            assert len(results) == 2
            assert mock_video_downloader.download_preview.call_count == 2

    @pytest.mark.asyncio
    async def test_download_many_previews_with_callback(
        self, mock_video_downloader, mock_scored_video, temp_dir
    ):
        """Test progress callback is called during downloads."""
        callback_calls = []

        def progress_callback(progress: DownloadProgress):
            callback_calls.append(progress.get_status_message())

        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=1,
                timeout=10,
                stagger_delay=0.01,
                video_downloader=mock_video_downloader,
            )

            await downloader.download_many_previews(
                [mock_scored_video],
                temp_dir,
                progress_callback=progress_callback,
            )

            assert len(callback_calls) >= 1

    @pytest.mark.asyncio
    async def test_download_many_previews_timeout(
        self, mock_scored_video, temp_dir
    ):
        """Test timeout handling during downloads."""
        # Create a mock that takes too long
        slow_downloader = Mock(spec=VideoDownloader)

        def slow_download(*args, **kwargs):
            import time
            time.sleep(2)  # Sleep longer than timeout
            return "/path/preview.mp4"

        slow_downloader.download_preview = slow_download

        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=1,
                timeout=0.1,  # Very short timeout
                stagger_delay=0.01,
                video_downloader=slow_downloader,
            )

            results = await downloader.download_many_previews(
                [mock_scored_video],
                temp_dir,
            )

            # Should return empty due to timeout
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_download_many_previews_failure(
        self, mock_scored_video, temp_dir
    ):
        """Test handling download failures."""
        failing_downloader = Mock(spec=VideoDownloader)
        failing_downloader.download_preview = Mock(return_value=None)

        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=1,
                timeout=10,
                stagger_delay=0.01,
                video_downloader=failing_downloader,
            )

            results = await downloader.download_many_previews(
                [mock_scored_video],
                temp_dir,
            )

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_download_many_clips_empty_list(self):
        """Test downloading clips with empty list."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(max_concurrent=5)
            results = await downloader.download_many_clips([])

            assert results == []

    @pytest.mark.asyncio
    async def test_download_many_clips_success(
        self, mock_video_downloader, mock_scored_video, temp_dir
    ):
        """Test successful parallel clip downloads."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=2,
                timeout=10,
                stagger_delay=0.01,
                video_downloader=mock_video_downloader,
            )

            # Create mock segments
            mock_segment = Mock()
            mock_segment.start_time = 5.0
            mock_segment.end_time = 10.0

            videos_with_segments = [
                (mock_scored_video, [mock_segment], str(temp_dir)),
            ]

            results = await downloader.download_many_clips(videos_with_segments)

            assert len(results) == 1
            assert mock_video_downloader.download_clip_sections.call_count == 1

    @pytest.mark.asyncio
    async def test_download_many_full_videos_success(
        self, mock_video_downloader, mock_scored_video, temp_dir
    ):
        """Test successful parallel full video downloads."""
        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=2,
                timeout=10,
                stagger_delay=0.01,
                video_downloader=mock_video_downloader,
            )

            results = await downloader.download_many_full_videos(
                [mock_scored_video],
                temp_dir,
            )

            assert len(results) == 1
            assert mock_video_downloader.download_single_video_to_folder.call_count == 1

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(
        self, mock_scored_video, temp_dir
    ):
        """Test that semaphore properly limits concurrent downloads."""
        concurrent_count = 0
        max_concurrent_seen = 0

        def track_concurrency(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            import time
            time.sleep(0.05)
            concurrent_count -= 1
            return "/path/preview.mp4"

        tracking_downloader = Mock(spec=VideoDownloader)
        tracking_downloader.download_preview = track_concurrency

        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=2,  # Limit to 2 concurrent
                timeout=10,
                stagger_delay=0.01,
                video_downloader=tracking_downloader,
            )

            # Try to download 5 videos
            videos = [mock_scored_video for _ in range(5)]
            await downloader.download_many_previews(videos, temp_dir)

            # Should never exceed 2 concurrent
            assert max_concurrent_seen <= 2

    @pytest.mark.asyncio
    async def test_stagger_delay_applied(
        self, mock_video_downloader, mock_scored_video, temp_dir
    ):
        """Test that stagger delay is applied between downloads."""
        import time

        start_times = []

        def record_start_time(*args, **kwargs):
            start_times.append(time.time())
            return "/path/preview.mp4"

        mock_video_downloader.download_preview = record_start_time

        with patch("services.video_downloader.load_config") as mock_config:
            mock_config.return_value = {}
            downloader = ParallelDownloader(
                max_concurrent=10,  # High limit so stagger is the constraint
                timeout=10,
                stagger_delay=0.1,  # 100ms stagger
                video_downloader=mock_video_downloader,
            )

            videos = [mock_scored_video for _ in range(3)]
            await downloader.download_many_previews(videos, temp_dir)

            # Check that there's at least some delay between starts
            if len(start_times) >= 2:
                delays = [
                    start_times[i + 1] - start_times[i]
                    for i in range(len(start_times) - 1)
                ]
                # At least one delay should be close to stagger_delay
                assert any(d >= 0.05 for d in delays)
