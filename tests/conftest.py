"""Shared pytest fixtures for stockpile tests."""

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Dict, Generator
from unittest.mock import Mock

import pytest

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config() -> Dict:
    """Sample configuration for testing."""
    return {
        "gemini_api_key": "test_gemini_key",
        "openai_api_key": "test_openai_key",
        "whisper_model": "base",
        "gemini_model": "gemini-3-flash-preview",
        "local_input_folder": "input",
        "local_output_folder": "output",
        "max_videos_per_phrase": 3,
        "max_video_duration_seconds": 900,
        "clips_per_minute": 2,
        "clip_extraction_enabled": True,
        "min_clip_duration": 4.0,
        "max_clip_duration": 15.0,
        "max_clips_per_video": 3,
        "delete_original_after_extraction": True,
        "use_two_pass_download": True,
        "competitive_analysis_enabled": True,
        "previews_per_need": 2,
        "clips_per_need_target": 1,
    }


@pytest.fixture
def mock_ai_service():
    """Mock AIService for testing."""
    mock = Mock()
    mock.plan_broll_needs = Mock()
    mock.evaluate_videos = Mock()
    return mock


@pytest.fixture
def mock_video_downloader():
    """Mock VideoDownloader for testing."""
    mock = Mock()
    mock.download_videos = Mock(return_value=[])
    mock.download_single_video_to_folder = Mock(return_value="/path/to/video.mp4")
    return mock


@pytest.fixture
def mock_clip_extractor():
    """Mock ClipExtractor for testing."""
    mock = Mock()
    mock.process_downloaded_video = Mock(return_value=([], False))
    mock.analyze_video = Mock()
    mock.extract_clip = Mock()
    return mock


@pytest.fixture
def sample_transcript() -> str:
    """Sample video transcript for testing."""
    return """
    [0:00] Welcome to this video about AI and automation.
    [0:15] Today we'll discuss how artificial intelligence is changing content creation.
    [0:30] Let's start by looking at some real-world examples of AI tools.
    [0:45] Many creators are now using AI for video editing and enhancement.
    [1:00] This technology can save hours of manual work.
    """


@pytest.fixture
def sample_broll_plan():
    """Sample B-roll plan for integration testing."""
    from models.broll_need import BRollNeed, BRollPlan

    return BRollPlan(
        source_duration=60.0,
        needs=[
            BRollNeed(
                timestamp=10.0,
                search_phrase="city skyline aerial",
                description="Aerial view of city skyline",
                context="urban development discussion",
            ),
        ],
        clips_per_minute=2.0,
    )


@pytest.fixture
def sample_video_results():
    """Sample video results for integration testing."""
    from models.video import VideoResult

    return [
        VideoResult(
            video_id="vid_001",
            title="City Skyline Drone Footage",
            url="https://youtube.com/watch?v=vid001",
            duration=45,
            description="Beautiful aerial drone footage of city skyline at sunset",
        ),
    ]


@pytest.fixture
def sample_transcript_result():
    """Sample TranscriptResult for integration testing."""
    from models.broll_need import TranscriptResult, TranscriptSegment

    segments = [
        TranscriptSegment(
            start=0.0, end=15.0, text="Welcome to this video about AI and automation."
        ),
        TranscriptSegment(
            start=15.0,
            end=30.0,
            text="Today we'll discuss how artificial intelligence is changing content creation.",
        ),
        TranscriptSegment(
            start=30.0,
            end=45.0,
            text="Let's start by looking at some real-world examples.",
        ),
        TranscriptSegment(
            start=45.0,
            end=60.0,
            text="Many creators are now using AI for video editing and enhancement.",
        ),
    ]

    return TranscriptResult(
        text="Welcome to this video about AI and automation. Today we'll discuss how artificial intelligence is changing content creation. Let's start by looking at some real-world examples. Many creators are now using AI for video editing and enhancement.",
        segments=segments,
        duration=60.0,
        language="en",
    )


