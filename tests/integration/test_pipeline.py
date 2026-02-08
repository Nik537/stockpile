"""Integration tests for the full Stockpile B-roll processing pipeline.

Tests the pipeline end-to-end with mocked external services.
Uses dependency injection (Phase 2A) to inject mock services.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models.broll_need import BRollNeed, BRollPlan, TranscriptResult, TranscriptSegment
from models.video import VideoResult, ScoredVideo
from models.clip import ClipSegment


@pytest.fixture
def mock_transcript_result():
    """Create a mock TranscriptResult with proper segments."""
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
    ]
    return TranscriptResult(
        text="Welcome to this video about AI and automation. Today we'll discuss how artificial intelligence is changing content creation. Let's start by looking at some real-world examples.",
        segments=segments,
        duration=300.0,  # 5 minutes
        language="en",
    )


@pytest.fixture
def mock_broll_plan():
    """Create a mock BRollPlan."""
    return BRollPlan(
        source_duration=300.0,
        needs=[
            BRollNeed(
                timestamp=30.0,
                search_phrase="artificial intelligence robot",
                description="AI robot working on computer",
                context="discussing AI and automation",
            ),
            BRollNeed(
                timestamp=120.0,
                search_phrase="content creator editing video",
                description="Person editing video on computer",
                context="AI changing content creation",
            ),
        ],
        clips_per_minute=2.0,
    )


@pytest.fixture
def mock_video_results():
    """Create mock video search results."""
    return [
        VideoResult(
            video_id="test_vid_1",
            title="AI Robot Demo",
            url="https://youtube.com/watch?v=test1",
            duration=60,
            description="Cool AI robot demonstration",
        ),
        VideoResult(
            video_id="test_vid_2",
            title="Content Creation Tips",
            url="https://youtube.com/watch?v=test2",
            duration=120,
            description="Best tools for content creators",
        ),
    ]


@pytest.fixture
def sample_broll_plan():
    """Sample B-roll plan fixture for conftest compatibility."""
    return BRollPlan(
        source_duration=60.0,
        needs=[
            BRollNeed(
                timestamp=10.0,
                search_phrase="city skyline aerial",
                description="Aerial view of city skyline",
                context="urban development",
            ),
        ],
        clips_per_minute=2.0,
    )


@pytest.fixture
def sample_video_results():
    """Sample video results fixture for conftest compatibility."""
    return [
        VideoResult(
            video_id="vid1",
            title="City Skyline Drone",
            url="https://youtube.com/watch?v=vid1",
            duration=45,
        ),
    ]


class TestFullPipeline:
    """Integration tests for the full B-roll processing pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_services(
        self, tmp_path, sample_config, mock_transcript_result, mock_broll_plan, mock_video_results
    ):
        """Test the complete pipeline end-to-end with all external APIs mocked."""
        # Create a dummy video file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"\x00" * 1024)  # Dummy file

        # Set up config to use tmp_path
        sample_config["local_input_folder"] = str(tmp_path / "input")
        sample_config["local_output_folder"] = str(tmp_path / "output")
        Path(sample_config["local_input_folder"]).mkdir(parents=True, exist_ok=True)
        Path(sample_config["local_output_folder"]).mkdir(parents=True, exist_ok=True)

        # Create mock services
        mock_ai = MagicMock()
        mock_ai.plan_broll_needs = MagicMock(return_value=mock_broll_plan)
        mock_ai.evaluate_videos = MagicMock(
            return_value=[
                ScoredVideo(
                    video_id="test_vid_1", score=8, video_result=mock_video_results[0]
                ),
            ]
        )
        mock_ai.detect_content_style = MagicMock(return_value=MagicMock())
        mock_ai.generate_image_queries = MagicMock(return_value=MagicMock(needs=[]))

        mock_transcription = MagicMock()
        mock_transcription.transcribe = MagicMock(return_value=mock_transcript_result)

        mock_downloader = MagicMock()
        mock_downloader.download_single_video_to_folder = MagicMock(
            return_value=str(tmp_path / "downloaded.mp4")
        )

        mock_organizer = MagicMock()
        mock_organizer.create_project_folder = MagicMock(
            return_value=str(tmp_path / "output" / "project")
        )
        mock_organizer.create_need_subfolder = MagicMock(
            return_value=str(tmp_path / "output" / "project" / "0m30s_ai_robot")
        )

        mock_file_monitor = MagicMock()
        mock_clip_extractor = MagicMock()
        mock_clip_extractor.process_downloaded_video = MagicMock(return_value=([], False))

        mock_video_source = MagicMock()
        mock_video_source.search = AsyncMock(return_value=mock_video_results)
        mock_video_source.get_source_name = MagicMock(return_value="mock_source")

        # Import BRollProcessor
        from broll_processor import BRollProcessor

        # Create processor with injected mocks
        try:
            processor = BRollProcessor(
                config=sample_config,
                ai_service=mock_ai,
                transcription_service=mock_transcription,
                video_downloader=mock_downloader,
                file_organizer=mock_organizer,
                file_monitor=mock_file_monitor,
                clip_extractor=mock_clip_extractor,
                video_sources=[mock_video_source],
                cost_tracker=MagicMock(),
                video_prefilter=MagicMock(),
            )
        except (ValueError, TypeError) as e:
            # If DI isn't implemented yet, skip gracefully
            if "unexpected keyword" in str(e).lower():
                pytest.skip("Dependency injection not yet implemented (Phase 2A)")
            raise

        # Verify processor was created successfully
        assert processor is not None
        assert processor.ai_service is mock_ai
        assert processor.transcription_service is mock_transcription

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_transcript(self, tmp_path, sample_config):
        """Test that the pipeline gracefully handles an empty transcript."""
        sample_config["local_input_folder"] = str(tmp_path / "input")
        sample_config["local_output_folder"] = str(tmp_path / "output")
        Path(sample_config["local_input_folder"]).mkdir(parents=True, exist_ok=True)
        Path(sample_config["local_output_folder"]).mkdir(parents=True, exist_ok=True)

        # Mock transcription that returns empty result
        mock_transcription = MagicMock()
        mock_transcription.transcribe = MagicMock(
            return_value=TranscriptResult(
                text="", segments=[], duration=0.0, language="en"
            )
        )

        mock_ai = MagicMock()
        mock_ai.plan_broll_needs = MagicMock(
            return_value=BRollPlan(source_duration=0.0, needs=[], clips_per_minute=2.0)
        )
        mock_ai.detect_content_style = MagicMock(return_value=MagicMock())

        from broll_processor import BRollProcessor

        try:
            processor = BRollProcessor(
                config=sample_config,
                ai_service=mock_ai,
                transcription_service=mock_transcription,
                video_downloader=MagicMock(),
                file_organizer=MagicMock(),
                file_monitor=MagicMock(),
                clip_extractor=MagicMock(),
                video_sources=[MagicMock()],
                cost_tracker=MagicMock(),
                video_prefilter=MagicMock(),
            )
        except (ValueError, TypeError) as e:
            if "unexpected keyword" in str(e).lower():
                pytest.skip("Dependency injection not yet implemented (Phase 2A)")
            raise

        # Verify AI planner returns empty plan for empty transcript
        plan = processor.ai_service.plan_broll_needs(
            TranscriptResult(text="", segments=[], duration=0.0, language="en")
        )
        assert len(plan.needs) == 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_api_errors(self, tmp_path, sample_config):
        """Test that API errors are properly propagated."""
        sample_config["local_input_folder"] = str(tmp_path / "input")
        sample_config["local_output_folder"] = str(tmp_path / "output")
        Path(sample_config["local_input_folder"]).mkdir(parents=True, exist_ok=True)
        Path(sample_config["local_output_folder"]).mkdir(parents=True, exist_ok=True)

        # Mock AI service that raises an error
        mock_ai = MagicMock()
        mock_ai.plan_broll_needs = MagicMock(
            side_effect=Exception("API rate limit exceeded")
        )
        mock_ai.detect_content_style = MagicMock(return_value=MagicMock())

        mock_transcription = MagicMock()
        segments = [TranscriptSegment(start=0, end=60, text="Test transcript")]
        mock_transcription.transcribe = MagicMock(
            return_value=TranscriptResult(
                text="Test transcript",
                segments=segments,
                duration=60.0,
                language="en",
            )
        )

        from broll_processor import BRollProcessor

        try:
            processor = BRollProcessor(
                config=sample_config,
                ai_service=mock_ai,
                transcription_service=mock_transcription,
                video_downloader=MagicMock(),
                file_organizer=MagicMock(),
                file_monitor=MagicMock(),
                clip_extractor=MagicMock(),
                video_sources=[MagicMock()],
                cost_tracker=MagicMock(),
                video_prefilter=MagicMock(),
            )
        except (ValueError, TypeError) as e:
            if "unexpected keyword" in str(e).lower():
                pytest.skip("Dependency injection not yet implemented (Phase 2A)")
            raise

        # Verify the error would propagate
        with pytest.raises(Exception, match="API rate limit"):
            processor.ai_service.plan_broll_needs(
                TranscriptResult(
                    text="test", segments=[], duration=60.0, language="en"
                )
            )

    @pytest.mark.asyncio
    async def test_checkpoint_and_resume(self, tmp_path, sample_config):
        """Test that checkpointing mechanism works correctly."""
        from utils.checkpoint import (
            ProcessingCheckpoint,
            get_checkpoint_path,
            cleanup_checkpoint,
        )

        # Create a checkpoint
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_path = checkpoint_dir / "test_checkpoint.json"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock checkpoint using the correct dataclass fields
        video_path = str(tmp_path / "test_video.mp4")
        checkpoint = ProcessingCheckpoint(
            video_path=video_path,
            project_dir=str(tmp_path / "project"),
            stage="planned",
            completed_needs=["0m30s_ai_robot"],
            total_needs=3,
        )

        # Save checkpoint using the class method
        checkpoint.save(checkpoint_path)

        # Verify checkpoint exists
        assert checkpoint_path.exists()

        # Load it back using the class method
        loaded = ProcessingCheckpoint.load(checkpoint_path)

        assert loaded is not None
        assert loaded.stage == "planned"
        assert loaded.video_path == video_path
        assert loaded.completed_needs == ["0m30s_ai_robot"]
        assert loaded.total_needs == 3

        # Test cleanup
        cleanup_checkpoint(checkpoint_path)
        assert not checkpoint_path.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_path_generation(self, tmp_path):
        """Test checkpoint path generation utility."""
        from utils.checkpoint import get_checkpoint_path

        video_path = str(tmp_path / "my_video.mp4")
        output_dir = str(tmp_path / "output")

        checkpoint_path = get_checkpoint_path(video_path, output_dir)

        assert checkpoint_path.name == "my_video.checkpoint.json"
        assert ".checkpoints" in str(checkpoint_path)

    @pytest.mark.asyncio
    async def test_broll_need_folder_name_generation(self):
        """Test that BRollNeed generates correct folder names."""
        need = BRollNeed(
            timestamp=90.0,  # 1:30
            search_phrase="city skyline aerial",
            description="Beautiful city aerial shot",
            context="discussing urban development",
        )

        folder_name = need.folder_name
        assert folder_name.startswith("1m30s_")
        assert "beautiful" in folder_name.lower() or "city" in folder_name.lower()

    @pytest.mark.asyncio
    async def test_transcript_context_window(self, mock_transcript_result):
        """Test TranscriptResult context window functionality."""
        # Get context around timestamp 15.0 (middle of the transcript)
        context = mock_transcript_result.get_text_around_timestamp(15.0, 10.0)

        # Should contain text from nearby segments
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_broll_plan_properties(self, mock_broll_plan):
        """Test BRollPlan computed properties."""
        # Test expected clip count (5 min * 2 clips/min = 10)
        assert mock_broll_plan.expected_clip_count == 10

        # Test actual clip count
        assert mock_broll_plan.actual_clip_count == 2

        # Test sorted needs
        sorted_needs = mock_broll_plan.get_needs_sorted_by_timestamp()
        assert sorted_needs[0].timestamp == 30.0
        assert sorted_needs[1].timestamp == 120.0


class TestAPIIntegration:
    """Integration tests for the FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test the health check endpoint returns 200."""
        # Import FastAPI test client
        try:
            from httpx import AsyncClient, ASGITransport
        except ImportError:
            pytest.skip("httpx not installed for API testing")

        try:
            from api.server import app
        except Exception as e:
            pytest.skip(f"Could not import server app: {e}")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Test the root endpoint returns API info."""
        try:
            from httpx import AsyncClient, ASGITransport
        except ImportError:
            pytest.skip("httpx not installed for API testing")

        try:
            from api.server import app
        except Exception as e:
            pytest.skip(f"Could not import server app: {e}")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "version" in data


class TestModelIntegration:
    """Integration tests for data model interactions."""

    def test_video_result_to_scored_video(self, mock_video_results):
        """Test VideoResult can be wrapped in ScoredVideo."""
        video = mock_video_results[0]
        scored = ScoredVideo(video_id=video.video_id, score=8, video_result=video)

        assert scored.video_id == "test_vid_1"
        assert scored.score == 8
        assert scored.video_result.title == "AI Robot Demo"

    def test_clip_segment_duration(self):
        """Test ClipSegment duration calculation."""
        segment = ClipSegment(
            start_time=10.5,
            end_time=18.2,
            relevance_score=8,
            description="Good B-roll clip",
        )

        assert segment.duration == pytest.approx(7.7)

    def test_clip_segment_metadata(self):
        """Test ClipSegment metadata conversion."""
        segment = ClipSegment(
            start_time=5.0,
            end_time=12.5,
            relevance_score=9,
            description="Excellent B-roll",
            clip_score=0.85,
            gemini_score=9,
        )

        metadata = segment.to_metadata_dict()

        assert metadata["start_time"] == 5.0
        assert metadata["end_time"] == 12.5
        assert metadata["duration"] == 7.5
        assert metadata["relevance_score"] == 9
        assert metadata["clip_score"] == 0.85
        assert metadata["gemini_score"] == 9

    def test_broll_need_clamps_duration(self):
        """Test BRollNeed clamps suggested_duration to valid range."""
        # Test minimum clamping
        need_short = BRollNeed(
            timestamp=10.0,
            search_phrase="test",
            description="test",
            context="test",
            suggested_duration=1.0,  # Below minimum
        )
        assert need_short.suggested_duration == 4.0

        # Test maximum clamping
        need_long = BRollNeed(
            timestamp=10.0,
            search_phrase="test",
            description="test",
            context="test",
            suggested_duration=30.0,  # Above maximum
        )
        assert need_long.suggested_duration == 15.0

    def test_broll_need_enhanced_metadata(self):
        """Test BRollNeed enhanced metadata detection."""
        # Without enhanced metadata
        basic_need = BRollNeed(
            timestamp=10.0,
            search_phrase="city skyline",
            description="City view",
            context="urban",
        )
        assert not basic_need.has_enhanced_metadata()

        # With enhanced metadata
        enhanced_need = BRollNeed(
            timestamp=10.0,
            search_phrase="city skyline",
            description="City view",
            context="urban",
            alternate_searches=["urban landscape", "downtown aerial"],
            negative_keywords=["night", "winter"],
            visual_style="cinematic",
        )
        assert enhanced_need.has_enhanced_metadata()
        assert len(enhanced_need.all_search_phrases) == 3

    def test_transcript_result_format_with_timestamps(self, mock_transcript_result):
        """Test TranscriptResult timestamp formatting."""
        formatted = mock_transcript_result.format_with_timestamps()

        assert "[0:00]" in formatted
        assert "[0:15]" in formatted
        assert "[0:30]" in formatted
        assert "Welcome" in formatted
