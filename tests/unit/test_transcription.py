"""Unit tests for TranscriptionService with faster-whisper."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Create a mock for the faster_whisper module before any tests run
mock_faster_whisper = MagicMock()
mock_faster_whisper.WhisperModel = MagicMock()
sys.modules["faster_whisper"] = mock_faster_whisper


def create_mock_transcription_result():
    """Create mock segments and info for transcription results."""
    mock_segment1 = MagicMock()
    mock_segment1.start = 0.0
    mock_segment1.end = 5.0
    mock_segment1.text = " Hello, this is a test."

    mock_segment2 = MagicMock()
    mock_segment2.start = 5.0
    mock_segment2.end = 10.0
    mock_segment2.text = " This is the second segment."

    mock_info = MagicMock()
    mock_info.duration = 10.0
    mock_info.language = "en"

    return [mock_segment1, mock_segment2], mock_info


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset module cache before each test."""
    # Clear the transcription module from cache so each test gets fresh import
    if "services.transcription" in sys.modules:
        del sys.modules["services.transcription"]
    yield


@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel."""
    mock_model = MagicMock()
    segments, info = create_mock_transcription_result()
    mock_model.transcribe.return_value = (iter(segments), info)
    return mock_model


class TestTranscriptionServiceInit:
    """Tests for TranscriptionService initialization."""

    def test_init_with_defaults(self):
        """Test service initializes with default parameters."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model
        mock_faster_whisper.WhisperModel.reset_mock()

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        assert service.model_name == "base"
        assert service.device == "auto"
        assert service.compute_type == "auto"
        mock_faster_whisper.WhisperModel.assert_called_once_with(
            "base", device="auto", compute_type="auto"
        )

    def test_init_with_custom_params(self):
        """Test service initializes with custom parameters."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model
        mock_faster_whisper.WhisperModel.reset_mock()

        from services.transcription import TranscriptionService
        service = TranscriptionService(
            model_name="large-v3",
            device="cuda",
            compute_type="float16"
        )

        assert service.model_name == "large-v3"
        assert service.device == "cuda"
        assert service.compute_type == "float16"
        mock_faster_whisper.WhisperModel.assert_called_with(
            "large-v3", device="cuda", compute_type="float16"
        )

    def test_init_model_load_failure(self):
        """Test service handles model load failure gracefully."""
        mock_faster_whisper.WhisperModel.side_effect = RuntimeError("Failed to load model")

        from services.transcription import TranscriptionService

        with pytest.raises(RuntimeError, match="Failed to load model"):
            TranscriptionService()

        # Reset side_effect for other tests
        mock_faster_whisper.WhisperModel.side_effect = None


class TestTranscriptionServiceTranscribe:
    """Tests for transcription functionality."""

    @pytest.mark.anyio
    async def test_transcribe_audio_with_timestamps(self, temp_dir):
        """Test transcription returns TranscriptResult with segments."""
        # Create mock segments
        segments, info = create_mock_transcription_result()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        # Create a dummy audio file
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"dummy audio data")

        result = await service.transcribe_audio(
            str(audio_file), with_timestamps=True
        )

        # Verify result structure
        assert result.text == "Hello, this is a test. This is the second segment."
        assert result.duration == 10.0
        assert result.language == "en"
        assert len(result.segments) == 2

        # Verify first segment
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 5.0
        assert result.segments[0].text == "Hello, this is a test."

        # Verify second segment
        assert result.segments[1].start == 5.0
        assert result.segments[1].end == 10.0
        assert result.segments[1].text == "This is the second segment."

    @pytest.mark.anyio
    async def test_transcribe_audio_without_timestamps(self, temp_dir):
        """Test transcription returns just text when with_timestamps=False."""
        segments, info = create_mock_transcription_result()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"dummy audio data")

        result = await service.transcribe_audio(
            str(audio_file), with_timestamps=False
        )

        assert isinstance(result, str)
        assert result == "Hello, this is a test. This is the second segment."

    @pytest.mark.anyio
    async def test_transcribe_file_not_found(self):
        """Test transcription raises error for non-existent file."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        with pytest.raises(FileNotFoundError):
            await service.transcribe_audio("/nonexistent/file.wav")

    @pytest.mark.anyio
    async def test_transcribe_video_file(self, temp_dir):
        """Test transcription of video file extracts audio first."""
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = " Test text."

        mock_info = MagicMock()
        mock_info.duration = 5.0
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"dummy video data")

        # Mock subprocess for ffmpeg audio extraction
        with patch("services.transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # The file will be deleted by cleanup, so we need to create it
            with patch.object(Path, "unlink"):
                result = await service.transcribe_audio(str(video_file))

                # Verify ffmpeg was called for audio extraction
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "ffmpeg"
                assert "-vn" in call_args  # Video disabled (audio only)
                assert "-acodec" in call_args


class TestTranscriptionServiceHelpers:
    """Tests for helper methods."""

    def test_is_video_file(self):
        """Test video file detection."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]

        for ext in video_extensions:
            path = Path(f"/test/video{ext}")
            assert service._is_video_file(path) is True

        # Non-video files
        non_video = [".mp3", ".wav", ".txt", ".pdf"]
        for ext in non_video:
            path = Path(f"/test/file{ext}")
            assert service._is_video_file(path) is False

    def test_is_audio_file(self):
        """Test audio file detection."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"]

        for ext in audio_extensions:
            path = Path(f"/test/audio{ext}")
            assert service._is_audio_file(path) is True

        # Non-audio files
        non_audio = [".mp4", ".txt", ".pdf"]
        for ext in non_audio:
            path = Path(f"/test/file{ext}")
            assert service._is_audio_file(path) is False

    def test_is_supported_file(self):
        """Test supported file detection."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        # Supported files
        supported = [
            "/test/video.mp4",
            "/test/audio.mp3",
            "/test/video.mkv",
            "/test/audio.wav"
        ]
        for path in supported:
            assert service.is_supported_file(path) is True

        # Unsupported files
        unsupported = ["/test/doc.txt", "/test/data.json", "/test/image.jpg"]
        for path in unsupported:
            assert service.is_supported_file(path) is False


class TestAudioExtraction:
    """Tests for audio extraction from video."""

    def test_extract_audio_success(self, temp_dir):
        """Test successful audio extraction from video."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"dummy video data")

        with patch("services.transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            audio_path = service._extract_audio_from_video(video_file)

            # Verify ffmpeg was called with correct arguments
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]

            assert call_args[0] == "ffmpeg"
            assert "-i" in call_args
            assert str(video_file) in call_args
            assert "-vn" in call_args  # No video
            assert "-acodec" in call_args
            assert "pcm_s16le" in call_args  # WAV codec
            assert "-ar" in call_args
            assert "16000" in call_args  # 16kHz sample rate
            assert "-ac" in call_args
            assert "1" in call_args  # Mono

            # Cleanup
            Path(audio_path).unlink(missing_ok=True)

    def test_extract_audio_ffmpeg_failure(self, temp_dir):
        """Test audio extraction handles ffmpeg failure."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"dummy video data")

        with patch("services.transcription.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ffmpeg", stderr="FFmpeg error: codec not found"
            )

            with pytest.raises(RuntimeError, match="Audio extraction failed"):
                service._extract_audio_from_video(video_file)


class TestCudaDetection:
    """Tests for CUDA availability detection."""

    def test_cuda_available_with_torch(self):
        """Test CUDA detection when torch is available."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        # Test when torch.cuda.is_available() returns True
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True
            assert service._cuda_available() is True

            mock_torch.cuda.is_available.return_value = False
            assert service._cuda_available() is False

    def test_cuda_available_without_torch(self):
        """Test CUDA detection returns False when torch not installed."""
        mock_model = MagicMock()
        mock_faster_whisper.WhisperModel.return_value = mock_model

        from services.transcription import TranscriptionService
        service = TranscriptionService()

        # Simulate torch not being installed
        def import_error_side_effect(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return MagicMock()

        with patch("builtins.__import__", side_effect=import_error_side_effect):
            assert service._cuda_available() is False
