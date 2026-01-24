"""Unit tests for TranscriptionService with faster-whisper."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock the faster_whisper module before importing TranscriptionService
# This allows tests to run without actually installing faster-whisper


@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel."""
    mock_model = MagicMock()

    # Mock transcribe method to return segments generator and info
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

    # Return generator-like object and info
    mock_model.transcribe.return_value = (
        iter([mock_segment1, mock_segment2]),
        mock_info,
    )

    return mock_model


@pytest.fixture
def mock_whisper_model_class(mock_whisper_model):
    """Patch the WhisperModel class to return our mock."""
    with patch("services.transcription.WhisperModel") as mock_class:
        mock_class.return_value = mock_whisper_model
        yield mock_class


@pytest.fixture
def transcription_service(mock_whisper_model_class):
    """Create a TranscriptionService with mocked WhisperModel."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from services.transcription import TranscriptionService

    service = TranscriptionService(
        model_name="base",
        device="cpu",
        compute_type="int8"
    )
    return service


class TestTranscriptionServiceInit:
    """Tests for TranscriptionService initialization."""

    def test_init_with_defaults(self, mock_whisper_model_class):
        """Test service initializes with default parameters."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from services.transcription import TranscriptionService

        service = TranscriptionService()

        assert service.model_name == "base"
        assert service.device == "auto"
        assert service.compute_type == "auto"
        mock_whisper_model_class.assert_called_once_with(
            "base", device="auto", compute_type="auto"
        )

    def test_init_with_custom_params(self, mock_whisper_model_class):
        """Test service initializes with custom parameters."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from services.transcription import TranscriptionService

        service = TranscriptionService(
            model_name="large-v3",
            device="cuda",
            compute_type="float16"
        )

        assert service.model_name == "large-v3"
        assert service.device == "cuda"
        assert service.compute_type == "float16"
        mock_whisper_model_class.assert_called_with(
            "large-v3", device="cuda", compute_type="float16"
        )

    def test_init_model_load_failure(self, mock_whisper_model_class):
        """Test service handles model load failure gracefully."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        from services.transcription import TranscriptionService

        mock_whisper_model_class.side_effect = RuntimeError("Failed to load model")

        with pytest.raises(RuntimeError, match="Failed to load model"):
            TranscriptionService()


class TestTranscriptionServiceTranscribe:
    """Tests for transcription functionality."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_with_timestamps(
        self, transcription_service, mock_whisper_model, temp_dir
    ):
        """Test transcription returns TranscriptResult with segments."""
        # Create a dummy audio file
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"dummy audio data")

        result = await transcription_service.transcribe_audio(
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

    @pytest.mark.asyncio
    async def test_transcribe_audio_without_timestamps(
        self, transcription_service, temp_dir
    ):
        """Test transcription returns just text when with_timestamps=False."""
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"dummy audio data")

        result = await transcription_service.transcribe_audio(
            str(audio_file), with_timestamps=False
        )

        assert isinstance(result, str)
        assert result == "Hello, this is a test. This is the second segment."

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, transcription_service):
        """Test transcription raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            await transcription_service.transcribe_audio("/nonexistent/file.wav")

    @pytest.mark.asyncio
    async def test_transcribe_video_file(
        self, transcription_service, mock_whisper_model, temp_dir
    ):
        """Test transcription of video file extracts audio first."""
        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"dummy video data")

        # Mock subprocess for ffmpeg audio extraction
        with patch("services.transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # The file will be deleted by cleanup, so we need to create it
            with patch.object(Path, "unlink"):
                result = await transcription_service.transcribe_audio(str(video_file))

                # Verify ffmpeg was called for audio extraction
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert call_args[0] == "ffmpeg"
                assert "-vn" in call_args  # Video disabled (audio only)
                assert "-acodec" in call_args


class TestTranscriptionServiceHelpers:
    """Tests for helper methods."""

    def test_is_video_file(self, transcription_service):
        """Test video file detection."""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]

        for ext in video_extensions:
            path = Path(f"/test/video{ext}")
            assert transcription_service._is_video_file(path) is True

        # Non-video files
        non_video = [".mp3", ".wav", ".txt", ".pdf"]
        for ext in non_video:
            path = Path(f"/test/file{ext}")
            assert transcription_service._is_video_file(path) is False

    def test_is_audio_file(self, transcription_service):
        """Test audio file detection."""
        audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"]

        for ext in audio_extensions:
            path = Path(f"/test/audio{ext}")
            assert transcription_service._is_audio_file(path) is True

        # Non-audio files
        non_audio = [".mp4", ".txt", ".pdf"]
        for ext in non_audio:
            path = Path(f"/test/file{ext}")
            assert transcription_service._is_audio_file(path) is False

    def test_is_supported_file(self, transcription_service):
        """Test supported file detection."""
        # Supported files
        supported = [
            "/test/video.mp4",
            "/test/audio.mp3",
            "/test/video.mkv",
            "/test/audio.wav"
        ]
        for path in supported:
            assert transcription_service.is_supported_file(path) is True

        # Unsupported files
        unsupported = ["/test/doc.txt", "/test/data.json", "/test/image.jpg"]
        for path in unsupported:
            assert transcription_service.is_supported_file(path) is False


class TestAudioExtraction:
    """Tests for audio extraction from video."""

    def test_extract_audio_success(self, transcription_service, temp_dir):
        """Test successful audio extraction from video."""
        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"dummy video data")

        with patch("services.transcription.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            audio_path = transcription_service._extract_audio_from_video(video_file)

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

    def test_extract_audio_ffmpeg_failure(self, transcription_service, temp_dir):
        """Test audio extraction handles ffmpeg failure."""
        import subprocess

        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"dummy video data")

        with patch("services.transcription.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ffmpeg", stderr="FFmpeg error: codec not found"
            )

            with pytest.raises(RuntimeError, match="Audio extraction failed"):
                transcription_service._extract_audio_from_video(video_file)


class TestCudaDetection:
    """Tests for CUDA availability detection."""

    def test_cuda_available_with_torch(self, transcription_service):
        """Test CUDA detection when torch is available."""
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            import sys
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True

            assert transcription_service._cuda_available() is True

            mock_torch.cuda.is_available.return_value = False
            assert transcription_service._cuda_available() is False

    def test_cuda_available_without_torch(self, transcription_service):
        """Test CUDA detection returns False when torch not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            # Import error should return False
            with patch("builtins.__import__", side_effect=ImportError):
                assert transcription_service._cuda_available() is False
