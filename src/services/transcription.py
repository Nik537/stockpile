"""Audio transcription service using Whisper (faster-whisper or openai-whisper)."""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from models.broll_need import TranscriptResult, TranscriptSegment
from utils.config import get_supported_audio_formats, get_supported_video_formats
from utils.retry import retry_api_call, retry_file_operation

logger = logging.getLogger(__name__)

# Try to import faster-whisper first, fall back to regular whisper
_USE_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel
    _USE_FASTER_WHISPER = True
    logger.debug("Using faster-whisper backend")
except ImportError:
    import whisper
    logger.debug("Using openai-whisper backend (faster-whisper not available)")


class TranscriptionService:
    """Service for transcribing audio content using Whisper."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._transcription_lock = asyncio.Lock()
        self._use_faster_whisper = _USE_FASTER_WHISPER
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")

            if self._use_faster_whisper:
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info(
                    f"Loaded faster-whisper model {self.model_name} "
                    f"(device={self.device}, compute_type={self.compute_type})"
                )
            else:
                self.model = whisper.load_model(self.model_name)
                is_multilingual = (
                    "multilingual" if self.model.is_multilingual else "English-only"
                )
                param_count = sum(p.numel() for p in self.model.parameters())
                logger.info(
                    f"Loaded {is_multilingual} Whisper model with {param_count:,} parameters"
                )

        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}")
            raise

    def _cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @retry_api_call(max_retries=3, base_delay=2.0)
    async def transcribe_audio(
        self, input_file_path: str, with_timestamps: bool = True
    ) -> Union[TranscriptResult, str]:
        """Transcribe audio file to text with optional timestamps.

        Args:
            input_file_path: Path to audio or video file
            with_timestamps: If True, return full TranscriptResult with segments.
                           If False, return just the text string (legacy behavior).

        Returns:
            TranscriptResult with segments and duration, or just text string
        """
        file_path: Path = Path(input_file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Starting transcription of: {file_path.name}")

        try:
            if self._is_video_file(file_path):
                audio_path = self._extract_audio_from_video(file_path)
                cleanup_audio = True
            else:
                audio_path = file_path
                cleanup_audio = False

            async with self._transcription_lock:
                result = await asyncio.to_thread(
                    self._transcribe_with_whisper, str(audio_path)
                )

            if cleanup_audio:
                Path(audio_path).unlink(missing_ok=True)

            # Return full result or just text based on flag
            if with_timestamps:
                return result
            else:
                return result.text

        except Exception as e:
            logger.error(f"Transcription failed for {file_path}: {e}")
            raise

    def _transcribe_with_whisper(self, audio_path: str) -> TranscriptResult:
        """Run Whisper transcription and return full result with timestamps."""
        try:
            if not self.model:
                raise ValueError("Whisper model not loaded")

            if self._use_faster_whisper:
                return self._transcribe_faster_whisper(audio_path)
            else:
                return self._transcribe_openai_whisper(audio_path)

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    def _transcribe_faster_whisper(self, audio_path: str) -> TranscriptResult:
        """Transcribe using faster-whisper backend."""
        # faster-whisper returns a generator of segments and info
        segments_generator, info = self.model.transcribe(audio_path)

        # Collect segments from generator
        segments = []
        text_parts = []
        for seg in segments_generator:
            segments.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=str(seg.text).strip(),
                )
            )
            text_parts.append(str(seg.text).strip())

        # Combine all segment texts
        text = " ".join(text_parts)

        # Get duration and language from info
        duration = float(info.duration)
        language = info.language

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{duration:.1f}s duration, language: {language}"
        )

        return TranscriptResult(
            text=text.strip(),
            segments=segments,
            duration=duration,
            language=language,
        )

    def _transcribe_openai_whisper(self, audio_path: str) -> TranscriptResult:
        """Transcribe using openai-whisper backend."""
        result = self.model.transcribe(
            str(audio_path),
            language=None,
            task="transcribe",
            fp16=False,
            verbose=False,
        )

        # Extract text
        text = result.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Transcription result is not a string")

        # Extract segments with timing info
        raw_segments = result.get("segments", [])
        segments = []
        for seg in raw_segments:
            segments.append(
                TranscriptSegment(
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                    text=str(seg.get("text", "")).strip(),
                )
            )

        # Get duration (last segment end time, or explicit duration if available)
        duration = 0.0
        if segments:
            duration = segments[-1].end
        if "duration" in result:
            duration = float(result["duration"])

        # Get detected language
        language = result.get("language")

        logger.info(
            f"Transcription complete: {len(segments)} segments, "
            f"{duration:.1f}s duration, language: {language}"
        )

        return TranscriptResult(
            text=text.strip(),
            segments=segments,
            duration=duration,
            language=language,
        )

    @retry_file_operation(max_retries=3, base_delay=1.0)
    def _extract_audio_from_video(self, video_path: Path) -> str:
        """Extract audio from video file using ffmpeg.

        Args:
            video_path: Path to video file

        Returns:
            Path to extracted audio file
        """
        logger.info(f"Extracting audio from video: {video_path.name}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_path = temp_file.name

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                audio_path,
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger.debug(f"Audio extraction completed: {audio_path}")
            return audio_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            Path(audio_path).unlink(missing_ok=True)
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")

        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            Path(audio_path).unlink(missing_ok=True)
            raise

    def _is_video_file(self, file_path: Path) -> bool:
        video_extensions = get_supported_video_formats()
        return file_path.suffix.lower() in video_extensions

    def _is_audio_file(self, file_path: Path) -> bool:
        audio_extensions = get_supported_audio_formats()
        return file_path.suffix.lower() in audio_extensions

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file format is supported for transcription.

        Args:
            file_path: Path to file

        Returns:
            True if file format is supported
        """
        path = Path(file_path)
        return self._is_video_file(path) or self._is_audio_file(path)
