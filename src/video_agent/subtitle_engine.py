"""Hormozi/CapCut-style animated subtitle generator.

Generates word-level animated subtitles using faster-whisper for timing
and pysubs2 for ASS/SSA subtitle file generation. Subtitles are burned
into video via FFmpeg.
"""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import pysubs2

from video_agent.models import SubtitleStyle, WordTiming

logger = logging.getLogger(__name__)


class SubtitleEngine:
    """Generates word-level animated subtitles (Hormozi/CapCut style).

    Uses faster-whisper for word-level timing and pysubs2 for ASS/SSA generation.
    Subtitles are burned into video via FFmpeg.
    """

    def __init__(self, whisper_model: str = "base", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self._whisper_model_name = whisper_model
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _get_model(self):
        """Lazy-load the faster-whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
                self._whisper_model_name, self._device, self._compute_type,
            )
            self._model = WhisperModel(
                self._whisper_model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    async def generate_word_timestamps(self, audio_path: Path) -> list[WordTiming]:
        """Get word-level timestamps from audio using faster-whisper.

        Args:
            audio_path: Path to audio or video file.

        Returns:
            List of WordTiming objects with per-word start/end times.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract audio to WAV if input is a video file
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
        if audio_path.suffix.lower() in video_exts:
            wav_path = await self._extract_audio(audio_path)
            cleanup = True
        else:
            wav_path = audio_path
            cleanup = False

        try:
            model = self._get_model()
            word_timings = await asyncio.to_thread(
                self._transcribe_words, model, str(wav_path),
            )
            logger.info("Generated %d word timestamps from %s", len(word_timings), audio_path.name)
            return word_timings
        finally:
            if cleanup:
                Path(wav_path).unlink(missing_ok=True)

    def _transcribe_words(self, model, audio_path: str) -> list[WordTiming]:
        """Run faster-whisper transcription with word-level timestamps."""
        segments, _info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
        )

        word_timings: list[WordTiming] = []
        for segment in segments:
            if segment.words is None:
                continue
            for w in segment.words:
                word_timings.append(WordTiming(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    confidence=float(w.probability) if hasattr(w, "probability") else 1.0,
                ))
        return word_timings

    async def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video to a temporary WAV file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp.name,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg audio extraction failed: {stderr.decode()}")
        return Path(tmp.name)

    # ------------------------------------------------------------------
    # Word grouping
    # ------------------------------------------------------------------

    def _group_words(
        self,
        word_timings: list[WordTiming],
        words_per_group: int = 3,
    ) -> list[tuple[list[WordTiming], float, float]]:
        """Group words into display chunks for subtitle rendering.

        Args:
            word_timings: Full list of word timings.
            words_per_group: How many words per subtitle line.

        Returns:
            List of (words, group_start_seconds, group_end_seconds).
        """
        if not word_timings:
            return []

        groups: list[tuple[list[WordTiming], float, float]] = []
        for i in range(0, len(word_timings), words_per_group):
            chunk = word_timings[i : i + words_per_group]
            start = chunk[0].start
            end = chunk[-1].end
            groups.append((chunk, start, end))
        return groups

    def _group_words_sentences(
        self,
        word_timings: list[WordTiming],
        max_words: int = 10,
        max_duration: float = 4.0,
    ) -> list[tuple[list[WordTiming], float, float]]:
        """Group words into sentence-like chunks for documentary/minimal styles.

        Splits on sentence-ending punctuation or when limits are reached.
        """
        if not word_timings:
            return []

        sentence_enders = {".", "!", "?"}
        groups: list[tuple[list[WordTiming], float, float]] = []
        current: list[WordTiming] = []

        for wt in word_timings:
            current.append(wt)
            duration = current[-1].end - current[0].start
            ends_sentence = any(wt.word.rstrip().endswith(p) for p in sentence_enders)

            if ends_sentence or len(current) >= max_words or duration >= max_duration:
                groups.append((list(current), current[0].start, current[-1].end))
                current = []

        if current:
            groups.append((current, current[0].start, current[-1].end))

        return groups

    # ------------------------------------------------------------------
    # ASS subtitle generation
    # ------------------------------------------------------------------

    def generate_ass_subtitles(
        self,
        word_timings: list[WordTiming],
        style: SubtitleStyle = SubtitleStyle.HORMOZI,
        video_width: int = 1920,
        video_height: int = 1080,
    ) -> str:
        """Generate ASS subtitle file content using pysubs2.

        Args:
            word_timings: Word-level timing data.
            style: Subtitle style preset.
            video_width: Video resolution width.
            video_height: Video resolution height.

        Returns:
            ASS file content as a string.
        """
        subs = pysubs2.SSAFile()
        subs.info["PlayResX"] = str(video_width)
        subs.info["PlayResY"] = str(video_height)

        if style == SubtitleStyle.HORMOZI:
            self._build_hormozi(subs, word_timings, video_width, video_height)
        elif style == SubtitleStyle.DOCUMENTARY:
            self._build_documentary(subs, word_timings, video_width, video_height)
        elif style == SubtitleStyle.MINIMAL:
            self._build_minimal(subs, word_timings, video_width, video_height)
        else:
            raise ValueError(f"Unknown subtitle style: {style}")

        return subs.to_string("ass")

    # -- Hormozi style ------------------------------------------------

    def _build_hormozi(
        self,
        subs: pysubs2.SSAFile,
        word_timings: list[WordTiming],
        width: int,
        height: int,
    ) -> None:
        """Build Hormozi/CapCut style: bold, center-screen, word-by-word pop."""
        hormozi = pysubs2.SSAStyle(
            fontname="Arial Black",
            fontsize=80,
            bold=True,
            primarycolor=pysubs2.Color(255, 255, 255, 0),       # white, opaque
            secondarycolor=pysubs2.Color(255, 255, 0, 0),       # yellow highlight
            outlinecolor=pysubs2.Color(0, 0, 0, 0),             # black outline
            backcolor=pysubs2.Color(0, 0, 0, 100),              # semi-transparent bg
            outline=4.0,
            shadow=2.0,
            borderstyle=1,  # outline + drop shadow
            alignment=pysubs2.Alignment.BOTTOM_CENTER,
            marginv=int(height * 0.12),  # ~12% from bottom
            scalex=100,
            scaley=100,
            spacing=1.0,
        )
        subs.styles["Hormozi"] = hormozi

        groups = self._group_words(word_timings, words_per_group=3)

        for words, group_start, group_end in groups:
            # Build karaoke-tagged text: each word gets a \k duration tag
            # \k duration is in centiseconds (1/100 second)
            parts: list[str] = []
            for i, wt in enumerate(words):
                word_dur_cs = max(1, int((wt.end - wt.start) * 100))
                # Pop-in scale effect on each word
                pop = r"{\t(0,80,\fscx110\fscy110)\t(80,160,\fscx100\fscy100)}"
                parts.append(f"{{\\kf{word_dur_cs}}}{pop}{wt.word}")

            text = " ".join(parts)

            event = pysubs2.SSAEvent(
                start=int(group_start * 1000),
                end=int(group_end * 1000),
                text=text,
                style="Hormozi",
            )
            subs.events.append(event)

    # -- Documentary style --------------------------------------------

    def _build_documentary(
        self,
        subs: pysubs2.SSAFile,
        word_timings: list[WordTiming],
        width: int,
        height: int,
    ) -> None:
        """Build documentary style: clean, sentence-based, fade in/out."""
        doc_style = pysubs2.SSAStyle(
            fontname="Arial",
            fontsize=50,
            bold=False,
            primarycolor=pysubs2.Color(255, 255, 255, 0),
            outlinecolor=pysubs2.Color(0, 0, 0, 0),
            backcolor=pysubs2.Color(0, 0, 0, 80),
            outline=2.0,
            shadow=1.0,
            borderstyle=1,
            alignment=pysubs2.Alignment.BOTTOM_CENTER,
            marginv=60,
        )
        subs.styles["Documentary"] = doc_style

        groups = self._group_words_sentences(word_timings, max_words=10, max_duration=4.0)

        for words, group_start, group_end in groups:
            text_str = " ".join(wt.word for wt in words)
            # Fade in 200ms, fade out 200ms
            text = r"{\fad(200,200)}" + text_str

            event = pysubs2.SSAEvent(
                start=int(group_start * 1000),
                end=int(group_end * 1000),
                text=text,
                style="Documentary",
            )
            subs.events.append(event)

    # -- Minimal style ------------------------------------------------

    def _build_minimal(
        self,
        subs: pysubs2.SSAFile,
        word_timings: list[WordTiming],
        width: int,
        height: int,
    ) -> None:
        """Build minimal style: small, lower-left, simple fade."""
        minimal_style = pysubs2.SSAStyle(
            fontname="Arial",
            fontsize=40,
            bold=False,
            primarycolor=pysubs2.Color(220, 220, 220, 0),
            outlinecolor=pysubs2.Color(0, 0, 0, 50),
            backcolor=pysubs2.Color(0, 0, 0, 150),
            outline=1.5,
            shadow=0.5,
            borderstyle=1,
            alignment=pysubs2.Alignment.BOTTOM_LEFT,
            marginl=40,
            marginv=40,
        )
        subs.styles["Minimal"] = minimal_style

        groups = self._group_words_sentences(word_timings, max_words=8, max_duration=3.5)

        for words, group_start, group_end in groups:
            text_str = " ".join(wt.word for wt in words)
            text = r"{\fad(150,150)}" + text_str

            event = pysubs2.SSAEvent(
                start=int(group_start * 1000),
                end=int(group_end * 1000),
                text=text,
                style="Minimal",
            )
            subs.events.append(event)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_ass_file(self, ass_content: str, output_path: Path) -> Path:
        """Save ASS subtitle content to a file.

        Args:
            ass_content: The ASS file content string.
            output_path: Where to write the file.

        Returns:
            The path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(ass_content, encoding="utf-8")
        logger.info("Saved ASS subtitle file: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Burn subtitles into video
    # ------------------------------------------------------------------

    def burn_subtitles(
        self,
        video_path: Path,
        ass_path: Path,
        output_path: Path | None = None,
    ) -> Path:
        """Burn ASS subtitles into video using FFmpeg.

        Args:
            video_path: Source video file.
            ass_path: ASS subtitle file to burn in.
            output_path: Destination video file. Defaults to
                ``<video_stem>_subtitled.mp4`` next to the original.

        Returns:
            Path to the output video with burned-in subtitles.
        """
        video_path = Path(video_path)
        ass_path = Path(ass_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not ass_path.exists():
            raise FileNotFoundError(f"ASS file not found: {ass_path}")

        if output_path is None:
            output_path = video_path.with_name(f"{video_path.stem}_subtitled.mp4")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Escape special characters in ass_path for FFmpeg filter syntax
        # FFmpeg subtitle filter uses : and \ as special chars
        escaped_ass = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{escaped_ass}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ]

        logger.info("Burning subtitles into video: %s", video_path.name)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg subtitle burn failed (exit {result.returncode}): {result.stderr}"
            )

        logger.info("Subtitled video saved: %s", output_path)
        return output_path
