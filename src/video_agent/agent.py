"""Video Production Agent - autonomous video generation pipeline.

Takes a topic and produces a complete faceless YouTube video:
topic -> script -> narration -> assets -> subtitles -> composition -> output
"""

import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import wave
from datetime import datetime
from pathlib import Path
from collections.abc import Awaitable, Callable

from video_agent.models import (
    DraftReview,
    FixRequest,
    Script,
    SceneScript,
    SubtitleStyle,
    Timeline,
    TimelineScene,
    VisualType,
    VisualTypeDecision,
    WordTiming,
)
from video_agent.script_generator import ScriptGenerator
from video_agent.director import DirectorAgent
from video_agent.video_composer import VideoComposer
from video_agent.broll_adapter import BRollAdapter

from services.tts_service import TTSService
from services.music_service import MusicService
from services.ai_service import AIService
from services.image_generation_service import ImageGenerationService
from services.video_sources.pexels import PexelsVideoSource
from services.video_sources.pixabay import PixabayVideoSource
from services.video_sources.youtube import YouTubeVideoSource
from services.image_sources.google import GoogleImageSource
from services.image_sources.pexels import PexelsImageSource
from services.image_sources.pixabay import PixabayImageSource
from services.video_acquisition_service import VideoAcquisitionService
from services.video_search_service import VideoSearchService
from services.clip_extractor import ClipExtractor
from services.video_filter import VideoPreFilter
from services.semantic_verifier import SemanticVerifier
from services.video_downloader import VideoDownloader
from services.file_organizer import FileOrganizer
from models.image_generation import ImageEditRequest, ImageGenerationModel, ImageGenerationRequest
from models.image import ImageResult
from services.prompts._base import strip_markdown_code_blocks
from services.prompts.evaluation import VIDEO_AGENT_IMAGE_EVALUATOR
from utils.config import load_config

logger = logging.getLogger(__name__)

# Maximum B-roll clip duration in seconds
MAX_BROLL_CLIP_SECONDS = 4

# Target clip duration range for Gemini analysis (seconds)
MIN_BROLL_CLIP_SECONDS = 3
TARGET_BROLL_CLIP_SECONDS = 4

# Minimum score for Gemini video analysis segment acceptance
MIN_GEMINI_SEGMENT_SCORE = 7

# Maximum time to wait for Gemini File API processing (seconds)
GEMINI_FILE_POLL_TIMEOUT = 120
GEMINI_FILE_POLL_INTERVAL = 3

# TTS preprocessing: number-to-words helper (no external deps)
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


def _number_to_words(n: int) -> str:
    """Convert an integer to English words. Handles 0 to 999_999_999_999."""
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _number_to_words(-n)

    parts: list[str] = []
    if n >= 1_000_000_000:
        parts.append(_number_to_words(n // 1_000_000_000) + " billion")
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_number_to_words(n // 1_000_000) + " million")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(_number_to_words(n // 1_000) + " thousand")
        n %= 1_000
    if n >= 100:
        parts.append(_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        tail = _ONES[n % 10]
        parts.append(_TENS[n // 10] + ("-" + tail if tail else ""))
    elif n > 0:
        parts.append(_ONES[n])

    return " ".join(parts)


def _year_to_words(y: int) -> str:
    """Convert a year (1000-2099) to spoken English."""
    if 2000 <= y <= 2009:
        return "two thousand" + (" " + _ONES[y - 2000] if y > 2000 else "")
    if 2010 <= y <= 2099:
        return "twenty " + _number_to_words(y - 2000)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return _number_to_words(hi) + " hundred"
    return _number_to_words(hi) + " " + (
        _number_to_words(lo) if lo >= 10 else "oh " + _ONES[lo]
    )


# Abbreviations that should be spelled letter-by-letter with periods
# Excludes word-acronyms (NASA, OPEC, SCUBA) which are pronounced as words
_ABBREVIATION_MAP: dict[str, str] = {
    "AI": "A.I.",
    "CEO": "C.E.O.",
    "CFO": "C.F.O.",
    "CTO": "C.T.O.",
    "FBI": "F.B.I.",
    "CIA": "C.I.A.",
    "USA": "U.S.A.",
    "UK": "U.K.",
    "US": "U.S.",
    "GDP": "G.D.P.",
    "IPO": "I.P.O.",
    "DIY": "D.I.Y.",
}


class VideoProductionAgent:
    """Autonomous video production pipeline.

    Orchestrates all existing stockpile services + new video agent modules
    to produce a complete video from a topic prompt.
    """

    _PARALINGUISTIC_TAG_RE = re.compile(r"\[(?:laugh|sigh|gasp|chuckle|cough|sniff|groan)\]")

    def __init__(self, config: dict | None = None):
        """Initialize with all services.

        Uses dependency injection pattern from BRollProcessor.
        Load config from environment if not provided.
        """
        if config is None:
            config = load_config()

        self.config = config

        # Existing services (reused)
        self.tts = TTSService()
        self.ai = AIService(
            api_key=config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", ""),
            model_name=config.get("gemini_model", "gemini-3-pro-preview"),
        )
        self.music = MusicService()
        self.image_gen = ImageGenerationService()

        # Build B-roll source list ordered by configured priority
        # Default: YouTube -> Pexels -> Pixabay
        source_priority = config.get("broll_source_priority", ["youtube", "pexels", "pixabay"])
        youtube_broll_max_dur = config.get("youtube_broll_max_duration", 120)
        youtube_broll_max_res = config.get("youtube_broll_max_results", 15)
        all_sources = {
            "youtube": lambda: YouTubeVideoSource(max_results=youtube_broll_max_res, max_duration=youtube_broll_max_dur),
            "pexels": lambda: PexelsVideoSource(),
            "pixabay": lambda: PixabayVideoSource(),
        }
        self.video_sources = []
        for name in source_priority:
            factory = all_sources.get(name)
            if factory:
                self.video_sources.append(factory())
        # Ensure at least Pexels/Pixabay are present if priority list was empty
        if not self.video_sources:
            self.video_sources = [PexelsVideoSource(), PixabayVideoSource()]

        # Stock image sources: Google web search (primary), then Pexels/Pixabay (secondary)
        self.image_sources = [GoogleImageSource(), PexelsImageSource(), PixabayImageSource()]

        # New video agent modules
        self.script_gen = ScriptGenerator(self.ai)
        self.composer: VideoComposer | None = None  # Initialized per-project

        # Director agent for draft review loop
        director_enabled = config.get("director_review_enabled", True)
        self.director: DirectorAgent | None = None
        if director_enabled:
            self.director = DirectorAgent(
                model=config.get("director_model", "gemini-3-pro-preview"),
                approval_threshold=config.get("director_approval_threshold", 7),
            )
        self.director_max_iterations = config.get("director_max_iterations", 2)

        # Output directory
        self.output_dir = Path(config.get("local_output_folder", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature flag: use B-roll processor pipeline (new) vs inline methods (old)
        self.use_processor_broll = config.get("use_processor_broll", True)
        self.video_only = config.get("video_only", False)

        # B-Roll processor services (used when use_processor_broll=True)
        if self.use_processor_broll:
            gemini_key = config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
            self.clip_extractor = ClipExtractor(
                api_key=gemini_key,
                min_clip_duration=float(MIN_BROLL_CLIP_SECONDS),
                max_clip_duration=float(MAX_BROLL_CLIP_SECONDS),
                max_clips_per_video=1,
            )
            self.video_pre_filter = VideoPreFilter()
            self.semantic_verifier = SemanticVerifier() if gemini_key else None
            self.video_search_service = VideoSearchService(
                video_sources=self.video_sources,
                video_prefilter=self.video_pre_filter,
                ai_service=self.ai,
            )
            self.video_downloader = VideoDownloader(
                output_dir=str(self.output_dir / "downloads"),
            )
            self.file_organizer = FileOrganizer(
                base_output_dir=str(self.output_dir),
            )
            self.video_acquisition = VideoAcquisitionService(
                video_downloader=self.video_downloader,
                clip_extractor=self.clip_extractor,
                semantic_verifier=self.semantic_verifier,
                file_organizer=self.file_organizer,
                video_search_service=self.video_search_service,
                ai_service=self.ai,
                clips_per_need_target=1,
                min_clip_duration=float(MIN_BROLL_CLIP_SECONDS),
                max_clip_duration=float(MAX_BROLL_CLIP_SECONDS),
            )
            self.broll_adapter = BRollAdapter()

    async def produce(
        self,
        topic: str,
        voice_ref: str | None = None,
        style: str = "documentary",
        target_duration: int = 8,
        subtitle_style: str = "hormozi",
    ) -> Path:
        """Full autonomous pipeline: topic -> complete video.

        Args:
            topic: Video topic (e.g., "The History of Coffee")
            voice_ref: Optional path to voice reference WAV for cloning
            style: Video style ("documentary", "motivational", "educational", "hormozi")
            target_duration: Target video duration in minutes
            subtitle_style: Caption style ("hormozi", "documentary", "minimal")

        Returns:
            Path to final rendered video
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic[:30].replace(" ", "_").replace("/", "_")
        project_dir = self.output_dir / f"video_{timestamp}_{safe_topic}"
        project_dir.mkdir(parents=True, exist_ok=True)

        # Set up composer to output into project_dir
        self.composer = VideoComposer(output_dir=project_dir)

        logger.info(f"=== VIDEO PRODUCTION START: '{topic}' ===")
        logger.info(f"Style: {style}, Duration: {target_duration}min, Output: {project_dir}")

        # -- Phase 1: Pre-production --
        logger.info("-- Phase 1: Pre-production --")

        # 1. Generate script (sync call, run in thread pool)
        logger.info("Generating script...")
        script = await asyncio.to_thread(
            self.script_gen.generate, topic, style, target_duration
        )
        logger.info(f"Script generated: '{script.title}' with {len(script.scenes)} scenes")

        # Save script JSON for debugging
        self._save_script_json(script, project_dir)

        # 2. Generate narration audio per scene
        logger.info("Generating narration...")
        scene_audios = await self._generate_narration(script, project_dir, voice_ref)

        # 3. Merge scene audio into master track
        master_audio = project_dir / "master_audio.wav"
        self._merge_audio_files(scene_audios, master_audio)
        logger.info(f"Master audio: {master_audio}")

        # 4. Get word-level timestamps for subtitles
        logger.info("Generating word timestamps...")
        word_timings = await self._generate_word_timestamps(master_audio)
        logger.info(f"Got {len(word_timings)} word timings")

        # -- Phase 2: Asset Acquisition (parallel) --
        logger.info("-- Phase 2: Asset Acquisition --")

        broll_task = asyncio.create_task(
            self._acquire_visuals(script, project_dir)
        )
        music_task = asyncio.create_task(
            self._generate_music(script, project_dir)
        )

        visual_paths, music_path = await asyncio.gather(broll_task, music_task)
        logger.info(f"Assets acquired: {len(visual_paths)} visuals, music: {music_path}")

        # -- Phase 3: Assembly --
        logger.info("-- Phase 3: Assembly --")

        # Build timeline
        timeline = self._build_timeline(
            script, scene_audios, visual_paths,
            master_audio, music_path, word_timings,
            subtitle_style, project_dir,
        )

        # Generate subtitle file
        ass_path = await self._generate_subtitles(
            word_timings, subtitle_style, project_dir
        )
        if ass_path:
            timeline.subtitle_path = ass_path

        # -- Phase 4: Director Review Loop --
        if self.director:
            logger.info("-- Phase 4: Director Review Loop --")
            timeline = await self._director_review_loop(
                script, timeline, project_dir
            )

        # -- Phase 5: Final Composition --
        logger.info("-- Phase 5: Final Composition --")
        logger.info("Composing final video (1080p)...")
        final_video = await self.composer.compose(timeline, draft=False)
        logger.info(f"=== VIDEO COMPLETE: {final_video} ===")

        return final_video

    # ------------------------------------------------------------------
    # Phase 1 helpers
    # ------------------------------------------------------------------

    async def _generate_narration(
        self,
        script: Script,
        project_dir: Path,
        voice_ref: str | None,
    ) -> list[Path]:
        """Generate TTS audio for each scene.

        Returns list of audio file paths in scene order.
        Generates hook audio first, then each scene.
        """
        audio_dir = project_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        audio_paths: list[Path] = []

        # Build ordered list: hook first, then scenes
        all_texts = [script.hook.voiceover] + [s.voiceover for s in script.scenes]

        for i, text in enumerate(all_texts):
            label = "hook" if i == 0 else f"scene_{i:03d}"
            audio_path = audio_dir / f"{label}.wav"

            # Preprocess text for TTS (number-to-words, abbreviation expansion, etc.)
            text = self._preprocess_for_tts(text)

            logger.info(f"TTS [{label}]: {len(text)} chars")

            try:
                audio_bytes = await self._tts_generate(text, voice_ref)
                # TTS may return MP3 or other formats; ensure WAV for merging
                raw_path = audio_dir / f"{label}_raw"
                raw_path.write_bytes(audio_bytes)
                self._ensure_wav(raw_path, audio_path)
                raw_path.unlink(missing_ok=True)
                audio_paths.append(audio_path)
            except Exception as e:
                logger.error(f"TTS failed for {label}: {e}")
                # Create a short silence placeholder so timeline stays aligned
                self._write_silence_wav(audio_path, duration_seconds=2.0)
                audio_paths.append(audio_path)

        return audio_paths

    async def _tts_generate(self, text: str, voice_ref: str | None) -> bytes:
        """Generate TTS audio bytes using best available backend.

        Tries backends in priority order:
        1. Chatterbox Extended (custom RunPod endpoint)
        2. RunPod custom endpoint
        3. Public Chatterbox Turbo endpoint
        """
        if self.tts.is_chatterbox_ext_configured():
            # Base model — strip paralinguistic tags (not supported)
            clean_text = self._PARALINGUISTIC_TAG_RE.sub("", text).strip()
            return await self.tts.generate_chatterbox_extended(
                text=clean_text,
                voice_ref_path=voice_ref,
                exaggeration=0.4,
                cfg_weight=0.5,
                temperature=0.7,
            )

        if self.tts.is_runpod_configured():
            # Base model — strip paralinguistic tags (not supported)
            clean_text = self._PARALINGUISTIC_TAG_RE.sub("", text).strip()
            return await self.tts.generate_runpod(
                text=clean_text,
                voice_ref_path=voice_ref,
            )

        if self.tts.is_public_endpoint_configured():
            # Public Chatterbox Turbo supports paralinguistic tags natively
            audio_bytes, _cost = await self.tts.generate_public_audio(text)
            return audio_bytes

        raise RuntimeError(
            "No TTS backend configured. Set RUNPOD_API_KEY in .env"
        )

    def _merge_audio_files(self, audio_paths: list[Path], output_path: Path) -> None:
        """Merge multiple WAV files into a single master audio file.

        Simple concatenation of WAV data. All chunks should have same params
        since they come from the same TTS engine.
        """
        if not audio_paths:
            raise ValueError("No audio files to merge")

        frames: list[bytes] = []
        params: tuple | None = None

        for path in audio_paths:
            with wave.open(str(path), "rb") as wf:
                current_params = (
                    wf.getnchannels(),
                    wf.getsampwidth(),
                    wf.getframerate(),
                )
                if params is None:
                    params = current_params
                elif current_params != params:
                    logger.warning(
                        f"Audio param mismatch in {path.name}: "
                        f"{current_params} vs {params}, resampling may be needed"
                    )
                frames.append(wf.readframes(wf.getnframes()))

        if params is None:
            raise ValueError("Could not read audio parameters")

        with wave.open(str(output_path), "wb") as wf_out:
            wf_out.setnchannels(params[0])
            wf_out.setsampwidth(params[1])
            wf_out.setframerate(params[2])
            wf_out.writeframes(b"".join(frames))

        logger.info(
            f"Merged {len(audio_paths)} audio files -> {output_path.name}"
        )

    async def _generate_word_timestamps(
        self, master_audio: Path
    ) -> list[WordTiming]:
        """Generate word-level timestamps from the master audio.

        Uses the SubtitleEngine if available, otherwise falls back to
        a simple uniform distribution based on word count and audio duration.
        """
        try:
            from video_agent.subtitle_engine import SubtitleEngine
            engine = SubtitleEngine()
            return await engine.generate_word_timestamps(master_audio)
        except ImportError:
            logger.warning(
                "SubtitleEngine not available, using uniform word timing fallback"
            )
            return self._fallback_word_timings(master_audio)
        except Exception as e:
            logger.warning(f"Word timestamp generation failed: {e}, using fallback")
            return self._fallback_word_timings(master_audio)

    def _fallback_word_timings(self, audio_path: Path) -> list[WordTiming]:
        """Generate approximate word timings from audio duration.

        Simple heuristic: assumes ~3 words/second and distributes evenly.
        """
        duration = self._get_wav_duration(audio_path)
        # We don't have the transcript text here, so return empty
        # The subtitle engine is the proper way to get word timings
        return []

    # ------------------------------------------------------------------
    # Phase 2 helpers
    # ------------------------------------------------------------------

    async def _acquire_visuals(
        self,
        script: Script,
        project_dir: Path,
        progress_callback: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> dict[int, Path]:
        """Acquire visual assets for each scene using director-guided type selection.

        For each scene, fetches BOTH a video candidate AND an image candidate
        in parallel. Then the AI director (Gemini 3 Pro) evaluates all candidates
        in context of the full video narrative and decides which type works best
        per scene.

        Args:
            script: The video script.
            project_dir: Project output directory.
            progress_callback: Optional async callback(percent, message) for progress updates.

        Returns:
            Dict mapping scene_id -> visual_path.
        """
        visuals_dir = project_dir / "visuals"
        visuals_dir.mkdir(exist_ok=True)
        visual_paths: dict[int, Path] = {}

        sem = asyncio.Semaphore(6)

        # -- Build scene list (hook = scene 0) --
        # Tuple: (scene_id, keywords, style, voiceover, SceneScript|None)

        all_scenes: list[tuple[int, list[str], str, str, SceneScript | None]] = [
            (0, script.hook.visual_keywords,
             getattr(script.hook, "visual_style", "cinematic"),
             script.hook.voiceover, None),
        ]
        all_scenes += [
            (s.id, s.visual_keywords, s.visual_style, s.voiceover, s)
            for s in script.scenes
        ]

        total_count = 2 * len(all_scenes)
        completed_count = 0

        # -- Fetch helpers with timeout + progress reporting --

        async def fetch_and_report_video(
            sid: int, kw: list[str], style: str, vo: str,
            scene: SceneScript | None = None,
        ) -> tuple[int, Path | None]:
            nonlocal completed_count
            path: Path | None = None
            async with sem:
                try:
                    path = await asyncio.wait_for(
                        self._search_and_download_broll(
                            kw,
                            visuals_dir / f"scene_{sid:03d}_video.mp4",
                            visual_style=style,
                            voiceover_context=vo,
                            scene=scene,
                            script=script,
                        ),
                        timeout=90.0,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    logger.warning(f"Video candidate TIMED OUT for scene {sid} (90s)")
                except Exception as e:
                    logger.error(f"Video candidate failed for scene {sid}: {e}")
            completed_count += 1
            if progress_callback:
                pct = 40 + int((completed_count / total_count) * 22)
                status = "acquired" if path else "skipped"
                await progress_callback(
                    pct,
                    f"Scene {sid} video: {status} ({completed_count}/{total_count})",
                )
            return sid, path

        async def fetch_and_report_image(
            sid: int, kw: list[str], style: str,
        ) -> tuple[int, Path | None]:
            nonlocal completed_count
            path: Path | None = None
            async with sem:
                try:
                    path = await asyncio.wait_for(
                        self._generate_image(
                            kw,
                            style,
                            visuals_dir / f"scene_{sid:03d}_image.png",
                        ),
                        timeout=60.0,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    logger.warning(f"Image candidate TIMED OUT for scene {sid} (60s)")
                except Exception as e:
                    logger.error(f"Image candidate failed for scene {sid}: {e}")
            completed_count += 1
            if progress_callback:
                pct = 40 + int((completed_count / total_count) * 22)
                status = "acquired" if path else "skipped"
                await progress_callback(
                    pct,
                    f"Scene {sid} image: {status} ({completed_count}/{total_count})",
                )
            return sid, path

        # -- Phase 1: Run ALL fetches in parallel --

        if self.video_only:
            # Video-only mode: skip image generation entirely
            total_count = len(all_scenes)
            all_results = await asyncio.gather(
                *[fetch_and_report_video(sid, kw, style, vo, scene=sc)
                  for sid, kw, style, vo, sc in all_scenes],
                return_exceptions=True,
            )
            video_results = all_results
            image_results = []
        else:
            all_results = await asyncio.gather(
                *[fetch_and_report_video(sid, kw, style, vo, scene=sc)
                  for sid, kw, style, vo, sc in all_scenes],
                *[fetch_and_report_image(sid, kw, style)
                  for sid, kw, style, _vo, _sc in all_scenes],
                return_exceptions=True,
            )
            n = len(all_scenes)
            video_results = all_results[:n]
            image_results = all_results[n:]

        video_candidates: dict[int, Path | None] = {}
        image_candidates: dict[int, Path | None] = {}

        for result in video_results:
            if isinstance(result, Exception):
                logger.error(f"Video candidate task exception: {result}")
                continue
            sid, path = result
            video_candidates[sid] = path

        for result in image_results:
            if isinstance(result, Exception):
                logger.error(f"Image candidate task exception: {result}")
                continue
            sid, path = result
            image_candidates[sid] = path

        videos_found = sum(1 for v in video_candidates.values() if v)
        images_found = sum(1 for v in image_candidates.values() if v)
        logger.info(
            f"Dual candidates fetched: {videos_found} videos, {images_found} images"
            + (" (video-only mode)" if self.video_only else "")
        )

        # -- Phase 2: Director decides which type to use per scene --

        candidates_info: dict[int, dict] = {}
        for scene_id, kw, _style, _vo, _sc in all_scenes:
            v_path = video_candidates.get(scene_id)
            i_path = image_candidates.get(scene_id)
            candidates_info[scene_id] = {
                "has_video": v_path is not None,
                "has_image": i_path is not None,
                "video_desc": (
                    f"Video clip from search: {' '.join(kw[:3])}"
                    if v_path else "No video found"
                ),
                "image_desc": (
                    f"Stock/AI image for: {' '.join(kw[:3])}"
                    if i_path else "No image found"
                ),
            }

        scenes_with_both = [
            sid for sid, info in candidates_info.items()
            if info["has_video"] and info["has_image"]
        ]

        decision_map: dict[int, VisualTypeDecision] = {}
        if self.video_only:
            # Video-only mode: force all scenes to use video, skip director
            logger.info("Video-only mode: skipping director visual type decisions")
        elif scenes_with_both and self.director:
            if progress_callback:
                await progress_callback(
                    63,
                    f"Director choosing visual types for {len(scenes_with_both)} scenes...",
                )
            try:
                decisions = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.director.decide_visual_types, script, candidates_info
                    ),
                    timeout=30.0,
                )
                decision_map = {d.scene_id: d for d in decisions}
                if progress_callback:
                    vid_n = sum(1 for d in decisions if d.chosen_type == VisualType.BROLL_VIDEO)
                    img_n = sum(1 for d in decisions if d.chosen_type == VisualType.GENERATED_IMAGE)
                    await progress_callback(
                        65, f"Director decided: {vid_n} videos, {img_n} images"
                    )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "Director visual type decision timed out (30s). Using fallback."
                )
            except Exception as e:
                logger.warning(
                    f"Director visual type decision failed: {e}. Using fallback."
                )

        # -- Phase 3: Apply decisions and clean up unchosen --

        for scene_id, _kw, _style, _vo, _sc in all_scenes:
            v_path = video_candidates.get(scene_id)
            i_path = image_candidates.get(scene_id)
            decision = decision_map.get(scene_id)

            if decision and v_path and i_path:
                # Director made a choice
                if decision.chosen_type == VisualType.BROLL_VIDEO:
                    final_path = visuals_dir / f"scene_{scene_id:03d}.mp4"
                    if v_path != final_path:
                        v_path.rename(final_path)
                    if i_path and i_path.exists():
                        i_path.unlink(missing_ok=True)
                    visual_paths[scene_id] = final_path
                    logger.info(
                        f"Scene {scene_id}: Director chose VIDEO "
                        f"({decision.rationale[:60]})"
                    )
                else:
                    suffix = i_path.suffix if i_path else ".png"
                    final_path = visuals_dir / f"scene_{scene_id:03d}{suffix}"
                    if i_path != final_path:
                        i_path.rename(final_path)
                    if v_path and v_path.exists():
                        v_path.unlink(missing_ok=True)
                    visual_paths[scene_id] = final_path
                    logger.info(
                        f"Scene {scene_id}: Director chose IMAGE "
                        f"({decision.rationale[:60]})"
                    )
            elif v_path:
                # Only video available
                final_path = visuals_dir / f"scene_{scene_id:03d}.mp4"
                if v_path != final_path:
                    v_path.rename(final_path)
                visual_paths[scene_id] = final_path
            elif i_path:
                # Only image available
                suffix = i_path.suffix if i_path else ".png"
                final_path = visuals_dir / f"scene_{scene_id:03d}{suffix}"
                if i_path != final_path:
                    i_path.rename(final_path)
                visual_paths[scene_id] = final_path
            else:
                logger.warning(f"Scene {scene_id}: No visual candidates found")

        # -- Phase 4: Fallback for TEXT_GRAPHIC scenes with no candidates --
        for scene in script.scenes:
            if scene.id not in visual_paths and scene.visual_type == VisualType.TEXT_GRAPHIC:
                text_path = visuals_dir / f"scene_{scene.id:03d}.txt"
                text_content = self._make_display_text(
                    scene.visual_keywords, scene.voiceover
                )
                text_path.write_text(text_content)
                visual_paths[scene.id] = text_path

        if progress_callback:
            await progress_callback(
                68, f"Visuals finalized: {len(visual_paths)} assets ready"
            )

        return visual_paths

    @staticmethod
    def _make_display_text(keywords: list[str], voiceover: str) -> str:
        """Create clean display text for text_graphic scenes.

        Uses the first keyword as a title if it's short enough,
        otherwise extracts a short summary from the voiceover.
        Avoids dumping raw search terms on screen.
        """
        # If the first keyword looks like a clean title (short, no search noise), use it
        if keywords:
            first_kw = keywords[0].strip()
            # Use if it's a short, title-like phrase (under 5 words, no filler)
            word_count = len(first_kw.split())
            if 1 <= word_count <= 5:
                return first_kw.title()

        # Extract a short title from the voiceover (first sentence, truncated)
        if voiceover:
            # Take first sentence or first 40 chars
            first_sentence = voiceover.split(".")[0].split("!")[0].split("?")[0].strip()
            if len(first_sentence) <= 40:
                return first_sentence
            # Truncate to last full word within 40 chars
            truncated = first_sentence[:40].rsplit(" ", 1)[0]
            return truncated

        return "..."

    @staticmethod
    def _preprocess_for_tts(text: str) -> str:
        """Preprocess script text for TTS synthesis.

        Safety net that catches what the LLM prompt misses:
        - Normalizes whitespace
        - Replaces problematic characters (em dashes, semicolons, etc.)
        - Expands abbreviations (AI → A.I.)
        - Converts numbers to words ($500 → five hundred dollars)
        - Ensures terminal punctuation on every sentence
        - Preserves paralinguistic tags ([laugh], [sigh], etc.)
        """
        import re as _re

        if not text or not text.strip():
            return text

        # Protect paralinguistic tags with placeholders
        tag_pattern = _re.compile(r"\[(?:laugh|sigh|gasp|chuckle|cough|sniff|groan)\]")
        tags_found: list[str] = []

        def _save_tag(m: _re.Match) -> str:
            tags_found.append(m.group(0))
            return f"__TAG{len(tags_found) - 1}__"

        text = tag_pattern.sub(_save_tag, text)

        # --- Character replacements ---
        text = text.replace("\u2014", ", ")   # em dash
        text = text.replace("\u2013", ", ")   # en dash
        text = text.replace("\u2026", ".")    # ellipsis character
        text = text.replace("...", ".")       # triple dot
        text = text.replace(";", ".")
        text = text.replace("&", " and ")
        text = text.replace("(", ", ").replace(")", ", ")
        text = text.replace("#", " number ")

        # Slashes between words → "or" (but not in URLs or paths)
        text = _re.sub(r"(?<=[a-zA-Z])/(?=[a-zA-Z])", " or ", text)

        # --- Currency ---
        def _currency_replace(m: _re.Match) -> str:
            num_str = m.group(1).replace(",", "")
            suffix = m.group(2).strip() if m.group(2) else ""
            try:
                num = int(num_str)
                word = _number_to_words(num)
            except ValueError:
                word = num_str
            if suffix:
                return f"{word} {suffix} dollars"
            return f"{word} dollars"

        text = _re.sub(
            r"\$(\d[\d,]*)\s*(billion|million|thousand|hundred)?",
            _currency_replace,
            text,
            flags=_re.IGNORECASE,
        )

        # --- Percentages ---
        def _percent_replace(m: _re.Match) -> str:
            try:
                num = int(m.group(1))
                return f"{_number_to_words(num)} percent"
            except ValueError:
                return m.group(0)

        text = _re.sub(r"(\d+)%", _percent_replace, text)

        # --- Number ranges (e.g., 15-20) ---
        def _range_replace(m: _re.Match) -> str:
            try:
                a, b = int(m.group(1)), int(m.group(2))
                return f"{_number_to_words(a)} to {_number_to_words(b)}"
            except ValueError:
                return m.group(0)

        text = _re.sub(r"\b(\d{1,4})-(\d{1,4})\b", _range_replace, text)

        # --- Years (standalone 4-digit numbers 1000-2099) ---
        def _year_replace(m: _re.Match) -> str:
            y = int(m.group(0))
            if 1000 <= y <= 2099:
                return _year_to_words(y)
            return m.group(0)

        text = _re.sub(r"\b([12]\d{3})\b", _year_replace, text)

        # --- Remaining standalone numbers ---
        def _num_replace(m: _re.Match) -> str:
            try:
                n = int(m.group(0).replace(",", ""))
                return _number_to_words(n)
            except ValueError:
                return m.group(0)

        text = _re.sub(r"\b\d[\d,]*\b", _num_replace, text)

        # --- Abbreviation expansion (word-boundary, case-sensitive) ---
        for abbr, expanded in _ABBREVIATION_MAP.items():
            text = _re.sub(rf"\b{abbr}\b", expanded, text)

        # --- Whitespace normalization ---
        text = _re.sub(r"[ \t]+", " ", text)
        text = _re.sub(r" ,", ",", text)   # fix space before comma
        text = _re.sub(r",{2,}", ",", text)  # fix double commas
        text = _re.sub(r"\.{2,}", ".", text)  # fix double periods
        text = text.strip()

        # --- Terminal punctuation enforcement ---
        sentences = _re.split(r"(?<=[.!?])\s+", text)
        fixed_sentences: list[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Don't add punctuation to tag placeholders standing alone
            if _re.match(r"^__TAG\d+__$", s):
                fixed_sentences.append(s)
                continue
            if s and s[-1] not in ".!?":
                s += "."
            fixed_sentences.append(s)
        text = " ".join(fixed_sentences)

        # --- Restore paralinguistic tags ---
        for i, tag in enumerate(tags_found):
            text = text.replace(f"__TAG{i}__", tag)

        return text.strip()

    async def _search_and_download_broll(
        self,
        keywords: list[str],
        output_path: Path,
        visual_style: str = "",
        voiceover_context: str = "",
        scene: SceneScript | None = None,
        script: Script | None = None,
    ) -> Path | None:
        """Search video sources and download best B-roll clip.

        Uses the B-roll processor pipeline (VideoAcquisitionService) when
        use_processor_broll=True, with fallback to the legacy inline pipeline.

        When a SceneScript is provided, uses BRollAdapter for richer metadata.
        Otherwise builds a synthetic BRollNeed from keywords.

        Args:
            keywords: Visual keywords to search for
            output_path: Where to save the downloaded clip
            visual_style: Optional style hint for search enhancement
            voiceover_context: Optional voiceover text for AI evaluation context
            scene: Optional SceneScript for BRollAdapter conversion
            script: Optional full Script for timestamp calculation

        Returns:
            Path to downloaded clip, or None if all sources failed
        """
        if not self.use_processor_broll:
            logger.warning("[B-roll] Processor pipeline disabled, skipping B-roll search")
            return None
        return await self._search_broll_via_processor(
            keywords, output_path, visual_style, voiceover_context,
            scene=scene, script=script,
        )

    async def _search_broll_via_processor(
        self,
        keywords: list[str],
        output_path: Path,
        visual_style: str = "",
        voiceover_context: str = "",
        scene: SceneScript | None = None,
        script: Script | None = None,
    ) -> Path | None:
        """Search and download B-roll using the B-roll processor pipeline.

        When a SceneScript + Script are provided, uses BRollAdapter for
        rich metadata (alternate searches, required elements, timestamps).
        Otherwise builds a synthetic BRollNeed from the keywords.

        Args:
            keywords: Visual keywords to search for
            output_path: Where to save the downloaded clip
            visual_style: Optional style hint
            voiceover_context: Optional voiceover text for context
            scene: Optional SceneScript for adapter conversion
            script: Optional full Script for timestamp calculation

        Returns:
            Path to downloaded clip, or None if failed
        """
        from models.broll_need import BRollNeed

        # Use BRollAdapter when we have a full scene + script
        if scene is not None and script is not None:
            need = self.broll_adapter.scene_to_broll_need(scene, script)
        else:
            # Build a synthetic BRollNeed for hook or fix-up calls
            search_phrase = " ".join(keywords[:3])
            need = BRollNeed(
                timestamp=0.0,
                search_phrase=search_phrase,
                description=voiceover_context[:200] if voiceover_context else search_phrase,
                context=voiceover_context,
                suggested_duration=float(TARGET_BROLL_CLIP_SECONDS),
                original_context=voiceover_context,
                alternate_searches=[
                    f"{keywords[0]} footage" if keywords else "",
                    f"cinematic {search_phrase}",
                ],
                negative_keywords=["compilation", "reaction", "vlog", "meme", "review"],
                visual_style=visual_style if visual_style else None,
            )

        project_dir = str(output_path.parent)

        try:
            result = await self.video_acquisition.process_single_need(
                need=need,
                need_index=1,
                total_needs=1,
                project_dir=project_dir,
                video_prefilter=self.video_pre_filter,
            )

            if result:
                _folder_name, clip_files = result
                if clip_files:
                    first_clip = Path(clip_files[0])
                    if first_clip.exists():
                        import shutil
                        shutil.move(str(first_clip), str(output_path))
                        logger.info(
                            f"[B-roll] Processor pipeline: '{need.search_phrase}' "
                            f"-> {output_path.name}"
                        )
                        return output_path

        except Exception as e:
            logger.warning(f"[B-roll] Processor pipeline failed: {e}")

        logger.warning(f"[B-roll] No B-roll found via processor for: {need.search_phrase}")
        return None

    async def _evaluate_image_candidates(
        self,
        candidates: list[ImageResult],
        keywords: list[str],
        visual_style: str = "",
    ) -> int:
        """Score stock image candidates using Gemini Flash and pick the best.

        Args:
            candidates: Image search results to evaluate
            keywords: Visual keywords for the scene
            visual_style: Requested visual style

        Returns:
            Index of the best candidate (0-indexed), defaults to 0
        """
        if not candidates or len(candidates) <= 1:
            return 0

        search_query = " ".join(keywords[:3])

        # Format candidates for the prompt
        candidates_text = "\n".join(
            f"[{i}] Title: {c.title[:60]}, Source: {c.source}, "
            f"Resolution: {c.width}x{c.height}, "
            f"Description: {(c.description or 'N/A')[:80]}"
            for i, c in enumerate(candidates)
        )

        prompt = VIDEO_AGENT_IMAGE_EVALUATOR.format(
            search_query=search_query,
            visual_keywords=", ".join(keywords),
            visual_style=visual_style or "cinematic",
            candidates_text=candidates_text,
        )

        try:
            from google.genai import types

            response = await asyncio.to_thread(
                self.ai.client.models.generate_content,
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )

            if not response.text:
                return 0

            text = strip_markdown_code_blocks(response.text)
            result = json.loads(text)
            best_index = int(result.get("best_index", 0))

            if 0 <= best_index < len(candidates):
                logger.debug(
                    f"[Image eval] Selected #{best_index}: "
                    f"{result.get('reason', '')[:60]}"
                )
                return best_index

        except Exception as e:
            logger.warning(f"[Image eval] Evaluation failed: {e}")

        return 0

    def _is_style_enhancement_enabled(self) -> bool:
        """Check if Nano Banana Pro style enhancement is enabled.

        Controlled by IMAGE_STYLE_ENHANCEMENT config:
        - "auto" (default): enabled if RUNPOD_API_KEY is configured
        - "true": always enabled (will fail if no RUNPOD_API_KEY)
        - "false": always disabled
        """
        setting = self.config.get("image_style_enhancement", "auto")
        if setting == "true":
            return True
        if setting == "false":
            return False
        # "auto": enabled when RunPod is configured
        return self.image_gen.is_runpod_configured()

    async def _generate_image(
        self,
        keywords: list[str],
        style: str,
        output_path: Path,
    ) -> Path | None:
        """Acquire an image with stock-first fallback chain.

        Priority order:
        1. Stock photo from Pexels/Pixabay (FREE)
           -> Optional Nano Banana Pro style enhancement ($0.04)
        2. Gemini Flash text-to-image (FREE, 500/day)
        3. Runware Flux Klein 4B ($0.0006)

        Returns None only if all backends fail.
        """
        search_query = " ".join(keywords[:3])

        # --- Attempt 1: Stock photo (FREE) + optional style enhancement ---
        stock_path = await self._search_stock_image(
            search_query, output_path, keywords=keywords, visual_style=style
        )
        if stock_path:
            logger.info(f"[Image] Stock photo found: '{search_query}'")
            # Optionally enhance with Nano Banana Pro
            if self._is_style_enhancement_enabled():
                enhanced = await self._enhance_image_style(
                    stock_path, keywords, style
                )
                if enhanced:
                    return enhanced
            return stock_path

        # --- Attempt 2: Gemini Flash (FREE) ---
        prompt = f"{style} photograph of {', '.join(keywords)}, high quality, cinematic, 16:9"
        if self.image_gen.is_gemini_configured():
            try:
                request = ImageGenerationRequest(
                    prompt=prompt,
                    model=ImageGenerationModel.GEMINI_FLASH,
                    width=1920,
                    height=1080,
                )
                result = await self.image_gen.generate_gemini(request)
                saved = await self._save_generated_image(result, output_path)
                if saved:
                    logger.info(f"[Image] Gemini generated: {prompt[:50]}...")
                    return output_path
            except Exception as e:
                logger.warning(f"[Image] Gemini generation failed: {e}")

        # --- Attempt 3: Runware Flux Klein ($0.0006) ---
        if self.image_gen.is_runware_configured():
            try:
                request = ImageGenerationRequest(
                    prompt=prompt,
                    model=ImageGenerationModel.RUNWARE_FLUX_KLEIN_4B,
                    width=1920,
                    height=1080,
                )
                result = await self.image_gen.generate_runware(request)
                saved = await self._save_generated_image(result, output_path)
                if saved:
                    logger.info(f"[Image] Runware generated: {prompt[:50]}...")
                    return output_path
            except Exception as e:
                logger.warning(f"[Image] Runware generation failed: {e}")

        logger.warning(f"[Image] All image sources failed for: {search_query}")
        return None

    async def _enhance_image_style(
        self,
        stock_path: Path,
        keywords: list[str],
        style: str,
    ) -> Path | None:
        """Enhance a stock photo with Nano Banana Pro styling.

        Converts the local file to a base64 data URL, sends it to
        Nano Banana Pro with a style prompt, and saves the result.
        Returns the enhanced image path, or None on failure (graceful).
        """
        try:
            # Build style prompt from scene context
            kw_text = ", ".join(keywords)
            style_prompt = (
                f"{style} style, {kw_text}, "
                "cinematic lighting, high quality, 16:9 aspect ratio"
            )

            # Convert local file to data URL for the edit API
            image_bytes = stock_path.read_bytes()
            b64 = base64.b64encode(image_bytes).decode()
            suffix = stock_path.suffix.lower()
            mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            data_url = f"data:{mime};base64,{b64}"

            edit_request = ImageEditRequest(
                prompt=style_prompt,
                input_image_url=data_url,
                model=ImageGenerationModel.NANO_BANANA_PRO,
                strength=0.75,
            )
            result = await self.image_gen.edit_runpod(edit_request)

            # Save the enhanced image
            enhanced_path = stock_path.with_stem(stock_path.stem + "_styled")
            saved = await self._save_generated_image(result, enhanced_path)
            if saved:
                logger.info(
                    f"[Image] Nano Banana Pro enhanced: "
                    f"'{style_prompt[:60]}...' -> {enhanced_path.name}"
                )
                return enhanced_path

        except Exception as e:
            logger.warning(
                f"[Image] Nano Banana Pro enhancement failed, "
                f"using raw stock photo: {e}"
            )

        return None

    async def _save_generated_image(
        self,
        result,
        output_path: Path,
    ) -> bool:
        """Save a generated image result to disk.

        Handles both data URLs (Gemini) and HTTP URLs (Runware).
        Returns True if saved successfully.
        """
        if not result or not result.images:
            return False

        image = result.images[0]
        try:
            if image.url.startswith("data:"):
                b64_data = image.url.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_data)
            else:
                import httpx
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.get(image.url)
                    resp.raise_for_status()
                    image_bytes = resp.content

            if len(image_bytes) > 1024:
                output_path.write_bytes(image_bytes)
                return True
        except Exception as e:
            logger.warning(f"[Image] Failed to save image: {e}")

        return False

    async def _search_stock_image(
        self,
        query: str,
        output_path: Path,
        keywords: list[str] | None = None,
        visual_style: str = "",
    ) -> Path | None:
        """Search Google/Pexels/Pixabay for a stock photo and download it.

        Used as a free fallback when AI image generation fails.
        When multiple results are found, uses AI evaluation to pick the best one.
        """
        import httpx

        _watermark_domains = {
            "shutterstock", "gettyimages", "istockphoto",
            "dreamstime", "depositphotos", "123rf", "alamy",
            "bigstockphoto", "adobestock", "stock.adobe",
        }

        for source in self.image_sources:
            if not source.is_configured():
                continue

            source_name = source.get_source_name()
            try:
                results = await source.search_images(query, per_page=5)
                if not results:
                    continue

                # Filter out watermarked stock site images
                clean_results: list[ImageResult] = []
                for img_result in results:
                    if not img_result.download_url:
                        continue
                    url_lower = img_result.download_url.lower()
                    if any(domain in url_lower for domain in _watermark_domains):
                        logger.debug(f"[Image] Skipping watermarked source: {url_lower[:80]}")
                        continue
                    clean_results.append(img_result)

                if not clean_results:
                    continue

                # AI-evaluate candidates to pick the best one
                if len(clean_results) > 1 and keywords:
                    best_idx = await self._evaluate_image_candidates(
                        clean_results, keywords, visual_style
                    )
                else:
                    best_idx = 0

                # Reorder so AI-selected image is first, then try others as fallback
                ordered = [clean_results[best_idx]] + [
                    r for i, r in enumerate(clean_results) if i != best_idx
                ]

                for img_result in ordered:
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            resp = await client.get(img_result.download_url)
                            if resp.status_code == 200 and len(resp.content) > 1024:
                                # Ensure output has correct extension
                                content_type = resp.headers.get("content-type", "")
                                if "jpeg" in content_type or "jpg" in content_type:
                                    final_path = output_path.with_suffix(".jpg")
                                else:
                                    final_path = output_path.with_suffix(".png")
                                final_path.write_bytes(resp.content)
                                logger.info(
                                    f"[Image] Stock photo from {source_name}: "
                                    f"'{query}' ({len(resp.content) / 1024:.0f} KB)"
                                )
                                return final_path
                    except Exception as dl_err:
                        logger.warning(
                            f"[Image] Stock download failed ({source_name}): {dl_err}"
                        )
                        continue

            except Exception as e:
                logger.warning(f"[Image] Stock search failed ({source_name}): {e}")
                continue

        return None

    async def _generate_music(
        self,
        script: Script,
        project_dir: Path,
    ) -> Path | None:
        """Generate background music using MusicService (Stable Audio 2.5).

        Determines mood from script metadata and scene music_moods.
        """
        if not self.music.is_configured():
            logger.warning("Music service not configured (REPLICATE_API_KEY missing)")
            return None

        # Determine dominant mood from scenes
        moods = [s.music_mood for s in script.scenes]
        mood_counts: dict[str, int] = {}
        for m in moods:
            mood_counts[m] = mood_counts.get(m, 0) + 1
        dominant_mood = max(mood_counts, key=mood_counts.get) if mood_counts else "neutral"

        # Map mood to music prompt
        mood_to_prompt = {
            "tense": "dark ambient cinematic tension suspense",
            "uplifting": "uplifting inspirational cinematic motivational",
            "neutral": "ambient background cinematic subtle",
            "dark": "dark moody cinematic atmospheric",
            "energetic": "energetic upbeat cinematic driving",
            "hopeful": "hopeful uplifting cinematic warm",
            "mysterious": "mysterious ambient cinematic ethereal",
        }
        music_prompt = mood_to_prompt.get(dominant_mood, "ambient cinematic background")

        # Estimate total duration from scene audio estimates + hook buffer
        total_duration = sum(s.duration_est for s in script.scenes) + 10
        total_duration = min(int(total_duration), 190)  # Stable Audio max is 190s

        try:
            logger.info(f"Generating music: '{music_prompt}' ({total_duration}s)")
            audio_bytes = await self.music.generate_music(
                genres=music_prompt,
                output_seconds=total_duration,
            )
            music_path = project_dir / "background_music.wav"
            music_path.write_bytes(audio_bytes)
            return music_path
        except Exception as e:
            logger.warning(f"Music generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Phase 3 helpers
    # ------------------------------------------------------------------

    def _build_timeline(
        self,
        script: Script,
        scene_audios: list[Path],
        visual_paths: dict[int, Path],
        master_audio: Path,
        music_path: Path | None,
        word_timings: list[WordTiming],
        subtitle_style: str,
        project_dir: Path,
    ) -> Timeline:
        """Build a Timeline object from script + acquired assets.

        Maps each scene to its audio segment, visual asset, and word timings.
        Calculates audio_start/audio_end for each scene based on WAV durations.
        """
        # Calculate cumulative audio timestamps from scene audio files
        cumulative_offset = 0.0
        scene_boundaries: list[tuple[float, float]] = []

        for audio_path in scene_audios:
            duration = self._get_wav_duration(audio_path)
            scene_boundaries.append((cumulative_offset, cumulative_offset + duration))
            cumulative_offset += duration

        # Build TimelineScene objects
        # scene_audios[0] = hook, scene_audios[1..] = script.scenes
        timeline_scenes: list[TimelineScene] = []

        all_scenes_meta = (
            [(0, script.hook.visual_keywords, VisualType.BROLL_VIDEO, "cut", script.hook.sound_effect)]
            + [
                (s.id, s.visual_keywords, s.visual_type, s.transition_in, s.sound_effect)
                for s in script.scenes
            ]
        )

        for i, (scene_id, _keywords, visual_type, transition, sfx) in enumerate(all_scenes_meta):
            if i >= len(scene_boundaries):
                break

            audio_start, audio_end = scene_boundaries[i]
            audio_path = scene_audios[i]

            # Resolve visual path with fallback
            visual_path = visual_paths.get(scene_id)
            if visual_path is None:
                # Create placeholder text graphic
                placeholder = project_dir / "visuals" / f"placeholder_{scene_id:03d}.txt"
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                placeholder.write_text("...")
                visual_path = placeholder
                visual_type = VisualType.TEXT_GRAPHIC
            else:
                # Infer visual_type from file extension - a TEXT_GRAPHIC scene
                # may have been upgraded to GENERATED_IMAGE during acquisition
                ext = visual_path.suffix.lower()
                if ext in (".png", ".jpg", ".jpeg", ".webp"):
                    visual_type = VisualType.GENERATED_IMAGE
                elif ext in (".mp4", ".mkv", ".webm", ".mov"):
                    visual_type = VisualType.BROLL_VIDEO
                elif ext == ".txt":
                    visual_type = VisualType.TEXT_GRAPHIC

            # Collect word timings that fall within this scene's time range
            scene_words = [
                wt for wt in word_timings
                if audio_start <= wt.start < audio_end
            ]

            timeline_scenes.append(
                TimelineScene(
                    scene_id=scene_id,
                    audio_path=audio_path,
                    audio_start=audio_start,
                    audio_end=audio_end,
                    visual_path=visual_path,
                    visual_type=visual_type,
                    transition=transition,
                    sound_effect=sfx,
                    word_timings=scene_words,
                )
            )

        total_duration = cumulative_offset

        return Timeline(
            scenes=timeline_scenes,
            master_audio=master_audio,
            music_path=music_path,
            subtitle_path=None,  # Set later after generation
            color_grade="dark_cinematic",
            total_duration=total_duration,
        )

    async def _generate_subtitles(
        self,
        word_timings: list[WordTiming],
        subtitle_style: str,
        project_dir: Path,
    ) -> Path | None:
        """Generate ASS subtitle file from word timings.

        Returns path to .ass file or None if subtitle engine is unavailable.
        """
        if not word_timings:
            logger.warning("No word timings available, skipping subtitle generation")
            return None

        try:
            from video_agent.subtitle_engine import SubtitleEngine
            engine = SubtitleEngine()

            style_enum = SubtitleStyle(subtitle_style)
            ass_content = engine.generate_ass_subtitles(word_timings, style_enum)
            ass_path = project_dir / "subtitles.ass"
            engine.save_ass_file(ass_content, ass_path)
            logger.info(f"Subtitles generated: {ass_path}")
            return ass_path
        except ImportError:
            logger.warning("SubtitleEngine not available, skipping subtitles")
            return None
        except Exception as e:
            logger.warning(f"Subtitle generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Phase 4: Director review loop
    # ------------------------------------------------------------------

    async def _director_review_loop(
        self,
        script: Script,
        timeline: Timeline,
        project_dir: Path,
    ) -> Timeline:
        """Run the director review loop on draft renders.

        Renders a draft (480p), has the DirectorAgent review it, applies
        fixes if needed, and re-renders. Max iterations from config.

        Args:
            script: The video script
            timeline: Current timeline with resolved assets
            project_dir: Project output directory

        Returns:
            Updated timeline (may have modified visual assets or transitions)
        """
        for iteration in range(self.director_max_iterations):
            logger.info(f"Director review: iteration {iteration + 1}/{self.director_max_iterations}")

            # Render draft at 480p for fast review
            logger.info("Rendering draft (480p) for director review...")
            draft_path = await self.composer.compose(timeline, draft=True)

            # Director reviews the draft
            logger.info("Director reviewing draft...")
            review = self.director.review_draft(
                draft_video_path=str(draft_path),
                script=script,
                timeline=timeline,
                iteration=iteration,
            )

            logger.info(
                f"Director review: score={review.overall_score}/10, "
                f"approved={review.approved}, fixes={len(review.fix_requests)}, "
                f"notes={review.notes[:100]}"
            )

            # Clean up draft file
            if draft_path.exists():
                draft_path.unlink(missing_ok=True)

            if review.approved:
                logger.info("Director APPROVED the draft!")
                break

            # Apply fixes to timeline
            if review.fix_requests:
                logger.info(f"Applying {len(review.fix_requests)} fixes...")
                timeline = await self._apply_fixes(
                    timeline, script, review.fix_requests, project_dir
                )
            else:
                logger.info("No fixes requested but not approved. Proceeding anyway.")
                break

        return timeline

    async def _apply_fixes(
        self,
        timeline: Timeline,
        script: Script,
        fixes: list[FixRequest],
        project_dir: Path,
    ) -> Timeline:
        """Apply director fix requests to the timeline.

        Handles visual_mismatch (re-acquire assets), transition_jarring
        (change transition type), and pacing_issue (log for manual review).

        Args:
            timeline: Current timeline
            script: Original script
            fixes: List of FixRequest objects from director review
            project_dir: Project output directory

        Returns:
            Updated timeline with fixes applied
        """
        visuals_dir = project_dir / "visuals"

        for fix in fixes:
            scene_idx = None
            for i, scene in enumerate(timeline.scenes):
                if scene.scene_id == fix.scene_id:
                    scene_idx = i
                    break

            if scene_idx is None:
                logger.warning(f"Fix references unknown scene_id={fix.scene_id}, skipping")
                continue

            scene = timeline.scenes[scene_idx]

            if fix.issue_type == "visual_mismatch":
                # Re-acquire visual using suggested keywords
                keywords = fix.suggested_keywords if fix.suggested_keywords else []
                if not keywords:
                    # Fall back to original script keywords
                    for s in script.scenes:
                        if s.id == fix.scene_id:
                            keywords = s.visual_keywords
                            break

                if keywords:
                    new_visual = await self._search_and_download_broll(
                        keywords,
                        visuals_dir / f"scene_{fix.scene_id:03d}_fix.mp4",
                    )
                    if new_visual:
                        scene.visual_path = new_visual
                        scene.visual_type = VisualType.BROLL_VIDEO
                        logger.info(
                            f"Fixed scene {fix.scene_id}: replaced visual with new B-roll"
                        )

            elif fix.issue_type == "transition_jarring":
                # Change transition to a smoother option
                if "dissolve" in fix.suggested_fix.lower():
                    scene.transition = "dissolve"
                elif "fade" in fix.suggested_fix.lower():
                    scene.transition = "fade"
                else:
                    scene.transition = "dissolve"  # Safe default
                logger.info(
                    f"Fixed scene {fix.scene_id}: changed transition to {scene.transition}"
                )

            elif fix.issue_type == "pacing_issue":
                # Log pacing issues - auto-fix is too risky for timing
                logger.info(
                    f"Pacing issue in scene {fix.scene_id}: {fix.description} "
                    f"(manual review recommended)"
                )

            elif fix.issue_type == "audio_sync":
                # Audio sync issues can't be auto-fixed
                logger.info(
                    f"Audio sync issue in scene {fix.scene_id}: {fix.description} "
                    f"(manual review recommended)"
                )

            else:
                logger.debug(
                    f"Unhandled fix type '{fix.issue_type}' for scene {fix.scene_id}"
                )

        return timeline

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _ensure_wav(self, input_path: Path, output_path: Path) -> None:
        """Convert any audio file to 24kHz mono WAV using ffmpeg.

        If the file is already valid WAV, copies it directly.
        Otherwise uses ffmpeg to transcode.
        """
        # Check if already valid WAV
        try:
            with wave.open(str(input_path), "rb") as wf:
                wf.getnframes()
            # Valid WAV - just move it
            import shutil
            shutil.copy2(input_path, output_path)
            return
        except Exception:
            pass  # Not valid WAV, convert with ffmpeg

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(input_path),
                    "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )
            logger.info(f"Converted audio to WAV: {output_path.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {e.stderr.decode()[:200]}")
            raise RuntimeError(f"Audio conversion failed: {e.stderr.decode()[:100]}")

    def _get_wav_duration(self, wav_path: Path) -> float:
        """Get duration of a WAV file in seconds."""
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate == 0:
                    return 0.0
                return frames / rate
        except Exception as e:
            logger.warning(f"Could not read WAV duration for {wav_path}: {e}")
            return 0.0

    def _write_silence_wav(
        self, path: Path, duration_seconds: float = 2.0, sample_rate: int = 24000
    ) -> None:
        """Write a silent WAV file as a placeholder."""
        n_frames = int(sample_rate * duration_seconds)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * n_frames)

    def _save_script_json(self, script: Script, project_dir: Path) -> None:
        """Save script data as JSON for debugging."""
        try:
            data = {
                "title": script.title,
                "hook": {
                    "voiceover": script.hook.voiceover,
                    "visual_description": script.hook.visual_description,
                    "visual_keywords": script.hook.visual_keywords,
                },
                "scenes": [
                    {
                        "id": s.id,
                        "duration_est": s.duration_est,
                        "voiceover": s.voiceover,
                        "visual_keywords": s.visual_keywords,
                        "visual_style": s.visual_style,
                        "visual_type": s.visual_type.value,
                        "transition_in": s.transition_in,
                        "music_mood": s.music_mood,
                    }
                    for s in script.scenes
                ],
                "metadata": script.metadata,
            }
            script_path = project_dir / "script.json"
            script_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            logger.info(f"Script saved: {script_path}")
        except Exception as e:
            logger.warning(f"Failed to save script JSON: {e}")

    async def close(self) -> None:
        """Clean up HTTP clients and services."""
        await self.tts.close()
        await self.music.close()
        await self.image_gen.close()
        if self.director:
            await self.director.close()
