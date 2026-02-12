"""Video Production Agent - autonomous video generation pipeline.

Takes a topic and produces a complete faceless YouTube video:
topic -> script -> narration -> assets -> subtitles -> composition -> output
"""

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

from video_agent.models import (
    DraftReview,
    FixRequest,
    Script,
    SceneScript,
    SubtitleStyle,
    Timeline,
    TimelineScene,
    VisualType,
    WordTiming,
)
from video_agent.script_generator import ScriptGenerator
from video_agent.director import DirectorAgent
from video_agent.video_composer import VideoComposer

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
from models.image_generation import ImageEditRequest, ImageGenerationModel, ImageGenerationRequest
from models.video import ScoredVideo, VideoResult
from models.image import ImageResult
from services.prompts._base import strip_markdown_code_blocks
from services.prompts.evaluation import VIDEO_AGENT_BROLL_EVALUATOR, VIDEO_AGENT_IMAGE_EVALUATOR
from utils.config import load_config

logger = logging.getLogger(__name__)

# Maximum B-roll clip duration in seconds
MAX_BROLL_CLIP_SECONDS = 4

# Target clip duration range for Gemini analysis (seconds)
MIN_BROLL_CLIP_SECONDS = 3
TARGET_BROLL_CLIP_SECONDS = 4

# Minimum score threshold for AI-evaluated B-roll candidates (metadata evaluation)
MIN_BROLL_SCORE = 6

# Minimum score for Gemini video analysis segment acceptance
MIN_GEMINI_SEGMENT_SCORE = 7

# Maximum time to wait for Gemini File API processing (seconds)
GEMINI_FILE_POLL_TIMEOUT = 120
GEMINI_FILE_POLL_INTERVAL = 3

# Blocked title keywords for metadata pre-filtering
# Videos with these words in the title are unlikely to be clean B-roll
BLOCKED_TITLE_KEYWORDS = {
    "compilation", "top 10", "top 5", "top 20", "reaction", "review",
    "unboxing", "haul", "vlog", "podcast", "interview", "behind the scenes",
    "gameplay", "let's play", "tutorial", "how to", "explained", "tier list",
    "ranking", "worst", "cringe", "funny moments", "try not to laugh",
}

# Minimum view count to filter out very low quality/spam videos
MIN_VIEW_COUNT = 1000


class VideoProductionAgent:
    """Autonomous video production pipeline.

    Orchestrates all existing stockpile services + new video agent modules
    to produce a complete video from a topic prompt.
    """

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
            return await self.tts.generate_chatterbox_extended(
                text=text,
                voice_ref_path=voice_ref,
                exaggeration=0.4,
                cfg_weight=0.5,
                temperature=0.7,
            )

        if self.tts.is_runpod_configured():
            return await self.tts.generate_runpod(
                text=text,
                voice_ref_path=voice_ref,
            )

        if self.tts.is_public_endpoint_configured():
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
    ) -> dict[int, Path]:
        """Acquire visual assets for each scene.

        Returns dict mapping scene_id -> visual_path.

        For BROLL_VIDEO scenes: Search YouTube -> Pexels -> Pixabay (priority order), download best match
        For GENERATED_IMAGE scenes: Generate with Gemini/Runware, fallback to stock photo
        For TEXT_GRAPHIC scenes: Try generating an image first, fallback to styled text
        """
        visuals_dir = project_dir / "visuals"
        visuals_dir.mkdir(exist_ok=True)
        visual_paths: dict[int, Path] = {}

        sem = asyncio.Semaphore(3)  # Limit concurrent downloads

        async def acquire_one(scene: SceneScript) -> tuple[int, Path | None]:
            async with sem:
                try:
                    if scene.visual_type == VisualType.BROLL_VIDEO:
                        return scene.id, await self._search_and_download_broll(
                            scene.visual_keywords,
                            visuals_dir / f"scene_{scene.id:03d}.mp4",
                            visual_style=getattr(scene, "visual_style", ""),
                            voiceover_context=scene.voiceover,
                        )
                    elif scene.visual_type == VisualType.GENERATED_IMAGE:
                        path = await self._generate_image(
                            scene.visual_keywords,
                            scene.visual_style,
                            visuals_dir / f"scene_{scene.id:03d}.png",
                        )
                        return scene.id, path
                    else:  # TEXT_GRAPHIC
                        # Try generating an image first for text_graphic scenes
                        img_path = await self._generate_image(
                            scene.visual_keywords,
                            scene.visual_style or "minimalist",
                            visuals_dir / f"scene_{scene.id:03d}.png",
                        )
                        if img_path:
                            return scene.id, img_path
                        # Fallback: write clean display text (not raw keywords)
                        text_path = visuals_dir / f"scene_{scene.id:03d}.txt"
                        text_content = self._make_display_text(
                            scene.visual_keywords, scene.voiceover
                        )
                        text_path.write_text(text_content)
                        return scene.id, text_path
                except Exception as e:
                    logger.error(f"Asset acquisition failed for scene {scene.id}: {e}")
                    return scene.id, None

        # Also handle hook visual - respect visual_type if available
        async def acquire_hook() -> tuple[int, Path | None]:
            async with sem:
                try:
                    hook_visual_type = getattr(script.hook, "visual_type", None)
                    if hook_visual_type == VisualType.GENERATED_IMAGE:
                        path = await self._generate_image(
                            script.hook.visual_keywords,
                            getattr(script.hook, "visual_style", "cinematic"),
                            visuals_dir / "scene_000_hook.png",
                        )
                        if path:
                            return 0, path
                    # Default / fallback: search for B-roll video
                    return 0, await self._search_and_download_broll(
                        script.hook.visual_keywords,
                        visuals_dir / "scene_000_hook.mp4",
                        voiceover_context=script.hook.voiceover,
                    )
                except Exception as e:
                    logger.error(f"Hook visual acquisition failed: {e}")
                    return 0, None

        tasks = [acquire_hook()] + [acquire_one(scene) for scene in script.scenes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Asset acquisition task failed: {result}")
                continue
            scene_id, path = result
            if path:
                visual_paths[scene_id] = path

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

    async def _search_and_download_broll(
        self,
        keywords: list[str],
        output_path: Path,
        visual_style: str = "",
        voiceover_context: str = "",
    ) -> Path | None:
        """Search video sources and download best B-roll clip.

        For YouTube, tries multiple query variations before falling through:
          1. Enhanced: keywords + style + "stock footage b-roll"
          2. Simpler: keywords + "cinematic footage"
          3. Bare: keywords only

        All sources go through metadata pre-filtering and AI evaluation
        before downloading. Only videos scoring >= MIN_BROLL_SCORE are tried.

        Only falls through to Pexels/Pixabay if ALL YouTube attempts fail.
        Non-YouTube sources use a single standard search.

        Args:
            keywords: Visual keywords to search for
            output_path: Where to save the downloaded clip
            visual_style: Optional style hint for search enhancement
            voiceover_context: Optional voiceover text for AI evaluation context

        Returns:
            Path to downloaded clip, or None if all sources failed
        """
        import httpx

        search_query = " ".join(keywords[:3])

        for source in self.video_sources:
            source_name = source.get_source_name()
            if not source.is_configured():
                logger.debug(f"[B-roll] Skipping {source_name} (not configured)")
                continue

            try:
                # YouTube: try multiple query variations with max_results=10 each
                if source_name == "youtube" and isinstance(source, YouTubeVideoSource):
                    yt_result = await self._youtube_multi_query_search(
                        source, keywords, visual_style, output_path, search_query,
                        voiceover_context=voiceover_context,
                    )
                    if yt_result:
                        return yt_result
                    # All YouTube attempts exhausted, fall through to next source
                    continue

                # Non-YouTube sources: single standard search
                results = await source.search_videos_async(search_query)

                if not results:
                    logger.debug(f"[B-roll] No results from {source_name} for '{search_query}'")
                    continue

                # Pre-filter and AI-evaluate non-YouTube results
                filtered = self._prefilter_video_results(results)
                if filtered:
                    scored = await self._evaluate_broll_candidates(
                        filtered, keywords, visual_style, voiceover_context
                    )
                    download_order = (
                        [sv.video_result for sv in scored] if scored
                        else filtered[:3]
                    )
                else:
                    download_order = results[:3]

                # Try downloads in evaluated order
                for video_result in download_order:
                    try:
                        download_url = video_result.download_url
                        if not download_url:
                            continue

                        async with httpx.AsyncClient(timeout=60.0) as client:
                            resp = await client.get(download_url)
                            if resp.status_code == 200 and len(resp.content) > 1024:
                                output_path.write_bytes(resp.content)
                                logger.info(
                                    f"[B-roll] Downloaded from {source_name}: '{search_query}' -> "
                                    f"{output_path.name} ({len(resp.content) / 1024:.0f} KB)"
                                )
                                # Trim non-YouTube clips that are too long
                                trimmed = await self._trim_if_too_long(
                                    output_path, keywords, voiceover_context
                                )
                                return trimmed or output_path

                    except Exception as dl_err:
                        logger.warning(
                            f"[B-roll] Download failed ({source_name}, {video_result.video_id}): {dl_err}"
                        )
                        continue

            except Exception as e:
                logger.warning(
                    f"[B-roll] Search failed ({source_name}): {e}"
                )
                continue

        logger.warning(f"[B-roll] No B-roll found from any source for: {search_query}")
        return None

    async def _youtube_multi_query_search(
        self,
        source: "YouTubeVideoSource",
        keywords: list[str],
        visual_style: str,
        output_path: Path,
        search_query: str,
        voiceover_context: str = "",
    ) -> Path | None:
        """Try multiple YouTube query variations with AI evaluation before downloading.

        Attempts three progressively simpler queries:
          1. Enhanced: keywords + visual_style + "stock footage b-roll"
          2. Simpler: keywords + "cinematic footage"
          3. Bare: keywords only

        Each attempt: search -> pre-filter -> AI evaluate -> download best scoring.

        Args:
            source: The YouTubeVideoSource instance
            keywords: Visual keywords list
            visual_style: Style hint for first query
            output_path: Where to save the clip
            search_query: Pre-joined keywords string for logging
            voiceover_context: Voiceover text for AI evaluation context

        Returns:
            Path to downloaded clip, or None if all attempts failed
        """
        base_keywords = " ".join(keywords[:3])
        query_descriptions = [
            f"{base_keywords} {visual_style} stock footage b-roll no watermark".strip(),
            f"{base_keywords} cinematic footage free",
            f"{base_keywords} no watermark",
        ]

        for attempt, desc in enumerate(query_descriptions, 1):
            logger.info(f"[B-roll] YouTube attempt {attempt}/3: '{desc}'")

            try:
                if attempt == 1:
                    results = await source.search_broll_async(
                        keywords, visual_style=visual_style, max_results=10
                    )
                elif attempt == 2:
                    results = await source.search_broll_async(
                        keywords, visual_style="cinematic footage", max_results=10
                    )
                else:
                    results = await source.search_videos_async(base_keywords)

                if not results:
                    logger.debug(
                        f"[B-roll] YouTube attempt {attempt}/3 returned no results"
                    )
                    continue

                # Step 1: Metadata pre-filtering
                filtered = self._prefilter_video_results(results)
                if not filtered:
                    logger.debug(
                        f"[B-roll] All results filtered out (attempt {attempt}/3)"
                    )
                    continue

                # Step 2: AI evaluation - score candidates
                scored = await self._evaluate_broll_candidates(
                    filtered, keywords, visual_style, voiceover_context
                )

                # Step 3: Try downloads in score-descending order
                download_candidates = (
                    [sv.video_result for sv in scored] if scored
                    else filtered[:3]  # Fallback if evaluation fails
                )

                for video_result in download_candidates:
                    try:
                        downloaded = await self._download_youtube_broll(
                            video_result.url, output_path,
                            visual_keywords=keywords,
                            voiceover_context=voiceover_context,
                        )
                        if downloaded:
                            logger.info(
                                f"[B-roll] Downloaded from YouTube (attempt {attempt}/3): "
                                f"'{desc}' -> {output_path.name}"
                            )
                            return output_path
                    except Exception as dl_err:
                        logger.warning(
                            f"[B-roll] YouTube download failed "
                            f"(attempt {attempt}/3, {video_result.video_id}): {dl_err}"
                        )
                        continue

            except Exception as e:
                logger.warning(
                    f"[B-roll] YouTube search failed (attempt {attempt}/3): {e}"
                )
                continue

        logger.debug(f"[B-roll] All 3 YouTube attempts exhausted for: {search_query}")
        return None

    def _prefilter_video_results(
        self,
        results: list[VideoResult],
    ) -> list[VideoResult]:
        """Pre-filter video search results using metadata heuristics.

        Removes videos that are unlikely to be good B-roll based on:
        - Blocked title keywords (compilations, reactions, vlogs, etc.)
        - Very low view count (< MIN_VIEW_COUNT, if available)
        - Prefers Creative Commons licensed content (sorts CC first)

        Args:
            results: Raw search results to filter

        Returns:
            Filtered and re-ordered list of VideoResult objects
        """
        filtered = []
        for video in results:
            title_lower = video.title.lower()

            # Check blocked title keywords
            if any(kw in title_lower for kw in BLOCKED_TITLE_KEYWORDS):
                logger.debug(
                    f"[Pre-filter] Blocked keyword in title: '{video.title[:60]}'"
                )
                continue

            # Filter out very low view count videos (if view_count is available)
            if video.view_count is not None and video.view_count < MIN_VIEW_COUNT:
                logger.debug(
                    f"[Pre-filter] Low views ({video.view_count}): '{video.title[:60]}'"
                )
                continue

            filtered.append(video)

        # Sort: Creative Commons first, then by view_count descending
        def sort_key(v: VideoResult) -> tuple[int, int]:
            is_cc = 1 if v.license and "creative commons" in v.license.lower() else 0
            views = v.view_count or 0
            return (-is_cc, -views)

        filtered.sort(key=sort_key)

        if len(filtered) < len(results):
            logger.info(
                f"[Pre-filter] {len(results)} -> {len(filtered)} results "
                f"(removed {len(results) - len(filtered)})"
            )

        return filtered

    async def _evaluate_broll_candidates(
        self,
        results: list[VideoResult],
        keywords: list[str],
        visual_style: str = "",
        voiceover_context: str = "",
    ) -> list[ScoredVideo]:
        """Score video search results using Gemini Flash before downloading.

        Sends title/description metadata to AI for relevance scoring.
        Only videos scoring >= MIN_BROLL_SCORE are returned.

        Args:
            results: Pre-filtered video search results
            keywords: Visual keywords for the scene
            visual_style: Requested visual style
            voiceover_context: Voiceover text for context

        Returns:
            List of ScoredVideo objects sorted by score descending
        """
        if not results:
            return []

        search_query = " ".join(keywords[:3])

        # Format results for the prompt
        results_text = "\n".join(
            f"ID: {v.video_id}\n"
            f"Title: {v.title}\n"
            f"Description: {(v.description or 'N/A')[:200]}\n"
            f"Duration: {v.duration}s\n"
            f"Views: {v.view_count or 'N/A'}\n"
            "---"
            for v in results
        )

        prompt = VIDEO_AGENT_BROLL_EVALUATOR.format(
            search_query=search_query,
            visual_keywords=", ".join(keywords),
            visual_style=visual_style or "cinematic",
            voiceover_context=voiceover_context[:300] if voiceover_context else "N/A",
            results_text=results_text,
        )

        try:
            # Use Gemini Flash for fast, cheap evaluation
            from google.genai import types

            response = await asyncio.to_thread(
                self.ai.client.models.generate_content,
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )

            if not response.text:
                logger.warning("[B-roll eval] Empty AI response, skipping evaluation")
                return []

            text = strip_markdown_code_blocks(response.text)
            scored_data = json.loads(text)

            if not isinstance(scored_data, list):
                logger.warning("[B-roll eval] AI response is not a list")
                return []

            # Build scored video objects
            video_lookup = {v.video_id: v for v in results}
            scored_videos = []
            for item in scored_data:
                if not isinstance(item, dict):
                    continue
                vid_id = item.get("video_id", "")
                score = item.get("score", 0)
                if vid_id in video_lookup and isinstance(score, (int, float)) and score >= MIN_BROLL_SCORE:
                    scored_videos.append(
                        ScoredVideo(
                            video_id=vid_id,
                            score=int(score),
                            video_result=video_lookup[vid_id],
                        )
                    )

            scored_videos.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                f"[B-roll eval] '{search_query}': "
                f"{len(scored_videos)}/{len(results)} scored >= {MIN_BROLL_SCORE}"
            )
            return scored_videos

        except json.JSONDecodeError as e:
            logger.warning(f"[B-roll eval] JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"[B-roll eval] Evaluation failed: {e}")
            return []

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

    async def _download_youtube_broll(
        self,
        url: str,
        output_path: Path,
        visual_keywords: list[str] | None = None,
        voiceover_context: str = "",
    ) -> bool:
        """Download YouTube B-roll using two-pass: 360p preview -> Gemini analysis -> FFmpeg extract.

        Pass 1: Download 360p preview (fast, small file)
        Pass 2: Upload to Gemini, identify best 3-5s segment
        Pass 3: Extract segment with FFmpeg at higher quality from preview

        If Gemini analysis fails or no good segment found, falls back to
        downloading the first MAX_BROLL_CLIP_SECONDS of the video.

        Args:
            url: YouTube video URL
            output_path: Where to save the final extracted clip
            visual_keywords: Keywords describing desired visuals for Gemini context
            voiceover_context: Scene voiceover text for Gemini context

        Returns:
            True if download and extraction succeeded, False otherwise
        """
        import yt_dlp

        max_duration = self.config.get("youtube_broll_max_duration", 120)
        preview_path = output_path.parent / f"_preview_{output_path.stem}.mp4"

        try:
            # -- Pass 1: Download 360p preview --
            logger.info(f"[Two-pass] Pass 1: Downloading 360p preview for {url}")
            preview_ok = await asyncio.to_thread(
                self._ytdlp_download_preview, url, preview_path, max_duration
            )
            if not preview_ok or not preview_path.exists():
                logger.warning("[Two-pass] Preview download failed, skipping")
                return False

            preview_size_kb = preview_path.stat().st_size / 1024
            logger.info(f"[Two-pass] Preview downloaded: {preview_size_kb:.0f} KB")

            # -- Pass 2: Gemini analysis to find best segment --
            keyword_str = ", ".join(visual_keywords[:5]) if visual_keywords else ""
            segment = await self._analyze_video_for_best_segment(
                preview_path, keyword_str, voiceover_context
            )

            if segment:
                start_time, end_time, score = segment
                logger.info(
                    f"[Two-pass] Gemini found segment: {start_time:.1f}s-{end_time:.1f}s "
                    f"(score: {score}/10)"
                )

                # -- Pass 3: Extract clip with FFmpeg --
                extracted = await asyncio.to_thread(
                    self._extract_clip_ffmpeg,
                    preview_path, output_path, start_time, end_time,
                )
                if extracted:
                    logger.info(
                        f"[Two-pass] Extracted clip: {output_path.name} "
                        f"({end_time - start_time:.1f}s)"
                    )
                    return True

                logger.warning("[Two-pass] FFmpeg extraction failed, using fallback trim")

            else:
                logger.info("[Two-pass] No good segment found, using first few seconds")

            # -- Fallback: trim first MAX_BROLL_CLIP_SECONDS from preview --
            duration = self._get_video_duration_ffprobe(preview_path)
            trim_end = min(float(MAX_BROLL_CLIP_SECONDS), duration)
            extracted = await asyncio.to_thread(
                self._extract_clip_ffmpeg,
                preview_path, output_path, 0.0, trim_end,
            )
            return extracted

        except Exception as e:
            logger.warning(f"[Two-pass] Failed for {url}: {e}")
            return False

        finally:
            # Always clean up preview file
            if preview_path.exists():
                preview_path.unlink(missing_ok=True)
                logger.debug(f"[Two-pass] Cleaned up preview: {preview_path.name}")

    @staticmethod
    def _ytdlp_download_preview(
        url: str, preview_path: Path, max_duration: int = 120
    ) -> bool:
        """Download a 360p preview video with yt-dlp (synchronous, for thread pool).

        Args:
            url: YouTube video URL
            preview_path: Where to save the preview
            max_duration: Skip videos longer than this (seconds)

        Returns:
            True if download succeeded
        """
        import yt_dlp

        ydl_opts = {
            "format": "worst[height<=360]/worst",
            "outtmpl": str(preview_path.with_suffix(".%(ext)s")),
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            "match_filter": lambda info, *_: (
                f"Duration {info.get('duration', 0)}s exceeds max {max_duration}s"
                if info.get("duration", 0) > max_duration
                else None
            ),
            "quiet": True,
            "no_progress": True,
            "no_warnings": True,
            "retries": 3,
            "ignoreerrors": True,
            "socket_timeout": 30,
            "postprocessor_args": {
                "FFmpeg": ["-v", "quiet", "-nostats", "-loglevel", "error"],
                "Merger+ffmpeg": ["-v", "quiet", "-nostats", "-loglevel", "error"],
                "VideoConvertor+ffmpeg": ["-v", "quiet", "-nostats", "-loglevel", "error"],
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # yt-dlp may produce a file with different extension, find and rename
            stem = preview_path.stem
            parent = preview_path.parent
            candidates = list(parent.glob(f"{stem}.*"))
            if not candidates:
                return False

            actual_file = candidates[0]
            if actual_file != preview_path:
                actual_file.rename(preview_path)

            return preview_path.exists() and preview_path.stat().st_size > 1024

        except Exception as e:
            logger.warning(f"[Two-pass] yt-dlp preview error: {e}")
            return False

    async def _analyze_video_for_best_segment(
        self,
        video_path: Path,
        visual_keywords: str,
        voiceover_context: str = "",
    ) -> tuple[float, float, int] | None:
        """Upload video to Gemini File API and find the best 3-5s segment.

        Args:
            video_path: Path to the preview video file
            visual_keywords: Comma-separated keywords for what to look for
            voiceover_context: The scene's voiceover text for context

        Returns:
            Tuple of (start_time, end_time, score) or None if no good segment found
        """
        try:
            import google.genai as genai

            api_key = self.config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning("[Gemini clip] No GEMINI_API_KEY configured")
                return None

            client = genai.Client(api_key=api_key)

            # Get video duration for the prompt
            duration = self._get_video_duration_ffprobe(video_path)

            # Upload video to Gemini File API
            logger.info(f"[Gemini clip] Uploading video ({video_path.stat().st_size / 1024:.0f} KB)...")
            uploaded_file = await asyncio.to_thread(
                client.files.upload, file=str(video_path)
            )

            # Wait for file processing
            file_ready = await asyncio.to_thread(
                self._wait_for_gemini_file, client, uploaded_file.name
            )
            if not file_ready:
                logger.warning("[Gemini clip] File processing timed out")
                return None

            # Build analysis prompt
            context_section = ""
            if voiceover_context:
                context_section = f'\nVOICEOVER CONTEXT: "{voiceover_context[:300]}"\n'

            prompt = f"""Watch this video and identify the single best continuous segment of {MIN_BROLL_CLIP_SECONDS}-{MAX_BROLL_CLIP_SECONDS + 1} seconds that best matches these visuals: "{visual_keywords}".
{context_section}
VIDEO DURATION: {duration:.1f} seconds

REQUIREMENTS:
- The segment MUST have clean visuals: no watermarks, no text overlays, no logos, no talking heads
- Prefer cinematic shots: establishing shots, smooth camera movement, good lighting
- The segment should be visually compelling and work as background B-roll footage
- Duration must be between {MIN_BROLL_CLIP_SECONDS} and {MAX_BROLL_CLIP_SECONDS + 1} seconds

Return ONLY a single JSON object (no markdown, no extra text):
{{"start_time": 12.5, "end_time": 16.0, "score": 8, "rationale": "Clean aerial shot of city skyline"}}

SCORING:
- 10: Perfect match, stunning visuals, no distractions
- 8: Strong match, good quality, minor imperfections
- 7: Decent match, usable as B-roll
- 6 or below: Not good enough (DO NOT RETURN segments below 7)

If no segment scores 7 or higher, return: {{"score": 0}}"""

            # Call Gemini for analysis
            from google.genai import types

            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.0-flash",
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(temperature=0.2),
            )

            # Clean up uploaded file
            try:
                await asyncio.to_thread(client.files.delete, name=uploaded_file.name)
            except Exception:
                pass

            if not response.text:
                logger.warning("[Gemini clip] Empty response")
                return None

            # Parse JSON response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            score = int(result.get("score", 0))
            if score < MIN_GEMINI_SEGMENT_SCORE:
                logger.info(
                    f"[Gemini clip] Best segment scored {score}/10 "
                    f"(below threshold {MIN_GEMINI_SEGMENT_SCORE})"
                )
                return None

            start_time = float(result.get("start_time", 0))
            end_time = float(result.get("end_time", 0))

            # Validate timestamps
            if end_time <= start_time or start_time < 0:
                logger.warning(f"[Gemini clip] Invalid timestamps: {start_time}-{end_time}")
                return None

            seg_duration = end_time - start_time
            if seg_duration < MIN_BROLL_CLIP_SECONDS - 0.5:
                logger.warning(f"[Gemini clip] Segment too short: {seg_duration:.1f}s")
                return None
            if seg_duration > MAX_BROLL_CLIP_SECONDS + 2:
                # Trim to target duration
                end_time = start_time + TARGET_BROLL_CLIP_SECONDS

            # Clamp to video duration
            if end_time > duration:
                end_time = duration

            rationale = result.get("rationale", "")
            logger.info(f"[Gemini clip] Selected: {start_time:.1f}s-{end_time:.1f}s, score={score}, {rationale}")

            return (start_time, end_time, score)

        except json.JSONDecodeError as e:
            logger.warning(f"[Gemini clip] JSON parse error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[Gemini clip] Analysis failed: {e}")
            return None

    @staticmethod
    def _wait_for_gemini_file(client, file_name: str) -> bool:
        """Wait for Gemini File API to finish processing an uploaded file.

        Args:
            client: google.genai Client
            file_name: Name of the uploaded file

        Returns:
            True if file is ready, False if timed out or failed
        """
        start = time.time()
        while time.time() - start < GEMINI_FILE_POLL_TIMEOUT:
            file_info = client.files.get(name=file_name)
            if file_info.state.name == "ACTIVE":
                return True
            if file_info.state.name == "FAILED":
                logger.error(f"[Gemini clip] File processing failed: {file_name}")
                return False
            time.sleep(GEMINI_FILE_POLL_INTERVAL)

        logger.error(f"[Gemini clip] File processing timed out after {GEMINI_FILE_POLL_TIMEOUT}s")
        return False

    @staticmethod
    def _extract_clip_ffmpeg(
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
    ) -> bool:
        """Extract a video segment using FFmpeg with re-encoding for precise cuts.

        Args:
            input_path: Source video file
            output_path: Where to save the extracted clip
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            True if extraction succeeded and output is valid
        """
        duration = end_time - start_time
        if duration <= 0:
            return False

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-force_key_frames", f"expr:gte(t,{start_time})",
            "-movflags", "+faststart",
            "-v", "quiet",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"[FFmpeg] Extraction failed: {result.stderr[:200]}")
                return False

            # Validate output
            if output_path.exists() and output_path.stat().st_size > 10 * 1024:
                return True

            logger.warning(
                f"[FFmpeg] Output too small or missing: "
                f"{output_path.stat().st_size if output_path.exists() else 0} bytes"
            )
            output_path.unlink(missing_ok=True)
            return False

        except subprocess.TimeoutExpired:
            logger.warning("[FFmpeg] Extraction timed out (60s)")
            output_path.unlink(missing_ok=True)
            return False
        except Exception as e:
            logger.warning(f"[FFmpeg] Extraction error: {e}")
            output_path.unlink(missing_ok=True)
            return False

    @staticmethod
    def _get_video_duration_ffprobe(video_path: Path) -> float:
        """Get video duration in seconds using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds, defaults to 30.0 on error
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"[ffprobe] Could not get duration: {e}")
            return 30.0

    @staticmethod
    def _ytdlp_download(url: str, ydl_opts: dict) -> bool:
        """Run yt-dlp download synchronously (called from thread pool).

        Args:
            url: Video URL to download
            ydl_opts: yt-dlp options dict

        Returns:
            True if download succeeded
        """
        import yt_dlp

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        except Exception as e:
            logger.warning(f"[YouTube B-roll] yt-dlp error: {e}")
            return False

    async def _trim_if_too_long(
        self,
        clip_path: Path,
        keywords: list[str] | None = None,
        voiceover_context: str = "",
    ) -> Path | None:
        """Trim a non-YouTube clip to MAX_BROLL_CLIP_SECONDS if it's too long.

        For clips > MAX_BROLL_CLIP_SECONDS + 1: uses Gemini analysis to find
        the best segment, falling back to trimming from the start.

        Args:
            clip_path: Path to the downloaded clip
            keywords: Visual keywords for Gemini context
            voiceover_context: Voiceover text for Gemini context

        Returns:
            Path to the trimmed clip, or None if no trimming needed/failed
        """
        duration = self._get_video_duration_ffprobe(clip_path)
        max_dur = float(MAX_BROLL_CLIP_SECONDS) + 1.0

        if duration <= max_dur:
            return None  # No trimming needed

        logger.info(f"[Trim] Clip is {duration:.1f}s (max {max_dur:.0f}s), trimming...")

        # Try Gemini analysis first for longer clips
        start_time = 0.0
        end_time = float(TARGET_BROLL_CLIP_SECONDS)

        if duration > 10.0 and keywords:
            keyword_str = ", ".join(keywords[:5])
            segment = await self._analyze_video_for_best_segment(
                clip_path, keyword_str, voiceover_context
            )
            if segment:
                start_time, end_time, _score = segment
                logger.info(f"[Trim] Gemini selected {start_time:.1f}s-{end_time:.1f}s")
            else:
                logger.info("[Trim] Gemini found no good segment, trimming from start")

        # Extract the trimmed clip
        trimmed_path = clip_path.with_stem(clip_path.stem + "_trimmed")
        extracted = await asyncio.to_thread(
            self._extract_clip_ffmpeg, clip_path, trimmed_path, start_time, end_time
        )

        if extracted and trimmed_path.exists():
            # Replace original with trimmed version
            clip_path.unlink(missing_ok=True)
            trimmed_path.rename(clip_path)
            logger.info(f"[Trim] Trimmed to {end_time - start_time:.1f}s: {clip_path.name}")
            return clip_path

        # Trimming failed, keep original
        trimmed_path.unlink(missing_ok=True)
        return None

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
