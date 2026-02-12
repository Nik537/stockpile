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
from utils.config import load_config

logger = logging.getLogger(__name__)

# Maximum B-roll clip duration in seconds
MAX_BROLL_CLIP_SECONDS = 4


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
    ) -> Path | None:
        """Search video sources and download best B-roll clip.

        For YouTube, tries multiple query variations before falling through:
          1. Enhanced: keywords + style + "stock footage b-roll"
          2. Simpler: keywords + "cinematic footage"
          3. Bare: keywords only

        Only falls through to Pexels/Pixabay if ALL YouTube attempts fail.
        Non-YouTube sources use a single standard search.

        Args:
            keywords: Visual keywords to search for
            output_path: Where to save the downloaded clip
            visual_style: Optional style hint for search enhancement

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

                # Try top results from this source
                for video_result in results[:3]:
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
                                return output_path

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
    ) -> Path | None:
        """Try multiple YouTube query variations before giving up.

        Attempts three progressively simpler queries:
          1. Enhanced: keywords + visual_style + "stock footage b-roll"
          2. Simpler: keywords + "cinematic footage"
          3. Bare: keywords only

        Args:
            source: The YouTubeVideoSource instance
            keywords: Visual keywords list
            visual_style: Style hint for first query
            output_path: Where to save the clip
            search_query: Pre-joined keywords string for logging

        Returns:
            Path to downloaded clip, or None if all attempts failed
        """
        base_keywords = " ".join(keywords[:3])
        query_descriptions = [
            f"{base_keywords} {visual_style} stock footage b-roll".strip(),
            f"{base_keywords} cinematic footage",
            base_keywords,
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

                for video_result in results[:3]:
                    try:
                        downloaded = await self._download_youtube_broll(
                            video_result.url, output_path
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

    async def _download_youtube_broll(
        self,
        url: str,
        output_path: Path,
    ) -> bool:
        """Download a YouTube video as B-roll clip using yt-dlp.

        Downloads at 720p max for B-roll (background clips don't need full quality).
        Enforces max duration from config to avoid downloading hour-long videos.

        Args:
            url: YouTube video URL
            output_path: Where to save the downloaded clip

        Returns:
            True if download succeeded, False otherwise
        """
        import yt_dlp

        max_duration = self.config.get("youtube_broll_max_duration", 120)

        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "outtmpl": str(output_path.with_suffix(".%(ext)s")),
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
            result = await asyncio.to_thread(self._ytdlp_download, url, ydl_opts)
            if not result:
                return False

            # yt-dlp may produce a file with different name/ext, find it
            stem = output_path.stem
            parent = output_path.parent
            candidates = list(parent.glob(f"{stem}.*"))
            if not candidates:
                return False

            # Rename to expected output path if needed
            actual_file = candidates[0]
            if actual_file != output_path:
                actual_file.rename(output_path)

            # Validate the file
            if output_path.exists() and output_path.stat().st_size > 1024:
                return True

            output_path.unlink(missing_ok=True)
            return False

        except Exception as e:
            logger.warning(f"[YouTube B-roll] yt-dlp download failed for {url}: {e}")
            return False

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
        stock_path = await self._search_stock_image(search_query, output_path)
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
    ) -> Path | None:
        """Search Pexels/Pixabay for a stock photo and download it.

        Used as a free fallback when AI image generation fails.
        Downloads the first matching landscape photo.
        """
        import httpx

        for source in self.image_sources:
            if not source.is_configured():
                continue

            source_name = source.get_source_name()
            try:
                results = await source.search_images(query, per_page=3)
                if not results:
                    continue

                for img_result in results:
                    if not img_result.download_url:
                        continue
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
