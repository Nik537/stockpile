"""FFmpeg-based video assembly engine for the video agent pipeline.

Takes a Timeline object (scenes with audio, visuals, subtitles) and renders
a complete video using FFmpeg subprocess calls. No MoviePy dependency.

Supports xfade transitions between scenes (dissolve, zoom_in, swipe, fade)
and SFX audio mixing infrastructure.

All intermediate files are managed in a temp directory and cleaned up after
composition completes.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from video_agent.models import Timeline, TimelineScene, VisualType

logger = logging.getLogger(__name__)

# Resolution and encoding defaults
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30

# Draft mode settings (fast preview)
DRAFT_WIDTH = 960
DRAFT_HEIGHT = 480
DRAFT_PRESET = "ultrafast"
DRAFT_CRF = 28

# Final render settings
FINAL_CRF = 18
FINAL_PRESET = "medium"

# Color grade filter presets (FFmpeg eq + curves filters)
COLOR_GRADES = {
    "dark_cinematic": (
        "eq=contrast=1.2:brightness=-0.05:saturation=0.85,"
        "curves=m='0/0 0.25/0.15 0.5/0.45 0.75/0.8 1/1'"
    ),
    "warm": (
        "eq=contrast=1.05:saturation=1.1,"
        "colorbalance=rs=0.05:gs=0.02:bs=-0.03:rm=0.03:gm=0.01:bm=-0.02"
    ),
    "cool": (
        "eq=contrast=1.05:saturation=0.95,"
        "colorbalance=rs=-0.03:gs=0.0:bs=0.06:rm=-0.02:gm=0.01:bm=0.04"
    ),
    "neutral": "",
}

# Xfade transition mapping: script transition_in -> FFmpeg xfade transition name
XFADE_TRANSITIONS = {
    "dissolve": {"transition": "dissolve", "duration": 0.5},
    "zoom_in": {"transition": "circlecrop", "duration": 0.4},
    "swipe": {"transition": "slideleft", "duration": 0.4},
    "fade": {"transition": "fade", "duration": 0.6},
}

# SFX name -> file path mapping (populated at runtime from sfx/ directory)
# Keys are cue names from SceneScript.sound_effect, values are audio file paths.
SFX_LIBRARY: dict[str, Path] = {}


class VideoComposerError(Exception):
    """Raised when an FFmpeg operation fails during composition."""

    pass


class VideoComposer:
    """Assembles final video from Timeline using FFmpeg.

    FFmpeg-first approach: all rendering via subprocess, no MoviePy.
    This avoids memory issues with long videos and gives precise control.

    Supports xfade transitions between scenes and SFX audio mixing.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        sfx_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sfx_dir = sfx_dir
        self._sfx_library: dict[str, Path] = dict(SFX_LIBRARY)

        # Auto-discover SFX files from sfx_dir if provided
        if self.sfx_dir and self.sfx_dir.is_dir():
            self._discover_sfx(self.sfx_dir)

    def _discover_sfx(self, sfx_dir: Path) -> None:
        """Scan directory for SFX audio files and populate library.

        Maps filename stems to file paths. For example:
        sfx/whoosh.wav -> {"whoosh": Path("sfx/whoosh.wav")}
        sfx/dramatic_whoosh.mp3 -> {"dramatic_whoosh": Path("sfx/dramatic_whoosh.mp3")}
        """
        audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
        for f in sfx_dir.iterdir():
            if f.is_file() and f.suffix.lower() in audio_extensions:
                self._sfx_library[f.stem.lower()] = f
        if self._sfx_library:
            logger.info(
                f"SFX library: {len(self._sfx_library)} effects loaded from {sfx_dir}"
            )

    async def compose(self, timeline: Timeline, draft: bool = False) -> Path:
        """Compose full video from timeline.

        Steps:
        1. Render each scene individually (normalize + visual effects)
        2. Concatenate/transition all scene clips (xfade or concat demuxer)
        3. Add master audio track
        4. Mix in background music with ducking (if provided)
        5. Mix in SFX audio (if any scenes have sound effects)
        6. Burn subtitles (if provided)
        7. Apply color grade (if specified)
        8. Final encode

        Args:
            timeline: Complete timeline with scenes, audio, music, subs
            draft: If True, render at 480p ultrafast for preview

        Returns:
            Path to rendered video file
        """
        if not timeline.scenes:
            raise VideoComposerError("Timeline has no scenes to compose")

        width = DRAFT_WIDTH if draft else DEFAULT_WIDTH
        height = DRAFT_HEIGHT if draft else DEFAULT_HEIGHT

        tmp_dir = Path(tempfile.mkdtemp(prefix="stockpile_compose_"))
        logger.info(
            f"Composing video: {len(timeline.scenes)} scenes, "
            f"{width}x{height}, draft={draft}"
        )

        try:
            # Step 1: Render each scene
            scene_clips: list[Path] = []
            for i, scene in enumerate(timeline.scenes):
                logger.info(f"Rendering scene {i + 1}/{len(timeline.scenes)}")
                rendered = await self._render_scene(
                    scene, width, height, tmp_dir, i, draft
                )
                scene_clips.append(rendered)

            # Step 2: Concatenate/transition all scene clips
            raw_video = tmp_dir / "concatenated.mp4"
            transitions = [s.transition for s in timeline.scenes]
            has_xfade = any(t != "cut" for t in transitions)

            if has_xfade and len(scene_clips) > 1:
                await asyncio.to_thread(
                    self._concatenate_with_xfade,
                    scene_clips,
                    transitions,
                    raw_video,
                )
            else:
                await asyncio.to_thread(
                    self._concatenate_scenes, scene_clips, raw_video
                )

            # Step 3: Prepare audio track
            current_audio = timeline.master_audio

            # Step 4: Mix background music if provided
            if timeline.music_path and timeline.music_path.exists():
                mixed_audio = tmp_dir / "mixed_audio.wav"
                await asyncio.to_thread(
                    self._mix_music,
                    timeline.master_audio,
                    timeline.music_path,
                    mixed_audio,
                )
                current_audio = mixed_audio

            # Step 5: Mix SFX if any scenes have sound effects
            scenes_with_sfx = [
                s for s in timeline.scenes
                if s.sound_effect and s.sound_effect != "none"
            ]
            if scenes_with_sfx:
                sfx_audio = tmp_dir / "sfx_mixed_audio.wav"
                sfx_applied = await asyncio.to_thread(
                    self._mix_sfx,
                    current_audio,
                    timeline,
                    sfx_audio,
                )
                if sfx_applied:
                    current_audio = sfx_audio

            # Step 6: Add audio to video
            video_with_audio = tmp_dir / "with_audio.mp4"
            await asyncio.to_thread(
                self._add_audio, raw_video, current_audio, video_with_audio
            )
            current_video = video_with_audio

            # Step 7: Burn subtitles (skip in draft mode)
            if (
                not draft
                and timeline.subtitle_path
                and timeline.subtitle_path.exists()
            ):
                video_with_subs = tmp_dir / "with_subs.mp4"
                await asyncio.to_thread(
                    self._burn_subtitles,
                    current_video,
                    timeline.subtitle_path,
                    video_with_subs,
                )
                current_video = video_with_subs

            # Step 8: Apply color grade (skip in draft mode)
            grade = timeline.color_grade or "neutral"
            if not draft and grade != "neutral" and grade in COLOR_GRADES:
                graded_video = tmp_dir / "graded.mp4"
                await asyncio.to_thread(
                    self._apply_color_grade,
                    current_video,
                    graded_video,
                    grade,
                )
                current_video = graded_video

            # Move final output to output directory
            suffix = "_draft" if draft else ""
            final_path = self.output_dir / f"composed{suffix}.mp4"
            shutil.move(str(current_video), str(final_path))

            logger.info(f"Composition complete: {final_path}")
            return final_path

        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temp dir: {tmp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")

    # ------------------------------------------------------------------
    # Scene rendering
    # ------------------------------------------------------------------

    async def _render_scene(
        self,
        scene: TimelineScene,
        width: int,
        height: int,
        tmp_dir: Path,
        index: int,
        draft: bool = False,
    ) -> Path:
        """Render a single scene clip.

        For BROLL_VIDEO: normalize to target resolution, trim to duration
        For GENERATED_IMAGE: apply Ken Burns effect (zoompan)
        For TEXT_GRAPHIC: create text overlay on black background
        """
        duration = scene.audio_end - scene.audio_start
        output_path = tmp_dir / f"scene_{index:03d}.mp4"

        if scene.visual_type == VisualType.BROLL_VIDEO:
            await asyncio.to_thread(
                self._normalize_video,
                scene.visual_path,
                output_path,
                width,
                height,
                duration,
            )
        elif scene.visual_type == VisualType.GENERATED_IMAGE:
            direction = "zoom_in" if index % 2 == 0 else "zoom_out"
            await asyncio.to_thread(
                self._apply_ken_burns,
                scene.visual_path,
                output_path,
                duration,
                width,
                height,
                direction,
            )
        elif scene.visual_type == VisualType.TEXT_GRAPHIC:
            # visual_path stores a text file with the graphic content,
            # or we treat the path stem as the text itself
            text = self._read_text_graphic(scene.visual_path)
            await asyncio.to_thread(
                self._create_text_graphic,
                text,
                output_path,
                duration,
                width,
                height,
            )
        else:
            raise VideoComposerError(
                f"Unknown visual type: {scene.visual_type}"
            )

        if not output_path.exists():
            raise VideoComposerError(
                f"Scene rendering failed: {output_path} not created"
            )

        return output_path

    def _normalize_video(
        self,
        input_path: Path,
        output_path: Path,
        width: int,
        height: int,
        duration: Optional[float] = None,
    ) -> None:
        """Normalize a video clip to target resolution.

        Scales to fit within target dimensions preserving aspect ratio,
        pads with black bars if needed, and sets consistent pixel format
        and frame rate.
        """
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={DEFAULT_FPS}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", DRAFT_PRESET,
            "-pix_fmt", "yuv420p",
            "-an",
        ]

        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.append(str(output_path))

        self._run_ffmpeg(cmd, f"normalize {input_path.name}")

    def _apply_ken_burns(
        self,
        image_path: Path,
        output_path: Path,
        duration: float,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        direction: str = "zoom_in",
    ) -> None:
        """Apply Ken Burns zoom/pan effect to a static image.

        Creates a video from a still image with a slow zoom or pan effect
        to add visual interest.
        """
        total_frames = int(duration * DEFAULT_FPS)

        if direction == "zoom_out":
            zoom_expr = "if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))"
        elif direction == "pan_left":
            zoom_expr = "1.2"
        else:
            # Default: zoom_in
            zoom_expr = "min(zoom+0.0015,1.5)"

        # For pan_left, move x; otherwise center the zoom
        if direction == "pan_left":
            x_expr = "if(gte(on,1),x+1,0)"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"

        zoompan_filter = (
            f"zoompan=z='{zoom_expr}':"
            f"d={total_frames}:"
            f"x='{x_expr}':y='{y_expr}':"
            f"s={width}x{height}:fps={DEFAULT_FPS},"
            f"fps={DEFAULT_FPS}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", zoompan_filter,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", DRAFT_PRESET,
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, f"ken_burns {image_path.name} ({direction})")

    def _create_text_graphic(
        self,
        text: str,
        output_path: Path,
        duration: float,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        """Create a styled text graphic clip with gradient background.

        Renders centered white text on a dark gradient background.
        Applies word wrapping (max 4 words per line) and uses large
        bold font for readability. Text is vertically and horizontally centered.
        """
        # Word wrap: max 4 words per line for readability
        words = text.split()
        lines = []
        for i in range(0, len(words), 4):
            lines.append(" ".join(words[i : i + 4]))
        wrapped_text = "\n".join(lines) if lines else text

        # Escape text for FFmpeg drawtext filter
        escaped_text = (
            wrapped_text.replace("\\", "\\\\")
            .replace("'", "\u2019")
            .replace(":", "\\:")
            .replace("%", "%%")
        )

        # Larger font for fewer words, scale down for longer text
        total_words = len(words)
        if total_words <= 3:
            fontsize = 96
        elif total_words <= 6:
            fontsize = 80
        elif total_words <= 10:
            fontsize = 64
        else:
            fontsize = 52

        # Dark gradient background: dark charcoal (#1a1a2e) to near-black (#0f0f23)
        # Using a color source + gradient overlay for a more polished look
        bg_filter = (
            f"color=c=#0f0f23:s={width}x{height}:d={duration}:r={DEFAULT_FPS}[bg];"
            f"color=c=#1a1a2e@0.6:s={width}x{height // 2}:d={duration}:r={DEFAULT_FPS},"
            f"format=yuva420p[grad];"
            f"[bg][grad]overlay=0:0:format=auto[canvas];"
            f"[canvas]drawtext=text='{escaped_text}':"
            f"fontsize={fontsize}:fontcolor=white:"
            f"x=(w-tw)/2:y=(h-th)/2:"
            f"line_spacing=20"
        )

        cmd = [
            "ffmpeg", "-y",
            "-filter_complex", bg_filter,
            "-c:v", "libx264",
            "-preset", DRAFT_PRESET,
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(output_path),
        ]

        self._run_ffmpeg(cmd, f"text_graphic ({len(text)} chars)")

    # ------------------------------------------------------------------
    # Scene concatenation and transitions
    # ------------------------------------------------------------------

    @staticmethod
    def _select_transition(transition_in: str) -> tuple[str, float]:
        """Select FFmpeg xfade transition type and duration.

        Maps scene transition_in field to FFmpeg xfade parameters.

        Args:
            transition_in: Transition name from SceneScript ("cut", "dissolve", etc.)

        Returns:
            Tuple of (ffmpeg_transition_name, duration_seconds).
            Returns ("cut", 0.0) for unrecognized or "cut" transitions.
        """
        if transition_in in XFADE_TRANSITIONS:
            spec = XFADE_TRANSITIONS[transition_in]
            return spec["transition"], spec["duration"]
        return "cut", 0.0

    def _get_video_duration(self, video_path: Path) -> float:
        """Get the precise duration of a video file using ffprobe.

        Returns duration in seconds. Falls back to 0.0 on error.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data["format"]["duration"])
        except Exception as e:
            logger.warning(f"ffprobe failed for {video_path}: {e}")
        return 0.0

    def _concatenate_scenes(
        self, scene_clips: list[Path], output_path: Path
    ) -> None:
        """Concatenate scene clips using FFmpeg concat demuxer (fast, cut-only).

        All inputs must have matching codecs, resolution, and frame rate,
        which is enforced by _render_scene.
        """
        if not scene_clips:
            raise VideoComposerError("No scene clips to concatenate")

        if len(scene_clips) == 1:
            shutil.copy2(str(scene_clips[0]), str(output_path))
            return

        # Write concat list file
        concat_file = output_path.parent / "concat.txt"
        lines = [f"file '{clip}'" for clip in scene_clips]
        concat_file.write_text("\n".join(lines))

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, f"concatenate {len(scene_clips)} scenes (cut)")

    def _concatenate_with_xfade(
        self,
        scene_clips: list[Path],
        transitions: list[str],
        output_path: Path,
    ) -> None:
        """Concatenate scene clips using FFmpeg xfade filter_complex.

        Builds a dynamic filter chain for N scenes with transition effects.
        For "cut" transitions within the chain, uses dissolve with 0.0 duration
        to maintain the filter graph structure, which is equivalent to a cut.

        For N scenes, we build N-1 xfade filters chained together:
            [0][1]xfade=transition=T1:duration=D1:offset=O1[v01];
            [v01][2]xfade=transition=T2:duration=D2:offset=O2[v012]; ...

        The offset for transition i = sum of durations of clips 0..i minus
        the sum of all previous transition durations (overlap).

        Args:
            scene_clips: Rendered scene clip paths in order
            transitions: Transition type per scene (transitions[0] is ignored
                         since scene 0 has no preceding scene to transition from)
            output_path: Where to write the concatenated output
        """
        if len(scene_clips) < 2:
            if scene_clips:
                shutil.copy2(str(scene_clips[0]), str(output_path))
            return

        # Get precise durations for each clip
        durations = [self._get_video_duration(clip) for clip in scene_clips]

        # Resolve transitions for each boundary (N-1 boundaries for N clips)
        # transitions[i] is the transition INTO scene i, so boundary i uses
        # transitions[i+1] (the transition into the next scene)
        boundary_transitions: list[tuple[str, float]] = []
        for i in range(1, len(scene_clips)):
            t_name = transitions[i] if i < len(transitions) else "cut"
            ffmpeg_t, t_dur = self._select_transition(t_name)
            # For "cut" in an xfade chain, use fade with minimal overlap
            # (xfade requires duration > 0, so use ~1 frame as near-instant cut)
            if t_name == "cut":
                ffmpeg_t = "fade"
                t_dur = 0.04
            boundary_transitions.append((ffmpeg_t, t_dur))

        # Build filter_complex string
        filter_parts: list[str] = []
        cumulative_duration = durations[0]

        for i, (ffmpeg_t, t_dur) in enumerate(boundary_transitions):
            # Offset = cumulative visible duration so far minus the overlap
            offset = cumulative_duration - t_dur
            # Clamp offset to avoid negative values
            offset = max(0.0, offset)

            if i == 0:
                src_label = "[0][1]"
            else:
                src_label = f"[v{i - 1}][{i + 1}]"

            if i == len(boundary_transitions) - 1:
                # Last xfade: output is the final video stream
                out_label = "[vout]"
            else:
                out_label = f"[v{i}]"

            filter_parts.append(
                f"{src_label}xfade=transition={ffmpeg_t}:"
                f"duration={t_dur:.3f}:offset={offset:.3f}{out_label}"
            )

            # After this xfade, the output stream duration is:
            # offset + durations[i+1] (the transition starts at offset,
            # and the new clip's full duration extends from there)
            cumulative_duration = offset + durations[i + 1]

        filter_complex = ";".join(filter_parts)

        # Build FFmpeg command with all inputs
        cmd = ["ffmpeg", "-y"]
        for clip in scene_clips:
            cmd.extend(["-i", str(clip)])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264",
            "-preset", DRAFT_PRESET,
            "-pix_fmt", "yuv420p",
            str(output_path),
        ])

        transition_summary = ", ".join(
            f"{t[0]}({t[1]:.1f}s)" for t in boundary_transitions
        )
        self._run_ffmpeg(
            cmd,
            f"xfade {len(scene_clips)} scenes [{transition_summary}]",
        )

    # ------------------------------------------------------------------
    # Audio mixing
    # ------------------------------------------------------------------

    def _add_audio(
        self, video_path: Path, audio_path: Path, output_path: Path
    ) -> None:
        """Replace/add audio track to video.

        Maps the video stream from the first input and the audio stream
        from the second input. Uses -shortest to match durations.
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "add audio track")

    def _mix_music(
        self,
        narration_path: Path,
        music_path: Path,
        output_path: Path,
        music_volume: float = 0.12,
    ) -> None:
        """Mix background music under narration with volume ducking.

        Lowers music volume to the specified fraction of narration and
        mixes both tracks. Output duration matches narration length.
        """
        filter_complex = (
            f"[1:a]volume={music_volume}[music];"
            f"[0:a][music]amix=inputs=2:duration=first[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(narration_path),
            "-i", str(music_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "mix background music")

    def _mix_sfx(
        self,
        master_audio_path: Path,
        timeline: Timeline,
        output_path: Path,
        sfx_volume: float = 0.7,
    ) -> bool:
        """Mix sound effects into the audio track at scene timestamps.

        Finds SFX files matching each scene's sound_effect field, then
        overlays them at the correct timeline positions using FFmpeg
        adelay + amix filters.

        Args:
            master_audio_path: Current audio track (narration + music)
            timeline: Timeline with scenes that have sound_effect fields
            output_path: Where to write the mixed audio
            sfx_volume: Volume level for SFX (0.0 - 1.0)

        Returns:
            True if SFX were applied, False if skipped (no matching files)
        """
        # Collect SFX cues with their timestamps
        sfx_cues: list[tuple[float, Path]] = []

        for scene in timeline.scenes:
            if not scene.sound_effect or scene.sound_effect == "none":
                continue

            sfx_name = scene.sound_effect.lower().strip()
            sfx_path = self._sfx_library.get(sfx_name)

            if sfx_path is None:
                # Try partial matching (e.g., "dramatic_whoosh" matches "whoosh")
                for lib_name, lib_path in self._sfx_library.items():
                    if lib_name in sfx_name or sfx_name in lib_name:
                        sfx_path = lib_path
                        break

            if sfx_path is None:
                logger.debug(
                    f"SFX '{sfx_name}' not found in library "
                    f"(available: {list(self._sfx_library.keys())})"
                )
                continue

            if not sfx_path.exists():
                logger.warning(f"SFX file missing: {sfx_path}")
                continue

            sfx_cues.append((scene.audio_start, sfx_path))

        if not sfx_cues:
            logger.debug("No matching SFX files found, skipping SFX mix")
            return False

        logger.info(f"Mixing {len(sfx_cues)} SFX cues into audio")

        # Build FFmpeg filter_complex for SFX overlay
        # Input 0 = master audio, inputs 1..N = SFX files
        # Each SFX is delayed to its scene start time, then all are mixed
        cmd = ["ffmpeg", "-y", "-i", str(master_audio_path)]
        filter_parts: list[str] = []

        for i, (timestamp, sfx_path) in enumerate(sfx_cues):
            cmd.extend(["-i", str(sfx_path)])
            input_idx = i + 1
            delay_ms = int(timestamp * 1000)
            # Volume adjust and delay each SFX to its position
            filter_parts.append(
                f"[{input_idx}:a]volume={sfx_volume},"
                f"adelay={delay_ms}|{delay_ms}[sfx{i}]"
            )

        # Mix all SFX streams with the master audio
        # Build the amix input list
        mix_inputs = "[0:a]" + "".join(f"[sfx{i}]" for i in range(len(sfx_cues)))
        total_inputs = 1 + len(sfx_cues)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={total_inputs}:"
            f"duration=first:dropout_transition=0[out]"
        )

        filter_complex = ";".join(filter_parts)

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[out]",
            str(output_path),
        ])

        self._run_ffmpeg(cmd, f"mix {len(sfx_cues)} SFX cues")
        return True

    # ------------------------------------------------------------------
    # Color grading and subtitles
    # ------------------------------------------------------------------

    def _apply_color_grade(
        self,
        video_path: Path,
        output_path: Path,
        lut_name: str = "dark_cinematic",
    ) -> None:
        """Apply color grade via FFmpeg eq/curves/colorbalance filters.

        Uses built-in FFmpeg filters rather than .cube LUT files for
        portability.
        """
        vf = COLOR_GRADES.get(lut_name, "")
        if not vf:
            shutil.copy2(str(video_path), str(output_path))
            return

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", FINAL_PRESET,
            "-crf", str(FINAL_CRF),
            "-c:a", "copy",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, f"color grade: {lut_name}")

    def _burn_subtitles(
        self, video_path: Path, sub_path: Path, output_path: Path
    ) -> None:
        """Burn subtitle file into video.

        Supports ASS and SRT formats via the FFmpeg ass/subtitles filter.
        """
        # Use 'ass' filter for .ass files, 'subtitles' for .srt
        if sub_path.suffix.lower() == ".ass":
            # Escape colons and backslashes in path for FFmpeg filter
            escaped_path = str(sub_path).replace("\\", "/").replace(":", "\\:")
            vf = f"ass='{escaped_path}'"
        else:
            escaped_path = str(sub_path).replace("\\", "/").replace(":", "\\:")
            vf = f"subtitles='{escaped_path}'"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", FINAL_PRESET,
            "-crf", str(FINAL_CRF),
            "-c:a", "copy",
            str(output_path),
        ]

        self._run_ffmpeg(cmd, "burn subtitles")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, cmd: list[str], description: str = "") -> None:
        """Run FFmpeg command with error handling.

        Args:
            cmd: FFmpeg command as list of arguments
            description: Human-readable description for logging

        Raises:
            VideoComposerError: If FFmpeg returns a non-zero exit code
        """
        logger.info(f"FFmpeg: {description}")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr[-1000:]}")
            raise VideoComposerError(
                f"FFmpeg failed ({description}): {result.stderr[:500]}"
            )

    @staticmethod
    def _read_text_graphic(path: Path) -> str:
        """Read text content for a text graphic scene.

        If the path points to an existing .txt file, reads its content.
        Otherwise treats the file stem as the display text.
        """
        if path.exists() and path.suffix == ".txt":
            return path.read_text().strip()
        return path.stem.replace("_", " ")
