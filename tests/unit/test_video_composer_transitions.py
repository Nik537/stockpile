"""Unit tests for VideoComposer xfade transitions and SFX mixing.

Tests FFmpeg command generation for:
- Xfade transitions (dissolve, zoom_in, swipe, fade)
- "cut" transitions using fast concat demuxer
- Mixed transitions (some cut, some xfade) via xfade chain
- Offset calculation for xfade chains
- SFX mixing with adelay + amix filters
- Graceful handling when SFX files are missing
- Scene rendering dispatch (BROLL_VIDEO, GENERATED_IMAGE, TEXT_GRAPHIC)

All tests mock subprocess.run so no actual FFmpeg execution occurs.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from video_agent.models import Timeline, TimelineScene, VisualType, WordTiming
from video_agent.video_composer import (
    COLOR_GRADES,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DRAFT_HEIGHT,
    DRAFT_WIDTH,
    XFADE_TRANSITIONS,
    VideoComposer,
    VideoComposerError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def composer(temp_dir):
    """Create a VideoComposer with output_dir set to temp dir."""
    return VideoComposer(output_dir=temp_dir)


@pytest.fixture
def composer_with_sfx(temp_dir):
    """Create a VideoComposer with SFX directory populated."""
    sfx_dir = temp_dir / "sfx"
    sfx_dir.mkdir()
    # Create fake SFX files
    (sfx_dir / "whoosh.wav").write_bytes(b"\x00" * 100)
    (sfx_dir / "typing.mp3").write_bytes(b"\x00" * 100)
    (sfx_dir / "dramatic_hit.wav").write_bytes(b"\x00" * 100)
    return VideoComposer(output_dir=temp_dir / "output", sfx_dir=sfx_dir)


@pytest.fixture
def fake_scene_clips(temp_dir):
    """Create fake scene clip files."""
    clips = []
    for i in range(3):
        clip = temp_dir / f"scene_{i:03d}.mp4"
        clip.write_bytes(b"\x00" * 1024)
        clips.append(clip)
    return clips


@pytest.fixture
def sample_timeline(temp_dir):
    """Create a sample Timeline for testing."""
    audio_path = temp_dir / "master_audio.wav"
    audio_path.write_bytes(b"\x00" * 100)

    visual_path = temp_dir / "visual_001.mp4"
    visual_path.write_bytes(b"\x00" * 100)

    scenes = [
        TimelineScene(
            scene_id=0,
            audio_path=temp_dir / "audio_000.wav",
            audio_start=0.0,
            audio_end=5.0,
            visual_path=visual_path,
            visual_type=VisualType.BROLL_VIDEO,
            transition="cut",
        ),
        TimelineScene(
            scene_id=1,
            audio_path=temp_dir / "audio_001.wav",
            audio_start=5.0,
            audio_end=10.0,
            visual_path=visual_path,
            visual_type=VisualType.BROLL_VIDEO,
            transition="dissolve",
        ),
        TimelineScene(
            scene_id=2,
            audio_path=temp_dir / "audio_002.wav",
            audio_start=10.0,
            audio_end=15.0,
            visual_path=visual_path,
            visual_type=VisualType.BROLL_VIDEO,
            transition="fade",
        ),
    ]

    # Create audio files
    for s in scenes:
        s.audio_path.write_bytes(b"\x00" * 100)

    return Timeline(
        scenes=scenes,
        master_audio=audio_path,
        total_duration=15.0,
    )


# ---------------------------------------------------------------------------
# _select_transition tests
# ---------------------------------------------------------------------------


class TestSelectTransition:
    """Tests for _select_transition static method."""

    def test_dissolve_transition(self):
        """dissolve maps to FFmpeg 'dissolve' with 0.5s duration."""
        name, dur = VideoComposer._select_transition("dissolve")
        assert name == "dissolve"
        assert dur == 0.5

    def test_zoom_in_transition(self):
        """zoom_in maps to FFmpeg 'circlecrop' with 0.4s duration."""
        name, dur = VideoComposer._select_transition("zoom_in")
        assert name == "circlecrop"
        assert dur == 0.4

    def test_swipe_transition(self):
        """swipe maps to FFmpeg 'slideleft' with 0.4s duration."""
        name, dur = VideoComposer._select_transition("swipe")
        assert name == "slideleft"
        assert dur == 0.4

    def test_fade_transition(self):
        """fade maps to FFmpeg 'fade' with 0.6s duration."""
        name, dur = VideoComposer._select_transition("fade")
        assert name == "fade"
        assert dur == 0.6

    def test_cut_transition(self):
        """cut returns ('cut', 0.0)."""
        name, dur = VideoComposer._select_transition("cut")
        assert name == "cut"
        assert dur == 0.0

    def test_unrecognized_transition_defaults_to_cut(self):
        """Unrecognized transitions default to cut."""
        name, dur = VideoComposer._select_transition("wipe_diagonal")
        assert name == "cut"
        assert dur == 0.0

    def test_empty_string_defaults_to_cut(self):
        """Empty string transition defaults to cut."""
        name, dur = VideoComposer._select_transition("")
        assert name == "cut"
        assert dur == 0.0


# ---------------------------------------------------------------------------
# _concatenate_scenes (cut-only) tests
# ---------------------------------------------------------------------------


class TestConcatenateScenes:
    """Tests for _concatenate_scenes using FFmpeg concat demuxer."""

    def test_single_clip_copies_directly(self, composer, temp_dir):
        """A single clip is copied, not concatenated."""
        clip = temp_dir / "only_clip.mp4"
        clip.write_bytes(b"video_data")
        output = temp_dir / "output.mp4"

        composer._concatenate_scenes([clip], output)

        assert output.exists()
        assert output.read_bytes() == b"video_data"

    def test_empty_clips_raises_error(self, composer, temp_dir):
        """Empty clip list raises VideoComposerError."""
        with pytest.raises(VideoComposerError, match="No scene clips"):
            composer._concatenate_scenes([], temp_dir / "output.mp4")

    def test_multiple_clips_use_concat_demuxer(self, composer, fake_scene_clips, temp_dir):
        """Multiple clips use FFmpeg concat demuxer with -c copy."""
        output = temp_dir / "concatenated.mp4"
        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            composer._concatenate_scenes(fake_scene_clips, output)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]

        # Should use concat demuxer
        assert "-f" in cmd
        concat_idx = cmd.index("-f")
        assert cmd[concat_idx + 1] == "concat"

        # Should use stream copy (no re-encoding)
        assert "-c" in cmd
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "copy"

    def test_concat_file_lists_all_clips(self, composer, fake_scene_clips, temp_dir):
        """Concat list file contains all clip paths."""
        output = temp_dir / "concatenated.mp4"

        def mock_run(cmd, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            composer._concatenate_scenes(fake_scene_clips, output)

        # Check that concat.txt was created
        concat_file = output.parent / "concat.txt"
        assert concat_file.exists()
        content = concat_file.read_text()
        for clip in fake_scene_clips:
            assert str(clip) in content


# ---------------------------------------------------------------------------
# _concatenate_with_xfade tests
# ---------------------------------------------------------------------------


class TestConcatenateWithXfade:
    """Tests for _concatenate_with_xfade FFmpeg filter_complex generation."""

    def test_single_clip_copies_directly(self, composer, temp_dir):
        """A single clip is copied without xfade."""
        clip = temp_dir / "only_clip.mp4"
        clip.write_bytes(b"video_data")
        output = temp_dir / "xfade_output.mp4"

        composer._concatenate_with_xfade([clip], ["cut"], output)

        assert output.exists()
        assert output.read_bytes() == b"video_data"

    def test_two_clips_dissolve_generates_correct_filter(self, composer, fake_scene_clips, temp_dir):
        """Two clips with dissolve transition generate correct xfade filter."""
        clips = fake_scene_clips[:2]
        transitions = ["cut", "dissolve"]  # First scene has no transition
        output = temp_dir / "xfade_output.mp4"
        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run), \
             patch.object(composer, "_get_video_duration", return_value=5.0):
            composer._concatenate_with_xfade(clips, transitions, output)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]

        # Should have -filter_complex
        assert "-filter_complex" in cmd
        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]

        # Should contain xfade=transition=dissolve
        assert "xfade=transition=dissolve" in filter_str
        assert "duration=0.500" in filter_str

        # Should output to [vout]
        assert "[vout]" in filter_str

        # Should map [vout]
        assert "-map" in cmd
        map_idx = cmd.index("-map")
        assert cmd[map_idx + 1] == "[vout]"

    def test_three_clips_chain_produces_intermediate_labels(
        self, composer, fake_scene_clips, temp_dir
    ):
        """Three clips produce chained xfade with intermediate labels."""
        transitions = ["cut", "dissolve", "fade"]
        output = temp_dir / "xfade_output.mp4"
        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run), \
             patch.object(composer, "_get_video_duration", return_value=5.0):
            composer._concatenate_with_xfade(fake_scene_clips, transitions, output)

        cmd = captured_cmds[0]
        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]

        # Should have two xfade filters separated by semicolon
        parts = filter_str.split(";")
        assert len(parts) == 2

        # First filter: [0][1]xfade...  -> intermediate label
        assert parts[0].startswith("[0][1]xfade")
        assert "[v0]" in parts[0]

        # Second filter: [v0][2]xfade... -> [vout]
        assert "[v0][2]xfade" in parts[1]
        assert "[vout]" in parts[1]

    def test_cut_in_xfade_chain_uses_fade_with_minimal_duration(
        self, composer, fake_scene_clips, temp_dir
    ):
        """A 'cut' within an xfade chain uses fade with 0.04s (near-instant)."""
        transitions = ["cut", "cut", "dissolve"]  # Middle is cut
        output = temp_dir / "xfade_output.mp4"
        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run), \
             patch.object(composer, "_get_video_duration", return_value=5.0):
            composer._concatenate_with_xfade(fake_scene_clips, transitions, output)

        cmd = captured_cmds[0]
        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]

        parts = filter_str.split(";")
        # First boundary (transitions[1] = "cut") should use fade with 0.04s
        assert "xfade=transition=fade" in parts[0]
        assert "duration=0.040" in parts[0]

    def test_offset_calculation_accounts_for_overlap(self, composer, temp_dir):
        """Offsets decrease by transition duration (overlap)."""
        clips = []
        for i in range(3):
            c = temp_dir / f"clip_{i}.mp4"
            c.write_bytes(b"\x00" * 100)
            clips.append(c)

        transitions = ["cut", "dissolve", "dissolve"]
        output = temp_dir / "xfade_output.mp4"
        captured_cmds = []

        # Each clip is 10.0 seconds
        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run), \
             patch.object(composer, "_get_video_duration", return_value=10.0):
            composer._concatenate_with_xfade(clips, transitions, output)

        cmd = captured_cmds[0]
        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]

        # dissolve duration is 0.5s
        # First boundary offset: 10.0 - 0.5 = 9.5
        assert "offset=9.500" in filter_str

        # Second boundary: cumulative = 9.5 + 10.0 = 19.5, offset = 19.5 - 0.5 = 19.0
        assert "offset=19.000" in filter_str

    def test_all_inputs_listed_in_ffmpeg_command(self, composer, fake_scene_clips, temp_dir):
        """All scene clips appear as -i inputs in the FFmpeg command."""
        transitions = ["cut", "dissolve", "fade"]
        output = temp_dir / "xfade_output.mp4"
        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run), \
             patch.object(composer, "_get_video_duration", return_value=5.0):
            composer._concatenate_with_xfade(fake_scene_clips, transitions, output)

        cmd = captured_cmds[0]
        # Count -i occurrences
        input_indices = [i for i, v in enumerate(cmd) if v == "-i"]
        assert len(input_indices) == 3  # One per clip

        # Each clip path should appear after its -i
        for idx, clip in zip(input_indices, fake_scene_clips):
            assert cmd[idx + 1] == str(clip)


# ---------------------------------------------------------------------------
# SFX mixing tests
# ---------------------------------------------------------------------------


class TestSFXMixing:
    """Tests for _mix_sfx audio mixing method."""

    def test_sfx_library_discovery(self, temp_dir):
        """SFX files are auto-discovered from sfx_dir."""
        sfx_dir = temp_dir / "sfx"
        sfx_dir.mkdir()
        (sfx_dir / "whoosh.wav").write_bytes(b"\x00" * 100)
        (sfx_dir / "click.mp3").write_bytes(b"\x00" * 100)
        (sfx_dir / "notes.txt").write_bytes(b"not audio")  # Should be ignored

        composer = VideoComposer(output_dir=temp_dir / "out", sfx_dir=sfx_dir)

        assert "whoosh" in composer._sfx_library
        assert "click" in composer._sfx_library
        assert "notes" not in composer._sfx_library

    def test_mix_sfx_returns_false_when_no_matching_files(self, composer, temp_dir):
        """Returns False when no SFX files match scene cues."""
        master_audio = temp_dir / "master.wav"
        master_audio.write_bytes(b"\x00" * 100)

        timeline = Timeline(
            scenes=[
                TimelineScene(
                    scene_id=0,
                    audio_path=temp_dir / "a.wav",
                    audio_start=0.0,
                    audio_end=5.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="nonexistent_sfx",
                ),
            ],
            master_audio=master_audio,
            total_duration=5.0,
        )

        result = composer._mix_sfx(master_audio, timeline, temp_dir / "output.wav")
        assert result is False

    def test_mix_sfx_skips_none_sound_effects(self, composer_with_sfx, temp_dir):
        """Scenes with sound_effect='none' or None are skipped."""
        master_audio = temp_dir / "master.wav"
        master_audio.write_bytes(b"\x00" * 100)

        timeline = Timeline(
            scenes=[
                TimelineScene(
                    scene_id=0,
                    audio_path=temp_dir / "a.wav",
                    audio_start=0.0,
                    audio_end=5.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="none",
                ),
                TimelineScene(
                    scene_id=1,
                    audio_path=temp_dir / "b.wav",
                    audio_start=5.0,
                    audio_end=10.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="",
                ),
            ],
            master_audio=master_audio,
            total_duration=10.0,
        )

        result = composer_with_sfx._mix_sfx(master_audio, timeline, temp_dir / "output.wav")
        assert result is False

    def test_mix_sfx_generates_correct_filter(self, composer_with_sfx, temp_dir):
        """SFX mixing generates correct adelay + amix filter chain."""
        master_audio = temp_dir / "master.wav"
        master_audio.write_bytes(b"\x00" * 100)

        timeline = Timeline(
            scenes=[
                TimelineScene(
                    scene_id=0,
                    audio_path=temp_dir / "a.wav",
                    audio_start=0.0,
                    audio_end=5.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="whoosh",
                ),
                TimelineScene(
                    scene_id=1,
                    audio_path=temp_dir / "b.wav",
                    audio_start=5.0,
                    audio_end=10.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="typing",
                ),
            ],
            master_audio=master_audio,
            total_duration=10.0,
        )

        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            result = composer_with_sfx._mix_sfx(
                master_audio, timeline, temp_dir / "output.wav"
            )

        assert result is True
        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]

        # Should have -filter_complex
        assert "-filter_complex" in cmd
        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]

        # Check SFX delay values
        # Scene 0 at audio_start=0.0 -> adelay=0|0
        assert "adelay=0|0" in filter_str
        # Scene 1 at audio_start=5.0 -> adelay=5000|5000
        assert "adelay=5000|5000" in filter_str

        # Check amix with correct input count (1 master + 2 SFX = 3)
        assert "amix=inputs=3" in filter_str

        # Check volume adjustment
        assert "volume=0.7" in filter_str

        # Should have 3 inputs: master + 2 SFX files
        input_count = sum(1 for v in cmd if v == "-i")
        assert input_count == 3

    def test_mix_sfx_partial_match(self, composer_with_sfx, temp_dir):
        """SFX partial matching works (e.g., 'dramatic' matches 'dramatic_hit')."""
        master_audio = temp_dir / "master.wav"
        master_audio.write_bytes(b"\x00" * 100)

        timeline = Timeline(
            scenes=[
                TimelineScene(
                    scene_id=0,
                    audio_path=temp_dir / "a.wav",
                    audio_start=0.0,
                    audio_end=5.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="dramatic",  # Partial match for "dramatic_hit"
                ),
            ],
            master_audio=master_audio,
            total_duration=5.0,
        )

        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            result = composer_with_sfx._mix_sfx(
                master_audio, timeline, temp_dir / "output.wav"
            )

        assert result is True

    def test_mix_sfx_missing_file_skipped(self, temp_dir):
        """SFX entry in library pointing to missing file is skipped."""
        sfx_dir = temp_dir / "sfx"
        sfx_dir.mkdir()
        # Create file then delete it after composer discovers it
        ghost = sfx_dir / "ghost.wav"
        ghost.write_bytes(b"\x00" * 100)
        composer = VideoComposer(output_dir=temp_dir / "out", sfx_dir=sfx_dir)
        ghost.unlink()  # File no longer exists

        master_audio = temp_dir / "master.wav"
        master_audio.write_bytes(b"\x00" * 100)

        timeline = Timeline(
            scenes=[
                TimelineScene(
                    scene_id=0,
                    audio_path=temp_dir / "a.wav",
                    audio_start=0.0,
                    audio_end=5.0,
                    visual_path=temp_dir / "v.mp4",
                    visual_type=VisualType.BROLL_VIDEO,
                    sound_effect="ghost",
                ),
            ],
            master_audio=master_audio,
            total_duration=5.0,
        )

        result = composer._mix_sfx(master_audio, timeline, temp_dir / "output.wav")
        assert result is False


# ---------------------------------------------------------------------------
# Scene rendering dispatch tests
# ---------------------------------------------------------------------------


class TestRenderScene:
    """Tests for _render_scene visual type dispatch."""

    @pytest.mark.asyncio
    async def test_broll_video_calls_normalize(self, composer, temp_dir):
        """BROLL_VIDEO scenes call _normalize_video."""
        scene = TimelineScene(
            scene_id=0,
            audio_path=temp_dir / "a.wav",
            audio_start=0.0,
            audio_end=5.0,
            visual_path=temp_dir / "v.mp4",
            visual_type=VisualType.BROLL_VIDEO,
        )
        scene.visual_path.write_bytes(b"\x00" * 100)

        with patch.object(composer, "_normalize_video") as mock_norm:
            # Make the output file exist
            def create_output(*args, **kwargs):
                output_path = args[1]
                output_path.write_bytes(b"\x00" * 100)

            mock_norm.side_effect = create_output

            result = await composer._render_scene(
                scene, DEFAULT_WIDTH, DEFAULT_HEIGHT, temp_dir, 0
            )

            mock_norm.assert_called_once()
            assert result.exists()

    @pytest.mark.asyncio
    async def test_generated_image_calls_ken_burns(self, composer, temp_dir):
        """GENERATED_IMAGE scenes call _apply_ken_burns."""
        scene = TimelineScene(
            scene_id=0,
            audio_path=temp_dir / "a.wav",
            audio_start=0.0,
            audio_end=5.0,
            visual_path=temp_dir / "image.png",
            visual_type=VisualType.GENERATED_IMAGE,
        )
        scene.visual_path.write_bytes(b"\x00" * 100)

        with patch.object(composer, "_apply_ken_burns") as mock_kb:
            def create_output(*args, **kwargs):
                output_path = args[1]
                output_path.write_bytes(b"\x00" * 100)

            mock_kb.side_effect = create_output

            result = await composer._render_scene(
                scene, DEFAULT_WIDTH, DEFAULT_HEIGHT, temp_dir, 0
            )

            mock_kb.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_graphic_calls_create_text(self, composer, temp_dir):
        """TEXT_GRAPHIC scenes call _create_text_graphic."""
        text_file = temp_dir / "text.txt"
        text_file.write_text("Hello World")

        scene = TimelineScene(
            scene_id=0,
            audio_path=temp_dir / "a.wav",
            audio_start=0.0,
            audio_end=5.0,
            visual_path=text_file,
            visual_type=VisualType.TEXT_GRAPHIC,
        )

        with patch.object(composer, "_create_text_graphic") as mock_txt:
            def create_output(*args, **kwargs):
                output_path = args[1]
                output_path.write_bytes(b"\x00" * 100)

            mock_txt.side_effect = create_output

            result = await composer._render_scene(
                scene, DEFAULT_WIDTH, DEFAULT_HEIGHT, temp_dir, 0
            )

            mock_txt.assert_called_once()
            # First arg should be the text content
            assert mock_txt.call_args[0][0] == "Hello World"

    @pytest.mark.asyncio
    async def test_render_scene_raises_on_missing_output(self, composer, temp_dir):
        """Raises VideoComposerError if rendered file doesn't exist."""
        scene = TimelineScene(
            scene_id=0,
            audio_path=temp_dir / "a.wav",
            audio_start=0.0,
            audio_end=5.0,
            visual_path=temp_dir / "v.mp4",
            visual_type=VisualType.BROLL_VIDEO,
        )
        scene.visual_path.write_bytes(b"\x00" * 100)

        with patch.object(composer, "_normalize_video"):  # Does NOT create file
            with pytest.raises(VideoComposerError, match="Scene rendering failed"):
                await composer._render_scene(
                    scene, DEFAULT_WIDTH, DEFAULT_HEIGHT, temp_dir, 0
                )


# ---------------------------------------------------------------------------
# Full compose flow tests
# ---------------------------------------------------------------------------


class TestCompose:
    """Tests for the full compose pipeline."""

    @pytest.mark.asyncio
    async def test_compose_raises_on_empty_timeline(self, composer, temp_dir):
        """compose raises VideoComposerError for empty timeline."""
        timeline = Timeline(
            scenes=[],
            master_audio=temp_dir / "audio.wav",
            total_duration=0.0,
        )

        with pytest.raises(VideoComposerError, match="no scenes"):
            await composer.compose(timeline)

    @pytest.mark.asyncio
    async def test_compose_draft_uses_lower_resolution(self, composer, sample_timeline, temp_dir):
        """Draft mode renders at 480p (DRAFT_WIDTH x DRAFT_HEIGHT)."""
        render_widths = []

        async def mock_render_scene(scene, width, height, tmp, idx, draft=False):
            render_widths.append(width)
            output = tmp / f"scene_{idx:03d}.mp4"
            output.write_bytes(b"\x00" * 100)
            return output

        with patch.object(composer, "_render_scene", side_effect=mock_render_scene), \
             patch.object(composer, "_concatenate_with_xfade"), \
             patch.object(composer, "_concatenate_scenes"), \
             patch.object(composer, "_add_audio") as mock_audio, \
             patch("shutil.move"):

            # Make the with_audio.mp4 exist
            def create_audio_output(*args):
                Path(args[2]).write_bytes(b"\x00" * 100)

            mock_audio.side_effect = create_audio_output

            await composer.compose(sample_timeline, draft=True)

        # All scenes should have been rendered at draft width
        for w in render_widths:
            assert w == DRAFT_WIDTH

    @pytest.mark.asyncio
    async def test_compose_uses_xfade_when_transitions_present(
        self, composer, sample_timeline, temp_dir
    ):
        """Compose uses xfade when any scene has non-cut transition."""
        async def mock_render_scene(scene, width, height, tmp, idx, draft=False):
            output = tmp / f"scene_{idx:03d}.mp4"
            output.write_bytes(b"\x00" * 100)
            return output

        with patch.object(composer, "_render_scene", side_effect=mock_render_scene), \
             patch.object(composer, "_concatenate_with_xfade") as mock_xfade, \
             patch.object(composer, "_concatenate_scenes") as mock_concat, \
             patch.object(composer, "_add_audio") as mock_audio, \
             patch.object(composer, "_apply_color_grade") as mock_grade, \
             patch("shutil.move"):

            def create_audio_output(*args):
                Path(args[2]).write_bytes(b"\x00" * 100)

            mock_audio.side_effect = create_audio_output

            def create_graded_output(*args):
                Path(args[1]).write_bytes(b"\x00" * 100)

            mock_grade.side_effect = create_graded_output

            await composer.compose(sample_timeline, draft=False)

        # sample_timeline has "dissolve" and "fade" transitions, so xfade should be used
        mock_xfade.assert_called_once()
        mock_concat.assert_not_called()


# ---------------------------------------------------------------------------
# Color grade tests
# ---------------------------------------------------------------------------


class TestColorGrade:
    """Tests for _apply_color_grade."""

    def test_neutral_grade_copies_file(self, composer, temp_dir):
        """Neutral color grade just copies the file."""
        video = temp_dir / "input.mp4"
        video.write_bytes(b"original_video")
        output = temp_dir / "graded.mp4"

        composer._apply_color_grade(video, output, "neutral")

        assert output.exists()
        assert output.read_bytes() == b"original_video"

    def test_dark_cinematic_runs_ffmpeg(self, composer, temp_dir):
        """dark_cinematic applies eq+curves filters via FFmpeg."""
        video = temp_dir / "input.mp4"
        video.write_bytes(b"\x00" * 100)
        output = temp_dir / "graded.mp4"
        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            composer._apply_color_grade(video, output, "dark_cinematic")

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "-vf" in cmd
        vf_idx = cmd.index("-vf")
        vf_filter = cmd[vf_idx + 1]
        assert "contrast=1.2" in vf_filter
        assert "brightness=-0.05" in vf_filter


# ---------------------------------------------------------------------------
# Read text graphic utility tests
# ---------------------------------------------------------------------------


class TestReadTextGraphic:
    """Tests for _read_text_graphic static method."""

    def test_reads_existing_txt_file(self, temp_dir):
        """Reads text from existing .txt file."""
        txt = temp_dir / "content.txt"
        txt.write_text("Hello World!")
        result = VideoComposer._read_text_graphic(txt)
        assert result == "Hello World!"

    def test_reads_stem_for_non_txt(self, temp_dir):
        """Uses filename stem (without extension) for non-.txt paths."""
        path = temp_dir / "hello_world.mp4"
        result = VideoComposer._read_text_graphic(path)
        assert result == "hello world"

    def test_reads_stem_for_missing_file(self):
        """Uses filename stem for nonexistent paths."""
        path = Path("/nonexistent/my_title.txt")
        result = VideoComposer._read_text_graphic(path)
        assert result == "my title"


# ---------------------------------------------------------------------------
# FFmpeg error handling tests
# ---------------------------------------------------------------------------


class TestRunFFmpeg:
    """Tests for _run_ffmpeg error handling."""

    def test_raises_on_nonzero_exit(self, composer):
        """Raises VideoComposerError on FFmpeg failure."""
        def mock_run(cmd, **kwargs):
            result = Mock()
            result.returncode = 1
            result.stderr = "Error: codec not found"
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(VideoComposerError, match="FFmpeg failed"):
                composer._run_ffmpeg(["ffmpeg", "-version"], "test command")

    def test_succeeds_on_zero_exit(self, composer):
        """No exception on successful FFmpeg run."""
        def mock_run(cmd, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            composer._run_ffmpeg(["ffmpeg", "-version"], "test command")
