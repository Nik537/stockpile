"""Integration tests for the director review loop in the video production pipeline.

Tests the full director review flow with mocked services:
- Draft rendered at 480p for review
- Director reviews and returns structured feedback
- Fixes are applied to timeline (visual_mismatch, transition_jarring, pacing_issue)
- Max 2 review loops (doesn't loop forever)
- Director disabled config skips review entirely
- Approval after fixes applied stops the loop
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from video_agent.models import (
    DraftReview,
    FixRequest,
    HookScript,
    Script,
    SceneScript,
    Timeline,
    TimelineScene,
    VisualType,
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
def sample_script():
    """Create a sample Script for integration testing."""
    return Script(
        title="Test Video",
        hook=HookScript(
            voiceover="This is the hook voiceover text for testing.",
            visual_description="Dramatic opening shot",
            visual_keywords=["dramatic", "opening"],
            sound_effect="whoosh",
        ),
        scenes=[
            SceneScript(
                id=1,
                duration_est=8.0,
                voiceover="Scene one voiceover discussing the main topic in detail.",
                visual_keywords=["main", "topic", "detail"],
                visual_style="documentary",
                visual_type=VisualType.BROLL_VIDEO,
                transition_in="dissolve",
                music_mood="neutral",
            ),
            SceneScript(
                id=2,
                duration_est=10.0,
                voiceover="Scene two explores the secondary topic with different visuals.",
                visual_keywords=["secondary", "topic"],
                visual_style="cinematic",
                visual_type=VisualType.BROLL_VIDEO,
                transition_in="fade",
                music_mood="uplifting",
            ),
        ],
        metadata={"topic": "test", "style": "documentary"},
    )


@pytest.fixture
def sample_timeline(temp_dir):
    """Create a sample Timeline for integration testing."""
    audio = temp_dir / "master.wav"
    audio.write_bytes(b"\x00" * 100)

    visual = temp_dir / "visual.mp4"
    visual.write_bytes(b"\x00" * 1024)

    scenes = [
        TimelineScene(
            scene_id=0,
            audio_path=temp_dir / "a0.wav",
            audio_start=0.0,
            audio_end=5.0,
            visual_path=visual,
            visual_type=VisualType.BROLL_VIDEO,
            transition="cut",
        ),
        TimelineScene(
            scene_id=1,
            audio_path=temp_dir / "a1.wav",
            audio_start=5.0,
            audio_end=13.0,
            visual_path=visual,
            visual_type=VisualType.BROLL_VIDEO,
            transition="dissolve",
        ),
        TimelineScene(
            scene_id=2,
            audio_path=temp_dir / "a2.wav",
            audio_start=13.0,
            audio_end=23.0,
            visual_path=visual,
            visual_type=VisualType.BROLL_VIDEO,
            transition="fade",
        ),
    ]

    for s in scenes:
        s.audio_path.write_bytes(b"\x00" * 100)

    return Timeline(
        scenes=scenes,
        master_audio=audio,
        total_duration=23.0,
        color_grade="dark_cinematic",
    )


def _make_agent(config_overrides=None):
    """Create a VideoProductionAgent with mocked services."""
    config = {
        "gemini_api_key": "test_key",
        "gemini_model": "gemini-3-flash-preview",
        "local_output_folder": "/tmp/test_output",
        "director_review_enabled": True,
        "director_model": "gemini-2.5-flash",
        "director_approval_threshold": 7,
        "director_max_iterations": 2,
        "broll_source_priority": ["youtube", "pexels", "pixabay"],
    }
    if config_overrides:
        config.update(config_overrides)

    with patch("video_agent.agent.TTSService"), \
         patch("video_agent.agent.AIService"), \
         patch("video_agent.agent.MusicService"), \
         patch("video_agent.agent.ImageGenerationService"), \
         patch("video_agent.agent.ScriptGenerator"), \
         patch("video_agent.director.Client"):
        from video_agent.agent import VideoProductionAgent
        return VideoProductionAgent(config=config)


# ---------------------------------------------------------------------------
# Director review loop integration tests
# ---------------------------------------------------------------------------


class TestDirectorReviewLoop:
    """Tests for _director_review_loop integration."""

    @pytest.mark.asyncio
    async def test_approved_on_first_review(self, sample_script, sample_timeline, temp_dir):
        """Loop ends after first review if draft is approved."""
        agent = _make_agent()
        agent.composer = MagicMock()

        # Mock compose to return a draft path
        draft_path = temp_dir / "draft.mp4"
        draft_path.write_bytes(b"\x00" * 1024)
        agent.composer.compose = AsyncMock(return_value=draft_path)

        # Mock director to approve immediately
        approved_review = DraftReview(
            overall_score=8,
            fix_requests=[],
            approved=True,
            iteration=0,
            notes="Looks great.",
        )
        agent.director.review_draft = Mock(return_value=approved_review)

        result = await agent._director_review_loop(sample_script, sample_timeline, temp_dir)

        # Should compose only once (draft) and review once
        agent.composer.compose.assert_called_once()
        call_kwargs = agent.composer.compose.call_args
        assert call_kwargs[1]["draft"] is True or call_kwargs[0][1] is True
        agent.director.review_draft.assert_called_once()

        # Timeline returned unchanged
        assert result is sample_timeline

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self, sample_script, sample_timeline, temp_dir):
        """Loop stops after director_max_iterations even if not approved."""
        agent = _make_agent({"director_max_iterations": 2})
        agent.composer = MagicMock()

        draft_path = temp_dir / "draft.mp4"
        draft_path.write_bytes(b"\x00" * 1024)
        agent.composer.compose = AsyncMock(return_value=draft_path)

        # Mock director to never approve, always request fixes
        rejected_review = DraftReview(
            overall_score=4,
            fix_requests=[
                FixRequest(
                    scene_id=1,
                    issue_type="visual_mismatch",
                    description="Wrong visual",
                    suggested_keywords=["better", "keywords"],
                    priority="high",
                ),
            ],
            approved=False,
            iteration=0,
        )
        agent.director.review_draft = Mock(return_value=rejected_review)

        # Mock _apply_fixes to return timeline unchanged
        agent._apply_fixes = AsyncMock(return_value=sample_timeline)

        result = await agent._director_review_loop(sample_script, sample_timeline, temp_dir)

        # Should have done exactly 2 iterations
        assert agent.director.review_draft.call_count == 2
        assert agent.composer.compose.call_count == 2
        assert agent._apply_fixes.call_count == 2

    @pytest.mark.asyncio
    async def test_fixes_applied_between_iterations(
        self, sample_script, sample_timeline, temp_dir
    ):
        """Fixes from first review are applied before second render."""
        agent = _make_agent({"director_max_iterations": 2})
        agent.composer = MagicMock()

        draft_path = temp_dir / "draft.mp4"
        draft_path.write_bytes(b"\x00" * 1024)
        agent.composer.compose = AsyncMock(return_value=draft_path)

        fix_request = FixRequest(
            scene_id=1,
            issue_type="visual_mismatch",
            description="Wrong visual",
            suggested_keywords=["correct", "visual"],
            priority="high",
        )

        # First review: rejected with fixes. Second review: approved.
        reviews = [
            DraftReview(
                overall_score=4,
                fix_requests=[fix_request],
                approved=False,
                iteration=0,
            ),
            DraftReview(
                overall_score=8,
                fix_requests=[],
                approved=True,
                iteration=1,
            ),
        ]
        agent.director.review_draft = Mock(side_effect=reviews)
        agent._apply_fixes = AsyncMock(return_value=sample_timeline)

        result = await agent._director_review_loop(sample_script, sample_timeline, temp_dir)

        # _apply_fixes should have been called once (after first rejection)
        agent._apply_fixes.assert_called_once()
        # And the fix_requests should match what the director returned
        call_args = agent._apply_fixes.call_args
        assert call_args[0][2] == [fix_request]

    @pytest.mark.asyncio
    async def test_no_fixes_but_not_approved_breaks_loop(
        self, sample_script, sample_timeline, temp_dir
    ):
        """If no fixes requested but not approved, loop breaks."""
        agent = _make_agent({"director_max_iterations": 3})
        agent.composer = MagicMock()

        draft_path = temp_dir / "draft.mp4"
        draft_path.write_bytes(b"\x00" * 1024)
        agent.composer.compose = AsyncMock(return_value=draft_path)

        # Review returns low score but no fixes
        review = DraftReview(
            overall_score=5,
            fix_requests=[],
            approved=False,
            iteration=0,
        )
        agent.director.review_draft = Mock(return_value=review)

        result = await agent._director_review_loop(sample_script, sample_timeline, temp_dir)

        # Should only do 1 iteration (breaks because no fixes)
        assert agent.director.review_draft.call_count == 1

    @pytest.mark.asyncio
    async def test_draft_rendered_at_480p(self, sample_script, sample_timeline, temp_dir):
        """Draft is rendered with draft=True for 480p preview."""
        agent = _make_agent()
        agent.composer = MagicMock()

        draft_path = temp_dir / "draft.mp4"
        draft_path.write_bytes(b"\x00" * 1024)
        agent.composer.compose = AsyncMock(return_value=draft_path)

        approved_review = DraftReview(
            overall_score=9,
            fix_requests=[],
            approved=True,
            iteration=0,
        )
        agent.director.review_draft = Mock(return_value=approved_review)

        await agent._director_review_loop(sample_script, sample_timeline, temp_dir)

        # compose should be called with draft=True
        compose_call = agent.composer.compose.call_args
        if compose_call[1]:
            assert compose_call[1].get("draft") is True
        else:
            assert compose_call[0][1] is True  # positional arg


# ---------------------------------------------------------------------------
# _apply_fixes integration tests
# ---------------------------------------------------------------------------


class TestApplyFixes:
    """Tests for _apply_fixes method."""

    @pytest.mark.asyncio
    async def test_visual_mismatch_reacquires_asset(self, sample_script, sample_timeline, temp_dir):
        """visual_mismatch fix re-downloads B-roll with suggested keywords."""
        agent = _make_agent()
        new_visual = temp_dir / "new_visual.mp4"
        new_visual.write_bytes(b"\x00" * 1024)

        agent._search_and_download_broll = AsyncMock(return_value=new_visual)

        fixes = [
            FixRequest(
                scene_id=1,
                issue_type="visual_mismatch",
                description="Wrong visual",
                suggested_keywords=["correct", "keywords"],
                priority="high",
            ),
        ]

        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)

        # Should have called _search_and_download_broll with suggested keywords
        agent._search_and_download_broll.assert_called_once()
        call_kwargs = agent._search_and_download_broll.call_args
        assert call_kwargs[0][0] == ["correct", "keywords"]

        # Scene 1 visual_path should be updated
        scene_1 = [s for s in result.scenes if s.scene_id == 1][0]
        assert scene_1.visual_path == new_visual
        assert scene_1.visual_type == VisualType.BROLL_VIDEO

    @pytest.mark.asyncio
    async def test_visual_mismatch_falls_back_to_script_keywords(
        self, sample_script, sample_timeline, temp_dir
    ):
        """visual_mismatch with no suggested_keywords uses script keywords."""
        agent = _make_agent()
        new_visual = temp_dir / "new_visual.mp4"
        new_visual.write_bytes(b"\x00" * 1024)

        agent._search_and_download_broll = AsyncMock(return_value=new_visual)

        fixes = [
            FixRequest(
                scene_id=1,
                issue_type="visual_mismatch",
                description="Wrong visual",
                suggested_keywords=[],  # Empty - should fall back to script
                priority="medium",
            ),
        ]

        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)

        # Should have used script scene 1 keywords as fallback
        call_kwargs = agent._search_and_download_broll.call_args
        assert call_kwargs[0][0] == ["main", "topic", "detail"]

    @pytest.mark.asyncio
    async def test_transition_jarring_changes_transition(
        self, sample_script, sample_timeline, temp_dir
    ):
        """transition_jarring fix changes the scene transition."""
        agent = _make_agent()

        fixes = [
            FixRequest(
                scene_id=1,
                issue_type="transition_jarring",
                description="Abrupt cut",
                suggested_fix="Use a dissolve for smoother transition",
                priority="medium",
            ),
        ]

        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)

        scene_1 = [s for s in result.scenes if s.scene_id == 1][0]
        assert scene_1.transition == "dissolve"

    @pytest.mark.asyncio
    async def test_transition_jarring_fade_suggestion(
        self, sample_script, sample_timeline, temp_dir
    ):
        """transition_jarring fix with 'fade' in suggestion uses fade."""
        agent = _make_agent()

        fixes = [
            FixRequest(
                scene_id=1,
                issue_type="transition_jarring",
                description="Too abrupt",
                suggested_fix="A slow fade would work better here",
                priority="low",
            ),
        ]

        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)

        scene_1 = [s for s in result.scenes if s.scene_id == 1][0]
        assert scene_1.transition == "fade"

    @pytest.mark.asyncio
    async def test_transition_jarring_default_dissolve(
        self, sample_script, sample_timeline, temp_dir
    ):
        """transition_jarring fix defaults to dissolve if no match."""
        agent = _make_agent()

        fixes = [
            FixRequest(
                scene_id=1,
                issue_type="transition_jarring",
                description="Too abrupt",
                suggested_fix="Try something smoother",
                priority="low",
            ),
        ]

        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)

        scene_1 = [s for s in result.scenes if s.scene_id == 1][0]
        assert scene_1.transition == "dissolve"

    @pytest.mark.asyncio
    async def test_pacing_issue_is_logged_only(self, sample_script, sample_timeline, temp_dir):
        """pacing_issue fix is logged but timeline is unchanged."""
        agent = _make_agent()

        original_scenes = [
            (s.scene_id, s.audio_start, s.audio_end) for s in sample_timeline.scenes
        ]

        fixes = [
            FixRequest(
                scene_id=1,
                issue_type="pacing_issue",
                description="Scene too long",
                priority="medium",
            ),
        ]

        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)

        # Timing should be unchanged (pacing is manual)
        result_scenes = [(s.scene_id, s.audio_start, s.audio_end) for s in result.scenes]
        assert result_scenes == original_scenes

    @pytest.mark.asyncio
    async def test_unknown_scene_id_skipped(self, sample_script, sample_timeline, temp_dir):
        """Fix referencing non-existent scene_id is skipped."""
        agent = _make_agent()

        fixes = [
            FixRequest(
                scene_id=99,  # Doesn't exist
                issue_type="visual_mismatch",
                description="Wrong",
                suggested_keywords=["test"],
                priority="high",
            ),
        ]

        # Should not raise
        result = await agent._apply_fixes(sample_timeline, sample_script, fixes, temp_dir)
        assert result is sample_timeline


# ---------------------------------------------------------------------------
# Director disabled config tests
# ---------------------------------------------------------------------------


class TestDirectorDisabled:
    """Tests for director_review_enabled=False config."""

    def test_director_not_created_when_disabled(self):
        """Director agent is None when disabled in config."""
        agent = _make_agent({"director_review_enabled": False})
        assert agent.director is None

    def test_director_created_when_enabled(self):
        """Director agent is created when enabled in config."""
        agent = _make_agent({"director_review_enabled": True})
        assert agent.director is not None


# ---------------------------------------------------------------------------
# Director max iterations config tests
# ---------------------------------------------------------------------------


class TestDirectorMaxIterations:
    """Tests for director_max_iterations config."""

    def test_default_max_iterations(self):
        """Default max iterations is 2."""
        agent = _make_agent()
        assert agent.director_max_iterations == 2

    def test_custom_max_iterations(self):
        """Custom max iterations from config."""
        agent = _make_agent({"director_max_iterations": 5})
        assert agent.director_max_iterations == 5

    @pytest.mark.asyncio
    async def test_single_iteration_loop(self, sample_script, sample_timeline, temp_dir):
        """director_max_iterations=1 does only one review."""
        agent = _make_agent({"director_max_iterations": 1})
        agent.composer = MagicMock()

        draft_path = temp_dir / "draft.mp4"
        draft_path.write_bytes(b"\x00" * 1024)
        agent.composer.compose = AsyncMock(return_value=draft_path)

        rejected_review = DraftReview(
            overall_score=4,
            fix_requests=[
                FixRequest(
                    scene_id=1,
                    issue_type="visual_mismatch",
                    description="Wrong",
                    priority="high",
                ),
            ],
            approved=False,
            iteration=0,
        )
        agent.director.review_draft = Mock(return_value=rejected_review)
        agent._apply_fixes = AsyncMock(return_value=sample_timeline)

        await agent._director_review_loop(sample_script, sample_timeline, temp_dir)

        # Exactly 1 review
        assert agent.director.review_draft.call_count == 1
