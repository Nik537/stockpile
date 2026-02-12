"""Unit tests for the DirectorAgent draft review system.

Tests:
- DraftReview parsing from Gemini JSON responses
- review_draft() with approved response (score >= 7)
- review_draft() with fixes needed (score < 7)
- FixRequest parsing for each issue_type
- Graceful degradation when Gemini API fails (returns pass-through review)
- Review prompt includes script and timeline content
- Priority parsing and validation
- Score clamping to 1-10 range
- High-priority fix overrides approval even if score is high
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from video_agent.director import (
    DEFAULT_APPROVAL_THRESHOLD,
    DIRECTOR_REVIEW_PROMPT,
    DirectorAgent,
)
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
    """Create a sample Script for testing."""
    return Script(
        title="The History of Coffee",
        hook=HookScript(
            voiceover="Did you know that coffee was first discovered by goats?",
            visual_description="Close-up of coffee beans being roasted",
            visual_keywords=["coffee beans", "roasting", "close-up"],
            sound_effect="whoosh",
        ),
        scenes=[
            SceneScript(
                id=1,
                duration_est=8.0,
                voiceover="In the 9th century, Ethiopian goat herders noticed their goats dancing after eating coffee cherries.",
                visual_keywords=["ethiopia", "goat", "coffee plant"],
                visual_style="documentary",
                visual_type=VisualType.BROLL_VIDEO,
                transition_in="dissolve",
                music_mood="neutral",
            ),
            SceneScript(
                id=2,
                duration_est=10.0,
                voiceover="Coffee then spread through the Arab world, becoming a staple of social life.",
                visual_keywords=["arab", "coffee house", "traditional"],
                visual_style="cinematic",
                visual_type=VisualType.BROLL_VIDEO,
                transition_in="fade",
                music_mood="uplifting",
            ),
        ],
        metadata={"topic": "coffee history", "style": "documentary"},
    )


@pytest.fixture
def sample_timeline(temp_dir):
    """Create a sample Timeline for testing."""
    audio_path = temp_dir / "master.wav"
    audio_path.write_bytes(b"\x00" * 100)

    visual = temp_dir / "visual.mp4"
    visual.write_bytes(b"\x00" * 100)

    return Timeline(
        scenes=[
            TimelineScene(
                scene_id=0,
                audio_path=temp_dir / "a0.wav",
                audio_start=0.0,
                audio_end=5.0,
                visual_path=visual,
                visual_type=VisualType.BROLL_VIDEO,
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
        ],
        master_audio=audio_path,
        total_duration=23.0,
        color_grade="dark_cinematic",
    )


@pytest.fixture
def director():
    """Create a DirectorAgent with mocked Gemini client."""
    with patch("video_agent.director.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        agent = DirectorAgent(api_key="test_key", approval_threshold=7)
        return agent


# ---------------------------------------------------------------------------
# _parse_review_response tests
# ---------------------------------------------------------------------------


class TestParseReviewResponse:
    """Tests for _parse_review_response JSON parsing."""

    def test_parses_approved_review(self, director):
        """Parses an approved review with high score."""
        response = json.dumps({
            "overall_score": 8,
            "approved": True,
            "notes": "Great quality draft, minimal issues.",
            "fixes": [],
        })

        review = director._parse_review_response(response, iteration=0)

        assert review.overall_score == 8
        assert review.approved is True
        assert review.notes == "Great quality draft, minimal issues."
        assert review.fix_requests == []
        assert review.iteration == 0

    def test_parses_rejected_review_with_fixes(self, director):
        """Parses a rejected review with fix requests."""
        response = json.dumps({
            "overall_score": 4,
            "approved": False,
            "notes": "Significant visual mismatches.",
            "fixes": [
                {
                    "scene_id": 1,
                    "issue_type": "visual_mismatch",
                    "description": "Scene 1 shows urban footage, but script calls for Ethiopian countryside",
                    "suggested_fix": "Search for Ethiopian rural landscape footage",
                    "suggested_keywords": ["ethiopia", "countryside", "rural"],
                    "priority": "high",
                },
                {
                    "scene_id": 2,
                    "issue_type": "transition_jarring",
                    "description": "Abrupt cut between scenes 1 and 2",
                    "suggested_fix": "Use dissolve transition",
                    "priority": "medium",
                },
            ],
        })

        review = director._parse_review_response(response, iteration=1)

        assert review.overall_score == 4
        assert review.approved is False
        assert len(review.fix_requests) == 2
        assert review.iteration == 1

        # Check first fix
        fix1 = review.fix_requests[0]
        assert fix1.scene_id == 1
        assert fix1.issue_type == "visual_mismatch"
        assert fix1.priority == "high"
        assert fix1.suggested_keywords == ["ethiopia", "countryside", "rural"]

        # Check second fix
        fix2 = review.fix_requests[1]
        assert fix2.scene_id == 2
        assert fix2.issue_type == "transition_jarring"
        assert fix2.priority == "medium"

    def test_parses_each_issue_type(self, director):
        """Parses all recognized issue types correctly."""
        issue_types = [
            "visual_mismatch",
            "pacing_issue",
            "transition_jarring",
            "audio_sync",
            "content_gap",
        ]

        for issue_type in issue_types:
            response = json.dumps({
                "overall_score": 5,
                "approved": False,
                "notes": "Testing",
                "fixes": [
                    {
                        "scene_id": 1,
                        "issue_type": issue_type,
                        "description": f"Test {issue_type}",
                        "priority": "medium",
                    }
                ],
            })

            review = director._parse_review_response(response, iteration=0)
            assert review.fix_requests[0].issue_type == issue_type

    def test_score_clamped_to_1_10_range(self, director):
        """Scores are clamped to [1, 10] range."""
        # Score below 1
        response = json.dumps({
            "overall_score": -5,
            "approved": False,
            "notes": "",
            "fixes": [],
        })
        review = director._parse_review_response(response, iteration=0)
        assert review.overall_score == 1

        # Score above 10
        response = json.dumps({
            "overall_score": 15,
            "approved": True,
            "notes": "",
            "fixes": [],
        })
        review = director._parse_review_response(response, iteration=0)
        assert review.overall_score == 10

    def test_high_priority_fix_prevents_approval(self, director):
        """Score >= 7 but high-priority fix overrides to not approved."""
        response = json.dumps({
            "overall_score": 8,
            "approved": True,
            "notes": "Good overall but critical visual issue.",
            "fixes": [
                {
                    "scene_id": 1,
                    "issue_type": "visual_mismatch",
                    "description": "Completely wrong visual",
                    "priority": "high",
                }
            ],
        })

        review = director._parse_review_response(response, iteration=0)
        assert review.overall_score == 8
        assert review.approved is False  # High priority overrides

    def test_low_score_not_approved_even_if_ai_says_approved(self, director):
        """Score < threshold is not approved even if AI says approved."""
        response = json.dumps({
            "overall_score": 5,
            "approved": True,  # AI says approved but score is too low
            "notes": "",
            "fixes": [],
        })

        review = director._parse_review_response(response, iteration=0)
        assert review.approved is False  # Threshold logic overrides

    def test_invalid_priority_defaults_to_medium(self, director):
        """Invalid priority values default to 'medium'."""
        response = json.dumps({
            "overall_score": 5,
            "approved": False,
            "notes": "",
            "fixes": [
                {
                    "scene_id": 1,
                    "issue_type": "visual_mismatch",
                    "description": "Test",
                    "priority": "critical",  # Not a valid priority
                }
            ],
        })

        review = director._parse_review_response(response, iteration=0)
        assert review.fix_requests[0].priority == "medium"

    def test_invalid_json_returns_passthrough(self, director):
        """Invalid JSON response returns pass-through review."""
        review = director._parse_review_response("not valid json at all", iteration=0)

        assert review.approved is True
        assert review.overall_score == 7
        assert review.fix_requests == []
        assert "Failed to parse" in review.notes

    def test_strips_markdown_code_blocks(self, director):
        """JSON wrapped in markdown fences is parsed correctly."""
        inner_json = json.dumps({
            "overall_score": 9,
            "approved": True,
            "notes": "Perfect.",
            "fixes": [],
        })
        response = f"```json\n{inner_json}\n```"

        review = director._parse_review_response(response, iteration=0)
        assert review.overall_score == 9
        assert review.approved is True

    def test_missing_fields_have_defaults(self, director):
        """Missing JSON fields use sensible defaults."""
        response = json.dumps({})  # All fields missing

        review = director._parse_review_response(response, iteration=0)
        assert review.overall_score == 5  # Default
        assert review.notes == ""
        assert review.fix_requests == []

    def test_suggested_keywords_handles_non_list(self, director):
        """Non-list suggested_keywords converted to empty list."""
        response = json.dumps({
            "overall_score": 5,
            "approved": False,
            "notes": "",
            "fixes": [
                {
                    "scene_id": 1,
                    "issue_type": "visual_mismatch",
                    "description": "Test",
                    "suggested_keywords": "not a list",
                    "priority": "medium",
                }
            ],
        })

        review = director._parse_review_response(response, iteration=0)
        assert review.fix_requests[0].suggested_keywords == []

    def test_non_dict_fixes_are_skipped(self, director):
        """Non-dict entries in fixes list are skipped."""
        response = json.dumps({
            "overall_score": 5,
            "approved": False,
            "notes": "",
            "fixes": ["not_a_dict", 123, None],
        })

        review = director._parse_review_response(response, iteration=0)
        assert review.fix_requests == []


# ---------------------------------------------------------------------------
# _build_review_prompt tests
# ---------------------------------------------------------------------------


class TestBuildReviewPrompt:
    """Tests for _build_review_prompt content."""

    def test_prompt_includes_script_title(self, director, sample_script, sample_timeline):
        """Prompt includes the script title."""
        prompt = director._build_review_prompt(sample_script, sample_timeline)
        assert "The History of Coffee" in prompt

    def test_prompt_includes_hook_voiceover(self, director, sample_script, sample_timeline):
        """Prompt includes the hook voiceover text."""
        prompt = director._build_review_prompt(sample_script, sample_timeline)
        assert "coffee was first discovered by goats" in prompt

    def test_prompt_includes_scene_details(self, director, sample_script, sample_timeline):
        """Prompt includes scene voiceover, keywords, and style."""
        prompt = director._build_review_prompt(sample_script, sample_timeline)
        assert "Scene 1" in prompt
        assert "Scene 2" in prompt
        assert "documentary" in prompt
        assert "cinematic" in prompt

    def test_prompt_includes_timeline_metadata(self, director, sample_script, sample_timeline):
        """Prompt includes timeline duration, scene count, color grade."""
        prompt = director._build_review_prompt(sample_script, sample_timeline)
        assert "23.0s" in prompt
        assert "3" in prompt  # num_scenes
        assert "dark_cinematic" in prompt

    def test_prompt_includes_approval_threshold(self, director, sample_script, sample_timeline):
        """Prompt includes the configured approval threshold."""
        prompt = director._build_review_prompt(sample_script, sample_timeline)
        assert str(director.approval_threshold) in prompt


# ---------------------------------------------------------------------------
# review_draft tests
# ---------------------------------------------------------------------------


class TestReviewDraft:
    """Tests for the full review_draft() method."""

    def test_missing_video_file_returns_passthrough(self, director, sample_script, sample_timeline):
        """Missing draft video file returns pass-through review."""
        review = director.review_draft(
            "/nonexistent/draft.mp4",
            sample_script,
            sample_timeline,
            iteration=0,
        )

        assert review.approved is True
        assert review.overall_score == 7
        assert "not found" in review.notes

    def test_no_api_key_returns_passthrough(self, sample_script, sample_timeline, temp_dir):
        """No API key configured returns pass-through review."""
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            agent = DirectorAgent(api_key="")

        # Ensure client is None (no API key)
        assert agent.client is None

        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        review = agent.review_draft(str(draft), sample_script, sample_timeline, iteration=0)

        assert review.approved is True
        assert "No API key" in review.notes

    def test_successful_review_approved(self, director, sample_script, sample_timeline, temp_dir):
        """Successful Gemini review with score >= 7 is approved."""
        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        mock_file = MagicMock()
        mock_file.name = "test_file"
        director.client.files.upload.return_value = mock_file

        # Mock file processing state
        file_info = MagicMock()
        file_info.state = "ACTIVE"
        director.client.files.get.return_value = file_info

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "overall_score": 8,
            "approved": True,
            "notes": "Great quality.",
            "fixes": [],
        })
        director.client.models.generate_content.return_value = mock_response

        review = director.review_draft(str(draft), sample_script, sample_timeline, iteration=0)

        assert review.approved is True
        assert review.overall_score == 8
        assert review.fix_requests == []
        director.client.files.upload.assert_called_once()

    def test_successful_review_rejected(self, director, sample_script, sample_timeline, temp_dir):
        """Successful Gemini review with score < 7 is rejected."""
        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        mock_file = MagicMock()
        mock_file.name = "test_file"
        director.client.files.upload.return_value = mock_file

        file_info = MagicMock()
        file_info.state = "ACTIVE"
        director.client.files.get.return_value = file_info

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "overall_score": 4,
            "approved": False,
            "notes": "Needs work.",
            "fixes": [
                {
                    "scene_id": 1,
                    "issue_type": "visual_mismatch",
                    "description": "Wrong visual",
                    "suggested_keywords": ["correct", "keywords"],
                    "priority": "high",
                }
            ],
        })
        director.client.models.generate_content.return_value = mock_response

        review = director.review_draft(str(draft), sample_script, sample_timeline, iteration=0)

        assert review.approved is False
        assert review.overall_score == 4
        assert len(review.fix_requests) == 1

    def test_empty_response_returns_passthrough(
        self, director, sample_script, sample_timeline, temp_dir
    ):
        """Empty Gemini response returns pass-through review."""
        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        mock_file = MagicMock()
        mock_file.name = "test_file"
        director.client.files.upload.return_value = mock_file

        file_info = MagicMock()
        file_info.state = "ACTIVE"
        director.client.files.get.return_value = file_info

        mock_response = MagicMock()
        mock_response.text = ""  # Empty response
        director.client.models.generate_content.return_value = mock_response

        review = director.review_draft(str(draft), sample_script, sample_timeline, iteration=0)

        assert review.approved is True
        assert "Empty AI response" in review.notes

    def test_api_exception_returns_passthrough(
        self, director, sample_script, sample_timeline, temp_dir
    ):
        """General API exception returns pass-through review (graceful degradation)."""
        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        mock_file = MagicMock()
        mock_file.name = "test_file"
        director.client.files.upload.return_value = mock_file

        file_info = MagicMock()
        file_info.state = "ACTIVE"
        director.client.files.get.return_value = file_info

        director.client.models.generate_content.side_effect = ValueError("Unexpected API error")

        review = director.review_draft(str(draft), sample_script, sample_timeline, iteration=0)

        assert review.approved is True
        assert "ValueError" in review.notes

    def test_api_rate_limit_is_reraised(
        self, director, sample_script, sample_timeline, temp_dir
    ):
        """APIRateLimitError is re-raised (for retry decorator)."""
        from utils.retry import APIRateLimitError

        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        mock_file = MagicMock()
        mock_file.name = "test_file"
        director.client.files.upload.return_value = mock_file

        file_info = MagicMock()
        file_info.state = "ACTIVE"
        director.client.files.get.return_value = file_info

        director.client.models.generate_content.side_effect = APIRateLimitError("Rate limited")

        with pytest.raises(APIRateLimitError):
            director.review_draft(str(draft), sample_script, sample_timeline, iteration=0)

    def test_network_error_is_reraised(
        self, director, sample_script, sample_timeline, temp_dir
    ):
        """NetworkError is re-raised (for retry decorator)."""
        from utils.retry import NetworkError

        draft = temp_dir / "draft.mp4"
        draft.write_bytes(b"\x00" * 1024)

        mock_file = MagicMock()
        mock_file.name = "test_file"
        director.client.files.upload.return_value = mock_file

        file_info = MagicMock()
        file_info.state = "ACTIVE"
        director.client.files.get.return_value = file_info

        director.client.models.generate_content.side_effect = NetworkError("Network down")

        with pytest.raises(NetworkError):
            director.review_draft(str(draft), sample_script, sample_timeline, iteration=0)


# ---------------------------------------------------------------------------
# _passthrough_review tests
# ---------------------------------------------------------------------------


class TestPassthroughReview:
    """Tests for _passthrough_review fallback."""

    def test_passthrough_is_approved(self, director):
        """Pass-through review is always approved."""
        review = director._passthrough_review(iteration=0, notes="test")
        assert review.approved is True
        assert review.overall_score == 7
        assert review.fix_requests == []

    def test_passthrough_preserves_iteration(self, director):
        """Pass-through preserves the iteration number."""
        review = director._passthrough_review(iteration=3, notes="test")
        assert review.iteration == 3

    def test_passthrough_default_notes(self, director):
        """Pass-through has default notes when none provided."""
        review = director._passthrough_review(iteration=0)
        assert "API unavailable" in review.notes


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestDirectorInit:
    """Tests for DirectorAgent initialization."""

    def test_default_approval_threshold(self):
        """Default approval threshold is 7."""
        with patch("video_agent.director.Client"):
            agent = DirectorAgent(api_key="test")
        assert agent.approval_threshold == DEFAULT_APPROVAL_THRESHOLD

    def test_custom_approval_threshold(self):
        """Custom approval threshold is respected."""
        with patch("video_agent.director.Client"):
            agent = DirectorAgent(api_key="test", approval_threshold=5)
        assert agent.approval_threshold == 5

    def test_no_client_without_key(self):
        """No Gemini client created when API key is empty."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            agent = DirectorAgent(api_key="")
        assert agent.client is None

    def test_custom_model(self):
        """Custom model name is stored."""
        with patch("video_agent.director.Client"):
            agent = DirectorAgent(api_key="test", model="gemini-2.5-pro")
        assert agent.model == "gemini-2.5-pro"
