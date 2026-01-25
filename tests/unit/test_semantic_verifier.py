"""Unit tests for SemanticVerifier service.

Tests the semantic verification functionality that validates clips match
the original transcript context and contain required visual elements.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.models.broll_need import BRollNeed
from src.models.clip import ClipSegment
from src.services.semantic_verifier import SemanticVerifier, VerificationResult


# --- Fixtures ---


@pytest.fixture
def mock_genai():
    """Mock the google.genai module."""
    with patch("src.services.semantic_verifier.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock files API
        mock_file = MagicMock()
        mock_file.name = "test_file_123"
        mock_client.files.upload.return_value = mock_file

        # Mock file state for processing completion
        mock_file_info = MagicMock()
        mock_file_info.state.name = "ACTIVE"
        mock_client.files.get.return_value = mock_file_info

        # Mock models API with default response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "similarity_score": 0.95,
            "matched_elements": ["people", "laptops", "coffee shop"],
            "missing_elements": [],
            "rationale": "The clip shows a busy coffee shop with remote workers on laptops."
        })
        mock_client.models.generate_content.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_config():
    """Mock the config loader with test values."""
    with patch("src.services.semantic_verifier.load_config") as mock_load:
        mock_load.return_value = {
            "gemini_api_key": "test_api_key",
            "semantic_match_threshold": 0.9,
            "semantic_verification_enabled": True,
            "reject_below_threshold": True,
            "min_required_elements_match": 0.8,
        }
        yield mock_load


@pytest.fixture
def sample_broll_need():
    """Create a sample BRollNeed for testing."""
    return BRollNeed(
        timestamp=30.0,
        search_phrase="coffee shop interior",
        description="Busy cafe scene",
        context="Talking about remote work culture",
        original_context="The coffee shop was packed with remote workers on laptops",
        required_elements=["people", "laptops", "coffee shop", "busy"],
    )


@pytest.fixture
def sample_broll_need_no_context():
    """Create a BRollNeed without original_context for fallback testing."""
    return BRollNeed(
        timestamp=30.0,
        search_phrase="coffee shop interior",
        description="Busy cafe scene as fallback description",
        context="Talking about remote work culture",
        original_context="",  # Empty - should fall back to description
        required_elements=["people", "laptops"],
    )


@pytest.fixture
def sample_clip_segment():
    """Create a sample ClipSegment for testing."""
    return ClipSegment(
        start_time=5.0,
        end_time=12.0,
        relevance_score=8,
        description="Coffee shop scene with customers working",
    )


@pytest.fixture
def temp_clip_file(tmp_path):
    """Create a temporary clip file for testing."""
    clip_file = tmp_path / "test_clip.mp4"
    clip_file.write_bytes(b"fake video data")
    return clip_file


# --- VerificationResult Dataclass Tests ---


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_verification_result_dataclass(self):
        """Test VerificationResult can be created with all fields."""
        result = VerificationResult(
            passed=True,
            similarity_score=0.95,
            matched_elements=["people", "laptops", "coffee shop"],
            missing_elements=["busy"],
            rationale="The clip shows a coffee shop with workers.",
        )

        assert result.passed is True
        assert result.similarity_score == 0.95
        assert result.matched_elements == ["people", "laptops", "coffee shop"]
        assert result.missing_elements == ["busy"]
        assert result.rationale == "The clip shows a coffee shop with workers."

    def test_elements_match_ratio_all_matched(self):
        """Test elements_match_ratio when all elements are matched."""
        result = VerificationResult(
            passed=True,
            similarity_score=1.0,
            matched_elements=["a", "b", "c"],
            missing_elements=[],
            rationale="All elements present",
        )
        assert result.elements_match_ratio == 1.0

    def test_elements_match_ratio_partial_match(self):
        """Test elements_match_ratio with partial matches."""
        result = VerificationResult(
            passed=False,
            similarity_score=0.6,
            matched_elements=["a", "b"],
            missing_elements=["c", "d"],
            rationale="Half elements present",
        )
        assert result.elements_match_ratio == 0.5

    def test_elements_match_ratio_no_elements(self):
        """Test elements_match_ratio when no elements required."""
        result = VerificationResult(
            passed=True,
            similarity_score=0.9,
            matched_elements=[],
            missing_elements=[],
            rationale="No elements to check",
        )
        # No elements means 100% match by default
        assert result.elements_match_ratio == 1.0

    def test_to_dict_method(self):
        """Test to_dict serialization method."""
        result = VerificationResult(
            passed=True,
            similarity_score=0.85,
            matched_elements=["a", "b"],
            missing_elements=["c"],
            rationale="Test rationale",
        )
        result_dict = result.to_dict()

        assert result_dict["passed"] is True
        assert result_dict["similarity_score"] == 0.85
        assert result_dict["elements_match_ratio"] == pytest.approx(0.666, rel=0.01)
        assert result_dict["matched_elements"] == ["a", "b"]
        assert result_dict["missing_elements"] == ["c"]
        assert result_dict["rationale"] == "Test rationale"


# --- SemanticVerifier Tests ---


class TestSemanticVerifierVerification:
    """Tests for SemanticVerifier verification logic."""

    @pytest.mark.asyncio
    async def test_verification_passes_matching_clip(
        self, mock_genai, mock_config, sample_broll_need, temp_clip_file
    ):
        """Test verification passes when Gemini returns high similarity score."""
        # Configure mock to return high similarity
        mock_genai.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.95,
            "matched_elements": ["people", "laptops", "coffee shop", "busy"],
            "missing_elements": [],
            "rationale": "Perfect match - all elements visible in busy coffee shop scene."
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

        assert result.passed is True
        assert result.similarity_score == 0.95
        assert len(result.matched_elements) == 4
        assert len(result.missing_elements) == 0

    @pytest.mark.asyncio
    async def test_verification_fails_missing_elements(
        self, mock_genai, mock_config, sample_broll_need, temp_clip_file
    ):
        """Test verification fails when Gemini returns low similarity with missing elements."""
        # Configure mock to return low similarity with missing elements
        mock_genai.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.5,
            "matched_elements": ["coffee shop"],
            "missing_elements": ["people", "laptops", "busy"],
            "rationale": "Clip shows empty coffee shop, missing key elements."
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

        assert result.passed is False
        assert result.similarity_score == 0.5
        assert "people" in result.missing_elements
        assert "laptops" in result.missing_elements

    @pytest.mark.asyncio
    async def test_verification_uses_original_context(
        self, mock_genai, mock_config, sample_broll_need, temp_clip_file
    ):
        """Verify the prompt sent to Gemini includes original_context, not just search_phrase."""
        verifier = SemanticVerifier(api_key="test_key")
        await verifier.verify_clip(temp_clip_file, sample_broll_need)

        # Check the prompt that was sent to Gemini
        call_args = mock_genai.models.generate_content.call_args
        prompt_content = str(call_args)

        # The prompt should contain the original_context
        assert "coffee shop was packed with remote workers on laptops" in prompt_content
        # The prompt should also contain required_elements
        assert "people" in prompt_content
        assert "laptops" in prompt_content

    @pytest.mark.asyncio
    async def test_verification_falls_back_to_description(
        self, mock_genai, mock_config, sample_broll_need_no_context, temp_clip_file
    ):
        """Test that verification falls back to description when original_context is empty."""
        verifier = SemanticVerifier(api_key="test_key")
        await verifier.verify_clip(temp_clip_file, sample_broll_need_no_context)

        # Check the prompt that was sent to Gemini
        call_args = mock_genai.models.generate_content.call_args
        prompt_content = str(call_args)

        # Should use description as fallback
        assert "Busy cafe scene as fallback description" in prompt_content


class TestSemanticVerifierThresholds:
    """Tests for threshold configuration."""

    @pytest.mark.asyncio
    async def test_verification_threshold_configurable(
        self, mock_genai, temp_clip_file, sample_broll_need
    ):
        """Test that different thresholds work correctly."""
        # Test with low threshold (0.5) - should pass a 0.6 score
        with patch("src.services.semantic_verifier.load_config") as mock_load:
            mock_load.return_value = {
                "gemini_api_key": "test_key",
                "semantic_match_threshold": 0.5,  # Low threshold
                "semantic_verification_enabled": True,
                "reject_below_threshold": True,
                "min_required_elements_match": 0.5,
            }

            mock_genai.models.generate_content.return_value.text = json.dumps({
                "similarity_score": 0.6,
                "matched_elements": ["people", "coffee shop"],
                "missing_elements": ["laptops", "busy"],
                "rationale": "Partial match"
            })

            verifier = SemanticVerifier(api_key="test_key")
            result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

            # Score 0.6 should pass threshold 0.5
            assert result.passed is True
            assert result.similarity_score == 0.6

    @pytest.mark.asyncio
    async def test_verification_fails_high_threshold(
        self, mock_genai, temp_clip_file, sample_broll_need
    ):
        """Test that high threshold rejects moderate scores."""
        with patch("src.services.semantic_verifier.load_config") as mock_load:
            mock_load.return_value = {
                "gemini_api_key": "test_key",
                "semantic_match_threshold": 0.95,  # Very high threshold
                "semantic_verification_enabled": True,
                "reject_below_threshold": True,
                "min_required_elements_match": 0.8,
            }

            mock_genai.models.generate_content.return_value.text = json.dumps({
                "similarity_score": 0.85,
                "matched_elements": ["people", "laptops", "coffee shop"],
                "missing_elements": ["busy"],
                "rationale": "Good but not perfect match"
            })

            verifier = SemanticVerifier(api_key="test_key")
            result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

            # Score 0.85 should fail threshold 0.95
            assert result.passed is False
            assert result.similarity_score == 0.85


class TestSemanticVerifierBatchFiltering:
    """Tests for batch verification and filtering."""

    @pytest.mark.asyncio
    async def test_verify_and_filter_clips_returns_passing(
        self, mock_genai, mock_config, sample_broll_need, tmp_path
    ):
        """Test batch verification returns only passing clips."""
        # Create multiple clip files
        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"
        clip3 = tmp_path / "clip3.mp4"
        for clip in [clip1, clip2, clip3]:
            clip.write_bytes(b"fake video data")

        # Create clip segments
        segment1 = ClipSegment(0.0, 5.0, 9, "Great match")
        segment2 = ClipSegment(5.0, 10.0, 5, "Poor match")
        segment3 = ClipSegment(10.0, 15.0, 8, "Good match")

        clips = [
            (clip1, segment1),
            (clip2, segment2),
            (clip3, segment3),
        ]

        # Mock different responses for each verification
        # Note: Threshold is 0.9 for similarity, 0.8 for elements ratio
        responses = [
            # Clip 1 - passes (high score, all 4 elements matched = 100%)
            json.dumps({
                "similarity_score": 0.95,
                "matched_elements": ["people", "laptops", "coffee shop", "busy"],
                "missing_elements": [],
                "rationale": "Perfect match"
            }),
            # Clip 2 - fails (low score, only 1/4 elements = 25%)
            json.dumps({
                "similarity_score": 0.4,
                "matched_elements": ["coffee shop"],
                "missing_elements": ["people", "laptops", "busy"],
                "rationale": "Poor match"
            }),
            # Clip 3 - passes (above threshold, 4/5 elements = 80%)
            json.dumps({
                "similarity_score": 0.92,
                "matched_elements": ["people", "laptops", "coffee shop", "busy"],
                "missing_elements": ["extra"],  # One extra element missing is OK
                "rationale": "Good match"
            }),
        ]

        # Set up mock to return different responses sequentially
        mock_genai.models.generate_content.side_effect = [
            MagicMock(text=resp) for resp in responses
        ]

        verifier = SemanticVerifier(api_key="test_key")
        results = await verifier.verify_and_filter_clips(clips, sample_broll_need)

        # Should return only the 2 passing clips (clip1 and clip3)
        assert len(results) == 2
        # All returned should have passed=True
        for _path, _segment, verification in results:
            assert verification.passed is True
        # Verify scores are as expected (sorted by score descending)
        assert results[0][2].similarity_score == 0.95
        assert results[1][2].similarity_score == 0.92

    @pytest.mark.asyncio
    async def test_verify_and_filter_clips_returns_best_when_none_pass(
        self, mock_genai, mock_config, sample_broll_need, tmp_path
    ):
        """Test that when no clips pass, returns the best scoring one with warning."""
        # Create clip files
        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"
        for clip in [clip1, clip2]:
            clip.write_bytes(b"fake video data")

        segment1 = ClipSegment(0.0, 5.0, 5, "Below threshold")
        segment2 = ClipSegment(5.0, 10.0, 4, "Also below threshold")

        clips = [(clip1, segment1), (clip2, segment2)]

        # Both clips fail threshold (all scores below 0.9)
        responses = [
            json.dumps({
                "similarity_score": 0.6,  # Best score but still fails
                "matched_elements": ["people"],
                "missing_elements": ["laptops", "coffee shop", "busy"],
                "rationale": "Partial match"
            }),
            json.dumps({
                "similarity_score": 0.3,
                "matched_elements": [],
                "missing_elements": ["people", "laptops", "coffee shop", "busy"],
                "rationale": "Poor match"
            }),
        ]
        mock_genai.models.generate_content.side_effect = [
            MagicMock(text=resp) for resp in responses
        ]

        verifier = SemanticVerifier(api_key="test_key")
        results = await verifier.verify_and_filter_clips(clips, sample_broll_need, min_clips=1)

        # Should return the best scoring clip even though it fails
        assert len(results) == 1
        _path, _segment, verification = results[0]

        # Should be marked as passed (with warning in rationale)
        assert verification.passed is True
        assert "[WARNING: Below threshold]" in verification.rationale
        # Should be the higher-scoring clip
        assert verification.similarity_score == 0.6


class TestSemanticVerifierDisabled:
    """Tests for disabled verification mode."""

    @pytest.mark.asyncio
    async def test_disabled_verification_single_clip(
        self, mock_genai, sample_broll_need, temp_clip_file
    ):
        """Test that disabled verifier auto-passes clips."""
        with patch("src.services.semantic_verifier.load_config") as mock_load:
            mock_load.return_value = {
                "gemini_api_key": "test_key",
                "semantic_match_threshold": 0.9,
                "semantic_verification_enabled": False,  # DISABLED
                "reject_below_threshold": True,
                "min_required_elements_match": 0.8,
            }

            verifier = SemanticVerifier(api_key="test_key")
            result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

            # Should auto-pass with score 1.0
            assert result.passed is True
            assert result.similarity_score == 1.0
            assert "disabled" in result.rationale.lower()
            # Should NOT have called Gemini API
            mock_genai.models.generate_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_verification_batch(
        self, mock_genai, sample_broll_need, tmp_path
    ):
        """Test that disabled verifier returns all clips as passed in batch mode."""
        with patch("src.services.semantic_verifier.load_config") as mock_load:
            mock_load.return_value = {
                "gemini_api_key": "test_key",
                "semantic_match_threshold": 0.9,
                "semantic_verification_enabled": False,  # DISABLED
                "reject_below_threshold": True,
                "min_required_elements_match": 0.8,
            }

            # Create multiple clips
            clip1 = tmp_path / "clip1.mp4"
            clip2 = tmp_path / "clip2.mp4"
            for clip in [clip1, clip2]:
                clip.write_bytes(b"fake video data")

            segment1 = ClipSegment(0.0, 5.0, 8, "Test clip 1")
            segment2 = ClipSegment(5.0, 10.0, 7, "Test clip 2")

            clips = [(clip1, segment1), (clip2, segment2)]

            verifier = SemanticVerifier(api_key="test_key")
            results = await verifier.verify_and_filter_clips(clips, sample_broll_need)

            # All clips should be returned and passed
            assert len(results) == 2
            for _path, _segment, verification in results:
                assert verification.passed is True
                assert verification.similarity_score == 1.0
                assert "disabled" in verification.rationale.lower()

            # Should NOT have called Gemini API
            mock_genai.models.generate_content.assert_not_called()


class TestSemanticVerifierEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_missing_clip_file(self, mock_genai, mock_config, sample_broll_need):
        """Test verification handles missing clip file."""
        verifier = SemanticVerifier(api_key="test_key")
        nonexistent_path = Path("/nonexistent/clip.mp4")

        result = await verifier.verify_clip(nonexistent_path, sample_broll_need)

        assert result.passed is False
        assert result.similarity_score == 0.0
        assert "not found" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_empty_clips_list(self, mock_genai, mock_config, sample_broll_need):
        """Test verify_and_filter_clips handles empty list."""
        verifier = SemanticVerifier(api_key="test_key")
        results = await verifier.verify_and_filter_clips([], sample_broll_need)

        assert results == []

    @pytest.mark.asyncio
    async def test_malformed_json_response(
        self, mock_genai, mock_config, sample_broll_need, temp_clip_file
    ):
        """Test verification handles malformed JSON from Gemini."""
        mock_genai.models.generate_content.return_value.text = "not valid json"

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

        # Should fail gracefully
        assert result.passed is False
        assert result.similarity_score == 0.0
        assert "parse" in result.rationale.lower() or "failed" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_json_with_markdown_wrapper(
        self, mock_genai, mock_config, sample_broll_need, temp_clip_file
    ):
        """Test verification handles JSON wrapped in markdown code blocks."""
        mock_genai.models.generate_content.return_value.text = """```json
{
    "similarity_score": 0.92,
    "matched_elements": ["people", "laptops"],
    "missing_elements": ["busy"],
    "rationale": "Good match with markdown wrapper"
}
```"""

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

        # Should parse correctly despite markdown wrapper
        assert result.similarity_score == 0.92
        assert "people" in result.matched_elements

    @pytest.mark.asyncio
    async def test_score_clamping(
        self, mock_genai, mock_config, sample_broll_need, temp_clip_file
    ):
        """Test that scores outside 0-1 range are clamped."""
        mock_genai.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 1.5,  # Invalid: above 1.0
            "matched_elements": ["people"],
            "missing_elements": [],
            "rationale": "Overly enthusiastic AI"
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need)

        # Score should be clamped to 1.0
        assert result.similarity_score == 1.0


class TestSemanticVerifierInit:
    """Tests for SemanticVerifier initialization."""

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        with patch("src.services.semantic_verifier.load_config") as mock_load:
            mock_load.return_value = {
                "gemini_api_key": None,  # No API key
                "semantic_verification_enabled": True,
            }

            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                SemanticVerifier()

    def test_init_with_explicit_api_key(self, mock_genai, mock_config):
        """Test initialization with explicit API key."""
        verifier = SemanticVerifier(api_key="explicit_key")
        assert verifier is not None
        assert verifier.enabled is True

    def test_init_loads_thresholds_from_config(self, mock_genai):
        """Test that thresholds are loaded from config."""
        with patch("src.services.semantic_verifier.load_config") as mock_load:
            mock_load.return_value = {
                "gemini_api_key": "test_key",
                "semantic_match_threshold": 0.75,
                "semantic_verification_enabled": True,
                "reject_below_threshold": False,
                "min_required_elements_match": 0.6,
            }

            verifier = SemanticVerifier(api_key="test_key")

            assert verifier.threshold == 0.75
            assert verifier.enabled is True
            assert verifier.reject_below_threshold is False
            assert verifier.min_elements_match == 0.6
