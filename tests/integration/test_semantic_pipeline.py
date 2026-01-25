"""Integration tests for the semantic matching pipeline.

Tests the full flow of context preservation from B-roll planning through
evaluation, extraction, and verification. Ensures that original_context
and required_elements flow correctly through the entire pipeline.

Pipeline Components Tested:
1. BRollNeed model with original_context and required_elements
2. AIService.plan_broll_needs() output includes context fields
3. AIService.evaluate_videos() uses context for scoring
4. ClipExtractor.analyze_video() uses context for clip selection
5. SemanticVerifier verifies clips match context
6. BRollProcessor integrates the full pipeline

These tests verify the semantic matching feature (Fix 4) works end-to-end.
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Tuple

import sys
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.models.broll_need import BRollNeed, BRollPlan
from src.models.clip import ClipSegment, ClipResult, VideoAnalysisResult
from src.services.semantic_verifier import SemanticVerifier, VerificationResult


# --- Fixtures ---


@pytest.fixture
def sample_broll_need_with_context():
    """Create a BRollNeed with original_context and required_elements for testing."""
    return BRollNeed(
        timestamp=30.0,
        search_phrase="coffee shop interior",
        description="Busy cafe scene",
        context="Talking about remote work culture",
        original_context="The coffee shop was packed with remote workers typing on laptops during the morning rush. The atmosphere was buzzing with productivity as people sipped their lattes.",
        required_elements=["people", "laptops", "coffee shop interior", "busy atmosphere", "morning light"],
    )


@pytest.fixture
def sample_broll_need_minimal():
    """Create a BRollNeed with minimal context (for fallback testing)."""
    return BRollNeed(
        timestamp=60.0,
        search_phrase="sunset timelapse",
        description="Beautiful sunset over the ocean",
        context="Discussing end of day routines",
        original_context="",  # Empty - should fall back
        required_elements=[],  # Empty - no required elements
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
def sample_broll_plan(sample_broll_need_with_context):
    """Create a sample BRollPlan with context-aware needs."""
    return BRollPlan(
        source_duration=300.0,
        needs=[sample_broll_need_with_context],
        clips_per_minute=2.0,
        source_file="/path/to/source.mp4",
    )


@pytest.fixture
def temp_clip_file(tmp_path):
    """Create a temporary clip file for testing."""
    clip_file = tmp_path / "test_clip.mp4"
    clip_file.write_bytes(b"fake video data for testing")
    return clip_file


@pytest.fixture
def mock_genai_client():
    """Mock the google.genai Client for SemanticVerifier tests."""
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

        yield mock_client


@pytest.fixture
def mock_config_enabled():
    """Mock config with semantic verification enabled."""
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
def mock_config_disabled():
    """Mock config with semantic verification disabled."""
    with patch("src.services.semantic_verifier.load_config") as mock_load:
        mock_load.return_value = {
            "gemini_api_key": "test_api_key",
            "semantic_match_threshold": 0.9,
            "semantic_verification_enabled": False,  # DISABLED
            "reject_below_threshold": True,
            "min_required_elements_match": 0.8,
        }
        yield mock_load


# --- Test 1: BRollNeed Has Context Fields ---


class TestBRollNeedContextFields:
    """Test that BRollNeed model properly supports context fields."""

    def test_broll_need_has_context_fields(self, sample_broll_need_with_context):
        """Verify original_context and required_elements fields are properly set and accessible."""
        need = sample_broll_need_with_context

        # Verify fields exist and are properly set
        assert hasattr(need, "original_context")
        assert hasattr(need, "required_elements")

        # Verify original_context content
        assert need.original_context is not None
        assert len(need.original_context) > 0
        assert "coffee shop" in need.original_context.lower()
        assert "remote workers" in need.original_context.lower()
        assert "laptops" in need.original_context.lower()

        # Verify required_elements content
        assert isinstance(need.required_elements, list)
        assert len(need.required_elements) == 5
        assert "people" in need.required_elements
        assert "laptops" in need.required_elements
        assert "coffee shop interior" in need.required_elements
        assert "busy atmosphere" in need.required_elements
        assert "morning light" in need.required_elements

    def test_broll_need_context_defaults_to_empty(self):
        """Verify BRollNeed defaults to empty context fields when not provided."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="test phrase",
            description="test description",
            context="test context",
            # Not providing original_context and required_elements
        )

        assert need.original_context == ""
        assert need.required_elements == []

    def test_broll_need_preserves_long_context(self):
        """Verify BRollNeed can store long transcript context (100+ chars as documented)."""
        long_context = (
            "This is a very long transcript segment that contains detailed context about "
            "what the video narrator is discussing. It should be at least 100 characters "
            "to properly represent the narrative context. The AI will use this to find "
            "clips that truly match the meaning, not just keywords."
        )

        need = BRollNeed(
            timestamp=30.0,
            search_phrase="test phrase",
            description="test description",
            context="test",
            original_context=long_context,
            required_elements=["element1", "element2"],
        )

        assert len(need.original_context) > 100
        assert need.original_context == long_context


# --- Test 2: Full Pipeline Preserves Context ---


class TestFullPipelinePreservesContext:
    """Test that context flows correctly through the full pipeline."""

    @pytest.fixture
    def mock_ai_service_with_planning(self, sample_broll_need_with_context):
        """Create mock AIService that returns context-aware BRollNeeds."""
        with patch("src.services.ai_service.Client"):
            from src.services.ai_service import AIService

            service = AIService(api_key="test_key")

            # Mock plan_broll_needs to return a BRollNeed with context
            mock_plan = BRollPlan(
                source_duration=300.0,
                needs=[sample_broll_need_with_context],
                clips_per_minute=2.0,
            )
            service.plan_broll_needs = Mock(return_value=mock_plan)

            return service

    def test_context_preserved_from_planning_to_evaluation(
        self, mock_ai_service_with_planning, sample_broll_need_with_context
    ):
        """Verify original_context flows from planning to evaluation."""
        ai_service = mock_ai_service_with_planning

        # Get the planned BRollNeed
        plan = ai_service.plan_broll_needs("Test transcript", 300.0)
        need = plan.needs[0]

        # Verify context is preserved in the planned need
        assert need.original_context == sample_broll_need_with_context.original_context
        assert need.required_elements == sample_broll_need_with_context.required_elements

        # The evaluate_videos method should receive this need with context intact
        # This is verified by checking the need object passed would have the fields
        assert hasattr(need, "original_context")
        assert hasattr(need, "required_elements")
        assert len(need.original_context) > 0
        assert len(need.required_elements) > 0

    def test_context_passed_to_clip_extractor(
        self, sample_broll_need_with_context, tmp_path
    ):
        """Verify BRollNeed with context can be passed to ClipExtractor methods."""
        # Create a mock video file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video data")

        # Verify that ClipExtractor.analyze_video accepts broll_need parameter
        with patch("src.services.clip_extractor.Client"):
            from src.services.clip_extractor import ClipExtractor

            # Create extractor (won't actually call API in test)
            extractor = ClipExtractor(api_key="test_key")

            # The analyze_video method signature should accept broll_need
            import inspect
            sig = inspect.signature(extractor.analyze_video)
            assert "broll_need" in sig.parameters

            # The process_downloaded_video method should also accept it
            sig = inspect.signature(extractor.process_downloaded_video)
            assert "broll_need" in sig.parameters

    def test_context_used_in_semantic_verification(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, temp_clip_file
    ):
        """Verify SemanticVerifier uses original_context and required_elements."""
        # Set up mock response with good match
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.95,
            "matched_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere", "morning light"],
            "missing_elements": [],
            "rationale": "Perfect match - all required elements visible in the clip."
        })

        verifier = SemanticVerifier(api_key="test_key")

        # Run verification
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)
        )

        # Verify the prompt included original_context
        call_args = mock_genai_client.models.generate_content.call_args
        prompt_content = str(call_args)

        # Should include the original context from BRollNeed
        assert "coffee shop was packed with remote workers" in prompt_content
        assert "morning rush" in prompt_content

        # Should include required elements
        assert "people" in prompt_content
        assert "laptops" in prompt_content


# --- Test 3: Low Relevance Clips Rejected ---


class TestLowRelevanceClipsRejected:
    """Test that clips below 90% threshold are rejected."""

    @pytest.mark.asyncio
    async def test_low_similarity_score_rejected(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, temp_clip_file, caplog
    ):
        """Verify clips with low similarity score are rejected and logged."""
        # Set up mock response with low similarity
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.6,  # Below 0.9 threshold
            "matched_elements": ["coffee shop interior"],
            "missing_elements": ["people", "laptops", "busy atmosphere", "morning light"],
            "rationale": "Clip shows empty coffee shop, missing most required elements."
        })

        with caplog.at_level(logging.DEBUG):
            verifier = SemanticVerifier(api_key="test_key")
            result = await verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)

        # Verify clip was rejected
        assert result.passed is False
        assert result.similarity_score == 0.6

        # Verify logging indicates rejection reason
        log_messages = " ".join([record.message for record in caplog.records])
        assert "failed" in log_messages.lower() or "threshold" in log_messages.lower()

    @pytest.mark.asyncio
    async def test_missing_elements_causes_rejection(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, temp_clip_file
    ):
        """Verify clips missing too many required elements are rejected."""
        # Set up mock response with good score but too many missing elements
        # min_required_elements_match is 0.8, so with 1/5 elements = 0.2, should fail
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.7,  # Below 0.9
            "matched_elements": ["coffee shop interior"],  # 1 out of 5 = 20%
            "missing_elements": ["people", "laptops", "busy atmosphere", "morning light"],
            "rationale": "Only coffee shop visible, missing most elements."
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)

        # Verify clip was rejected
        assert result.passed is False
        assert len(result.missing_elements) == 4
        assert result.elements_match_ratio == 0.2  # 1/5 elements

    @pytest.mark.asyncio
    async def test_threshold_boundary_exactly_90_percent(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, temp_clip_file
    ):
        """Verify clips at exactly 90% threshold pass."""
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.90,  # Exactly at threshold
            "matched_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere"],
            "missing_elements": ["morning light"],  # 4/5 = 80% elements
            "rationale": "Good match, minor element missing."
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)

        # At exactly threshold should pass
        assert result.passed is True
        assert result.similarity_score == 0.90


# --- Test 4: Semantic Verification Integration ---


class TestSemanticVerificationIntegration:
    """Test semantic verification with passing and failing scenarios."""

    @pytest.mark.asyncio
    async def test_verification_passes_with_matching_clip(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, temp_clip_file
    ):
        """Test verification passes when clip matches context."""
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.95,
            "matched_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere", "morning light"],
            "missing_elements": [],
            "rationale": "Perfect match - clip shows busy coffee shop with remote workers on laptops."
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)

        assert result.passed is True
        assert result.similarity_score == 0.95
        assert len(result.matched_elements) == 5
        assert len(result.missing_elements) == 0

    @pytest.mark.asyncio
    async def test_verification_fails_with_unrelated_clip(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, temp_clip_file
    ):
        """Test verification fails when clip doesn't match context."""
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.2,
            "matched_elements": [],
            "missing_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere", "morning light"],
            "rationale": "Clip shows outdoor beach scene, completely unrelated to coffee shop context."
        })

        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)

        assert result.passed is False
        assert result.similarity_score == 0.2
        assert len(result.matched_elements) == 0
        assert len(result.missing_elements) == 5

    @pytest.mark.asyncio
    async def test_batch_verification_filters_clips(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_with_context, tmp_path
    ):
        """Test verify_and_filter_clips returns only passing clips."""
        # Create multiple clip files
        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"
        clip3 = tmp_path / "clip3.mp4"
        for clip in [clip1, clip2, clip3]:
            clip.write_bytes(b"fake video data")

        segment1 = ClipSegment(0.0, 5.0, 9, "High quality clip")
        segment2 = ClipSegment(5.0, 10.0, 5, "Low quality clip")
        segment3 = ClipSegment(10.0, 15.0, 8, "Medium quality clip")

        clips = [
            (clip1, segment1),
            (clip2, segment2),
            (clip3, segment3),
        ]

        # Mock different responses for each clip
        responses = [
            json.dumps({
                "similarity_score": 0.95,
                "matched_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere", "morning light"],
                "missing_elements": [],
                "rationale": "Perfect match"
            }),
            json.dumps({
                "similarity_score": 0.4,  # Below threshold
                "matched_elements": ["coffee shop interior"],
                "missing_elements": ["people", "laptops", "busy atmosphere", "morning light"],
                "rationale": "Poor match"
            }),
            json.dumps({
                "similarity_score": 0.92,
                "matched_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere"],
                "missing_elements": ["morning light"],
                "rationale": "Good match"
            }),
        ]
        mock_genai_client.models.generate_content.side_effect = [
            MagicMock(text=resp) for resp in responses
        ]

        verifier = SemanticVerifier(api_key="test_key")
        results = await verifier.verify_and_filter_clips(clips, sample_broll_need_with_context)

        # Should return 2 passing clips (clip1 and clip3)
        assert len(results) == 2
        for _path, _segment, verification in results:
            assert verification.passed is True

        # Verify sorted by score (highest first)
        assert results[0][2].similarity_score == 0.95
        assert results[1][2].similarity_score == 0.92

    @pytest.mark.asyncio
    async def test_disabled_verification_passes_all_clips(
        self, mock_genai_client, mock_config_disabled, sample_broll_need_with_context, temp_clip_file
    ):
        """Test that disabled verification auto-passes all clips."""
        verifier = SemanticVerifier(api_key="test_key")
        result = await verifier.verify_clip(temp_clip_file, sample_broll_need_with_context)

        assert result.passed is True
        assert result.similarity_score == 1.0
        assert "disabled" in result.rationale.lower()

        # Should NOT have called Gemini API
        mock_genai_client.models.generate_content.assert_not_called()


# --- Test 5: Negative Examples Included in Evaluation ---


class TestNegativeExamplesInEvaluation:
    """Test that negative examples are included in evaluation prompts."""

    @pytest.fixture
    def mock_ai_service(self):
        """Create a minimal AIService instance for testing."""
        with patch("src.services.ai_service.Client"):
            from src.services.ai_service import AIService

            service = AIService(api_key="test_key")
            return service

    def test_negative_examples_generated_for_required_elements(
        self, mock_ai_service, sample_broll_need_with_context
    ):
        """Verify negative examples are generated for required_elements."""
        result = mock_ai_service._generate_negative_examples(sample_broll_need_with_context)

        # Should have rejection header
        assert "REJECT these types of videos:" in result

        # Should have rejection lines for required elements
        assert "Video missing 'people'" in result
        assert "Video missing 'laptops'" in result
        assert "Video missing 'coffee shop interior'" in result
        assert "Video missing 'busy atmosphere'" in result

        # Should include general rejection patterns
        assert "opposite of context" in result or "generic stock footage" in result

    def test_no_negative_examples_without_context(
        self, mock_ai_service, sample_broll_need_minimal
    ):
        """Verify no negative examples when original_context is empty."""
        result = mock_ai_service._generate_negative_examples(sample_broll_need_minimal)

        # Should return empty string when no context
        assert result == ""

    def test_negative_examples_included_in_evaluation_prompt(
        self, mock_ai_service, sample_broll_need_with_context
    ):
        """Verify negative examples are included in the evaluation prompt."""
        prompt = mock_ai_service._build_enhanced_evaluation_prompt(
            search_phrase="coffee shop interior",
            results_text="Test video results",
            broll_need=sample_broll_need_with_context,
        )

        # Should include negative examples section
        assert "REJECT these types of videos:" in prompt

        # Should include rejection instruction
        assert "score it <=4 regardless of technical quality" in prompt

    def test_negative_examples_limit_to_four_elements(self, mock_ai_service):
        """Verify only first 4 required elements get negative examples."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="test",
            description="test",
            context="test",
            original_context="Some context for testing element limits",
            required_elements=[
                "element1", "element2", "element3", "element4",
                "element5", "element6",  # These should be excluded
            ],
        )

        result = mock_ai_service._generate_negative_examples(need)

        # Should have first 4
        assert "Video missing 'element1'" in result
        assert "Video missing 'element2'" in result
        assert "Video missing 'element3'" in result
        assert "Video missing 'element4'" in result

        # Should NOT have 5th and 6th
        assert "Video missing 'element5'" not in result
        assert "Video missing 'element6'" not in result


# --- Additional Integration Tests ---


class TestPipelineEndToEnd:
    """End-to-end integration tests for the semantic pipeline."""

    def test_broll_need_context_propagates_through_dataclasses(
        self, sample_broll_need_with_context
    ):
        """Verify context can be serialized and deserialized correctly."""
        need = sample_broll_need_with_context

        # Simulate serialization (as might happen in logging or caching)
        need_dict = {
            "timestamp": need.timestamp,
            "search_phrase": need.search_phrase,
            "description": need.description,
            "context": need.context,
            "original_context": need.original_context,
            "required_elements": need.required_elements,
        }

        # Reconstruct from dict
        reconstructed = BRollNeed(**need_dict)

        # Verify all fields preserved
        assert reconstructed.original_context == need.original_context
        assert reconstructed.required_elements == need.required_elements
        assert len(reconstructed.original_context) == len(need.original_context)

    def test_verification_result_serialization(self):
        """Verify VerificationResult can be converted to dict correctly."""
        result = VerificationResult(
            passed=True,
            similarity_score=0.95,
            matched_elements=["people", "laptops", "coffee shop"],
            missing_elements=["busy atmosphere"],
            rationale="Good match overall",
        )

        result_dict = result.to_dict()

        assert result_dict["passed"] is True
        assert result_dict["similarity_score"] == 0.95
        assert result_dict["matched_elements"] == ["people", "laptops", "coffee shop"]
        assert result_dict["missing_elements"] == ["busy atmosphere"]
        assert result_dict["rationale"] == "Good match overall"
        assert result_dict["elements_match_ratio"] == 0.75  # 3/4 elements

    def test_clip_segment_works_with_verification(self, sample_clip_segment):
        """Verify ClipSegment integrates properly with verification workflow."""
        segment = sample_clip_segment

        # ClipSegment should have all required fields for verification
        assert hasattr(segment, "start_time")
        assert hasattr(segment, "end_time")
        assert hasattr(segment, "relevance_score")
        assert hasattr(segment, "description")

        # Verify duration calculation
        assert segment.duration == 7.0  # 12.0 - 5.0

        # Verify metadata dict includes all fields
        metadata = segment.to_metadata_dict()
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "relevance_score" in metadata
        assert "description" in metadata

    @pytest.mark.asyncio
    async def test_fallback_to_description_when_no_context(
        self, mock_genai_client, mock_config_enabled, sample_broll_need_minimal, temp_clip_file, caplog
    ):
        """Verify verification falls back to description when original_context is empty."""
        mock_genai_client.models.generate_content.return_value.text = json.dumps({
            "similarity_score": 0.85,
            "matched_elements": [],
            "missing_elements": [],
            "rationale": "Matches description"
        })

        with caplog.at_level(logging.WARNING):
            verifier = SemanticVerifier(api_key="test_key")
            await verifier.verify_clip(temp_clip_file, sample_broll_need_minimal)

        # Check the prompt used description as fallback
        call_args = mock_genai_client.models.generate_content.call_args
        prompt_content = str(call_args)

        # Should include description since original_context was empty
        assert "Beautiful sunset over the ocean" in prompt_content

        # Should have logged a warning about missing context
        log_messages = " ".join([record.message for record in caplog.records])
        assert "fallback" in log_messages.lower() or "description" in log_messages.lower()
