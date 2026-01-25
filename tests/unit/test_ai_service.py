"""Unit tests for AIService including negative examples generation (Fix 5)."""

import pytest
from unittest.mock import Mock, patch

from src.models.broll_need import BRollNeed


class TestGenerateNegativeExamples:
    """Tests for AIService._generate_negative_examples() method - Fix 5."""

    @pytest.fixture
    def mock_ai_service(self):
        """Create a minimal AIService instance for testing."""
        with patch("src.services.ai_service.Client"):
            from src.services.ai_service import AIService

            service = AIService(api_key="test_key")
            return service

    def test_returns_empty_when_no_context(self, mock_ai_service):
        """Should return empty string when original_context is missing."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="",  # Empty
            required_elements=["people", "laptops"],
        )

        result = mock_ai_service._generate_negative_examples(broll_need)

        assert result == ""

    def test_returns_empty_when_no_required_elements(self, mock_ai_service):
        """Should return empty string when required_elements is empty."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="The coffee shop was packed with remote workers",
            required_elements=[],  # Empty
        )

        result = mock_ai_service._generate_negative_examples(broll_need)

        assert result == ""

    def test_generates_examples_for_required_elements(self, mock_ai_service):
        """Should generate rejection examples for each required element."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="The coffee shop was packed with remote workers typing on laptops",
            required_elements=["people", "laptops", "morning light"],
        )

        result = mock_ai_service._generate_negative_examples(broll_need)

        # Check header
        assert "REJECT these types of videos:" in result

        # Check each required element has a rejection line
        assert "Video missing 'people'" in result
        assert "Video missing 'laptops'" in result
        assert "Video missing 'morning light'" in result

    def test_limits_to_four_elements(self, mock_ai_service):
        """Should only generate examples for first 4 required elements."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="Full transcript context here",
            required_elements=[
                "element1",
                "element2",
                "element3",
                "element4",
                "element5",
                "element6",
            ],
        )

        result = mock_ai_service._generate_negative_examples(broll_need)

        # Should have first 4 elements
        assert "Video missing 'element1'" in result
        assert "Video missing 'element2'" in result
        assert "Video missing 'element3'" in result
        assert "Video missing 'element4'" in result

        # Should NOT have 5th and 6th elements
        assert "Video missing 'element5'" not in result
        assert "Video missing 'element6'" not in result

    def test_includes_general_rejection_patterns(self, mock_ai_service):
        """Should include general rejection patterns for context mismatches."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="The coffee shop was packed with remote workers",
            required_elements=["people"],
        )

        result = mock_ai_service._generate_negative_examples(broll_need)

        # Should include general rejection patterns
        assert "opposite of context" in result
        assert "generic stock footage" in result


class TestBuildEnhancedEvaluationPrompt:
    """Tests for negative examples integration in _build_enhanced_evaluation_prompt()."""

    @pytest.fixture
    def mock_ai_service(self):
        """Create a minimal AIService instance for testing."""
        with patch("src.services.ai_service.Client"):
            from src.services.ai_service import AIService

            service = AIService(api_key="test_key")
            return service

    def test_includes_negative_examples_in_prompt(self, mock_ai_service):
        """Should include negative examples section when broll_need has context and elements."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="The coffee shop was packed with remote workers typing on laptops",
            required_elements=["people", "laptops", "coffee shop interior"],
        )

        prompt = mock_ai_service._build_enhanced_evaluation_prompt(
            search_phrase="coffee shop interior",
            results_text="Test video results",
            broll_need=broll_need,
        )

        # Should include negative examples
        assert "REJECT these types of videos:" in prompt
        assert "Video missing 'people'" in prompt

        # Should include the scoring instruction for rejections
        assert "score it <=4 regardless of technical quality" in prompt

    def test_no_negative_examples_without_context(self, mock_ai_service):
        """Should not include negative examples section when context is missing."""
        broll_need = BRollNeed(
            timestamp=30.0,
            search_phrase="coffee shop interior",
            description="Coffee shop scene",
            context="Testing",
            original_context="",  # Empty - no context
            required_elements=["people"],
        )

        prompt = mock_ai_service._build_enhanced_evaluation_prompt(
            search_phrase="coffee shop interior",
            results_text="Test video results",
            broll_need=broll_need,
        )

        # Should NOT include negative examples rejection header
        # (The REJECTION CRITERIA section is different - it's always present)
        assert "REJECT these types of videos:" not in prompt
