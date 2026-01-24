"""Unit tests for Q2 Enhanced Search Phrase Generation.

Tests the enhanced BRollNeed metadata fields, search_with_fallback method,
negative keyword filtering, and enhanced evaluation prompt generation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.models.broll_need import BRollNeed
from src.models.video import VideoResult


class TestBRollNeedEnhanced:
    """Tests for enhanced BRollNeed metadata fields (Q2)."""

    def test_create_with_enhanced_metadata(self):
        """Test creating a BRollNeed with enhanced metadata."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline aerial",
            description="Urban establishing shot",
            context="talking about urban development",
            alternate_searches=[
                "downtown buildings aerial",
                "metropolitan skyline",
            ],
            negative_keywords=["night", "rain", "logo"],
            visual_style="cinematic",
            time_of_day="golden hour",
            movement="drone push-in",
        )

        assert need.search_phrase == "city skyline aerial"
        assert len(need.alternate_searches) == 2
        assert "downtown buildings aerial" in need.alternate_searches
        assert len(need.negative_keywords) == 3
        assert "night" in need.negative_keywords
        assert need.visual_style == "cinematic"
        assert need.time_of_day == "golden hour"
        assert need.movement == "drone push-in"

    def test_has_enhanced_metadata_true(self):
        """Test has_enhanced_metadata returns True when metadata present."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            alternate_searches=["downtown buildings"],
        )
        assert need.has_enhanced_metadata() is True

    def test_has_enhanced_metadata_negative_keywords(self):
        """Test has_enhanced_metadata returns True with negative keywords."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            negative_keywords=["logo", "watermark"],
        )
        assert need.has_enhanced_metadata() is True

    def test_has_enhanced_metadata_visual_style(self):
        """Test has_enhanced_metadata returns True with visual style."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            visual_style="cinematic",
        )
        assert need.has_enhanced_metadata() is True

    def test_has_enhanced_metadata_false(self):
        """Test has_enhanced_metadata returns False when no metadata."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
        )
        assert need.has_enhanced_metadata() is False

    def test_primary_search_property(self):
        """Test primary_search property returns search_phrase."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline aerial",
            description="Urban shot",
            context="",
        )
        assert need.primary_search == "city skyline aerial"

    def test_all_search_phrases_property(self):
        """Test all_search_phrases includes primary and alternates."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline aerial",
            description="Urban shot",
            context="",
            alternate_searches=["downtown buildings", "metropolitan area"],
        )
        all_phrases = need.all_search_phrases
        assert len(all_phrases) == 3
        assert all_phrases[0] == "city skyline aerial"
        assert "downtown buildings" in all_phrases
        assert "metropolitan area" in all_phrases

    def test_all_search_phrases_no_alternates(self):
        """Test all_search_phrases with no alternates returns only primary."""
        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline aerial",
            description="Urban shot",
            context="",
        )
        all_phrases = need.all_search_phrases
        assert len(all_phrases) == 1
        assert all_phrases[0] == "city skyline aerial"


class TestNegativeKeywordFilter:
    """Tests for negative keyword filtering in AIService."""

    def test_filter_removes_matching_titles(self):
        """Test that videos with negative keywords in title are filtered."""
        from src.services.ai_service import AIService

        # Create mock AIService (we just need the filter method)
        ai_service = Mock(spec=AIService)
        ai_service.filter_by_negative_keywords = AIService.filter_by_negative_keywords.__get__(
            ai_service
        )

        videos = [
            VideoResult(
                video_id="v1",
                title="Beautiful city skyline at sunset",
                url="http://test.com/v1",
                duration=60,
                description="A peaceful scene",
            ),
            VideoResult(
                video_id="v2",
                title="City skyline with LOGO overlay",
                url="http://test.com/v2",
                duration=60,
                description="Urban footage",
            ),
            VideoResult(
                video_id="v3",
                title="Night city lights panorama",
                url="http://test.com/v3",
                duration=60,
                description="After dark",
            ),
        ]

        negative_keywords = ["logo", "night"]
        filtered = ai_service.filter_by_negative_keywords(videos, negative_keywords)

        assert len(filtered) == 1
        assert filtered[0].video_id == "v1"

    def test_filter_removes_matching_descriptions(self):
        """Test that videos with negative keywords in description are filtered."""
        from src.services.ai_service import AIService

        ai_service = Mock(spec=AIService)
        ai_service.filter_by_negative_keywords = AIService.filter_by_negative_keywords.__get__(
            ai_service
        )

        videos = [
            VideoResult(
                video_id="v1",
                title="City skyline",
                url="http://test.com/v1",
                duration=60,
                description="Beautiful daytime footage with watermark",
            ),
            VideoResult(
                video_id="v2",
                title="Urban panorama",
                url="http://test.com/v2",
                duration=60,
                description="Clean footage, no overlays",
            ),
        ]

        negative_keywords = ["watermark"]
        filtered = ai_service.filter_by_negative_keywords(videos, negative_keywords)

        assert len(filtered) == 1
        assert filtered[0].video_id == "v2"

    def test_filter_empty_keywords_returns_all(self):
        """Test that empty negative keywords returns all videos."""
        from src.services.ai_service import AIService

        ai_service = Mock(spec=AIService)
        ai_service.filter_by_negative_keywords = AIService.filter_by_negative_keywords.__get__(
            ai_service
        )

        videos = [
            VideoResult(
                video_id="v1",
                title="Video 1",
                url="http://test.com/v1",
                duration=60,
                description="",
            ),
            VideoResult(
                video_id="v2",
                title="Video 2",
                url="http://test.com/v2",
                duration=60,
                description="",
            ),
        ]

        filtered = ai_service.filter_by_negative_keywords(videos, [])
        assert len(filtered) == 2

    def test_filter_case_insensitive(self):
        """Test that keyword matching is case insensitive."""
        from src.services.ai_service import AIService

        ai_service = Mock(spec=AIService)
        ai_service.filter_by_negative_keywords = AIService.filter_by_negative_keywords.__get__(
            ai_service
        )

        videos = [
            VideoResult(
                video_id="v1",
                title="Video with WATERMARK",
                url="http://test.com/v1",
                duration=60,
                description="",
            ),
            VideoResult(
                video_id="v2",
                title="Clean Video",
                url="http://test.com/v2",
                duration=60,
                description="",
            ),
        ]

        negative_keywords = ["watermark"]  # lowercase
        filtered = ai_service.filter_by_negative_keywords(videos, negative_keywords)

        assert len(filtered) == 1
        assert filtered[0].video_id == "v2"


class TestEnhancedEvaluationPrompt:
    """Tests for enhanced evaluation prompt building."""

    def test_build_enhanced_prompt_with_metadata(self):
        """Test that enhanced prompt includes visual preferences."""
        from src.services.ai_service import AIService

        ai_service = Mock(spec=AIService)
        ai_service._build_enhanced_evaluation_prompt = (
            AIService._build_enhanced_evaluation_prompt.__get__(ai_service)
        )

        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            visual_style="cinematic",
            time_of_day="golden hour",
            movement="drone",
            negative_keywords=["logo", "watermark"],
        )

        prompt = ai_service._build_enhanced_evaluation_prompt(
            search_phrase="city skyline",
            results_text="ID: v1\nTitle: Test",
            broll_need=need,
            transcript_segment="talking about cities",
        )

        # Check visual preferences are included
        assert "cinematic" in prompt
        assert "golden hour" in prompt
        assert "drone" in prompt
        assert "logo" in prompt or "watermark" in prompt
        assert "VISUAL PREFERENCES" in prompt
        assert "TRANSCRIPT CONTEXT" in prompt

    def test_build_enhanced_prompt_no_metadata(self):
        """Test that prompt works with minimal metadata."""
        from src.services.ai_service import AIService

        ai_service = Mock(spec=AIService)
        ai_service._build_enhanced_evaluation_prompt = (
            AIService._build_enhanced_evaluation_prompt.__get__(ai_service)
        )

        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
        )

        prompt = ai_service._build_enhanced_evaluation_prompt(
            search_phrase="city skyline",
            results_text="ID: v1\nTitle: Test",
            broll_need=need,
            transcript_segment=None,
        )

        # Basic prompt should still work
        assert "city skyline" in prompt
        assert "VISUAL PREFERENCES" not in prompt  # No enhanced metadata


class TestSearchWithFallback:
    """Tests for search_with_fallback method."""

    @pytest.mark.asyncio
    async def test_search_with_fallback_primary_sufficient(self):
        """Test that primary search is used when results are sufficient."""
        from src.broll_processor import BRollProcessor

        # Mock the processor
        processor = Mock(spec=BRollProcessor)
        processor.search_youtube_videos = AsyncMock(
            return_value=[
                VideoResult(
                    video_id=f"v{i}",
                    title=f"Video {i}",
                    url=f"http://test.com/v{i}",
                    duration=60,
                    description="",
                )
                for i in range(10)
            ]
        )
        processor.ai_service = Mock()
        processor.ai_service.filter_by_negative_keywords = lambda x, y: x

        # Bind the method to our mock
        processor.search_with_fallback = BRollProcessor.search_with_fallback.__get__(
            processor
        )

        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            alternate_searches=["downtown", "urban"],
        )

        results = await processor.search_with_fallback(need, min_results=5)

        # Should only call search once (primary is sufficient)
        assert len(results) == 10
        processor.search_youtube_videos.assert_called_once_with("city skyline")

    @pytest.mark.asyncio
    async def test_search_with_fallback_uses_alternates(self):
        """Test that alternate searches are used when primary is insufficient."""
        from src.broll_processor import BRollProcessor

        # Mock the processor
        processor = Mock(spec=BRollProcessor)

        # Primary returns 2 results (insufficient)
        # Each alternate returns 3 results
        call_count = [0]

        async def mock_search(phrase):
            call_count[0] += 1
            if call_count[0] == 1:  # Primary
                return [
                    VideoResult(
                        video_id=f"v{i}",
                        title=f"Primary {i}",
                        url=f"http://test.com/v{i}",
                        duration=60,
                        description="",
                    )
                    for i in range(2)
                ]
            else:  # Alternates
                return [
                    VideoResult(
                        video_id=f"alt{call_count[0]}_{i}",
                        title=f"Alt {call_count[0]} Video {i}",
                        url=f"http://test.com/alt{call_count[0]}_{i}",
                        duration=60,
                        description="",
                    )
                    for i in range(3)
                ]

        processor.search_youtube_videos = mock_search
        processor.ai_service = Mock()
        processor.ai_service.filter_by_negative_keywords = lambda x, y: x

        processor.search_with_fallback = BRollProcessor.search_with_fallback.__get__(
            processor
        )

        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            alternate_searches=["downtown", "urban"],
        )

        results = await processor.search_with_fallback(need, min_results=5)

        # Should have results from primary + alternates
        assert len(results) >= 5
        assert call_count[0] >= 2  # At least primary + one alternate

    @pytest.mark.asyncio
    async def test_search_with_fallback_applies_negative_filter(self):
        """Test that negative keywords filter is applied."""
        from src.broll_processor import BRollProcessor

        # Mock the processor
        processor = Mock(spec=BRollProcessor)
        processor.search_youtube_videos = AsyncMock(
            return_value=[
                VideoResult(
                    video_id="v1",
                    title="City skyline",
                    url="http://test.com/v1",
                    duration=60,
                    description="",
                ),
                VideoResult(
                    video_id="v2",
                    title="Night skyline with logo",
                    url="http://test.com/v2",
                    duration=60,
                    description="",
                ),
            ]
        )

        # Mock the filter to actually filter
        def mock_filter(videos, keywords):
            return [v for v in videos if not any(k.lower() in v.title.lower() for k in keywords)]

        processor.ai_service = Mock()
        processor.ai_service.filter_by_negative_keywords = mock_filter

        processor.search_with_fallback = BRollProcessor.search_with_fallback.__get__(
            processor
        )

        need = BRollNeed(
            timestamp=30.0,
            search_phrase="city skyline",
            description="Urban shot",
            context="",
            negative_keywords=["logo", "night"],
        )

        results = await processor.search_with_fallback(need, min_results=1)

        # Should filter out the video with "Night" and "logo" in title
        assert len(results) == 1
        assert results[0].video_id == "v1"
