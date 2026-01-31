"""Tests for Feature 1, 2, and 3: Style Detection, Context Window, Feedback Loop."""

import json
import tempfile
from pathlib import Path

import pytest

from models.broll_need import BRollNeed, TranscriptResult, TranscriptSegment
from models.feedback import ContentFeedback, FeedbackStore, RejectionFilter
from models.image import ImageNeed, ImageResult
from models.style import ContentStyle, VisualStyle, ColorTone, PacingStyle
from services.feedback_service import FeedbackService


class TestContentStyle:
    """Tests for Feature 1: Style/Mood Detection models."""

    def test_create_content_style_defaults(self):
        """Test ContentStyle creation with defaults."""
        style = ContentStyle(
            visual_style=VisualStyle.CINEMATIC,
            color_tone=ColorTone.WARM,
            pacing=PacingStyle.MODERATE,
        )
        assert style.visual_style == VisualStyle.CINEMATIC
        assert style.color_tone == ColorTone.WARM
        assert style.topic == ""
        assert style.target_audience == ""
        assert style.preferred_imagery == []
        assert style.avoid_imagery == []

    def test_create_content_style_full(self):
        """Test ContentStyle creation with all fields."""
        style = ContentStyle(
            visual_style=VisualStyle.RAW,
            color_tone=ColorTone.DESATURATED,
            pacing=PacingStyle.FAST,
            topic="MMA training and conditioning",
            topic_keywords=["MMA", "BJJ", "fighter", "gym"],
            target_audience="Competitive fighters",
            audience_level="advanced",
            content_type="educational",
            tone="serious",
            preferred_imagery=["professional athletes", "intense training"],
            avoid_imagery=["casual gym-goers", "stock fitness models"],
            mood_keywords=["intense", "dedicated"],
            avoid_keywords=["easy", "beginner"],
        )
        assert style.topic == "MMA training and conditioning"
        assert "MMA" in style.topic_keywords
        assert style.audience_level == "advanced"
        assert len(style.preferred_imagery) == 2
        assert len(style.avoid_imagery) == 2

    def test_to_prompt_context(self):
        """Test ContentStyle.to_prompt_context() generates useful output."""
        style = ContentStyle(
            visual_style=VisualStyle.DOCUMENTARY,
            color_tone=ColorTone.NEUTRAL,
            pacing=PacingStyle.SLOW,
            topic="Ocean conservation",
            target_audience="Environmental enthusiasts",
            preferred_imagery=["underwater footage", "marine life"],
            avoid_imagery=["pollution", "dead animals"],
        )
        context = style.to_prompt_context()
        assert "DOCUMENTARY" in context or "documentary" in context.lower()
        assert "Ocean conservation" in context
        assert "Environmental enthusiasts" in context

    def test_visual_style_enum(self):
        """Test VisualStyle enum values."""
        assert VisualStyle.CINEMATIC.value == "cinematic"
        assert VisualStyle.DOCUMENTARY.value == "documentary"
        assert VisualStyle.RAW.value == "raw"
        assert VisualStyle.PROFESSIONAL.value == "professional"
        assert VisualStyle.MODERN.value == "modern"


class TestContextWindow:
    """Tests for Feature 2: Context Window (±10 seconds)."""

    def test_image_need_context_fields(self):
        """Test ImageNeed has context window fields."""
        need = ImageNeed(
            timestamp=30.0,
            search_phrase="coffee shop remote work",
            context="people working on laptops",
            context_before="talking about remote work culture",
            context_after="discussing productivity tips",
            themes=["work", "productivity", "remote"],
            emotional_tone="professional",
        )
        assert need.context_before == "talking about remote work culture"
        assert need.context_after == "discussing productivity tips"
        assert "work" in need.themes
        assert need.emotional_tone == "professional"

    def test_image_need_get_enhanced_context(self):
        """Test ImageNeed.get_enhanced_context() method."""
        # Test with full_context set - returns full_context
        need_with_full = ImageNeed(
            timestamp=30.0,
            search_phrase="coffee shop",
            context="current context",
            context_before="before context",
            context_after="after context",
            full_context="before context current context after context",
            themes=["work", "coffee"],
        )
        enhanced = need_with_full.get_enhanced_context()
        assert "before context" in enhanced
        assert "current context" in enhanced
        assert "after context" in enhanced

        # Test fallback - returns context when full_context is empty
        need_without_full = ImageNeed(
            timestamp=30.0,
            search_phrase="coffee shop",
            context="current context",
            context_before="before context",
            context_after="after context",
            themes=["work", "coffee"],
        )
        fallback = need_without_full.get_enhanced_context()
        assert fallback == "current context"

    def test_broll_need_context_fields(self):
        """Test BRollNeed has context window fields."""
        need = BRollNeed(
            timestamp=60.0,
            search_phrase="athlete training",
            description="Show athlete doing exercises",
            context="talking about training",
            context_before="introduced the topic",
            context_after="will discuss nutrition next",
            themes=["fitness", "training"],
            emotional_tone="motivational",
        )
        assert need.context_before == "introduced the topic"
        assert need.context_after == "will discuss nutrition next"
        assert "fitness" in need.themes

    def test_transcript_result_context_window(self):
        """Test TranscriptResult.get_context_window() method."""
        segments = [
            TranscriptSegment(start=0.0, end=5.0, text="Welcome to the show."),
            TranscriptSegment(start=5.0, end=10.0, text="Today we discuss fitness."),
            TranscriptSegment(start=10.0, end=15.0, text="Let's start with cardio."),
            TranscriptSegment(start=15.0, end=20.0, text="Running is great."),
            TranscriptSegment(start=20.0, end=25.0, text="Now let's do weights."),
        ]
        transcript = TranscriptResult(
            text="Welcome to the show. Today we discuss fitness. Let's start with cardio. Running is great. Now let's do weights.",
            segments=segments,
            duration=25.0,
        )

        before, at, after = transcript.get_context_window(12.5, window_seconds=5.0)
        # Should capture context around 12.5s (±5s = 7.5s to 17.5s)
        assert isinstance(before, str)
        assert isinstance(at, str)
        assert isinstance(after, str)

    def test_transcript_result_full_context_window(self):
        """Test TranscriptResult.get_full_context_window() method."""
        segments = [
            TranscriptSegment(start=0.0, end=10.0, text="First part."),
            TranscriptSegment(start=10.0, end=20.0, text="Second part."),
            TranscriptSegment(start=20.0, end=30.0, text="Third part."),
        ]
        transcript = TranscriptResult(
            text="First part. Second part. Third part.",
            segments=segments,
            duration=30.0,
        )

        full_context = transcript.get_full_context_window(15.0, window_seconds=10.0)
        assert isinstance(full_context, str)


class TestFeedbackModels:
    """Tests for Feature 3: Feedback Loop models."""

    def test_content_feedback_creation(self):
        """Test ContentFeedback dataclass."""
        feedback = ContentFeedback(
            content_id="pexels_12345",
            content_type="image",
            source="pexels",
            search_phrase="athlete training",
            was_rejected=True,
            rejection_reason="too_generic",
            title="Generic gym photo",
        )
        assert feedback.content_id == "pexels_12345"
        assert feedback.was_rejected is True
        assert feedback.rejection_reason == "too_generic"

    def test_content_feedback_to_dict(self):
        """Test ContentFeedback serialization."""
        feedback = ContentFeedback(
            content_id="yt_abc123",
            content_type="video",
            source="youtube",
            search_phrase="cooking tutorial",
            was_rejected=False,
        )
        data = feedback.to_dict()
        assert data["content_id"] == "yt_abc123"
        assert data["was_rejected"] is False
        assert "feedback_time" in data

    def test_content_feedback_from_dict(self):
        """Test ContentFeedback deserialization."""
        data = {
            "content_id": "px_999",
            "content_type": "image",
            "source": "pixabay",
            "search_phrase": "sunset",
            "was_rejected": True,
            "rejection_reason": "wrong_subject",
        }
        feedback = ContentFeedback.from_dict(data)
        assert feedback.content_id == "px_999"
        assert feedback.rejection_reason == "wrong_subject"

    def test_feedback_store_add_rejection(self):
        """Test FeedbackStore.add_rejection() method."""
        store = FeedbackStore()
        feedback = ContentFeedback(
            content_id="test_1",
            content_type="image",
            source="pexels",
            search_phrase="gym workout",
            was_rejected=True,
            title="Stock gym photo",
        )

        store.add_rejection(feedback)

        assert store.total_rejections == 1
        assert len(store.rejections) == 1
        assert "pexels" in store.rejected_sources
        assert "test_1" in store.rejected_content_ids

    def test_feedback_store_add_approval(self):
        """Test FeedbackStore.add_approval() method."""
        store = FeedbackStore()
        feedback = ContentFeedback(
            content_id="test_2",
            content_type="video",
            source="youtube",
            search_phrase="mma training",
            was_rejected=False,
            title="Pro MMA fighter workout",
        )

        store.add_approval(feedback)

        assert store.total_approvals == 1
        assert len(store.approvals) == 1
        assert len(store.successful_patterns) > 0

    def test_feedback_store_rejection_score(self):
        """Test FeedbackStore.get_rejection_score() method."""
        store = FeedbackStore()

        # Add multiple rejections for same source
        for i in range(3):
            store.add_rejection(ContentFeedback(
                content_id=f"px_{i}",
                content_type="image",
                source="pixabay",
                search_phrase="generic photo",
                was_rejected=True,
                title="Stock photo",
            ))

        # Score should be higher for pixabay source now
        score = store.get_rejection_score(
            content_id="px_new",
            source="pixabay",
            title="Another stock photo",
        )
        assert score > 0.0  # Should have some rejection likelihood

    def test_feedback_store_serialization(self):
        """Test FeedbackStore to_dict/from_dict round-trip."""
        store = FeedbackStore()
        store.add_rejection(ContentFeedback(
            content_id="test_1",
            content_type="image",
            source="pexels",
            search_phrase="test",
            was_rejected=True,
        ))
        store.add_approval(ContentFeedback(
            content_id="test_2",
            content_type="video",
            source="youtube",
            search_phrase="test",
            was_rejected=False,
        ))

        data = store.to_dict()
        restored = FeedbackStore.from_dict(data)

        assert restored.total_rejections == 1
        assert restored.total_approvals == 1

    def test_rejection_filter_to_prompt_context(self):
        """Test RejectionFilter.to_prompt_context() method."""
        filter_obj = RejectionFilter(
            avoid_sources=["pixabay"],
            avoid_keywords=["generic", "stock"],
            boost_sources=["pexels"],
            learned_preferences="Prefers action shots over static",
        )

        context = filter_obj.to_prompt_context()
        assert "FEEDBACK HISTORY" in context
        assert "pixabay" in context.lower()
        assert "pexels" in context.lower()


class TestFeedbackService:
    """Tests for Feature 3: FeedbackService."""

    def test_feedback_service_initialization(self):
        """Test FeedbackService initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)
            assert service.feedback_dir == Path(tmpdir)
            assert service.store is not None

    def test_feedback_service_record_rejection(self):
        """Test FeedbackService.record_rejection() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)
            service.record_rejection(
                content_id="test_123",
                content_type="image",
                source="pexels",
                search_phrase="athlete training",
                reason="too_generic",
                title="Stock athlete photo",
            )

            assert service.store.total_rejections == 1
            # Check persistence
            assert service.feedback_file.exists()

    def test_feedback_service_record_approval(self):
        """Test FeedbackService.record_approval() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)
            service.record_approval(
                content_id="test_456",
                content_type="video",
                source="youtube",
                search_phrase="mma fighter",
                title="Pro MMA training",
            )

            assert service.store.total_approvals == 1

    def test_feedback_service_persistence(self):
        """Test FeedbackService persists and loads data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            service1 = FeedbackService(tmpdir)
            service1.record_rejection(
                content_id="persist_test",
                content_type="image",
                source="google",
                search_phrase="test",
            )

            # Load in new instance
            service2 = FeedbackService(tmpdir)
            assert service2.store.total_rejections == 1
            assert "persist_test" in service2.store.rejected_content_ids

    def test_feedback_service_get_rejection_filter(self):
        """Test FeedbackService.get_rejection_filter() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)

            # Add enough rejections to trigger filtering
            for i in range(4):
                service.record_rejection(
                    content_id=f"bad_{i}",
                    content_type="image",
                    source="pixabay",
                    search_phrase="generic photo",
                    reason="too_generic",
                )

            filter_obj = service.get_rejection_filter("test phrase")
            assert "pixabay" in filter_obj.avoid_sources

    def test_feedback_service_apply_to_image_candidates(self):
        """Test FeedbackService.apply_to_image_candidates() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)

            # Record rejection for specific content
            service.record_rejection(
                content_id="bad_image",
                content_type="image",
                source="pexels",
                search_phrase="test",
            )

            # Create candidates
            candidates = [
                ImageResult(
                    image_id="bad_image",
                    title="Bad Image",
                    url="http://example.com/bad.jpg",
                    download_url="http://example.com/bad.jpg",
                    width=1920,
                    height=1080,
                    source="pexels",
                ),
                ImageResult(
                    image_id="good_image",
                    title="Good Image",
                    url="http://example.com/good.jpg",
                    download_url="http://example.com/good.jpg",
                    width=1920,
                    height=1080,
                    source="pexels",
                ),
            ]

            filtered = service.apply_to_image_candidates(candidates, "test")
            # bad_image should be deprioritized or filtered
            assert len(filtered) >= 1

    def test_feedback_service_get_prompt_additions(self):
        """Test FeedbackService.get_prompt_additions() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)

            # No rejections yet
            empty_context = service.get_prompt_additions("test")
            assert empty_context == ""

            # Add rejections
            for i in range(3):
                service.record_rejection(
                    content_id=f"rej_{i}",
                    content_type="image",
                    source="google",
                    search_phrase="test photo",
                    reason="poor_quality",
                )

            context = service.get_prompt_additions("test photo")
            # Should have some content now
            assert isinstance(context, str)

    def test_feedback_service_statistics(self):
        """Test FeedbackService.get_statistics() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)
            service.record_rejection(
                content_id="stat_1",
                content_type="image",
                source="pexels",
                search_phrase="test",
            )
            service.record_approval(
                content_id="stat_2",
                content_type="video",
                source="youtube",
                search_phrase="test",
            )

            stats = service.get_statistics()
            assert stats["total_rejections"] == 1
            assert stats["total_approvals"] == 1
            assert "top_rejected_sources" in stats

    def test_feedback_service_clear(self):
        """Test FeedbackService.clear_feedback() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FeedbackService(tmpdir)
            service.record_rejection(
                content_id="clear_test",
                content_type="image",
                source="pexels",
                search_phrase="test",
            )
            assert service.store.total_rejections == 1

            service.clear_feedback()
            assert service.store.total_rejections == 0
            assert service.store.total_approvals == 0
