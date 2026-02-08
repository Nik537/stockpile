"""Video search service for multi-source video discovery.

This module extracts video search functionality from BRollProcessor into a dedicated
service that can be tested independently and reused across the codebase.

Responsibilities:
- Search multiple video sources (YouTube, Pexels, Pixabay) for matching videos
- Implement fallback search with alternate search phrases
- Evaluate and score videos using AI
- Filter results using negative keywords and pre-filtering

This is part of the BRollProcessor decomposition into focused service classes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models.broll_need import BRollNeed, TranscriptResult
    from models.style import ContentStyle
    from models.user_preferences import UserPreferences
    from models.video import ScoredVideo, VideoResult
    from services.ai_service import AIService
    from services.feedback_service import FeedbackService
    from services.video_filter import VideoPreFilter
    from services.video_sources import VideoSource

logger = logging.getLogger(__name__)


class VideoSearchService:
    """Service for searching and evaluating videos across multiple sources.

    This service encapsulates video search logic that was previously embedded
    in BRollProcessor, providing a clean interface for:
    - Multi-source video search (YouTube, Pexels, Pixabay)
    - Fallback search with alternate phrases
    - AI-based video evaluation and scoring
    - Integration with content style and feedback systems
    """

    def __init__(
        self,
        video_sources: list["VideoSource"],
        video_prefilter: "VideoPreFilter",
        ai_service: "AIService",
        feedback_service: Optional["FeedbackService"] = None,
        content_style: Optional["ContentStyle"] = None,
        content_filter: Optional[str] = None,
        max_videos_per_phrase: int = 3,
        evaluation_context_seconds: float = 30.0,
    ):
        """Initialize the video search service.

        Args:
            video_sources: List of video source instances (YouTube, Pexels, etc.)
            video_prefilter: VideoPreFilter instance for metadata-based filtering
            ai_service: AIService instance for video evaluation
            feedback_service: Optional FeedbackService for learning from rejections
            content_style: Optional ContentStyle for style-aware evaluation
            content_filter: Optional content filter string (e.g., "men only, no women")
            max_videos_per_phrase: Maximum videos to return per search (default: 3)
            evaluation_context_seconds: Seconds of context for evaluation (default: 30.0)
        """
        self.video_sources = video_sources
        self.video_prefilter = video_prefilter
        self.ai_service = ai_service
        self.feedback_service = feedback_service
        self.content_style = content_style
        self.content_filter = content_filter
        self.max_videos_per_phrase = max_videos_per_phrase
        self.evaluation_context_seconds = evaluation_context_seconds

        logger.info(
            f"[VideoSearchService] Initialized with {len(video_sources)} sources, "
            f"max_videos={max_videos_per_phrase}"
        )

    def set_content_style(self, content_style: Optional["ContentStyle"]) -> None:
        """Update the content style for evaluation.

        Args:
            content_style: New content style to use for evaluations
        """
        self.content_style = content_style

    async def search_youtube_videos(self, phrase: str) -> list["VideoResult"]:
        """Search video sources for videos matching the phrase.

        Currently searches all configured sources and combines results.

        Args:
            phrase: Search phrase to find videos for

        Returns:
            List of VideoResult objects from all sources
        """
        if not phrase or not phrase.strip():
            logger.warning("Empty search phrase provided")
            return []

        all_results: list["VideoResult"] = []

        for source in self.video_sources:
            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, source.search_videos, phrase)
                all_results.extend(results)
                logger.debug(f"Source '{source.get_source_name()}' returned {len(results)} videos")
            except Exception as e:
                logger.warning(f"Search failed for source '{source.get_source_name()}': {e}")
                continue

        logger.info(f"Total videos from all sources: {len(all_results)}")
        return all_results

    async def search_with_fallback(
        self, need: "BRollNeed", min_results: int = 5
    ) -> list["VideoResult"]:
        """Search for videos with fallback to alternate search phrases.

        Q2 ENHANCEMENT: Uses enhanced B-roll need metadata to search with
        primary phrase first, then alternates if insufficient results.
        Also filters out videos with negative keywords.

        Args:
            need: BRollNeed with primary search and optional alternates
            min_results: Minimum results before trying alternates (default: 5)

        Returns:
            List of VideoResult objects, filtered by negative keywords
        """
        # Search with primary phrase first
        results = await self.search_youtube_videos(need.search_phrase)
        logger.info(f"Primary search '{need.search_phrase}': {len(results)} results")

        # If insufficient results and we have alternates, try them
        if len(results) < min_results and need.alternate_searches:
            logger.info(
                f"Insufficient results ({len(results)} < {min_results}), "
                f"trying {len(need.alternate_searches)} alternate searches"
            )

            for i, alt_phrase in enumerate(need.alternate_searches):
                if len(results) >= min_results * 2:
                    logger.info(f"Sufficient results ({len(results)}), stopping alternates")
                    break

                alt_results = await self.search_youtube_videos(alt_phrase)
                logger.info(f"Alternate search {i+1} '{alt_phrase}': {len(alt_results)} results")

                # Add only new results (deduplicate by video_id)
                existing_ids = {r.video_id for r in results}
                new_results = [r for r in alt_results if r.video_id not in existing_ids]
                results.extend(new_results)

            logger.info(f"Total after fallback searches: {len(results)} results")

        # Q2 ENHANCEMENT: Filter by negative keywords
        if need.negative_keywords and results:
            original_count = len(results)
            results = self.ai_service.filter_by_negative_keywords(
                results, need.negative_keywords
            )
            if len(results) < original_count:
                logger.info(
                    f"Negative keyword filter: {original_count} -> {len(results)} videos"
                )

        return results

    async def evaluate_videos(
        self, phrase: str, videos: list["VideoResult"]
    ) -> list["ScoredVideo"]:
        """Evaluate videos using Gemini AI (legacy method).

        NOTE: For Q4 context-aware evaluation, use evaluate_videos_enhanced instead.

        Feature 1 & 3: Now includes content_style and feedback_context for better selection.

        Args:
            phrase: Search phrase for context
            videos: List of videos to evaluate

        Returns:
            List of scored videos, limited to max_videos_per_phrase
        """
        if not videos:
            logger.info(f"No videos to evaluate for phrase: {phrase}")
            return []

        # Feature 3: Get feedback context
        feedback_context = ""
        if self.feedback_service:
            feedback_context = self.feedback_service.get_prompt_additions(phrase)

        loop = asyncio.get_event_loop()
        scored_videos = await loop.run_in_executor(
            None,
            lambda: self.ai_service.evaluate_videos(
                phrase,
                videos,
                self.content_filter,
                content_style=self.content_style,  # Feature 1
                feedback_context=feedback_context,  # Feature 3
            )
        )

        limited_videos = scored_videos[:self.max_videos_per_phrase]
        return limited_videos

    async def evaluate_videos_enhanced(
        self,
        need: "BRollNeed",
        videos: list["VideoResult"],
        transcript_result: Optional["TranscriptResult"] = None,
        user_preferences: Optional["UserPreferences"] = None,
    ) -> list["ScoredVideo"]:
        """Evaluate videos using context-aware AI scoring (Q4 improvement).

        This enhanced evaluation method passes full context to the AI:
        - Transcript segment around the B-roll timestamp
        - B-roll metadata (visual_style, time_of_day, movement, etc.)
        - User preferences (style, content to avoid, mood)

        This results in more relevant scoring compared to the legacy evaluate_videos method.

        Args:
            need: BRollNeed with search phrase, timestamp, and enhanced metadata
            videos: List of video results to evaluate
            transcript_result: Optional TranscriptResult to extract context segment
            user_preferences: Optional UserPreferences for style customization

        Returns:
            List of scored videos, limited to max_videos_per_phrase
        """
        if not videos:
            logger.info(f"No videos to evaluate for phrase: {need.search_phrase}")
            return []

        # Q4 IMPROVEMENT: Extract transcript segment around the B-roll timestamp
        transcript_segment = ""
        if transcript_result:
            # Get transcript text around the B-roll timestamp
            transcript_segment = transcript_result.get_text_around_timestamp(
                need.timestamp, self.evaluation_context_seconds
            )
            if transcript_segment:
                logger.debug(
                    f"Evaluation context for '{need.search_phrase}' at {need.timestamp:.1f}s: "
                    f"'{transcript_segment[:100]}...'"
                )

        # Feature 3: Get feedback context
        feedback_context = ""
        if self.feedback_service:
            feedback_context = self.feedback_service.get_prompt_additions(need.search_phrase)

        loop = asyncio.get_event_loop()
        # Q2/Q4 IMPROVEMENT: Pass full context to evaluation for better scoring
        # BRollNeed includes enhanced metadata: visual_style, time_of_day, movement, negative_keywords
        # Feature 1 & 3: Also pass content_style and feedback_context
        scored_videos = await loop.run_in_executor(
            None,
            lambda: self.ai_service.evaluate_videos(
                search_phrase=need.search_phrase,
                video_results=videos,
                content_filter=self.content_filter,
                transcript_segment=transcript_segment,
                broll_need=need,  # Q2: Passes visual_style, movement, negative_keywords
                content_style=self.content_style,  # Feature 1
                feedback_context=feedback_context,  # Feature 3
            ),
        )

        limited_videos = scored_videos[:self.max_videos_per_phrase]
        return limited_videos
