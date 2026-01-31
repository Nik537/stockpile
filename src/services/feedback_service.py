"""Feedback service for storing and applying user feedback.

Feature 3: Feedback Loop
Learns from user rejections to improve future content selection.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from models.feedback import ContentFeedback, FeedbackStore, RejectionFilter
from models.image import ImageResult
from models.video import VideoResult

logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for storing and applying user feedback.

    Maintains a persistent feedback store that learns from user rejections
    to improve future content selection.
    """

    def __init__(self, feedback_dir: str, feedback_filename: str = "feedback.json"):
        """Initialize the feedback service.

        Args:
            feedback_dir: Directory to store feedback file
            feedback_filename: Name of the feedback JSON file
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / feedback_filename
        self.store = self._load_feedback()

        logger.info(
            f"[FeedbackService] Initialized: {len(self.store.rejections)} rejections, "
            f"{len(self.store.approvals)} approvals loaded from {self.feedback_file}"
        )

    def _load_feedback(self) -> FeedbackStore:
        """Load feedback store from disk.

        Returns:
            FeedbackStore, either loaded from file or new empty store
        """
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, "r") as f:
                    data = json.load(f)
                return FeedbackStore.from_dict(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[FeedbackService] Failed to load feedback: {e}")

        return FeedbackStore()

    def _save_feedback(self) -> None:
        """Save feedback store to disk."""
        try:
            with open(self.feedback_file, "w") as f:
                json.dump(self.store.to_dict(), f, indent=2)
            logger.debug(f"[FeedbackService] Saved feedback to {self.feedback_file}")
        except Exception as e:
            logger.error(f"[FeedbackService] Failed to save feedback: {e}")

    def record_rejection(
        self,
        content_id: str,
        content_type: str,
        source: str,
        search_phrase: str,
        reason: Optional[str] = None,
        timestamp: float = 0.0,
        title: str = "",
        description: str = "",
    ) -> None:
        """Record a user rejection.

        Args:
            content_id: Unique content identifier
            content_type: 'image' or 'video'
            source: Content source (pexels, pixabay, youtube, google)
            search_phrase: Search phrase used to find this content
            reason: Optional rejection reason
            timestamp: Timestamp in source video
            title: Content title
            description: Content description
        """
        feedback = ContentFeedback(
            content_id=content_id,
            content_type=content_type,
            source=source,
            search_phrase=search_phrase,
            was_rejected=True,
            rejection_reason=reason,
            timestamp=timestamp,
            title=title,
            description=description,
        )

        self.store.add_rejection(feedback)
        self._save_feedback()

        logger.info(
            f"[FeedbackService] Recorded rejection: {content_type} from {source}, "
            f"reason={reason or 'unspecified'}"
        )

    def record_approval(
        self,
        content_id: str,
        content_type: str,
        source: str,
        search_phrase: str,
        timestamp: float = 0.0,
        title: str = "",
        description: str = "",
    ) -> None:
        """Record an explicit user approval (positive feedback).

        Args:
            content_id: Unique content identifier
            content_type: 'image' or 'video'
            source: Content source
            search_phrase: Search phrase used
            timestamp: Timestamp in source video
            title: Content title
            description: Content description
        """
        feedback = ContentFeedback(
            content_id=content_id,
            content_type=content_type,
            source=source,
            search_phrase=search_phrase,
            was_rejected=False,
            timestamp=timestamp,
            title=title,
            description=description,
        )

        self.store.add_approval(feedback)
        self._save_feedback()

        logger.debug(f"[FeedbackService] Recorded approval: {content_type} from {source}")

    def get_rejection_filter(self, search_phrase: str = "") -> RejectionFilter:
        """Get filters based on past rejections for this search type.

        Args:
            search_phrase: Optional search phrase for context-specific filtering

        Returns:
            RejectionFilter with sources/keywords to avoid and boost
        """
        filter_result = RejectionFilter()

        # Identify sources to avoid (rejected 3+ times)
        for source, count in self.store.rejected_sources.items():
            if count >= 3:
                filter_result.avoid_sources.append(source)

        # Identify sources to boost (from successful patterns)
        for pattern in self.store.successful_patterns:
            if ":" in pattern:
                source = pattern.split(":")[-1]
                if source not in filter_result.boost_sources:
                    filter_result.boost_sources.append(source)

        # Get top rejected keywords (rejected 2+ times)
        sorted_keywords = sorted(
            self.store.rejected_keywords.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        filter_result.avoid_keywords = [
            kw for kw, count in sorted_keywords[:15] if count >= 2
        ]

        # Generate learned preferences summary
        preferences = []
        if filter_result.avoid_sources:
            preferences.append(f"Avoid {', '.join(filter_result.avoid_sources)} sources")
        if filter_result.boost_sources:
            preferences.append(f"Prefer {', '.join(filter_result.boost_sources)} sources")
        if self.store.total_rejections > 5:
            # Summarize common rejection reasons
            reasons = {}
            for rejection in self.store.rejections[-20:]:  # Last 20 rejections
                if rejection.rejection_reason:
                    reasons[rejection.rejection_reason] = (
                        reasons.get(rejection.rejection_reason, 0) + 1
                    )
            if reasons:
                top_reason = max(reasons, key=reasons.get)
                preferences.append(f"Common rejection: {top_reason}")

        filter_result.learned_preferences = "; ".join(preferences)

        return filter_result

    def apply_to_image_candidates(
        self,
        candidates: List[ImageResult],
        search_phrase: str,
    ) -> List[ImageResult]:
        """Filter and reorder image candidates based on feedback history.

        Args:
            candidates: List of ImageResult objects
            search_phrase: The search phrase used

        Returns:
            Filtered and reordered list with high-rejection items deprioritized
        """
        if not candidates or self.store.total_rejections == 0:
            return candidates

        # Score each candidate by rejection likelihood
        scored_candidates = []
        for candidate in candidates:
            rejection_score = self.store.get_rejection_score(
                content_id=candidate.image_id,
                source=candidate.source,
                title=candidate.title,
            )

            # Check if exact ID was rejected before
            if candidate.image_id in self.store.rejected_content_ids:
                rejection_score = 1.0  # Definitely skip this one

            scored_candidates.append((candidate, rejection_score))

        # Sort by rejection score (lower is better)
        scored_candidates.sort(key=lambda x: x[1])

        # Filter out high-rejection candidates and return
        filtered = [
            candidate for candidate, score in scored_candidates
            if score < 0.8  # Allow candidates with <80% rejection likelihood
        ]

        if len(filtered) < len(candidates):
            logger.info(
                f"[FeedbackService] Filtered {len(candidates) - len(filtered)} "
                f"high-rejection images"
            )

        return filtered if filtered else candidates[:3]  # Always keep some candidates

    def apply_to_video_candidates(
        self,
        candidates: List[VideoResult],
        search_phrase: str,
    ) -> List[VideoResult]:
        """Filter and reorder video candidates based on feedback history.

        Args:
            candidates: List of VideoResult objects
            search_phrase: The search phrase used

        Returns:
            Filtered and reordered list with high-rejection items deprioritized
        """
        if not candidates or self.store.total_rejections == 0:
            return candidates

        # Score each candidate by rejection likelihood
        scored_candidates = []
        for candidate in candidates:
            rejection_score = self.store.get_rejection_score(
                content_id=candidate.video_id,
                source=candidate.source if hasattr(candidate, 'source') else 'youtube',
                title=candidate.title,
            )

            # Check if exact ID was rejected before
            if candidate.video_id in self.store.rejected_content_ids:
                rejection_score = 1.0

            scored_candidates.append((candidate, rejection_score))

        # Sort by rejection score (lower is better)
        scored_candidates.sort(key=lambda x: x[1])

        # Filter out high-rejection candidates
        filtered = [
            candidate for candidate, score in scored_candidates
            if score < 0.8
        ]

        if len(filtered) < len(candidates):
            logger.info(
                f"[FeedbackService] Filtered {len(candidates) - len(filtered)} "
                f"high-rejection videos"
            )

        return filtered if filtered else candidates[:3]

    def get_prompt_additions(self, search_phrase: str = "") -> str:
        """Get additional prompt context from feedback history.

        Args:
            search_phrase: Optional search phrase for context

        Returns:
            Formatted string to add to AI prompts
        """
        if self.store.total_rejections == 0:
            return ""

        rejection_filter = self.get_rejection_filter(search_phrase)
        return rejection_filter.to_prompt_context()

    def get_statistics(self) -> dict:
        """Get feedback statistics summary.

        Returns:
            Dictionary with feedback statistics
        """
        return {
            "total_rejections": self.store.total_rejections,
            "total_approvals": self.store.total_approvals,
            "unique_rejected_sources": len(self.store.rejected_sources),
            "unique_rejected_keywords": len(self.store.rejected_keywords),
            "successful_patterns": len(self.store.successful_patterns),
            "top_rejected_sources": dict(
                sorted(
                    self.store.rejected_sources.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ),
            "top_rejected_keywords": dict(
                sorted(
                    self.store.rejected_keywords.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }

    def clear_feedback(self) -> None:
        """Clear all feedback data (for testing or reset)."""
        self.store = FeedbackStore()
        self._save_feedback()
        logger.info("[FeedbackService] Cleared all feedback data")
