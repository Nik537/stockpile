"""Data models for feedback loop system.

Feature 3: Feedback Loop
Stores user rejections and learns from them to improve future selections.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


@dataclass
class ContentFeedback:
    """User feedback on a selected image or video.

    Represents a single rejection or approval of content,
    enabling the system to learn from user preferences.
    """

    content_id: str
    """Unique identifier for the content (image_id or video_id)."""

    content_type: str
    """Type of content: 'image' or 'video'."""

    source: str
    """Source of the content: pexels, pixabay, google, youtube."""

    search_phrase: str
    """The search phrase used to find this content."""

    was_rejected: bool
    """Whether the user rejected this content."""

    rejection_reason: Optional[str] = None
    """Optional reason for rejection: 'too_generic', 'wrong_subject', 'poor_quality', etc."""

    timestamp: float = 0.0
    """Timestamp in source video where this content was used."""

    title: str = ""
    """Title of the content for pattern matching."""

    description: str = ""
    """Description of the content for pattern analysis."""

    feedback_time: float = field(default_factory=time.time)
    """When this feedback was recorded."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "source": self.source,
            "search_phrase": self.search_phrase,
            "was_rejected": self.was_rejected,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp,
            "title": self.title,
            "description": self.description,
            "feedback_time": self.feedback_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentFeedback":
        """Create ContentFeedback from dictionary."""
        return cls(
            content_id=data.get("content_id", ""),
            content_type=data.get("content_type", "unknown"),
            source=data.get("source", "unknown"),
            search_phrase=data.get("search_phrase", ""),
            was_rejected=data.get("was_rejected", True),
            rejection_reason=data.get("rejection_reason"),
            timestamp=data.get("timestamp", 0.0),
            title=data.get("title", ""),
            description=data.get("description", ""),
            feedback_time=data.get("feedback_time", time.time()),
        )


@dataclass
class FeedbackStore:
    """Persistent storage for feedback data.

    Aggregates user feedback to identify patterns and improve
    future content selection.
    """

    rejections: List[ContentFeedback] = field(default_factory=list)
    """List of all recorded rejections."""

    approvals: List[ContentFeedback] = field(default_factory=list)
    """List of explicitly approved content (optional positive feedback)."""

    # Learned patterns (aggregated from rejections)
    rejected_sources: Dict[str, int] = field(default_factory=dict)
    """Source -> rejection count mapping."""

    rejected_keywords: Dict[str, int] = field(default_factory=dict)
    """Keyword -> rejection count mapping."""

    rejected_content_ids: Dict[str, int] = field(default_factory=dict)
    """Content ID -> rejection count (for exact matches)."""

    successful_patterns: List[str] = field(default_factory=list)
    """Patterns that have worked well (from approvals)."""

    # Statistics
    total_rejections: int = 0
    total_approvals: int = 0
    last_updated: float = field(default_factory=time.time)

    def add_rejection(self, feedback: ContentFeedback) -> None:
        """Record a user rejection and update learned patterns.

        Args:
            feedback: ContentFeedback with was_rejected=True
        """
        self.rejections.append(feedback)
        self.total_rejections += 1
        self.last_updated = time.time()

        # Update source rejection count
        source = feedback.source.lower()
        self.rejected_sources[source] = self.rejected_sources.get(source, 0) + 1

        # Update keyword rejection counts (extract from title/search phrase)
        keywords = self._extract_keywords(feedback.title, feedback.search_phrase)
        for keyword in keywords:
            self.rejected_keywords[keyword] = self.rejected_keywords.get(keyword, 0) + 1

        # Track exact content ID
        self.rejected_content_ids[feedback.content_id] = (
            self.rejected_content_ids.get(feedback.content_id, 0) + 1
        )

    def add_approval(self, feedback: ContentFeedback) -> None:
        """Record an explicit user approval.

        Args:
            feedback: ContentFeedback with was_rejected=False
        """
        self.approvals.append(feedback)
        self.total_approvals += 1
        self.last_updated = time.time()

        # Track successful patterns
        keywords = self._extract_keywords(feedback.title, feedback.search_phrase)
        for keyword in keywords:
            pattern = f"{keyword}:{feedback.source}"
            if pattern not in self.successful_patterns:
                self.successful_patterns.append(pattern)

    def _extract_keywords(self, title: str, search_phrase: str) -> List[str]:
        """Extract meaningful keywords from title and search phrase.

        Args:
            title: Content title
            search_phrase: Search phrase used

        Returns:
            List of lowercase keywords
        """
        # Combine and normalize
        combined = f"{title} {search_phrase}".lower()

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are",
            "been", "be", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "stock", "footage", "video", "image", "photo", "picture",
        }

        # Split and filter
        words = combined.split()
        keywords = [
            word.strip(".,!?\"'()[]{}") for word in words
            if len(word) > 2 and word not in stop_words
        ]

        return keywords[:10]  # Limit to top 10 keywords

    def get_rejection_score(self, content_id: str, source: str, title: str) -> float:
        """Calculate a rejection likelihood score for content.

        Higher scores indicate higher likelihood of rejection based on history.

        Args:
            content_id: Content identifier
            source: Content source
            title: Content title

        Returns:
            Score from 0.0 (unlikely rejection) to 1.0 (likely rejection)
        """
        score = 0.0

        # Check if exact content was rejected before
        if content_id in self.rejected_content_ids:
            score += 0.5 * min(self.rejected_content_ids[content_id], 3) / 3

        # Check source rejection rate
        source_lower = source.lower()
        if source_lower in self.rejected_sources:
            total_from_source = self.rejected_sources[source_lower]
            if total_from_source >= 3:
                score += 0.2
            elif total_from_source >= 5:
                score += 0.3

        # Check keyword patterns
        keywords = self._extract_keywords(title, "")
        rejected_keyword_count = sum(
            1 for kw in keywords if kw in self.rejected_keywords
        )
        if rejected_keyword_count > 0:
            score += 0.1 * min(rejected_keyword_count, 3) / 3

        return min(score, 1.0)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rejections": [r.to_dict() for r in self.rejections],
            "approvals": [a.to_dict() for a in self.approvals],
            "rejected_sources": self.rejected_sources,
            "rejected_keywords": self.rejected_keywords,
            "rejected_content_ids": self.rejected_content_ids,
            "successful_patterns": self.successful_patterns,
            "total_rejections": self.total_rejections,
            "total_approvals": self.total_approvals,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackStore":
        """Create FeedbackStore from dictionary."""
        store = cls()
        store.rejections = [
            ContentFeedback.from_dict(r) for r in data.get("rejections", [])
        ]
        store.approvals = [
            ContentFeedback.from_dict(a) for a in data.get("approvals", [])
        ]
        store.rejected_sources = data.get("rejected_sources", {})
        store.rejected_keywords = data.get("rejected_keywords", {})
        store.rejected_content_ids = data.get("rejected_content_ids", {})
        store.successful_patterns = data.get("successful_patterns", [])
        store.total_rejections = data.get("total_rejections", 0)
        store.total_approvals = data.get("total_approvals", 0)
        store.last_updated = data.get("last_updated", time.time())
        return store


@dataclass
class RejectionFilter:
    """Filters to apply based on feedback history.

    Generated by FeedbackService.get_rejection_filter() for use in
    content selection.
    """

    avoid_sources: List[str] = field(default_factory=list)
    """Sources to deprioritize (e.g., ['pixabay'] if often rejected)."""

    avoid_keywords: List[str] = field(default_factory=list)
    """Keywords to avoid in content selection."""

    boost_sources: List[str] = field(default_factory=list)
    """Sources to prioritize (from successful patterns)."""

    learned_preferences: str = ""
    """Human-readable summary of learned preferences for AI prompts."""

    rejection_threshold: float = 0.5
    """Content with rejection score above this should be filtered."""

    def to_prompt_context(self) -> str:
        """Generate prompt context for AI selection.

        Returns:
            Formatted string for injection into AI prompts
        """
        parts = []

        if self.avoid_sources:
            parts.append(f"DEPRIORITIZE sources: {', '.join(self.avoid_sources)}")

        if self.boost_sources:
            parts.append(f"PREFER sources: {', '.join(self.boost_sources)}")

        if self.avoid_keywords:
            parts.append(f"AVOID content with: {', '.join(self.avoid_keywords[:10])}")

        if self.learned_preferences:
            parts.append(f"User preference: {self.learned_preferences}")

        if not parts:
            return ""

        return "FEEDBACK HISTORY:\n" + "\n".join(parts)
