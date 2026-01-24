"""Video pre-filtering service for filtering YouTube results before download.

This module provides aggressive pre-filtering of YouTube search results based on
metadata to avoid wasting bandwidth on poor quality or irrelevant videos.

Filters include:
- Minimum view count threshold
- Maximum duration limit
- Blocked title keywords
- Creative Commons license preference (when available)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from models.video import VideoResult
from utils.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for video filtering."""

    min_view_count: int = 1000
    max_prefilter_duration: int = 600  # 10 minutes
    blocked_title_keywords: list[str] = field(
        default_factory=lambda: ["compilation", "top 10", "reaction", "review"]
    )
    prefer_creative_commons: bool = True

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "FilterConfig":
        """Create FilterConfig from application config or environment variables."""
        if config is None:
            config = load_config()

        blocked_keywords_str = config.get(
            "blocked_title_keywords",
            os.getenv("BLOCKED_TITLE_KEYWORDS", "compilation,top 10,reaction,review"),
        )

        if isinstance(blocked_keywords_str, str):
            blocked_keywords = [kw.strip() for kw in blocked_keywords_str.split(",") if kw.strip()]
        else:
            blocked_keywords = blocked_keywords_str

        return cls(
            min_view_count=config.get(
                "min_view_count", int(os.getenv("MIN_VIEW_COUNT", "1000"))
            ),
            max_prefilter_duration=config.get(
                "max_prefilter_duration",
                int(os.getenv("MAX_PREFILTER_DURATION", "600")),
            ),
            blocked_title_keywords=blocked_keywords,
            prefer_creative_commons=config.get(
                "prefer_creative_commons",
                os.getenv("PREFER_CREATIVE_COMMONS", "true").lower() == "true",
            ),
        )


@dataclass
class FilterStats:
    """Statistics from filtering a batch of videos."""

    total_input: int = 0
    total_passed: int = 0
    total_filtered: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def filter_rate(self) -> float:
        """Return the percentage of videos that were filtered out."""
        if self.total_input == 0:
            return 0.0
        return (self.total_filtered / self.total_input) * 100

    def to_dict(self) -> dict:
        """Convert stats to dictionary for reporting."""
        return {
            "total_input": self.total_input,
            "total_passed": self.total_passed,
            "total_filtered": self.total_filtered,
            "filter_rate_percent": round(self.filter_rate, 1),
            "reasons": dict(self.reasons),
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Filtered {self.total_filtered}/{self.total_input} videos "
            f"({self.filter_rate:.1f}%), {self.total_passed} passed"
        )


def should_download(
    video: VideoResult, filter_config: Optional[FilterConfig] = None
) -> tuple[bool, str]:
    """
    Determine if a video should be downloaded based on metadata.

    Applies a series of filter rules to the video metadata to determine
    if it's worth downloading for B-roll extraction.

    Args:
        video: VideoResult object with metadata
        filter_config: Optional filter configuration (uses defaults if not provided)

    Returns:
        Tuple of (should_download, reason):
        - should_download: True if video passes all filters
        - reason: Human-readable explanation of the decision
    """
    if filter_config is None:
        filter_config = FilterConfig.from_config()

    # Rule 1: Check minimum view count
    # Low view count often indicates poor quality content
    if hasattr(video, "view_count") and video.view_count is not None:
        if video.view_count < filter_config.min_view_count:
            return (
                False,
                f"Low view count: {video.view_count:,} < {filter_config.min_view_count:,}",
            )

    # Rule 2: Check duration limit
    # Very long videos are less likely to have good B-roll segments
    if video.duration > filter_config.max_prefilter_duration:
        return (
            False,
            f"Too long: {video.duration}s > {filter_config.max_prefilter_duration}s",
        )

    # Rule 3: Check for blocked keywords in title
    # Certain video types are rarely good B-roll sources
    title_lower = video.title.lower()
    for keyword in filter_config.blocked_title_keywords:
        keyword_lower = keyword.strip().lower()
        if keyword_lower and keyword_lower in title_lower:
            return False, f"Blocked keyword in title: '{keyword}'"

    # Rule 4: Creative Commons preference (boost, not filter)
    # This is handled at ranking stage, not filtering
    # Videos with CC license are preferred but non-CC isn't rejected

    return True, "Passed all filters"


def filter_videos(
    videos: list[VideoResult],
    filter_config: Optional[FilterConfig] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> tuple[list[VideoResult], FilterStats]:
    """
    Filter a list of videos based on metadata, logging reasons for skipped videos.

    This is the main entry point for pre-filtering YouTube search results
    before downloading.

    Args:
        videos: List of VideoResult objects to filter
        filter_config: Optional filter configuration (uses defaults if not provided)
        log_callback: Optional callback for logging filtered videos

    Returns:
        Tuple of (filtered_videos, stats):
        - filtered_videos: List of videos that passed all filters
        - stats: FilterStats object with filtering statistics
    """
    if filter_config is None:
        filter_config = FilterConfig.from_config()

    filtered = []
    stats = FilterStats(total_input=len(videos))

    for video in videos:
        should_dl, reason = should_download(video, filter_config)

        if should_dl:
            filtered.append(video)
            stats.total_passed += 1
        else:
            stats.total_filtered += 1
            # Track reason counts
            stats.reasons[reason] = stats.reasons.get(reason, 0) + 1

            if log_callback:
                log_callback(f"Skipping '{video.title}': {reason}")
            else:
                logger.debug(f"Pre-filter: Skipping '{video.title}': {reason}")

    # Log summary
    if stats.total_filtered > 0:
        logger.info(f"Pre-filter: {stats}")

    return filtered, stats


class VideoPreFilter:
    """
    Service class for video pre-filtering with cumulative statistics tracking.

    Use this class when you need to track filtering statistics across
    multiple batches of videos (e.g., across all B-roll needs in a project).
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize the pre-filter service.

        Args:
            config: Optional application configuration dictionary
        """
        self.filter_config = FilterConfig.from_config(config)
        self.cumulative_stats = FilterStats()

    def filter(
        self,
        videos: list[VideoResult],
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> list[VideoResult]:
        """
        Filter videos and accumulate statistics.

        Args:
            videos: List of videos to filter
            log_callback: Optional callback for logging

        Returns:
            List of videos that passed all filters
        """
        filtered, stats = filter_videos(videos, self.filter_config, log_callback)

        # Accumulate stats
        self.cumulative_stats.total_input += stats.total_input
        self.cumulative_stats.total_passed += stats.total_passed
        self.cumulative_stats.total_filtered += stats.total_filtered
        for reason, count in stats.reasons.items():
            self.cumulative_stats.reasons[reason] = (
                self.cumulative_stats.reasons.get(reason, 0) + count
            )

        return filtered

    def get_stats(self) -> FilterStats:
        """Get cumulative filtering statistics."""
        return self.cumulative_stats

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self.cumulative_stats = FilterStats()

    def get_report(self) -> str:
        """Generate a human-readable report of filtering activity."""
        stats = self.cumulative_stats

        if stats.total_input == 0:
            return "No videos processed by pre-filter."

        lines = [
            "=== Pre-Filter Report ===",
            f"Total videos processed: {stats.total_input}",
            f"Videos passed: {stats.total_passed} ({100 - stats.filter_rate:.1f}%)",
            f"Videos filtered: {stats.total_filtered} ({stats.filter_rate:.1f}%)",
            "",
        ]

        if stats.reasons:
            lines.append("Filter reasons:")
            # Sort by count descending
            sorted_reasons = sorted(stats.reasons.items(), key=lambda x: -x[1])
            for reason, count in sorted_reasons:
                lines.append(f"  - {reason}: {count}")

        return "\n".join(lines)
