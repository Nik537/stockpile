"""Data models for YouTube outlier finder."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OutlierVideo:
    """A video that significantly outperforms its channel's average."""

    # Core fields
    video_id: str
    title: str
    url: str
    thumbnail_url: str
    view_count: int
    outlier_score: float  # video_views / channel_avg (legacy, kept for compatibility)
    channel_average_views: float
    channel_name: str
    upload_date: str  # YYYYMMDD format
    outlier_tier: str  # "solid" (3-5x), "strong" (5-10x), "exceptional" (10x+)

    # Engagement metrics
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    engagement_rate: Optional[float] = None  # (likes + comments) / views * 100

    # Velocity metrics
    days_since_upload: Optional[int] = None
    views_per_day: Optional[float] = None
    velocity_score: Optional[float] = None  # views_per_day / channel_median_velocity

    # Composite scoring (multi-layer)
    composite_score: Optional[float] = None  # Weighted combination of all scores
    statistical_score: Optional[float] = None  # IQR-based score
    engagement_score: Optional[float] = None  # Normalized engagement score

    # Reddit integration
    found_on_reddit: bool = False
    reddit_score: Optional[int] = None
    reddit_subreddit: Optional[str] = None

    # Momentum tracking
    momentum_score: Optional[float] = None  # recent_velocity / historical_velocity
    is_trending: bool = False  # momentum > 1.5

    @classmethod
    def calculate_tier(cls, score: float) -> str:
        """Calculate the outlier tier based on score.

        Args:
            score: The outlier score (video_views / channel_avg)

        Returns:
            Tier string: "solid", "strong", or "exceptional"
        """
        if score >= 10.0:
            return "exceptional"
        elif score >= 5.0:
            return "strong"
        else:
            return "solid"


@dataclass
class ChannelStats:
    """Statistics for a YouTube channel."""

    channel_id: str
    channel_name: str
    average_views: float
    median_views: float
    total_videos_analyzed: int
    subscriber_count: Optional[int] = None

    # IQR-based statistics for outlier detection
    q1_views: Optional[float] = None  # 25th percentile
    q3_views: Optional[float] = None  # 75th percentile
    iqr_views: Optional[float] = None  # Q3 - Q1
    upper_bound: Optional[float] = None  # Q3 + 1.5 * IQR

    # Velocity statistics
    median_views_per_day: Optional[float] = None
    average_views_per_day: Optional[float] = None


@dataclass
class OutlierSearchResult:
    """Result of an outlier search operation."""

    outliers: List[OutlierVideo] = field(default_factory=list)
    channels_analyzed: int = 0
    total_videos_scanned: int = 0
    topic: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export.

        Returns:
            Dictionary representation of the search result
        """
        return {
            "topic": self.topic,
            "channels_analyzed": self.channels_analyzed,
            "total_videos_scanned": self.total_videos_scanned,
            "outliers_found": len(self.outliers),
            "outliers": [self._outlier_to_dict(o) for o in self.outliers],
        }

    @staticmethod
    def _outlier_to_dict(o: "OutlierVideo") -> dict:
        """Convert a single outlier to dictionary."""
        return {
            # Core fields
            "video_id": o.video_id,
            "title": o.title,
            "url": o.url,
            "thumbnail_url": o.thumbnail_url,
            "view_count": o.view_count,
            "outlier_score": round(o.outlier_score, 2),
            "channel_average_views": round(o.channel_average_views, 0),
            "channel_name": o.channel_name,
            "upload_date": o.upload_date,
            "outlier_tier": o.outlier_tier,
            # Engagement metrics
            "like_count": o.like_count,
            "comment_count": o.comment_count,
            "engagement_rate": round(o.engagement_rate, 2) if o.engagement_rate else None,
            # Velocity metrics
            "days_since_upload": o.days_since_upload,
            "views_per_day": round(o.views_per_day, 0) if o.views_per_day else None,
            "velocity_score": round(o.velocity_score, 2) if o.velocity_score else None,
            # Composite scoring
            "composite_score": round(o.composite_score, 2) if o.composite_score else None,
            "statistical_score": (
                round(o.statistical_score, 2) if o.statistical_score else None
            ),
            "engagement_score": (
                round(o.engagement_score, 2) if o.engagement_score else None
            ),
            # Reddit integration
            "found_on_reddit": o.found_on_reddit,
            "reddit_score": o.reddit_score,
            "reddit_subreddit": o.reddit_subreddit,
            # Momentum
            "momentum_score": (
                round(o.momentum_score, 2) if o.momentum_score else None
            ),
            "is_trending": o.is_trending,
        }
