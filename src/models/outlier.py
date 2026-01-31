"""Data models for YouTube outlier finder."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OutlierVideo:
    """A video that significantly outperforms its channel's average."""

    video_id: str
    title: str
    url: str
    thumbnail_url: str
    view_count: int
    outlier_score: float  # video_views / channel_avg
    channel_average_views: float
    channel_name: str
    upload_date: str  # YYYYMMDD format
    outlier_tier: str  # "solid" (3-5x), "strong" (5-10x), "exceptional" (10x+)

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
            "outliers": [
                {
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
                }
                for o in self.outliers
            ],
        }
