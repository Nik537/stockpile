"""Video-related data models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoResult:
    """Represents a video search result from any source.

    Supports YouTube, Pexels, Pixabay, and other video sources.
    The source and license fields enable tracking where footage came from
    and what licensing restrictions apply.
    """

    video_id: str
    title: str
    url: str
    duration: int  # in seconds
    description: Optional[str] = None
    # Multi-source tracking (Q3 improvement)
    source: str = "youtube"  # youtube, pexels, pixabay, etc.
    license: Optional[str] = None  # CC0, YouTube Standard, etc.
    # Direct download URL for stock footage (Pexels/Pixabay provide direct links)
    download_url: Optional[str] = None
    # Metadata for pre-filtering (S5 improvement)
    view_count: Optional[int] = None  # Number of views
    like_count: Optional[int] = None  # Number of likes
    channel: Optional[str] = None  # Channel/creator name


@dataclass
class ScoredVideo:
    """Represents a video with AI evaluation score."""

    video_id: str
    score: int  # 1-10 rating from AI evaluator
    video_result: VideoResult
