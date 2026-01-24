"""Video sources package for multi-platform B-roll acquisition."""

from services.video_sources.base import VideoSource
from services.video_sources.youtube import YouTubeVideoSource

__all__ = ["VideoSource", "YouTubeVideoSource"]
