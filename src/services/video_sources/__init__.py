"""Video sources package for multi-platform B-roll acquisition."""

from services.video_sources.base import VideoSource
from services.video_sources.youtube import YouTubeVideoSource
from services.video_sources.pexels import PexelsVideoSource
from services.video_sources.pixabay import PixabayVideoSource

__all__ = ["VideoSource", "YouTubeVideoSource", "PexelsVideoSource", "PixabayVideoSource"]
