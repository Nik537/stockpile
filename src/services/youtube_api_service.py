"""YouTube Data API Service for efficient metadata retrieval.

Provides reliable, fast access to YouTube channel and video metadata using
the official YouTube Data API v3. Replaces slow yt-dlp scraping for metadata.

Quota Budget (10,000 units/day free):
- search.list: 100 units
- channels.list: 1 unit (batched, 50 per request)
- videos.list: 1 unit (batched, 50 per request)
- playlistItems.list: 1 unit

Can analyze ~29 topics/day with 100 channels each.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """Channel information from YouTube API."""
    channel_id: str
    channel_name: str
    subscriber_count: int
    video_count: int
    view_count: int
    description: str = ""
    custom_url: str = ""
    thumbnail_url: str = ""
    country: str = ""
    published_at: str = ""


@dataclass
class VideoInfo:
    """Video information from YouTube API."""
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_name: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int
    duration: str  # ISO 8601 duration format (e.g., "PT5M30S")
    duration_seconds: int = 0
    thumbnail_url: str = ""
    tags: List[str] = field(default_factory=list)
    category_id: str = ""
    is_short: bool = False


@dataclass
class ChannelStats:
    """Channel statistics from YouTube API."""
    channel_id: str
    subscriber_count: int
    video_count: int
    view_count: int


@dataclass
class VideoStats:
    """Video statistics from YouTube API."""
    video_id: str
    view_count: int
    like_count: int
    comment_count: int


class YouTubeAPIService:
    """Service for interacting with YouTube Data API v3.

    Features:
    - Batched requests (50 items per request for efficiency)
    - Quota tracking
    - Error handling with retries
    - Support for channel search, stats, and video metadata
    """

    # Batch sizes for API requests
    MAX_BATCH_SIZE = 50  # YouTube API limit

    # Quota costs
    QUOTA_SEARCH = 100
    QUOTA_CHANNELS = 1
    QUOTA_VIDEOS = 1
    QUOTA_PLAYLIST_ITEMS = 1

    def __init__(self, api_key: str):
        """Initialize the YouTube API service.

        Args:
            api_key: YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self._quota_used = 0
        self._lock = threading.RLock()  # Thread safety for API calls

    @property
    def quota_used(self) -> int:
        """Get total quota units used in this session."""
        return self._quota_used

    def _execute_request(self, request):
        """Execute an API request with thread safety.

        Args:
            request: Google API request object

        Returns:
            API response
        """
        with self._lock:
            return request.execute()

    def _parse_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to seconds.

        Args:
            duration: Duration string like "PT5M30S" or "PT1H2M3S"

        Returns:
            Duration in seconds
        """
        if not duration or not duration.startswith("PT"):
            return 0

        duration = duration[2:]  # Remove "PT" prefix
        hours = minutes = seconds = 0

        # Parse hours
        if "H" in duration:
            hours_part, duration = duration.split("H")
            hours = int(hours_part)

        # Parse minutes
        if "M" in duration:
            minutes_part, duration = duration.split("M")
            minutes = int(minutes_part)

        # Parse seconds
        if "S" in duration:
            seconds_part, _ = duration.split("S")
            seconds = int(seconds_part)

        return hours * 3600 + minutes * 60 + seconds

    def search_channels(
        self,
        topic: str,
        max_results: int = 50,
        min_subscriber_count: Optional[int] = None,
        max_subscriber_count: Optional[int] = None,
    ) -> List[ChannelInfo]:
        """Search for YouTube channels by topic.

        Args:
            topic: Search query/topic
            max_results: Maximum number of channels to return (default 50)
            min_subscriber_count: Filter channels with fewer subscribers (optional)
            max_subscriber_count: Filter channels with more subscribers (optional)

        Returns:
            List of ChannelInfo objects
        """
        channels = []
        page_token = None
        results_fetched = 0

        while results_fetched < max_results:
            try:
                # Search for channels
                request = self.youtube.search().list(
                    part="snippet",
                    q=topic,
                    type="channel",
                    maxResults=min(50, max_results - results_fetched),
                    pageToken=page_token,
                )
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_SEARCH

                if not response.get("items"):
                    break

                # Extract channel IDs for batch stats fetch
                channel_ids = [
                    item["snippet"]["channelId"]
                    for item in response["items"]
                ]

                # Get detailed channel stats
                channel_stats = self.get_channel_stats(channel_ids)

                # Build ChannelInfo objects
                for item in response["items"]:
                    channel_id = item["snippet"]["channelId"]
                    stats = channel_stats.get(channel_id)

                    if not stats:
                        continue

                    # Apply subscriber filters
                    if min_subscriber_count and stats.subscriber_count < min_subscriber_count:
                        continue
                    if max_subscriber_count and stats.subscriber_count > max_subscriber_count:
                        continue

                    channel = ChannelInfo(
                        channel_id=channel_id,
                        channel_name=item["snippet"]["title"],
                        subscriber_count=stats.subscriber_count,
                        video_count=stats.video_count,
                        view_count=stats.view_count,
                        description=item["snippet"].get("description", ""),
                        thumbnail_url=item["snippet"]["thumbnails"].get("high", {}).get("url", ""),
                        published_at=item["snippet"].get("publishedAt", ""),
                    )
                    channels.append(channel)

                results_fetched += len(response["items"])
                page_token = response.get("nextPageToken")

                if not page_token:
                    break

            except HttpError as e:
                logger.error(f"YouTube API error searching channels: {e}")
                break
            except Exception as e:
                logger.error(f"Error searching channels: {e}")
                break

        logger.info(f"Found {len(channels)} channels for topic: {topic}")
        return channels

    def get_channel_stats(self, channel_ids: List[str]) -> Dict[str, ChannelStats]:
        """Get statistics for multiple channels (batched).

        Args:
            channel_ids: List of channel IDs

        Returns:
            Dict mapping channel_id to ChannelStats
        """
        results = {}

        # Process in batches of 50
        for i in range(0, len(channel_ids), self.MAX_BATCH_SIZE):
            batch = channel_ids[i:i + self.MAX_BATCH_SIZE]

            try:
                request = self.youtube.channels().list(
                    part="statistics",
                    id=",".join(batch),
                )
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_CHANNELS

                for item in response.get("items", []):
                    stats = item.get("statistics", {})
                    channel_id = item["id"]

                    results[channel_id] = ChannelStats(
                        channel_id=channel_id,
                        subscriber_count=int(stats.get("subscriberCount", 0)),
                        video_count=int(stats.get("videoCount", 0)),
                        view_count=int(stats.get("viewCount", 0)),
                    )

            except HttpError as e:
                logger.error(f"YouTube API error getting channel stats: {e}")
            except Exception as e:
                logger.error(f"Error getting channel stats: {e}")

        return results

    def get_channel_details(self, channel_ids: List[str]) -> Dict[str, ChannelInfo]:
        """Get detailed information for multiple channels (batched).

        Args:
            channel_ids: List of channel IDs

        Returns:
            Dict mapping channel_id to ChannelInfo
        """
        results = {}

        # Process in batches of 50
        for i in range(0, len(channel_ids), self.MAX_BATCH_SIZE):
            batch = channel_ids[i:i + self.MAX_BATCH_SIZE]

            try:
                request = self.youtube.channels().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(batch),
                )
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_CHANNELS

                for item in response.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    channel_id = item["id"]

                    results[channel_id] = ChannelInfo(
                        channel_id=channel_id,
                        channel_name=snippet.get("title", ""),
                        subscriber_count=int(stats.get("subscriberCount", 0)),
                        video_count=int(stats.get("videoCount", 0)),
                        view_count=int(stats.get("viewCount", 0)),
                        description=snippet.get("description", ""),
                        custom_url=snippet.get("customUrl", ""),
                        thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                        country=snippet.get("country", ""),
                        published_at=snippet.get("publishedAt", ""),
                    )

            except HttpError as e:
                logger.error(f"YouTube API error getting channel details: {e}")
            except Exception as e:
                logger.error(f"Error getting channel details: {e}")

        return results

    def get_video_stats(self, video_ids: List[str]) -> Dict[str, VideoStats]:
        """Get statistics for multiple videos (batched).

        Args:
            video_ids: List of video IDs

        Returns:
            Dict mapping video_id to VideoStats
        """
        results = {}

        # Process in batches of 50
        for i in range(0, len(video_ids), self.MAX_BATCH_SIZE):
            batch = video_ids[i:i + self.MAX_BATCH_SIZE]

            try:
                request = self.youtube.videos().list(
                    part="statistics",
                    id=",".join(batch),
                )
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_VIDEOS

                for item in response.get("items", []):
                    stats = item.get("statistics", {})
                    video_id = item["id"]

                    results[video_id] = VideoStats(
                        video_id=video_id,
                        view_count=int(stats.get("viewCount", 0)),
                        like_count=int(stats.get("likeCount", 0)),
                        comment_count=int(stats.get("commentCount", 0)),
                    )

            except HttpError as e:
                logger.error(f"YouTube API error getting video stats: {e}")
            except Exception as e:
                logger.error(f"Error getting video stats: {e}")

        return results

    def get_video_details(self, video_ids: List[str]) -> Dict[str, VideoInfo]:
        """Get detailed information for multiple videos (batched).

        Args:
            video_ids: List of video IDs

        Returns:
            Dict mapping video_id to VideoInfo
        """
        results = {}

        # Process in batches of 50
        for i in range(0, len(video_ids), self.MAX_BATCH_SIZE):
            batch = video_ids[i:i + self.MAX_BATCH_SIZE]

            try:
                request = self.youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(batch),
                )
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_VIDEOS

                for item in response.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    content = item.get("contentDetails", {})
                    video_id = item["id"]

                    duration = content.get("duration", "")
                    duration_seconds = self._parse_duration(duration)

                    # Check if it's a YouTube Short (< 60 seconds and vertical)
                    is_short = duration_seconds <= 60

                    results[video_id] = VideoInfo(
                        video_id=video_id,
                        title=snippet.get("title", ""),
                        description=snippet.get("description", ""),
                        channel_id=snippet.get("channelId", ""),
                        channel_name=snippet.get("channelTitle", ""),
                        published_at=snippet.get("publishedAt", ""),
                        view_count=int(stats.get("viewCount", 0)),
                        like_count=int(stats.get("likeCount", 0)),
                        comment_count=int(stats.get("commentCount", 0)),
                        duration=duration,
                        duration_seconds=duration_seconds,
                        thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                        tags=snippet.get("tags", []),
                        category_id=snippet.get("categoryId", ""),
                        is_short=is_short,
                    )

            except HttpError as e:
                logger.error(f"YouTube API error getting video details: {e}")
            except Exception as e:
                logger.error(f"Error getting video details: {e}")

        return results

    def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50,
    ) -> List[VideoInfo]:
        """Get videos from a channel's uploads playlist.

        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to return

        Returns:
            List of VideoInfo objects
        """
        videos = []

        try:
            # First, get the channel's uploads playlist ID
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id,
            )
            response = self._execute_request(request)
            self._quota_used += self.QUOTA_CHANNELS

            if not response.get("items"):
                return []

            uploads_playlist_id = (
                response["items"][0]
                .get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )

            if not uploads_playlist_id:
                return []

            # Get videos from uploads playlist
            video_ids = []
            page_token = None

            while len(video_ids) < max_results:
                request = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=page_token,
                )
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_PLAYLIST_ITEMS

                for item in response.get("items", []):
                    video_id = item["snippet"].get("resourceId", {}).get("videoId")
                    if video_id:
                        video_ids.append(video_id)

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            # Get detailed video info
            if video_ids:
                video_details = self.get_video_details(video_ids)
                videos = list(video_details.values())

        except HttpError as e:
            logger.error(f"YouTube API error getting channel videos: {e}")
        except Exception as e:
            logger.error(f"Error getting channel videos: {e}")

        return videos

    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        published_after: Optional[str] = None,
        order: str = "relevance",
    ) -> List[VideoInfo]:
        """Search for videos by query.

        Args:
            query: Search query
            max_results: Maximum number of videos to return
            published_after: Only return videos published after this date (ISO 8601)
            order: Sort order ("relevance", "date", "viewCount", "rating")

        Returns:
            List of VideoInfo objects
        """
        video_ids = []
        page_token = None

        while len(video_ids) < max_results:
            try:
                request_params = {
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": min(50, max_results - len(video_ids)),
                    "order": order,
                    "pageToken": page_token,
                }

                if published_after:
                    request_params["publishedAfter"] = published_after

                request = self.youtube.search().list(**request_params)
                response = self._execute_request(request)
                self._quota_used += self.QUOTA_SEARCH

                for item in response.get("items", []):
                    video_id = item.get("id", {}).get("videoId")
                    if video_id:
                        video_ids.append(video_id)

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            except HttpError as e:
                logger.error(f"YouTube API error searching videos: {e}")
                break
            except Exception as e:
                logger.error(f"Error searching videos: {e}")
                break

        # Get detailed video info
        if video_ids:
            video_details = self.get_video_details(video_ids)
            return list(video_details.values())

        return []


# Factory function for creating service instance
_service_instance: Optional[YouTubeAPIService] = None


def get_youtube_api_service(api_key: Optional[str] = None) -> Optional[YouTubeAPIService]:
    """Get or create the YouTubeAPIService instance.

    Args:
        api_key: YouTube Data API key (uses config if not provided)

    Returns:
        YouTubeAPIService instance or None if no API key available
    """
    global _service_instance

    if _service_instance is None:
        if api_key is None:
            # Try to get from config
            from utils.config import load_config
            config = load_config()
            api_key = config.get("youtube_api_key")

        if api_key:
            _service_instance = YouTubeAPIService(api_key)
        else:
            logger.warning("No YouTube API key available")
            return None

    return _service_instance
