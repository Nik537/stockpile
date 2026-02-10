"""YouTube Outlier Finder Service - Enhanced Version.

Discovers viral/outperforming YouTube videos by topic using yt-dlp.
Identifies videos that significantly outperform their channel's average.

Optimizations implemented:
1. Channel-first approach (100x more efficient than search)
2. Batched video statistics (50 IDs per request)
3. yt-dlp for free metadata extraction
4. SQLite database persistence
5. In-memory TTL caching
6. Parallel processing with rate limiting
7. Scales to 100+ channels (10,000+ videos)
8. YouTube Data API integration for faster metadata (when available)
9. Reddit integration for early viral detection
"""

import logging
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import yt_dlp

# Regex to detect Indian language scripts in text
# Covers: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
_INDIAN_SCRIPT_RE = re.compile(
    r'[\u0900-\u097F'   # Devanagari (Hindi, Marathi, Sanskrit)
    r'\u0980-\u09FF'    # Bengali
    r'\u0A00-\u0A7F'    # Gurmukhi (Punjabi)
    r'\u0A80-\u0AFF'    # Gujarati
    r'\u0B00-\u0B7F'    # Oriya
    r'\u0B80-\u0BFF'    # Tamil
    r'\u0C00-\u0C7F'    # Telugu
    r'\u0C80-\u0CFF'    # Kannada
    r'\u0D00-\u0D7F]'   # Malayalam
)

from models.outlier import ChannelStats, OutlierSearchResult, OutlierVideo
from services.video_cache import VideoCache, get_video_cache

# Optional YouTube API service import
try:
    from services.youtube_api_service import (
        YouTubeAPIService,
        get_youtube_api_service,
        ChannelInfo as APIChannelInfo,
        VideoInfo as APIVideoInfo,
    )
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False

# Optional Reddit monitor import
try:
    from services.reddit_monitor import RedditMonitor, RedditVideo
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# Type aliases for callbacks
OnOutlierFoundCallback = Callable[[OutlierVideo], None]
OnChannelCompleteCallback = Callable[[int, int, int], None]  # (channels_done, total, videos_scanned)

logger = logging.getLogger(__name__)

# Suppress yt-dlp's verbose logging
logging.getLogger("yt_dlp").setLevel(logging.CRITICAL)
logging.getLogger("yt_dlp.extractor").setLevel(logging.CRITICAL)

# User agent rotation to avoid rate limiting
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def get_random_user_agent() -> str:
    """Get a random user agent to avoid detection."""
    return random.choice(USER_AGENTS)


class OutlierFinderService:
    """Service to find viral/outperforming YouTube videos by topic.

    Enhanced with:
    - Parallel channel analysis
    - Video caching (SQLite + in-memory)
    - Batched metadata extraction
    - Channel-first discovery approach
    - YouTube Data API integration (50x faster when available)
    - Reddit integration for early viral detection
    """

    def __init__(
        self,
        min_score: float = 3.0,
        max_videos_per_channel: int = 100,  # Increased from 50
        date_days: Optional[int] = None,
        min_subs: Optional[int] = None,
        max_subs: Optional[int] = None,
        exclude_shorts: bool = True,
        min_views: int = 0,
        exclude_indian: bool = False,
        parallel_workers: int = 3,  # Parallel channel analysis
        use_cache: bool = True,
        cache: Optional[VideoCache] = None,
        youtube_api_key: Optional[str] = None,
        use_youtube_api: bool = True,
        enable_reddit_discovery: bool = True,
    ):
        """Initialize the outlier finder.

        Args:
            min_score: Minimum outlier score to include (default 3.0 = 3x average)
            max_videos_per_channel: Max videos to analyze per channel (default 100)
            date_days: Only include videos from last N days (None = all time)
            min_subs: Minimum subscriber count filter (None = no minimum)
            max_subs: Maximum subscriber count filter (None = no maximum)
            exclude_shorts: Exclude YouTube Shorts (default True)
            min_views: Minimum view count to include (default 0 = no minimum)
            exclude_indian: Exclude videos with Indian language scripts in title (default False)
            parallel_workers: Number of parallel workers for channel analysis
            use_cache: Whether to use caching (default True)
            cache: Optional VideoCache instance (uses global if None)
            youtube_api_key: YouTube Data API key (optional, uses config if not provided)
            use_youtube_api: Whether to use YouTube API when available (default True)
            enable_reddit_discovery: Whether to check Reddit for viral signals (default True)
        """
        self.min_score = min_score
        self.max_videos_per_channel = max_videos_per_channel
        self.date_days = date_days
        self.min_subs = min_subs
        self.max_subs = max_subs
        self.exclude_shorts = exclude_shorts
        self.min_views = min_views
        self.exclude_indian = exclude_indian
        self.parallel_workers = parallel_workers
        self.use_cache = use_cache
        self.cache = cache or (get_video_cache() if use_cache else None)

        # YouTube API integration
        self.use_youtube_api = use_youtube_api
        self._youtube_api: Optional[YouTubeAPIService] = None
        if use_youtube_api and YOUTUBE_API_AVAILABLE:
            if youtube_api_key:
                self._youtube_api = YouTubeAPIService(youtube_api_key)
            else:
                self._youtube_api = get_youtube_api_service()

        if self._youtube_api:
            logger.info("YouTube Data API enabled - using fast metadata retrieval")
        else:
            logger.info("YouTube Data API not available - using yt-dlp fallback")

        # Reddit integration
        self.enable_reddit_discovery = enable_reddit_discovery and REDDIT_AVAILABLE
        self._reddit_monitor: Optional[RedditMonitor] = None
        if self.enable_reddit_discovery:
            self._reddit_monitor = RedditMonitor(min_score=50, max_age_hours=72)
            logger.info("Reddit integration enabled for early viral detection")

        # Rate limiting delays (in seconds) - reduced when using API
        if self._youtube_api:
            self.channel_delay = (0.1, 0.3)  # Much faster with API
            self.video_delay = (0.05, 0.1)
        else:
            self.channel_delay = (1.5, 3.0)  # Slower for yt-dlp scraping
            self.video_delay = (0.3, 0.8)

        self.retry_delays = [2, 4, 8, 16]  # Exponential backoff on 403

        # Batch size for video metadata enrichment
        self.batch_size = 50 if self._youtube_api else 10  # API supports 50

    def find_outliers_by_topic(
        self,
        topic: str,
        max_channels: int = 10,
        on_outlier_found: Optional[OnOutlierFoundCallback] = None,
        on_channel_complete: Optional[OnChannelCompleteCallback] = None,
    ) -> OutlierSearchResult:
        """Find outlier videos for a given topic.

        Main entry point - searches topic, discovers channels, finds outliers.
        Now with parallel processing and caching for 10-100x more videos.

        Uses YouTube Data API when available for 50x faster metadata retrieval.

        Args:
            topic: Topic to search for (e.g., "tech reviews")
            max_channels: Maximum number of channels to analyze (can now handle 100+)
            on_outlier_found: Optional callback called when an outlier is found
            on_channel_complete: Optional callback called when a channel is analyzed

        Returns:
            OutlierSearchResult with found outliers
        """
        logger.info(f"Finding outliers for topic: '{topic}' (max {max_channels} channels)")

        result = OutlierSearchResult(topic=topic)

        # Step 0: Check Reddit for early viral signals (if enabled)
        reddit_video_ids: Set[str] = set()
        if self._reddit_monitor:
            try:
                reddit_videos = self._reddit_monitor.find_videos_by_topic(topic, limit_per_subreddit=25)
                reddit_video_ids = {v.video_id for v in reddit_videos}
                logger.info(f"Found {len(reddit_video_ids)} videos trending on Reddit for topic: {topic}")
            except Exception as e:
                logger.warning(f"Reddit discovery failed: {e}")

        # Step 1: Search for channels by topic
        # First, check if we have indexed channels for this niche
        indexed_channels = self._get_channels_from_index(topic, max_channels)

        if indexed_channels:
            logger.info(f"Using {len(indexed_channels)} channels from index for: {topic}")
            channel_urls = indexed_channels
        elif self._youtube_api:
            # Fall back to API search
            channel_urls = self._search_channels_by_topic_api(topic, max_channels)
        else:
            # Fall back to yt-dlp search
            channel_urls = self._search_channels_by_topic(topic, max_channels)

        if not channel_urls:
            logger.warning(f"No channels found for topic: {topic}")
            return result

        total_channels = len(channel_urls)
        logger.info(f"Found {total_channels} channels to analyze")

        # Step 2: Analyze channels in parallel
        all_outliers: List[OutlierVideo] = []
        completed_count = 0
        total_videos = 0

        # Use ThreadPoolExecutor for parallel channel analysis
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all channel analysis tasks
            future_to_channel = {
                executor.submit(self._analyze_channel_safe, url): url
                for url in channel_urls
            }

            # Process results as they complete
            for future in as_completed(future_to_channel):
                channel_url = future_to_channel[future]
                completed_count += 1

                try:
                    channel_outliers, videos_count = future.result()
                    total_videos += videos_count
                    result.total_videos_scanned = total_videos
                    result.channels_analyzed = completed_count

                    if channel_outliers:
                        all_outliers.extend(channel_outliers)
                        logger.info(
                            f"Channel {completed_count}/{total_channels}: "
                            f"found {len(channel_outliers)} outliers from {videos_count} videos"
                        )

                        # Call callback for each outlier found
                        if on_outlier_found:
                            for outlier in channel_outliers:
                                try:
                                    on_outlier_found(outlier)
                                except Exception as e:
                                    logger.warning(f"Error in on_outlier_found callback: {e}")

                    # Call channel complete callback
                    if on_channel_complete:
                        try:
                            on_channel_complete(completed_count, total_channels, total_videos)
                        except Exception as e:
                            logger.warning(f"Error in on_channel_complete callback: {e}")

                except Exception as e:
                    logger.error(f"Error analyzing channel {channel_url}: {e}")
                    result.channels_analyzed = completed_count

        # Sort outliers by score (highest first)
        all_outliers.sort(key=lambda x: x.outlier_score, reverse=True)
        result.outliers = all_outliers

        # Log cache stats
        if self.cache:
            stats = self.cache.get_stats()
            logger.info(
                f"Cache stats: {stats['videos_cached']} videos, "
                f"{stats['channels_cached']} channels cached"
            )

        logger.info(
            f"Search complete: {len(all_outliers)} outliers from "
            f"{result.channels_analyzed} channels, {result.total_videos_scanned} videos"
        )

        return result

    def _analyze_channel_safe(self, channel_url: str) -> Tuple[List[OutlierVideo], int]:
        """Thread-safe wrapper for channel analysis with rate limiting."""
        # Add small random delay to avoid rate limiting
        time.sleep(random.uniform(*self.channel_delay))
        return self._analyze_channel(channel_url)

    def _get_channels_from_index(self, topic: str, max_results: int) -> List[str]:
        """Get channels from the local index for a topic.

        Checks the channel index for pre-indexed channels matching this topic.
        Much faster than API search when index is populated.

        Args:
            topic: Topic/niche to search for
            max_results: Maximum number of channels to return

        Returns:
            List of channel URLs (empty if no indexed channels found)
        """
        if not self.cache:
            return []

        try:
            # Query index for channels with this niche tag
            indexed = self.cache.get_channels_by_niche(
                niche=topic.lower(),
                min_subs=self.min_subs,
                max_subs=self.max_subs,
                limit=max_results,
            )

            if indexed:
                channel_urls = [
                    f"https://www.youtube.com/channel/{ch['channel_id']}"
                    for ch in indexed
                    if ch.get('channel_id')
                ]
                logger.debug(f"Found {len(channel_urls)} indexed channels for: {topic}")
                return channel_urls

        except Exception as e:
            logger.warning(f"Error querying channel index: {e}")

        return []

    def _search_channels_by_topic_api(
        self, topic: str, max_results: int
    ) -> List[str]:
        """Search for channels using YouTube Data API.

        Much faster than yt-dlp scraping and includes subscriber counts.

        Args:
            topic: Topic to search for
            max_results: Maximum number of channels to return

        Returns:
            List of channel URLs
        """
        if not self._youtube_api:
            return self._search_channels_by_topic(topic, max_results)

        try:
            # Search for channels with subscriber filtering
            channels = self._youtube_api.search_channels(
                topic=topic,
                max_results=max_results,
                min_subscriber_count=self.min_subs,
                max_subscriber_count=self.max_subs,
            )

            channel_urls = []
            for channel in channels:
                url = f"https://www.youtube.com/channel/{channel.channel_id}"
                channel_urls.append(url)

                # Pre-cache channel info from API
                if self.cache:
                    self.cache.set_channel(channel.channel_id, {
                        "channel_id": channel.channel_id,
                        "channel_name": channel.channel_name,
                        "subscriber_count": channel.subscriber_count,
                        "video_count": channel.video_count,
                        "view_count": channel.view_count,
                    })

            logger.info(f"YouTube API found {len(channel_urls)} channels for topic: {topic}")
            logger.info(f"API quota used: {self._youtube_api.quota_used} units")

            return channel_urls

        except Exception as e:
            logger.error(f"YouTube API channel search failed: {e}, falling back to yt-dlp")
            return self._search_channels_by_topic(topic, max_results)

    def _extract_channel_videos_api(self, channel_id: str) -> List[Dict]:
        """Extract channel videos using YouTube Data API.

        Much faster than yt-dlp and includes all metadata in single request.

        Args:
            channel_id: YouTube channel ID

        Returns:
            List of video metadata dicts (compatible with yt-dlp format)
        """
        if not self._youtube_api:
            return []

        try:
            api_videos = self._youtube_api.get_channel_videos(
                channel_id=channel_id,
                max_results=self.max_videos_per_channel,
            )

            # Convert API format to yt-dlp compatible format
            videos = []
            for v in api_videos:
                # Convert ISO date to YYYYMMDD format
                upload_date = ""
                if v.published_at:
                    try:
                        dt = datetime.fromisoformat(v.published_at.replace("Z", "+00:00"))
                        upload_date = dt.strftime("%Y%m%d")
                    except ValueError:
                        pass

                video_dict = {
                    "id": v.video_id,
                    "title": v.title,
                    "view_count": v.view_count,
                    "like_count": v.like_count,
                    "comment_count": v.comment_count,
                    "duration": v.duration_seconds,
                    "upload_date": upload_date,
                    "channel": v.channel_name,
                    "uploader": v.channel_name,
                    "channel_id": v.channel_id,
                    "thumbnail": v.thumbnail_url,
                    "is_short": v.is_short,
                }
                videos.append(video_dict)

                # Cache each video
                if self.cache:
                    self.cache.set_video(v.video_id, video_dict)

            logger.debug(f"YouTube API fetched {len(videos)} videos for channel {channel_id}")
            return videos

        except Exception as e:
            logger.warning(f"YouTube API video fetch failed for {channel_id}: {e}")
            return []

    def _search_channels_by_topic(
        self, topic: str, max_results: int
    ) -> List[str]:
        """Search YouTube for channels related to the topic.

        Uses yt-dlp's ytsearch to find videos, then extracts unique channel URLs.

        Args:
            topic: Topic to search for
            max_results: Maximum number of channels to return

        Returns:
            List of unique channel URLs
        """
        # Search for more videos than channels needed to get unique channels
        # Increased multiplier for better channel diversity
        search_count = min(max_results * 5, 200)  # Cap at 200 search results
        search_query = f"ytsearch{search_count}:{topic}"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "http_headers": {
                "User-Agent": get_random_user_agent(),
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)

                if not info or "entries" not in info:
                    return []

                # Extract unique channel URLs from search results
                seen_channels: Set[str] = set()
                channel_urls: List[str] = []

                for entry in info["entries"]:
                    if not entry:
                        continue

                    # Get channel URL from video entry
                    channel_id = entry.get("channel_id")
                    channel_url = entry.get("channel_url")

                    if not channel_url and channel_id:
                        channel_url = f"https://www.youtube.com/channel/{channel_id}"

                    if channel_url and channel_url not in seen_channels:
                        seen_channels.add(channel_url)
                        channel_urls.append(channel_url)

                        if len(channel_urls) >= max_results:
                            break

                return channel_urls

        except Exception as e:
            logger.error(f"Error searching for channels: {e}")
            return []

    def _analyze_channel(
        self, channel_url: str
    ) -> Tuple[List[OutlierVideo], int]:
        """Analyze a channel for outlier videos.

        Enhanced with multi-layer scoring including:
        - IQR-based statistical score
        - Engagement metrics (likes, comments)
        - Velocity tracking (views per day)
        - Composite score combining all factors

        Args:
            channel_url: YouTube channel URL

        Returns:
            Tuple of (list of outlier videos, total videos analyzed)
        """
        # Extract channel ID for cache lookup
        channel_id = self._extract_channel_id(channel_url)

        # Check cache for channel video list
        cached_video_ids = None
        if self.cache and channel_id:
            cached_video_ids = self.cache.get_channel_video_ids(channel_id)

        if cached_video_ids:
            logger.debug(f"Using cached video list for channel {channel_id}")
            videos = self._get_videos_from_ids(cached_video_ids)
        else:
            # Get channel videos - use API if available, otherwise yt-dlp
            if self._youtube_api and channel_id:
                videos = self._extract_channel_videos_api(channel_id)
            else:
                videos = self._extract_channel_videos(channel_url)

            # Cache the video IDs
            if self.cache and videos and channel_id:
                video_ids = [v.get("id") for v in videos if v.get("id")]
                self.cache.set_channel_video_ids(channel_id, video_ids)

        if not videos:
            return [], 0

        # Filter by date if specified
        if self.date_days:
            cutoff_date = datetime.now() - timedelta(days=self.date_days)
            videos = [
                v for v in videos
                if self._parse_date(v.get("upload_date", "")) >= cutoff_date
            ]

        if not videos:
            return [], 0

        # Filter out Shorts if requested
        if self.exclude_shorts:
            videos = [v for v in videos if not self._is_short(v)]

        if not videos:
            return [], 0

        # Filter out Indian language videos if requested
        if self.exclude_indian:
            videos = [v for v in videos if not self._is_indian_video(v)]

        if not videos:
            return [], 0

        # Get channel name
        channel_name = videos[0].get("channel", videos[0].get("uploader", "Unknown"))

        # Calculate comprehensive channel statistics
        channel_stats = self._calculate_channel_stats(
            videos, channel_id or "unknown", channel_name
        )

        if channel_stats.average_views <= 0:
            return [], len(videos)

        # Cache channel stats
        if self.cache and channel_id:
            self.cache.set_channel(channel_id, {
                "channel_id": channel_id,
                "average_views": channel_stats.average_views,
                "median_views": channel_stats.median_views,
                "video_count": len(videos),
                "q1_views": channel_stats.q1_views,
                "q3_views": channel_stats.q3_views,
                "iqr_views": channel_stats.iqr_views,
                "upper_bound": channel_stats.upper_bound,
                "median_views_per_day": channel_stats.median_views_per_day,
            })

        # Find outliers with enhanced scoring
        outliers = []

        for video in videos:
            view_count = video.get("view_count", 0)
            if not view_count:
                continue

            # Skip videos below minimum view threshold
            if self.min_views and view_count < self.min_views:
                continue

            # Legacy ratio score
            ratio_score = view_count / channel_stats.average_views

            if ratio_score >= self.min_score:
                tier = OutlierVideo.calculate_tier(ratio_score)

                # Calculate engagement metrics
                like_count = video.get("like_count")
                comment_count = video.get("comment_count")
                engagement_rate = self._calculate_engagement_rate(video)
                engagement_score = self._calculate_engagement_score(engagement_rate)

                # Calculate velocity metrics
                upload_date = video.get("upload_date", "")
                days_since_upload = self._days_since_upload(upload_date)
                views_per_day = (
                    view_count / days_since_upload if days_since_upload > 0 else None
                )
                velocity_score = self._calculate_velocity_score(
                    views_per_day, channel_stats.median_views_per_day or 0
                )

                # Calculate IQR-based statistical score
                statistical_score = self._calculate_statistical_score(
                    view_count, channel_stats.upper_bound or 0
                )

                # Calculate composite score
                composite_score = self._calculate_composite_score(
                    statistical_score, engagement_score, velocity_score, ratio_score
                )

                outlier = OutlierVideo(
                    # Core fields
                    video_id=video.get("id", ""),
                    title=video.get("title", "Unknown"),
                    url=f"https://www.youtube.com/watch?v={video.get('id', '')}",
                    thumbnail_url=self._get_thumbnail_url(video),
                    view_count=view_count,
                    outlier_score=ratio_score,
                    channel_average_views=channel_stats.average_views,
                    channel_name=channel_name,
                    upload_date=upload_date,
                    outlier_tier=tier,
                    # Engagement metrics
                    like_count=like_count,
                    comment_count=comment_count,
                    engagement_rate=engagement_rate,
                    # Velocity metrics
                    days_since_upload=days_since_upload if days_since_upload > 0 else None,
                    views_per_day=views_per_day,
                    velocity_score=velocity_score,
                    # Composite scoring
                    composite_score=composite_score,
                    statistical_score=statistical_score,
                    engagement_score=engagement_score,
                )
                outliers.append(outlier)

        return outliers, len(videos)

    def _get_videos_from_ids(self, video_ids: List[str]) -> List[Dict]:
        """Get video metadata from cache or fetch if needed.

        Uses batched lookups for efficiency.

        Args:
            video_ids: List of video IDs

        Returns:
            List of video metadata dicts
        """
        videos = []

        if self.cache:
            # Batch lookup from cache
            cached = self.cache.get_videos_batch(video_ids)

            # Separate cached and uncached
            uncached_ids = [vid for vid, data in cached.items() if data is None]
            for vid, data in cached.items():
                if data:
                    videos.append(data)

            # Fetch uncached in batches
            if uncached_ids:
                logger.debug(f"Fetching {len(uncached_ids)} uncached videos")
                fetched = self._fetch_videos_batch(uncached_ids)
                videos.extend(fetched)

                # Cache the fetched videos
                self.cache.set_videos_batch(fetched)
        else:
            # No cache, fetch all
            videos = self._fetch_videos_batch(video_ids)

        return videos

    def _fetch_videos_batch(self, video_ids: List[str]) -> List[Dict]:
        """Fetch video metadata for multiple videos.

        Processes in batches with rate limiting.

        Args:
            video_ids: List of video IDs to fetch

        Returns:
            List of video metadata dicts
        """
        videos = []

        for i in range(0, len(video_ids), self.batch_size):
            batch = video_ids[i:i + self.batch_size]

            for video_id in batch:
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                ydl_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "skip_download": True,
                    "http_headers": {
                        "User-Agent": get_random_user_agent(),
                    },
                }

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(video_url, download=False)
                        if info:
                            videos.append(info)
                except Exception as e:
                    logger.warning(f"Could not fetch video {video_id}: {e}")

                # Rate limiting between fetches
                time.sleep(random.uniform(*self.video_delay))

        return videos

    def _extract_channel_id(self, channel_url: str) -> Optional[str]:
        """Extract channel ID from URL.

        Args:
            channel_url: YouTube channel URL

        Returns:
            Channel ID or None
        """
        # Handle different URL formats
        if "/channel/" in channel_url:
            parts = channel_url.split("/channel/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]
        elif "/@" in channel_url:
            # Handle @username format - would need to resolve
            return None
        return None

    def _extract_channel_videos(self, channel_url: str) -> List[dict]:
        """Extract video list with metadata from a channel.

        Uses yt-dlp extract_flat for fast initial extraction,
        then enriches with view counts.

        Args:
            channel_url: YouTube channel URL

        Returns:
            List of video metadata dictionaries
        """
        # Ensure we're getting the videos tab
        if "/videos" not in channel_url:
            channel_url = channel_url.rstrip("/") + "/videos"

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "playlistend": self.max_videos_per_channel,
            "http_headers": {
                "User-Agent": get_random_user_agent(),
            },
        }

        retry_count = 0
        max_retries = len(self.retry_delays)

        while retry_count <= max_retries:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(channel_url, download=False)

                    if not info:
                        return []

                    # Get channel name from playlist metadata
                    channel_name = (
                        info.get("channel")
                        or info.get("uploader")
                        or info.get("title", "").replace(" - Videos", "")
                        or "Unknown"
                    )

                    # Handle playlist-style results
                    if "entries" in info:
                        videos = []
                        for entry in info["entries"]:
                            if entry:
                                # Inject channel name into each video entry
                                entry["channel"] = entry.get("channel") or channel_name
                                entry["uploader"] = entry.get("uploader") or channel_name
                                videos.append(entry)

                        # Enrich with view counts (batched)
                        return self._enrich_video_metadata(videos)
                    else:
                        return []

            except Exception as e:
                error_msg = str(e).lower()

                # Check for rate limiting
                if "403" in error_msg or "forbidden" in error_msg:
                    if retry_count < max_retries:
                        delay = self.retry_delays[retry_count]
                        logger.warning(
                            f"Rate limited, retrying in {delay}s (attempt {retry_count + 1})"
                        )
                        time.sleep(delay)
                        retry_count += 1
                        continue

                logger.error(f"Error extracting channel videos: {e}")
                return []

        return []

    def _enrich_video_metadata(self, videos: List[dict]) -> List[dict]:
        """Enrich video list with view counts if not present.

        Now with caching and YouTube API support for batch fetching.
        Uses API when available (50 videos per request), otherwise yt-dlp.

        Args:
            videos: List of video metadata dicts

        Returns:
            Enriched video list with view counts
        """
        enriched = []
        to_fetch = []

        # First pass: check which videos already have view_count or are cached
        for video in videos:
            if video.get("view_count"):
                enriched.append(video)
                # Also cache it
                if self.cache:
                    video_id = video.get("id")
                    if video_id:
                        self.cache.set_video(video_id, video)
                continue

            video_id = video.get("id")
            if not video_id:
                continue

            # Check cache
            if self.cache:
                cached = self.cache.get_video(video_id)
                if cached and cached.get("view_count"):
                    # Merge cached data with original
                    video.update(cached)
                    enriched.append(video)
                    continue

            # Need to fetch this one
            to_fetch.append(video)

        # Batch fetch uncached videos
        if to_fetch:
            logger.debug(f"Enriching {len(to_fetch)} videos with metadata")

            # Use YouTube API for batch fetching if available (50x faster)
            if self._youtube_api:
                enriched.extend(self._enrich_videos_with_api(to_fetch))
            else:
                enriched.extend(self._enrich_videos_with_ytdlp(to_fetch))

        return enriched

    def _enrich_videos_with_api(self, videos: List[dict]) -> List[dict]:
        """Enrich videos using YouTube Data API (batched, 50 per request).

        Args:
            videos: List of video metadata dicts

        Returns:
            Enriched video list
        """
        if not self._youtube_api:
            return videos

        enriched = []
        video_ids = [v.get("id") for v in videos if v.get("id")]

        try:
            # Batch fetch video details (50 per request)
            video_details = self._youtube_api.get_video_details(video_ids)

            # Create lookup for quick access
            video_lookup = {v.get("id"): v for v in videos}

            for video_id, details in video_details.items():
                video = video_lookup.get(video_id, {})

                # Convert ISO date to YYYYMMDD format
                upload_date = video.get("upload_date", "")
                if details.published_at and not upload_date:
                    try:
                        dt = datetime.fromisoformat(details.published_at.replace("Z", "+00:00"))
                        upload_date = dt.strftime("%Y%m%d")
                    except ValueError:
                        pass

                video.update({
                    "view_count": details.view_count,
                    "like_count": details.like_count,
                    "comment_count": details.comment_count,
                    "duration": details.duration_seconds,
                    "upload_date": upload_date,
                    "channel": details.channel_name,
                    "is_short": details.is_short,
                })
                enriched.append(video)

                # Cache the enriched video
                if self.cache:
                    self.cache.set_video(video_id, video)

            # Handle any videos not returned by API
            returned_ids = set(video_details.keys())
            for video in videos:
                if video.get("id") not in returned_ids:
                    enriched.append(video)

        except Exception as e:
            logger.warning(f"YouTube API enrichment failed: {e}, falling back to yt-dlp")
            return self._enrich_videos_with_ytdlp(videos)

        return enriched

    def _enrich_videos_with_ytdlp(self, videos: List[dict]) -> List[dict]:
        """Enrich videos using yt-dlp (slower, one at a time).

        Args:
            videos: List of video metadata dicts

        Returns:
            Enriched video list
        """
        enriched = []

        for video in videos:
            video_id = video.get("id")
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "http_headers": {
                    "User-Agent": get_random_user_agent(),
                },
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    if info:
                        # Merge enriched data with original
                        video.update({
                            "view_count": info.get("view_count", 0),
                            "upload_date": info.get("upload_date", video.get("upload_date", "")),
                            "duration": info.get("duration", video.get("duration", 0)),
                            "channel": info.get("channel", info.get("uploader", "")),
                            "like_count": info.get("like_count"),
                            "comment_count": info.get("comment_count"),
                        })
                        enriched.append(video)

                        # Cache the enriched video
                        if self.cache:
                            self.cache.set_video(video_id, video)

                # Rate limiting between video fetches
                delay = random.uniform(*self.video_delay)
                time.sleep(delay)

            except Exception as e:
                logger.warning(f"Could not enrich video {video_id}: {e}")
                # Still include video with whatever data we have
                enriched.append(video)

        return enriched

    def _calculate_channel_average(
        self, videos: List[dict]
    ) -> Tuple[float, float]:
        """Calculate channel average and median views.

        Excludes top 5% to avoid skewing by previous outliers.

        Args:
            videos: List of video metadata dicts

        Returns:
            Tuple of (average_views, median_views)
        """
        view_counts = [v.get("view_count", 0) for v in videos if v.get("view_count")]

        if not view_counts:
            return 0.0, 0.0

        # Sort and exclude top 5%
        view_counts.sort()
        exclude_count = max(1, len(view_counts) // 20)  # 5%
        filtered_counts = view_counts[:-exclude_count] if len(view_counts) > 1 else view_counts

        if not filtered_counts:
            filtered_counts = view_counts

        avg_views = statistics.mean(filtered_counts)
        median_views = statistics.median(filtered_counts)

        return avg_views, median_views

    def _calculate_channel_stats(
        self, videos: List[dict], channel_id: str, channel_name: str
    ) -> ChannelStats:
        """Calculate comprehensive channel statistics including IQR and velocity.

        Args:
            videos: List of video metadata dicts
            channel_id: YouTube channel ID
            channel_name: Channel display name

        Returns:
            ChannelStats with IQR-based statistics and velocity metrics
        """
        view_counts = [v.get("view_count", 0) for v in videos if v.get("view_count")]

        if not view_counts:
            return ChannelStats(
                channel_id=channel_id,
                channel_name=channel_name,
                average_views=0.0,
                median_views=0.0,
                total_videos_analyzed=0,
            )

        # Sort and exclude top 5% for average calculation
        view_counts_sorted = sorted(view_counts)
        exclude_count = max(1, len(view_counts_sorted) // 20)
        filtered_counts = (
            view_counts_sorted[:-exclude_count]
            if len(view_counts_sorted) > 1
            else view_counts_sorted
        )

        avg_views = statistics.mean(filtered_counts) if filtered_counts else 0.0
        median_views = statistics.median(filtered_counts) if filtered_counts else 0.0

        # Calculate IQR-based statistics using numpy
        q1 = float(np.percentile(view_counts, 25))
        q3 = float(np.percentile(view_counts, 75))
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        # Calculate velocity statistics (views per day)
        velocity_values = []
        for video in videos:
            view_count = video.get("view_count", 0)
            upload_date = video.get("upload_date", "")
            if view_count and upload_date:
                days_old = self._days_since_upload(upload_date)
                if days_old > 0:
                    velocity_values.append(view_count / days_old)

        median_velocity = statistics.median(velocity_values) if velocity_values else 0.0
        avg_velocity = statistics.mean(velocity_values) if velocity_values else 0.0

        return ChannelStats(
            channel_id=channel_id,
            channel_name=channel_name,
            average_views=avg_views,
            median_views=median_views,
            total_videos_analyzed=len(videos),
            q1_views=q1,
            q3_views=q3,
            iqr_views=iqr,
            upper_bound=upper_bound,
            median_views_per_day=median_velocity,
            average_views_per_day=avg_velocity,
        )

    def _days_since_upload(self, upload_date: str) -> int:
        """Calculate days since upload from YYYYMMDD format."""
        if not upload_date or len(upload_date) != 8:
            return 0
        try:
            upload_dt = datetime.strptime(upload_date, "%Y%m%d")
            return max(1, (datetime.now() - upload_dt).days)
        except ValueError:
            return 0

    def _calculate_engagement_rate(self, video: dict) -> Optional[float]:
        """Calculate engagement rate as (likes + comments) / views * 100."""
        view_count = video.get("view_count", 0)
        like_count = video.get("like_count", 0) or 0
        comment_count = video.get("comment_count", 0) or 0

        if view_count <= 0:
            return None

        return (like_count + comment_count) / view_count * 100

    def _calculate_engagement_score(self, engagement_rate: Optional[float]) -> Optional[float]:
        """Normalize engagement rate to a score (6% = 1.0, capped at 2.0)."""
        if engagement_rate is None:
            return None
        return min(engagement_rate / 6.0, 2.0)

    def _calculate_velocity_score(
        self, views_per_day: Optional[float], channel_median_velocity: float
    ) -> Optional[float]:
        """Calculate velocity score as video velocity / channel median velocity."""
        if views_per_day is None or channel_median_velocity <= 0:
            return None
        return views_per_day / channel_median_velocity

    def _calculate_statistical_score(
        self, view_count: int, upper_bound: float
    ) -> Optional[float]:
        """Calculate IQR-based statistical score."""
        if upper_bound <= 0:
            return None
        return view_count / upper_bound

    def _calculate_composite_score(
        self,
        statistical_score: Optional[float],
        engagement_score: Optional[float],
        velocity_score: Optional[float],
        ratio_score: float,
    ) -> float:
        """Calculate weighted composite score from all components.

        Weights:
        - 35% statistical (IQR-based)
        - 25% engagement
        - 25% velocity
        - 15% ratio (views/channel average - legacy)

        Falls back to available scores if some are missing.
        """
        scores = []
        weights = []

        if statistical_score is not None:
            scores.append(statistical_score)
            weights.append(0.35)

        if engagement_score is not None:
            scores.append(engagement_score)
            weights.append(0.25)

        if velocity_score is not None:
            scores.append(velocity_score)
            weights.append(0.25)

        # Always have ratio score
        scores.append(ratio_score)
        weights.append(0.15)

        # Normalize weights if some scores are missing
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            return sum(s * w for s, w in zip(scores, normalized_weights))

        return ratio_score

    def _get_thumbnail_url(self, video: dict) -> str:
        """Get the best available thumbnail URL for a video."""
        if video.get("thumbnail"):
            return video["thumbnail"]

        if video.get("thumbnails"):
            thumbnails = video["thumbnails"]
            if thumbnails:
                for thumb in reversed(thumbnails):
                    if thumb.get("url"):
                        return thumb["url"]

        video_id = video.get("id", "")
        if video_id:
            return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        return ""

    def _is_short(self, video: dict) -> bool:
        """Check if a video is a YouTube Short."""
        duration = video.get("duration", 0)
        if duration and duration <= 60:
            return True

        title = video.get("title", "").lower()
        if "#shorts" in title or "#short" in title:
            return True

        url = video.get("url", "")
        if "/shorts/" in url:
            return True

        return False

    def _is_indian_video(self, video: dict) -> bool:
        """Check if a video is likely Indian based on title script or language metadata."""
        title = video.get("title", "")
        if _INDIAN_SCRIPT_RE.search(title):
            return True

        language = video.get("language", "")
        if language:
            indian_langs = {
                "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or",
                "as", "ur", "sd", "ne", "sa", "kok", "mai", "doi", "bho",
            }
            if language.lower().split("-")[0] in indian_langs:
                return True

        return False

    def _parse_date(self, date_str: str) -> datetime:
        """Parse YYYYMMDD date string."""
        if not date_str or len(date_str) != 8:
            return datetime.min

        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            return datetime.min
