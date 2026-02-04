"""Reddit Social Signal Monitor for Early Viral Video Detection.

Monitors Reddit for YouTube video links to catch viral content
24-72 hours before it peaks on YouTube.

Implements recommendation #8: Monitor social signals for early detection.

Enhanced with:
- PRAW support for authenticated API access (higher rate limits)
- RedditVideoFinder class for targeted discovery
- Cross-reference with outlier finding
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

import requests

# Try to import PRAW for authenticated access
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RedditVideo:
    """A YouTube video discovered on Reddit."""
    video_id: str
    title: str
    url: str
    subreddit: str
    reddit_score: int  # Reddit upvotes
    reddit_comments: int
    reddit_url: str
    discovered_at: datetime
    post_title: str
    # Enhanced fields for cross-referencing
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    upvote_ratio: float = 0.0
    is_crosspost: bool = False
    flair: str = ""
    awards: int = 0
    # Computed viral score
    viral_score: float = field(default=0.0)

    def __post_init__(self):
        """Calculate viral score after initialization."""
        if self.viral_score == 0.0:
            # Score based on engagement
            self.viral_score = (
                self.reddit_score * 1.0 +
                self.reddit_comments * 2.0 +  # Comments indicate high engagement
                self.awards * 50.0
            ) * self.upvote_ratio if self.upvote_ratio > 0 else (
                self.reddit_score + self.reddit_comments * 2 + self.awards * 50
            )


# Default subreddits to monitor for viral videos
DEFAULT_SUBREDDITS = [
    "videos",
    "youtubehaiku",
    "mealtimevideos",
    "listentothis",
    "documentaries",
    "lectures",
    "artisanvideos",
    "deepintoyoutube",
    "internetisbeautiful",
]

# Topic-specific subreddit mappings
TOPIC_SUBREDDITS = {
    "tech": ["technology", "gadgets", "android", "apple", "hardware", "buildapc"],
    "gaming": ["gaming", "games", "pcgaming", "nintendo", "playstation", "xbox"],
    "cooking": ["cooking", "food", "recipes", "gifrecipes", "mealprep"],
    "fitness": ["fitness", "bodybuilding", "weightlifting", "running", "crossfit"],
    "music": ["music", "listentothis", "hiphopheads", "indieheads", "metal"],
    "science": ["science", "space", "physics", "biology", "chemistry"],
    "business": ["entrepreneur", "smallbusiness", "startups", "business"],
    "education": ["education", "learnprogramming", "datascience", "machinelearning"],
    "travel": ["travel", "backpacking", "solotravel", "digitalnomad"],
    "cars": ["cars", "autos", "cartalk", "justrolledintotheshop"],
}


class RedditMonitor:
    """Monitor Reddit for YouTube videos to find early viral signals.

    Uses Reddit's public JSON API (no authentication required for read-only).
    Rate limited to respect Reddit's guidelines.
    """

    # Reddit API rate limits
    REQUESTS_PER_MINUTE = 10
    MIN_DELAY_SECONDS = 6  # 60 / 10 = 6 seconds between requests

    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        min_score: int = 100,  # Minimum Reddit upvotes
        max_age_hours: int = 48,  # Only videos posted in last 48 hours
    ):
        """Initialize the Reddit monitor.

        Args:
            subreddits: List of subreddits to monitor (uses defaults if None)
            min_score: Minimum Reddit score (upvotes) to consider
            max_age_hours: Maximum age of Reddit posts to consider
        """
        self.subreddits = subreddits or DEFAULT_SUBREDDITS
        self.min_score = min_score
        self.max_age_hours = max_age_hours

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; StockpileBot/1.0; +https://github.com/stockpile)"
        })

        self._last_request_time = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_DELAY_SECONDS:
            time.sleep(self.MIN_DELAY_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL.

        Handles various URL formats:
        - youtube.com/watch?v=VIDEO_ID
        - youtu.be/VIDEO_ID
        - youtube.com/embed/VIDEO_ID
        - youtube.com/v/VIDEO_ID

        Args:
            url: URL to parse

        Returns:
            Video ID or None
        """
        if not url:
            return None

        # Handle youtu.be format
        if "youtu.be/" in url:
            match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
            if match:
                return match.group(1)

        # Handle youtube.com formats
        if "youtube.com" in url:
            parsed = urlparse(url)

            # /watch?v=VIDEO_ID
            if parsed.path == "/watch":
                params = parse_qs(parsed.query)
                if "v" in params:
                    return params["v"][0]

            # /embed/VIDEO_ID or /v/VIDEO_ID
            match = re.search(r"/(embed|v)/([a-zA-Z0-9_-]{11})", parsed.path)
            if match:
                return match.group(2)

        return None

    def find_videos_in_subreddit(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 100,
    ) -> List[RedditVideo]:
        """Find YouTube videos posted in a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort order ("hot", "new", "top", "rising")
            limit: Maximum posts to fetch (max 100)

        Returns:
            List of RedditVideo objects
        """
        self._rate_limit()

        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {"limit": min(limit, 100)}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error fetching r/{subreddit}: {e}")
            return []

        videos = []
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)

        for post in data.get("data", {}).get("children", []):
            post_data = post.get("data", {})

            # Check post age
            created_utc = post_data.get("created_utc", 0)
            post_time = datetime.fromtimestamp(created_utc)
            if post_time < cutoff_time:
                continue

            # Check score
            score = post_data.get("score", 0)
            if score < self.min_score:
                continue

            # Check if it's a YouTube link
            post_url = post_data.get("url", "")
            video_id = self._extract_video_id(post_url)

            if not video_id:
                # Check if there's a YouTube link in the post content
                selftext = post_data.get("selftext", "")
                for word in selftext.split():
                    video_id = self._extract_video_id(word)
                    if video_id:
                        break

            if video_id:
                videos.append(RedditVideo(
                    video_id=video_id,
                    title=post_data.get("title", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    subreddit=subreddit,
                    reddit_score=score,
                    reddit_comments=post_data.get("num_comments", 0),
                    reddit_url=f"https://www.reddit.com{post_data.get('permalink', '')}",
                    discovered_at=post_time,
                    post_title=post_data.get("title", ""),
                ))

        return videos

    def find_videos_by_topic(
        self,
        topic: str,
        limit_per_subreddit: int = 50,
    ) -> List[RedditVideo]:
        """Find YouTube videos related to a topic across relevant subreddits.

        Args:
            topic: Topic to search for (e.g., "tech", "gaming", "cooking")
            limit_per_subreddit: Max posts per subreddit

        Returns:
            List of RedditVideo objects, sorted by Reddit score
        """
        # Determine which subreddits to search
        topic_lower = topic.lower()
        subreddits = set(DEFAULT_SUBREDDITS)

        # Add topic-specific subreddits
        for keyword, subs in TOPIC_SUBREDDITS.items():
            if keyword in topic_lower:
                subreddits.update(subs)

        logger.info(f"Searching {len(subreddits)} subreddits for topic: {topic}")

        all_videos: List[RedditVideo] = []
        seen_ids: Set[str] = set()

        for subreddit in subreddits:
            try:
                videos = self.find_videos_in_subreddit(
                    subreddit,
                    sort="hot",
                    limit=limit_per_subreddit
                )

                # Deduplicate
                for video in videos:
                    if video.video_id not in seen_ids:
                        seen_ids.add(video.video_id)
                        all_videos.append(video)

                logger.debug(f"r/{subreddit}: found {len(videos)} videos")

            except Exception as e:
                logger.error(f"Error searching r/{subreddit}: {e}")

        # Sort by Reddit score (highest first)
        all_videos.sort(key=lambda v: v.reddit_score, reverse=True)

        logger.info(f"Found {len(all_videos)} unique videos from Reddit")
        return all_videos

    def search_reddit(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        time_filter: str = "week",
        limit: int = 100,
    ) -> List[RedditVideo]:
        """Search Reddit for YouTube videos matching a query.

        Args:
            query: Search query
            subreddit: Optional subreddit to limit search
            sort: Sort order ("relevance", "hot", "top", "new", "comments")
            time_filter: Time filter ("hour", "day", "week", "month", "year", "all")
            limit: Maximum results

        Returns:
            List of RedditVideo objects
        """
        self._rate_limit()

        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
        else:
            url = "https://www.reddit.com/search.json"

        params = {
            "q": f"{query} site:youtube.com OR site:youtu.be",
            "sort": sort,
            "t": time_filter,
            "limit": min(limit, 100),
        }
        if subreddit:
            params["restrict_sr"] = "on"

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            return []

        videos = []
        seen_ids: Set[str] = set()
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)

        for post in data.get("data", {}).get("children", []):
            post_data = post.get("data", {})

            # Check post age
            created_utc = post_data.get("created_utc", 0)
            post_time = datetime.fromtimestamp(created_utc)
            if post_time < cutoff_time:
                continue

            # Check score
            score = post_data.get("score", 0)
            if score < self.min_score:
                continue

            # Extract video ID
            post_url = post_data.get("url", "")
            video_id = self._extract_video_id(post_url)

            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                videos.append(RedditVideo(
                    video_id=video_id,
                    title=post_data.get("title", ""),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    subreddit=post_data.get("subreddit", ""),
                    reddit_score=score,
                    reddit_comments=post_data.get("num_comments", 0),
                    reddit_url=f"https://www.reddit.com{post_data.get('permalink', '')}",
                    discovered_at=post_time,
                    post_title=post_data.get("title", ""),
                ))

        return videos

    def get_trending_videos(
        self,
        on_video_found: Optional[Callable[[RedditVideo], None]] = None,
    ) -> List[RedditVideo]:
        """Get currently trending YouTube videos from Reddit.

        Searches r/videos and other popular subreddits for hot content.

        Args:
            on_video_found: Optional callback for each video found

        Returns:
            List of trending videos sorted by score
        """
        all_videos: List[RedditVideo] = []
        seen_ids: Set[str] = set()

        # Search hot and rising from default subreddits
        for subreddit in self.subreddits[:5]:  # Limit to top 5
            for sort in ["hot", "rising"]:
                try:
                    videos = self.find_videos_in_subreddit(subreddit, sort=sort, limit=50)
                    for video in videos:
                        if video.video_id not in seen_ids:
                            seen_ids.add(video.video_id)
                            all_videos.append(video)
                            if on_video_found:
                                on_video_found(video)
                except Exception as e:
                    logger.error(f"Error fetching r/{subreddit}/{sort}: {e}")

        # Sort by combined score (upvotes + comments)
        all_videos.sort(key=lambda v: v.reddit_score + v.reddit_comments, reverse=True)

        return all_videos


class RedditVideoFinder:
    """Enhanced Reddit video discovery with PRAW support.

    Provides higher-level methods for finding viral YouTube videos
    with optional authenticated API access for better rate limits.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "stockpile:v2.0 (by /u/stockpile_bot)",
        min_score: int = 50,
        max_age_hours: int = 72,
    ):
        """Initialize the video finder.

        Args:
            client_id: Reddit API client ID (for PRAW)
            client_secret: Reddit API client secret (for PRAW)
            user_agent: User agent string
            min_score: Minimum Reddit score to consider
            max_age_hours: Maximum age of posts
        """
        self.min_score = min_score
        self.max_age_hours = max_age_hours
        self._reddit = None
        self._unauthenticated_monitor = RedditMonitor(
            min_score=min_score,
            max_age_hours=max_age_hours,
        )

        # Try to initialize PRAW for authenticated access
        if PRAW_AVAILABLE and client_id and client_secret:
            try:
                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                # Test connection
                self._reddit.user.me()
                logger.info("PRAW authenticated successfully - using higher rate limits")
            except Exception as e:
                logger.warning(f"PRAW authentication failed: {e}, using public API")
                self._reddit = None

    def find_viral_videos(
        self,
        subreddits: Optional[List[str]] = None,
        time_filter: str = "week",
        limit: int = 100,
    ) -> List[RedditVideo]:
        """Find viral YouTube videos across multiple subreddits.

        Args:
            subreddits: List of subreddit names (uses defaults if None)
            time_filter: Time filter ("hour", "day", "week", "month", "year", "all")
            limit: Maximum results per subreddit

        Returns:
            List of RedditVideo objects sorted by viral score
        """
        if subreddits is None:
            subreddits = DEFAULT_SUBREDDITS

        all_videos: List[RedditVideo] = []
        seen_ids: Set[str] = set()

        if self._reddit:
            # Use PRAW for authenticated access
            all_videos = self._find_viral_with_praw(subreddits, time_filter, limit, seen_ids)
        else:
            # Fall back to public API
            for subreddit in subreddits:
                videos = self._unauthenticated_monitor.find_videos_in_subreddit(
                    subreddit, sort="top", limit=limit
                )
                for video in videos:
                    if video.video_id not in seen_ids:
                        seen_ids.add(video.video_id)
                        all_videos.append(video)

        # Sort by viral score
        all_videos.sort(key=lambda v: v.viral_score, reverse=True)
        return all_videos

    def _find_viral_with_praw(
        self,
        subreddits: List[str],
        time_filter: str,
        limit: int,
        seen_ids: Set[str],
    ) -> List[RedditVideo]:
        """Find videos using PRAW authenticated API."""
        videos = []
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)

        for subreddit_name in subreddits:
            try:
                subreddit = self._reddit.subreddit(subreddit_name)

                for submission in subreddit.top(time_filter=time_filter, limit=limit):
                    # Check age
                    post_time = datetime.fromtimestamp(submission.created_utc)
                    if post_time < cutoff_time:
                        continue

                    # Check score
                    if submission.score < self.min_score:
                        continue

                    # Extract video ID
                    video_id = self._extract_video_id(submission.url)
                    if not video_id and hasattr(submission, 'selftext'):
                        # Check selftext for YouTube links
                        for word in submission.selftext.split():
                            video_id = self._extract_video_id(word)
                            if video_id:
                                break

                    if video_id and video_id not in seen_ids:
                        seen_ids.add(video_id)
                        videos.append(RedditVideo(
                            video_id=video_id,
                            title=submission.title,
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            subreddit=subreddit_name,
                            reddit_score=submission.score,
                            reddit_comments=submission.num_comments,
                            reddit_url=f"https://www.reddit.com{submission.permalink}",
                            discovered_at=post_time,
                            post_title=submission.title,
                            upvote_ratio=submission.upvote_ratio,
                            is_crosspost=hasattr(submission, 'crosspost_parent'),
                            flair=submission.link_flair_text or "",
                            awards=submission.total_awards_received,
                        ))

            except Exception as e:
                logger.warning(f"Error fetching r/{subreddit_name} with PRAW: {e}")

        return videos

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        return self._unauthenticated_monitor._extract_video_id(url)

    def search_topic(
        self,
        topic: str,
        limit: int = 100,
    ) -> List[RedditVideo]:
        """Search Reddit for YouTube videos about a topic.

        Args:
            topic: Topic to search for
            limit: Maximum results

        Returns:
            List of RedditVideo objects
        """
        if self._reddit:
            return self._search_with_praw(topic, limit)
        else:
            return self._unauthenticated_monitor.search_reddit(
                query=topic,
                time_filter="week",
                limit=limit,
            )

    def _search_with_praw(self, query: str, limit: int) -> List[RedditVideo]:
        """Search using PRAW authenticated API."""
        videos = []
        seen_ids: Set[str] = set()
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)

        try:
            # Search across all subreddits
            for submission in self._reddit.subreddit("all").search(
                f"{query} site:youtube.com OR site:youtu.be",
                sort="relevance",
                time_filter="week",
                limit=limit,
            ):
                # Check age
                post_time = datetime.fromtimestamp(submission.created_utc)
                if post_time < cutoff_time:
                    continue

                # Check score
                if submission.score < self.min_score:
                    continue

                # Extract video ID
                video_id = self._extract_video_id(submission.url)
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    videos.append(RedditVideo(
                        video_id=video_id,
                        title=submission.title,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        subreddit=submission.subreddit.display_name,
                        reddit_score=submission.score,
                        reddit_comments=submission.num_comments,
                        reddit_url=f"https://www.reddit.com{submission.permalink}",
                        discovered_at=post_time,
                        post_title=submission.title,
                        upvote_ratio=submission.upvote_ratio,
                        awards=submission.total_awards_received,
                    ))

        except Exception as e:
            logger.error(f"PRAW search failed: {e}")

        videos.sort(key=lambda v: v.viral_score, reverse=True)
        return videos

    def get_trending_youtube(self) -> List[RedditVideo]:
        """Get currently trending YouTube videos from r/videos and related.

        Returns:
            List of trending videos sorted by viral score
        """
        trending_subreddits = [
            "videos",
            "youtubehaiku",
            "mealtimevideos",
            "listentothis",
            "documentaries",
        ]

        return self.find_viral_videos(
            subreddits=trending_subreddits,
            time_filter="day",
            limit=50,
        )


# Convenience function
def find_reddit_videos(
    topic: Optional[str] = None,
    min_score: int = 100,
    max_age_hours: int = 48,
) -> List[RedditVideo]:
    """Find YouTube videos trending on Reddit.

    Args:
        topic: Optional topic to focus on
        min_score: Minimum Reddit upvotes
        max_age_hours: Maximum age of posts

    Returns:
        List of RedditVideo objects
    """
    monitor = RedditMonitor(min_score=min_score, max_age_hours=max_age_hours)

    if topic:
        return monitor.find_videos_by_topic(topic)
    else:
        return monitor.get_trending_videos()


def get_reddit_video_finder(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> RedditVideoFinder:
    """Get a RedditVideoFinder instance.

    Tries to load credentials from config if not provided.

    Args:
        client_id: Reddit API client ID
        client_secret: Reddit API client secret

    Returns:
        RedditVideoFinder instance
    """
    if client_id is None or client_secret is None:
        try:
            from utils.config import load_config
            config = load_config()
            client_id = client_id or config.get("reddit_client_id")
            client_secret = client_secret or config.get("reddit_client_secret")
        except Exception:
            pass

    return RedditVideoFinder(
        client_id=client_id,
        client_secret=client_secret,
    )
