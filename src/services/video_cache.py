"""Video metadata caching with SQLite persistence and in-memory TTL cache.

Implements recommendations #4 (database persistence) and #5 (TTL caching).
Dramatically reduces API calls by caching video/channel statistics.
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """In-memory cache entry with TTL."""
    value: Any
    expires_at: float


class VideoCache:
    """Multi-tier caching for YouTube video and channel data.

    Tier 1: In-memory TTL cache (fast, volatile)
    Tier 2: SQLite database (persistent, slower)

    TTL Strategy (based on data volatility):
    - View counts: 15 minutes (frequently changing)
    - Video metadata: 24 hours (stable)
    - Channel info: 7 days (rarely changes)
    """

    # Default TTLs in seconds
    TTL_VIEW_COUNTS = 900        # 15 minutes
    TTL_VIDEO_METADATA = 86400   # 24 hours
    TTL_CHANNEL_INFO = 604800    # 7 days
    TTL_CHANNEL_VIDEOS = 3600    # 1 hour (list of video IDs)

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the cache.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            cache_dir = Path.home() / ".stockpile" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "video_cache.db")

        self.db_path = db_path
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

        # Initialize database
        self._init_db()

        logger.info(f"VideoCache initialized at {db_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Video metadata table
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    channel_id TEXT,
                    channel_name TEXT,
                    view_count INTEGER,
                    like_count INTEGER,
                    comment_count INTEGER,
                    duration INTEGER,
                    upload_date TEXT,
                    thumbnail_url TEXT,
                    is_short BOOLEAN,
                    metadata_json TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Channel metadata table
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id TEXT PRIMARY KEY,
                    channel_name TEXT,
                    subscriber_count INTEGER,
                    video_count INTEGER,
                    average_views REAL,
                    median_views REAL,
                    metadata_json TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Channel videos list (cached video IDs)
                CREATE TABLE IF NOT EXISTS channel_videos (
                    channel_id TEXT,
                    video_id TEXT,
                    position INTEGER,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (channel_id, video_id)
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
                CREATE INDEX IF NOT EXISTS idx_videos_fetched ON videos(fetched_at);
                CREATE INDEX IF NOT EXISTS idx_channel_videos_fetched ON channel_videos(fetched_at);

                -- View count history for trend analysis
                CREATE TABLE IF NOT EXISTS view_history (
                    video_id TEXT,
                    view_count INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (video_id, recorded_at)
                );
            """)

    # =========================================================================
    # In-Memory Cache Operations
    # =========================================================================

    def _memory_get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache if not expired."""
        with self._lock:
            entry = self._memory_cache.get(key)
            if entry and entry.expires_at > time.time():
                return entry.value
            elif entry:
                # Expired, remove it
                del self._memory_cache[key]
            return None

    def _memory_set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in in-memory cache with TTL."""
        with self._lock:
            self._memory_cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )

    def _memory_clear_expired(self) -> int:
        """Clear expired entries from memory cache. Returns count cleared."""
        now = time.time()
        cleared = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._memory_cache.items()
                if v.expires_at <= now
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                cleared += 1
        return cleared

    # =========================================================================
    # Video Operations
    # =========================================================================

    def get_video(self, video_id: str) -> Optional[Dict]:
        """Get video metadata from cache.

        Checks in-memory cache first, then SQLite.

        Args:
            video_id: YouTube video ID

        Returns:
            Video metadata dict or None if not cached/expired
        """
        cache_key = f"video:{video_id}"

        # Check memory cache first
        cached = self._memory_get(cache_key)
        if cached:
            return cached

        # Check SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?",
                (video_id,)
            ).fetchone()

            if row:
                # Check if data is fresh enough for view counts
                fetched_at = datetime.fromisoformat(row["updated_at"])
                age_seconds = (datetime.now() - fetched_at).total_seconds()

                if age_seconds < self.TTL_VIDEO_METADATA:
                    video_data = dict(row)
                    # Parse metadata JSON if present
                    if video_data.get("metadata_json"):
                        video_data.update(json.loads(video_data["metadata_json"]))

                    # Cache in memory for fast access
                    self._memory_set(cache_key, video_data, self.TTL_VIEW_COUNTS)
                    return video_data

        return None

    def get_videos_batch(self, video_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """Get multiple videos from cache in batch.

        Args:
            video_ids: List of video IDs

        Returns:
            Dict mapping video_id to metadata (None if not cached)
        """
        results = {}
        uncached_ids = []

        # Check memory cache first
        for vid in video_ids:
            cached = self._memory_get(f"video:{vid}")
            if cached:
                results[vid] = cached
            else:
                uncached_ids.append(vid)

        # Batch query SQLite for remaining
        if uncached_ids:
            placeholders = ",".join("?" * len(uncached_ids))
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM videos WHERE video_id IN ({placeholders})",
                    uncached_ids
                ).fetchall()

                for row in rows:
                    video_data = dict(row)
                    if video_data.get("metadata_json"):
                        video_data.update(json.loads(video_data["metadata_json"]))

                    vid = video_data["video_id"]
                    results[vid] = video_data
                    self._memory_set(f"video:{vid}", video_data, self.TTL_VIEW_COUNTS)

        # Fill in None for truly uncached videos
        for vid in video_ids:
            if vid not in results:
                results[vid] = None

        return results

    def set_video(self, video_id: str, data: Dict) -> None:
        """Store video metadata in cache.

        Args:
            video_id: YouTube video ID
            data: Video metadata dict
        """
        cache_key = f"video:{video_id}"
        self._memory_set(cache_key, data, self.TTL_VIEW_COUNTS)

        # Store in SQLite
        with sqlite3.connect(self.db_path) as conn:
            # Extract known fields, store rest as JSON
            known_fields = {
                "title", "channel_id", "channel_name", "view_count",
                "like_count", "comment_count", "duration", "upload_date",
                "thumbnail_url", "is_short"
            }
            extra_data = {k: v for k, v in data.items() if k not in known_fields and k != "video_id"}

            conn.execute("""
                INSERT OR REPLACE INTO videos (
                    video_id, title, channel_id, channel_name, view_count,
                    like_count, comment_count, duration, upload_date,
                    thumbnail_url, is_short, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                video_id,
                data.get("title"),
                data.get("channel_id"),
                data.get("channel_name") or data.get("channel") or data.get("uploader"),
                data.get("view_count"),
                data.get("like_count"),
                data.get("comment_count"),
                data.get("duration"),
                data.get("upload_date"),
                data.get("thumbnail_url") or data.get("thumbnail"),
                data.get("is_short", False),
                json.dumps(extra_data) if extra_data else None
            ))

            # Also record view count history
            if data.get("view_count"):
                conn.execute("""
                    INSERT OR IGNORE INTO view_history (video_id, view_count)
                    VALUES (?, ?)
                """, (video_id, data["view_count"]))

    def set_videos_batch(self, videos: List[Dict]) -> None:
        """Store multiple videos in cache.

        Args:
            videos: List of video metadata dicts (must have 'id' or 'video_id')
        """
        for video in videos:
            video_id = video.get("video_id") or video.get("id")
            if video_id:
                self.set_video(video_id, video)

    # =========================================================================
    # Channel Operations
    # =========================================================================

    def get_channel(self, channel_id: str) -> Optional[Dict]:
        """Get channel metadata from cache."""
        cache_key = f"channel:{channel_id}"

        cached = self._memory_get(cache_key)
        if cached:
            return cached

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM channels WHERE channel_id = ?",
                (channel_id,)
            ).fetchone()

            if row:
                fetched_at = datetime.fromisoformat(row["updated_at"])
                age_seconds = (datetime.now() - fetched_at).total_seconds()

                if age_seconds < self.TTL_CHANNEL_INFO:
                    channel_data = dict(row)
                    if channel_data.get("metadata_json"):
                        channel_data.update(json.loads(channel_data["metadata_json"]))
                    self._memory_set(cache_key, channel_data, self.TTL_CHANNEL_INFO)
                    return channel_data

        return None

    def set_channel(self, channel_id: str, data: Dict) -> None:
        """Store channel metadata in cache."""
        cache_key = f"channel:{channel_id}"
        self._memory_set(cache_key, data, self.TTL_CHANNEL_INFO)

        with sqlite3.connect(self.db_path) as conn:
            extra_data = {k: v for k, v in data.items()
                         if k not in {"channel_id", "channel_name", "subscriber_count",
                                     "video_count", "average_views", "median_views"}}

            conn.execute("""
                INSERT OR REPLACE INTO channels (
                    channel_id, channel_name, subscriber_count, video_count,
                    average_views, median_views, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                channel_id,
                data.get("channel_name") or data.get("channel") or data.get("uploader"),
                data.get("subscriber_count") or data.get("channel_follower_count"),
                data.get("video_count"),
                data.get("average_views"),
                data.get("median_views"),
                json.dumps(extra_data) if extra_data else None
            ))

    def get_channel_video_ids(self, channel_id: str) -> Optional[List[str]]:
        """Get cached list of video IDs for a channel."""
        cache_key = f"channel_videos:{channel_id}"

        cached = self._memory_get(cache_key)
        if cached:
            return cached

        with sqlite3.connect(self.db_path) as conn:
            # Check if cache is fresh
            row = conn.execute("""
                SELECT MAX(fetched_at) as latest FROM channel_videos
                WHERE channel_id = ?
            """, (channel_id,)).fetchone()

            if row and row[0]:
                fetched_at = datetime.fromisoformat(row[0])
                age_seconds = (datetime.now() - fetched_at).total_seconds()

                if age_seconds < self.TTL_CHANNEL_VIDEOS:
                    video_ids = [r[0] for r in conn.execute("""
                        SELECT video_id FROM channel_videos
                        WHERE channel_id = ?
                        ORDER BY position
                    """, (channel_id,)).fetchall()]

                    if video_ids:
                        self._memory_set(cache_key, video_ids, self.TTL_CHANNEL_VIDEOS)
                        return video_ids

        return None

    def set_channel_video_ids(self, channel_id: str, video_ids: List[str]) -> None:
        """Cache list of video IDs for a channel."""
        cache_key = f"channel_videos:{channel_id}"
        self._memory_set(cache_key, video_ids, self.TTL_CHANNEL_VIDEOS)

        with sqlite3.connect(self.db_path) as conn:
            # Clear old entries
            conn.execute("DELETE FROM channel_videos WHERE channel_id = ?", (channel_id,))

            # Insert new entries
            conn.executemany("""
                INSERT INTO channel_videos (channel_id, video_id, position)
                VALUES (?, ?, ?)
            """, [(channel_id, vid, i) for i, vid in enumerate(video_ids)])

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            channel_count = conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
            history_count = conn.execute("SELECT COUNT(*) FROM view_history").fetchone()[0]

        return {
            "videos_cached": video_count,
            "channels_cached": channel_count,
            "view_history_entries": history_count,
            "memory_cache_entries": len(self._memory_cache),
            "db_path": self.db_path
        }

    def clear_old_data(self, days: int = 30) -> int:
        """Clear data older than N days.

        Args:
            days: Remove data older than this many days

        Returns:
            Number of rows deleted
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        deleted = 0
        with sqlite3.connect(self.db_path) as conn:
            deleted += conn.execute(
                "DELETE FROM videos WHERE updated_at < ?", (cutoff_str,)
            ).rowcount
            deleted += conn.execute(
                "DELETE FROM channels WHERE updated_at < ?", (cutoff_str,)
            ).rowcount
            deleted += conn.execute(
                "DELETE FROM channel_videos WHERE fetched_at < ?", (cutoff_str,)
            ).rowcount
            deleted += conn.execute(
                "DELETE FROM view_history WHERE recorded_at < ?", (cutoff_str,)
            ).rowcount

        logger.info(f"Cleared {deleted} old cache entries")
        return deleted


# Singleton instance for easy access
_cache_instance: Optional[VideoCache] = None


def get_video_cache() -> VideoCache:
    """Get the global VideoCache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = VideoCache()
    return _cache_instance
