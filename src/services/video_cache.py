"""Video metadata caching with SQLite persistence and in-memory TTL cache.

Implements recommendations #4 (database persistence) and #5 (TTL caching).
Dramatically reduces API calls by caching video/channel statistics.

Enhanced with:
- Turso cloud database support with embedded replicas
- Channel indexing for niche-based discovery
- Fallback to local SQLite when Turso not configured
"""

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import libsql for Turso support
try:
    import libsql_experimental as libsql
    LIBSQL_AVAILABLE = True
except ImportError:
    LIBSQL_AVAILABLE = False
    logger.debug("libsql not available, using local SQLite only")


@dataclass
class CacheEntry:
    """In-memory cache entry with TTL."""
    value: Any
    expires_at: float


class VideoCache:
    """Multi-tier caching for YouTube video and channel data.

    Tier 1: In-memory TTL cache (fast, volatile)
    Tier 2: SQLite/Turso database (persistent, slower)

    TTL Strategy (based on data volatility):
    - View counts: 15 minutes (frequently changing)
    - Video metadata: 24 hours (stable)
    - Channel info: 7 days (rarely changes)

    Turso Integration:
    - When configured, uses Turso as cloud database with local replica
    - Automatic sync for offline support
    - Falls back to local SQLite when Turso not available
    """

    # Default TTLs in seconds
    TTL_VIEW_COUNTS = 900        # 15 minutes
    TTL_VIDEO_METADATA = 86400   # 24 hours
    TTL_CHANNEL_INFO = 604800    # 7 days
    TTL_CHANNEL_VIDEOS = 3600    # 1 hour (list of video IDs)

    def __init__(
        self,
        db_path: Optional[str] = None,
        turso_url: Optional[str] = None,
        turso_auth_token: Optional[str] = None,
        use_cloud: bool = False,
    ):
        """Initialize the cache.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
            turso_url: Turso database URL (e.g., libsql://db-name.turso.io)
            turso_auth_token: Turso authentication token
            use_cloud: Whether to use cloud sync when Turso is configured
        """
        if db_path is None:
            cache_dir = Path.home() / ".stockpile" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(cache_dir / "video_cache.db")

        self.db_path = db_path
        self.turso_url = turso_url
        self.turso_auth_token = turso_auth_token
        self.use_cloud = use_cloud and LIBSQL_AVAILABLE and turso_url and turso_auth_token

        self._memory_cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._conn: Optional[Union[sqlite3.Connection, Any]] = None

        # Initialize database
        self._init_db()

        if self.use_cloud:
            logger.info(f"VideoCache initialized with Turso cloud sync at {db_path}")
        else:
            logger.info(f"VideoCache initialized locally at {db_path}")

    def _get_connection(self) -> Union[sqlite3.Connection, Any]:
        """Get database connection (Turso or SQLite)."""
        if self._conn is not None:
            return self._conn

        if self.use_cloud and LIBSQL_AVAILABLE:
            try:
                self._conn = libsql.connect(
                    database=self.db_path,
                    sync_url=self.turso_url,
                    auth_token=self.turso_auth_token,
                )
                # Initial sync from cloud
                self._conn.sync()
                logger.debug("Connected to Turso with embedded replica")
            except Exception as e:
                logger.warning(f"Turso connection failed: {e}, falling back to local SQLite")
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)

        return self._conn

    def sync_to_cloud(self) -> bool:
        """Sync local changes to Turso cloud.

        Returns:
            True if sync succeeded, False otherwise
        """
        if not self.use_cloud:
            return False

        try:
            conn = self._get_connection()
            if hasattr(conn, 'sync'):
                conn.sync()
                logger.debug("Synced to Turso cloud")
                return True
        except Exception as e:
            logger.error(f"Cloud sync failed: {e}")

        return False

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = self._get_connection()

        # Use executescript for SQLite, execute for libsql
        schema = """
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

            -- View count history for trend analysis
            CREATE TABLE IF NOT EXISTS view_history (
                video_id TEXT,
                view_count INTEGER,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (video_id, recorded_at)
            );

            -- Channel niche tagging for discovery (Phase 2 enhancement)
            CREATE TABLE IF NOT EXISTS channel_niches (
                channel_id TEXT,
                niche TEXT,
                confidence REAL DEFAULT 1.0,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (channel_id, niche)
            );

            -- Channel index for fast lookup (Phase 2 enhancement)
            CREATE TABLE IF NOT EXISTS channel_index (
                channel_id TEXT PRIMARY KEY,
                channel_name TEXT,
                subscriber_count INTEGER,
                last_analyzed TIMESTAMP,
                total_videos INTEGER,
                average_views REAL,
                median_views REAL,
                status TEXT DEFAULT 'active'
            );
        """

        # Execute each statement separately for compatibility
        for statement in schema.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except Exception as e:
                    # Ignore "table already exists" errors
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema statement failed: {e}")

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_videos_fetched ON videos(fetched_at)",
            "CREATE INDEX IF NOT EXISTS idx_channel_videos_fetched ON channel_videos(fetched_at)",
            "CREATE INDEX IF NOT EXISTS idx_channel_niches_niche ON channel_niches(niche)",
            "CREATE INDEX IF NOT EXISTS idx_channel_index_subs ON channel_index(subscriber_count)",
        ]

        for idx in indexes:
            try:
                conn.execute(idx)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index creation failed: {e}")

        conn.commit()

        # Sync to cloud if using Turso
        if self.use_cloud:
            self.sync_to_cloud()

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

        Checks in-memory cache first, then SQLite/Turso.

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

        # Check database
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?",
                (video_id,)
            )
            row = cursor.fetchone()

            if row:
                # Convert row to dict
                columns = [description[0] for description in cursor.description]
                video_data = dict(zip(columns, row))

                # Check if data is fresh enough
                updated_at = video_data.get("updated_at", "")
                if updated_at:
                    try:
                        fetched_at = datetime.fromisoformat(str(updated_at))
                        age_seconds = (datetime.now() - fetched_at).total_seconds()

                        if age_seconds < self.TTL_VIDEO_METADATA:
                            # Parse metadata JSON if present
                            if video_data.get("metadata_json"):
                                video_data.update(json.loads(video_data["metadata_json"]))

                            # Cache in memory for fast access
                            self._memory_set(cache_key, video_data, self.TTL_VIEW_COUNTS)
                            return video_data
                    except ValueError:
                        pass
        except Exception as e:
            logger.warning(f"Error fetching video {video_id}: {e}")

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

        # Batch query database for remaining
        if uncached_ids:
            placeholders = ",".join("?" * len(uncached_ids))
            conn = self._get_connection()

            try:
                cursor = conn.execute(
                    f"SELECT * FROM videos WHERE video_id IN ({placeholders})",
                    uncached_ids
                )
                columns = [description[0] for description in cursor.description]

                for row in cursor.fetchall():
                    video_data = dict(zip(columns, row))
                    if video_data.get("metadata_json"):
                        video_data.update(json.loads(video_data["metadata_json"]))

                    vid = video_data["video_id"]
                    results[vid] = video_data
                    self._memory_set(f"video:{vid}", video_data, self.TTL_VIEW_COUNTS)
            except Exception as e:
                logger.warning(f"Error batch fetching videos: {e}")

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

        # Store in database
        conn = self._get_connection()
        try:
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

            conn.commit()
        except Exception as e:
            logger.warning(f"Error storing video {video_id}: {e}")

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

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM channels WHERE channel_id = ?",
                (channel_id,)
            )
            row = cursor.fetchone()

            if row:
                columns = [description[0] for description in cursor.description]
                channel_data = dict(zip(columns, row))

                updated_at = channel_data.get("updated_at", "")
                if updated_at:
                    try:
                        fetched_at = datetime.fromisoformat(str(updated_at))
                        age_seconds = (datetime.now() - fetched_at).total_seconds()

                        if age_seconds < self.TTL_CHANNEL_INFO:
                            if channel_data.get("metadata_json"):
                                channel_data.update(json.loads(channel_data["metadata_json"]))
                            self._memory_set(cache_key, channel_data, self.TTL_CHANNEL_INFO)
                            return channel_data
                    except ValueError:
                        pass
        except Exception as e:
            logger.warning(f"Error fetching channel {channel_id}: {e}")

        return None

    def set_channel(self, channel_id: str, data: Dict) -> None:
        """Store channel metadata in cache."""
        cache_key = f"channel:{channel_id}"
        self._memory_set(cache_key, data, self.TTL_CHANNEL_INFO)

        conn = self._get_connection()
        try:
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
            conn.commit()
        except Exception as e:
            logger.warning(f"Error storing channel {channel_id}: {e}")

    def get_channel_video_ids(self, channel_id: str) -> Optional[List[str]]:
        """Get cached list of video IDs for a channel."""
        cache_key = f"channel_videos:{channel_id}"

        cached = self._memory_get(cache_key)
        if cached:
            return cached

        conn = self._get_connection()
        try:
            # Check if cache is fresh
            row = conn.execute("""
                SELECT MAX(fetched_at) as latest FROM channel_videos
                WHERE channel_id = ?
            """, (channel_id,)).fetchone()

            if row and row[0]:
                try:
                    fetched_at = datetime.fromisoformat(str(row[0]))
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
                except ValueError:
                    pass
        except Exception as e:
            logger.warning(f"Error fetching channel videos for {channel_id}: {e}")

        return None

    def set_channel_video_ids(self, channel_id: str, video_ids: List[str]) -> None:
        """Cache list of video IDs for a channel."""
        cache_key = f"channel_videos:{channel_id}"
        self._memory_set(cache_key, video_ids, self.TTL_CHANNEL_VIDEOS)

        conn = self._get_connection()
        try:
            # Clear old entries
            conn.execute("DELETE FROM channel_videos WHERE channel_id = ?", (channel_id,))

            # Insert new entries
            for i, vid in enumerate(video_ids):
                conn.execute("""
                    INSERT INTO channel_videos (channel_id, video_id, position)
                    VALUES (?, ?, ?)
                """, (channel_id, vid, i))

            conn.commit()
        except Exception as e:
            logger.warning(f"Error storing channel videos for {channel_id}: {e}")

    # =========================================================================
    # Channel Index Operations (Phase 2 Enhancement)
    # =========================================================================

    def add_channel_to_index(
        self,
        channel_id: str,
        channel_name: str,
        subscriber_count: int,
        total_videos: int = 0,
        average_views: float = 0.0,
        median_views: float = 0.0,
        niches: Optional[List[str]] = None,
    ) -> None:
        """Add or update a channel in the index.

        Args:
            channel_id: YouTube channel ID
            channel_name: Channel display name
            subscriber_count: Subscriber count
            total_videos: Number of videos analyzed
            average_views: Average views per video
            median_views: Median views per video
            niches: List of niche tags for this channel
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO channel_index (
                    channel_id, channel_name, subscriber_count, last_analyzed,
                    total_videos, average_views, median_views, status
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, 'active')
            """, (
                channel_id, channel_name, subscriber_count,
                total_videos, average_views, median_views
            ))

            # Add niche tags
            if niches:
                for niche in niches:
                    conn.execute("""
                        INSERT OR REPLACE INTO channel_niches (channel_id, niche, confidence)
                        VALUES (?, ?, 1.0)
                    """, (channel_id, niche.lower()))

            conn.commit()

            # Sync to cloud periodically
            if self.use_cloud:
                self.sync_to_cloud()

        except Exception as e:
            logger.warning(f"Error adding channel to index: {e}")

    def get_channels_by_niche(
        self,
        niche: str,
        min_subs: Optional[int] = None,
        max_subs: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get channels tagged with a specific niche.

        Args:
            niche: Niche tag to search for
            min_subs: Minimum subscriber count filter
            max_subs: Maximum subscriber count filter
            limit: Maximum number of results

        Returns:
            List of channel dicts from index
        """
        conn = self._get_connection()
        channels = []

        try:
            query = """
                SELECT ci.* FROM channel_index ci
                JOIN channel_niches cn ON ci.channel_id = cn.channel_id
                WHERE cn.niche = ?
                AND ci.status = 'active'
            """
            params: List[Any] = [niche.lower()]

            if min_subs:
                query += " AND ci.subscriber_count >= ?"
                params.append(min_subs)
            if max_subs:
                query += " AND ci.subscriber_count <= ?"
                params.append(max_subs)

            query += " ORDER BY ci.subscriber_count DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, tuple(params))
            columns = [description[0] for description in cursor.description]

            for row in cursor.fetchall():
                channels.append(dict(zip(columns, row)))

        except Exception as e:
            logger.warning(f"Error fetching channels by niche: {e}")

        return channels

    def get_stale_channels(self, days_old: int = 7, limit: int = 100) -> List[str]:
        """Get channel IDs that haven't been analyzed recently.

        Args:
            days_old: Consider channels stale after this many days
            limit: Maximum number of results

        Returns:
            List of channel IDs needing refresh
        """
        conn = self._get_connection()
        cutoff = datetime.now() - timedelta(days=days_old)

        try:
            rows = conn.execute("""
                SELECT channel_id FROM channel_index
                WHERE last_analyzed < ?
                AND status = 'active'
                ORDER BY last_analyzed ASC
                LIMIT ?
            """, (cutoff.isoformat(), limit)).fetchall()

            return [row[0] for row in rows]
        except Exception as e:
            logger.warning(f"Error fetching stale channels: {e}")
            return []

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        conn = self._get_connection()

        try:
            video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            channel_count = conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
            history_count = conn.execute("SELECT COUNT(*) FROM view_history").fetchone()[0]
            index_count = conn.execute("SELECT COUNT(*) FROM channel_index").fetchone()[0]
            niche_count = conn.execute("SELECT COUNT(DISTINCT niche) FROM channel_niches").fetchone()[0]
        except Exception:
            video_count = channel_count = history_count = index_count = niche_count = 0

        return {
            "videos_cached": video_count,
            "channels_cached": channel_count,
            "view_history_entries": history_count,
            "indexed_channels": index_count,
            "unique_niches": niche_count,
            "memory_cache_entries": len(self._memory_cache),
            "db_path": self.db_path,
            "cloud_sync_enabled": self.use_cloud,
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
        conn = self._get_connection()

        try:
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
            conn.commit()

            # Sync cleanup to cloud
            if self.use_cloud:
                self.sync_to_cloud()

        except Exception as e:
            logger.warning(f"Error clearing old data: {e}")

        logger.info(f"Cleared {deleted} old cache entries")
        return deleted


# Singleton instance for easy access
_cache_instance: Optional[VideoCache] = None


def get_video_cache(
    turso_url: Optional[str] = None,
    turso_auth_token: Optional[str] = None,
    use_cloud: Optional[bool] = None,
) -> VideoCache:
    """Get the global VideoCache instance.

    Args:
        turso_url: Turso database URL (uses config if not provided)
        turso_auth_token: Turso auth token (uses config if not provided)
        use_cloud: Whether to use cloud sync (uses config if not provided)

    Returns:
        VideoCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        # Try to get config values
        try:
            from utils.config import load_config
            config = load_config()

            if turso_url is None:
                turso_url = config.get("turso_database_url")
            if turso_auth_token is None:
                turso_auth_token = config.get("turso_auth_token")
            if use_cloud is None:
                use_cloud = config.get("use_cloud_cache", False)
        except Exception:
            pass

        _cache_instance = VideoCache(
            turso_url=turso_url,
            turso_auth_token=turso_auth_token,
            use_cloud=use_cloud or False,
        )

    return _cache_instance
