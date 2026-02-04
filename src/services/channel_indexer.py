"""Channel Indexer Service for Background Niche Discovery.

Pre-builds a database of YouTube channels organized by niche/topic
to enable fast outlier searches without API-heavy discovery.

Features:
- Index channels by niche/topic
- Update stale channel data
- Support for popular niche presets
- Integration with YouTube API and video cache
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from services.video_cache import VideoCache, get_video_cache

logger = logging.getLogger(__name__)

# Try to import YouTube API service
try:
    from services.youtube_api_service import YouTubeAPIService, get_youtube_api_service
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False


# Popular niches for preset indexing
POPULAR_NICHES = [
    # Tech & Business
    "tech reviews",
    "programming tutorials",
    "startup advice",
    "personal finance",
    "productivity tips",
    # Creative & Lifestyle
    "cooking tutorials",
    "fitness workouts",
    "travel vlog",
    "photography tips",
    "music production",
    # Entertainment & Gaming
    "gaming commentary",
    "movie reviews",
    "comedy sketches",
    "animation",
    # Education & Self-Improvement
    "self improvement",
    "language learning",
    "science explained",
    "history documentaries",
    "educational content",
    # Niche Interests
    "woodworking",
    "car reviews",
    "home renovation",
    "gardening tips",
    "pet care",
]


@dataclass
class IndexResult:
    """Result of an indexing operation."""
    niche: str
    channels_indexed: int
    channels_skipped: int
    duration_seconds: float
    errors: List[str]


class ChannelIndexer:
    """Service for building and maintaining a channel index by niche.

    Uses YouTube Data API to discover channels in specific niches
    and stores them in the video cache for fast outlier searching.
    """

    def __init__(
        self,
        youtube_api_key: Optional[str] = None,
        cache: Optional[VideoCache] = None,
        min_subscriber_count: int = 1000,
        max_subscriber_count: int = 10_000_000,
    ):
        """Initialize the channel indexer.

        Args:
            youtube_api_key: YouTube Data API key (uses config if not provided)
            cache: VideoCache instance (uses global if not provided)
            min_subscriber_count: Minimum subscribers for indexing
            max_subscriber_count: Maximum subscribers for indexing
        """
        self.cache = cache or get_video_cache()
        self.min_subs = min_subscriber_count
        self.max_subs = max_subscriber_count

        # Initialize YouTube API
        self._youtube_api: Optional[YouTubeAPIService] = None
        if YOUTUBE_API_AVAILABLE:
            if youtube_api_key:
                self._youtube_api = YouTubeAPIService(youtube_api_key)
            else:
                self._youtube_api = get_youtube_api_service()

        if not self._youtube_api:
            logger.warning("YouTube API not available - channel indexing will be limited")

    def index_niche(
        self,
        niche: str,
        max_channels: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> IndexResult:
        """Index channels for a specific niche/topic.

        Searches YouTube for channels related to the niche and stores
        them in the cache with niche tags.

        Args:
            niche: Niche/topic to index (e.g., "tech reviews")
            max_channels: Maximum number of channels to index
            on_progress: Optional callback for progress (indexed, total)

        Returns:
            IndexResult with statistics
        """
        start_time = time.time()
        errors: List[str] = []
        channels_indexed = 0
        channels_skipped = 0

        logger.info(f"Indexing niche: {niche} (max {max_channels} channels)")

        if not self._youtube_api:
            errors.append("YouTube API not available")
            return IndexResult(
                niche=niche,
                channels_indexed=0,
                channels_skipped=0,
                duration_seconds=0,
                errors=errors,
            )

        try:
            # Search for channels in this niche
            channels = self._youtube_api.search_channels(
                topic=niche,
                max_results=max_channels,
                min_subscriber_count=self.min_subs,
                max_subscriber_count=self.max_subs,
            )

            total = len(channels)
            logger.info(f"Found {total} channels for niche: {niche}")

            for i, channel in enumerate(channels):
                try:
                    # Add to cache index with niche tag
                    self.cache.add_channel_to_index(
                        channel_id=channel.channel_id,
                        channel_name=channel.channel_name,
                        subscriber_count=channel.subscriber_count,
                        total_videos=channel.video_count,
                        average_views=0.0,  # Will be populated on first outlier search
                        median_views=0.0,
                        niches=[niche],
                    )
                    channels_indexed += 1

                    if on_progress:
                        on_progress(channels_indexed, total)

                except Exception as e:
                    logger.warning(f"Error indexing channel {channel.channel_id}: {e}")
                    channels_skipped += 1
                    errors.append(f"Channel {channel.channel_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error searching for niche {niche}: {e}")
            errors.append(str(e))

        duration = time.time() - start_time
        logger.info(
            f"Indexed {channels_indexed} channels for '{niche}' in {duration:.1f}s "
            f"({channels_skipped} skipped)"
        )

        return IndexResult(
            niche=niche,
            channels_indexed=channels_indexed,
            channels_skipped=channels_skipped,
            duration_seconds=duration,
            errors=errors,
        )

    def index_preset(
        self,
        preset: str = "popular",
        max_channels_per_niche: int = 50,
        on_niche_complete: Optional[Callable[[str, IndexResult], None]] = None,
    ) -> Dict[str, IndexResult]:
        """Index multiple niches from a preset.

        Args:
            preset: Preset name ("popular" for default niches)
            max_channels_per_niche: Max channels per niche
            on_niche_complete: Optional callback when each niche completes

        Returns:
            Dict mapping niche to IndexResult
        """
        if preset == "popular":
            niches = POPULAR_NICHES
        else:
            logger.warning(f"Unknown preset: {preset}, using popular")
            niches = POPULAR_NICHES

        results = {}
        total_niches = len(niches)

        logger.info(f"Indexing {total_niches} niches from '{preset}' preset")

        for i, niche in enumerate(niches, 1):
            logger.info(f"[{i}/{total_niches}] Indexing niche: {niche}")

            result = self.index_niche(niche, max_channels=max_channels_per_niche)
            results[niche] = result

            if on_niche_complete:
                on_niche_complete(niche, result)

            # Small delay between niches to avoid rate limiting
            time.sleep(1)

        return results

    def update_stale_channels(
        self,
        days_old: int = 7,
        max_channels: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Update channels that haven't been analyzed recently.

        Re-fetches metadata for stale channels and updates their
        statistics in the index.

        Args:
            days_old: Consider channels stale after this many days
            max_channels: Maximum number of channels to update
            on_progress: Optional callback for progress (updated, total)

        Returns:
            Number of channels updated
        """
        if not self._youtube_api:
            logger.warning("YouTube API not available - cannot update channels")
            return 0

        # Get stale channel IDs from cache
        stale_ids = self.cache.get_stale_channels(days_old=days_old, limit=max_channels)

        if not stale_ids:
            logger.info("No stale channels to update")
            return 0

        logger.info(f"Updating {len(stale_ids)} stale channels")

        updated = 0
        total = len(stale_ids)

        # Batch fetch channel stats
        try:
            channel_details = self._youtube_api.get_channel_details(stale_ids)

            for channel_id, details in channel_details.items():
                try:
                    # Update in index
                    self.cache.add_channel_to_index(
                        channel_id=details.channel_id,
                        channel_name=details.channel_name,
                        subscriber_count=details.subscriber_count,
                        total_videos=details.video_count,
                    )
                    updated += 1

                    if on_progress:
                        on_progress(updated, total)

                except Exception as e:
                    logger.warning(f"Error updating channel {channel_id}: {e}")

        except Exception as e:
            logger.error(f"Error fetching channel details: {e}")

        logger.info(f"Updated {updated}/{total} stale channels")
        return updated

    def get_indexed_channels(
        self,
        niche: str,
        min_subs: Optional[int] = None,
        max_subs: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get channels from the index for a specific niche.

        Args:
            niche: Niche to search for
            min_subs: Minimum subscriber count filter
            max_subs: Maximum subscriber count filter
            limit: Maximum number of results

        Returns:
            List of channel dicts
        """
        return self.cache.get_channels_by_niche(
            niche=niche,
            min_subs=min_subs or self.min_subs,
            max_subs=max_subs or self.max_subs,
            limit=limit,
        )

    def get_index_stats(self) -> Dict:
        """Get statistics about the channel index.

        Returns:
            Dict with index statistics
        """
        cache_stats = self.cache.get_stats()

        return {
            "indexed_channels": cache_stats.get("indexed_channels", 0),
            "unique_niches": cache_stats.get("unique_niches", 0),
            "cloud_sync_enabled": cache_stats.get("cloud_sync_enabled", False),
            "api_available": self._youtube_api is not None,
            "api_quota_used": self._youtube_api.quota_used if self._youtube_api else 0,
        }


# Factory function
def get_channel_indexer(
    youtube_api_key: Optional[str] = None,
    cache: Optional[VideoCache] = None,
) -> ChannelIndexer:
    """Get a ChannelIndexer instance.

    Args:
        youtube_api_key: YouTube API key (uses config if not provided)
        cache: VideoCache instance (uses global if not provided)

    Returns:
        ChannelIndexer instance
    """
    return ChannelIndexer(youtube_api_key=youtube_api_key, cache=cache)
