"""AI response caching to avoid redundant API calls and save costs.

S2 IMPROVEMENT: AI Response Caching
- Cache Gemini API responses keyed by content hash
- 100% cost savings on re-runs with same content
- TTL-based invalidation for stale cache entries
- Automatic size management with LRU eviction
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

try:
    from diskcache import Cache
    DISKCACHE_AVAILABLE = True
except ImportError:
    Cache = None  # type: ignore[assignment,misc]
    DISKCACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIResponseCache:
    """Cache for AI API responses to reduce costs and improve speed.

    Caches responses from Gemini and other AI services to avoid redundant API calls.
    Particularly useful when re-processing the same video or similar content.

    S2 IMPROVEMENT: Provides 100% cost savings on re-runs by caching:
    - B-roll planning responses (keyed by transcript hash)
    - Video evaluation responses (keyed by video URL + evaluation prompt)
    - Clip extraction analysis (keyed by video file hash)

    Example usage:
        cache = AIResponseCache()

        # Check cache before API call
        cached = cache.get("my_prompt", "gemini-3-flash-preview")
        if cached:
            return parse_cached_response(cached)

        # Make API call if not cached
        response = ai_service.call_api(prompt)

        # Store in cache for future use
        cache.set("my_prompt", "gemini-3-flash-preview", response)
    """

    def __init__(
        self,
        cache_dir: str = ".cache/ai_responses",
        ttl_days: int = 30,
        max_size_gb: float = 1.0,
        enabled: bool = True,
    ):
        """Initialize AI response cache.

        Args:
            cache_dir: Directory for cache storage
            ttl_days: Time-to-live for cache entries in days (default: 30)
            max_size_gb: Maximum cache size in GB (default: 1.0)
            enabled: Whether caching is enabled (default: True)
        """
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.max_size_gb = max_size_gb

        # Statistics tracking
        self.hits = 0
        self.misses = 0

        if not enabled or not DISKCACHE_AVAILABLE:
            if not DISKCACHE_AVAILABLE:
                logger.info("diskcache not installed, AI response caching disabled")
            else:
                logger.info("AI response caching is DISABLED")
            self.cache = None
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # TTL in seconds
        self.default_ttl = ttl_days * 24 * 60 * 60

        # Max cache size in bytes
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

        # Initialize diskcache with size limit
        self.cache = Cache(
            str(self.cache_dir),
            size_limit=max_size_bytes,
            eviction_policy="least-recently-used",
        )

        logger.info(
            f"Initialized AI cache at {cache_dir} (TTL: {ttl_days} days, "
            f"Max size: {max_size_gb}GB)"
        )

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached AI response if exists.

        Args:
            prompt: The prompt sent to the AI (or cache key content)
            model: Model name (e.g., 'gemini-3-flash-preview')

        Returns:
            Cached response text or None if not found
        """
        if not self.enabled or self.cache is None:
            return None

        cache_key = self._generate_key(prompt, model)

        try:
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                self.hits += 1
                logger.debug(f"Cache HIT for {model} (hit rate: {self.hit_rate:.1%})")
                return cached_value
            else:
                self.misses += 1
                logger.debug(f"Cache MISS for {model} (hit rate: {self.hit_rate:.1%})")
                return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            self.misses += 1
            return None

    def set(
        self, prompt: str, model: str, response: str, ttl: Optional[int] = None
    ) -> None:
        """Cache an AI response.

        Args:
            prompt: The prompt sent to the AI (or cache key content)
            model: Model name (e.g., 'gemini-3-flash-preview')
            response: The AI's response text
            ttl: Optional custom TTL in seconds (default: use configured TTL)
        """
        if not self.enabled or self.cache is None:
            return

        cache_key = self._generate_key(prompt, model)
        ttl_seconds = ttl if ttl is not None else self.default_ttl

        try:
            self.cache.set(cache_key, response, expire=ttl_seconds)
            logger.debug(f"Cached response for {model} (TTL: {ttl_seconds}s)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries that were cleared
        """
        if not self.enabled or self.cache is None:
            logger.info("Cache is disabled, nothing to clear")
            return 0

        try:
            entry_count = len(self.cache)
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info(f"Cache cleared successfully ({entry_count} entries removed)")
            return entry_count
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or self.cache is None:
            return {
                "enabled": False,
                "total_requests": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "size_bytes": 0,
                "size_mb": 0.0,
                "entry_count": 0,
                "ttl_days": self.ttl_days,
                "max_size_gb": self.max_size_gb,
            }

        return {
            "enabled": True,
            "total_requests": self.hits + self.misses,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "size_bytes": self.cache.volume(),
            "size_mb": round(self.cache.volume() / (1024 * 1024), 2),
            "entry_count": len(self.cache),
            "ttl_days": self.ttl_days,
            "max_size_gb": self.max_size_gb,
            "cache_dir": str(self.cache_dir),
        }

    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        if not stats.get("enabled", True):
            logger.info("Cache Stats - DISABLED")
            return

        logger.info(
            f"Cache Stats - Requests: {stats['total_requests']}, "
            f"Hit Rate: {stats['hit_rate']:.1%}, "
            f"Size: {stats['size_mb']}MB, "
            f"Entries: {stats['entry_count']}"
        )

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _generate_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model.

        Uses SHA-256 hash of prompt + model to create a unique,
        fixed-length key that works as a filesystem-safe identifier.

        Args:
            prompt: The prompt text
            model: Model name

        Returns:
            SHA-256 hash as hex string
        """
        # Combine prompt and model for unique key
        combined = f"{model}:{prompt}"

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(combined.encode("utf-8"))
        return hash_obj.hexdigest()

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Note: diskcache automatically handles expiration,
        but this can be called to force cleanup.

        Returns:
            Number of expired entries that were cleaned up
        """
        if not self.enabled or self.cache is None:
            return 0

        try:
            # Iterate through cache and check expiration
            expired_count = 0
            for key in list(self.cache.iterkeys()):
                # Touch the key to check if expired
                if self.cache.get(key) is None:
                    expired_count += 1

            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
            return 0

    def close(self) -> None:
        """Close the cache and release resources."""
        if self.cache is not None:
            try:
                self.cache.close()
                logger.debug("Cache closed successfully")
            except Exception as e:
                logger.warning(f"Error closing cache: {e}")


def load_cache_from_config(config: dict) -> AIResponseCache:
    """Load cache from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        AIResponseCache instance (may be disabled based on config)
    """
    enabled = config.get("cache_enabled", True)
    cache_dir = config.get("cache_dir", ".cache/ai_responses")
    ttl_days = config.get("cache_ttl_days", 30)
    max_size_gb = config.get("cache_max_size_gb", 1.0)

    return AIResponseCache(
        cache_dir=cache_dir,
        ttl_days=ttl_days,
        max_size_gb=max_size_gb,
        enabled=enabled,
    )


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file for cache keying.

    Used for caching video analysis results by file content.

    Args:
        file_path: Path to file to hash
        chunk_size: Size of chunks to read (default: 8KB)

    Returns:
        SHA-256 hash as hex string
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute file hash for {file_path}: {e}")
        # Return a hash of the file path as fallback
        return hashlib.sha256(file_path.encode()).hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of string content for cache keying.

    Args:
        content: String content to hash

    Returns:
        SHA-256 hash as hex string
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
