"""AI response caching to avoid redundant API calls and save costs."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from diskcache import Cache

logger = logging.getLogger(__name__)


class AIResponseCache:
    """Cache for AI API responses to reduce costs and improve speed.

    Caches responses from Gemini and other AI services to avoid redundant API calls.
    Particularly useful when re-processing the same video or similar content.

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
    ):
        """Initialize AI response cache.

        Args:
            cache_dir: Directory for cache storage
            ttl_days: Time-to-live for cache entries in days (default: 30)
            max_size_gb: Maximum cache size in GB (default: 1.0)
        """
        self.cache_dir = Path(cache_dir)
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

        # Statistics tracking
        self.hits = 0
        self.misses = 0

        logger.info(
            f"Initialized AI cache at {cache_dir} (TTL: {ttl_days} days, "
            f"Max size: {max_size_gb}GB)"
        )

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached AI response if exists.

        Args:
            prompt: The prompt sent to the AI
            model: Model name (e.g., 'gemini-3-flash-preview')

        Returns:
            Cached response text or None if not found
        """
        cache_key = self._generate_key(prompt, model)

        try:
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                self.hits += 1
                logger.debug(
                    f"Cache HIT for {model} (hit rate: {self.hit_rate:.1%})"
                )
                return cached_value
            else:
                self.misses += 1
                logger.debug(
                    f"Cache MISS for {model} (hit rate: {self.hit_rate:.1%})"
                )
                return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            self.misses += 1
            return None

    def set(self, prompt: str, model: str, response: str, ttl: Optional[int] = None):
        """Cache an AI response.

        Args:
            prompt: The prompt sent to the AI
            model: Model name (e.g., 'gemini-3-flash-preview')
            response: The AI's response text
            ttl: Optional custom TTL in seconds (default: use configured TTL)
        """
        cache_key = self._generate_key(prompt, model)
        ttl_seconds = ttl if ttl is not None else self.default_ttl

        try:
            self.cache.set(cache_key, response, expire=ttl_seconds)
            logger.debug(f"Cached response for {model} (TTL: {ttl_seconds}s)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        """Clear all cached entries."""
        try:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_requests": self.hits + self.misses,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "size_bytes": self.cache.volume(),
            "size_mb": round(self.cache.volume() / (1024 * 1024), 2),
            "entry_count": len(self.cache),
        }

    def log_stats(self):
        """Log cache statistics."""
        stats = self.get_stats()
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

    def cleanup_expired(self):
        """Remove expired entries from cache.

        Note: diskcache automatically handles expiration,
        but this can be called to force cleanup.
        """
        try:
            # Iterate through cache and check expiration
            expired_count = 0
            for key in list(self.cache.iterkeys()):
                # Touch the key to check if expired
                if self.cache.get(key) is None:
                    expired_count += 1

            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired cache entries")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")


def load_cache_from_config(config: dict) -> Optional[AIResponseCache]:
    """Load cache from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        AIResponseCache instance or None if disabled
    """
    if not config.get("cache_enabled", True):
        logger.info("AI response caching is disabled")
        return None

    cache_dir = config.get("cache_dir", ".cache/ai_responses")
    ttl_days = config.get("cache_ttl_days", 30)
    max_size_gb = config.get("cache_max_size_gb", 1.0)

    return AIResponseCache(
        cache_dir=cache_dir, ttl_days=ttl_days, max_size_gb=max_size_gb
    )
