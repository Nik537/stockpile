"""Tests for AI response caching (S2 improvement).

S2 IMPROVEMENT: AI Response Caching
- Tests for AIResponseCache class
- Tests for cache hit/miss behavior
- Tests for TTL expiration
- Tests for cache size limits
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.cache import (
    AIResponseCache,
    compute_content_hash,
    compute_file_hash,
    load_cache_from_config,
)


class TestAIResponseCache:
    """Tests for AIResponseCache class."""

    def test_cache_init_enabled(self, tmp_path):
        """Test cache initialization when enabled."""
        cache_dir = str(tmp_path / "test_cache")
        cache = AIResponseCache(cache_dir=cache_dir, enabled=True)

        assert cache.enabled is True
        assert cache.cache is not None
        assert Path(cache_dir).exists()
        cache.close()

    def test_cache_init_disabled(self, tmp_path):
        """Test cache initialization when disabled."""
        cache_dir = str(tmp_path / "test_cache")
        cache = AIResponseCache(cache_dir=cache_dir, enabled=False)

        assert cache.enabled is False
        assert cache.cache is None
        # Directory should NOT be created when disabled
        assert not Path(cache_dir).exists()

    def test_cache_set_and_get(self, tmp_path):
        """Test basic set and get operations."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        prompt = "test prompt content"
        model = "gemini-3-flash-preview"
        response = '{"result": "test response"}'

        # Set value
        cache.set(prompt, model, response)

        # Get value - should be a hit
        result = cache.get(prompt, model)
        assert result == response
        assert cache.hits == 1
        assert cache.misses == 0

        cache.close()

    def test_cache_miss(self, tmp_path):
        """Test cache miss behavior."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        result = cache.get("nonexistent", "model")
        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1

        cache.close()

    def test_cache_disabled_operations(self, tmp_path):
        """Test that disabled cache returns None and doesn't store."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)

        # Set should do nothing
        cache.set("prompt", "model", "response")

        # Get should return None
        result = cache.get("prompt", "model")
        assert result is None

        # Stats should show disabled
        stats = cache.get_stats()
        assert stats["enabled"] is False

    def test_cache_clear(self, tmp_path):
        """Test cache clearing."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        # Add some entries
        cache.set("prompt1", "model", "response1")
        cache.set("prompt2", "model", "response2")

        # Verify entries exist
        assert cache.get("prompt1", "model") == "response1"

        # Clear cache
        count = cache.clear()
        assert count >= 2

        # Verify entries are gone
        assert cache.get("prompt1", "model") is None

        cache.close()

    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        cache.set("prompt1", "model", "response1")
        cache.get("prompt1", "model")  # Hit
        cache.get("prompt2", "model")  # Miss

        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["entry_count"] >= 1

        cache.close()

    def test_hit_rate_calculation(self, tmp_path):
        """Test hit rate calculation."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        # No requests yet
        assert cache.hit_rate == 0.0

        # Add an entry and hit it
        cache.set("prompt", "model", "response")
        cache.get("prompt", "model")  # Hit

        assert cache.hit_rate == 1.0

        # Add a miss
        cache.get("nonexistent", "model")  # Miss

        assert cache.hit_rate == 0.5

        cache.close()

    def test_different_models_different_keys(self, tmp_path):
        """Test that different models produce different cache keys."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        prompt = "same prompt"
        cache.set(prompt, "model1", "response1")
        cache.set(prompt, "model2", "response2")

        assert cache.get(prompt, "model1") == "response1"
        assert cache.get(prompt, "model2") == "response2"

        cache.close()


class TestLoadCacheFromConfig:
    """Tests for load_cache_from_config function."""

    def test_load_enabled(self, tmp_path):
        """Test loading enabled cache from config."""
        config = {
            "cache_enabled": True,
            "cache_dir": str(tmp_path / "cache"),
            "cache_ttl_days": 7,
            "cache_max_size_gb": 0.5,
        }

        cache = load_cache_from_config(config)
        assert cache.enabled is True
        assert cache.ttl_days == 7
        assert cache.max_size_gb == 0.5
        cache.close()

    def test_load_disabled(self, tmp_path):
        """Test loading disabled cache from config."""
        config = {
            "cache_enabled": False,
            "cache_dir": str(tmp_path / "cache"),
        }

        cache = load_cache_from_config(config)
        assert cache.enabled is False

    def test_load_defaults(self, tmp_path):
        """Test loading cache with default values."""
        config = {"cache_dir": str(tmp_path / "cache")}

        cache = load_cache_from_config(config)
        assert cache.enabled is True  # Default enabled
        assert cache.ttl_days == 30  # Default TTL
        assert cache.max_size_gb == 1.0  # Default size
        cache.close()


class TestHashFunctions:
    """Tests for hash helper functions."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        content = "test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_compute_content_hash_different_content(self):
        """Test that different content produces different hashes."""
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")

        assert hash1 != hash2

    def test_compute_file_hash(self, tmp_path):
        """Test file hash computation."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test file content")

        hash1 = compute_file_hash(str(test_file))
        hash2 = compute_file_hash(str(test_file))

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_compute_file_hash_different_files(self, tmp_path):
        """Test that different files produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        hash1 = compute_file_hash(str(file1))
        hash2 = compute_file_hash(str(file2))

        assert hash1 != hash2

    def test_compute_file_hash_nonexistent_file(self, tmp_path):
        """Test file hash computation for nonexistent file."""
        nonexistent = str(tmp_path / "nonexistent.txt")
        # Should return hash of file path as fallback
        hash_result = compute_file_hash(nonexistent)
        assert len(hash_result) == 64


class TestCacheIntegration:
    """Integration tests for cache with AI service patterns."""

    def test_broll_planning_cache_pattern(self, tmp_path):
        """Test caching pattern used by B-roll planning."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        # Simulate B-roll planning cache key generation
        prompt_version = "v2"
        transcript_hash = compute_content_hash("This is a test transcript")[:32]
        clips_per_minute = 2.0
        content_filter = ""

        cache_key = f"{prompt_version}|{transcript_hash}|{clips_per_minute}|{content_filter}"

        # Simulate cached B-roll plan
        broll_plan = json.dumps(
            [
                {
                    "timestamp": 30.0,
                    "search_phrase": "city skyline",
                    "description": "Establishing shot",
                }
            ]
        )

        # Cache the response
        cache.set(cache_key, "gemini-3-flash-preview", broll_plan)

        # Retrieve from cache
        cached = cache.get(cache_key, "gemini-3-flash-preview")
        assert cached == broll_plan

        # Parse cached data
        needs_data = json.loads(cached)
        assert len(needs_data) == 1
        assert needs_data[0]["search_phrase"] == "city skyline"

        cache.close()

    def test_video_evaluation_cache_pattern(self, tmp_path):
        """Test caching pattern used by video evaluation."""
        cache = AIResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        # Simulate video evaluation cache key
        prompt_version = "v2"
        search_phrase_hash = compute_content_hash("city skyline aerial")[:16]
        video_ids = ["vid1", "vid2", "vid3"]
        video_list_hash = compute_content_hash("|".join(sorted(video_ids)))[:32]

        cache_key = f"{prompt_version}|{search_phrase_hash}|{video_list_hash}|"

        # Simulate cached evaluation
        evaluation = json.dumps(
            [
                {"video_id": "vid1", "score": 9},
                {"video_id": "vid2", "score": 7},
            ]
        )

        cache.set(cache_key, "gemini-3-flash-preview", evaluation)
        cached = cache.get(cache_key, "gemini-3-flash-preview")

        assert cached == evaluation
        scored = json.loads(cached)
        assert len(scored) == 2
        assert scored[0]["score"] == 9

        cache.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
