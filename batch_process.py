#!/usr/bin/env python
"""Batch processing script for processing multiple videos with shared configuration."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from broll_processor import BRollProcessor
from models.user_preferences import UserPreferences
from utils.config import load_config, setup_logging


def load_batch_config(config_path: Path) -> dict[str, Any]:
    """Load batch configuration from JSON or YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with batch configuration

    Configuration format:
        {
            "videos": [
                "path/to/video1.mp4",
                "path/to/video2.mp4"
            ],
            "preferences": {
                "style": "cinematic",
                "avoid": "text overlays",
                "time_of_day": "golden hour"
            },
            "concurrency": 2
        }
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Batch config file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        return json.load(f)


async def process_batch(config_path: Path) -> None:
    """Process a batch of videos based on configuration file.

    Args:
        config_path: Path to batch configuration file (JSON or YAML)
    """
    # Load batch configuration
    batch_config = load_batch_config(config_path)

    video_paths = batch_config.get("videos", [])
    if not video_paths:
        logger.error("No videos specified in batch config")
        return

    # User preferences (optional)
    prefs_dict = batch_config.get("preferences", {})
    user_preferences = UserPreferences(**prefs_dict) if prefs_dict else None

    # Concurrency settings
    max_concurrent = batch_config.get("concurrency", 2)

    logger.info("🎬 Starting batch processing:")
    logger.info(f"   Videos: {len(video_paths)}")
    logger.info(f"   Concurrency: {max_concurrent}")
    if user_preferences:
        logger.info(f"   Preferences: {prefs_dict}")

    # Initialize processor
    system_config = load_config()
    processor = BRollProcessor(system_config)

    # Process videos with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(video_path: str, index: int) -> tuple[str, bool]:
        """Process a single video with semaphore limiting."""
        async with semaphore:
            logger.info(f"\n[{index}/{len(video_paths)}] Processing: {video_path}")
            try:
                await processor.process_video(video_path, user_preferences)
                logger.info(f"✅ [{index}/{len(video_paths)}] Completed: {video_path}")
                return (video_path, True)
            except Exception as e:
                logger.error(f"❌ [{index}/{len(video_paths)}] Failed: {video_path} - {e}")
                return (video_path, False)

    # Process all videos in parallel (with concurrency limit)
    tasks = [
        process_with_limit(video_path, i + 1) for i, video_path in enumerate(video_paths)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Summary
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful

    logger.info("\n" + "=" * 60)
    logger.info("📊 Batch Processing Complete")
    logger.info(f"   Total: {len(video_paths)} videos")
    logger.info(f"   ✅ Successful: {successful}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info("=" * 60)


async def main() -> None:
    """Main entry point for batch processing."""
    # Set up logging
    setup_logging("INFO")

    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <config_file.json|yaml>")
        print("\nExample config file (JSON):")
        print(
            """
{
  "videos": [
    "input/video1.mp4",
    "input/video2.mp4"
  ],
  "preferences": {
    "style": "cinematic",
    "avoid": "text overlays"
  },
  "concurrency": 2
}
        """
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])

    try:
        await process_batch(config_path)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Run async main
    asyncio.run(main())
