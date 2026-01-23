#!/usr/bin/env python3
"""Non-interactive runner with predefined preferences."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from broll_processor import BRollProcessor
from models.user_preferences import UserPreferences
from utils.config import setup_logging, load_config


async def main():
    """Run stockpile with predefined preferences."""
    setup_logging()

    # Video path
    video_path = "/Users/niknoavak/Desktop/YT/stockpile/input/descriptS&C.mp4"

    # Create preferences based on user's answers
    preferences = UserPreferences(
        visual_style="MMA fighting, men only, martial arts focused",
        pace="fast",
        time_of_day="any",
        content_filter="men only, MMA, fighting, martial arts"
    )

    print("\n" + "="*70)
    print("STOCKPILE - Processing with your preferences")
    print("="*70)
    print(f"\nVideo: {Path(video_path).name}")
    print(f"\nPreferences:")
    print(f"  Visual Style: {preferences.visual_style}")
    print(f"  Pacing: {preferences.pace}")
    print(f"  Time of Day: {preferences.time_of_day}")
    print(f"  Content Filter: {preferences.content_filter}")
    print("\n" + "="*70 + "\n")

    # Initialize processor
    config = load_config()
    processor = BRollProcessor(config)

    # Process video
    print("Starting processing... (this will take 10-15 minutes)\n")
    await processor.process_video(video_path, preferences)

    print("\n" + "="*70)
    print("✓ Processing complete!")
    print(f"Check output folder: {config['local_output_folder']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
