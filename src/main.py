"""Main application entry point for stockpile."""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Dict

from tqdm import tqdm

from broll_processor import BRollProcessor
from services.interactive_ui import InteractiveUI
from utils.config import setup_logging, load_config
from utils.progress import ProcessingStatus, format_eta

logger = logging.getLogger(__name__)


class ProgressBarCallback:
    """Progress bar callback for displaying real-time progress."""

    def __init__(self):
        """Initialize progress bars."""
        self.progress_bars: Dict[str, tqdm] = {}
        self.overall_bar: Optional[tqdm] = None

    def __call__(self, status: ProcessingStatus):
        """Update progress bars based on status.

        Args:
            status: ProcessingStatus object with current progress
        """
        # Create or update overall progress bar
        if self.overall_bar is None:
            self.overall_bar = tqdm(
                total=100,
                desc="Overall Progress",
                unit="%",
                position=0,
                leave=True,
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}% [{elapsed}<{remaining}, {rate_fmt}]",
            )

        # Update overall progress
        self.overall_bar.n = status.overall_progress
        self.overall_bar.refresh()

        # Create or update stage-specific progress bars
        for stage_name, stage in status.stages.items():
            if stage_name not in self.progress_bars:
                # Create new progress bar for this stage
                self.progress_bars[stage_name] = tqdm(
                    total=stage.total_items,
                    desc=f"  {stage_name.replace('_', ' ').title()}",
                    unit="items",
                    position=len(self.progress_bars) + 1,
                    leave=False,
                )

            # Update stage progress
            bar = self.progress_bars[stage_name]
            bar.n = stage.completed_items
            bar.total = stage.total_items

            # Update description with ETA
            if stage.status == "in_progress" and stage.eta_seconds:
                eta_str = format_eta(stage.eta_seconds)
                bar.set_description(f"  {stage_name.replace('_', ' ').title()} (ETA: {eta_str})")
            elif stage.status == "completed":
                bar.set_description(f"  {stage_name.replace('_', ' ').title()} ✓")
            elif stage.status == "failed":
                bar.set_description(f"  {stage_name.replace('_', ' ').title()} ✗")

            bar.refresh()

    def close(self):
        """Close all progress bars."""
        if self.overall_bar:
            self.overall_bar.close()
        for bar in self.progress_bars.values():
            bar.close()


class StockpileApp:
    """Main application class for stockpile."""

    def __init__(self, interactive: bool = False, video_file: Optional[str] = None):
        self.processor: Optional[BRollProcessor] = None
        self.running = False
        self.interactive = interactive
        self.video_file = video_file
        self.ui = InteractiveUI() if interactive else None

    async def start(self) -> None:
        """Start the stockpile application."""
        # Setup logging
        setup_logging()

        if self.interactive:
            await self._start_interactive()
        else:
            await self._start_daemon()

    async def _start_daemon(self) -> None:
        """Start the daemon mode (file watching)."""
        try:
            logger.info("Starting stockpile in daemon mode...")

            # Initialize processor
            config = load_config()
            self.processor = BRollProcessor(config)

            # Start processor
            self.processor.start()
            self.running = True

            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("stockpile started successfully (daemon mode)")

            # Keep the application running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            sys.exit(1)

    async def _start_interactive(self) -> None:
        """Start the interactive mode (single video with Q&A)."""
        try:
            # Validate video file
            if not self.video_file:
                self.ui.display_error("No video file specified. Usage: python main.py -i <video_file>")
                sys.exit(1)

            video_path = Path(self.video_file)
            if not video_path.exists():
                self.ui.display_error(f"Video file not found: {self.video_file}")
                sys.exit(1)

            # Display welcome
            self.ui.display_welcome()

            # Initialize processor
            config = load_config()
            self.processor = BRollProcessor(config)

            # Step 1: Transcribe video
            self.ui.display_processing_status("Transcribing audio... (this may take a moment)")
            transcript = await self.processor.transcribe_and_prompt(str(video_path))

            # Step 2: Display video info and transcript preview
            self.ui.display_video_info(str(video_path), transcript)
            self.ui.display_transcript_preview(transcript)

            # Step 3: Ask predefined questions
            preferences = self.ui.ask_predefined_questions()

            # Step 4: Generate and ask context-specific questions
            max_questions = config.get("interactive_max_questions", 3)
            self.ui.display_processing_status("Generating context-specific questions based on your video...")

            questions = self.processor.ai_service.generate_context_questions(
                transcript, max_questions
            )

            if questions:
                preferences = self.ui.ask_generated_questions(questions, preferences)

            # Step 5: Confirm preferences
            confirmed = self.ui.confirm_preferences(preferences)

            if not confirmed:
                self.ui.display_error("Preferences not confirmed. Exiting.")
                sys.exit(0)

            # Step 6: Process video with preferences
            self.ui.display_processing_status("Processing video with your preferences...")
            self.ui.display_processing_status("This may take several minutes...")

            # Create progress callback
            progress_callback = ProgressBarCallback()

            try:
                await self.processor.process_video(
                    str(video_path),
                    preferences,
                    status_callback=progress_callback
                )
            finally:
                # Close progress bars
                progress_callback.close()

            self.ui.display_success("\nProcessing complete! Check output folder for organized B-roll clips.")

        except KeyboardInterrupt:
            self.ui.display_error("Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Interactive mode failed: {e}", exc_info=True)
            self.ui.display_error(f"Processing failed: {e}")
            sys.exit(1)

    def _signal_handler(self, signum, _):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Stockpile B-roll processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  stockpile                     # Run in daemon mode (watches for videos)
  stockpile -i video.mp4        # Interactive mode for single video
        """
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode with user prompts"
    )
    parser.add_argument(
        "video_file",
        nargs="?",
        help="Video file to process (required for interactive mode)"
    )

    args = parser.parse_args()

    # Create app with parsed arguments
    app = StockpileApp(interactive=args.interactive, video_file=args.video_file)

    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
