"""Interactive terminal UI for stockpile using Rich library."""

import logging
from typing import List, Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from models.user_preferences import UserPreferences, GeneratedQuestion
from models.broll_need import TranscriptResult

logger = logging.getLogger(__name__)


class InteractiveUI:
    """Rich-based interactive terminal interface for stockpile."""

    def __init__(self):
        self.console = Console()
        self.preferences = UserPreferences()

    def display_welcome(self) -> None:
        """Display welcome banner."""
        banner = """
 _____ _             _           _ _
|   __| |_ ___ ___ _| |___ _|

 |___ | '_| . |  _| '_| . | | |  _|
 |_____|_| |___|___|_,_|  _|_|_|___|
                      |_|

        AI-Powered B-Roll Processing Studio
        """
        self.console.print(
            Panel(
                banner,
                style="bold blue",
                border_style="blue",
            )
        )
        self.console.print()

    def display_video_info(self, file_path: str, transcript: TranscriptResult) -> None:
        """Show video metadata and transcript summary.

        Args:
            file_path: Path to the video file
            transcript: Transcription result with metadata
        """
        duration_minutes = transcript.duration / 60
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("File", Path(file_path).name)
        info_table.add_row("Duration", f"{duration_minutes:.1f} minutes")
        info_table.add_row("File Size", f"{file_size_mb:.1f} MB")
        info_table.add_row("Language", transcript.language if hasattr(transcript, 'language') else "Detected automatically")

        self.console.print(
            Panel(
                info_table,
                title="[bold]Video Information[/bold]",
                border_style="green",
            )
        )
        self.console.print()

    def display_transcript_preview(self, transcript: TranscriptResult, max_lines: int = 8) -> None:
        """Show a preview of the transcript with timestamps.

        Args:
            transcript: Transcription result
            max_lines: Maximum number of segments to display
        """
        self.console.print("[bold cyan]Transcript Preview:[/bold cyan]\n")

        if hasattr(transcript, 'segments') and transcript.segments:
            # Show first few segments
            for i, segment in enumerate(transcript.segments[:max_lines]):
                timestamp = f"[dim][{int(segment.start // 60):02d}:{int(segment.start % 60):02d}][/dim]"
                text = segment.text.strip()
                self.console.print(f"  {timestamp} {text}")

            if len(transcript.segments) > max_lines:
                remaining = len(transcript.segments) - max_lines
                self.console.print(f"\n  [dim]... ({remaining} more segments)[/dim]")
        else:
            # Fallback: show first portion of full text
            preview_text = transcript.text[:500]
            if len(transcript.text) > 500:
                preview_text += "..."
            self.console.print(f"  {preview_text}")

        self.console.print()

    def ask_predefined_questions(self) -> UserPreferences:
        """Ask predefined common questions about B-roll preferences.

        Returns:
            UserPreferences object with answers to predefined questions.
        """
        preferences = UserPreferences()

        self.console.print(
            "[bold yellow]Let's customize your B-roll preferences[/bold yellow]\n"
        )

        # Question 1: Visual Style
        self.console.print("[bold]1. Visual Style[/bold]")
        self.console.print("[dim]What visual style matches your content?[/dim]\n")

        self.console.print("  [1] Documentary (factual, observational)")
        self.console.print("  [2] Cinematic (dramatic, high production)")
        self.console.print("  [3] Vintage (period-appropriate, archival)")
        self.console.print("  [4] Modern (contemporary, sleek)")
        self.console.print("  [c] Custom\n")

        style_choice = Prompt.ask(
            "Choose",
            choices=["1", "2", "3", "4", "c"],
            default="1",
            show_choices=False,
        )

        style_map = {
            "1": "documentary",
            "2": "cinematic",
            "3": "vintage",
            "4": "modern",
        }

        if style_choice == "c":
            preferences.visual_style = Prompt.ask("  Enter custom style")
        else:
            preferences.visual_style = style_map[style_choice]

        self.console.print(f"  [green]✓[/green] {preferences.visual_style}\n")

        # Question 2: Pacing
        self.console.print("[bold]2. Pacing[/bold]")
        self.console.print("[dim]What pacing would work best?[/dim]\n")

        self.console.print("  [1] Fast (energetic, quick cuts)")
        self.console.print("  [2] Medium (balanced, versatile)")
        self.console.print("  [3] Slow (contemplative, lingering shots)")
        self.console.print("  [c] Custom\n")

        pace_choice = Prompt.ask(
            "Choose",
            choices=["1", "2", "3", "c"],
            default="2",
            show_choices=False,
        )

        pace_map = {
            "1": "fast",
            "2": "medium",
            "3": "slow",
        }

        if pace_choice == "c":
            preferences.pace = Prompt.ask("  Enter custom pace")
        else:
            preferences.pace = pace_map[pace_choice]

        self.console.print(f"  [green]✓[/green] {preferences.pace}\n")

        # Question 3: Time of Day
        self.console.print("[bold]3. Time of Day[/bold]")
        self.console.print("[dim]Preferred lighting/time of day?[/dim]\n")

        self.console.print("  [1] Any (no preference)")
        self.console.print("  [2] Daytime (bright, clear)")
        self.console.print("  [3] Night (dark, moody)")
        self.console.print("  [4] Golden hour (warm, cinematic)")
        self.console.print("  [c] Custom\n")

        time_choice = Prompt.ask(
            "Choose",
            choices=["1", "2", "3", "4", "c"],
            default="1",
            show_choices=False,
        )

        time_map = {
            "1": "any",
            "2": "daytime",
            "3": "night",
            "4": "golden hour",
        }

        if time_choice == "c":
            preferences.time_of_day = Prompt.ask("  Enter custom time preference")
        else:
            preferences.time_of_day = time_map[time_choice]

        self.console.print(f"  [green]✓[/green] {preferences.time_of_day}\n")

        return preferences

    def ask_generated_questions(
        self,
        questions: List[GeneratedQuestion],
        preferences: UserPreferences
    ) -> UserPreferences:
        """Present AI-generated questions and collect answers.

        Args:
            questions: List of context-aware questions from AI
            preferences: Existing preferences to update

        Returns:
            Updated UserPreferences with answers to generated questions.
        """
        if not questions:
            return preferences

        self.console.print(
            "\n[bold yellow]Based on your video content, I have a few more questions:[/bold yellow]\n"
        )

        for i, question in enumerate(questions, 1):
            self.console.print(f"[bold]{i}. {question.question_text}[/bold]")

            if question.context_reason:
                self.console.print(f"[dim]({question.context_reason})[/dim]")

            self.console.print()

            if question.options:
                # Display options
                for j, option in enumerate(question.options, 1):
                    self.console.print(f"  [{j}] {option}")

                if question.allows_custom:
                    self.console.print("  [c] Custom answer")

                self.console.print()

                # Get choice
                valid_choices = [str(j) for j in range(1, len(question.options) + 1)]
                if question.allows_custom:
                    valid_choices.append("c")

                choice = Prompt.ask(
                    "Choose",
                    choices=valid_choices,
                    default="1",
                    show_choices=False,
                )

                if choice == "c":
                    answer = Prompt.ask("  Enter your answer")
                else:
                    answer = question.options[int(choice) - 1]
            else:
                # Free-form answer
                answer = Prompt.ask("Your answer")

            # Map answer to preference field
            setattr(preferences, question.preference_field, answer)
            self.console.print(f"  [green]✓[/green] {answer}\n")

        return preferences

    def confirm_preferences(self, preferences: UserPreferences) -> bool:
        """Show summary of collected preferences and ask for confirmation.

        Args:
            preferences: Collected user preferences

        Returns:
            True if user confirms, False to re-enter preferences.
        """
        self.console.print("\n[bold cyan]Your Preferences Summary[/bold cyan]\n")

        summary_table = Table(show_header=True, box=None, padding=(0, 2))
        summary_table.add_column("Preference", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="white")

        if preferences.visual_style:
            summary_table.add_row("Visual Style", preferences.visual_style)
        if preferences.pace:
            summary_table.add_row("Pacing", preferences.pace)
        if preferences.time_of_day:
            summary_table.add_row("Time of Day", preferences.time_of_day)
        if preferences.era_period:
            summary_table.add_row("Era/Period", preferences.era_period)
        if preferences.location_type:
            summary_table.add_row("Location Type", preferences.location_type)
        if preferences.color_mood:
            summary_table.add_row("Color/Mood", preferences.color_mood)
        if preferences.content_focus:
            summary_table.add_row("Content Focus", preferences.content_focus)
        if preferences.content_filter:
            summary_table.add_row("Content Filter", preferences.content_filter)
        if preferences.custom_notes:
            summary_table.add_row("Custom Notes", "; ".join(preferences.custom_notes))

        self.console.print(
            Panel(
                summary_table,
                border_style="green",
            )
        )
        self.console.print()

        return Confirm.ask("[bold]Proceed with these preferences?[/bold]", default=True)

    def display_processing_status(self, message: str, style: str = "yellow") -> None:
        """Show processing status update.

        Args:
            message: Status message to display
            style: Rich style for the message
        """
        self.console.print(f"[{style}]⠿ {message}[/{style}]")

    def display_error(self, message: str) -> None:
        """Display error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"\n[bold red]Error:[/bold red] {message}\n")

    def display_success(self, message: str) -> None:
        """Display success message.

        Args:
            message: Success message to display
        """
        self.console.print(f"\n[bold green]✓[/bold green] {message}\n")
