#!/usr/bin/env python3
"""YouTube Outlier Finder CLI.

Find viral/outperforming YouTube videos by topic for title and thumbnail inspiration.
Uses yt-dlp for free YouTube metadata extraction.

Usage:
    python src/outlier_finder.py -t "tech reviews"
    python src/outlier_finder.py -t "cooking tutorials" --days 90 --min-score 5
    python src/outlier_finder.py -t "productivity" -o outliers.json --max-channels 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from services.outlier_finder_service import OutlierFinderService
from models.outlier import OutlierSearchResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_views(count: int) -> str:
    """Format view count for display.

    Args:
        count: View count

    Returns:
        Formatted string (e.g., "1.2M", "500K", "10K")
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}K"
    else:
        return str(count)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text with ellipsis if needed.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def get_tier_style(tier: str) -> str:
    """Get rich style for tier display.

    Args:
        tier: Outlier tier

    Returns:
        Rich style string
    """
    styles = {
        "exceptional": "bold red",
        "strong": "bold yellow",
        "solid": "green",
    }
    return styles.get(tier, "white")


def display_results(result: OutlierSearchResult, console: Console) -> None:
    """Display outlier results in a rich table.

    Args:
        result: OutlierSearchResult to display
        console: Rich console instance
    """
    if not result.outliers:
        console.print("\n[yellow]No outliers found matching your criteria.[/yellow]")
        console.print(
            f"Scanned {result.total_videos_scanned} videos from "
            f"{result.channels_analyzed} channels."
        )
        return

    # Create summary panel
    summary = Text()
    summary.append(f"Topic: ", style="bold")
    summary.append(f"{result.topic}\n")
    summary.append(f"Channels analyzed: ", style="bold")
    summary.append(f"{result.channels_analyzed}\n")
    summary.append(f"Videos scanned: ", style="bold")
    summary.append(f"{result.total_videos_scanned}\n")
    summary.append(f"Outliers found: ", style="bold")
    summary.append(f"{len(result.outliers)}", style="green bold")

    console.print(Panel(summary, title="YouTube Outlier Finder", border_style="blue"))

    # Create results table
    table = Table(
        title="Outlier Videos",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
        row_styles=["", "dim"],
    )

    table.add_column("Title", style="white", max_width=40)
    table.add_column("Channel", style="cyan", max_width=15)
    table.add_column("Views", justify="right", style="green")
    table.add_column("Avg", justify="right", style="yellow")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Tier", justify="center")
    table.add_column("Date", justify="center", style="dim")

    for outlier in result.outliers:
        tier_style = get_tier_style(outlier.outlier_tier)

        # Format date for display
        date_display = ""
        if outlier.upload_date and len(outlier.upload_date) == 8:
            date_display = f"{outlier.upload_date[4:6]}/{outlier.upload_date[6:8]}/{outlier.upload_date[2:4]}"

        table.add_row(
            truncate_text(outlier.title, 40),
            truncate_text(outlier.channel_name, 15),
            format_views(outlier.view_count),
            format_views(int(outlier.channel_average_views)),
            f"{outlier.outlier_score:.1f}x",
            Text(outlier.outlier_tier[:5], style=tier_style),
            date_display,
        )

    console.print(table)

    # Show tier legend
    legend = Text()
    legend.append("Tiers: ")
    legend.append("exceptional", style="bold red")
    legend.append(" (10x+) | ")
    legend.append("strong", style="bold yellow")
    legend.append(" (5-10x) | ")
    legend.append("solid", style="green")
    legend.append(" (3-5x)")

    console.print(legend)


def export_json(result: OutlierSearchResult, output_path: str) -> None:
    """Export results to JSON file.

    Args:
        result: OutlierSearchResult to export
        output_path: Path to output file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Results exported to: {output_path}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Find viral/outperforming YouTube videos by topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/outlier_finder.py -t "tech reviews"
  python src/outlier_finder.py -t "cooking" --days 90 --min-score 5
  python src/outlier_finder.py -t "productivity" -o outliers.json
  python src/outlier_finder.py -t "gaming" --max-channels 20 --include-shorts
        """,
    )

    parser.add_argument(
        "-t", "--topic",
        required=True,
        help="Topic to search for (e.g., 'tech reviews')",
    )

    parser.add_argument(
        "--max-channels",
        type=int,
        default=10,
        help="Maximum channels to analyze (default: 10)",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=50,
        help="Maximum videos per channel to analyze (default: 50)",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=3.0,
        help="Minimum outlier score to include (default: 3.0 = 3x average)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only include videos from last N days (default: all time)",
    )

    parser.add_argument(
        "--min-subs",
        type=int,
        default=None,
        help="Minimum channel subscriber count (default: no minimum)",
    )

    parser.add_argument(
        "--max-subs",
        type=int,
        default=None,
        help="Maximum channel subscriber count (default: no maximum)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Export results to JSON file",
    )

    parser.add_argument(
        "--include-shorts",
        action="store_true",
        help="Include YouTube Shorts (default: exclude)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("services.outlier_finder_service").setLevel(logging.INFO)

    console = Console()

    console.print(f"\n[bold blue]YouTube Outlier Finder[/bold blue]")
    console.print(f"Searching for outliers in topic: [cyan]{args.topic}[/cyan]")
    console.print(f"Analyzing up to {args.max_channels} channels...\n")

    # Initialize service
    service = OutlierFinderService(
        min_score=args.min_score,
        max_videos_per_channel=args.max_videos,
        date_days=args.days,
        min_subs=args.min_subs,
        max_subs=args.max_subs,
        exclude_shorts=not args.include_shorts,
    )

    try:
        # Run the search
        result = service.find_outliers_by_topic(args.topic, args.max_channels)

        # Display results
        display_results(result, console)

        # Export to JSON if requested
        if args.output:
            export_json(result, args.output)
            console.print(f"\n[green]Results saved to: {args.output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Search cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
