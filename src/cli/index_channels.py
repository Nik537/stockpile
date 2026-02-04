#!/usr/bin/env python3
"""CLI for indexing YouTube channels by niche.

Usage:
    # Index a single niche
    python -m cli.index_channels --niche "tech reviews" --max-channels 200

    # Index from predefined popular niches
    python -m cli.index_channels --preset popular

    # Update stale channels
    python -m cli.index_channels --update-stale --days 7

    # Show index statistics
    python -m cli.index_channels --stats
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from services.channel_indexer import ChannelIndexer, get_channel_indexer, POPULAR_NICHES
from utils.config import load_config, setup_logging


console = Console()


def index_single_niche(indexer: ChannelIndexer, niche: str, max_channels: int) -> None:
    """Index a single niche with progress display."""
    console.print(f"\n[bold blue]Indexing niche: {niche}[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Indexing {niche}...", total=max_channels)

        def on_progress(indexed: int, total: int) -> None:
            progress.update(task, completed=indexed, total=total)

        result = indexer.index_niche(niche, max_channels=max_channels, on_progress=on_progress)

    # Display results
    if result.channels_indexed > 0:
        console.print(f"[green]✓ Indexed {result.channels_indexed} channels[/green]")
    if result.channels_skipped > 0:
        console.print(f"[yellow]⚠ Skipped {result.channels_skipped} channels[/yellow]")
    if result.errors:
        console.print(f"[red]✗ {len(result.errors)} errors occurred[/red]")
        for error in result.errors[:5]:  # Show first 5 errors
            console.print(f"  [dim]{error}[/dim]")

    console.print(f"[dim]Duration: {result.duration_seconds:.1f}s[/dim]")


def index_preset(indexer: ChannelIndexer, preset: str, max_per_niche: int) -> None:
    """Index all niches in a preset."""
    console.print(f"\n[bold blue]Indexing preset: {preset}[/bold blue]")
    console.print(f"[dim]This will index {len(POPULAR_NICHES)} niches with up to {max_per_niche} channels each[/dim]\n")

    total_indexed = 0
    total_skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing niches...", total=len(POPULAR_NICHES))

        def on_niche_complete(niche: str, result) -> None:
            nonlocal total_indexed, total_skipped
            total_indexed += result.channels_indexed
            total_skipped += result.channels_skipped
            progress.advance(task)

        results = indexer.index_preset(
            preset=preset,
            max_channels_per_niche=max_per_niche,
            on_niche_complete=on_niche_complete,
        )

    # Summary table
    table = Table(title="Indexing Results")
    table.add_column("Niche", style="cyan")
    table.add_column("Indexed", justify="right", style="green")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Duration", justify="right")

    for niche, result in results.items():
        table.add_row(
            niche,
            str(result.channels_indexed),
            str(result.channels_skipped),
            f"{result.duration_seconds:.1f}s",
        )

    console.print(table)
    console.print(f"\n[bold green]Total: {total_indexed} channels indexed, {total_skipped} skipped[/bold green]")


def update_stale(indexer: ChannelIndexer, days: int, max_channels: int) -> None:
    """Update stale channels."""
    console.print(f"\n[bold blue]Updating channels older than {days} days[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Updating stale channels...", total=max_channels)

        def on_progress(updated: int, total: int) -> None:
            progress.update(task, completed=updated, total=total)

        updated = indexer.update_stale_channels(
            days_old=days,
            max_channels=max_channels,
            on_progress=on_progress,
        )

    console.print(f"[green]✓ Updated {updated} channels[/green]")


def show_stats(indexer: ChannelIndexer) -> None:
    """Display index statistics."""
    stats = indexer.get_index_stats()

    table = Table(title="Channel Index Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Indexed Channels", str(stats["indexed_channels"]))
    table.add_row("Unique Niches", str(stats["unique_niches"]))
    table.add_row("Cloud Sync", "✓ Enabled" if stats["cloud_sync_enabled"] else "✗ Disabled")
    table.add_row("YouTube API", "✓ Available" if stats["api_available"] else "✗ Unavailable")
    table.add_row("API Quota Used", str(stats["api_quota_used"]))

    console.print(table)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index YouTube channels by niche for fast outlier searching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index a single niche
    python -m cli.index_channels --niche "tech reviews" --max-channels 200

    # Index popular niches preset
    python -m cli.index_channels --preset popular --max-per-niche 50

    # Update stale channels
    python -m cli.index_channels --update-stale --days 7

    # Show statistics
    python -m cli.index_channels --stats
        """,
    )

    parser.add_argument(
        "--niche",
        type=str,
        help="Single niche to index (e.g., 'tech reviews')",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["popular"],
        help="Index all niches from a preset",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=100,
        help="Maximum channels to index per niche (default: 100)",
    )
    parser.add_argument(
        "--max-per-niche",
        type=int,
        default=50,
        help="Maximum channels per niche when using preset (default: 50)",
    )
    parser.add_argument(
        "--update-stale",
        action="store_true",
        help="Update channels that haven't been refreshed recently",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Consider channels stale after this many days (default: 7)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    # Load config
    config = load_config()

    # Check for YouTube API key
    if not config.get("youtube_api_key"):
        console.print("[red]Error: YOUTUBE_API_KEY not configured[/red]")
        console.print("[dim]Set YOUTUBE_API_KEY in your .env file[/dim]")
        sys.exit(1)

    # Create indexer
    indexer = get_channel_indexer(youtube_api_key=config.get("youtube_api_key"))

    # Execute requested operation
    if args.stats:
        show_stats(indexer)
    elif args.update_stale:
        update_stale(indexer, args.days, args.max_channels)
    elif args.preset:
        index_preset(indexer, args.preset, args.max_per_niche)
    elif args.niche:
        index_single_niche(indexer, args.niche, args.max_channels)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
