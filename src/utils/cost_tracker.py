"""Cost tracking for API usage across all services."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APICost:
    """Represents a single API cost entry."""

    service: str  # "gemini", "openai", "youtube"
    operation: str  # "transcribe", "plan", "evaluate", "extract", etc.
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CostReport:
    """Aggregated cost report for a video processing job."""

    video_path: str
    total_cost_usd: float = 0.0
    api_calls: list[APICost] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_path": self.video_path,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "api_calls": [asdict(call) for call in self.api_calls],
            "generated_at": datetime.now().isoformat(),
        }


class CostTracker:
    """Tracks API costs across all services."""

    # Pricing constants (as of January 2026)
    # Gemini Flash pricing per million tokens
    GEMINI_FLASH_INPUT_PER_M = 0.075  # $0.075 per 1M input tokens
    GEMINI_FLASH_OUTPUT_PER_M = 0.30  # $0.30 per 1M output tokens

    # OpenAI Whisper pricing per minute
    WHISPER_PER_MINUTE = 0.006  # $0.006 per minute

    # Note: YouTube search is free via yt-dlp

    def __init__(self, budget_limit_usd: float | None = None):
        """Initialize cost tracker.

        Args:
            budget_limit_usd: Optional budget limit in USD. Warning will be logged if exceeded.
        """
        self.budget_limit = budget_limit_usd
        self.api_calls: list[APICost] = []
        self.total_cost = 0.0
        self.breakdown: dict[str, float] = {}

    def track_gemini_call(
        self,
        operation: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Track a Gemini API call.

        Args:
            operation: Operation name (e.g., "plan_broll", "evaluate_videos")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD for this call
        """
        cost = (
            (input_tokens / 1_000_000) * self.GEMINI_FLASH_INPUT_PER_M
            + (output_tokens / 1_000_000) * self.GEMINI_FLASH_OUTPUT_PER_M
        )

        api_cost = APICost(
            service="gemini",
            operation=operation,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cost_usd=cost,
        )

        self.api_calls.append(api_cost)
        self.total_cost += cost
        self.breakdown["gemini"] = self.breakdown.get("gemini", 0.0) + cost

        self._check_budget()
        return cost

    def track_whisper_call(
        self,
        duration_minutes: float,
    ) -> float:
        """Track an OpenAI Whisper transcription call.

        Args:
            duration_minutes: Audio duration in minutes

        Returns:
            Cost in USD for this call
        """
        cost = duration_minutes * self.WHISPER_PER_MINUTE

        api_cost = APICost(
            service="openai",
            operation="transcribe",
            tokens_input=0,
            tokens_output=0,
            cost_usd=cost,
        )

        self.api_calls.append(api_cost)
        self.total_cost += cost
        self.breakdown["openai"] = self.breakdown.get("openai", 0.0) + cost

        self._check_budget()
        return cost

    def get_report(self, video_path: str) -> CostReport:
        """Generate a cost report for the current session.

        Args:
            video_path: Path to the processed video

        Returns:
            CostReport object with all cost information
        """
        return CostReport(
            video_path=video_path,
            total_cost_usd=self.total_cost,
            api_calls=self.api_calls.copy(),
            breakdown=self.breakdown.copy(),
        )

    def save_report(self, output_dir: Path, video_path: str) -> Path:
        """Save cost report to JSON file.

        Args:
            output_dir: Directory to save the report
            video_path: Path to the processed video

        Returns:
            Path to the saved report file
        """
        report = self.get_report(video_path)

        # Create cost_report.json in the output directory
        report_path = output_dir / "cost_report.json"

        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(
            f"Cost report saved: {report_path} "
            f"(Total: ${self.total_cost:.4f}, "
            f"{len(self.api_calls)} API calls)"
        )

        return report_path

    def _check_budget(self) -> None:
        """Check if budget limit has been exceeded and log warning."""
        if self.budget_limit and self.total_cost > self.budget_limit:
            logger.warning(
                f"⚠️  BUDGET EXCEEDED: ${self.total_cost:.4f} > ${self.budget_limit:.2f} limit"
            )
