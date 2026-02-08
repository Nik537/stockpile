"""Transcription and B-roll planning service.

This module extracts transcription and B-roll planning functionality from
BRollProcessor into a dedicated service that handles:
- Audio transcription using Whisper
- Search phrase extraction from transcripts
- Timeline-aware B-roll need planning

This is part of the BRollProcessor decomposition into focused service classes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from models.broll_need import BRollNeed, BRollPlan, TranscriptResult
    from models.user_preferences import UserPreferences
    from services.ai_service import AIService
    from services.transcription import TranscriptionService

logger = logging.getLogger(__name__)


class TranscriptionPlanningService:
    """Service for transcription and B-roll planning.

    This service encapsulates transcription and planning logic that was previously
    embedded in BRollProcessor, providing a clean interface for:
    - Audio transcription with timestamps
    - Search phrase extraction from transcripts
    - Timeline-aware B-roll need planning
    """

    def __init__(
        self,
        transcription_service: "TranscriptionService",
        ai_service: "AIService",
        clips_per_minute: float = 2.0,
        content_filter: Optional[str] = None,
    ):
        """Initialize the transcription planning service.

        Args:
            transcription_service: TranscriptionService instance for Whisper transcription
            ai_service: AIService instance for AI-powered planning
            clips_per_minute: Target B-roll density (default: 2.0 clips per minute)
            content_filter: Optional content filter string (e.g., "men only, no women")
        """
        self.transcription_service = transcription_service
        self.ai_service = ai_service
        self.clips_per_minute = clips_per_minute
        self.content_filter = content_filter

        logger.info(
            f"[TranscriptionPlanningService] Initialized with "
            f"clips_per_minute={clips_per_minute}"
        )

    async def transcribe_audio(self, file_path: str) -> "TranscriptResult":
        """Transcribe audio content using Whisper with timestamps.

        Args:
            file_path: Path to the audio/video file to transcribe

        Returns:
            TranscriptResult with text, segments (with timestamps), and duration

        Raises:
            ValueError: If the file format is not supported
        """
        if not self.transcription_service.is_supported_file(file_path):
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")

        # Get full transcript with timestamps
        transcript_result = await self.transcription_service.transcribe_audio(
            file_path, with_timestamps=True
        )
        return transcript_result

    async def extract_search_phrases(self, transcript: str) -> list[str]:
        """Extract relevant search phrases using Gemini AI (legacy method).

        Args:
            transcript: Full transcript text to analyze

        Returns:
            List of search phrases for B-roll footage
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided for phrase extraction")
            return []

        loop = asyncio.get_event_loop()
        search_phrases = await loop.run_in_executor(
            None, self.ai_service.extract_search_phrases, transcript
        )
        return search_phrases

    async def plan_broll_needs(
        self,
        transcript_result: "TranscriptResult",
        source_file: str,
        user_preferences: Optional["UserPreferences"] = None,
    ) -> "BRollPlan":
        """Plan timeline-aware B-roll needs from transcript.

        Uses AI to identify specific moments in the source video that need
        B-roll footage, with target density of clips_per_minute.

        Args:
            transcript_result: TranscriptResult with segments and timestamps
            source_file: Path to source file for reference
            user_preferences: Optional user preferences for B-roll customization

        Returns:
            BRollPlan with list of BRollNeed objects
        """
        # Import here to avoid circular imports
        from models.broll_need import BRollPlan

        if not transcript_result.text or not transcript_result.text.strip():
            logger.warning("Empty transcript provided for B-roll planning")
            return BRollPlan(
                source_duration=transcript_result.duration,
                needs=[],
                clips_per_minute=self.clips_per_minute,
                source_file=source_file,
            )

        loop = asyncio.get_event_loop()
        broll_plan = await loop.run_in_executor(
            None,
            lambda: self.ai_service.plan_broll_needs(
                transcript_result,
                self.clips_per_minute,
                source_file,
                self.content_filter,
                user_preferences,
            ),
        )
        return broll_plan

    async def transcribe_and_prompt(self, file_path: str) -> "TranscriptResult":
        """Transcribe video and return result for interactive prompting.

        Used by interactive mode to get transcript before asking questions.
        This allows the UI to show transcript preview and generate context-aware
        questions before proceeding with B-roll planning.

        Args:
            file_path: Path to video file to transcribe

        Returns:
            TranscriptResult with text, segments, and duration
        """
        return await self.transcribe_audio(file_path)
