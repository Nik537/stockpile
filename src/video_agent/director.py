"""AI Director Agent for draft video review.

Reviews rendered draft videos against the original script and timeline,
producing structured DraftReview feedback with fix requests for the
iterative refinement loop.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from google.genai import Client, types

from services.prompts import strip_markdown_code_blocks
from utils.retry import APIRateLimitError, NetworkError, retry_api_call
from video_agent.models import (
    DraftReview,
    FixRequest,
    Script,
    Timeline,
)

logger = logging.getLogger(__name__)

# Approval threshold: drafts scoring at or above this with no high-priority
# fixes are automatically approved.
DEFAULT_APPROVAL_THRESHOLD = 7

DIRECTOR_REVIEW_PROMPT = """You are an expert video director reviewing a draft video.
Compare the draft against the original script and timeline, then produce a structured review.

## Original Script

**Title:** {title}

### Hook
- Voiceover: "{hook_voiceover}"
- Visual description: "{hook_visual_description}"
- Keywords: {hook_keywords}

### Scenes
{scenes_text}

## Timeline Info
- Total duration: {total_duration:.1f}s
- Number of scenes: {num_scenes}
- Color grade: {color_grade}
- Has music: {has_music}
- Has subtitles: {has_subtitles}

## Review Instructions

Watch the draft video carefully and evaluate:

1. **Visual-Script Alignment**: Does each scene's visual match the script's visual_keywords and visual_style?
2. **Pacing**: Are scene durations appropriate? Do transitions feel natural?
3. **Flow**: Does the video flow smoothly from hook through all scenes?
4. **Audio Sync**: Does the voiceover timing match the visuals?
5. **Overall Quality**: Professional polish, color consistency, subtitle readability.

## Scoring Guide
- 9-10: Broadcast ready, no changes needed
- 7-8: Good quality, minor polish only
- 5-6: Acceptable but needs specific fixes
- 3-4: Significant issues, multiple scenes need rework
- 1-2: Major problems, near-complete rework needed

## Output Format

Return a JSON object with this exact structure:
{{
  "overall_score": <int 1-10>,
  "approved": <bool>,
  "notes": "<general reviewer notes>",
  "fixes": [
    {{
      "scene_id": <int>,
      "issue_type": "<visual_mismatch|pacing_issue|transition_jarring|audio_sync|content_gap>",
      "description": "<what is wrong>",
      "suggested_fix": "<how to fix it>",
      "suggested_keywords": ["<alt search terms if visual_mismatch>"],
      "priority": "<low|medium|high>"
    }}
  ]
}}

Rules:
- Set approved=true ONLY if overall_score >= {approval_threshold} AND there are no "high" priority fixes.
- Be specific in descriptions. Reference scene IDs.
- For visual_mismatch issues, always provide suggested_keywords with better search terms.
- Return an empty fixes list if the draft is perfect.
- Return ONLY the JSON object, no markdown fences or extra text.
"""


class DirectorAgent:
    """AI director that reviews draft videos against their scripts.

    Uses Gemini's multimodal capabilities to analyze draft video files
    and compare them against the original script and timeline, producing
    structured DraftReview feedback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-pro-preview",
        approval_threshold: int = DEFAULT_APPROVAL_THRESHOLD,
    ):
        """Initialize the DirectorAgent.

        Args:
            api_key: Google GenAI API key. Falls back to GEMINI_API_KEY env var.
            model: Gemini model to use. Defaults to Flash for speed.
            approval_threshold: Minimum score for auto-approval (default 7).
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            logger.warning(
                "DirectorAgent initialized without API key. "
                "review_draft() will return pass-through reviews."
            )
        self.client = Client(api_key=resolved_key) if resolved_key else None
        self.model = model
        self.approval_threshold = approval_threshold

        logger.info(
            f"DirectorAgent initialized: model={model}, "
            f"approval_threshold={approval_threshold}"
        )

    @retry_api_call(max_retries=3, base_delay=2.0)
    def review_draft(
        self,
        draft_video_path: str,
        script: Script,
        timeline: Timeline,
        iteration: int = 0,
    ) -> DraftReview:
        """Review a rendered draft video against the script and timeline.

        Uploads the draft video to Gemini and sends a detailed review prompt
        comparing it against the original script. Returns a structured
        DraftReview with scoring and fix requests.

        Args:
            draft_video_path: Path to the draft video file (480p recommended).
            script: The Script the video was built from.
            timeline: The Timeline used for composition.
            iteration: Current review iteration number.

        Returns:
            DraftReview with score, approval status, and fix requests.
        """
        video_path = Path(draft_video_path)

        if not video_path.exists():
            logger.error(f"Draft video not found: {draft_video_path}")
            return self._passthrough_review(
                iteration, notes=f"Draft file not found: {draft_video_path}"
            )

        if self.client is None:
            logger.warning("No Gemini API key configured, returning pass-through review")
            return self._passthrough_review(
                iteration, notes="No API key configured, auto-approved"
            )

        # Build the review prompt
        prompt = self._build_review_prompt(script, timeline)

        logger.info(
            f"Reviewing draft (iteration {iteration}): "
            f"{video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )

        try:
            # Upload video file to Gemini
            uploaded_file = self.client.files.upload(file=str(video_path))

            # Wait for file to be processed
            self._wait_for_file_processing(uploaded_file)

            # Send review request with video + prompt
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )

            if not response.text:
                logger.error("Empty response from Gemini for draft review")
                return self._passthrough_review(
                    iteration, notes="Empty AI response, auto-approved"
                )

            # Parse the structured review
            review = self._parse_review_response(response.text, iteration)

            logger.info(
                f"Draft review complete: score={review.overall_score}/10, "
                f"approved={review.approved}, fixes={len(review.fix_requests)}"
            )

            return review

        except (APIRateLimitError, NetworkError):
            raise  # Let retry decorator handle these
        except Exception as e:
            logger.error(f"Draft review failed: {e}")

            # Graceful degradation: don't block production pipeline
            return self._passthrough_review(
                iteration, notes=f"Review failed ({type(e).__name__}), auto-approved"
            )

    def _build_review_prompt(self, script: Script, timeline: Timeline) -> str:
        """Build the review prompt from script and timeline data.

        Args:
            script: The original Script.
            timeline: The composition Timeline.

        Returns:
            Formatted prompt string.
        """
        # Format scenes text
        scenes_lines = []
        for scene in script.scenes:
            scenes_lines.append(
                f"- Scene {scene.id}: voiceover=\"{scene.voiceover[:80]}...\", "
                f"keywords={scene.visual_keywords}, "
                f"style={scene.visual_style}, "
                f"transition={scene.transition_in}, "
                f"music_mood={scene.music_mood}, "
                f"duration_est={scene.duration_est}s"
            )
        scenes_text = "\n".join(scenes_lines)

        return DIRECTOR_REVIEW_PROMPT.format(
            title=script.title,
            hook_voiceover=script.hook.voiceover[:120],
            hook_visual_description=script.hook.visual_description[:120],
            hook_keywords=script.hook.visual_keywords,
            scenes_text=scenes_text,
            total_duration=timeline.total_duration,
            num_scenes=len(timeline.scenes),
            color_grade=timeline.color_grade,
            has_music=timeline.music_path is not None,
            has_subtitles=timeline.subtitle_path is not None,
            approval_threshold=self.approval_threshold,
        )

    def _wait_for_file_processing(self, uploaded_file, timeout: float = 120.0) -> None:
        """Wait for an uploaded file to finish server-side processing.

        Args:
            uploaded_file: The file object returned by client.files.upload().
            timeout: Maximum wait time in seconds.

        Raises:
            TimeoutError: If processing exceeds timeout.
            RuntimeError: If processing fails.
        """
        start = time.time()
        while True:
            file_info = self.client.files.get(name=uploaded_file.name)
            state = getattr(file_info, "state", None)

            if state is None or str(state) == "ACTIVE":
                return  # Ready

            if str(state) == "FAILED":
                raise RuntimeError(
                    f"Gemini file processing failed for {uploaded_file.name}"
                )

            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"File processing timed out after {timeout}s for {uploaded_file.name}"
                )

            logger.debug(
                f"Waiting for file processing... state={state}, elapsed={elapsed:.0f}s"
            )
            time.sleep(2.0)

    def _parse_review_response(self, response_text: str, iteration: int) -> DraftReview:
        """Parse Gemini's JSON response into a DraftReview.

        Args:
            response_text: Raw text from Gemini response.
            iteration: Current review iteration number.

        Returns:
            Populated DraftReview object.
        """
        cleaned = strip_markdown_code_blocks(response_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse review JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return self._passthrough_review(
                iteration, notes="Failed to parse AI review response, auto-approved"
            )

        # Parse overall score
        overall_score = int(data.get("overall_score", 5))
        overall_score = max(1, min(10, overall_score))

        # Parse notes
        notes = str(data.get("notes", "")).strip()

        # Parse fix requests
        fix_requests = []
        has_high_priority = False
        for fix_data in data.get("fixes", []):
            if not isinstance(fix_data, dict):
                continue

            priority = str(fix_data.get("priority", "medium")).strip().lower()
            if priority not in ("low", "medium", "high"):
                priority = "medium"
            if priority == "high":
                has_high_priority = True

            suggested_keywords = fix_data.get("suggested_keywords", [])
            if isinstance(suggested_keywords, list):
                suggested_keywords = [str(k).strip() for k in suggested_keywords if k]
            else:
                suggested_keywords = []

            fix = FixRequest(
                scene_id=int(fix_data.get("scene_id", 0)),
                issue_type=str(fix_data.get("issue_type", "content_gap")).strip(),
                description=str(fix_data.get("description", "")).strip(),
                suggested_keywords=suggested_keywords,
                suggested_fix=str(fix_data.get("suggested_fix", "")).strip(),
                priority=priority,
            )
            fix_requests.append(fix)

        # Determine approval: score >= threshold AND no high-priority fixes
        approved = data.get("approved", False)
        # Override AI's decision with our threshold logic for consistency
        if overall_score >= self.approval_threshold and not has_high_priority:
            approved = True
        elif overall_score < self.approval_threshold or has_high_priority:
            approved = False

        return DraftReview(
            overall_score=overall_score,
            fix_requests=fix_requests,
            approved=approved,
            iteration=iteration,
            notes=notes,
        )

    def _passthrough_review(self, iteration: int, notes: str = "") -> DraftReview:
        """Create a pass-through review that auto-approves the draft.

        Used as graceful degradation when the API is unavailable or
        encounters errors - the production pipeline should not be blocked
        by a failed review.

        Args:
            iteration: Current review iteration number.
            notes: Explanation of why pass-through was used.

        Returns:
            An approved DraftReview with score 7 and no fix requests.
        """
        logger.info(f"Pass-through review (iteration {iteration}): {notes}")
        return DraftReview(
            overall_score=7,
            fix_requests=[],
            approved=True,
            iteration=iteration,
            notes=notes or "Pass-through review (API unavailable)",
        )

    async def close(self) -> None:
        """Clean up resources.

        Currently a no-op since google.genai Client does not require
        explicit cleanup, but included for interface consistency with
        other agents.
        """
        logger.debug("DirectorAgent closed")
