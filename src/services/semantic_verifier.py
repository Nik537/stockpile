"""Semantic verification service for verifying clips match original transcript context.

This service uses Gemini AI to analyze extracted clips and verify they semantically
match the original transcript context and contain the required visual elements.

Configuration (from src/utils/config.py):
- semantic_match_threshold: Minimum similarity score to pass (0.0-1.0)
- semantic_verification_enabled: Whether verification is active
- reject_below_threshold: Whether to reject clips below threshold
- min_required_elements_match: Minimum ratio of required elements that must be present
"""

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from google.genai import Client, types
from models.broll_need import BRollNeed
from models.clip import ClipSegment
from utils.config import load_config
from utils.retry import APIRateLimitError, NetworkError, retry_api_call

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Prompt version for cache invalidation
SEMANTIC_VERIFICATION_PROMPT_VERSION = "v1"


@dataclass
class VerificationResult:
    """Result of semantic verification for a clip.

    Attributes:
        passed: Whether the clip passes the semantic verification threshold
        similarity_score: How well the clip matches the original context (0.0 to 1.0)
        matched_elements: List of required elements that ARE visible in the clip
        missing_elements: List of required elements that are NOT visible in the clip
        rationale: AI-generated explanation of the verification result
    """

    passed: bool
    similarity_score: float  # 0.0 to 1.0
    matched_elements: list[str] = field(default_factory=list)
    missing_elements: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def elements_match_ratio(self) -> float:
        """Calculate the ratio of matched elements to total required elements."""
        total = len(self.matched_elements) + len(self.missing_elements)
        if total == 0:
            return 1.0  # No required elements means 100% match
        return len(self.matched_elements) / total

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "similarity_score": self.similarity_score,
            "elements_match_ratio": self.elements_match_ratio,
            "matched_elements": self.matched_elements,
            "missing_elements": self.missing_elements,
            "rationale": self.rationale,
        }


class SemanticVerifier:
    """Service for verifying clips semantically match transcript context.

    Uses Gemini's multimodal capabilities to analyze video clips and verify
    they match the original transcript context and contain required visual elements.

    Example usage:
        verifier = SemanticVerifier()
        result = await verifier.verify_clip(clip_path, broll_need)
        if result.passed:
            print(f"Clip verified with score: {result.similarity_score:.2f}")
        else:
            print(f"Clip rejected: {result.rationale}")
    """

    # Configuration constants
    FILE_POLL_INTERVAL = 5.0  # Seconds between file status checks
    FILE_POLL_TIMEOUT = 300.0  # Max seconds to wait for file processing

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ):
        """Initialize the semantic verifier.

        Args:
            model_name: Gemini model to use for verification
            api_key: Optional API key. If not provided, loads from config.
        """
        config = load_config()

        # Load API key from config if not provided
        if api_key is None:
            api_key = config.get("gemini_api_key")

        if not api_key:
            msg = "GEMINI_API_KEY is required for SemanticVerifier"
            raise ValueError(msg)

        self.model_name = model_name
        self.client = Client(api_key=api_key)

        # Load verification thresholds from config
        self.threshold = float(config.get("semantic_match_threshold", 0.9))
        self.enabled = bool(config.get("semantic_verification_enabled", True))
        self.reject_below_threshold = bool(config.get("reject_below_threshold", True))
        self.min_elements_match = float(config.get("min_required_elements_match", 0.8))

        logger.info(
            f"Initialized SemanticVerifier with model: {model_name} "
            f"(threshold: {self.threshold}, enabled: {self.enabled})"
        )

    async def verify_clip(
        self,
        clip_path: Path,
        broll_need: BRollNeed,
    ) -> VerificationResult:
        """Verify a clip matches the original transcript context.

        Uploads the clip to Gemini and prompts the AI to analyze whether
        the visual content matches the original transcript context and
        contains the required visual elements.

        Args:
            clip_path: Path to the extracted clip file
            broll_need: BRollNeed with original_context and required_elements

        Returns:
            VerificationResult with verification details
        """
        if not self.enabled:
            logger.debug("Semantic verification disabled, auto-passing clip")
            return VerificationResult(
                passed=True,
                similarity_score=1.0,
                matched_elements=list(broll_need.required_elements),
                missing_elements=[],
                rationale="Verification disabled - auto-passed",
            )

        clip_file = Path(clip_path)
        if not clip_file.exists():
            logger.error(f"Clip file not found: {clip_path}")
            return VerificationResult(
                passed=False,
                similarity_score=0.0,
                matched_elements=[],
                missing_elements=list(broll_need.required_elements),
                rationale=f"Clip file not found: {clip_path}",
            )

        # If no original_context, we can't verify semantically
        if not broll_need.original_context:
            logger.warning(
                f"No original_context in BRollNeed for {clip_path.name}, "
                "using description as fallback"
            )
            context_to_check = broll_need.description
        else:
            context_to_check = broll_need.original_context

        try:
            result = await self._perform_verification(
                clip_path=clip_file,
                original_context=context_to_check,
                required_elements=broll_need.required_elements,
                search_phrase=broll_need.search_phrase,
            )

            logger.info(
                f"Verified clip {clip_file.name}: "
                f"score={result.similarity_score:.2f}, "
                f"passed={result.passed}, "
                f"elements={len(result.matched_elements)}/{len(result.matched_elements) + len(result.missing_elements)}"
            )

            return result

        except Exception as e:
            logger.error(f"Verification failed for {clip_path}: {e}")
            # On error, return a failing result if reject_below_threshold is True
            # Otherwise, pass the clip with a warning
            if self.reject_below_threshold:
                return VerificationResult(
                    passed=False,
                    similarity_score=0.0,
                    matched_elements=[],
                    missing_elements=list(broll_need.required_elements),
                    rationale=f"Verification error: {e}",
                )
            return VerificationResult(
                passed=True,
                similarity_score=0.5,
                matched_elements=[],
                missing_elements=list(broll_need.required_elements),
                rationale=f"Verification error (auto-passed): {e}",
            )

    @retry_api_call(max_retries=3, base_delay=2.0)
    async def _perform_verification(
        self,
        clip_path: Path,
        original_context: str,
        required_elements: list[str],
        search_phrase: str,
    ) -> VerificationResult:
        """Perform the actual verification using Gemini.

        Args:
            clip_path: Path to clip file
            original_context: Original transcript context
            required_elements: List of required visual elements
            search_phrase: The search phrase used to find this clip

        Returns:
            VerificationResult from AI analysis
        """
        temp_file_path = None
        uploaded_file = None

        try:
            # Handle Unicode filenames by copying to temp file
            upload_path = str(clip_path)
            try:
                str(clip_path).encode("ascii")
            except UnicodeEncodeError:
                logger.info("Filename contains Unicode characters, using temp file for upload")
                suffix = clip_path.suffix
                temp_fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
                os.close(temp_fd)
                shutil.copy2(str(clip_path), temp_file_path)
                upload_path = temp_file_path

            # Upload clip to Gemini File API
            logger.debug(f"Uploading clip for verification: {clip_path.name}")
            uploaded_file = self.client.files.upload(file=upload_path)

            # Wait for file processing
            file_ready = self._wait_for_file_processing(uploaded_file.name)
            if not file_ready:
                return VerificationResult(
                    passed=False,
                    similarity_score=0.0,
                    matched_elements=[],
                    missing_elements=list(required_elements),
                    rationale="File processing timed out or failed",
                )

            # Build verification prompt
            prompt = self._build_verification_prompt(
                original_context=original_context,
                required_elements=required_elements,
                search_phrase=search_phrase,
            )

            # Generate verification response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Low temperature for consistent evaluation
                ),
            )

            # Parse response into VerificationResult
            return self._parse_verification_response(
                response_text=response.text,
                required_elements=required_elements,
            )

        except Exception as e:
            logger.error(f"Verification API call failed: {e}")
            error_str = str(e).lower()
            if "rate limit" in error_str:
                msg = f"Rate limit hit: {e}"
                raise APIRateLimitError(msg) from e
            if "network" in error_str:
                msg = f"Network error: {e}"
                raise NetworkError(msg) from e
            raise

        finally:
            # Clean up uploaded file
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete uploaded file: {e}")

            # Clean up temp file
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def _wait_for_file_processing(self, file_name: str) -> bool:
        """Wait for uploaded file to finish processing.

        Args:
            file_name: Name of the uploaded file

        Returns:
            True if file is ready, False if failed/timeout
        """
        start_time = time.time()

        while time.time() - start_time < self.FILE_POLL_TIMEOUT:
            file_info = self.client.files.get(name=file_name)

            if file_info.state.name == "ACTIVE":
                logger.debug(f"File ready for verification: {file_name}")
                return True
            if file_info.state.name == "FAILED":
                logger.error(f"File processing failed: {file_name}")
                return False

            logger.debug(f"File still processing: {file_info.state.name}")
            time.sleep(self.FILE_POLL_INTERVAL)

        logger.error(f"File processing timeout: {file_name}")
        return False

    def _build_verification_prompt(
        self,
        original_context: str,
        required_elements: list[str],
        search_phrase: str,
    ) -> str:
        """Build the prompt for semantic verification.

        Args:
            original_context: The original transcript context
            required_elements: List of visual elements that should be present
            search_phrase: The search phrase used to find this clip

        Returns:
            Prompt string for Gemini
        """
        elements_list = (
            "\n".join(f"  - {elem}" for elem in required_elements)
            if required_elements
            else "  (none specified)"
        )

        return f"""You are a Semantic B-Roll Verifier. Your task is to analyze this video clip and verify it semantically matches the original transcript context it's meant to support.

ORIGINAL TRANSCRIPT CONTEXT:
"{original_context}"

SEARCH PHRASE USED:
"{search_phrase}"

REQUIRED VISUAL ELEMENTS (must be visible in the clip):
{elements_list}

VERIFICATION TASKS:
1. Watch the video clip carefully
2. Determine how well the visual content matches the MEANING and TONE of the original context
3. Check which required elements ARE visible in the clip
4. Check which required elements are MISSING from the clip
5. Calculate a similarity score from 0.0 to 1.0

SCORING CRITERIA:
- 0.9-1.0: Perfect match - visuals directly illustrate the transcript context, all required elements present
- 0.7-0.89: Good match - visuals generally align with context, most required elements present
- 0.5-0.69: Partial match - some visual connection to context, some elements missing
- 0.3-0.49: Weak match - tenuous connection to context, many elements missing
- 0.0-0.29: No match - visuals unrelated to context, most/all elements missing

OUTPUT FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "similarity_score": 0.85,
  "matched_elements": ["element1", "element2"],
  "missing_elements": ["element3"],
  "rationale": "Brief explanation of why this score was given, focusing on what matches and what doesn't match the original context."
}}

RULES:
- similarity_score must be a float between 0.0 and 1.0
- matched_elements: only include elements from the required list that ARE clearly visible
- missing_elements: only include elements from the required list that are NOT visible
- rationale: 1-2 sentences explaining the score
- Be strict about semantic matching - the visual should truly support the narrative context
- No markdown, no extra text, just the JSON object"""

    def _parse_verification_response(
        self,
        response_text: str,
        required_elements: list[str],
    ) -> VerificationResult:
        """Parse AI response into VerificationResult.

        Args:
            response_text: Raw AI response text
            required_elements: Original list of required elements for validation

        Returns:
            VerificationResult parsed from response
        """
        if not response_text:
            return VerificationResult(
                passed=False,
                similarity_score=0.0,
                matched_elements=[],
                missing_elements=list(required_elements),
                rationale="Empty response from AI",
            )

        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse verification JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return VerificationResult(
                passed=False,
                similarity_score=0.0,
                matched_elements=[],
                missing_elements=list(required_elements),
                rationale=f"Failed to parse AI response: {e}",
            )

        # Extract and validate fields
        try:
            similarity_score = float(data.get("similarity_score", 0.0))
            similarity_score = max(0.0, min(1.0, similarity_score))  # Clamp to [0, 1]

            matched_elements = data.get("matched_elements", [])
            if not isinstance(matched_elements, list):
                matched_elements = []
            matched_elements = [str(e) for e in matched_elements]

            missing_elements = data.get("missing_elements", [])
            if not isinstance(missing_elements, list):
                missing_elements = []
            missing_elements = [str(e) for e in missing_elements]

            rationale = str(data.get("rationale", ""))

            # Determine if clip passes based on thresholds
            passed = self._determine_pass(
                similarity_score=similarity_score,
                matched_elements=matched_elements,
                missing_elements=missing_elements,
            )

            return VerificationResult(
                passed=passed,
                similarity_score=similarity_score,
                matched_elements=matched_elements,
                missing_elements=missing_elements,
                rationale=rationale,
            )

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid verification data: {data}, error: {e}")
            return VerificationResult(
                passed=False,
                similarity_score=0.0,
                matched_elements=[],
                missing_elements=list(required_elements),
                rationale=f"Invalid verification data: {e}",
            )

    def _determine_pass(
        self,
        similarity_score: float,
        matched_elements: list[str],
        missing_elements: list[str],
    ) -> bool:
        """Determine if the clip passes verification based on thresholds.

        Args:
            similarity_score: Semantic similarity score (0.0-1.0)
            matched_elements: List of elements that were found
            missing_elements: List of elements that were not found

        Returns:
            True if the clip passes verification thresholds
        """
        # Check similarity score threshold
        if similarity_score < self.threshold:
            logger.debug(
                f"Clip failed similarity threshold: {similarity_score:.2f} < {self.threshold:.2f}"
            )
            if self.reject_below_threshold:
                return False

        # Check required elements match ratio
        total_elements = len(matched_elements) + len(missing_elements)
        if total_elements > 0:
            elements_ratio = len(matched_elements) / total_elements
            if elements_ratio < self.min_elements_match:
                logger.debug(
                    f"Clip failed elements threshold: {elements_ratio:.2f} < {self.min_elements_match:.2f}"
                )
                if self.reject_below_threshold:
                    return False

        return True

    async def verify_and_filter_clips(
        self,
        clips: list[tuple[Path, ClipSegment]],
        broll_need: BRollNeed,
        min_clips: int = 1,
    ) -> list[tuple[Path, ClipSegment, VerificationResult]]:
        """Verify all clips and filter to those passing threshold.

        Verifies each clip against the BRollNeed context and returns only
        clips that pass the verification threshold. If no clips pass,
        returns the best-scoring clip with a warning.

        Args:
            clips: List of (clip_path, clip_segment) tuples
            broll_need: BRollNeed with original_context and required_elements
            min_clips: Minimum number of clips to return (default: 1)

        Returns:
            List of (clip_path, clip_segment, verification_result) tuples
            for clips that pass verification or best-scoring if none pass
        """
        if not self.enabled:
            logger.debug("Semantic verification disabled, returning all clips as passed")
            return [
                (
                    path,
                    segment,
                    VerificationResult(
                        passed=True,
                        similarity_score=1.0,
                        matched_elements=list(broll_need.required_elements),
                        missing_elements=[],
                        rationale="Verification disabled - auto-passed",
                    ),
                )
                for path, segment in clips
            ]

        if not clips:
            logger.warning("No clips provided for verification")
            return []

        # Verify all clips
        results: list[tuple[Path, ClipSegment, VerificationResult]] = []
        for clip_path, segment in clips:
            result = await self.verify_clip(clip_path, broll_need)
            results.append((clip_path, segment, result))

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x[2].similarity_score, reverse=True)

        # Filter to passing clips
        passing = [r for r in results if r[2].passed]

        if passing:
            logger.info(
                f"Verification complete: {len(passing)}/{len(clips)} clips passed threshold"
            )
            return passing

        # If no clips pass, return best-scoring clips up to min_clips with warning
        logger.warning(
            f"No clips passed verification threshold ({self.threshold:.2f}). "
            f"Returning top {min_clips} best-scoring clip(s) with warning."
        )

        best_clips = results[:min_clips]
        for _path, _segment, result in best_clips:
            # Mark as passed with warning in rationale
            result.passed = True
            result.rationale = f"[WARNING: Below threshold] {result.rationale}"

        return best_clips
