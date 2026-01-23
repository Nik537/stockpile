"""AI service for phrase extraction and video evaluation using Google GenAI."""

import json
import logging
from typing import List
from google.genai import Client
from google.genai import types

from utils.retry import retry_api_call, APIRateLimitError, NetworkError
from models.video import VideoResult, ScoredVideo
from models.broll_need import BRollNeed, BRollPlan, TranscriptResult
from models.user_preferences import UserPreferences, GeneratedQuestion

logger = logging.getLogger(__name__)


def strip_markdown_code_blocks(text: str) -> str:
    """Strip markdown code blocks from AI response text.

    Args:
        text: Raw text that may contain markdown code blocks

    Returns:
        Cleaned text with markdown code blocks removed
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove ```
    return text.strip()


class AIService:
    """Service for AI-powered phrase extraction and video evaluation using Gemini."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        """Initialize Google GenAI client.

        Args:
            api_key: Google GenAI API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Client(api_key=api_key)

        logger.info(f"Initialized AI service with model: {model_name}")

    @retry_api_call(max_retries=5, base_delay=2.0)
    def extract_search_phrases(self, transcript: str) -> List[str]:
        """Extract B-roll search phrases from transcript using Gemini.

        Args:
            transcript: Transcribed text content

        Returns:
            List of search phrases for B-roll footage
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript provided for phrase extraction")
            return []

        # B-Roll Extractor v6 prompt (from original n8n workflow)
        prompt = f"""You are B-RollExtractor v6.

GOAL
Turn the transcript into stock-footage search phrases an editor can paste into Pexels, YouTube, etc.

OUTPUT
Return one JSON string array and nothing else.

Example: ["Berlin Wall falling", "vintage CRT monitor close-up", "Hitler with Stalin", "Mao era parade"]

RULES
• ≥10 phrases.
• 2–6 words each.
• Must name a tangible scene, person, object or event (no pure ideas).
• Use simple connectors ("with", "in", "during") to relate entities.
• No duplicates or name-spamming combos ("Hitler Stalin Mao").
• No markdown, no extra keys, no surrounding text.

GOOD
"1930s Kremlin meeting"
"Stalin official portrait"
"Hitler with Stalin"

BAD
"policy shift"         (abstract)
"Power dynamics"        (abstract)
"Hitler Stalin Mao"     (unclear)
"massive power"         (no concrete noun)

TRANSCRIPT ↓
<<<
{transcript}
>>>"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                ),
            )

            # Parse JSON response
            try:
                # Strip markdown code blocks if present
                if not response.text:
                    logger.error("AI response is empty")
                    return []
                response_text = strip_markdown_code_blocks(response.text)

                phrases = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract phrases from text
                logger.warning(
                    "Failed to parse JSON response, attempting to extract phrases from text"
                )
                logger.info(f"Raw AI response: {response.text}")
                import re

                # Look for quoted phrases or lines that look like phrases
                text = response.text
                if not text:
                    logger.error("AI response is empty")
                    return []
                phrases = re.findall(r'"([^"]+)"', text)
                if not phrases:
                    # Fallback: split by lines and filter
                    lines = text.strip().split("\n")
                    phrases = [
                        line.strip(" -•*")
                        for line in lines
                        if line.strip() and len(line.strip()) < 50
                    ]

            # Validate and clean phrases
            if not isinstance(phrases, list):
                logger.error("AI response is not a list")
                return []

            # Filter and clean phrases
            cleaned_phrases = []
            for phrase in phrases:
                if isinstance(phrase, str) and phrase.strip():
                    clean_phrase = phrase.strip().lower()
                    if len(clean_phrase) > 0 and clean_phrase not in cleaned_phrases:
                        cleaned_phrases.append(clean_phrase)

            # Limit to 10 phrases
            final_phrases = cleaned_phrases[:10]
            return final_phrases

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(
                f"Raw response: {response.text if 'response' in locals() else 'No response'}"
            )
            return []

        except Exception as e:
            logger.error(f"Phrase extraction failed: {e}")
            # Convert specific errors to retryable errors
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit hit: {e}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(f"Network error: {e}")
            raise

    @retry_api_call(max_retries=3, base_delay=1.0)
    def generate_context_questions(
        self,
        transcript_result: TranscriptResult,
        max_questions: int = 3,
    ) -> List[GeneratedQuestion]:
        """Generate context-aware questions based on transcript content.

        Analyzes the transcript to generate relevant questions about B-roll preferences
        specific to the content being discussed.

        Args:
            transcript_result: TranscriptResult with text and metadata
            max_questions: Maximum number of questions to generate (default: 3)

        Returns:
            List of GeneratedQuestion objects with context-specific questions
        """
        if not transcript_result.text or not transcript_result.text.strip():
            logger.warning("Empty transcript provided for question generation")
            return []

        duration_minutes = transcript_result.duration / 60.0

        # Use first 1000 characters of transcript for analysis (enough for context)
        transcript_preview = transcript_result.text[:1000]
        if len(transcript_result.text) > 1000:
            transcript_preview += "..."

        prompt = f"""You are a creative B-roll planning assistant.

Analyze this video transcript and generate {max_questions} targeted questions to help select the best B-roll footage style.

TRANSCRIPT PREVIEW:
<<<
{transcript_preview}
>>>

VIDEO INFO:
- Duration: {duration_minutes:.1f} minutes
- Language: {transcript_result.language or 'English'}

GENERATE QUESTIONS ABOUT:
1. Visual style that matches the content tone (documentary, cinematic, raw, etc.)
2. Time period/era based on topics discussed
3. Location types that complement the subjects mentioned
4. Any specific content preferences based on subjects in the transcript

PREFERENCE FIELDS YOU CAN MAP TO:
- era_period (e.g., "modern", "historical", "1950s")
- location_type (e.g., "urban", "nature", "indoor")
- color_mood (e.g., "warm", "cold", "vibrant")
- content_focus (e.g., "people", "objects", "landscapes")
- custom_notes (for anything else)

OUTPUT FORMAT (JSON array):
[
  {{
    "question_id": "q1",
    "question_text": "Your video discusses urban development. What era should the B-roll emphasize?",
    "preference_field": "era_period",
    "options": ["Mix of historical and modern", "Primarily historical (pre-1950)", "Mid-century (1950-1990)", "Contemporary (1990-present)"],
    "allows_custom": true,
    "context_reason": "Video mentions city planning evolution over decades"
  }}
]

RULES:
1. Generate EXACTLY {max_questions} questions
2. Questions MUST be based on actual transcript content
3. Each question maps to ONE preference_field
4. Provide 3-4 relevant options per question
5. Always allow custom input (allows_custom: true)
6. Include brief context_reason explaining why you asked
7. Return ONLY the JSON array, no markdown, no extra text
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,  # Moderate temperature for creative questions
                ),
            )

            if not response.text:
                logger.error("AI response is empty for question generation")
                return []

            response_text = strip_markdown_code_blocks(response.text)

            try:
                questions_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for question generation: {e}")
                logger.debug(f"Raw response: {response.text}")
                return []

            if not isinstance(questions_data, list):
                logger.error("Question generation response is not a list")
                return []

            # Parse and validate each question
            questions = []
            for item in questions_data:
                try:
                    if not isinstance(item, dict):
                        continue

                    question = GeneratedQuestion(
                        question_id=str(item.get("question_id", f"q{len(questions) + 1}")),
                        question_text=str(item.get("question_text", "")).strip(),
                        preference_field=str(item.get("preference_field", "custom_notes")).strip(),
                        options=item.get("options") if isinstance(item.get("options"), list) else None,
                        allows_custom=bool(item.get("allows_custom", True)),
                        context_reason=str(item.get("context_reason", "")).strip() if item.get("context_reason") else None,
                    )

                    if question.question_text:  # Only add if question text is not empty
                        questions.append(question)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid question: {e}")
                    continue

            logger.info(f"Generated {len(questions)} context-aware questions")
            return questions[:max_questions]  # Ensure we don't exceed max

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            # Convert specific errors to retryable errors
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit hit: {e}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(f"Network error: {e}")
            return []  # Return empty list on error

    @retry_api_call(max_retries=5, base_delay=2.0)
    def plan_broll_needs(
        self,
        transcript_result: TranscriptResult,
        clips_per_minute: float = 2.0,
        source_file: str = None,
        content_filter: str = None,
        user_preferences: UserPreferences = None,
    ) -> BRollPlan:
        """Plan timeline-aware B-roll needs from transcript with timestamps.

        Analyzes the transcript to identify specific moments that need B-roll,
        returning a BRollPlan with timestamped needs spread across the video.

        Args:
            transcript_result: TranscriptResult with text, segments, and duration
            clips_per_minute: Target B-roll density (default: 2 clips per minute)
            source_file: Optional path to source file for reference
            content_filter: Optional filter for content (e.g., "men only, no women")

        Returns:
            BRollPlan with list of BRollNeed objects
        """
        if not transcript_result.text or not transcript_result.text.strip():
            logger.warning("Empty transcript provided for B-roll planning")
            return BRollPlan(
                source_duration=transcript_result.duration,
                needs=[],
                clips_per_minute=clips_per_minute,
                source_file=source_file,
            )

        # Calculate target number of B-roll clips
        duration_minutes = transcript_result.duration / 60.0
        total_clips_needed = max(1, int(duration_minutes * clips_per_minute))

        # Format transcript with timestamps for AI analysis
        timestamped_transcript = transcript_result.format_with_timestamps()

        prompt = f"""You are B-RollPlanner v1.

GOAL
Identify specific moments in this video that need B-roll footage. You must identify approximately {total_clips_needed} B-roll opportunities spread across the video.

INPUT
- Source video duration: {transcript_result.duration:.1f} seconds ({duration_minutes:.1f} minutes)
- Target density: {clips_per_minute} clips per minute = {total_clips_needed} total clips needed
- Language detected: {transcript_result.language or 'unknown'}

TRANSCRIPT WITH TIMESTAMPS
<<<
{timestamped_transcript}
>>>

OUTPUT FORMAT
Return ONLY a JSON array with this exact structure:
[
  {{
    "timestamp": 30.5,
    "search_phrase": "city skyline sunset aerial",
    "description": "Establishing shot for urban segment",
    "context": "talking about life in the city and how it changed",
    "suggested_duration": 5
  }}
]

RULES
1. Identify exactly {total_clips_needed} B-roll needs (±2 is acceptable)
2. Spread clips EVENLY across the video timeline:
   - First clip around {transcript_result.duration * 0.05:.0f}s
   - Last clip before {transcript_result.duration * 0.95:.0f}s
   - Even spacing between clips
3. timestamp: Specific moment in source video (seconds) where B-roll should appear
4. search_phrase: 2-6 words, MUST name tangible scenes/objects/events for YouTube search
   - GOOD: "Berlin Wall falling", "vintage CRT monitor", "coffee shop interior"
   - BAD: "power dynamics", "abstract concept", "feeling of change"
5. description: What the editor should see (under 40 characters)
6. context: 10-20 words from the transcript around this moment
7. suggested_duration: How long the B-roll should play (4-15 seconds)
8. NO duplicates or very similar search phrases
9. Return ONLY the JSON array, no markdown, no extra text
{f'10. CONTENT FILTER: {content_filter}' if content_filter else ''}

{self._format_user_preferences(user_preferences) if user_preferences and user_preferences.has_preferences() else ''}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Low temperature for consistent timestamps
                ),
            )

            # Parse JSON response
            if not response.text:
                logger.error("AI response is empty for B-roll planning")
                return BRollPlan(
                    source_duration=transcript_result.duration,
                    needs=[],
                    clips_per_minute=clips_per_minute,
                    source_file=source_file,
                )

            response_text = strip_markdown_code_blocks(response.text)

            try:
                needs_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for B-roll planning: {e}")
                logger.debug(f"Raw response: {response.text}")
                return BRollPlan(
                    source_duration=transcript_result.duration,
                    needs=[],
                    clips_per_minute=clips_per_minute,
                    source_file=source_file,
                )

            if not isinstance(needs_data, list):
                logger.error("B-roll planning response is not a list")
                return BRollPlan(
                    source_duration=transcript_result.duration,
                    needs=[],
                    clips_per_minute=clips_per_minute,
                    source_file=source_file,
                )

            # Parse and validate each B-roll need
            needs = []
            for item in needs_data:
                try:
                    if not isinstance(item, dict):
                        continue

                    timestamp = float(item.get("timestamp", 0))
                    search_phrase = str(item.get("search_phrase", "")).strip()
                    description = str(item.get("description", "")).strip()
                    context = str(item.get("context", "")).strip()
                    suggested_duration = float(item.get("suggested_duration", 5.0))

                    # Validate required fields
                    if not search_phrase or not description:
                        logger.warning(f"Skipping B-roll need with missing fields: {item}")
                        continue

                    # Validate timestamp is within video duration
                    if timestamp < 0:
                        timestamp = 0
                    if timestamp > transcript_result.duration:
                        timestamp = transcript_result.duration * 0.9

                    need = BRollNeed(
                        timestamp=timestamp,
                        search_phrase=search_phrase.lower(),
                        description=description,
                        context=context,
                        suggested_duration=suggested_duration,
                    )
                    needs.append(need)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid B-roll need data: {item}, error: {e}")
                    continue

            # Sort by timestamp
            needs.sort(key=lambda n: n.timestamp)

            logger.info(
                f"B-roll planning complete: {len(needs)} needs identified "
                f"for {duration_minutes:.1f} min video (target: {total_clips_needed})"
            )

            return BRollPlan(
                source_duration=transcript_result.duration,
                needs=needs,
                clips_per_minute=clips_per_minute,
                source_file=source_file,
            )

        except Exception as e:
            logger.error(f"B-roll planning failed: {e}")
            # Convert specific errors to retryable errors
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit hit: {e}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(f"Network error: {e}")
            raise

    def _apply_content_filter(
        self, video_results: List[VideoResult], content_filter: str
    ) -> List[VideoResult]:
        """Apply content filter to remove videos based on title/description keywords.

        Args:
            video_results: List of video results to filter
            content_filter: Filter string (e.g., "men only, no women")

        Returns:
            Filtered list of video results
        """
        filter_lower = content_filter.lower()

        # Define exclusion keywords based on common filter patterns
        exclude_keywords = []

        if "no women" in filter_lower or "men only" in filter_lower:
            exclude_keywords.extend([
                "woman", "women", "female", "girl", "girls", "lady", "ladies",
                "she ", "her ", "mom", "mother", "daughter", "wife", "girlfriend",
                "actress", "businesswoman", "sportswoman"
            ])

        if "no men" in filter_lower or "women only" in filter_lower:
            exclude_keywords.extend([
                "man ", "men ", " male", "boy ", "boys", " guy", "guys",
                "he ", "his ", "dad", "father", "son ", "husband", "boyfriend",
                "actor ", "businessman", "sportsman"
            ])

        if not exclude_keywords:
            return video_results

        filtered = []
        for video in video_results:
            # Combine title and description for checking
            text = f"{video.title} {video.description or ''}".lower()

            # Check if any exclusion keyword is present
            should_exclude = any(kw in text for kw in exclude_keywords)

            if not should_exclude:
                filtered.append(video)
            else:
                logger.debug(f"Filtered out video (content filter): {video.title[:50]}...")

        return filtered

    @retry_api_call(max_retries=3, base_delay=1.0)
    def evaluate_videos(
        self, search_phrase: str, video_results: List[VideoResult], content_filter: str = None
    ) -> List[ScoredVideo]:
        """Evaluate YouTube videos for B-roll suitability using Gemini.

        Args:
            search_phrase: The search phrase used to find videos
            video_results: List of video search results
            content_filter: Optional filter for content (e.g., "men only, no women")

        Returns:
            List of scored videos (score >= 6 only)
        """
        if not video_results:
            logger.info(f"No videos to evaluate for phrase: {search_phrase}")
            return []

        # Apply content filter pre-processing if specified
        if content_filter:
            filtered_results = self._apply_content_filter(video_results, content_filter)
            if len(filtered_results) < len(video_results):
                logger.info(
                    f"Content filter removed {len(video_results) - len(filtered_results)} videos "
                    f"({len(filtered_results)} remaining)"
                )
            video_results = filtered_results

            if not video_results:
                logger.info(f"No videos remaining after content filter for: {search_phrase}")
                return []

        # Format video results for AI evaluation
        results_text = "\n".join(
            [
                f"ID: {video.video_id}\n"
                f"Title: {video.title}\n"
                f"Description: {video.description[:200] if video.description else 'N/A'}...\n"
                f"Duration: {video.duration}s\n"
                f"URL: {video.url}\n"
                "---"
                for video in video_results
            ]
        )

        evaluator_prompt = f"""You are B-Roll Evaluator. Your goal is to select the most visually relevant YouTube videos for a given search phrase.

You will be given a search phrase and a list of YouTube search results including their titles and descriptions.

SEARCH PHRASE:
"{search_phrase}"

YOUTUBE RESULTS:
---
{results_text}
---

TASK:
1. Analyze the title and description of each video.
2. Compare them against the search phrase.
3. Choose the videos that are most likely to contain generic, high-quality B-roll footage matching the phrase.
4. Prioritize cinematic shots, stock footage, documentary clips. Avoid vlogs, talk shows, tutorials, or videos with prominent branding.
5. Rate each video from 1-10 based on B-roll potential.

OUTPUT:
Return a JSON array of objects with video_id and score for videos scoring 6 or higher, ordered by score (highest first).
Format: [{{"video_id": "abc123", "score": 9}}, {{"video_id": "def456", "score": 7}}]
Return only the JSON array, nothing else."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=evaluator_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent evaluation
                ),
            )

            try:
                if not response.text:
                    logger.error("AI response is empty")
                    return []
                text = strip_markdown_code_blocks(response.text)
                scored_results = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("JSON parsing failed, extracting video IDs from text")
                import re

                text = response.text
                if not text:
                    logger.error("AI response is empty")
                    return []
                matches = re.findall(
                    r'"video_id":\s*"([^"]+)".*?"score":\s*(\d+)', text
                )
                scored_results = [
                    {"video_id": vid_id, "score": int(score)}
                    for vid_id, score in matches
                    if int(score) >= 6
                ]
                if not scored_results:
                    logger.error(
                        "Could not extract any valid video evaluations from response"
                    )
                    return []

            # Validate response format
            if not isinstance(scored_results, list):
                logger.error("AI evaluation response is not a list")
                return []

            # Create ScoredVideo objects
            scored_videos = []
            video_lookup = {v.video_id: v for v in video_results}

            for item in scored_results:
                if (
                    not isinstance(item, dict)
                    or "video_id" not in item
                    or "score" not in item
                ):
                    logger.warning(f"Invalid scored video format: {item}")
                    continue

                video_id = item["video_id"]
                score = item["score"]

                # Validate score
                if not isinstance(score, (int, float)) or score < 6 or score > 10:
                    logger.warning(f"Invalid score {score} for video {video_id}")
                    continue

                # Find corresponding video result
                if video_id not in video_lookup:
                    logger.warning(f"Video ID {video_id} not found in original results")
                    continue

                scored_video = ScoredVideo(
                    video_id=video_id,
                    score=int(score),
                    video_result=video_lookup[video_id],
                )
                scored_videos.append(scored_video)

            # Sort by score (highest first)
            scored_videos.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                f"Evaluated videos for '{search_phrase}': {len(scored_videos)} videos scored >= 6"
            )
            return scored_videos

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse video evaluation response as JSON: {e}")
            logger.debug(
                f"Raw response: {response.text if 'response' in locals() else 'No response'}"
            )
            return []

        except Exception as e:
            logger.error(f"Video evaluation failed for phrase '{search_phrase}': {e}")
            # Convert specific errors to retryable errors
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit hit: {e}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(f"Network error: {e}")
            raise

    def _format_user_preferences(self, user_preferences: UserPreferences) -> str:
        """Format user preferences for injection into AI prompts.

        Args:
            user_preferences: UserPreferences object with collected preferences

        Returns:
            Formatted string section for prompt injection
        """
        if not user_preferences or not user_preferences.has_preferences():
            return ""

        preferences_text = user_preferences.to_prompt_instructions()
        return f"""
USER PREFERENCES
Apply these preferences when selecting B-roll:
{preferences_text}
"""
