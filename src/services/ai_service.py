"""AI service for phrase extraction and video evaluation using Google GenAI.

S2 IMPROVEMENT: AI Response Caching
- Caches AI responses keyed by content hash + prompt version
- 100% cost savings on re-runs with same content
- Increment PROMPT_VERSIONS when prompts change to invalidate cache
"""

import json
import logging
from typing import Optional

from google.genai import Client, types
from models.broll_need import BRollNeed, BRollPlan, TranscriptResult
from models.user_preferences import GeneratedQuestion, UserPreferences
from models.video import ScoredVideo, VideoResult
from utils.cache import AIResponseCache, compute_content_hash
from utils.retry import APIRateLimitError, NetworkError, retry_api_call

logger = logging.getLogger(__name__)


# S2 IMPROVEMENT: Prompt version identifiers for cache invalidation
# IMPORTANT: Increment these when prompts change to invalidate stale cached responses
PROMPT_VERSIONS = {
    "extract_search_phrases": "v6",
    "generate_context_questions": "v1",
    "plan_broll_needs": "v3",  # B-roll planning with original_context and required_elements
    "evaluate_videos": "v4",  # Video evaluation with semantic context scoring + negative examples
}


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

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-flash-preview",
        cache: Optional[AIResponseCache] = None,
    ):
        """Initialize Google GenAI client.

        Args:
            api_key: Google GenAI API key
            model_name: Gemini model to use
            cache: Optional AIResponseCache for caching responses
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = Client(api_key=api_key)
        self.cache = cache

        if cache:
            logger.info(f"Initialized AI service with model: {model_name} (caching enabled)")
        else:
            logger.info(f"Initialized AI service with model: {model_name}")

    @retry_api_call(max_retries=5, base_delay=2.0)
    def extract_search_phrases(self, transcript: str) -> list[str]:
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
    ) -> list[GeneratedQuestion]:
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
                        options=item.get("options")
                        if isinstance(item.get("options"), list)
                        else None,
                        allows_custom=bool(item.get("allows_custom", True)),
                        context_reason=str(item.get("context_reason", "")).strip()
                        if item.get("context_reason")
                        else None,
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

        S2 IMPROVEMENT: Results are cached by transcript hash for 100% savings on re-runs.
        Cache key includes prompt version, transcript hash, and parameters.

        Args:
            transcript_result: TranscriptResult with text, segments, and duration
            clips_per_minute: Target B-roll density (default: 2 clips per minute)
            source_file: Optional path to source file for reference
            content_filter: Optional filter for content (e.g., "men only, no women")
            user_preferences: Optional user preferences for B-roll customization

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

        # S2 IMPROVEMENT: Generate cache key from transcript content and parameters
        # Include user preferences hash if present to ensure different preferences = different cache
        user_prefs_hash = ""
        if user_preferences and user_preferences.has_preferences():
            user_prefs_hash = compute_content_hash(user_preferences.to_prompt_instructions())[:16]

        cache_key_content = (
            f"{PROMPT_VERSIONS['plan_broll_needs']}|"
            f"{self._generate_transcript_hash(transcript_result)}|"
            f"{clips_per_minute}|{content_filter or ''}|{user_prefs_hash}"
        )

        # S2 IMPROVEMENT: Check cache before making API call
        if self.cache:
            cached_response = self.cache.get(cache_key_content, self.model_name)
            if cached_response:
                try:
                    cached_data = json.loads(cached_response)
                    needs = self._parse_broll_needs_from_data(
                        cached_data, transcript_result.duration
                    )
                    logger.info(
                        f"[CACHE HIT] B-roll planning: {len(needs)} needs from cache "
                        f"(saved ~$0.01 API cost)"
                    )
                    return BRollPlan(
                        source_duration=transcript_result.duration,
                        needs=needs,
                        clips_per_minute=clips_per_minute,
                        source_file=source_file,
                    )
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to parse cached B-roll plan: {e}")
                    # Continue to make fresh API call

        # Format transcript with timestamps for AI analysis
        timestamped_transcript = transcript_result.format_with_timestamps()

        # Q2 ENHANCEMENT: Enhanced planning prompt with alternate searches, negatives, and visual metadata
        # Fix 4: Added original_context and required_elements for semantic context preservation
        prompt = f"""You are B-RollPlanner v3 (Semantic Context Aware).

GOAL
Identify specific moments in this video that need B-roll footage. You must identify approximately {total_clips_needed} B-roll opportunities spread across the video.

For each moment, provide ENHANCED search metadata AND semantic context:
- Primary search phrase (most specific)
- 2-3 alternate search phrases (synonyms, related terms)
- Negative keywords (what should NOT appear in results)
- Visual style preference (cinematic, documentary, raw, vlog)
- Time of day if relevant (golden hour, night, day)
- Camera movement preference (static, pan, drone, handheld, tracking)
- FULL ORIGINAL CONTEXT from transcript (50-150 characters)
- REQUIRED VISUAL ELEMENTS that must appear in the clip (3-6 specific items)

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
    "search_phrase": "coffee shop interior morning busy",
    "description": "Busy cafe scene with remote workers",
    "context": "talking about remote work culture",
    "original_context": "The coffee shop was packed with remote workers typing on laptops during the morning rush, everyone with their headphones in, completely focused on their screens.",
    "required_elements": ["people", "laptops", "coffee shop interior", "busy atmosphere", "morning light"],
    "suggested_duration": 5,
    "alternate_searches": [
      "cafe coworking space laptops",
      "remote workers coffee shop",
      "people working laptops cafe"
    ],
    "negative_keywords": ["empty", "night", "text overlay", "logo", "watermark"],
    "visual_style": "cinematic",
    "time_of_day": "morning",
    "movement": "slow pan"
  }}
]

RULES
1. Identify exactly {total_clips_needed} B-roll needs (+-2 is acceptable)
2. Spread clips EVENLY across the video timeline:
   - First clip around {transcript_result.duration * 0.05:.0f}s
   - Last clip before {transcript_result.duration * 0.95:.0f}s
   - Even spacing between clips
3. timestamp: Specific moment in source video (seconds) where B-roll should appear
4. search_phrase: 2-6 words, MUST name tangible scenes/objects/events for YouTube search
   - Include visual descriptors (aerial, close-up, wide shot)
   - GOOD: "Berlin Wall falling", "vintage CRT monitor close-up", "coffee shop interior"
   - BAD: "power dynamics", "abstract concept", "feeling of change"
5. alternate_searches: 2-3 synonym or related search phrases for fallback
6. negative_keywords: 3-5 terms that should NOT appear in video titles/descriptions
   - Always include: "watermark", "logo", "text overlay" for stock footage
   - Add content-specific exclusions based on context
7. visual_style: One of "cinematic", "documentary", "raw", "vlog", or null if any
8. time_of_day: "golden hour", "night", "day", "morning", or null if doesn't matter
9. movement: "static", "pan", "drone", "handheld", "tracking", or null if any
10. description: What the editor should see (under 40 characters)
11. context: 10-20 words summary from the transcript around this moment
12. suggested_duration: How long the B-roll should play (4-15 seconds)
13. original_context: CRITICAL - The EXACT transcript segment (50-150 chars) that this B-roll supports.
    - Copy the actual words from the transcript around this timestamp
    - This preserves the full meaning for downstream AI evaluation
    - Example: "More and more people are working from coffee shops and co-working spaces"
14. required_elements: 3-6 SPECIFIC visual elements that MUST appear in the selected clip
    - Be concrete: "laptops" not "technology", "morning light" not "lighting"
    - Include people, objects, settings, atmosphere, lighting as applicable
    - These will be used to filter clips during evaluation
15. NO duplicates or very similar search phrases
16. Return ONLY the JSON array, no markdown, no extra text
{f'17. CONTENT FILTER: {content_filter}' if content_filter else ''}

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

                    # Q2 ENHANCEMENT: Parse enhanced metadata fields
                    alternate_searches = item.get("alternate_searches", [])
                    if isinstance(alternate_searches, list):
                        alternate_searches = [
                            str(s).strip().lower() for s in alternate_searches if s
                        ]
                    else:
                        alternate_searches = []

                    negative_keywords = item.get("negative_keywords", [])
                    if isinstance(negative_keywords, list):
                        negative_keywords = [str(k).strip().lower() for k in negative_keywords if k]
                    else:
                        negative_keywords = []

                    visual_style = item.get("visual_style")
                    if visual_style:
                        visual_style = str(visual_style).strip().lower()

                    time_of_day = item.get("time_of_day")
                    if time_of_day:
                        time_of_day = str(time_of_day).strip().lower()

                    movement = item.get("movement")
                    if movement:
                        movement = str(movement).strip().lower()

                    # Fix 4: Parse semantic context preservation fields
                    original_context = str(item.get("original_context", "")).strip()
                    required_elements = item.get("required_elements", [])
                    if isinstance(required_elements, list):
                        required_elements = [str(e).strip().lower() for e in required_elements if e]
                    else:
                        required_elements = []

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
                        # Fix 4: Semantic context preservation fields
                        original_context=original_context,
                        required_elements=required_elements,
                        # Q2 ENHANCEMENT: Enhanced metadata fields
                        alternate_searches=alternate_searches,
                        negative_keywords=negative_keywords,
                        visual_style=visual_style,
                        time_of_day=time_of_day,
                        movement=movement,
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

            # S2 IMPROVEMENT: Cache the successful response
            if self.cache and response_text:
                self.cache.set(cache_key_content, self.model_name, response_text)
                logger.debug("[CACHE SAVE] B-roll planning response cached")

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
        self, video_results: list[VideoResult], content_filter: str
    ) -> list[VideoResult]:
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
            exclude_keywords.extend(
                [
                    "woman",
                    "women",
                    "female",
                    "girl",
                    "girls",
                    "lady",
                    "ladies",
                    "she ",
                    "her ",
                    "mom",
                    "mother",
                    "daughter",
                    "wife",
                    "girlfriend",
                    "actress",
                    "businesswoman",
                    "sportswoman",
                ]
            )

        if "no men" in filter_lower or "women only" in filter_lower:
            exclude_keywords.extend(
                [
                    "man ",
                    "men ",
                    " male",
                    "boy ",
                    "boys",
                    " guy",
                    "guys",
                    "he ",
                    "his ",
                    "dad",
                    "father",
                    "son ",
                    "husband",
                    "boyfriend",
                    "actor ",
                    "businessman",
                    "sportsman",
                ]
            )

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
        self,
        search_phrase: str,
        video_results: list[VideoResult],
        content_filter: str = None,
        transcript_segment: str = None,
        broll_need: Optional[BRollNeed] = None,
        user_preferences: Optional[UserPreferences] = None,
    ) -> list[ScoredVideo]:
        """Evaluate YouTube videos for B-roll suitability using Gemini.

        S2 IMPROVEMENT: Results are cached by video list hash for 100% savings on re-runs.
        Q2 ENHANCEMENT: Optionally uses BRollNeed metadata for context-aware evaluation.
        Q4 IMPROVEMENT: Context-aware evaluation with transcript segment and user preferences.
        Cache key includes prompt version, search phrase, video list hash, and content filter.

        Args:
            search_phrase: The search phrase used to find videos
            video_results: List of video search results
            content_filter: Optional filter for content (e.g., "men only, no women")
            transcript_segment: Optional transcript context for evaluation
            broll_need: Optional BRollNeed with enhanced metadata (visual_style, etc.)
            user_preferences: Optional UserPreferences for style customization

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

        # S2 IMPROVEMENT: Generate cache key from video list and parameters
        cache_key_content = (
            f"{PROMPT_VERSIONS['evaluate_videos']}|"
            f"{compute_content_hash(search_phrase)[:16]}|"
            f"{self._generate_video_list_hash(video_results)}|"
            f"{content_filter or ''}"
        )

        # S2 IMPROVEMENT: Check cache before making API call
        if self.cache:
            cached_response = self.cache.get(cache_key_content, self.model_name)
            if cached_response:
                try:
                    cached_data = json.loads(cached_response)
                    # Reconstruct ScoredVideo objects from cached data
                    video_lookup = {v.video_id: v for v in video_results}
                    scored_videos = []
                    for item in cached_data:
                        if item.get("video_id") in video_lookup:
                            scored_video = ScoredVideo(
                                video_id=item["video_id"],
                                score=int(item["score"]),
                                video_result=video_lookup[item["video_id"]],
                            )
                            scored_videos.append(scored_video)
                    logger.info(
                        f"[CACHE HIT] Video evaluation: {len(scored_videos)} videos "
                        f"for '{search_phrase}' (saved ~$0.003 API cost)"
                    )
                    return scored_videos
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Failed to parse cached video evaluation: {e}")
                    # Continue to make fresh API call

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

        # Q4 IMPROVEMENT: Use enhanced prompt if BRollNeed metadata or user preferences available
        # FIX 1 & 3: Also use enhanced prompt when original_context or required_elements are present
        has_context = (
            (broll_need and broll_need.has_enhanced_metadata())
            or (broll_need and broll_need.original_context)
            or (broll_need and broll_need.required_elements)
            or transcript_segment
            or (user_preferences and user_preferences.has_preferences())
        )
        if has_context:
            evaluator_prompt = self._build_enhanced_evaluation_prompt(
                search_phrase=search_phrase,
                results_text=results_text,
                broll_need=broll_need,
                transcript_segment=transcript_segment,
                user_preferences=user_preferences,
            )
            logger.debug(
                f"Using context-aware evaluation prompt: "
                f"broll_need={broll_need is not None}, "
                f"transcript_segment={bool(transcript_segment)}, "
                f"user_prefs={user_preferences.has_preferences() if user_preferences else False}"
            )
        else:
            # Original evaluation prompt (fallback for non-enhanced needs)
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
                matches = re.findall(r'"video_id":\s*"([^"]+)".*?"score":\s*(\d+)', text)
                scored_results = [
                    {"video_id": vid_id, "score": int(score)}
                    for vid_id, score in matches
                    if int(score) >= 6
                ]
                if not scored_results:
                    logger.error("Could not extract any valid video evaluations from response")
                    return []

            # Validate response format
            if not isinstance(scored_results, list):
                logger.error("AI evaluation response is not a list")
                return []

            # Create ScoredVideo objects
            scored_videos = []
            video_lookup = {v.video_id: v for v in video_results}

            for item in scored_results:
                if not isinstance(item, dict) or "video_id" not in item or "score" not in item:
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

            # S2 IMPROVEMENT: Cache the successful response
            if self.cache and scored_videos:
                # Cache the scored results as JSON for later retrieval
                cache_data = [{"video_id": sv.video_id, "score": sv.score} for sv in scored_videos]
                self.cache.set(cache_key_content, self.model_name, json.dumps(cache_data))
                logger.debug(f"[CACHE SAVE] Video evaluation cached ({len(scored_videos)} results)")

            logger.info(
                f"Evaluated videos for '{search_phrase}': "
                f"{len(scored_videos)} videos scored >= 6"
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

    # S2 IMPROVEMENT: Cache helper methods

    def _generate_transcript_hash(self, transcript_result: TranscriptResult) -> str:
        """Generate a hash of transcript content for cache keying.

        Args:
            transcript_result: TranscriptResult with text and segments

        Returns:
            SHA-256 hash of transcript content (first 32 chars)
        """
        # Use transcript text plus duration as the content to hash
        content = f"{transcript_result.text}|{transcript_result.duration}"
        return compute_content_hash(content)[:32]

    def _generate_video_list_hash(self, video_results: list[VideoResult]) -> str:
        """Generate a hash of video list for cache keying.

        Args:
            video_results: List of VideoResult objects

        Returns:
            SHA-256 hash of video IDs (first 32 chars)
        """
        # Hash the sorted list of video IDs
        video_ids = sorted([v.video_id for v in video_results])
        content = "|".join(video_ids)
        return compute_content_hash(content)[:32]

    def _parse_broll_needs_from_data(self, needs_data: list, duration: float) -> list[BRollNeed]:
        """Parse B-roll needs from cached JSON data.

        Args:
            needs_data: List of dicts with B-roll need data
            duration: Video duration for validation

        Returns:
            List of BRollNeed objects
        """
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

                # Parse Q2 enhanced metadata fields
                alternate_searches = item.get("alternate_searches", [])
                if isinstance(alternate_searches, list):
                    alternate_searches = [str(s).strip().lower() for s in alternate_searches if s]
                else:
                    alternate_searches = []

                negative_keywords = item.get("negative_keywords", [])
                if isinstance(negative_keywords, list):
                    negative_keywords = [str(k).strip().lower() for k in negative_keywords if k]
                else:
                    negative_keywords = []

                visual_style = item.get("visual_style")
                if visual_style:
                    visual_style = str(visual_style).strip().lower()

                time_of_day = item.get("time_of_day")
                if time_of_day:
                    time_of_day = str(time_of_day).strip().lower()

                movement = item.get("movement")
                if movement:
                    movement = str(movement).strip().lower()

                # Fix 4: Parse semantic context preservation fields
                original_context = str(item.get("original_context", "")).strip()
                required_elements = item.get("required_elements", [])
                if isinstance(required_elements, list):
                    required_elements = [str(e).strip().lower() for e in required_elements if e]
                else:
                    required_elements = []

                # Validate required fields
                if not search_phrase or not description:
                    continue

                # Validate timestamp is within video duration
                if timestamp < 0:
                    timestamp = 0
                if timestamp > duration:
                    timestamp = duration * 0.9

                need = BRollNeed(
                    timestamp=timestamp,
                    search_phrase=search_phrase.lower(),
                    description=description,
                    context=context,
                    suggested_duration=suggested_duration,
                    # Fix 4: Semantic context preservation fields
                    original_context=original_context,
                    required_elements=required_elements,
                    # Q2 Enhanced metadata fields
                    alternate_searches=alternate_searches,
                    negative_keywords=negative_keywords,
                    visual_style=visual_style,
                    time_of_day=time_of_day,
                    movement=movement,
                )
                needs.append(need)

            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid B-roll need data in cache: {item}, error: {e}")
                continue

        # Sort by timestamp
        needs.sort(key=lambda n: n.timestamp)
        return needs

    def filter_by_negative_keywords(
        self, video_results: list[VideoResult], negative_keywords: list[str]
    ) -> list[VideoResult]:
        """Filter video results by negative keywords.

        Part of Q2 Enhanced Search Phrase Generation improvement.

        Args:
            video_results: List of video results to filter
            negative_keywords: Keywords that should NOT appear

        Returns:
            Filtered list of video results
        """
        if not negative_keywords:
            return video_results

        filtered = []
        for video in video_results:
            # Combine title and description for checking
            text = f"{video.title} {video.description or ''}".lower()

            # Check if any negative keyword is present
            has_negative = any(kw.lower() in text for kw in negative_keywords)

            if not has_negative:
                filtered.append(video)
            else:
                logger.debug(f"Filtered out video (negative keyword): {video.title[:50]}...")

        if len(filtered) < len(video_results):
            logger.info(
                f"Negative keyword filter removed {len(video_results) - len(filtered)} videos"
            )

        return filtered

    def _generate_negative_examples(self, broll_need: BRollNeed) -> str:
        """Generate examples of videos to reject based on context and required elements.

        FIX 5: Prevents false positive matches by showing AI what NOT to match.
        This helps prevent high-quality generic clips from scoring well when they
        don't match the specific context.

        Args:
            broll_need: BRollNeed with original_context and required_elements

        Returns:
            Formatted string with rejection patterns, or empty string if insufficient data
        """
        if not broll_need.original_context or not broll_need.required_elements:
            return ""

        # Build negative examples based on missing each required element
        examples = ["REJECT these types of videos:"]

        # For each required element, describe what a video missing it would look like
        for element in broll_need.required_elements[:4]:  # Limit to 4 examples
            examples.append(f"- Video missing '{element}': Does not show {element}")

        # Add general rejection patterns based on context
        examples.append(
            "- Video showing opposite of context: e.g., if context mentions 'busy', reject 'empty'"
        )
        examples.append(
            "- Video matching keywords but wrong meaning: generic stock footage unrelated to specific context"
        )

        return "\n".join(examples)

    def _build_enhanced_evaluation_prompt(
        self,
        search_phrase: str,
        results_text: str,
        broll_need: Optional[BRollNeed] = None,
        transcript_segment: Optional[str] = None,
        user_preferences: Optional[UserPreferences] = None,
    ) -> str:
        """Build a context-aware evaluation prompt using all available context.

        Q2 ENHANCEMENT: Uses visual_style, time_of_day, movement, and negative_keywords
        to create a more targeted evaluation prompt.

        Q4 IMPROVEMENT: Now includes transcript context and user preferences for
        more relevant scoring based on what the narrator is discussing and user's
        style preferences.

        FIX 1 & 3: Enhanced to use original_context and required_elements from BRollNeed
        for semantic-aware scoring that prioritizes meaning over keywords.

        FIX 5: Now includes negative examples to prevent false positive matches.

        Args:
            search_phrase: The search phrase used to find videos
            results_text: Formatted text of video results
            broll_need: Optional BRollNeed with enhanced metadata
            transcript_segment: Optional transcript context around B-roll timestamp
            user_preferences: Optional UserPreferences for style customization

        Returns:
            Context-aware evaluator prompt string
        """
        # FIX 1: Build original context section from BRollNeed (highest priority)
        original_context_section = ""
        if broll_need and broll_need.original_context:
            original_context_section = f"""
ORIGINAL CONTEXT (from transcript - THIS IS THE PRIMARY SCORING CRITERION):
"{broll_need.original_context}"

Score based on match to ORIGINAL CONTEXT above, not just the search phrase.
The search phrase is derived from this context - but videos must match the MEANING, not just keywords.
"""

        # FIX 3: Build required elements section
        required_elements_section = ""
        if broll_need and broll_need.required_elements:
            elements_list = "\n".join(f"- {elem}" for elem in broll_need.required_elements)
            required_elements_section = f"""
REQUIRED VISUAL ELEMENTS (must be present in the video):
{elements_list}

CRITICAL: A video that matches the search phrase but is MISSING more than 2 required elements
should score 4 or lower. All required elements should ideally be present.
"""

        # FIX 5: Generate negative examples to prevent false positives
        negative_examples_section = ""
        if broll_need:
            negative_examples = self._generate_negative_examples(broll_need)
            if negative_examples:
                negative_examples_section = f"""
{negative_examples}

If a video matches any of the REJECT patterns above, score it <=4 regardless of technical quality.
"""

        # Build visual preference section from BRollNeed metadata
        visual_preferences = ""
        if broll_need and broll_need.has_enhanced_metadata():
            prefs = []
            if broll_need.visual_style:
                prefs.append(f"- Preferred style: {broll_need.visual_style}")
            if broll_need.time_of_day:
                prefs.append(f"- Preferred time of day: {broll_need.time_of_day}")
            if broll_need.movement:
                prefs.append(f"- Preferred camera movement: {broll_need.movement}")
            if broll_need.negative_keywords:
                prefs.append(f"- AVOID videos with: {', '.join(broll_need.negative_keywords)}")

            if prefs:
                visual_preferences = (
                    "\nVISUAL PREFERENCES (use these to boost/penalize scores):\n"
                    + "\n".join(prefs)
                )

        # Fallback to transcript_segment if no original_context (backwards compatibility)
        context_section = ""
        if not original_context_section and transcript_segment:
            context_section = f"""
TRANSCRIPT CONTEXT (what is being discussed at this moment):
"{transcript_segment[:300]}"

Consider: Does the video match the TOPIC and TONE of what's being discussed?
"""

        # Build user preferences section
        user_prefs_section = ""
        if user_preferences and user_preferences.has_preferences():
            prefs_text = user_preferences.to_prompt_instructions()
            user_prefs_section = f"""
USER PREFERENCES (apply these as scoring criteria):
{prefs_text}
"""

        return f"""You are B-Roll Evaluator v4 (Semantic Context-Aware).

Your task is to evaluate videos based on how well they match the ORIGINAL CONTEXT from the transcript,
not just the search keywords. A video can match keywords but completely miss the meaning.

DERIVED SEARCH PHRASE: "{search_phrase}"
{original_context_section}
{required_elements_section}
YOUTUBE RESULTS:
---
{results_text}
---
{visual_preferences}
{context_section}
{user_prefs_section}
{negative_examples_section}
SCORING WEIGHTS (total 100%):
1. **SEMANTIC MATCH TO ORIGINAL CONTEXT (50%)** - Does the video capture the MEANING of what's being discussed?
   - A video about "coffee shop remote work" should show people working, not just coffee being poured
   - The visual must support the NARRATIVE, not just contain matching keywords
2. **REQUIRED ELEMENTS PRESENT (20%)** - Are all the required visual elements visible?
   - Check each required element against the video title/description
   - Each missing element reduces the score significantly
3. **TECHNICAL QUALITY (20%)** - Resolution, stability, lighting, production value
   - Cinematic/stock footage quality preferred
   - Avoid amateur, shaky, or poorly lit content
4. **ABSENCE OF UNWANTED ELEMENTS (10%)** - No text overlays, logos, watermarks, branding
   - Deduct points for vlogs, talk shows, tutorials, reaction videos
   - Deduct points for prominent branding or advertising

REJECTION CRITERIA (automatic low scores):
- Score <=3: Video shows the OPPOSITE of what was described (e.g., search for "people working" but video shows empty office)
- Score <=4: More than 2 required elements are missing from the video
- Score <=5: Video matches keywords but completely misses the semantic meaning

RATING SCALE:
- 9-10: Perfect semantic match + all required elements + high quality + clean footage
- 7-8: Good semantic match + most required elements present + decent quality
- 6: Acceptable match but missing some elements or context alignment
- <6: Not suitable (don't include in output)

OUTPUT FORMAT:
Return a JSON array of objects with video_id and score for videos scoring 6 or higher.
Order by score (highest first).
Format: [{{"video_id": "abc123", "score": 9}}, {{"video_id": "def456", "score": 7}}]
Return ONLY the JSON array, nothing else."""
