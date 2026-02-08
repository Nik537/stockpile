"""B-Roll related prompt templates.

Contains prompts for:
- BROLL_EXTRACTOR_V6: Extract search phrases from transcript
- BROLL_PLANNER_V3: Plan timeline-aware B-roll needs
"""

# B-Roll Extractor v6 prompt (from original n8n workflow)
# Template placeholders: {transcript}
BROLL_EXTRACTOR_V6 = """You are B-RollExtractor v6.

GOAL
Turn the transcript into stock-footage search phrases an editor can paste into Pexels, YouTube, etc.

OUTPUT
Return one JSON string array and nothing else.

Example: ["Berlin Wall falling", "vintage CRT monitor close-up", "Hitler with Stalin", "Mao era parade"]

RULES
* >=10 phrases.
* 2-6 words each.
* Must name a tangible scene, person, object or event (no pure ideas).
* Use simple connectors ("with", "in", "during") to relate entities.
* No duplicates or name-spamming combos ("Hitler Stalin Mao").
* No markdown, no extra keys, no surrounding text.

GOOD
"1930s Kremlin meeting"
"Stalin official portrait"
"Hitler with Stalin"

BAD
"policy shift"         (abstract)
"Power dynamics"        (abstract)
"Hitler Stalin Mao"     (unclear)
"massive power"         (no concrete noun)

TRANSCRIPT
<<<
{transcript}
>>>"""


# B-Roll Planner v3 prompt (Semantic Context Aware)
# Template placeholders:
# - {total_clips_needed}
# - {source_duration}
# - {duration_minutes}
# - {language}
# - {timestamped_transcript}
# - {first_clip_time} (calculated as source_duration * 0.05)
# - {last_clip_time} (calculated as source_duration * 0.95)
# - {clips_per_minute}
#
# Optional sections appended separately:
# - content_filter_section (if content_filter is provided)
# - user_preferences_section (if user_preferences is provided)
BROLL_PLANNER_V3 = """You are B-RollPlanner v3 (Semantic Context Aware).

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
- Source video duration: {source_duration:.1f} seconds ({duration_minutes:.1f} minutes)
- Target density: {clips_per_minute} clips per minute = {total_clips_needed} total clips needed
- Language detected: {language}

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
   - First clip around {first_clip_time:.0f}s
   - Last clip before {last_clip_time:.0f}s
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
10. description: What the editor should see (40-55 characters, be descriptive)
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
16. Return ONLY the JSON array, no markdown, no extra text"""
