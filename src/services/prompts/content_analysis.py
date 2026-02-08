"""Content analysis prompt templates.

Contains prompts for:
- STYLE_ANALYZER: Content style and mood detection
- CONTEXT_QUESTION_GENERATOR: Context-aware question generation for B-roll preferences
"""

# Content Style Analyzer prompt
# Template placeholders: {transcript_preview}, {duration_minutes}, {language}
STYLE_ANALYZER = """You are a Content Style Analyzer for video B-roll matching.

TASK: Analyze this video transcript to understand its content, audience, and visual style requirements.
This analysis will be used to select matching B-roll images and video clips.

TRANSCRIPT:
<<<
{transcript_preview}
>>>

VIDEO INFO:
- Duration: {duration_minutes:.1f} minutes
- Language: {language}

ANALYZE THE FOLLOWING:

1. TOPIC ANALYSIS:
   - What is the main topic/subject?
   - What are the key topic keywords (5-10 words)?

2. AUDIENCE ANALYSIS:
   - Who is the target audience?
   - What skill/knowledge level is this for? (beginner/intermediate/advanced/professional/general)
   - Any demographic hints from the language used?

3. VISUAL STYLE:
   - What visual style matches this content? (cinematic/documentary/raw/professional/energetic/moody/vintage/modern)
   - What color tone fits? (warm/cool/neutral/desaturated/vibrant/high_contrast/low_contrast)
   - What pacing? (fast/moderate/slow/mixed)

4. MOOD/TONE:
   - What is the overall tone? (serious/casual/inspirational/technical/entertaining)
   - 3-5 mood keywords that describe this content

5. IMAGERY GUIDANCE:
   - What types of imagery should be used? (5-8 specific examples)
   - What types of imagery should be AVOIDED? (5-8 specific examples that would look out of place)

OUTPUT FORMAT (JSON):
{{
  "topic": "Main topic in 5-10 words",
  "topic_keywords": ["keyword1", "keyword2", "keyword3", ...],
  "target_audience": "Description of target viewer",
  "audience_level": "beginner|intermediate|advanced|professional|general",
  "audience_demographics": "Demographic hints if any",
  "content_type": "educational|motivational|tutorial|vlog|documentary|entertainment|corporate",
  "tone": "serious|casual|inspirational|technical|entertaining",
  "visual_style": "cinematic|documentary|raw|professional|energetic|moody|vintage|modern",
  "color_tone": "warm|cool|neutral|desaturated|vibrant|high_contrast|low_contrast",
  "pacing": "fast|moderate|slow|mixed",
  "mood_keywords": ["mood1", "mood2", "mood3"],
  "preferred_imagery": ["specific imagery type 1", "specific imagery type 2", ...],
  "avoid_imagery": ["imagery to avoid 1", "imagery to avoid 2", ...],
  "is_talking_head": true|false,
  "is_tutorial": true|false,
  "is_entertainment": true|false,
  "is_corporate": true|false
}}

Return ONLY the JSON object, no markdown, no extra text."""


# Context Question Generator prompt
# Template placeholders: {max_questions}, {transcript_preview}, {duration_minutes}, {language}
CONTEXT_QUESTION_GENERATOR = """You are a creative B-roll planning assistant.

Analyze this video transcript and generate {max_questions} targeted questions to help select the best B-roll footage style.

TRANSCRIPT PREVIEW:
<<<
{transcript_preview}
>>>

VIDEO INFO:
- Duration: {duration_minutes:.1f} minutes
- Language: {language}

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
7. Return ONLY the JSON array, no markdown, no extra text"""
