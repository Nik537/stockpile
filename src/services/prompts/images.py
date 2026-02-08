"""Image-related prompt templates.

Contains prompts for:
- IMAGE_QUERY_GENERATOR: Basic image query generation
- IMAGE_QUERY_CONTEXT_AWARE: Context-aware image query generation with extended context window
- IMAGE_SELECTOR: AI-powered image selection from candidates
"""

# Image Query Generator prompt
# Template placeholders: {transcript_formatted}, {timestamps}
IMAGE_QUERY_GENERATOR = """You are an Image Query Generator for video content.

TASK: Analyze this transcript and generate an optimized image search phrase for each timestamp.
The images will be used as visual aids (B-roll) at these specific moments in a video.

TRANSCRIPT:
{transcript_formatted}

TIMESTAMPS TO GENERATE QUERIES FOR:
{timestamps}

RULES FOR SEARCH PHRASES:
1. Each search phrase should be 2-5 words
2. Focus on concrete, visual concepts that stock photo sites can match
3. Avoid abstract concepts - prefer "coffee shop laptop" over "productivity"
4. Include visual style hints when relevant: "aerial", "close-up", "silhouette"
5. Match the emotional tone of the transcript at that moment
6. Consider what visual would BEST illustrate what's being discussed

OUTPUT FORMAT:
Return a JSON array with one object per timestamp:
[
  {{
    "timestamp": 0,
    "search_phrase": "city skyline sunset",
    "required_elements": ["buildings", "sky", "sunset colors"],
    "visual_style": "cinematic"
  }},
  {{
    "timestamp": 5,
    "search_phrase": "laptop coffee shop",
    "required_elements": ["laptop", "coffee", "workspace"],
    "visual_style": "lifestyle"
  }}
]

Return ONLY the JSON array, nothing else."""


# Context-Aware Image Query Generator prompt
# Template placeholders:
# - {context_window}
# - {transcript_with_timestamps}
# - {style_guidance} (optional, from ContentStyle)
# - {timestamp_contexts} (JSON array of timestamp/context pairs)
IMAGE_QUERY_CONTEXT_AWARE = """You are an Image Query Generator with Context Awareness.

TASK: Analyze this transcript and generate optimized image search phrases.
For each timestamp, you have access to +/-{context_window} seconds of surrounding context.
Use this extended context to understand what's being discussed and generate better queries.

TRANSCRIPT WITH TIMESTAMPS:
{transcript_with_timestamps}
{style_guidance}
TIMESTAMPS WITH CONTEXT WINDOWS:
{timestamp_contexts}

ANALYSIS REQUIREMENTS:
For each timestamp, analyze the +/-{context_window}s context window to:
1. Identify key THEMES (main topics being discussed)
2. Extract ENTITIES (people, places, objects mentioned)
3. Detect EMOTIONAL TONE (serious, excited, contemplative, etc.)
4. Generate a search phrase that captures the semantic meaning

RULES FOR SEARCH PHRASES:
1. 2-5 words, focused on concrete visuals
2. Consider the FULL context window, not just the exact timestamp
3. Match the emotional tone of the surrounding discussion
4. Avoid generic terms - be specific to what's being discussed
5. Include visual style hints when relevant

OUTPUT FORMAT:
Return a JSON array with enhanced context for each timestamp:
[
  {{
    "timestamp": 0,
    "search_phrase": "intense gym training session",
    "required_elements": ["athlete", "weights", "gym equipment"],
    "visual_style": "raw",
    "themes": ["fitness", "dedication", "training"],
    "entities": ["gym", "athlete"],
    "emotional_tone": "motivated"
  }}
]

Return ONLY the JSON array, nothing else."""


# Image Selector prompt
# Template placeholders:
# - {search_phrase}
# - {context}
# - {style_section} (optional, from ContentStyle)
# - {feedback_section} (optional, from feedback history)
# - {candidates_text}
IMAGE_SELECTOR = """You are an Image Selector for video content.

TASK: Select the BEST image from these candidates to use as a visual aid in a video.

SEARCH PHRASE: "{search_phrase}"
TRANSCRIPT CONTEXT: "{context}"
{style_section}{feedback_section}
CANDIDATES:
{candidates_text}

SELECTION CRITERIA (in order of importance):
1. RELEVANCE (40%): Which image best matches the search phrase AND transcript context?
2. STYLE CONSISTENCY (25%): If style info provided, match the source video's visual style
3. AVOIDS PAST ISSUES (20%): If feedback history provided, avoid patterns that were rejected before
4. SOURCE QUALITY (10%): Prefer web search results (google) for variety; stock sites (pexels, pixabay) are backup
5. RESOLUTION (5%): Higher resolution is better

OUTPUT:
Return ONLY a JSON object with the best index and brief reason:
{{"best_index": 0, "reason": "Best match for context"}}

Return ONLY the JSON object, nothing else."""
