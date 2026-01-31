"""Style and mood detection models for content matching."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class VisualStyle(Enum):
    """Visual style categories for content."""
    CINEMATIC = "cinematic"  # Dramatic, high production value
    DOCUMENTARY = "documentary"  # Factual, observational
    RAW = "raw"  # Unpolished, authentic, handheld
    PROFESSIONAL = "professional"  # Clean, corporate, polished
    ENERGETIC = "energetic"  # Fast-paced, dynamic, bright
    MOODY = "moody"  # Dark, atmospheric, dramatic lighting
    VINTAGE = "vintage"  # Retro, film grain, muted colors
    MODERN = "modern"  # Contemporary, sleek, minimal


class ColorTone(Enum):
    """Color tone categories."""
    WARM = "warm"  # Orange, yellow, red tones
    COOL = "cool"  # Blue, green, purple tones
    NEUTRAL = "neutral"  # Balanced, natural colors
    DESATURATED = "desaturated"  # Muted, low saturation
    VIBRANT = "vibrant"  # High saturation, punchy colors
    HIGH_CONTRAST = "high_contrast"  # Strong blacks and whites
    LOW_CONTRAST = "low_contrast"  # Flat, even lighting


class PacingStyle(Enum):
    """Content pacing categories."""
    FAST = "fast"  # Quick cuts, high energy
    MODERATE = "moderate"  # Standard pacing
    SLOW = "slow"  # Deliberate, contemplative
    MIXED = "mixed"  # Varies throughout


@dataclass
class ContentStyle:
    """Detected style and mood of source content.

    Used to match B-roll images and video clips to the source video's
    visual language and tone.

    Feature 1: Style/Mood Detection + Content Analysis
    Analyzes the source video to understand:
    - What it's about (topic, subject matter, key themes)
    - Who it's for (target audience, skill level, demographics)
    - Visual style (cinematic, documentary, raw, etc.)
    - Mood/tone (energetic, serious, educational, entertaining)
    """

    # Primary visual style
    visual_style: VisualStyle = VisualStyle.PROFESSIONAL

    # Color characteristics
    color_tone: ColorTone = ColorTone.NEUTRAL

    # Pacing/energy
    pacing: PacingStyle = PacingStyle.MODERATE

    # Content type hints
    is_talking_head: bool = False
    is_tutorial: bool = False
    is_entertainment: bool = False
    is_corporate: bool = False

    # Mood descriptors (for AI prompts)
    mood_keywords: List[str] = field(default_factory=list)

    # Things to avoid based on style
    avoid_keywords: List[str] = field(default_factory=list)

    # Confidence score (0-1)
    confidence: float = 0.8

    # NEW: Content analysis fields (Feature 1)
    topic: str = ""
    """Main subject of the video (e.g., 'MMA strength and conditioning')"""

    topic_keywords: List[str] = field(default_factory=list)
    """Key topic keywords for search enhancement (e.g., ['MMA', 'BJJ', 'wrestling', 'gym'])"""

    target_audience: str = ""
    """Description of target viewer (e.g., 'competitive fighters and serious martial artists')"""

    audience_level: str = "general"
    """Audience skill level: 'beginner', 'intermediate', 'advanced', 'professional', 'general'"""

    audience_demographics: str = ""
    """Demographic hints (e.g., '18-35 male athletes')"""

    content_type: str = ""
    """Content category: 'educational', 'motivational', 'tutorial', 'vlog', 'documentary', etc."""

    tone: str = ""
    """Overall tone: 'serious', 'casual', 'inspirational', 'technical', 'entertaining', etc."""

    preferred_imagery: List[str] = field(default_factory=list)
    """Types of imagery that match this content
    (e.g., ['professional athletes', 'high-intensity training', 'competition footage'])"""

    avoid_imagery: List[str] = field(default_factory=list)
    """Types of imagery to avoid for this content
    (e.g., ['casual gym-goers', 'beginner exercises', 'stock fitness models'])"""

    def to_prompt_context(self) -> str:
        """Generate prompt context for AI selection.

        Returns a formatted string suitable for injection into AI prompts
        for image/video selection.
        """
        parts = []

        # Content analysis section (Feature 1)
        if self.topic:
            parts.append(f"TOPIC: {self.topic}")

        if self.topic_keywords:
            parts.append(f"Topic keywords: {', '.join(self.topic_keywords)}")

        if self.target_audience:
            parts.append(f"TARGET AUDIENCE: {self.target_audience}")

        if self.audience_level and self.audience_level != "general":
            parts.append(f"Audience level: {self.audience_level}")

        if self.content_type:
            parts.append(f"Content type: {self.content_type}")

        if self.tone:
            parts.append(f"Tone: {self.tone}")

        # Visual style section
        parts.append(f"Visual style: {self.visual_style.value}")
        parts.append(f"Color tone: {self.color_tone.value}")
        parts.append(f"Pacing: {self.pacing.value}")

        if self.mood_keywords:
            parts.append(f"Mood: {', '.join(self.mood_keywords)}")

        # Imagery guidance (Feature 1)
        if self.preferred_imagery:
            parts.append(f"PREFERRED IMAGERY: {', '.join(self.preferred_imagery)}")

        if self.avoid_imagery:
            parts.append(f"AVOID IMAGERY: {', '.join(self.avoid_imagery)}")

        if self.avoid_keywords:
            parts.append(f"Avoid keywords: {', '.join(self.avoid_keywords)}")

        # Legacy content type flags
        content_types = []
        if self.is_talking_head:
            content_types.append("talking head")
        if self.is_tutorial:
            content_types.append("tutorial/educational")
        if self.is_entertainment:
            content_types.append("entertainment")
        if self.is_corporate:
            content_types.append("corporate/business")

        if content_types:
            parts.append(f"Format hints: {', '.join(content_types)}")

        return "\n".join(parts)

    def get_search_query_modifier(self, base_query: str) -> str:
        """Enhance a search query with style-appropriate terms.

        Args:
            base_query: The original search query

        Returns:
            Modified search query with style context
        """
        modifiers = self.get_search_modifiers()

        # Add topic-specific keywords if relevant
        if self.topic_keywords:
            # Add up to 2 topic keywords that don't duplicate base query
            base_lower = base_query.lower()
            topic_additions = [
                kw for kw in self.topic_keywords[:3]
                if kw.lower() not in base_lower
            ][:2]
            modifiers.extend(topic_additions)

        if not modifiers:
            return base_query

        # Combine base query with up to 2 modifiers
        selected_modifiers = modifiers[:2]
        return f"{base_query} {' '.join(selected_modifiers)}"

    def get_search_modifiers(self) -> List[str]:
        """Get search term modifiers based on style."""
        modifiers = []

        if self.visual_style == VisualStyle.CINEMATIC:
            modifiers.extend(["cinematic", "dramatic", "film"])
        elif self.visual_style == VisualStyle.DOCUMENTARY:
            modifiers.extend(["documentary", "real", "authentic"])
        elif self.visual_style == VisualStyle.RAW:
            modifiers.extend(["raw", "candid", "unposed"])
        elif self.visual_style == VisualStyle.PROFESSIONAL:
            modifiers.extend(["professional", "clean", "business"])
        elif self.visual_style == VisualStyle.ENERGETIC:
            modifiers.extend(["dynamic", "action", "vibrant"])
        elif self.visual_style == VisualStyle.MOODY:
            modifiers.extend(["moody", "dark", "atmospheric"])

        return modifiers


@dataclass
class TranscriptContext:
    """Extended transcript context for better search relevance.

    Instead of just a single phrase, includes surrounding context
    for more accurate image/video matching.
    """

    # Core search phrase
    search_phrase: str

    # Timestamp in source video
    timestamp: float

    # Extended context (±10 seconds of transcript)
    context_before: str = ""  # Text from previous ~10 seconds
    context_after: str = ""   # Text from next ~10 seconds

    # Full context window
    full_context: str = ""    # Combined context

    # Key themes/topics extracted from context
    themes: List[str] = field(default_factory=list)

    # Entities mentioned (people, places, things)
    entities: List[str] = field(default_factory=list)

    # Emotional tone at this point
    emotional_tone: Optional[str] = None

    def get_enhanced_search_phrase(self) -> str:
        """Get search phrase enhanced with context."""
        if self.themes:
            return f"{self.search_phrase} {' '.join(self.themes[:2])}"
        return self.search_phrase

    def to_prompt_context(self) -> str:
        """Generate context for AI prompts."""
        parts = [f'Search phrase: "{self.search_phrase}"']

        if self.full_context:
            parts.append(f'Transcript context: "{self.full_context}"')

        if self.themes:
            parts.append(f"Key themes: {', '.join(self.themes)}")

        if self.emotional_tone:
            parts.append(f"Tone: {self.emotional_tone}")

        return "\n".join(parts)
