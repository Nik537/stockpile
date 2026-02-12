"""Adapter layer bridging SceneScript models to BRollNeed models.

Converts the video agent's scene-level data into the B-roll processor's
planning format, enabling the existing B-roll acquisition pipeline to
serve the video production agent.
"""

from __future__ import annotations

import re
from typing import List

from models.broll_need import BRollNeed, BRollPlan
from video_agent.models import SceneScript, Script, VisualType

# Default negative keywords to exclude low-quality results
DEFAULT_NEGATIVE_KEYWORDS = [
    "compilation",
    "reaction",
    "vlog",
    "meme",
    "review",
]


class BRollAdapter:
    """Converts video agent SceneScript/Script objects into BRollNeed/BRollPlan objects."""

    @staticmethod
    def scene_to_broll_need(scene: SceneScript, script: Script) -> BRollNeed:
        """Convert a single SceneScript to a BRollNeed.

        Maps scene-level data (voiceover, visual keywords, style) into the
        BRollNeed format expected by the B-roll processor pipeline.

        Args:
            scene: The scene to convert.
            script: The full script (used for timestamp calculation and context).

        Returns:
            A fully populated BRollNeed instance.
        """
        # Calculate timestamp from scene position in script
        timestamp = BRollAdapter._calculate_timestamp(scene, script)

        # Build primary search phrase from visual keywords
        search_phrase = " ".join(scene.visual_keywords) if scene.visual_keywords else ""

        # Generate alternate search variations
        alternate_searches = BRollAdapter._generate_alternate_searches(scene.visual_keywords)

        # Extract required elements from voiceover text
        required_elements = BRollAdapter._extract_required_elements(scene.voiceover)

        # Determine suggested duration from scene estimate
        min_dur, max_dur = BRollAdapter.get_clip_duration_range(scene)
        suggested_duration = (min_dur + max_dur) / 2.0

        return BRollNeed(
            timestamp=timestamp,
            search_phrase=search_phrase,
            description=scene.voiceover[:200] if scene.voiceover else "",
            context=scene.voiceover,
            suggested_duration=suggested_duration,
            original_context=scene.voiceover,
            required_elements=required_elements,
            alternate_searches=alternate_searches,
            negative_keywords=list(DEFAULT_NEGATIVE_KEYWORDS),
            visual_style=scene.visual_style if scene.visual_style else None,
        )

    @staticmethod
    def scenes_to_broll_plan(script: Script) -> BRollPlan:
        """Convert a full Script into a BRollPlan.

        Only converts scenes with VisualType.BROLL_VIDEO; other visual types
        (generated images, text graphics) are skipped.

        Args:
            script: The complete video script.

        Returns:
            A BRollPlan containing BRollNeed objects for all B-roll scenes.
        """
        needs: List[BRollNeed] = []
        total_duration = 0.0

        for scene in script.scenes:
            total_duration += scene.duration_est

        for scene in script.scenes:
            if scene.visual_type != VisualType.BROLL_VIDEO:
                continue
            need = BRollAdapter.scene_to_broll_need(scene, script)
            needs.append(need)

        # Include hook duration in total
        hook_duration = 10.0  # Default hook duration estimate
        total_duration += hook_duration

        return BRollPlan(
            source_duration=total_duration,
            needs=needs,
            clips_per_minute=len(needs) / max(total_duration / 60.0, 1.0),
        )

    @staticmethod
    def get_clip_duration_range(scene: SceneScript) -> tuple[float, float]:
        """Determine the clip duration range for a scene.

        If the scene has explicit min/max_clip_duration set, those are used.
        Otherwise, the range is inferred from the scene's duration estimate.

        Args:
            scene: The scene to determine duration range for.

        Returns:
            Tuple of (min_duration, max_duration) in seconds.
        """
        # Prefer explicit overrides if set
        if scene.min_clip_duration is not None and scene.max_clip_duration is not None:
            return (scene.min_clip_duration, scene.max_clip_duration)

        # Infer from duration estimate
        if scene.duration_est <= 3:
            return (2.0, 4.0)
        elif scene.duration_est <= 6:
            return (3.0, 6.0)
        else:
            return (4.0, 8.0)

    @staticmethod
    def _calculate_timestamp(scene: SceneScript, script: Script) -> float:
        """Calculate the timestamp offset of a scene within the full script.

        Sums durations of all preceding scenes plus the hook to determine
        when this scene starts in the overall timeline.

        Args:
            scene: The target scene.
            script: The full script containing all scenes.

        Returns:
            Timestamp in seconds from the start of the video.
        """
        # Start after hook (estimate 10 seconds for hook)
        offset = 10.0

        for s in script.scenes:
            if s.id == scene.id:
                break
            offset += s.duration_est

        return offset

    @staticmethod
    def _generate_alternate_searches(keywords: List[str]) -> List[str]:
        """Generate 2-3 alternate search phrase variations from keywords.

        Creates variations by using subsets and reorderings of the original
        keywords to improve search coverage.

        Args:
            keywords: The visual keywords from the scene.

        Returns:
            List of 2-3 alternate search phrases.
        """
        if not keywords:
            return []

        alternates: List[str] = []

        # Variation 1: keywords in reverse order
        if len(keywords) >= 2:
            alternates.append(" ".join(reversed(keywords)))

        # Variation 2: first keyword + "footage" suffix
        alternates.append(f"{keywords[0]} footage")

        # Variation 3: first two keywords with "cinematic" prefix
        if len(keywords) >= 2:
            alternates.append(f"cinematic {keywords[0]} {keywords[1]}")

        return alternates[:3]

    @staticmethod
    def _extract_required_elements(voiceover: str) -> List[str]:
        """Extract concrete visual elements from voiceover text.

        Identifies nouns and noun phrases that represent visual elements
        the B-roll clip should contain.

        Args:
            voiceover: The narration text for the scene.

        Returns:
            List of required visual element strings.
        """
        if not voiceover:
            return []

        # Extract meaningful noun-like words (4+ chars, not common stopwords)
        stopwords = {
            "that", "this", "with", "from", "have", "been", "were", "will",
            "would", "could", "should", "about", "their", "there", "these",
            "those", "they", "your", "what", "when", "where", "which", "while",
            "more", "most", "some", "such", "than", "them", "then", "into",
            "also", "just", "very", "really", "here", "even", "only", "like",
            "over", "many", "much", "each", "every", "other", "being",
        }

        words = re.findall(r"\b[a-zA-Z]{4,}\b", voiceover.lower())
        elements = []
        seen = set()

        for word in words:
            if word not in stopwords and word not in seen:
                seen.add(word)
                elements.append(word)

            if len(elements) >= 5:
                break

        return elements
