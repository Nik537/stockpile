"""Script generator for the autonomous video production pipeline.

Uses Gemini AI to generate structured video scripts from a topic,
producing a Script dataclass with hook, scenes, and metadata.
"""

import json
import logging

from services.prompts import strip_markdown_code_blocks
from services.prompts.script_generation import SCRIPT_GENERATOR_V1
from utils.retry import APIRateLimitError, NetworkError, retry_api_call
from video_agent.models import HookScript, SceneScript, Script, VisualType

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """Generates structured video scripts from topics using Gemini AI.

    Takes an existing AIService instance for Gemini API access.
    """

    def __init__(self, ai_service):
        """Initialize with an AIService instance.

        Args:
            ai_service: Configured AIService with Gemini client
        """
        self.ai = ai_service

    @retry_api_call(max_retries=3, base_delay=2.0)
    def generate(
        self,
        topic: str,
        style: str = "documentary",
        target_duration_minutes: int = 8,
    ) -> Script:
        """Generate a structured video script from a topic using Gemini.

        Args:
            topic: The video topic/subject to script
            style: Visual style (documentary, cinematic, educational, etc.)
            target_duration_minutes: Target video length in minutes

        Returns:
            Script dataclass with hook, scenes, and metadata
        """
        # Calculate target scene count: ~10-15 scenes per minute for fast pacing
        target_scenes = max(10, int(target_duration_minutes * 12))

        logger.info(
            f"Generating script: topic='{topic[:60]}', style={style}, "
            f"duration={target_duration_minutes}min, target_scenes={target_scenes}"
        )

        prompt = SCRIPT_GENERATOR_V1.format(
            topic=topic,
            style=style,
            target_duration_minutes=target_duration_minutes,
            target_scenes=target_scenes,
        )

        try:
            from google.genai import types

            response = self.ai.client.models.generate_content(
                model=self.ai.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                logger.error("AI response is empty for script generation")
                raise ValueError("Empty AI response for script generation")

            response_text = strip_markdown_code_blocks(response.text)

            try:
                script_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for script generation: {e}")
                logger.debug(f"Raw response: {response.text[:500]}")
                raise ValueError(f"Failed to parse script JSON: {e}")

            script = self._parse_script(script_data)

            total_duration = sum(s.duration_est for s in script.scenes)
            logger.info(
                f"Script generated: '{script.title}' — "
                f"{len(script.scenes)} scenes, ~{total_duration:.0f}s total"
            )

            return script

        except (ValueError, json.JSONDecodeError):
            raise
        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            if "rate limit" in str(e).lower():
                raise APIRateLimitError(f"Rate limit hit: {e}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NetworkError(f"Network error: {e}")
            raise

    def _parse_script(self, data: dict) -> Script:
        """Parse a Script dataclass from Gemini JSON response.

        Args:
            data: Dictionary from JSON response

        Returns:
            Populated Script object

        Raises:
            ValueError: If required fields are missing
        """
        title = str(data.get("title", "")).strip()
        if not title:
            raise ValueError("Script response missing 'title'")

        # Parse hook
        hook_data = data.get("hook")
        if not hook_data or not isinstance(hook_data, dict):
            raise ValueError("Script response missing 'hook'")

        hook = HookScript(
            voiceover=str(hook_data.get("voiceover", "")).strip(),
            visual_description=str(hook_data.get("visual_description", "")).strip(),
            visual_keywords=self._parse_string_list(hook_data.get("visual_keywords", [])),
            sound_effect=str(hook_data.get("sound_effect", "none")).strip(),
        )

        if not hook.voiceover:
            raise ValueError("Hook is missing voiceover text")

        # Parse scenes
        scenes_data = data.get("scenes")
        if not scenes_data or not isinstance(scenes_data, list):
            raise ValueError("Script response missing 'scenes' list")

        scenes = []
        for i, scene_data in enumerate(scenes_data):
            if not isinstance(scene_data, dict):
                logger.warning(f"Skipping non-dict scene at index {i}")
                continue

            try:
                scene = self._parse_scene(scene_data, fallback_id=i + 1)
                scenes.append(scene)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid scene {i}: {e}")
                continue

        if not scenes:
            raise ValueError("No valid scenes parsed from script response")

        # Parse metadata
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return Script(
            title=title,
            hook=hook,
            scenes=scenes,
            metadata=metadata,
        )

    def _parse_scene(self, data: dict, fallback_id: int = 0) -> SceneScript:
        """Parse a single SceneScript from JSON data.

        Args:
            data: Dictionary with scene fields
            fallback_id: ID to use if not provided in data

        Returns:
            Populated SceneScript object
        """
        # Parse visual_type enum
        visual_type_str = str(data.get("visual_type", "broll_video")).strip().lower()
        try:
            visual_type = VisualType(visual_type_str)
        except ValueError:
            visual_type = VisualType.BROLL_VIDEO

        voiceover = str(data.get("voiceover", "")).strip()
        if not voiceover:
            raise ValueError(f"Scene {fallback_id} is missing voiceover text")

        return SceneScript(
            id=int(data.get("id", fallback_id)),
            duration_est=float(data.get("duration_est", 5.0)),
            voiceover=voiceover,
            visual_keywords=self._parse_string_list(data.get("visual_keywords", [])),
            visual_style=str(data.get("visual_style", "cinematic")).strip(),
            visual_type=visual_type,
            transition_in=str(data.get("transition_in", "cut")).strip(),
            music_mood=str(data.get("music_mood", "neutral")).strip(),
            sound_effect=str(data.get("sound_effect", "none")).strip(),
        )

    def _parse_string_list(self, raw: list) -> list[str]:
        """Parse and clean a list of strings.

        Args:
            raw: Raw list that may contain non-string items

        Returns:
            Cleaned list of non-empty strings
        """
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if item and str(item).strip()]
