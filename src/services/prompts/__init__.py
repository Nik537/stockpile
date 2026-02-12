"""Prompts module - centralized prompt templates for AI services.

This module extracts all hardcoded prompts from ai_service.py into a dedicated
location for better maintainability and version tracking.

Re-exports all prompt constants and utilities for easy importing:
    from services.prompts import PROMPT_VERSIONS, strip_markdown_code_blocks
    from services.prompts import BROLL_EXTRACTOR_V6, BROLL_PLANNER_V3
"""

from services.prompts._base import strip_markdown_code_blocks
from services.prompts.broll import BROLL_EXTRACTOR_V6, BROLL_PLANNER_V3
from services.prompts.evaluation import EVALUATOR_V4, BASIC_EVALUATOR
from services.prompts.images import (
    IMAGE_QUERY_GENERATOR,
    IMAGE_QUERY_CONTEXT_AWARE,
    IMAGE_SELECTOR,
)
from services.prompts.content_analysis import STYLE_ANALYZER, CONTEXT_QUESTION_GENERATOR
from services.prompts.bulk import BULK_IMAGE_PROMPT_GENERATOR
from services.prompts.script_generation import SCRIPT_GENERATOR_V1

# S2 IMPROVEMENT: Prompt version identifiers for cache invalidation
# IMPORTANT: Increment these when prompts change to invalidate stale cached responses
PROMPT_VERSIONS = {
    "extract_search_phrases": "v6",
    "generate_context_questions": "v1",
    "plan_broll_needs": "v3",  # B-roll planning with original_context and required_elements
    "evaluate_videos": "v4",  # Video evaluation with semantic context scoring + negative examples
    "generate_image_queries": "v1",  # Image query generation for parallel image acquisition
    "select_best_image": "v2",  # AI-powered image selection from candidates (v2: + ContentStyle)
    "detect_content_style": "v1",  # Feature 1: Content style/mood detection
    "generate_image_queries_with_context": "v1",  # Feature 2: Image queries with +-10s context window
    "generate_bulk_image_prompts": "v2",  # Bulk image prompt generation with diverse rendering styles
    "generate_script": "v1",  # Script generation from topic for video agent pipeline
}

__all__ = [
    # Utilities
    "strip_markdown_code_blocks",
    # Version tracking
    "PROMPT_VERSIONS",
    # B-Roll prompts
    "BROLL_EXTRACTOR_V6",
    "BROLL_PLANNER_V3",
    # Evaluation prompts
    "EVALUATOR_V4",
    "BASIC_EVALUATOR",
    # Image prompts
    "IMAGE_QUERY_GENERATOR",
    "IMAGE_QUERY_CONTEXT_AWARE",
    "IMAGE_SELECTOR",
    # Content analysis prompts
    "STYLE_ANALYZER",
    "CONTEXT_QUESTION_GENERATOR",
    # Bulk generation prompts
    "BULK_IMAGE_PROMPT_GENERATOR",
    # Script generation prompts
    "SCRIPT_GENERATOR_V1",
]
