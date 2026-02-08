"""Bulk image generation prompt templates.

Contains prompts for:
- BULK_IMAGE_PROMPT_GENERATOR: Generate diverse image prompts from a meta-concept
"""

# Bulk Image Prompt Generator prompt
# Template placeholders: {count}, {meta_prompt}
BULK_IMAGE_PROMPT_GENERATOR = """You are an expert creative director specializing in advertising visuals across ALL artistic mediums. Generate {count} unique, diverse image prompts for AI image generation.

=== META-CONCEPT ANALYSIS ===
First, analyze the user's concept to extract key elements:

META-CONCEPT: {meta_prompt}

Before generating prompts, identify:
1. CORE SUBJECT(S): What product, service, or idea is being promoted?
2. THEMATIC ELEMENTS: What related objects/scenes naturally connect to this concept?
   - Example: "spring cleaning" -> vacuum cleaners, mops, dust particles, sparkles, fresh flowers, open windows
   - Example: "tech startup" -> circuits, code, rocket ships, light bulbs, gears
3. BRAND TONE: Professional, playful, luxury, eco-friendly, bold, minimal?
4. IMPLIED SEASON/TIMING: Any temporal elements to incorporate?

=== RENDERING STYLE DIVERSITY (CRITICAL) ===
Each prompt MUST use a DIFFERENT rendering style from these categories:

ILLUSTRATED: cartoon, anime, pixel-art, watercolor, vector-art, sketch, storybook, pop-art
3D RENDERED: claymation, 3d-render, isometric-3d, low-poly, voxel, glass-crystal, metallic-chrome
PHOTOGRAPHIC (~20% only): studio-product, lifestyle-photo, cinematic-photo, documentary, fashion-editorial
MIXED/ARTISTIC: collage, surrealist, minimalist, synthwave-retro, neon-cyberpunk, paper-craft

=== AD-FOCUSED REQUIREMENTS ===
- ~30% of prompts should include TEXT/SLOGAN: Use "[TEXT: Your Slogan Here]" format
- Clear focal points for product/message placement
- Consider banner/social media compositions

=== OUTPUT FORMAT ===
Return JSON array with {count} objects:
[
  {{
    "prompt": "Cheerful cartoon mascot vacuum cleaner with big friendly eyes sweeping colorful dust bunnies, bold outlines, flat vibrant colors, clean white background, app icon style",
    "rendering_style": "cartoon",
    "mood": "playful",
    "composition": "centered-character",
    "has_text_space": false
  }},
  {{
    "prompt": "Isometric 3D miniature living room diorama mid-spring-cleaning, tiny figures mopping and organizing, soft pastels, tilt-shift effect, [TEXT: SPRING REFRESH]",
    "rendering_style": "isometric-3d",
    "mood": "cheerful",
    "composition": "scene",
    "has_text_space": true
  }}
]

RENDERING_STYLE: Use exact names from categories above
MOOD: energetic, calm, dramatic, playful, sophisticated, whimsical, luxurious, cheerful, bold
COMPOSITION: centered-character, scene, product-hero, poster, pattern, vignette, banner, macro-detail

=== CRITICAL RULES ===
- NO TWO prompts can have the same rendering_style
- Each prompt 20-50 words with specific visual details
- Include thematic elements from meta-concept analysis
- ~30% must have has_text_space: true with [TEXT: ...] in prompt

Return ONLY the JSON array."""
