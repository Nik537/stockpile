"""Base utilities for prompts module.

Contains shared helper functions used across prompt modules.
"""


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
