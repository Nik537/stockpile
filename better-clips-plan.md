# Better Clips Plan: Achieving 90%+ Semantic Match

## Problem Statement

The current pipeline optimizes against a **compressed 2-6 word search phrase**, not the original transcript context. This causes clips to match keywords but miss semantic intent entirely.

**Example failure:**
```
Original transcript: "The coffee shop was packed with remote workers on laptops"
Search phrase:       "coffee shop interior"
Found clip:          Empty aesthetic coffee shop at night
AI Score:            9/10 (matches search phrase perfectly)
Actual relevance:    0% (missing: busy, morning, workers, laptops)
```

---

## The 6 Gaps in Current Pipeline

| Gap | Location | File | Problem |
|-----|----------|------|---------|
| 1. Phrase Compression | `plan_broll_needs` | `ai_service.py` | 2-6 words loses critical details |
| 2. YouTube Ranking | `search_videos` | `youtube_service.py` | Popularity over semantic match |
| 3. Pre-Filter | `filter` | `video_filter.py` | Filters by views/duration, not content |
| 4. Evaluation | `evaluate_videos` | `ai_service.py` | Compares to search phrase, not original |
| 5. Clip Extraction | `analyze_video` | `clip_extractor.py` | Analyzes for search phrase only |
| 6. No Verification | Missing | N/A | Never validates final clip matches intent |

---

## Fix 1: Expand Search Phrase Context in Evaluation

### Current Behavior
```python
# ai_service.py line ~806
SEARCH PHRASE: "coffee shop interior"
```

### Required Change
```python
# Pass full context to evaluation prompt
ORIGINAL CONTEXT: "The coffee shop was packed with remote workers on laptops during morning rush"
DERIVED SEARCH: "coffee shop interior"

REQUIREMENT: Video must match the CONTEXT (busy, morning, people working), not just "interior"
```

### Implementation

**File: `src/models/broll_need.py`**
```python
@dataclass
class BRollNeed:
    timestamp: str
    search_phrase: str
    description: str
    alternate_searches: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    visual_style: Optional[str] = None
    time_of_day: Optional[str] = None
    movement: Optional[str] = None

    # NEW: Add original context field
    original_context: str = ""  # Full transcript segment (100+ chars)
    required_elements: List[str] = field(default_factory=list)  # Must-have visual elements
```

**File: `src/services/ai_service.py` - `plan_broll_needs()`**
```python
# Modify the planning prompt to output additional fields
PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "needs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "search_phrase": {"type": "string"},
                    "description": {"type": "string"},
                    # NEW FIELDS
                    "original_context": {
                        "type": "string",
                        "description": "The exact transcript segment this B-roll supports (50-150 chars)"
                    },
                    "required_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Visual elements that MUST appear in the clip (e.g., 'people', 'morning light', 'laptops')"
                    },
                    # ... existing fields
                }
            }
        }
    }
}
```

**File: `src/services/ai_service.py` - `evaluate_videos()`**
```python
# Update evaluation prompt to use original context
EVALUATION_PROMPT = f"""
You are evaluating YouTube videos for B-roll relevance.

ORIGINAL CONTEXT (from transcript):
"{broll_need.original_context}"

DERIVED SEARCH PHRASE: "{broll_need.search_phrase}"

REQUIRED VISUAL ELEMENTS (must be present):
{chr(10).join(f'- {elem}' for elem in broll_need.required_elements)}

CRITICAL: Score based on match to ORIGINAL CONTEXT, not just the search phrase.
A video matching the search phrase but missing required elements should score ≤5.

Videos to evaluate:
{video_list}
"""
```

---

## Fix 2: Add Semantic Verification Stage

### Purpose
After clip extraction, verify the clip actually matches the original intent before accepting it.

### Implementation

**New file: `src/services/semantic_verifier.py`**
```python
"""Semantic verification for extracted clips."""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import google.generativeai as genai

from src.models.broll_need import BRollNeed
from src.models.clip import ClipSegment

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of semantic verification."""
    passed: bool
    similarity_score: float  # 0.0 to 1.0
    matched_elements: List[str]
    missing_elements: List[str]
    rationale: str


class SemanticVerifier:
    """Verifies clips match original transcript context."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.similarity_threshold = 0.9  # 90% match required

    async def verify_clip(
        self,
        clip_path: Path,
        broll_need: BRollNeed,
        clip_segment: ClipSegment
    ) -> VerificationResult:
        """
        Verify a clip matches the original context.

        Args:
            clip_path: Path to extracted clip file
            broll_need: Original B-roll requirement with context
            clip_segment: Clip metadata from extraction

        Returns:
            VerificationResult with pass/fail and details
        """
        # Upload clip for analysis
        video_file = genai.upload_file(str(clip_path))

        prompt = f"""
Analyze this video clip for semantic match to the original context.

ORIGINAL CONTEXT (what the clip should depict):
"{broll_need.original_context}"

REQUIRED ELEMENTS (must be visually present):
{chr(10).join(f'- {elem}' for elem in broll_need.required_elements)}

SEARCH PHRASE USED: "{broll_need.search_phrase}"

TASK:
1. List which required elements ARE visible in the clip
2. List which required elements are MISSING from the clip
3. Calculate a similarity score (0.0 to 1.0) based on:
   - 1.0 = All required elements present, matches context perfectly
   - 0.9 = Minor element missing but overall matches well
   - 0.7 = Some elements present but missing key aspects
   - 0.5 = Matches search phrase but not the original context
   - 0.3 = Loosely related at best
   - 0.0 = Completely unrelated

Respond in JSON:
{{
    "matched_elements": ["element1", "element2"],
    "missing_elements": ["element3"],
    "similarity_score": 0.85,
    "rationale": "Brief explanation of the score"
}}
"""

        response = await self.model.generate_content_async(
            [video_file, prompt],
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )

        result = json.loads(response.text)

        return VerificationResult(
            passed=result["similarity_score"] >= self.similarity_threshold,
            similarity_score=result["similarity_score"],
            matched_elements=result["matched_elements"],
            missing_elements=result["missing_elements"],
            rationale=result["rationale"]
        )

    async def verify_and_filter_clips(
        self,
        clips: List[Tuple[Path, ClipSegment]],
        broll_need: BRollNeed,
        min_clips: int = 1
    ) -> List[Tuple[Path, ClipSegment, VerificationResult]]:
        """
        Verify multiple clips and return only those passing threshold.

        If no clips pass, returns the best-scoring clip anyway (with warning).
        """
        results = []

        for clip_path, segment in clips:
            verification = await self.verify_clip(clip_path, broll_need, segment)
            results.append((clip_path, segment, verification))

            logger.info(
                f"Clip verification: {clip_path.name} - "
                f"Score: {verification.similarity_score:.0%} - "
                f"{'PASS' if verification.passed else 'FAIL'}"
            )

        # Filter to passing clips
        passing = [r for r in results if r[2].passed]

        if passing:
            return passing

        # No clips passed - return best scoring with warning
        results.sort(key=lambda x: x[2].similarity_score, reverse=True)
        best = results[0]

        logger.warning(
            f"No clips passed {self.similarity_threshold:.0%} threshold. "
            f"Best available: {best[0].name} at {best[2].similarity_score:.0%}"
        )

        if min_clips > 0:
            return [best]
        return []
```

### Integration into BRollProcessor

**File: `src/broll_processor.py`**
```python
from src.services.semantic_verifier import SemanticVerifier

class BRollProcessor:
    def __init__(self, ...):
        # ... existing init
        self.semantic_verifier = SemanticVerifier()

    async def _process_broll_need(self, broll_need: BRollNeed, ...):
        # ... existing clip extraction logic

        # NEW: Add verification stage after extraction
        if extracted_clips:
            verified_clips = await self.semantic_verifier.verify_and_filter_clips(
                clips=[(clip.path, clip) for clip in extracted_clips],
                broll_need=broll_need,
                min_clips=1
            )

            # Log verification results
            for path, segment, result in verified_clips:
                if not result.passed:
                    logger.warning(
                        f"Clip {path.name} below threshold: "
                        f"Missing elements: {result.missing_elements}"
                    )

            # Use only verified clips
            extracted_clips = [segment for _, segment, _ in verified_clips]
```

---

## Fix 3: Modify Scoring Criteria Priority

### Current Priority (Problematic)
```
1. Technical quality (50%)
2. Keyword match to search phrase (30%)
3. Semantic relevance (20%)
```

### New Priority (Semantic-First)
```
1. Semantic match to original context (50%)
2. Required elements present (20%)
3. Technical quality (20%)
4. Absence of unwanted elements (10%)
```

### Implementation

**File: `src/services/ai_service.py` - Update evaluation prompt**
```python
EVALUATION_CRITERIA = """
SCORING CRITERIA (in priority order):

1. SEMANTIC MATCH TO ORIGINAL CONTEXT (50% weight)
   - Does the video depict EXACTLY what was described in the transcript?
   - Score 10: Perfect match to the described scene/action
   - Score 7: Good match with minor differences
   - Score 4: Matches search phrase but misses context
   - Score 1: Unrelated to original meaning

2. REQUIRED ELEMENTS PRESENT (20% weight)
   - Are ALL the required visual elements visible?
   - Each missing element reduces score by 2 points

3. TECHNICAL QUALITY (20% weight)
   - Resolution, lighting, stability, composition
   - Prefer 1080p+, good lighting, smooth motion

4. ABSENCE OF UNWANTED ELEMENTS (10% weight)
   - No watermarks, text overlays, talking heads
   - No logos or branding

REJECTION CRITERIA (automatic score ≤3):
- Video shows opposite of what was described
- More than 2 required elements missing
- Contains explicit content/violence
- Heavily watermarked or branded

EXAMPLE:
Context: "busy morning coffee shop with remote workers"
Required elements: ["people", "laptops", "morning light", "coffee shop interior"]

- Empty night cafe: Score 3 (missing: busy, morning, workers)
- Crowded afternoon cafe: Score 6 (missing: morning, laptops)
- Morning cafe with laptop users: Score 9 (matches all elements)
"""
```

**File: `src/services/clip_extractor.py` - Update extraction prompt**
```python
CLIP_EXTRACTION_PROMPT = f"""
Analyze this video to find segments matching the original context.

ORIGINAL CONTEXT: "{broll_need.original_context}"
REQUIRED ELEMENTS: {broll_need.required_elements}

Find segments (4-15 seconds) where MOST required elements are visible.

SCORING (for each segment):
- 10: All required elements clearly visible
- 8: Most elements visible, minor gaps
- 6: Some elements visible
- 4: Matches search phrase only
- 2: Loosely related
- 0: Unrelated

ONLY return segments scoring 6 or higher.
Prefer fewer high-quality matches over many mediocre ones.
"""
```

---

## Fix 4: Store Original Context Throughout Pipeline

### Data Flow Change

```
Before:
  Transcript → plan_broll_needs → search_phrase (2-6 words) → [rest of pipeline]
                                        ↓
                              Original context LOST

After:
  Transcript → plan_broll_needs → BRollNeed {
                                    search_phrase: "coffee shop interior",
                                    original_context: "The coffee shop was packed with remote workers on laptops",
                                    required_elements: ["people", "laptops", "coffee shop", "busy"]
                                  }
                                        ↓
                              Original context PRESERVED through entire pipeline
```

### Implementation Checklist

- [ ] Update `BRollNeed` dataclass with `original_context` and `required_elements` fields
- [ ] Update `plan_broll_needs()` prompt to output these new fields
- [ ] Update `evaluate_videos()` to use `original_context` in scoring
- [ ] Update `ClipExtractor.analyze_video()` to use `original_context`
- [ ] Update `SemanticVerifier` (new) to validate against `original_context`
- [ ] Update logging to show context preservation through pipeline

---

## Fix 5: Add Negative Examples to Evaluation Prompts

### Purpose
Prevent AI from accepting clips that technically match keywords but miss the intent.

### Implementation

**File: `src/services/ai_service.py`**
```python
def _generate_negative_examples(self, broll_need: BRollNeed) -> str:
    """Generate examples of what NOT to match based on context."""

    prompt = f"""
Based on this context and required elements, generate examples of
videos that would INCORRECTLY match the search phrase but miss the intent.

CONTEXT: "{broll_need.original_context}"
SEARCH PHRASE: "{broll_need.search_phrase}"
REQUIRED ELEMENTS: {broll_need.required_elements}

Generate 3 specific examples of videos to REJECT:

Format:
- [description of wrong video]: Missing [what's missing]
"""

    # This would be called during evaluation
    return self._call_ai(prompt)


# Example output for "busy morning coffee shop with remote workers":
NEGATIVE_EXAMPLES = """
REJECT these types of videos:
- Empty coffee shop interior at night: Missing "busy", "morning", "workers"
- Coffee beans being roasted: Missing "shop interior", "workers", "laptops"
- Barista making coffee close-up: Missing "remote workers", "laptops", "busy atmosphere"
- Coffee shop exterior shot: Missing "interior", "workers", "laptops"
"""
```

**Integrate into evaluation prompt:**
```python
EVALUATION_PROMPT = f"""
{base_evaluation_prompt}

{negative_examples}

If a video matches any of the REJECT patterns above, score it ≤4 regardless of quality.
"""
```

---

## Fix 6: Add CLIP Embedding Verification (Optional Advanced)

### Purpose
Use CLIP model for quantitative semantic similarity scoring between text and video frames.

### Implementation

**New file: `src/services/clip_embeddings.py`**
```python
"""CLIP-based semantic similarity for clip verification."""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple


class CLIPVerifier:
    """Uses CLIP model for text-to-image semantic similarity."""

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract_keyframes(self, video_path: Path, num_frames: int = 5) -> List[Image.Image]:
        """Extract evenly-spaced keyframes from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def compute_similarity(
        self,
        video_path: Path,
        text_description: str,
        num_frames: int = 5
    ) -> Tuple[float, List[float]]:
        """
        Compute semantic similarity between video frames and text.

        Returns:
            Tuple of (average_similarity, per_frame_similarities)
        """
        frames = self.extract_keyframes(video_path, num_frames)

        if not frames:
            return 0.0, []

        # Process text and images
        inputs = self.processor(
            text=[text_description],
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Normalize embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = (image_embeds @ text_embeds.T).squeeze().cpu().numpy()

        if isinstance(similarities, np.floating):
            similarities = [float(similarities)]
        else:
            similarities = similarities.tolist()

        avg_similarity = sum(similarities) / len(similarities)

        return avg_similarity, similarities

    def verify_clip_matches_context(
        self,
        video_path: Path,
        original_context: str,
        threshold: float = 0.25  # CLIP scores are typically 0.15-0.35 for good matches
    ) -> Tuple[bool, float]:
        """
        Verify if a clip semantically matches the original context.

        Returns:
            Tuple of (passed, similarity_score)
        """
        avg_sim, _ = self.compute_similarity(video_path, original_context)
        return avg_sim >= threshold, avg_sim
```

### Integration

**File: `src/services/semantic_verifier.py`**
```python
from src.services.clip_embeddings import CLIPVerifier

class SemanticVerifier:
    def __init__(self, use_clip: bool = True):
        self.clip_verifier = CLIPVerifier() if use_clip else None

    async def verify_clip(self, clip_path, broll_need, clip_segment):
        # Existing Gemini-based verification
        gemini_result = await self._gemini_verify(clip_path, broll_need)

        # Optional CLIP-based verification
        if self.clip_verifier:
            clip_passed, clip_score = self.clip_verifier.verify_clip_matches_context(
                clip_path,
                broll_need.original_context
            )

            # Combine scores (Gemini weight: 0.7, CLIP weight: 0.3)
            combined_score = (gemini_result.similarity_score * 0.7) + (clip_score * 0.3)

            return VerificationResult(
                passed=combined_score >= self.similarity_threshold,
                similarity_score=combined_score,
                # ... rest of fields
            )

        return gemini_result
```

---

## Implementation Priority

### Phase 1: Critical Fixes (Immediate Impact)
1. **Fix 4**: Store original context throughout pipeline
2. **Fix 1**: Expand search phrase context in evaluation
3. **Fix 3**: Modify scoring criteria priority

### Phase 2: Verification Layer
4. **Fix 2**: Add semantic verification stage
5. **Fix 5**: Add negative examples to prompts

### Phase 3: Advanced (Optional)
6. **Fix 6**: CLIP embedding verification

---

## Configuration

**Add to `.env`:**
```bash
# Semantic matching settings
SEMANTIC_MATCH_THRESHOLD=0.9          # 90% match required
SEMANTIC_VERIFICATION_ENABLED=true    # Enable verification stage
USE_CLIP_VERIFICATION=false           # Enable CLIP embeddings (requires GPU)
REJECT_BELOW_THRESHOLD=true           # Reject clips below threshold vs warn
MIN_REQUIRED_ELEMENTS_MATCH=0.8       # 80% of required elements must be present
```

**Add to `src/utils/config.py`:**
```python
SEMANTIC_MATCH_THRESHOLD = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.9"))
SEMANTIC_VERIFICATION_ENABLED = os.getenv("SEMANTIC_VERIFICATION_ENABLED", "true").lower() == "true"
USE_CLIP_VERIFICATION = os.getenv("USE_CLIP_VERIFICATION", "false").lower() == "true"
REJECT_BELOW_THRESHOLD = os.getenv("REJECT_BELOW_THRESHOLD", "true").lower() == "true"
MIN_REQUIRED_ELEMENTS_MATCH = float(os.getenv("MIN_REQUIRED_ELEMENTS_MATCH", "0.8"))
```

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Semantic match rate | ~40% | 90%+ |
| Clips rejected for poor match | 0% | ~50% |
| False positive rate (high score, wrong content) | High | Low |
| Required elements present | Unchecked | Verified |
| Original context preserved | No | Yes |

---

## Testing Plan

### Unit Tests
```python
# tests/unit/test_semantic_verifier.py

def test_verification_passes_matching_clip():
    """Clip with all required elements should pass."""

def test_verification_fails_missing_elements():
    """Clip missing required elements should fail."""

def test_verification_uses_original_context():
    """Verification should compare to original context, not search phrase."""

def test_negative_examples_reduce_score():
    """Clips matching negative examples should score low."""
```

### Integration Tests
```python
# tests/integration/test_semantic_pipeline.py

def test_full_pipeline_preserves_context():
    """Original context should flow from planning to verification."""

def test_low_relevance_clips_rejected():
    """Clips below 90% threshold should be rejected."""

def test_search_phrase_vs_context_mismatch_caught():
    """Clips matching search phrase but not context should fail verification."""
```

---

## Summary

The core problem is **information loss** - compressing rich transcript context into 2-6 word search phrases and never recovering it. These fixes:

1. Preserve original context throughout the entire pipeline
2. Evaluate against context, not just search phrases
3. Prioritize semantic match over technical quality
4. Add verification stage to reject false positives
5. Use negative examples to prevent common mismatches
6. Optionally add quantitative embedding-based verification

Together, these changes should achieve the 90%+ semantic match target.
