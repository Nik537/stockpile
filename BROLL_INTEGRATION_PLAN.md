# Integrating the B-Roll Processor into the Video Agent

## The Problem

The video agent's B-roll acquisition is noticeably worse than the standalone B-roll processor. The agent downloads ONE video per scene, uses simpler evaluation prompts, and skips the competitive analysis that makes the processor's output so much better. This document lays out exactly how to bring the processor's quality into the video agent.

---

## Gap Analysis: What the Processor Does That the Agent Doesn't

| Capability | B-Roll Processor | Video Agent | Impact |
|-----------|-----------------|-------------|--------|
| **Competitive analysis** | Downloads 2+ previews per need, picks the BEST clip across all | Downloads 1 video per scene, first success wins | **Biggest quality gap** |
| **Context-aware evaluation** | `EVALUATOR_V4` with 5 weighted criteria + original_context + required_elements + feedback history | `VIDEO_AGENT_BROLL_EVALUATOR` with 4 simpler criteria | Worse video selection |
| **Semantic verification** | Post-download verification that clips match transcript meaning | None | Bad clips slip through |
| **Style detection** | Detects visual style, color tone, pacing from transcript | Script provides `visual_style` per scene but no global detection | Less cohesive look |
| **Multi-clip extraction** | Extracts up to 3 clips per video, picks best | Extracts 1 segment per video | Fewer options |
| **Rich planning metadata** | `BRollNeed` with original_context, required_elements, alternate_searches, negative_keywords, visual_style, time_of_day, movement | `SceneScript` with visual_keywords + visual_style | Less search precision |
| **Feedback loop** | Learns from user rejections via `FeedbackService` | None | Repeats mistakes |
| **Caching** | Content-hash cache on all AI responses | None | Higher cost, no replay |
| **Pre-filtering** | `VideoPreFilter` blocks compilations/reactions/vlogs by metadata | `_prefilter_video_results()` has similar but lighter filtering | More junk downloads |
| **Checkpoint/resume** | Resumes from failure midway | No resume; full restart on failure | Wasted work |
| **Clip duration range** | 4-15 seconds | 3-5 seconds (hardcoded) | Agent clips too short for some scenes |

---

## Architecture: How to Wire It In

### Current Video Agent Flow (B-Roll Path)

```
SceneScript.visual_keywords
  -> _search_and_download_broll()
      -> _youtube_multi_query_search() / stock source search
      -> _prefilter_video_results()
      -> _evaluate_broll_candidates()  [VIDEO_AGENT_BROLL_EVALUATOR]
      -> _download_youtube_broll()     [single video, 360p preview -> Gemini -> FFmpeg]
  -> Director decides video vs image
  -> Timeline assembly
```

### Proposed Flow (Using B-Roll Processor Services)

```
SceneScript  ->  BRollNeed adapter  ->  VideoAcquisitionService
                                            |
                 +--------------------------+---------------------------+
                 |                          |                           |
          VideoSearchService       ClipExtractor              SemanticVerifier
          (multi-source +          (competitive analysis +    (post-download
           EVALUATOR_V4)            multi-clip extraction)     context match)
                 |                          |
          VideoPreFilter           VideoDownloader
          (metadata filter)        (two-pass + retry)
```

### Key Principle: Compose, Don't Rewrite

The B-roll processor's pipeline is already decomposed into injectable services:
- `VideoAcquisitionService` - orchestrates per-need processing
- `VideoSearchService` - multi-source search + AI evaluation
- `ClipExtractor` - Gemini analysis + FFmpeg
- `VideoDownloader` - yt-dlp wrapper with two-pass
- `VideoPreFilter` - metadata filtering
- `SemanticVerifier` - post-extraction validation

**The video agent should import and use these services directly**, not duplicate their logic.

---

## Implementation Plan

### Phase 1: Adapter Layer (BRollNeed Bridge)

The processor operates on `BRollNeed` objects. The video agent operates on `SceneScript` objects. Build an adapter that converts between them.

**New file: `src/video_agent/broll_adapter.py`**

```python
class BRollAdapter:
    """Converts video agent scene data into B-roll processor inputs."""

    @staticmethod
    def scene_to_broll_need(scene: SceneScript, script: Script) -> BRollNeed:
        """Convert a SceneScript to a BRollNeed for processor consumption."""
        return BRollNeed(
            timestamp=_calculate_timestamp(scene, script),
            search_phrase=scene.visual_keywords,
            description=scene.voiceover,
            context=scene.voiceover,
            original_context=scene.voiceover,  # KEY: preserves semantic meaning
            required_elements=_extract_required_elements(scene),
            alternate_searches=_generate_alternates(scene),
            negative_keywords=["compilation", "reaction", "vlog", "meme"],
            visual_style=scene.visual_style or "cinematic",
            suggested_duration=scene.duration_est,
        )

    @staticmethod
    def scenes_to_broll_plan(script: Script) -> BRollPlan:
        """Convert full script to a BRollPlan."""
        needs = [
            BRollAdapter.scene_to_broll_need(scene, script)
            for scene in script.scenes
            if scene.visual_type == VisualType.BROLL_VIDEO
        ]
        return BRollPlan(needs=needs, source_duration=script.metadata.get("target_duration", 60))
```

**Why this matters:** The `original_context` field is the #1 quality driver in the processor's evaluation prompts. By mapping `scene.voiceover` to `original_context`, every downstream AI call (evaluation, clip extraction, semantic verification) gets the full narrative context instead of just keywords.

### Phase 2: Replace `_search_and_download_broll()` with `VideoAcquisitionService`

This is the core swap. The video agent's `_search_and_download_broll()` (lines 904-1009) gets replaced with a call to the processor's `VideoAcquisitionService.process_single_need()`.

**Changes to `src/video_agent/agent.py`:**

1. **Add new service to `__init__()`:**
```python
from src.services.video_acquisition_service import VideoAcquisitionService
from src.services.video_search_service import VideoSearchService
from src.services.clip_extractor import ClipExtractor
from src.services.video_filter import VideoPreFilter
from src.services.semantic_verifier import SemanticVerifier
from src.video_agent.broll_adapter import BRollAdapter

class VideoProductionAgent:
    def __init__(self, config):
        # ... existing init ...

        # B-Roll processor services (replaces internal b-roll logic)
        self.clip_extractor = ClipExtractor(config)
        self.video_pre_filter = VideoPreFilter(config)
        self.semantic_verifier = SemanticVerifier(config)
        self.video_search_service = VideoSearchService(
            config=config,
            video_sources=self.video_sources,  # reuse existing sources
            ai_service=self._get_or_create_ai_service(),
            pre_filter=self.video_pre_filter,
        )
        self.video_acquisition = VideoAcquisitionService(
            config=config,
            search_service=self.video_search_service,
            downloader=self.downloader,  # reuse existing
            clip_extractor=self.clip_extractor,
            file_organizer=None,  # video agent manages its own files
            semantic_verifier=self.semantic_verifier,
        )
```

2. **Replace `_search_and_download_broll()` body:**
```python
async def _search_and_download_broll(self, scene: SceneScript, output_dir: Path) -> Optional[Path]:
    """Acquire B-roll using the full processor pipeline."""
    broll_need = BRollAdapter.scene_to_broll_need(scene, self.current_script)

    clips = await self.video_acquisition.process_single_need(
        need=broll_need,
        output_dir=output_dir,
        content_style=self.detected_style,  # from Phase 3
    )

    if clips:
        return clips[0]  # Return best clip path
    return None
```

3. **Remove now-unused internal methods:**
   - `_youtube_multi_query_search()` - replaced by `VideoSearchService.search_with_fallback()`
   - `_evaluate_broll_candidates()` - replaced by `VideoSearchService.evaluate_videos_enhanced()`
   - `_download_youtube_broll()` - replaced by `VideoAcquisitionService` two-pass logic
   - `_analyze_video_for_best_segment()` - replaced by `ClipExtractor.analyze_video()`
   - `_prefilter_video_results()` - replaced by `VideoPreFilter`

### Phase 3: Add Style Detection

The processor detects content style from the transcript to influence all downstream decisions. The video agent should do the same from the script.

**Add to `VideoProductionAgent.produce()` after script generation:**

```python
# After script generation, detect style for B-roll matching
self.detected_style = await self.ai_service.detect_content_style(
    transcript_text="\n".join(scene.voiceover for scene in script.scenes),
    language="en",
)
```

This `ContentStyle` object (with `visual_style`, `color_tone`, `pacing`, mood) then flows into every evaluation call, ensuring visual coherence across all B-roll clips.

### Phase 4: Add Competitive Analysis

This is the single highest-impact change. Instead of downloading 1 video per scene, download 2+ previews and pick the winner.

**Configuration additions to video agent config:**

```python
# In video agent config / VideoProduceRequest schema
competitive_analysis_enabled: bool = True
previews_per_need: int = 2          # Compare 2 candidates per scene
clips_per_need_target: int = 1       # Pick the 1 best
```

**No code changes needed** - `VideoAcquisitionService.process_single_need()` already handles competitive analysis when these config flags are set. The adapter layer from Phase 1 passes the config through.

### Phase 5: Add Semantic Verification

Post-download check that clips actually match the scene's narrative intent.

```python
# After clip download in _search_and_download_broll()
if self.semantic_verifier and clip_path:
    is_valid = await self.semantic_verifier.verify(
        clip_path=clip_path,
        original_context=scene.voiceover,
        required_elements=broll_need.required_elements,
    )
    if not is_valid:
        logger.warning(f"Scene {scene.id}: clip failed semantic verification, retrying...")
        # Try next best candidate
```

### Phase 6: Upgrade Evaluation Prompts

Replace `VIDEO_AGENT_BROLL_EVALUATOR` with `EVALUATOR_V4` from the processor.

**Changes to `src/services/prompts/evaluation.py`:**

Either:
- **Option A (clean):** Import and use `EVALUATOR_V4` directly from `src/services/ai_service.py` prompt constants
- **Option B (gradual):** Copy the key improvements into the video agent's evaluator:
  - Add `original_context` as PRIMARY scoring criterion (50% weight)
  - Add `required_elements` checking
  - Add `negative_examples` section
  - Add `feedback_context` from feedback history
  - Lower temperature from current value to 0.1

**Recommended: Option A.** The processor's prompts are battle-tested. Don't maintain two versions.

### Phase 7: Configurable Clip Duration

The video agent hardcodes 3-5s clips. This should be configurable per-scene based on scene duration.

```python
# In BRollAdapter
MIN_CLIP_DURATION = 3   # Keep short for fast-paced scenes
MAX_CLIP_DURATION = 8   # Allow longer for establishing shots

@staticmethod
def get_clip_duration_range(scene: SceneScript) -> tuple[float, float]:
    """Dynamic clip duration based on scene context."""
    if scene.duration_est <= 3:
        return (2, 4)   # Short scene = short clip
    elif scene.duration_est <= 6:
        return (3, 6)   # Medium scene
    else:
        return (4, 8)   # Long scene = allow longer establishing shots
```

---

## Configuration Mapping

How existing processor config maps to video agent usage:

| Processor Config | Default | Video Agent Equivalent | Recommended |
|-----------------|---------|----------------------|-------------|
| `CLIPS_PER_MINUTE` | 2.0 | N/A (script-driven) | Skip - scenes define needs |
| `MIN_CLIP_DURATION` | 4.0 | `MIN_BROLL_CLIP_SECONDS=3` | 3 (keep short for pacing) |
| `MAX_CLIP_DURATION` | 15.0 | `MAX_BROLL_CLIP_SECONDS=4` | 8 (allow longer) |
| `MAX_VIDEOS_PER_PHRASE` | 3 | N/A | 3 |
| `COMPETITIVE_ANALYSIS_ENABLED` | true | N/A (new) | true |
| `PREVIEWS_PER_NEED` | 2 | N/A (new) | 2 |
| `CLIPS_PER_NEED_TARGET` | 1 | N/A (new) | 1 |
| `USE_TWO_PASS_DOWNLOAD` | true | Already implemented | true |
| `PREVIEW_MAX_HEIGHT` | 360 | Already 360p | 360 |
| `SEMANTIC_VERIFICATION_ENABLED` | true | N/A (new) | true |
| `STYLE_DETECTION_ENABLED` | true | N/A (new) | true |
| `SEARCH_SOURCES` | youtube,pexels,pixabay | `broll_source_priority` | Keep existing |
| `FEEDBACK_ENABLED` | true | N/A (new) | true (Phase 8+) |

---

## Files to Modify

| File | Change | Effort |
|------|--------|--------|
| `src/video_agent/broll_adapter.py` | **NEW** - SceneScript -> BRollNeed adapter | Small |
| `src/video_agent/agent.py` | Replace `_search_and_download_broll()` + add service init + remove dead methods | Large |
| `src/video_agent/models.py` | Add optional clip duration fields to SceneScript | Small |
| `src/services/prompts/evaluation.py` | Remove `VIDEO_AGENT_BROLL_EVALUATOR` (replaced by processor's evaluator) | Small |
| `src/api/schemas.py` | Add competitive_analysis config to `VideoProduceRequest` | Small |
| `src/api/routers/video_agent.py` | Pass new config through to agent | Small |

**Files NOT modified (reused as-is):**
- `src/services/video_acquisition_service.py`
- `src/services/video_search_service.py`
- `src/services/clip_extractor.py`
- `src/services/video_downloader.py`
- `src/services/video_filter.py`
- `src/services/semantic_verifier.py`
- `src/services/ai_service.py`
- `src/services/feedback_service.py`

---

## Migration Strategy

### Step 1: Feature Flag (Day 1)
Add `use_processor_broll: bool = False` to video agent config. When false, existing behavior. When true, new pipeline. This lets you A/B test quality.

### Step 2: Adapter + Service Wiring (Day 1-2)
Build `BRollAdapter`, wire `VideoAcquisitionService` into agent `__init__()`. Behind feature flag.

### Step 3: Replace Core Method (Day 2-3)
Swap `_search_and_download_broll()` internals. Test with a few videos. Compare output quality.

### Step 4: Add Competitive Analysis (Day 3)
Enable `competitive_analysis_enabled=True`. This is the biggest quality jump - multiple previews per scene, best-of-N selection.

### Step 5: Add Style Detection + Semantic Verification (Day 4)
Layer on style coherence and post-download quality gates.

### Step 6: Remove Feature Flag (Day 5)
Once quality is confirmed better, remove the flag and delete the old internal methods (~400 lines of dead code).

---

## Expected Quality Improvements

| Improvement | Mechanism | Expected Impact |
|------------|-----------|----------------|
| Better video selection | `EVALUATOR_V4` with original_context as primary criterion | Clips match narrative meaning, not just keywords |
| Fewer bad clips | Competitive analysis (2+ candidates per scene) | Best-of-N selection instead of first-success |
| Visual coherence | Style detection flowing into all evaluations | Consistent look across all B-roll in video |
| Semantic accuracy | Post-download verification against scene voiceover | Catches "looks right but wrong context" clips |
| Richer search | alternate_searches + negative_keywords from adapter | Better YouTube results, fewer irrelevant downloads |
| Metadata filtering | `VideoPreFilter` blocks compilations/vlogs/reactions | Less junk even reaches evaluation |

**Conservative estimate:** B-roll quality parity with the standalone processor tab, with the added benefit of the Director's video-vs-image decisions and the full video production pipeline around it.

---

## What NOT to Bring Over

Some processor features don't make sense in the video agent context:

- **`BRollPlan` generation from transcript** - The video agent already has a script with per-scene visual needs. No need to re-plan from scratch.
- **`CLIPS_PER_MINUTE` density control** - Scene count is script-driven, not density-driven.
- **`FileOrganizer` timestamp-prefixed folders** - Video agent manages its own temp directory structure.
- **`FileMonitor` daemon mode** - Video agent is API-driven, not folder-watching.
- **`DriveService` / `NotificationService`** - Video agent has its own delivery (WebSocket + download endpoint).
- **`ProcessingCheckpoint`** - Nice to have eventually, but the video agent's job system already tracks state.
- **`ImageAcquisitionService`** - Video agent already has its own image generation pipeline (Flux, Google, stock).

---

## Cost Impact

The competitive analysis adds ~1 extra preview download + Gemini analysis per scene. For a typical 10-scene video:

| Cost Category | Current | With Processor Pipeline | Delta |
|--------------|---------|------------------------|-------|
| Preview downloads | 10 (360p) | 20 (360p) | +10 small downloads |
| Gemini evaluations | 10 | 10 | Same (evaluates same result sets) |
| Gemini video analysis | 10 | 20 | +10 (analyzing 2 previews per scene) |
| High-res downloads | 10 (1080p) | 10 (1080p) | Same (only winners) |
| Semantic verification | 0 | 10 | +10 quick Gemini calls |
| **Total API cost delta** | ~$0.10 | ~$0.18 | **+$0.08 per video** |

**Trade-off:** ~80% better clip selection for ~80% more API cost on B-roll. Given that B-roll is a small fraction of total video production cost (TTS, image gen, and rendering dominate), this is negligible.

---

## Summary

The B-roll processor already has battle-tested services for every step of the pipeline. The video agent should **compose these services via an adapter layer** rather than maintaining its own inferior B-roll logic. The key wins are:

1. **Competitive analysis** - compare multiple candidates per scene (biggest quality jump)
2. **Context-aware evaluation** - `EVALUATOR_V4` scores on narrative meaning, not just keywords
3. **Semantic verification** - catches clips that look right but are wrong context
4. **Style detection** - ensures visual coherence across all clips

Total effort: ~3-5 days. Most of it is wiring and cleanup, not new logic.
