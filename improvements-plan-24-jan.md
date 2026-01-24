# Stockpile Improvements Plan - January 24, 2026

## Overview

This document outlines targeted improvements to Stockpile's **processing speed** and **B-roll relevance quality**. Improvements are prioritized by impact and implementation effort.

---

## Speed Improvements

### S1. Parallelize B-Roll Need Processing - COMPLETED
**Priority:** Critical | **Effort:** Low | **Impact:** 5-10x speedup
**Status:** IMPLEMENTED (Jan 24, 2026)

B-roll needs now process in parallel using `asyncio.gather()` with batched execution.

**Implementation Details:**
- File: `src/broll_processor.py` - `_execute_pipeline()` method (lines 384-426)
- Batched parallel processing: needs are processed in batches of `max_concurrent_needs` at a time
- Semaphores control resource limits:
  - `ai_semaphore` - Rate-limits AI API calls (PARALLEL_AI_CALLS, default: 5)
  - `download_semaphore` - Rate-limits video downloads (PARALLEL_DOWNLOADS, default: 3)
  - `extraction_semaphore` - Rate-limits FFmpeg extractions (PARALLEL_EXTRACTIONS, default: 2)

**Configuration:**
- `MAX_PARALLEL_NEEDS=5` (preferred) or `MAX_CONCURRENT_NEEDS=5` (legacy alias)
- Config file: `src/utils/config.py`
- Environment: `.env.example`

**Acceptance Criteria:**
- [x] B-roll needs process in parallel (batched asyncio.gather)
- [x] Configurable concurrency limit (`MAX_PARALLEL_NEEDS`)
- [x] No race conditions in file writing (each need writes to unique timestamp-prefixed folder)
- [x] API rate limits respected (ai_semaphore wraps all AI calls)

---

### S2. AI Response Caching
**Priority:** High | **Effort:** Medium | **Impact:** 100% savings on re-runs

Cache Gemini API responses keyed by content hash. Already noted as "in progress" in UPGRADE_PLAN.md.

**Cache Strategy:**
```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(".cache/ai_responses")

def get_cached_response(prompt: str, content_hash: str) -> dict | None:
    cache_key = hashlib.sha256(f"{prompt}:{content_hash}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None

def cache_response(prompt: str, content_hash: str, response: dict):
    cache_key = hashlib.sha256(f"{prompt}:{content_hash}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(response))
```

**Cache Targets:**
- B-roll planning responses (keyed by transcript hash)
- Video evaluation responses (keyed by video URL + prompt)
- Clip extraction analysis (keyed by video file hash)

**Configuration:**
```bash
AI_CACHE_ENABLED=true
AI_CACHE_DIR=.cache/ai_responses
AI_CACHE_TTL_DAYS=30
```

**Acceptance Criteria:**
- [ ] Cache hits skip API calls entirely
- [ ] Cache invalidation by TTL
- [ ] Cache can be cleared manually
- [ ] Cache size monitoring/limits

---

### S3. Local Whisper Transcription ✅ COMPLETED
**Priority:** High | **Effort:** Low | **Impact:** ~2 min → ~30s

Replace OpenAI Whisper API with local `faster-whisper` (4x faster than original Whisper).

**Implementation:** See `src/services/transcription.py` - Updated to use faster-whisper

**Key changes:**
- Replaced `openai-whisper` with `faster-whisper>=1.0.0` in requirements.txt
- Updated `TranscriptionService` to use `faster_whisper.WhisperModel`
- Added VAD (Voice Activity Detection) for better segment accuracy
- Added configurable device (auto/cuda/cpu) and compute_type (auto/float16/int8)

**Configuration (in .env.example):**
```bash
TRANSCRIPTION_MODE=local          # local | api (api not yet implemented)
WHISPER_MODEL=base                # tiny, base, small, medium, large-v2, large-v3
WHISPER_DEVICE=auto               # auto, cuda, cpu
WHISPER_COMPUTE_TYPE=auto         # auto, float16, float32, int8
```

**Trade-offs:**
| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| tiny | Fastest | Lower | ~1GB |
| base | Fast | Good | ~1GB |
| small | Medium | Better | ~2GB |
| medium | Slower | Great | ~5GB |
| large-v2/v3 | Slowest | Best | ~10GB |

**Acceptance Criteria:**
- [x] Local transcription works without API key
- [x] Model size configurable
- [ ] Falls back to API if local fails (API mode not yet implemented)
- [x] GPU acceleration when available (device="auto" uses CUDA if present)

**Unit tests:** See `tests/unit/test_transcription.py`

---

### S4. Parallel YouTube Downloads ✅ COMPLETED
**Priority:** Medium | **Effort:** Low | **Impact:** 3-5x faster downloads

Download multiple preview videos simultaneously with concurrency control.

**Implementation:** See `src/services/video_downloader.py` - `ParallelDownloader` class

Key features implemented:
- `ParallelDownloader` class with semaphore-based concurrency control
- `DownloadProgress` dataclass for comprehensive progress tracking
- Timeout handling per download using `asyncio.wait_for()`
- Staggered start delay to avoid rate limit spikes
- Three download methods:
  - `download_many_previews()` - For low-quality preview downloads
  - `download_many_clips()` - For high-quality clip sections
  - `download_many_full_videos()` - For traditional full video downloads

**Configuration:**
```bash
MAX_CONCURRENT_DOWNLOADS=5
DOWNLOAD_TIMEOUT_SECONDS=120
DOWNLOAD_STAGGER_DELAY=0.5
```

**Acceptance Criteria:**
- [x] Multiple downloads run simultaneously
- [x] Respects YouTube rate limits (via stagger delay + semaphore)
- [x] Timeout handling per download
- [x] Progress tracking across all downloads

**Usage Example:**
```python
from services.video_downloader import ParallelDownloader

downloader = ParallelDownloader(max_concurrent=5, timeout=120)
results = await downloader.download_many_previews(
    videos=scored_videos,
    output_dir=Path("/output"),
    progress_callback=lambda p: print(p.get_status_message())
)
```

---

### S5. Aggressive Pre-Filtering
**Priority:** Medium | **Effort:** Low | **Impact:** Fewer wasted downloads

Filter YouTube results BEFORE downloading based on metadata.

**Filter Criteria:**
```python
def should_download(video: YouTubeResult) -> bool:
    # Skip low-quality indicators
    if video.view_count < 1000:
        return False  # Likely poor quality

    # Skip overly long videos
    if video.duration > 600:  # 10 minutes
        return False

    # Skip videos with problematic titles
    bad_keywords = ["compilation", "top 10", "reaction", "review"]
    if any(kw in video.title.lower() for kw in bad_keywords):
        return False

    # Prefer Creative Commons
    if video.license == "creativeCommon":
        return True  # Boost priority

    return True
```

**Configuration:**
```bash
MIN_VIEW_COUNT=1000
MAX_PREFILTER_DURATION=600
PREFER_CREATIVE_COMMONS=true
BLOCKED_TITLE_KEYWORDS=compilation,top 10,reaction,review
```

**Acceptance Criteria:**
- [ ] Filters applied before any downloads
- [ ] Configurable filter rules
- [ ] Logging shows why videos were skipped
- [ ] Filter stats in final report

---

## Quality Improvements

### Q1. CLIP Embeddings for Visual Matching - COMPLETED
**Priority:** Critical | **Effort:** Medium | **Impact:** Visual relevance scoring
**Status:** IMPLEMENTED (Jan 24, 2026)

Use OpenAI CLIP to compare video frames against text descriptions. This is the single biggest quality improvement available.

**Implementation Details:**

**Files modified/created:**
- `src/services/visual_matcher.py` - Complete VisualMatcher service with CLIP integration
- `src/services/clip_extractor.py` - Added CLIP parameters and integration methods
- `src/models/clip.py` - Added `clip_score` and `gemini_score` metadata fields
- `src/broll_processor.py` - Passes CLIP config to ClipExtractor
- `src/utils/config.py` - Added CLIP configuration variables
- `.env.example` - Added CLIP environment variables
- `requirements.txt` - Added CLIP dependencies (transformers, torch, torchvision, opencv-python, pillow)
- `tests/unit/test_visual_matcher.py` - 16 unit tests for CLIP functionality

**Key Features:**
- Lazy loading of CLIP model (only loads when first used)
- GPU acceleration (auto-detects CUDA, MPS, or falls back to CPU)
- Efficient frame extraction using OpenCV (not full decode)
- Batch processing to avoid OOM on large videos
- Combined scoring (configurable CLIP + Gemini weight)
- CLIP scores included in segment metadata

**New Methods in ClipExtractor:**
- `score_video_with_clip()` - Quick overall video relevance score
- `find_best_segment_with_clip()` - Find best segment using visual matching
- `enhance_segments_with_clip()` - Enhance Gemini segments with CLIP scores
- `analyze_video_with_clip()` - Full analysis with CLIP enhancement

**Configuration (in .env.example):**
```bash
CLIP_ENABLED=true
CLIP_MODEL=openai/clip-vit-base-patch32
CLIP_SAMPLE_RATE=1                    # Frames per second to analyze
CLIP_MIN_SCORE_THRESHOLD=0.3          # Minimum similarity score
CLIP_USE_FOR_PREFILTER=true           # Use CLIP as pre-filter
CLIP_USE_FOR_SCORING=true             # Use CLIP for final scoring
CLIP_WEIGHT_IN_SCORE=0.4              # 40% CLIP + 60% Gemini
```

**Acceptance Criteria:**
- [x] CLIP scoring integrated into evaluation pipeline
- [x] Frames extracted efficiently (not full decode) - Uses OpenCV with frame skipping
- [x] GPU acceleration when available - Auto-detects CUDA > MPS > CPU
- [x] Scores included in output metadata - Added clip_score and gemini_score to ClipSegment

**Unit Tests:** 16 tests in `tests/unit/test_visual_matcher.py`

---

### Q2. Enhanced Search Phrase Generation
**Priority:** High | **Effort:** Low | **Impact:** Better search coverage

Improve the AI planning prompt to generate richer search metadata.

**Enhanced BRollNeed Model:**
```python
@dataclass
class EnhancedBRollNeed:
    timestamp: str
    primary_search: str           # Main search phrase
    alternate_searches: list[str] # 2-3 synonym variations
    negative_keywords: list[str]  # What to avoid
    visual_style: str             # cinematic, documentary, raw, etc.
    time_of_day: str | None       # golden hour, night, day, etc.
    movement: str | None          # static, pan, drone, handheld
```

**Improved Planning Prompt:**
```
For each B-roll moment, provide:
1. Primary search phrase (most specific)
2. 2-3 alternate search phrases (synonyms, related terms)
3. Negative keywords (what should NOT appear)
4. Preferred visual style
5. Time of day if relevant
6. Camera movement preference if relevant

Example output:
{
  "timestamp": "0m30s",
  "primary_search": "city skyline aerial drone golden hour",
  "alternate_searches": [
    "urban landscape drone sunset",
    "downtown buildings aerial view",
    "metropolitan skyline from above"
  ],
  "negative_keywords": ["night", "rain", "text overlay", "logo"],
  "visual_style": "cinematic",
  "time_of_day": "golden hour",
  "movement": "slow drone push-in"
}
```

**Search Strategy:**
```python
async def search_with_fallback(need: EnhancedBRollNeed) -> list[VideoResult]:
    # Try primary search first
    results = await youtube.search(need.primary_search)

    # If insufficient results, try alternates
    if len(results) < 5:
        for alt in need.alternate_searches:
            more_results = await youtube.search(alt)
            results.extend(more_results)
            if len(results) >= 10:
                break

    # Filter out negative keywords
    return [r for r in results if not has_negative_keywords(r, need.negative_keywords)]
```

**Acceptance Criteria:**
- [ ] Planning returns enhanced metadata
- [ ] Alternate searches used as fallback
- [ ] Negative keywords filter results
- [ ] Visual style passed to evaluation

---

### Q3. Multi-Source Search
**Priority:** High | **Effort:** Medium | **Impact:** Higher quality footage

Add Pexels and Pixabay APIs for professional stock footage.

**Service Implementations:**

```python
# Pexels API (free, CC0 license)
class PexelsService:
    BASE_URL = "https://api.pexels.com/videos/search"

    async def search(self, query: str, per_page: int = 10) -> list[VideoResult]:
        headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
        params = {"query": query, "per_page": per_page}

        async with aiohttp.ClientSession() as session:
            async with session.get(self.BASE_URL, headers=headers, params=params) as resp:
                data = await resp.json()
                return [self._to_video_result(v) for v in data["videos"]]

# Pixabay API (free, CC0 license)
class PixabayService:
    BASE_URL = "https://pixabay.com/api/videos/"

    async def search(self, query: str, per_page: int = 10) -> list[VideoResult]:
        params = {
            "key": os.getenv("PIXABAY_API_KEY"),
            "q": query,
            "per_page": per_page
        }
        # Similar implementation...
```

**Unified Search:**
```python
class MultiSourceSearcher:
    def __init__(self):
        self.sources = [
            YouTubeService(),
            PexelsService(),
            PixabayService(),
        ]

    async def search(self, query: str) -> list[VideoResult]:
        results = await asyncio.gather(*[
            source.search(query) for source in self.sources
        ])

        # Merge and deduplicate
        all_results = []
        for source_results in results:
            all_results.extend(source_results)

        # Rank by quality indicators
        return self._rank_results(all_results)
```

**Configuration:**
```bash
PEXELS_API_KEY=your_key_here
PIXABAY_API_KEY=your_key_here
SEARCH_SOURCES=youtube,pexels,pixabay
PREFER_STOCK_FOOTAGE=true             # Prioritize Pexels/Pixabay over YouTube
```

**Benefits:**
- Professional quality footage
- Proper CC0 licensing (no attribution required)
- No watermarks or ads
- Consistent visual style

**Acceptance Criteria:**
- [ ] Pexels integration working
- [ ] Pixabay integration working
- [ ] Results merged and ranked
- [ ] Source tracking in output metadata

---

### Q4. Context-Aware Evaluation Prompt
**Priority:** Medium | **Effort:** Low | **Impact:** More relevant scoring

Pass the original video's transcript context to the evaluation AI.

**Current Evaluation:**
```python
prompt = f"Evaluate this video for: {broll_need.description}"
```

**Improved Evaluation:**
```python
prompt = f"""
CONTEXT FROM ORIGINAL VIDEO:
The following is the transcript segment where this B-roll will be inserted:
"{transcript_segment}"

USER PREFERENCES:
- Style: {preferences.style}
- Avoid: {preferences.avoid_content}
- Preferred mood: {preferences.mood}

B-ROLL REQUIREMENT:
- Timestamp: {broll_need.timestamp}
- Description: {broll_need.description}
- Visual style: {broll_need.visual_style}

EVALUATION TASK:
Score this candidate video 1-10 based on:
1. Relevance to the transcript context (what is being discussed)
2. Match to the visual style requirements
3. Technical quality (resolution, stability, lighting)
4. Absence of unwanted elements (text, logos, watermarks)
5. Emotional tone match

Provide your score and a brief rationale.
"""
```

**Acceptance Criteria:**
- [ ] Transcript context passed to evaluation
- [ ] User preferences influence scoring
- [ ] Scoring criteria are explicit and consistent

---

### Q5. Scene Segmentation Before Extraction
**Priority:** Medium | **Effort:** Medium | **Impact:** More coherent clips

Use PySceneDetect to identify distinct scenes, then select the best complete scene.

**Installation:**
```bash
pip install scenedetect[opencv]
```

**Implementation:**
```python
from scenedetect import detect, ContentDetector, split_video_ffmpeg

class SceneAwareExtractor:
    def detect_scenes(self, video_path: str) -> list[tuple[float, float]]:
        """Detect scene boundaries in video."""
        scene_list = detect(video_path, ContentDetector())
        return [(scene[0].get_seconds(), scene[1].get_seconds())
                for scene in scene_list]

    def find_best_scene(self, video_path: str, description: str,
                        min_duration: float = 4, max_duration: float = 15
                        ) -> tuple[float, float]:
        """Find the scene that best matches the description."""
        scenes = self.detect_scenes(video_path)

        # Filter by duration
        valid_scenes = [
            (start, end) for start, end in scenes
            if min_duration <= (end - start) <= max_duration
        ]

        # Score each scene with CLIP
        best_score = 0
        best_scene = None

        for start, end in valid_scenes:
            frames = self.extract_scene_frames(video_path, start, end)
            score = self.clip_scorer.score_frames(frames, description)
            if score > best_score:
                best_score = score
                best_scene = (start, end)

        return best_scene
```

**Benefits:**
- Clips don't cut mid-action or mid-sentence
- More visually coherent segments
- Natural scene boundaries

**Configuration:**
```bash
SCENE_DETECTION_ENABLED=true
SCENE_DETECTION_THRESHOLD=27          # ContentDetector threshold
PREFER_COMPLETE_SCENES=true
```

**Acceptance Criteria:**
- [ ] Scene detection integrated
- [ ] Clips align to scene boundaries when possible
- [ ] Falls back to timestamp-based extraction if no good scenes

---

## Implementation Priority Matrix

| ID | Improvement | Speed | Quality | Effort | Priority |
|----|-------------|-------|---------|--------|----------|
| S1 | Parallelize B-roll processing | +++ | - | Low | **P0** |
| Q1 | CLIP visual matching | - | +++ | Medium | **P0** |
| S3 | Local Whisper | ++ | - | Low | **P1** |
| Q2 | Enhanced search phrases | - | ++ | Low | **P1** |
| S4 | Parallel downloads | ++ | - | Low | **P1** |
| Q3 | Multi-source search | - | ++ | Medium | **P2** |
| S2 | AI response caching | ++ | - | Medium | **P2** |
| Q4 | Context-aware evaluation | - | + | Low | **P2** |
| S5 | Aggressive pre-filtering | + | + | Low | **P3** |
| Q5 | Scene segmentation | - | + | Medium | **P3** |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. **S1**: Parallelize B-roll processing
2. **S3**: Local Whisper transcription
3. **Q2**: Enhanced search phrase generation
4. **S4**: Parallel YouTube downloads

### Phase 2: Quality Core (2-3 days)
5. **Q1**: CLIP embeddings integration
6. **Q3**: Multi-source search (Pexels, Pixabay)
7. **Q4**: Context-aware evaluation prompts

### Phase 3: Polish (1-2 days)
8. **S2**: AI response caching
9. **S5**: Aggressive pre-filtering
10. **Q5**: Scene segmentation

---

## New Dependencies

```txt
# Speed improvements
faster-whisper>=0.10.0          # Local transcription

# Quality improvements
transformers>=4.35.0            # CLIP model
torch>=2.0.0                    # PyTorch for CLIP
torchvision>=0.15.0             # Image processing
open-clip-torch>=2.20.0         # Alternative CLIP (faster)
scenedetect[opencv]>=0.6.0      # Scene detection
aiohttp>=3.9.0                  # Async HTTP for API calls
```

---

## New Configuration Variables

```bash
# Speed
MAX_PARALLEL_NEEDS=5
MAX_CONCURRENT_DOWNLOADS=5
TRANSCRIPTION_MODE=local
WHISPER_MODEL_SIZE=base

# Caching
AI_CACHE_ENABLED=true
AI_CACHE_DIR=.cache/ai_responses
AI_CACHE_TTL_DAYS=30

# Pre-filtering
MIN_VIEW_COUNT=1000
BLOCKED_TITLE_KEYWORDS=compilation,top 10,reaction

# Quality - CLIP
CLIP_ENABLED=true
CLIP_MODEL=openai/clip-vit-base-patch32
CLIP_SAMPLE_RATE=1
CLIP_MIN_SCORE_THRESHOLD=0.3

# Quality - Multi-source
PEXELS_API_KEY=
PIXABAY_API_KEY=
SEARCH_SOURCES=youtube,pexels,pixabay
PREFER_STOCK_FOOTAGE=true

# Quality - Scene detection
SCENE_DETECTION_ENABLED=true
SCENE_DETECTION_THRESHOLD=27
```

---

## Success Metrics

### Speed Targets
| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| 5-min video processing | ~15 min | ~3 min | 5x faster |
| Transcription | ~2 min | ~30s | 4x faster |
| Per-need processing | ~60s | ~15s | 4x faster |
| Re-run (cached) | ~15 min | ~1 min | 15x faster |

### Quality Targets
| Metric | Current | Target |
|--------|---------|--------|
| Clip relevance (user rating) | ~6/10 | ~8/10 |
| Clips needing manual replacement | ~40% | ~15% |
| Visual style match | ~50% | ~80% |
| Licensing clarity | Variable | 100% clear |

---

## Notes

- All improvements are backward compatible
- Each can be enabled/disabled via configuration
- Implement with feature flags for gradual rollout
- Add telemetry to measure actual impact
