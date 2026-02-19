# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stockpile** is an AI-powered B-roll automation pipeline for video content creators. It processes videos to produce curated B-roll footage by transcribing content, identifying B-roll needs using AI, searching/downloading YouTube footage, and extracting optimal 4-15 second clips.

**Technology Stack:**
- Python 3.10+
- OpenAI Whisper (transcription)
- Google Gemini 3 Flash Preview (AI analysis, planning, evaluation)
- yt-dlp (YouTube downloads)
- FFmpeg (clip extraction)
- Google Drive API (optional cloud workflow)

**Development Status:** See `UPGRADE_PLAN.md` for the roadmap transforming stockpile from working prototype to production-ready tool. Key improvements in progress:
- ✅ Testing infrastructure (pytest, coverage, fixtures)
- ✅ Code quality tools (ruff, mypy, pre-commit hooks)
- 🔄 AI response caching (100% cost savings on re-processing)
- 🔄 Progress tracking and parallel processing (5-10x speed improvement)
- 📋 Web UI, checkpointing, batch processing (planned)

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with API keys: GEMINI_API_KEY, OPENAI_API_KEY
```

### Running the Application

**Daemon mode (watches input folder):**
```bash
python stockpile.py
```

**Interactive mode (single video with preferences):**
```bash
python stockpile.py -i path/to/video.mp4
```

**Non-interactive with predefined preferences:**
```bash
python run_with_preferences.py
```

### Testing

**Test framework:** pytest with pytest-asyncio, pytest-cov, pytest-mock

**Running tests:**
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Generate HTML coverage report
pytest --cov-report=html
```

**Test structure:**
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests for individual components
│   ├── test_models.py
│   ├── test_clip_extractor.py
│   ├── test_ai_service.py
│   └── test_video_downloader.py
└── integration/             # End-to-end workflow tests
    └── test_broll_processor.py
```

**Manual testing checklist:**
- Drop test videos in `input/` folder
- Verify output quality in `output/` folder
- Check log file: `src/broll_processor.log`
- Review clip extraction accuracy (4-15 second segments)

**Key verification points:**
- Clip duration matches `MIN_CLIP_DURATION` and `MAX_CLIP_DURATION` settings
- Output folder structure follows `{timestamp}_{description}/` pattern
- AI scoring appears in filenames: `clip1_5.2s-12.8s_score09_video.mp4`

### Code Quality Tools

**Development dependencies installed:**
- **ruff**: Fast Python linter and formatter (replaces black, isort, flake8)
- **mypy**: Static type checker
- **pre-commit**: Git hooks for automated quality checks

**Running quality checks:**
```bash
# Lint code (with auto-fix)
ruff check src/ --fix

# Format code
ruff format src/

# Type check
mypy src/

# Run all pre-commit hooks manually
pre-commit run --all-files

# Install pre-commit hooks (run once)
pre-commit install
```

**Pre-commit hooks automatically run on every commit:**
- Ruff linting and formatting
- Type checking with mypy
- File checks (trailing whitespace, EOF newlines, YAML/TOML/JSON syntax)
- Secret detection
- Prevents commits to main branch (use feature branches)

**Configuration files:**
- `pyproject.toml`: Ruff, mypy, pytest, coverage configuration
- `.pre-commit-config.yaml`: Pre-commit hook definitions

### Troubleshooting Commands

```bash
# Check FFmpeg installation
ffmpeg -version

# Verify Python environment
which python
python --version

# Check installed packages
pip list | grep -E "whisper|genai|yt-dlp"

# Monitor processing logs
tail -f src/broll_processor.log

# Test Google Drive authentication (if configured)
# Delete token.json and re-run to force re-auth
rm token.json
```

## Architecture

### Service-Oriented Design with Central Orchestrator

**Core Pattern:** `BRollProcessor` orchestrates all services:

```python
BRollProcessor (src/broll_processor.py)
    ├── TranscriptionService → Whisper API
    ├── AIService → Gemini API (planning, evaluation, analysis)
    ├── YouTubeService → Search API
    ├── VideoDownloader → yt-dlp wrapper
    ├── ClipExtractor → Gemini video analysis + FFmpeg
    ├── FileOrganizer → Folder structure management
    ├── DriveService → Google Drive upload/download
    ├── NotificationService → Gmail API
    └── FileMonitor → Watchdog + Drive polling
```

**All services are dependency-injected into `BRollProcessor.__init__()`** for testability and flexibility.

### Data Models (`src/models/`)

Critical data structures:

**`BRollNeed`** - Timeline-aware B-roll requirement:
- `timestamp`: When in the video this B-roll appears (e.g., "0m30s")
- `search_phrase`: What to search for (e.g., "city skyline aerial")
- `description`: Context for AI evaluation

**`BRollPlan`** - Complete planning result from AI:
- `needs`: List of `BRollNeed` objects
- Spread evenly across video timeline based on `CLIPS_PER_MINUTE` config

**`ClipSegment`** - AI-identified clip within a video:
- `start_time`, `end_time`: Precise timestamps in seconds
- `score`: Quality rating 1-10
- `rationale`: AI explanation of why this segment is good B-roll

**`ScoredVideo`** - Evaluated search result:
- `score`: Quality rating 1-10
- `url`, `title`, `duration`
- Used to rank which videos to download

### Two-Pass Download Optimization

**Critical performance feature** (85% bandwidth savings, 3-4x speed improvement):

1. **First pass:** Download low-quality preview (360p) for AI analysis
2. **AI analysis:** Identify optimal clip timestamps from preview
3. **Second pass:** Download ONLY the specific clips in high quality (1080p)
4. **Cleanup:** Delete preview video, keep only extracted clips

**Configuration:**
```bash
USE_TWO_PASS_DOWNLOAD=true           # Enable optimization
PREVIEW_MAX_HEIGHT=360               # Preview quality
CLIP_DOWNLOAD_FORMAT=bestvideo[height<=1080]+bestaudio/best
```

**Implementation:** `VideoDownloader.download_video_clips()` in `src/services/video_downloader.py`

### Competitive Analysis Pattern

**Recently implemented optimization** (80% clip reduction):

Instead of downloading one video per B-roll need, the system:
1. Downloads multiple preview videos per need (default: 2)
2. AI analyzes all candidates together using Gemini video analysis
3. Selects the SINGLE best clip across all candidates
4. Downloads only the winner in high resolution

**Configuration:**
```bash
COMPETITIVE_ANALYSIS_ENABLED=true
PREVIEWS_PER_NEED=2                  # Videos to compare
CLIPS_PER_NEED_TARGET=1              # Final clips per need
```

**Result:** Test video went from 15 clips → 3 clips while maintaining quality.

**Implementation:** `BRollProcessor._competitive_analysis()` in `src/broll_processor.py`

### Retry Pattern with Exponential Backoff

**Critical for reliability** when dealing with YouTube rate limits and network issues:

**Decorator-based retry logic** (`src/utils/retry.py`):
```python
@retry_api_call(max_retries=3, initial_delay=1.0)
async def your_api_call():
    # Will automatically retry on APIRateLimitError, NetworkError
    pass
```

**Custom error types:**
- `APIRateLimitError` - Gemini API rate limits
- `NetworkError` - Connection issues
- `YouTubeRateLimitError` - YouTube throttling

**Applied to:**
- All Gemini AI calls (planning, evaluation, video analysis)
- YouTube search requests
- Video downloads via yt-dlp

### File Monitoring Pattern

**Daemon mode supports dual sources:**

1. **Local filesystem:** Watchdog monitors `LOCAL_INPUT_FOLDER`
2. **Google Drive:** Polls `GOOGLE_DRIVE_INPUT_FOLDER_ID` every 30 seconds

**Implementation:** `FileMonitor` in `src/services/file_monitor.py`

**Critical behavior:**
- Processing files are tracked to prevent duplicate processing
- Input videos are protected from deletion (stored in `protected_input_files` set)
- After processing, local files remain; Drive files are optionally deleted

## Critical Implementation Details

### Clip Extraction with FFmpeg

**Two-stage process** in `ClipExtractor` (`src/services/clip_extractor.py`):

1. **AI Analysis:** Upload video to Gemini, get structured JSON response with timestamps
2. **FFmpeg Extraction:** Extract clips using precise timestamps

**FFmpeg command structure:**
```bash
ffmpeg -ss <start_time> -i <input_video> -t <duration> \
       -c:v libx264 -preset medium -crf 23 \
       -c:a aac -b:a 192k \
       -force_key_frames "expr:gte(t,<start_time>)" \
       <output_file>
```

**Critical flags:**
- `-force_key_frames` ensures frame-accurate cuts at start_time
- `-preset medium -crf 23` balances quality and file size
- `-ss` before `-i` for faster seeking (input seeking)

**Bug fix history:** Initial implementation had clips starting at 0s instead of start_time due to yt-dlp callback signature issues. Fixed by adding `force_keyframes_at_cuts`.

### YouTube Download Reliability

**Phase 2 improvements** for handling YouTube throttling:

**Rate limiting configuration:**
```bash
YTDLP_RATE_LIMIT=2000000            # 2MB/s limit
YTDLP_RETRIES=5                     # Retry attempts
```

**User agent rotation:** Configured in `VideoDownloader` to avoid bot detection.

**Format fallback strategy:**
```python
# Try high quality first, fall back to lower quality
formats = [
    'bestvideo[height<=1080]+bestaudio/best',
    'bestvideo[height<=720]+bestaudio/best',
    'best'
]
```

**Implementation:** `VideoDownloader._download_with_ytdlp()` in `src/services/video_downloader.py`

### Timeline-Aware B-Roll Planning

**AI generates B-roll needs spread evenly across video:**

**Configuration:**
```bash
CLIPS_PER_MINUTE=2                  # Density of B-roll needs
```

**Planning process** (`AIService.plan_broll_needs()` in `src/services/ai_service.py`):
1. Analyze full transcript with timestamps
2. Identify key moments requiring visual support
3. Generate `BRollNeed` objects with precise timestamps (e.g., "0m30s", "1m45s")
4. Ensure even distribution across video timeline

**Output structure:**
```
output/
  └── project_name_20250123_140523/
      ├── 0m30s_city_skyline_aerial/      ← Timestamp prefix
      │   └── clip1_5.2s-12.8s_score09_video.mp4
      ├── 1m15s_factory_workers/
      │   └── clip1_8.1s-14.3s_score07_video.mp4
      └── 2m45s_sunset_timelapse/
          └── clip1_10.5s-18.2s_score09_video.mp4
```

**Timestamp prefix enables video editors to quickly find where each B-roll belongs.**

### Interactive Mode Design

**Current implementation** (`src/services/interactive_ui.py`):

**Predefined questions + AI-generated context questions:**
```python
# Example predefined questions
"What style of B-roll? (cinematic, documentary, raw, etc.)"
"Any content to avoid? (text overlays, logos, specific subjects)"
"Preferred time of day? (golden hour, night, doesn't matter)"

# AI generates context-specific questions based on video content
# Example: For a tech review video, AI might ask:
"Should I focus on product close-ups or lifestyle shots?"
```

**User preferences are injected into AI prompts** during:
- B-roll planning (what to look for)
- Video evaluation (scoring criteria)
- Clip extraction (preferred visual style)

**Launch interactive mode:**
```bash
python stockpile.py -i path/to/video.mp4
```

## Configuration Deep Dive

### Required Environment Variables

```bash
GEMINI_API_KEY=              # Google AI Studio API key (ai.google.dev)
OPENAI_API_KEY=              # OpenAI platform key (for Whisper)
```

### Performance Tuning

**B-roll acquisition:**
```bash
MAX_VIDEOS_PER_PHRASE=3               # Videos to download per need
MAX_VIDEO_DURATION_SECONDS=900        # Skip videos longer than 15min
CLIPS_PER_MINUTE=2                    # Timeline density
```

**Clip extraction:**
```bash
CLIP_EXTRACTION_ENABLED=true          # Enable intelligent extraction
MIN_CLIP_DURATION=4                   # Minimum clip length
MAX_CLIP_DURATION=15                  # Maximum clip length
DELETE_ORIGINAL_AFTER_EXTRACTION=true # Clean up full videos
```

**Two-pass download:**
```bash
USE_TWO_PASS_DOWNLOAD=true
PREVIEW_MAX_HEIGHT=360                # Lower = faster analysis
CLIP_DOWNLOAD_FORMAT=bestvideo[height<=1080]+bestaudio/best
```

**Competitive analysis:**
```bash
COMPETITIVE_ANALYSIS_ENABLED=true
PREVIEWS_PER_NEED=2                   # More = better selection, slower
CLIPS_PER_NEED_TARGET=1               # Higher = more clips per need
```

### Google Drive Integration

**Optional cloud workflow:**

```bash
GOOGLE_DRIVE_INPUT_FOLDER_ID=         # Drive folder to monitor
GOOGLE_DRIVE_OUTPUT_FOLDER_ID=        # Drive folder for results
GOOGLE_CLIENT_ID=                     # OAuth client ID
GOOGLE_CLIENT_SECRET=                 # OAuth client secret
NOTIFICATION_EMAIL=                   # Email for completion notifications
```

**Folder IDs:** Extract from Drive URL: `https://drive.google.com/drive/folders/{FOLDER_ID}`

**OAuth setup:**
1. Create Google Cloud project
2. Enable Drive API and Gmail API
3. Create OAuth 2.0 Client ID (Desktop app)
4. First run will open browser for authorization
5. Token saved to `token.json` (auto-refresh)

## File Organization

```
stockpile/
├── src/
│   ├── broll_processor.py       # Central orchestrator (1,450 lines)
│   ├── main.py                  # Application entry point
│   ├── models/                  # Data structures
│   │   ├── broll_need.py        # Timeline-aware planning
│   │   ├── clip.py              # Extraction results
│   │   ├── user_preferences.py  # Interactive mode
│   │   └── video.py             # Search results
│   ├── services/                # Business logic layer
│   │   ├── ai_service.py        # Gemini integration (planning, evaluation)
│   │   ├── clip_extractor.py    # Video analysis + FFmpeg extraction
│   │   ├── drive_service.py     # Google Drive API
│   │   ├── file_monitor.py      # Watchdog + Drive polling
│   │   ├── file_organizer.py    # Output folder structure
│   │   ├── interactive_ui.py    # Terminal UI
│   │   ├── notification.py      # Gmail API
│   │   ├── transcription.py     # Whisper API
│   │   ├── video_downloader.py  # yt-dlp wrapper
│   │   └── youtube_service.py   # YouTube search API
│   └── utils/
│       ├── config.py            # .env loader and validator
│       └── retry.py             # Exponential backoff decorators
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared pytest fixtures
│   ├── unit/                    # Unit tests for components
│   │   ├── test_models.py
│   │   ├── test_clip_extractor.py
│   │   ├── test_ai_service.py
│   │   └── test_video_downloader.py
│   └── integration/             # End-to-end workflow tests
│       └── test_broll_processor.py
├── pyproject.toml               # Project config, ruff, mypy, pytest settings
├── .pre-commit-config.yaml      # Git hook configuration
├── requirements.txt             # Python dependencies
├── UPGRADE_PLAN.md              # Development roadmap
└── CLAUDE.md                    # This file - AI assistant guidance
```

## Common Development Tasks

### Adding a New Video Source

1. Create new service in `src/services/your_service.py`
2. Implement search interface returning `List[VideoResult]`
3. Inject service into `BRollProcessor.__init__()`
4. Add to search logic in `BRollProcessor._process_broll_need()`

### Modifying AI Prompts

**All AI prompts are in `AIService` (`src/services/ai_service.py`):**

- `plan_broll_needs()` - B-roll planning prompt
- `evaluate_videos()` - Video scoring prompt
- `ClipExtractor.analyze_video()` - Clip extraction prompt

**Prompt engineering tips:**
- Gemini 3 Flash Preview supports structured JSON output via `response_schema`
- Always provide examples in prompts for consistent formatting
- Use `temperature=0.2` for deterministic results

### Adjusting Clip Extraction Logic

**Modify `ClipExtractor` in `src/services/clip_extractor.py`:**

**Key parameters:**
- `min_clip_duration` / `max_clip_duration` - Length constraints
- `max_clips_per_video` - How many clips to extract per video

**AI analysis prompt** (`analyze_video()` method):
- Adjust scoring criteria
- Modify segment identification logic
- Change JSON response schema

**FFmpeg extraction** (`extract_clip()` method):
- Modify video codec settings (`-c:v libx264`)
- Adjust quality/size tradeoff (`-crf 23`)
- Change audio settings (`-c:a aac -b:a 192k`)

### Debugging AI Responses

**Enable verbose logging:**
```python
# In src/services/ai_service.py
logger.setLevel(logging.DEBUG)
```

**Log AI responses:**
```python
logger.debug(f"Raw AI response: {response.text}")
```

**Common issues:**
- Gemini returns malformed JSON → Add retry logic with reprompting
- Video analysis fails → Check video codec compatibility with Gemini
- Rate limit errors → Increase retry delay in `@retry_api_call` decorator

## Performance Characteristics

**Approximate processing time for 5-minute video:**
- Transcription: ~2 minutes (Whisper)
- B-roll planning: ~30 seconds (Gemini)
- Per B-roll need (×10): ~1 minute each
  - Search: 5 seconds
  - Evaluation: 10 seconds
  - Download: 20 seconds (with two-pass optimization)
  - Clip extraction: 25 seconds (AI analysis + FFmpeg)
- **Total: ~10-15 minutes**

**API costs per 5-minute video:**
- OpenAI Whisper: ~$0.03
- Gemini AI calls: ~$0.10
  - Planning: $0.01
  - Video evaluation (30 videos): $0.03
  - Clip extraction (30 videos): $0.06
- **Total: ~$0.13**

**Disk usage:**
- With clip extraction: ~200-500MB per project
- Without extraction: ~2-5GB per project

## Docker Deployment

**Dockerfile** uses Python 3.13-slim with FFmpeg:

```bash
# Build image
docker build -t stockpile .

# Run container
docker run -v $(pwd)/input:/app/input \
           -v $(pwd)/output:/app/output \
           -v $(pwd)/.env:/app/.env \
           stockpile
```

**Critical:** Mount `.env` file for API keys and mount `input`/`output` volumes for persistence.

## Known Limitations

1. **YouTube-only source:** No support for other video platforms yet (Pexels/Vimeo planned in Phase 3)
2. **No batch processing UI:** Can only monitor single input folder (batch processing planned in Phase 3)
3. **Limited error recovery:** Some failure modes require manual intervention (checkpointing system planned in Phase 3)
4. **No local LLM support:** Requires cloud AI APIs (Gemini, OpenAI)
5. ~~**Manual testing:** No automated test suite~~ **RESOLVED** - Automated testing infrastructure added in Phase 1

## Recent Bug Fixes & Improvements

**Jan 24, 2026 - Development Infrastructure (Phase 1):**
- Added pytest testing framework with coverage tracking
- Implemented code quality tools (ruff, mypy, pre-commit hooks)
- Created `tests/` directory with unit and integration test structure
- Added `pyproject.toml` with comprehensive linting/formatting/testing configuration
- Created `.pre-commit-config.yaml` with automated quality checks
- Updated `requirements.txt` with development and testing dependencies
- Created `UPGRADE_PLAN.md` documenting 10 improvements across 3 phases

**Jan 23, 2026 - Competitive Analysis:**
- Implemented multi-video comparison per B-roll need
- Reduced clip count by 80% (15 → 3 clips for test video)
- Added `PREVIEWS_PER_NEED` and `CLIPS_PER_NEED_TARGET` configuration

**Jan 23, 2026 - Clip Duration Fix:**
- Fixed clips downloading from 0s instead of specified start_time
- Added `force_keyframes_at_cuts` to FFmpeg command
- Corrected yt-dlp callback signature in `VideoDownloader`

**Phase 2 - YouTube Reliability:**
- Added rate limiting configuration (`YTDLP_RATE_LIMIT`)
- Implemented user agent rotation
- Added format fallback strategies
- Enhanced error handling for throttling

See `/COMPETITIVE_ANALYSIS_IMPLEMENTATION.md`, `/BUG_FIX_SUMMARY.md`, and `/RELIABILITY_IMPROVEMENTS_TEST_RESULTS.md` for details.

## RunPod Public Endpoints

RunPod provides pre-deployed public endpoints for popular AI models. These are managed by RunPod and require only an API key - no Docker deployment needed.

### Configuration

```bash
RUNPOD_API_KEY=              # RunPod API key (Settings > API Keys)
```

### API Format

**Base URL:** `https://api.runpod.ai/v2/{endpoint_slug}/{operation}`

**Operations:**
- `/runsync` - Synchronous (wait for result)
- `/run` - Asynchronous (returns job ID)
- `/status/{job_id}` - Check job status
- `/health` - Endpoint health check

**Authentication:** `Authorization: Bearer {RUNPOD_API_KEY}`

### Available Public Endpoints

#### Text-to-Speech

| Model | Slug | Input | Output | Pricing |
|-------|------|-------|--------|---------|
| Chatterbox Turbo | `chatterbox-turbo` | `prompt` | `audio_url` (WAV) | ~$0.06/1000 chars |

**Example:**
```bash
curl -X POST "https://api.runpod.ai/v2/chatterbox-turbo/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello world"}}'
```

**Response:**
```json
{
  "status": "COMPLETED",
  "output": {
    "audio_url": "https://image.runpod.ai/chatterbox-turbo/xxx.wav",
    "cost": 0.00432
  }
}
```

#### Image Generation

| Model | Slug | Input | Output | Pricing |
|-------|------|-------|--------|---------|
| Flux Dev | `black-forest-labs-flux-1-dev` | `prompt`, `width`, `height` | `images[].url` | $0.02/megapixel |
| Flux Schnell | `black-forest-labs-flux-1-schnell` | `prompt`, `width`, `height` | `images[].url` | $0.0024/megapixel |
| Flux Kontext | `black-forest-labs-flux-1-kontext-dev` | `prompt`, `image_url` | `images[].url` | ~$0.02/megapixel |
| Qwen Image 2.0 | Custom endpoint required | `prompt`, `width`, `height`, `seed` | `image` (base64) | ~$0.02/request |

**Example (Flux Dev):**
```bash
curl -X POST "https://api.runpod.ai/v2/black-forest-labs-flux-1-dev/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A futuristic city at sunset",
      "width": 1024,
      "height": 1024
    }
  }'
```

**Response:**
```json
{
  "status": "COMPLETED",
  "output": {
    "images": [{"url": "https://..."}],
    "seed": 12345
  }
}
```

### Public vs Custom Endpoints

| Feature | Public Endpoint | Custom Endpoint |
|---------|-----------------|-----------------|
| Setup | API key only | Docker build/push |
| Maintenance | RunPod handles | You handle |
| Customization | Limited | Full control |
| Voice cloning | No | Yes (with voice_reference) |
| Pricing | Usage-based | GPU time-based |

### Service Integration

The codebase supports both public and custom endpoints:

- **TTS:** `TTSService.generate_public()` uses `chatterbox-turbo`
- **Images:** `ImageGenerationService.generate_runpod()` uses Nano Banana Pro (editing)
- **Qwen Image 2.0:** `ImageGenerationService.generate_qwen_image()` requires custom endpoint (`RUNPOD_QWEN_IMAGE_ENDPOINT_ID`)

See `src/services/tts_service.py` and `src/services/image_generation_service.py` for implementation.
