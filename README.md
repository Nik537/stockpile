# 🎞️ stockpile

AI-powered B-roll pipeline that transforms your videos into organized, ready-to-edit footage. Drop a clip, get curated B-roll automatically.

Made for content creators who need B-roll fast. AI transcribes your video, identifies exactly where B-roll is needed, finds relevant footage on YouTube, scores it for quality, and extracts only the best clips.

## Example (with Google Drive)

**1. Drop video in input folder:**

<img src="media/input.gif" width="700" alt="Input Process">

**2. Get notification when processing completes:**

<img src="media/notif.gif" width="700" alt="Notification">

**3. Access organized b-roll folders:**

<img src="media/output.gif" width="700" alt="Output Result">

## ⚡ Quick Start (Local)

**Requirements:** Python, [FFmpeg](https://ffmpeg.org/download.html), [Gemini API key](https://aistudio.google.com/apikey), [OpenAI API key](https://platform.openai.com/api-keys)

```bash
# clone and install
git clone https://github.com/sasoder/stockpile.git
cd stockpile

# set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# configure
cp .env.example .env
# add your API keys to .env (GEMINI_API_KEY, OPENAI_API_KEY)

# run
python stockpile.py
```

Drop videos in `input/`, get organized B-roll clips in `output/`.

## How It Works

```
INPUT VIDEO
    ↓
1. TRANSCRIBE (OpenAI Whisper)
   → Timestamped transcript of your video
    ↓
2. AI PLANNING (Gemini)
   → Identifies specific moments needing B-roll
   → Example: "0m30s - city skyline aerial shot"
   → ~2 B-roll needs per minute
    ↓
3. YOUTUBE SEARCH
   → Finds candidate videos for each need
    ↓
4. AI EVALUATION (Gemini)
   → Scores videos 1-10 for B-roll quality
   → Filters by content preferences
    ↓
5. DOWNLOAD & EXTRACT
   → Downloads top-scored videos
   → AI analyzes and extracts best 4-15 second clips
   → Deletes originals (saves disk space)
    ↓
OUTPUT: ORGANIZED B-ROLL CLIPS
```

### Output Structure

```
📁 output/
  └── your_project_20250123_140523/
      ├── 0m30s_city_skyline_aerial/
      │   ├── clip1_5.2s-12.8s_score09_video.mp4
      │   └── clip2_15.0s-20.5s_score08_video.mp4
      ├── 1m15s_factory_workers_assembly/
      │   └── clip1_8.1s-14.3s_score07_video.mp4
      └── 2m45s_sunset_timelapse/
          └── clip1_10.5s-18.2s_score09_video.mp4
```

Each folder:
- **Timestamp prefix** shows where in your video this B-roll belongs
- **Descriptive name** explains what the B-roll shows
- **Score prefix** on clips indicates AI quality rating (06-10)
- **Time range** in filename shows the exact segment extracted

## 🎯 Key Features

### Intelligent Clip Extraction
Instead of downloading full 10-minute YouTube videos, stockpile uses AI to:
- Analyze each video for B-roll quality moments
- Extract only the best 4-15 second segments
- Name clips with timestamps: `clip1_5.2s-12.8s_score09_video.mp4`
- Delete originals automatically (massive disk space savings)

### Timeline-Aware Planning
AI understands your video's timeline and plans B-roll accordingly:
- Identifies specific moments that need B-roll
- Spreads clips evenly across your video
- Configurable density (default: ~2 clips per minute)

### Content Filtering
Optional filters let you specify preferences:
- Example: `"men only, no women"` or `"outdoor shots only"`
- Applied during AI video evaluation stage
- Helps maintain consistency in your content

### Cloud Workflow
Optional Google Drive integration:
- Drop videos in Drive input folder
- Get organized B-roll uploaded to Drive output folder
- Email notifications with Drive links
- Local cleanup after upload (saves space)

## ☁️ Google Drive Integration (Recommended)

For automated cloud workflow:

**1. Create OAuth Client:**

- Create a [Google Cloud project](https://console.cloud.google.com/)
- Enable Google Drive API and Gmail API
- Go to [OAuth Clients](https://console.cloud.google.com/auth/clients)
- Create OAuth 2.0 Client ID (Desktop app)
- Save client ID and secret to `.env`
- First run will prompt browser authorization

**2. Configure Drive folders in `.env`:**

```bash
# Google Drive Configuration
GOOGLE_DRIVE_INPUT_FOLDER_ID=your_input_folder_id
GOOGLE_DRIVE_OUTPUT_FOLDER_ID=your_output_folder_id
GOOGLE_CLIENT_ID=your_oauth_client_id
GOOGLE_CLIENT_SECRET=your_oauth_client_secret
NOTIFICATION_EMAIL=your@email.com
```

**Get Folder IDs:** Open the folder in Google Drive, copy ID from URL:
```
https://drive.google.com/drive/folders/FOLDER_ID_HERE
```

Now drop videos in your Drive input folder → get organized B-roll in output folder with email notifications.

## ⚙️ Configuration

Edit `.env` to customize behavior:

### Required
```bash
GEMINI_API_KEY=your_gemini_key        # Get from ai.google.dev
OPENAI_API_KEY=your_openai_key        # Get from platform.openai.com
```

### B-Roll Acquisition
```bash
MAX_VIDEOS_PER_PHRASE=3               # Videos to download per B-roll need
MAX_VIDEO_DURATION_SECONDS=900        # Skip videos longer than 15min
CLIPS_PER_MINUTE=2                    # B-roll density in planning
CONTENT_FILTER=""                     # Optional: "men only" or similar
```

### Clip Extraction
```bash
MIN_CLIP_DURATION=4                   # Minimum clip length in seconds
MAX_CLIP_DURATION=15                  # Maximum clip length in seconds
```

### Storage
```bash
# Local mode (default)
LOCAL_INPUT_FOLDER=input
LOCAL_OUTPUT_FOLDER=output

# Google Drive mode (optional)
GOOGLE_DRIVE_INPUT_FOLDER_ID=
GOOGLE_DRIVE_OUTPUT_FOLDER_ID=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
NOTIFICATION_EMAIL=
```

## 🚀 Planned: Interactive Terminal Mode

**Coming soon:** Terminal-based interactive workflow where AI asks you questions about the B-roll you want.

### Planned Workflow

```
1. Drop video or provide path
   ↓
2. AI transcribes and analyzes
   ↓
3. INTERACTIVE SESSION:

   🤖 AI: "I found your video discusses urban development.
          What style of B-roll are you looking for?"

   📝 You: "modern, cinematic cityscapes"

   🤖 AI: "Should I focus on daytime or nighttime footage?"

   📝 You: "both, but prefer golden hour"

   🤖 AI: "I identified 8 moments needing B-roll.
          How many clips per moment? (1-5)"

   📝 You: "2"

   ↓
4. AI proceeds with your preferences
   ↓
5. Get customized B-roll
```

### Why Interactive Mode?

- **Better results:** AI learns exactly what you're looking for
- **Flexibility:** Adjust preferences per project
- **Control:** Approve B-roll needs before searching
- **Efficiency:** No re-processing if results don't match vision

**Implementation planned for v2.0** - the AI will generate context-aware questions based on:
- Video content and topic
- Common B-roll preferences for that content type
- Your previous choices (learning from history)

*Track progress:* [GitHub Issue #X](#) *(create issue to track this feature)*

## 🛠️ Advanced Usage

### Run in Background

```bash
# Linux/Mac
nohup python stockpile.py > stockpile.log 2>&1 &

# Or use screen/tmux
screen -S stockpile
python stockpile.py
# Ctrl+A, D to detach
```

### Process Single Video (No Monitoring)

```python
from src.main import StockpileApp

app = StockpileApp()
await app.processor.process_video("/path/to/video.mp4")
```

### Custom Content Filters

Set `CONTENT_FILTER` in `.env`:
- `"men only, no women"` - gender filtering
- `"outdoor shots only"` - environment filtering
- `"no text overlays"` - technical filtering
- `"aerial drone footage preferred"` - style preference

Filters are applied during AI video evaluation stage.

## 📊 Performance & Costs

**Processing time** (approximate):
- 5-minute video → 10-15 minutes total
- Transcription: ~2 minutes
- B-roll planning: ~30 seconds
- Per B-roll need (x10): ~1 minute each
  - Search: 5 seconds
  - Evaluation: 10 seconds
  - Download: 20 seconds
  - Clip extraction: 25 seconds

**API costs** (per 5-minute video with 10 B-roll needs):
- OpenAI Whisper: ~$0.03
- Gemini AI calls: ~$0.10
  - Planning: $0.01
  - Video evaluation (30 videos): $0.03
  - Clip extraction (30 videos): $0.06
- **Total: ~$0.13 per video**

**Disk usage:**
- With clip extraction: ~200-500MB per project
- Without extraction: ~2-5GB per project (full videos)

## 🔧 Troubleshooting

**FFmpeg not found:**
```bash
# Mac
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

**Google Drive authorization fails:**
- Ensure Drive API and Gmail API are enabled
- Check OAuth consent screen is configured
- Delete `token.json` and re-authorize

**Video download fails:**
- Some videos may be geo-restricted or private
- yt-dlp automatically skips unavailable videos
- Check console output for specific errors

**No clips extracted from video:**
- AI may not find suitable 4-15 second segments
- Original video is kept in this case
- Adjust `MIN_CLIP_DURATION` / `MAX_CLIP_DURATION` in `.env`

## 🤝 Contributing

Contributions welcome! Areas of interest:
- [ ] Interactive terminal mode (see Planned section)
- [ ] Additional video sources beyond YouTube
- [ ] Custom AI model support (local LLMs)
- [ ] Web UI for configuration and monitoring
- [ ] Batch processing multiple videos

## 📝 License

MIT License - see LICENSE file

---

**Made for content creators by [@sasoder](https://github.com/sasoder)**

Built with: Python • OpenAI Whisper • Google Gemini • yt-dlp • FFmpeg
