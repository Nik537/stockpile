# MOSS-TTSD Deployment Guide

## What's Already Done

- **Frontend**: Per-speaker voice assignment UI in `TTSGenerator.tsx` — when MOSS-TTSD is selected with 2+ speakers, each speaker (S1-S5) gets a dropdown to assign a voice from the library
- **Backend**: `tts.py` accepts `voice_id_s1` through `voice_id_s5` form params, resolves each to a voice library file path, passes to `generate_moss_ttsd()`
- **RunPod Worker**: `runpod-moss-ttsd-worker/` contains Dockerfile + handler.py ready to build
- **Vercel**: Frontend pushed to `dev` branch and deployed
- **`.env`**: `RUNPOD_MOSS_TTSD_ENDPOINT_ID=` placeholder added (line 130)

## Remaining Steps

### 1. Build & Push Docker Image (Cloud Build Required)

Local build failed due to disk space (model is ~16GB). Build on a cloud VM instead.

**Option A: Vast.ai / RunPod Cloud (cheapest)**
```bash
# Rent a cheap CPU instance with 50GB+ disk
# SSH in, then:
git clone https://github.com/Nik537/stockpile.git
cd stockpile/runpod-moss-ttsd-worker

# Login to Docker Hub
docker login -u techtawn

# Build (takes ~20-30 min — mostly downloading model weights)
docker build --platform linux/amd64 -t techtawn/moss-ttsd-runpod:latest .

# Push
docker push techtawn/moss-ttsd-runpod:latest
```

**Option B: GitHub Actions (free but slow)**
Create `.github/workflows/build-moss.yml` in the repo to build and push on commit.

### 2. Create RunPod Serverless Endpoint

1. Go to https://runpod.io/console/serverless
2. Click **New Endpoint**
3. Settings:
   - **Docker Image**: `techtawn/moss-ttsd-runpod:latest`
   - **GPU**: A5000 (24GB) minimum, A100 40GB recommended
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 1-2
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 600 seconds
   - **Flash Boot**: Enable if available
4. Copy the **Endpoint ID** (e.g., `abc123xyz`)

### 3. Set Endpoint ID in `.env`

```bash
# In stockpile/.env, find this line and add your endpoint ID:
RUNPOD_MOSS_TTSD_ENDPOINT_ID=your-endpoint-id-here
```

Then restart the backend API server.

### 4. Verify It Works

1. Open https://social-media-multi-tool.vercel.app/
2. Go to TTS Generator
3. Select **MOSS-TTSD** model — should show "Configured" badge
4. Set speakers to 2+
5. Per-speaker voice dropdowns should appear
6. Enter dialogue text:
   ```
   [S1] Hello, how are you today?
   [S2] I'm doing great, thanks for asking!
   [S1] That's wonderful to hear.
   ```
7. Assign voices to S1 and S2 from the library
8. Click Generate Speech
9. First generation will be slow (~30-60s cold start), subsequent ones faster

### 5. Upload Custom Voice References (Optional)

Your voice library already has 43+ voices. To add more:

1. In the web app, scroll to the voice selector
2. Click **Add Voice**
3. Upload 5-10 seconds of clear speech (WAV/MP3)
4. Name it (e.g., "Deep Narrator", "Energetic Host")
5. The voice will be available for all TTS models including MOSS-TTSD

## Architecture Reference

```
Web App (Vercel)          Backend API (FastAPI)         RunPod Serverless
TTSGenerator.tsx    -->   /api/tts/generate        -->  moss-ttsd-runpod
  voice_id_s1             resolves voice IDs             handler.py
  voice_id_s2             to file paths                  MOSS-TTSD 8B model
  voice_id_s3             builds voice_ref_paths         voice cloning
  ...                     dict for each speaker          32kHz WAV output
```

## Key Files

| File | Purpose |
|------|---------|
| `web/src/components/TTSGenerator.tsx` | Frontend UI with per-speaker voice dropdowns |
| `web/src/components/TTSGenerator.css` | Styles for speaker voice assignment grid |
| `src/api/routers/tts.py` | Backend API accepting voice_id_s1-s5 params |
| `src/services/tts_service.py` | MOSS-TTSD service (generate_moss_ttsd method) |
| `runpod-moss-ttsd-worker/handler.py` | RunPod serverless handler |
| `runpod-moss-ttsd-worker/Dockerfile` | Docker build for the worker |
| `.env` | RUNPOD_MOSS_TTSD_ENDPOINT_ID config |

## Troubleshooting

- **"Not configured" badge**: `RUNPOD_MOSS_TTSD_ENDPOINT_ID` not set in `.env`
- **Timeout on first run**: Cold start takes 30-60s as model loads to GPU
- **Voice cloning poor quality**: Use clean, 5-10s audio with minimal background noise
- **Out of VRAM**: A5000 (24GB) is minimum; A100 40GB+ recommended for longer text
- **Docker build fails locally**: Need 50GB+ free disk space for model weights
