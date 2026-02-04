# TTS Setup - Stockpile

## Services Status

| Service | URL | Status |
|---------|-----|--------|
| **Backend** | http://localhost:8000 | Running |
| **Frontend** | http://localhost:5173 | Running |
| **API Docs** | http://localhost:8000/docs | Available |

---

## TTS Endpoints

### Primary: RunPod Serverless
- **Endpoint URL:** `https://api.runpod.ai/v2/z20vukds1ejniw/runsync`
- **Endpoint ID:** `z20vukds1ejniw`
- **API Key:** `rpa_7FU3ZZEZ2B1ATXQYQ0GZ05Z90LK01R663RX97PLP37wub3`
- **Console:** https://console.runpod.io/serverless/user/endpoint/z20vukds1ejniw

### Backup: Google Colab (via ngrok)
- **TTS URL:** `https://cultrate-kortney-nonmulched.ngrok-free.dev`
- **TTS Endpoint:** `POST /tts`
- **Health Check:** `GET /api/ui/initial-data`
- **Colab Notebook:** https://colab.research.google.com/drive/1SvKy_zy6_wbjEqhCj-gMo2Vse1ccf1Ap

---

## Environment Variables (in `.env`)

```bash
# RunPod Serverless TTS (PRIMARY)
RUNPOD_API_KEY=rpa_7FU3ZZEZ2B1ATXQYQ0GZ05Z90LK01R663RX97PLP37wub3
RUNPOD_ENDPOINT_ID=z20vukds1ejniw

# Colab TTS (BACKUP)
COLAB_TTS_URL=https://cultrate-kortney-nonmulched.ngrok-free.dev

# TTS Settings
TTS_PRIMARY=runpod
TTS_FALLBACK_TO_COLAB=true
```

---

## Usage Examples

### cURL - RunPod
```bash
curl -X POST https://api.runpod.ai/v2/z20vukds1ejniw/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer rpa_7FU3ZZEZ2B1ATXQYQ0GZ05Z90LK01R663RX97PLP37wub3' \
  -d '{"input":{"text":"Hello, this is a test.","exaggeration":0.5,"cfg_weight":0.5,"temperature":0.8}}'
```

### cURL - Colab
```bash
curl -X POST https://cultrate-kortney-nonmulched.ngrok-free.dev/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello, this is a test.","exaggeration":0.5,"cfg_weight":0.5,"temperature":0.8}' \
  --output audio.mp3
```

### Python - Using TTSService
```python
from src.services.tts_service import TTSService

tts = TTSService()

# RunPod (Primary)
audio = await tts.generate_runpod(
    text="Hello world",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8
)

# Colab (Backup)
tts.set_server_url("https://cultrate-kortney-nonmulched.ngrok-free.dev")
audio = await tts.generate(
    text="Hello world",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8
)

# Save audio
with open("output.mp3", "wb") as f:
    f.write(audio)
```

---

## Quick Links

| Resource | URL |
|----------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| RunPod Console | https://console.runpod.io/serverless/user/endpoint/z20vukds1ejniw |
| RunPod Settings | https://console.runpod.io/user/settings |
| Colab TTS | https://cultrate-kortney-nonmulched.ngrok-free.dev |
| Colab Notebook | https://colab.research.google.com/drive/1SvKy_zy6_wbjEqhCj-gMo2Vse1ccf1Ap |

---

## Troubleshooting

### RunPod 401 Unauthorized
1. Go to https://console.runpod.io/user/settings
2. Check API Keys section
3. Create a new key if needed
4. Update `RUNPOD_API_KEY` in `.env`

### Colab Disconnected
1. Open the Colab notebook
2. Click "Connect to a hosted runtime"
3. Run all cells
4. Wait for the ngrok URL to appear

### Restart Services
```bash
# Kill existing
pkill -f "python src/api/server.py"
pkill -f "vite"

# Start backend
cd /Users/niknoavak/Desktop/YT/stockpile
source .venv/bin/activate
python src/api/server.py > backend.log 2>&1 &

# Start frontend
cd web && npm run dev > ../frontend.log 2>&1 &
```
