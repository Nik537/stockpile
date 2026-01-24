# Starting the Stockpile Web UI

Quick guide to run the full-stack application (backend API + frontend web UI).

## Prerequisites

- Python 3.10+ environment with stockpile dependencies installed
- Node.js 18+ installed
- `.env` file configured with API keys

## Step-by-Step Instructions

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 2. Start the Backend API (Terminal 1)

**Easy way (recommended):**
```bash
./start_backend.sh
```

**Manual way:**
```bash
python src/api/server.py
```

You should see:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend is now running at `http://localhost:8000`**

### 3. Start the Frontend Dev Server (Terminal 2)

**Easy way (recommended):**
```bash
./start_frontend.sh
```

**Manual way:**
```bash
cd web
npm install  # First time only
npm run dev
```

You should see:
```
  VITE v5.0.12  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Frontend is now running at `http://localhost:5173`**

### 4. Access the Application

Open your browser and go to:

**http://localhost:5173**

## Using the Web UI

### Upload a Video

1. Drag and drop a video file onto the upload zone (or click to browse)
2. Optionally set user preferences:
   - **B-roll Style**: e.g., "cinematic", "documentary", "raw"
   - **Content to Avoid**: e.g., "text overlays, logos"
   - **Time of Day**: e.g., "golden hour", "night"
   - **Preferred Sources**: e.g., "nature footage, city aerials"
3. The video will upload and processing will start automatically

### Monitor Progress

- Real-time progress updates appear in the job card
- Progress bar shows completion percentage
- Status updates show current processing stage:
  - Transcribing audio
  - Planning B-roll needs
  - Searching and downloading videos
  - Extracting clips

### Download Results

When processing is complete:
1. Click the "⬇️ Download" button on the job card
2. A ZIP file containing all extracted B-roll clips will download

### Manage Jobs

- **View All Jobs**: The jobs section shows all processing jobs (past and present)
- **Delete Job**: Click "🗑️ Delete" to remove a job from the list

## Troubleshooting

### Backend Not Starting

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install backend dependencies
```bash
pip install -r requirements.txt
```

### Frontend Not Starting

**Error**: `node_modules not found`

**Solution**: Install frontend dependencies
```bash
cd web
npm install
```

### API Connection Failed

**Error**: Network errors in browser console

**Solution**:
1. Verify backend is running on port 8000
2. Check no firewall blocking localhost:8000
3. Restart both backend and frontend

### Upload Fails

**Error**: "Upload failed" message

**Solution**:
1. Check backend logs for errors
2. Verify `uploads/` directory exists (backend creates it automatically)
3. Check video format is supported (MP4, MOV, AVI, MKV, WebM)
4. Ensure sufficient disk space

### WebSocket Not Connecting

**Error**: Real-time updates not working

**Solution**:
1. Check browser console for WebSocket errors
2. Verify backend WebSocket endpoint is running
3. Try refreshing the page
4. Restart both servers

## Production Deployment

For production deployment:

### Build Frontend

```bash
cd web
npm run build
```

This creates optimized files in `web/dist/`.

### Serve Frontend with Backend

Configure FastAPI to serve the built frontend:

```python
# In src/api/server.py, add after app creation:
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
```

### Run Production Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Access at `http://localhost:8000`

## Architecture

```
Browser (localhost:5173)
    ↓ HTTP/WS requests via Vite proxy
FastAPI Backend (localhost:8000)
    ↓ Processes videos
Stockpile Core (BRollProcessor)
    ↓ Calls AI APIs
OpenAI Whisper + Google Gemini
```

## Next Steps

- Try uploading different video types
- Experiment with user preferences to see how they affect B-roll selection
- Monitor the processing stages in real-time
- Download and review the extracted clips

Happy B-roll hunting! 🎬
