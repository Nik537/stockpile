# Stockpile Web UI

Modern web interface for Stockpile - AI-powered B-roll automation.

## Features

- 📤 **Drag & Drop Upload**: Simple file upload with visual feedback
- ⚙️ **User Preferences**: Configure B-roll style, content to avoid, time of day preferences
- 📊 **Real-time Progress**: WebSocket-powered live updates during processing
- 📋 **Job History**: View all processing jobs with status and progress
- ⬇️ **Download Results**: One-click download of completed B-roll packages
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- 🎨 **Dark/Light Mode**: Automatic theme based on system preference

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Build Tool**: Vite
- **File Upload**: react-dropzone
- **Styling**: Pure CSS (no frameworks)
- **Backend**: FastAPI + WebSockets

## Development Setup

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm/bun
- Python 3.10+ with stockpile backend running

### Installation

```bash
# Install dependencies
npm install

# Start development server (with hot reload)
npm run dev
```

The dev server will start at `http://localhost:5173` with proxy to backend at `http://localhost:8000`.

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Running the Full Stack

### Terminal 1: Start Backend API

```bash
cd /path/to/stockpile
source .venv/bin/activate
python src/api/server.py
```

Backend runs at `http://localhost:8000`

### Terminal 2: Start Frontend Dev Server

```bash
cd web
npm run dev
```

Frontend runs at `http://localhost:5173`

### Access the Application

Open `http://localhost:5173` in your browser.

## API Endpoints

### REST API

- `GET /api/health` - Health check
- `POST /api/process` - Upload and process video
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `DELETE /api/jobs/{job_id}` - Delete job
- `GET /api/jobs/{job_id}/download` - Download results

### WebSocket

- `WS /ws/status/{job_id}` - Real-time job status updates

## Project Structure

```
web/
├── src/
│   ├── components/          # React components
│   │   ├── UploadForm.tsx   # File upload with preferences
│   │   ├── JobList.tsx      # List of all jobs
│   │   ├── JobCard.tsx      # Individual job display
│   │   └── ProgressBar.tsx  # Progress visualization
│   ├── types.ts             # TypeScript type definitions
│   ├── App.tsx              # Main application component
│   ├── App.css              # Application styles
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── vite.config.ts           # Vite config
└── README.md                # This file
```

## Configuration

### Vite Proxy

The development server proxies API and WebSocket requests to the backend:

- `/api/*` → `http://localhost:8000/api/*`
- `/ws/*` → `ws://localhost:8000/ws/*`

Configure in `vite.config.ts` if backend runs on different port.

## User Preferences

The UI supports these optional preferences for B-roll processing:

- **Style**: Visual style (e.g., "cinematic", "documentary", "raw")
- **Avoid**: Content to avoid (e.g., "text overlays, logos")
- **Time of Day**: Preferred time (e.g., "golden hour", "night")
- **Preferred Sources**: Video sources (e.g., "nature footage, city aerials")

These preferences are sent to the backend and used during AI processing.

## WebSocket Communication

Jobs receive real-time updates via WebSocket:

```typescript
// Connect to job status WebSocket
const ws = new WebSocket(`ws://localhost:5173/ws/status/${jobId}`)

// Receive updates
ws.onmessage = (event) => {
  const update = JSON.parse(event.data)
  // update.status: "queued" | "processing" | "completed" | "failed"
  // update.progress: { stage, percent, message }
  // update.error: error message if failed
}
```

## Troubleshooting

### Backend Connection Failed

Ensure backend is running:
```bash
python src/api/server.py
```

### WebSocket Connection Failed

Check that:
1. Backend WebSocket server is running
2. No firewall blocking WebSocket connections
3. Correct port in `vite.config.ts` proxy settings

### Upload Fails

Verify:
1. Backend `/api/process` endpoint is accessible
2. `uploads/` directory exists and is writable
3. Video file format is supported

## Future Enhancements

- [ ] Authentication/authorization
- [ ] Multi-user support with accounts
- [ ] Progress notifications (browser push, email)
- [ ] Job scheduling and queuing
- [ ] Advanced filtering and search in job history
- [ ] Real-time preview of extracted clips
- [ ] Batch upload support
- [ ] Job sharing and collaboration

## License

Part of the Stockpile project.
