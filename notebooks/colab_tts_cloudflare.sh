#!/bin/bash
# Chatterbox TTS Server with Cloudflare Tunnel (FREE static URL)
#
# SETUP (one-time):
# 1. Create free Cloudflare account: https://dash.cloudflare.com/sign-up
# 2. Go to: Zero Trust → Networks → Tunnels → Create a tunnel
# 3. Name it "tts-server" → Save tunnel
# 4. Copy the tunnel token (looks like: eyJhIjoiNjk...)
# 5. Replace YOUR_TUNNEL_TOKEN below
# 6. In Cloudflare, add a Public Hostname:
#    - Subdomain: tts (or whatever you want)
#    - Domain: your-domain.com (or use *.cfargotunnel.com)
#    - Service: http://localhost:8004
#
# Your permanent URL will be: https://tts.your-domain.com
#
# ============================================================

# Download cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared

# Install dependencies
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "chatterbox-tts @ git+https://github.com/devnen/chatterbox.git@master" fastapi uvicorn nest_asyncio

# Create server
cat > server.py << 'EOF'
import torch, threading, time, io
from chatterbox.tts import ChatterboxTTS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import torchaudio

print('Loading model...')
model = ChatterboxTTS.from_pretrained(device='cuda')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'], allow_credentials=True)

class R(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8

@app.get('/api/ui/initial-data')
async def h(): return {'status': 'ok'}

@app.post('/tts')
async def t(r: R):
    wav = model.generate(r.text, exaggeration=r.exaggeration, cfg_weight=r.cfg_weight, temperature=r.temperature)
    b = io.BytesIO(); torchaudio.save(b, wav, model.sr, format='mp3'); b.seek(0)
    return Response(content=b.read(), media_type='audio/mpeg')

import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8004, log_level='warning')
EOF

# Start server in background
python server.py &
sleep 5

# Start Cloudflare tunnel (replace with your token)
./cloudflared tunnel run --token YOUR_TUNNEL_TOKEN
