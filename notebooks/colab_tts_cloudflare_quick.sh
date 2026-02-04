#!/bin/bash
# Chatterbox TTS Server with Cloudflare Quick Tunnel
#
# NO SETUP REQUIRED - but URL changes each time
# For static URL, use colab_tts_cloudflare.sh instead
#
# ============================================================

# Download cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared

# Install dependencies
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "chatterbox-tts @ git+https://github.com/devnen/chatterbox.git@master" fastapi uvicorn

# Create server
cat > server.py << 'EOF'
import torch, threading, time, io, subprocess, re
from chatterbox.tts import ChatterboxTTS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import torchaudio

print('Loading model...')
model = ChatterboxTTS.from_pretrained(device='cuda')
print('✅ Model loaded!')

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
    print(f'Generating {len(r.text)} chars...')
    wav = model.generate(r.text, exaggeration=r.exaggeration, cfg_weight=r.cfg_weight, temperature=r.temperature)
    b = io.BytesIO(); torchaudio.save(b, wav, model.sr, format='mp3'); b.seek(0)
    print('✅ Done')
    return Response(content=b.read(), media_type='audio/mpeg')

# Start server in thread
def run_server():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8004, log_level='warning')

threading.Thread(target=run_server, daemon=True).start()
time.sleep(3)

# Start cloudflare tunnel and capture URL
print('\n🌐 Starting tunnel...\n')
process = subprocess.Popen(
    ['./cloudflared', 'tunnel', '--url', 'http://localhost:8004'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in process.stdout:
    print(line, end='')
    match = re.search(r'https://[\w-]+\.trycloudflare\.com', line)
    if match:
        url = match.group(0)
        print('\n' + '='*60)
        print('🎉 TTS SERVER READY!')
        print('='*60)
        print(f'\n📋 Copy this URL into Stockpile:\n\n   {url}\n')
        print('='*60 + '\n')
        break

# Keep alive
while True:
    time.sleep(60)
EOF

python server.py
