#!/bin/bash
# Chatterbox TTS Server Setup for Google Colab
#
# Usage: Copy and paste this entire script into Colab terminal
#
# Before running: Replace YOUR_NGROK_TOKEN with your token from:
# https://dashboard.ngrok.com/get-started/your-authtoken

ngrok config add-authtoken YOUR_NGROK_TOKEN && \
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install "chatterbox-tts @ git+https://github.com/devnen/chatterbox.git@master" fastapi uvicorn pyngrok nest_asyncio && \
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

threading.Thread(target=lambda: __import__('uvicorn').run(app, host='0.0.0.0', port=8004, log_level='warning'), daemon=True).start()
time.sleep(2)

from pyngrok import ngrok
print(f"\n🎉 URL: {ngrok.connect(8004).public_url}\n")

while True: time.sleep(60)
EOF
python server.py
