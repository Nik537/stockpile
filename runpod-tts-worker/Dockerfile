# RunPod Serverless Worker for Chatterbox TTS
# Build: docker build --platform linux/amd64 -t username/chatterbox-tts-runpod:latest .
# Push: docker push username/chatterbox-tts-runpod:latest

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Pre-download the model during build (optional but recommended)
# This makes cold starts faster
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || true

# Start handler
CMD ["python", "-u", "handler.py"]
