# RunPod Serverless Worker for Qwen3-TTS
# Build: docker build --platform linux/amd64 -t username/qwen3-tts-runpod:latest .
# Push: docker push username/qwen3-tts-runpod:latest

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FlashAttention2 for reduced VRAM usage
# Separate step because it requires --no-build-isolation and may fail on some platforms
RUN pip install flash-attn --no-build-isolation || echo "FlashAttention2 install failed, will use eager attention"

# Copy handler
COPY handler.py .

# Pre-download model weights during build (cache only, don't load into RAM)
# This avoids OOM during build while ensuring fast cold starts
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')"

# Start handler
CMD ["python", "-u", "handler.py"]
