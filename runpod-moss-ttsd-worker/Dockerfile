# RunPod Serverless Worker for MOSS-TTSD (Multi-Speaker Dialogue TTS)
# 8B parameter model - requires GPU with 24+ GB VRAM (A5000, A100, L40S)
#
# Build: docker build --platform linux/amd64 -t username/moss-ttsd-runpod:latest .
# Push: docker push username/moss-ttsd-runpod:latest

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to 2.9.1 with CUDA 12.6 (compatible with RunPod Ampere fleet drivers)
# Must also upgrade torchvision — base image ships torchvision for PyTorch 2.5.1
# which crashes on operator registration with torch 2.9.1
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchaudio==2.9.1 \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Install transformers 5.x (required by MOSS-TTSD)
RUN pip install --no-cache-dir "transformers>=5.0.0"

# Install FlashAttention2 for faster generation and lower VRAM
RUN pip install flash-attn --no-build-isolation || echo "FlashAttention2 install failed, will use SDPA"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RunPod execution timeout (10 minutes)
ENV RUNPOD_EXECUTION_TIMEOUT_MS=600000

# Copy handler
COPY handler.py .

# Pre-download model weights during build for fast cold starts
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading MOSS-TTSD v1.0...'); \
snapshot_download('OpenMOSS-Team/MOSS-TTSD-v1.0'); \
print('Downloading MOSS-Audio-Tokenizer...'); \
snapshot_download('OpenMOSS-Team/MOSS-Audio-Tokenizer'); \
print('Models cached.')"

# Start handler
CMD ["python", "-u", "handler.py"]
