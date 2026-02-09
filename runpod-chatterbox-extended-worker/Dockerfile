# RunPod Serverless Worker for Chatterbox TTS Extended
# Uses python:3.10-slim base (proven compatible with RunPod SDK)
# PyTorch CUDA support installed via pip (includes bundled CUDA runtime)
#
# Build: docker build --platform linux/amd64 -t techtawn/chatterbox-extended-runpod:v7 .
# Push: docker push techtawn/chatterbox-extended-runpod:v7

FROM python:3.10-slim

WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies (ffmpeg needed for denoising pipeline, git for cloning repo)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.7.0 with CUDA 12.8 support (pip version, not conda)
# This bundles the CUDA runtime - no NVIDIA base image needed
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Clone Chatterbox-TTS-Extended repo (not pip-installable, must be on PYTHONPATH)
RUN git clone --depth 1 https://github.com/petermg/Chatterbox-TTS-Extended.git /app/chatterbox-extended

# Add repo root to PYTHONPATH so "from chatterbox.src.chatterbox.tts import ..." works
ENV PYTHONPATH="/app/chatterbox-extended"

# RunPod execution timeout (10 minutes) to prevent premature job kills
ENV RUNPOD_EXECUTION_TIMEOUT_MS=600000

# Install only the deps that the handler actually needs from the repo's requirements
# (skip torch/torchaudio already installed, skip gradio/whisper/diffusers/etc not needed)
RUN pip install --no-cache-dir \
    omegaconf==2.3.0 \
    resemble-perth==1.0.1 \
    silero-vad==5.1.2 \
    conformer==0.3.2 \
    transformers==4.46.3 \
    resampy==0.4.3 \
    librosa==0.10.0 \
    s3tokenizer \
    diffusers==0.29.0

# Install additional dependencies for our worker
RUN pip install --no-cache-dir \
    pyrnnoise>=0.3.8 \
    faster-whisper \
    soundfile \
    numpy \
    nltk \
    Brotli

# Install RunPod SDK LAST
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt tokenizer data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy handler
COPY handler.py .

# Pre-download Chatterbox TTS model during build (faster cold starts)
RUN python -c "\
from chatterbox.src.chatterbox.tts import ChatterboxTTS; \
print('Downloading TTS model...'); \
ChatterboxTTS.from_pretrained(device='cpu'); \
print('TTS model cached.')"

# Pre-download faster-whisper base model during build
RUN python -c "\
from faster_whisper import WhisperModel; \
print('Downloading Whisper base model...'); \
WhisperModel('base', device='cpu', compute_type='int8'); \
print('Whisper model cached.')"

# Start handler
CMD ["python", "-u", "handler.py"]
