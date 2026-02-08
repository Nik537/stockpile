# =============================================================================
# Stockpile - Multi-stage Docker Build
# =============================================================================

# Build argument for optional CLIP/ML dependencies
ARG INSTALL_CLIP=false

# -----------------------------------------------------------------------------
# Stage 1: Builder - install dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optionally install CLIP/ML dependencies
ARG INSTALL_CLIP
RUN if [ "$INSTALL_CLIP" = "true" ]; then \
    pip install --no-cache-dir torch torchvision transformers opencv-python; \
    fi

# -----------------------------------------------------------------------------
# Stage 2: Runtime - slim production image
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd --create-home --shell /bin/bash stockpile

WORKDIR /app

# Copy application code
COPY --chown=stockpile:stockpile . .

# Create necessary directories with correct ownership
RUN mkdir -p input output uploads && chown -R stockpile:stockpile input output uploads

# Switch to non-root user
USER stockpile

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
