#!/usr/bin/env python
"""FastAPI server for stockpile web interface.

This module provides the main FastAPI application factory with all routers
included. The server has been refactored from a monolithic 2000-line file
into a modular router structure for better maintainability.

Routers:
    - core: Root and health check endpoints
    - broll: B-roll video processing endpoints
    - outliers: Outlier finder endpoints
    - tts: Text-to-speech endpoints
    - images: Image generation endpoints (Runware, Gemini, Nano Banana Pro)
    - bulk_images: Bulk image generation endpoints
"""

import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
# This must happen before importing any local modules
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from api.routers import broll, bulk_images, core, dataset_generator, images, music, outliers, tts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.config import load_config, validate_config

# Configure logging to output to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Media Multi Tool API",
    description="AI-Powered Content Creation Suite",
    version="1.0.0",
)

# CORS middleware
cors_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "tauri://localhost",
    "https://tauri.localhost",
]
extra_origin = os.environ.get("CORS_ORIGIN")
if extra_origin:
    cors_origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(core.router)
app.include_router(broll.router)
app.include_router(outliers.router)
app.include_router(tts.router)
app.include_router(images.router)
app.include_router(bulk_images.router)
app.include_router(music.router)
app.include_router(dataset_generator.router)


@app.on_event("startup")
async def startup_validation():
    """Validate configuration on startup."""
    config = load_config()
    errors = validate_config(config)
    if errors:
        logger.error(f"Configuration errors: {'; '.join(errors)}")
        # Don't crash - some routes may still work
    else:
        logger.info("Configuration validated successfully")


def create_app() -> FastAPI:
    """Application factory for uvicorn --factory mode."""
    return app


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "src.api.server:create_app",
        factory=True,
        host="127.0.0.1",
        port=port,
        reload=True,
    )
