"""Render startup script."""
import os
import uvicorn
from src.api.server import app

port = int(os.environ.get("PORT", "10000"))
uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
