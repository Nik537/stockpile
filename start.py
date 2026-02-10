"""Render startup - imports the real app with error handling."""
import os
import sys
import traceback

port = int(os.environ.get("PORT", "10000"))

# Try to import the real app
try:
    from src.api.server import app
    print("[start.py] Real app imported successfully", flush=True)
except Exception as e:
    print(f"[start.py] IMPORT FAILED: {e}", flush=True)
    traceback.print_exc()
    # Fallback to minimal app so we can see logs
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"error": str(e)}

    @app.get("/api/health")
    async def health():
        return {"status": "unhealthy", "error": str(e)}

import uvicorn
print(f"[start.py] Starting on port {port}", flush=True)
uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
