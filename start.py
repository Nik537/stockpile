"""Render startup - imports the real app with error handling."""
import os
import sys
import traceback

port = int(os.environ.get("PORT", "10000"))
import_error = None

# Try to import the real app
try:
    from src.api.server import app
    print("[start.py] Real app imported successfully", flush=True)
except Exception as e:
    import_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(f"[start.py] IMPORT FAILED: {import_error}", flush=True)
    # Fallback to minimal app that shows the error
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    app = FastAPI()
    err_msg = import_error  # capture in closure

    @app.get("/", response_class=PlainTextResponse)
    async def root():
        return f"Import error:\n{err_msg}"

    @app.get("/api/health", response_class=PlainTextResponse)
    async def health():
        return f"UNHEALTHY - Import error:\n{err_msg}"

import uvicorn
print(f"[start.py] Starting on port {port}", flush=True)
uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
