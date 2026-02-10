"""Minimal Render startup to test if basic FastAPI works."""
import os
import sys

port = int(os.environ.get("PORT", "10000"))
print(f"[start.py] Python {sys.version}", flush=True)
print(f"[start.py] CWD: {os.getcwd()}", flush=True)
print(f"[start.py] PORT: {port}", flush=True)

# Phase 1: Test minimal app
from fastapi import FastAPI
import uvicorn

test_app = FastAPI()

@test_app.get("/")
async def root():
    return {"message": "hello"}

@test_app.get("/api/health")
async def health():
    return {"status": "healthy"}

print(f"[start.py] Starting minimal test server on port {port}", flush=True)
uvicorn.run(test_app, host="0.0.0.0", port=port, log_level="info")
