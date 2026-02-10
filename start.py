"""Render startup wrapper with import error debugging."""
import os
import sys
import traceback

port = int(os.environ.get("PORT", "10000"))
print(f"[start.py] Python {sys.version}", flush=True)
print(f"[start.py] CWD: {os.getcwd()}", flush=True)
print(f"[start.py] PORT: {port}", flush=True)
print(f"[start.py] sys.path: {sys.path[:5]}", flush=True)

# Test imports one by one to find the failure
imports_to_test = [
    "fastapi", "uvicorn", "dotenv", "rich", "httpx",
    "google.genai", "numpy", "yt_dlp", "PIL", "praw", "boto3", "yaml",
]
for mod in imports_to_test:
    try:
        __import__(mod)
        print(f"[start.py] OK: {mod}", flush=True)
    except ImportError as e:
        print(f"[start.py] FAIL: {mod} -> {e}", flush=True)

# Try importing the app
try:
    print("[start.py] Importing src.api.server...", flush=True)
    from src.api.server import app  # noqa: F401
    print("[start.py] App imported successfully!", flush=True)
except Exception as e:
    print(f"[start.py] IMPORT ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Start the server
try:
    import uvicorn
    print(f"[start.py] Starting uvicorn on 0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
except Exception as e:
    print(f"[start.py] RUNTIME ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
