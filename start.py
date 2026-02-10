"""Render startup wrapper - helps debug import errors."""
import os
import sys
import traceback

try:
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    print(f"Starting server on port {port}", flush=True)
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=port, log_level="info")
except Exception as e:
    print(f"STARTUP ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
