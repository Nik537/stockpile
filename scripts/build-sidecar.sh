#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building Stockpile Backend Sidecar ==="

cd "$PROJECT_ROOT"

# Determine target triple
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    TARGET_TRIPLE="aarch64-apple-darwin"
elif [ "$ARCH" = "x86_64" ]; then
    TARGET_TRIPLE="x86_64-apple-darwin"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Build with PyInstaller
echo "Building sidecar for $TARGET_TRIPLE..."
source .venv/bin/activate 2>/dev/null || true

pyinstaller stockpile-backend.spec --noconfirm --clean

# Copy to Tauri binaries directory
BINARIES_DIR="web/src-tauri/binaries"
mkdir -p "$BINARIES_DIR"

SIDECAR_NAME="stockpile-backend-${TARGET_TRIPLE}"
cp "dist/stockpile-backend" "$BINARIES_DIR/$SIDECAR_NAME"
chmod +x "$BINARIES_DIR/$SIDECAR_NAME"

echo "=== Sidecar built: $BINARIES_DIR/$SIDECAR_NAME ==="
