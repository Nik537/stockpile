#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building Stockpile Desktop App ==="

cd "$PROJECT_ROOT"

# Step 1: Build Python sidecar
echo "Step 1/3: Building Python sidecar..."
"$SCRIPT_DIR/build-sidecar.sh"

# Step 2: Install web dependencies
echo "Step 2/3: Installing web dependencies..."
cd web
npm install

# Step 3: Build Tauri app
echo "Step 3/3: Building Tauri app..."
npm run tauri:build

echo "=== Build complete! ==="
echo "App bundle: web/src-tauri/target/release/bundle/"
