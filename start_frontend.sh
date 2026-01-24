#!/bin/bash
# Start the Stockpile React frontend

set -e  # Exit on error

echo "🎨 Starting Stockpile Frontend..."
echo ""

# Check if node_modules exists
if [[ ! -d "web/node_modules" ]]; then
    echo "📦 Installing frontend dependencies..."
    cd web
    npm install
    cd ..
    echo "✅ Dependencies installed"
    echo ""
fi

echo "Frontend will be available at:"
echo "  🌐 Web UI: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"
echo ""

# Start the dev server
cd web
npm run dev
