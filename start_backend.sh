#!/bin/bash
# Start the Stockpile FastAPI backend server

set -e  # Exit on error

echo "🚀 Starting Stockpile Backend API..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    echo ""
    exit 1
fi

# Check if .env file exists
if [[ ! -f .env ]]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys"
    echo ""
    exit 1
fi

# Check for required dependencies
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not installed!"
    echo "Installing backend dependencies..."
    pip install -q -r requirements.txt
    echo "✅ Dependencies installed"
    echo ""
fi

# Create uploads directory if it doesn't exist
mkdir -p uploads

echo "Backend will be available at:"
echo "  📡 API: http://localhost:8000"
echo "  📊 Docs: http://localhost:8000/docs"
echo "  ❤️  Health: http://localhost:8000/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"
echo ""

# Start the server
cd "$(dirname "$0")"
python src/api/server.py
