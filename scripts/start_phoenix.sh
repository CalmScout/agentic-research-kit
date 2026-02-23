#!/bin/bash
# Start Phoenix observability server for multi-agent RAG tracing

set -e

echo "🔥 Starting Phoenix Observability Server..."
echo ""

# Default port
PORT=${PHOENIX_PORT:-6006}

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT is already in use!"
    echo "   Killing existing Phoenix process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start Phoenix in background
echo "🚀 Starting Phoenix on port $PORT..."
echo "   UI will be available at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop Phoenix"
echo ""

# Start Phoenix
python -m phoenix.server.main serve --port $PORT

# If killed, cleanup
trap "echo '🛑 Phoenix stopped'; exit" INT TERM
