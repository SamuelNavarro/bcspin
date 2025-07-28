#!/bin/bash
set -e

# Use PORT environment variable if set, otherwise default to 8000
PORT=${PORT:-8000}

echo "Starting Sproxxo API on port $PORT"
echo "Working directory: $(pwd)"
echo "Contents of /app:"
ls -la /app/

echo "Checking for model files:"
if [ -d "/app/artifacts" ]; then
    echo "artifacts/ directory exists:"
    ls -la /app/artifacts/
else
    echo "artifacts/ directory not found"
fi

echo "Environment variables:"
echo "PORT=$PORT"
echo "ENVIRONMENT=${ENVIRONMENT:-not_set}"

# Start the application
echo "Starting uvicorn server..."
exec uv run uvicorn sproxxo.api.main:app --host 0.0.0.0 --port $PORT --log-level info
