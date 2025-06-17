#!/bin/bash
# Flexible entrypoint that either:
# 1. Executes passed commands (with environment properly set up)
# 2. Starts the speaches server (default behavior)

# https://stackoverflow.com/a/4319666
shopt -s huponexit

# Default port if not set
PORT=${PORT:-8000}

# Source the virtual environment
source /workspace/speaches/.venv/bin/activate

# If arguments were passed, execute them )
if [ $# -gt 0 ]; then
    exec "$@"
fi

# Run the server
cd /workspace/speaches
exec uvicorn --factory --host 0.0.0.0 --port ${PORT} speaches.main:create_app
