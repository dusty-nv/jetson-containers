#!/bin/bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit

# Default port if not set
PORT=${PORT:-8000}

# Change to the speaches directory
cd /workspace/speaches

# Activate the virtual environment
source .venv/bin/activate

# Check if additional arguments were provided
if [ "$#" -gt 0 ]; then
    # Run the server in the background and execute the provided command
    uvicorn --factory --host 0.0.0.0 --port ${PORT} speaches.main:create_app 2>&1 &
    echo ""
    sleep 5  # Give the server time to start
    echo "Running command:  $@"
    echo ""
    sleep 1
    "$@"
else
    # Run the server in the foreground
    exec uvicorn --factory --host 0.0.0.0 --port ${PORT} speaches.main:create_app
fi
