#!/usr/bin/env bash
set -x

echo "TESTING OLLAMA"

# Function to find an available port by trying to bind to it
find_available_port() {
    local start_port=$1
    local port=$start_port

    while [ $port -lt 65535 ]; do
        # Try to bind to the port using a temporary netcat process
        if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
            # Port is in use, try next
            port=$((port + 1))
        else
            # Port is available
            echo $port
            return 0
        fi
    done

    echo "No available ports found" >&2
    return 1
}

# Detect CUDA version dynamically
if [ -n "$CUDA_VERSION_MAJOR" ]; then
    CUDA_MAJOR="$CUDA_VERSION_MAJOR"
else
    # Try to detect CUDA version from nvidia-smi or nvcc
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | cut -d. -f1)
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_MAJOR=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\).*/\1/')
    else
        # Default fallback - this should be set by the container environment
        echo "Warning: CUDA version not detected, using default"
        CUDA_MAJOR=13
    fi
fi

echo "Detected CUDA version: $CUDA_MAJOR"

# stop any existing ollama processes (container-safe approach)
OLLAMA_PID=$(ps -ef | grep 'ollama serve' | grep -v grep | awk '{ print $2 }')

if [ -n "$OLLAMA_PID" ]; then
    echo "Stopping existing ollama process: $OLLAMA_PID"
    kill $OLLAMA_PID
    sleep 2
fi

# Find an available port starting from 11435
TEST_PORT=$(find_available_port 11435)
if [ $? -ne 0 ]; then
    echo "Failed to find available port. Exiting."
    exit 1
fi

echo "Using port $TEST_PORT for testing"

# start ollama using the detected CUDA version on the available port
OLLAMA_LLM_LIBRARY=cuda_v${CUDA_MAJOR} OLLAMA_DEBUG=1 OLLAMA_HOST=127.0.0.1:${TEST_PORT} /bin/ollama serve &
OLLAMA_PID=$!

# wait a moment for ollama to start
sleep 3

# verify ollama is running
if ! ps -p $OLLAMA_PID > /dev/null; then
    echo "ollama binary not running. exiting"
    exit 1
fi

echo "Ollama started with PID: $OLLAMA_PID on port $TEST_PORT using CUDA v${CUDA_MAJOR}"

# set the OLLAMA_HOST environment variable for the test
export OLLAMA_HOST=127.0.0.1:${TEST_PORT}

# run the test
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python3 $SCRIPT_DIR/test.py

# cleanup
kill $OLLAMA_PID
