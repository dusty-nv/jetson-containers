#!/bin/bash
# Script to download  models using speaches-cli

set -e 

echo "=== Downloading models for speaches ==="

cd /workspace/speaches

echo "Activating virtual environment..."
source .venv/bin/activate

# Models to download
MODELS=(
    "Systran/faster-whisper-large-v3"
    "speaches-ai/Kokoro-82M-v1.0-ONNX-fp16"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "Downloading model: $model"
    echo "Running: uvx speaches-cli model download $model"
    
    if uvx speaches-cli model download "$model"; then
        echo "✓ Successfully downloaded $model"
    else
        echo "✗ Failed to download $model"
        exit 1
    fi
done

echo ""
echo "=== All models downloaded successfully ✓ ===" 