#!/bin/bash
# https://stackoverflow.com/a/4319666
shopt -s huponexit

set -e

# Load environment
if [ -f "$HOME/.local/bin/env" ]; then
    source "$HOME/.local/bin/env"
fi

cd /opt/speaches

echo "===== Using global packages directly with Python's site-packages ====="

# Skip creating a virtual environment and use the global Python directly
echo "Using global Python installation with all system packages"
echo "Python info: $(which python3) - $(python3 --version)"

# Verify global packages are available
echo "Verifying required packages..."
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo "Warning: NumPy not available"
python3 -c "import ctranslate2; print(f'ctranslate2 version: {ctranslate2.__version__}')" || echo "Warning: ctranslate2 not available"
python3 -c "import faster_whisper; print(f'faster-whisper version: {faster_whisper.__version__}')" || echo "Warning: faster-whisper not available"

# Modify speaches configuration to not require faster-whisper as a dependency
echo "Modifying speaches configuration..."
sed -i 's|"faster-whisper>=1.1.1",||g' pyproject.toml
sed -i 's|"ctranslate2>=4.5.0",|"ctranslate2",|g' pyproject.toml
sed -i 's|enable_ui: bool = True|enable_ui: bool = False|g' src/speaches/config.py

cat pyproject.toml

# Print PYTHONPATH for debugging
echo "Current PYTHONPATH: $PYTHONPATH"
python3 -c "import sys; print('Python search paths:'); [print(f'  {p}') for p in sys.path]"

# Install speaches directly with pip (not uv) to avoid env isolation issues
echo "Installing speaches in development mode with pip..."
pip3 install -e .

# Install Gradio dependencies with pip
echo "Installing Gradio dependencies with pip..."
pip3 install gradio==5.13.0 gradio-client==1.6.0

# Verify Gradio installation
python3 -c "import gradio; print(f'Gradio version: {gradio.__version__}')"

# Start the server
echo "Starting the speaches server..."
#python3 -m uvicorn speaches.main:create_app --host 0.0.0.0 --factory

# Get the port from environment variable or use default
PORT=${SERVER_PORT:-8000}
echo "Using port: $PORT"


SPEACHES_DEFAULT_CMD="python3 -m uvicorn speaches.main:create_app --host 0.0.0.0 --port $PORT --factory"
SPEACHES_STARTUP_LAG=1

printf "Starting Speaches STT server:\n\n"
printf "  ${SPEACHES_DEFAULT_CMD}\n\n"

if [ "$#" -gt 0 ]; then
    ${SPEACHES_DEFAULT_CMD} &
    #echo "Letting server load for ${SPEACHES_STARTUP_LAG} seconds..."
    echo ""
    sleep ${SPEACHES_STARTUP_LAG}
    echo ""
    echo "Running command:  $@"
    echo ""
    sleep 1
    "$@"
else
    ${SPEACHES_DEFAULT_CMD}
fi
