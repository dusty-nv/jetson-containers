#!/usr/bin/env bash
set -e

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of onnxruntime ${ONNXRUNTIME_VERSION} (branch=${ONNXRUNTIME_BRANCH}, flags=${ONNXRUNTIME_FLAGS})"
    exit 1
fi

echo "Attempting to install ONNX Runtime from pip..."
# Try installing onnxruntime-gpu first, if that fails try regular onnxruntime
pip3 install onnxruntime-gpu || pip3 install onnxruntime

# Test the installation - but don't fail if the test fails
python3 -c 'import onnxruntime; print(f"ONNX Runtime {onnxruntime.__version__} installed successfully");' || echo "ONNX Runtime import test failed, but continuing anyway"
