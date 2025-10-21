#!/usr/bin/env bash
set -ex

echo "Building tinycudann ${TINYCUDANN_VERSION}"
cd /opt/tinycudann

# Configure source tree with cmake
export TCNN_CUDA_ARCHITECTURES=${CUDAARCHS}
export MAX_JOBS=$(nproc)

cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -- -j$(nproc)

cd bindings/torch

# Build and install python wheels
uv build --wheel --no-build-isolation . --out-dir $PIP_WHEEL_DIR --verbose
uv pip install $PIP_WHEEL_DIR/tinycudann*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/tinycudann*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
