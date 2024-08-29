#!/usr/bin/env bash 
set -ex

echo "Setting up environment for tinycudann ${TINYCUDANN_VERSION}"

echo "Building tinycudann ${TINYCUDANN_VERSION}"

cd /opt/tinycudann

export TCNN_CUDA_ARCHITECTURES=${CUDAARCHS}
export MAX_JOBS=$(nproc)

cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -- -j$(nproc)

cd bindings/torch
pip3 wheel . -w /opt/tinycudann/wheels
pip3 install --no-cache-dir --verbose /opt/tinycudann/wheels/tinycudann*.whl

