#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${DECORD_VERSION} --depth=1 --recursive https://github.com/dmlc/decord /opt/decord || \
git clone --depth=1 --recursive https://github.com/dmlc/decord /opt/decord

cd /opt/decord
export CUDA_HOME="/usr/local/cuda"
export NVCC_PATH="${CUDA_HOME}/bin/nvcc"
# -DCUDAToolkit_ROOT=$NVCC_PATH

mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make

cd ../python

pip3 wheel --no-build-isolation --wheel-dir=/opt/decord/wheels .
pip3 install --no-cache-dir --verbose /opt/decord/wheels/decord*.whl

cd /opt/decord


# Optionally upload to a repository using Twine
twine upload --verbose /opt/decord/wheels/decord*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
