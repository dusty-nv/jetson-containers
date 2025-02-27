#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${DECORD_VERSION} --depth=1 --recursive https://github.com/johnnynunez/decord /opt/decord || \
git clone --depth=1 --recursive https://github.com/johnnynunez/decord /opt/decord

cd /opt/decord
mkdir build && cd build
CUDA_HOME="/usr/local/cuda" \
NVCC_PATH="${CUDA_HOME}/bin/nvcc" \
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make

cd ../python

pip3 wheel --no-build-isolation --wheel-dir=/opt/decord/wheels .
pip3 install --no-cache-dir --verbose /opt/decord/wheels/decord*.whl

cd /opt/decord


# Optionally upload to a repository using Twine
twine upload --verbose /opt/decord/wheels/decord*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
