#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${DECORD_VERSION} --depth=1 --recursive https://github.com/johnnynunez/decord2 /opt/decord || \
git clone --depth=1 --recursive https://github.com/johnnynunez/decord2 /opt/decord

cd /opt/decord
mkdir build && cd build
CUDA_HOME="/usr/local/cuda" \
NVCC_PATH="${CUDA_HOME}/bin/nvcc" \
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
-DUSE_CUDA=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_ARCHITECTURES="80;90;100;110;120"
make -j$(nproc)
cp libdecord.so ..

cd ../python

uv build --wheel --no-build-isolation --out-dir /opt/decord/wheels .
uv pip install /opt/decord/wheels/decord*.whl
cd /opt/decord

# Optionally upload to a repository using Twine
twine upload --verbose /opt/decord/wheels/decord*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
