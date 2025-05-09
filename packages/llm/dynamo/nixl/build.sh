#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NIXL_VERSION} --depth=1 --recursive https://github.com/ai-dynamo/nixl /opt/nixl || \
git clone --depth=1 --recursive https://github.com/ai-dynamo/nixl /opt/nixl

# Navigate to the directory containing nixl's setup.py
cd /opt/nixl

pip3 install meson ninja pybind11
export MAX_JOBS=$(nproc)
pip3 wheel --no-build-isolation --wheel-dir=/opt/nixl/wheels . --verbose
pip3 install /opt/nixl/wheels/nixl*.whl

cd /opt/nixl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/nixl/wheels/nixl*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
