#!/usr/bin/env bash
set -ex

# Install dependencies of XGrammar
uv pip install pre-commit nanobind==2.5.0

# Clone the repository if it doesn't exist
git clone --branch=v${XGRAMMAR_VERSION} --recursive --depth=1 https://github.com/mlc-ai/xgrammar /opt/xgrammar ||
git clone --recursive --depth=1 https://github.com/mlc-ai/xgrammar /opt/xgrammar

# Create the wheel
export MAX_JOBS=$(nproc) # this is for AGX (max 4 working on Orin NX)
export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
cd /opt/xgrammar
uv build --wheel . --out-dir /opt/xgrammar/wheels/ --verbose

# Install the wheel
# Warning: version number is 0.1.5 even if actual version is 0.1.8, or 0.1.9 due to version.py not being adapted yet: https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/version.py
uv pip install /opt/xgrammar/wheels/xgrammar*.whl
uv pip install compressed-tensors

# Optionally upload to a repository using Twine
twine upload --verbose /opt/xgrammar/wheels/xgrammar*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
