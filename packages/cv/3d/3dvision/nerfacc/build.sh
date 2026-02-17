#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NERFACC_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfacc /opt/nerfacc || \
git clone --depth=1 --recursive https://github.com/nerfstudio-project/nerfacc /opt/nerfacc

cd /opt/nerfacc

export BUILD_NO_CUDA=0
export WITH_SYMBOLS=0
export LINE_INFO=1
export MAX_JOBS=$(nproc)

# Build and install python wheel
uv build --wheel --no-build-isolation . --out-dir $PIP_WHEEL_DIR --verbose
uv pip install lpips scipy
uv pip install $PIP_WHEEL_DIR/nerfacc*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/nerfacc*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
