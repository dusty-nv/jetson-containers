#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
GSPLAT_SRC="${GSPLAT_SRC:-/opt/gsplat}"

git clone --branch=v${GSPLAT_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/gsplat $GSPLAT_SRC || \
git clone --depth=1 --recursive https://github.com/nerfstudio-project/gsplat $GSPLAT_SRC

cd $GSPLAT_SRC

# build python wheel from source
export BUILD_NO_CUDA=0
export WITH_SYMBOLS=0
export LINE_INFO=1
export MAX_JOBS=$(nproc)

uv build --wheel --no-build-isolation . --out-dir $PIP_WHEEL_DIR --verbose
uv pip install $PIP_WHEEL_DIR/gsplat*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/gsplat*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
