#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${FAST_GAUSIAN_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/gsplat /opt/fast_gaussian_rasterization || \
git clone --depth=1 --recursive https://github.com/nerfstudio-project/gsplat /opt/fast_gaussian_rasterization

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/fast_gaussian_rasterization

export BUILD_NO_CUDA=0
export WITH_SYMBOLS=0
export LINE_INFO=1
MAX_JOBS=$(nproc) \
pip3 wheel . -w /opt/gsplat/wheels --verbose

pip3 install --no-cache-dir --verbose /opt/gsplat/wheels/gsplat*.whl

cd /opt/gsplat

# Optionally upload to a repository using Twine
twine upload --verbose /opt/gsplat/wheels/gsplat*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
