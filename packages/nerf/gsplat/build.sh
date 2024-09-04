#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${GSPLAT_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/gsplat /opt/gsplat || \
git clone --depth=1 --recursive https://github.com/nerfstudio-project/gsplat /opt/gsplat

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/gsplat

export BUILD_NO_CUDA=0
export WITH_SYMBOLS=0
export LINE_INFO=1
export MAX_JOBS=$(nproc)
pip3 wheel . -w /opt/gsplat/wheels

pip3 install --no-cache-dir --verbose /opt/hloc/gsplat/gsplat*.whl

cd /opt/gsplat
pip3 install 'numpy<2'
# Optionally upload to a repository using Twine
twine upload --verbose /opt/gsplat/wheels/gsplat*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
