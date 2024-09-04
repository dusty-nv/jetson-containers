#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${NERFACC_VERSION} --depth=1 --recursive https://github.com/nerfstudio-project/nerfacc /opt/nerfacc || \
git clone --depth=1 --recursive https://github.com/nerfstudio-project/nerfacc /opt/nerfacc

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/nerfacc

export BUILD_NO_CUDA=0
export WITH_SYMBOLS=0
export LINE_INFO=1
export MAX_JOBS=$(nproc)
pip3 wheel . -w /opt/nerfacc/wheels

pip3 install lpips scipy
pip3 install --no-cache-dir --verbose /opt/nerfacc/wheels/nerfacc*.whl

cd /opt/nerfacc
pip3 install 'numpy<2'
# Optionally upload to a repository using Twine
twine upload --verbose /opt/nerfacc/wheels/nerfacc*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
