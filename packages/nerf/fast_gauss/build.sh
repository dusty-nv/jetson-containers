#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${FAST_GAUSIAN_VERSION} --depth=1 --recursive https://github.com/dendenxu/fast-gaussian-rasterization /opt/fast_gauss || \
git clone --depth=1 --recursive https://github.com/dendenxu/fast-gaussian-rasterization /opt/fast_gauss

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/fast_gauss
pip3 install PyOpenGL PyOpenGL_accelerate pdbr tqdm ujson ruamel.yaml
MAX_JOBS=$(nproc) \
pip3 wheel . -w /opt/wheels --verbose


pip3 install --no-cache-dir --verbose /opt/wheels/fast_gauss*.whl

cd /opt/fast_gauss

# Optionally upload to a repository using Twine
twine upload --verbose /opt/wheels/fast_gauss*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
