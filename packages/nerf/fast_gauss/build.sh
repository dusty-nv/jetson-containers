#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${FAST_GAUSIAN_VERSION} --depth=1 --recursive https://github.com/dendenxu/fast-gaussian-rasterization /opt/fast_gauss || \
git clone --depth=1 --recursive https://github.com/dendenxu/fast-gaussian-rasterization /opt/fast_gauss

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/fast_gauss
pip3 install PyOpenGL PyOpenGL_accelerate pdbr tqdm ujson ruamel.yaml
MAX_JOBS=$(nproc) \
pip3 wheel . -w $PIP_WHEEL_DIR --verbose


pip3 install $PIP_WHEEL_DIR/fast_gauss*.whl

cd /opt/fast_gauss

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/fast_gauss*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
