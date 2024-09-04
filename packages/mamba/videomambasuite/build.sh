#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=v${VIDEOMAMBASUITE_VERSION} --depth=1 --recursive https://github.com/OpenGVLab/video-mamba-suite /opt/videomambasuite || \
git clone --depth=1 --recursive https://github.com/OpenGVLab/video-mamba-suite /opt/videomambasuite

# Navigate to the directory containing mamba's setup.py
cd /opt/videomambasuite
pip wheel --no-build-isolation --wheel-dir=/wheels .
pip3 install --no-cache-dir --verbose /opt/videomambasuite /wheels/videomamba*.whl

cd /opt/mamba

pip3 install -e .
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/pycolmap/wheels/videomamba*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
