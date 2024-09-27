#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${MAMBAVISION_VERSION} --depth=1 --recursive https://github.com/johnnynunez/MambaVision opt/mambavision || \
git clone --depth=1 --recursive https://github.com/johnnynunez/MambaVision /opt/mambavision

# Navigate to the directory containing mamba's setup.py
cd /opt/mambavision 
pip3 install -U einops timm 
pip3 wheel --no-build-isolation --no-deps --wheel-dir=/opt/mambavision/wheels .
pip3 install --no-cache-dir --verbose /opt/mambavision/wheels/mambavision*.whl

cd /opt/mambavision
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mambavision/wheels/mambavision*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
