#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${MAMBA_VISION} --depth=1 --recursive https://github.com/NVlabs/MambaVision opt/mambavision || \
git clone --depth=1 --recursive https://github.com/NVlabs/MambaVision /opt/mambavision

# Navigate to the directory containing mamba's setup.py
cd /opt/mambavision 
pip wheel --no-build-isolation --wheel-dir=/wheels .
pip3 install --no-cache-dir --verbose /wheels/mambavision*.whl

cd /opt/mamba

pip3 install -e .
pip3 install 'numpy<2'

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mambavision/wheels/mambavision*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
