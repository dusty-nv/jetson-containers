#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${MAMBAVISION_VERSION} --depth=1 --recursive https://github.com/johnnynunez/MambaVision opt/mambavision || \
git clone --depth=1 --recursive https://github.com/johnnynunez/MambaVision /opt/mambavision

# Navigate to the directory containing mamba's setup.py
cd /opt/mambavision
uv pip install -U einops timm
uv build --wheel --no-build-isolation --no-deps --out-dir /opt/mambavision/wheels .
uv pip install /opt/mambavision/wheels/mambavision*.whl

cd /opt/mambavision

# Optionally upload to a repository using Twine
twine upload --verbose /opt/mambavision/wheels/mambavision*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
