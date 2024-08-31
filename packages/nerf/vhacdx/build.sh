#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${VHACDX_VERSION} --depth=1 --recursive https://github.com/trimesh/vhacdx /opt/vhacdx || \
git clone --depth=1 --recursive https://github.com/trimesh/vhacdx /opt/vhacdx

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/vhacdx && \
pip3 wheel . -w /opt/vhacdx/wheels

# Verify the contents of the /opt directory
ls /opt/vhacdx/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/vhacdx/wheels/vhacdx*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/vhacdx/wheels/vhacdx*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
