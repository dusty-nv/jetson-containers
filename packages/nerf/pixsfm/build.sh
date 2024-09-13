#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${PIXSFM_VERSION} --depth=1 --recursive https://github.com/cvg/pixel-perfect-sfm /opt/pixsfm || \
git clone --depth=1 --recursive https://github.com/cvg/pixel-perfect-sfm /opt/pixsfm

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/pixsfm && \
pip3 wheel . -w /opt/pixsfm/wheels

# Verify the contents of the /opt directory
ls /opt/pixsfm/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/pixsfm/wheels/pixsfm*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/pixsfm/wheels/pixsfm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
