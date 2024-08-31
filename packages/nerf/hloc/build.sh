#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${HLOC_VERSION} --depth=1 --recursive https://github.com/cvg/Hierarchical-Localization /opt/hloc || \
git clone --depth=1 --recursive https://github.com/cvg/Hierarchical-Localization /opt/hloc

# Navigate to the directory containing PyMeshLab's setup.py
cd /opt/hloc && \
pip3 wheel . -w /opt/hloc/wheels

# Verify the contents of the /opt directory
ls /opt/hloc/wheels

# Return to the root directory
cd /

pip3 install --no-cache-dir --verbose /opt/hloc/wheels/hloc*.whl

# Optionally upload to a repository using Twine
twine upload --verbose /opt/hloc/wheels/hloc*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
