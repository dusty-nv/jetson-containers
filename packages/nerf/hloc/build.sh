#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone --branch=${HLOC_VERSION} --depth=1 --recursive https://github.com/cvg/Hierarchical-Localization /opt/hloc ||
git clone --depth=1 --recursive https://github.com/cvg/Hierarchical-Localization /opt/hloc

cd /opt/hloc

# Unpin some dependencies that are already installed
sed -i '/pycolmap/d' requirements.txt
sed -i '/opencv-python/d' requirements.txt

# Build & install the HLOC python wheel
pip3 wheel . -w $PIP_WHEEL_DIR --verbose
pip3 install $PIP_WHEEL_DIR/hloc*.whl

# Optionally upload to a repository using Twine
twine upload --verbose $PIP_WHEEL_DIR/hloc*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL}"
