#!/usr/bin/env bash
set -ex

if [ -z "$PYMESHLAB_VERSION" ]; then
    echo "Error: PYMESHLAB_VERSION is not set."
    exit 1
fi

if [ "$FORCE_BUILD" == "on" ]; then
    echo "Forcing build of pymeshlab ${PYMESHLAB_VERSION}"
    exit 1
fi

# Detect Python version and create a tag (e.g., cp310, cp311)
PYTHON_TAG=$(python3 -c "import sys; print('cp{}{}'.format(sys.version_info.major, sys.version_info.minor))")
echo "Detected Python tag: ${PYTHON_TAG}"

# Construct the GitHub release URL dynamically using the detected Python tag
WHL_URL="https://github.com/cnr-isti-vclab/PyMeshLab/releases/download/v${PYMESHLAB_VERSION}/pymeshlab-${PYMESHLAB_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-manylinux_2_35_aarch64.whl"
echo "Using wheel URL: ${WHL_URL}"

# Install the .whl file directly from the constructed URL
pip3 install "${WHL_URL}"

# Test the installation
python3 -c "import pymeshlab; ms = pymeshlab.MeshSet()"
