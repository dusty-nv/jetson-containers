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

# Detect aarch64/x86_64 tags
if [ "$(uname -m)" == "aarch64" ]; then
    PLATFORM_TAG="manylinux_2_35_aarch64"
else
    PLATFORM_TAG="manylinux_2_31_x86_64"
fi

# Construct the GitHub release URL dynamically using the detected tags
WHL_URL="https://github.com/cnr-isti-vclab/PyMeshLab/releases/download/v${PYMESHLAB_VERSION}/pymeshlab-${PYMESHLAB_VERSION}-${PYTHON_TAG}-${PYTHON_TAG}-${PLATFORM_TAG}.whl"
echo "Using wheel URL: ${WHL_URL}"

# Install the .whl file directly from the constructed URL
uv pip install "${WHL_URL}" || \
uv pip install pymeshlab==${PYMESHLAB_VERSION}

# Test the installation
python3 -c "import pymeshlab; ms = pymeshlab.MeshSet()"
