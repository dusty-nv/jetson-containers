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

# Construct the GitHub release URL dynamically
WHL_URL="https://github.com/cnr-isti-vclab/PyMeshLab/releases/download/v${PYMESHLAB_VERSION}/pymeshlab-${PYMESHLAB_VERSION}-cp310-cp310-manylinux_2_35_aarch64.whl"

# Install the .whl file directly from the constructed URL
pip3 install --no-cache-dir --verbose "${WHL_URL}"

python3 -c "import pymeshlab; ms = pymeshlab.MeshSet()"