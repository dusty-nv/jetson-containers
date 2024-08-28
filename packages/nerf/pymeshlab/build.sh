#!/usr/bin/env bash
set -ex

echo "Setting up environment for pymeshlab ${PYMESHLAB_VERSION}"

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

cd /opt/pymeshlab
# Run the 1_build.sh script to build PyMeshLab
sh scripts/Linux/1_build.sh -j$(nproc)

# Check if make_wheel.sh exists, and use it if available
if [ -f scripts/Linux/make_wheel.sh ]; then
    echo "Using make_wheel.sh to create the Python wheel"
    sh scripts/Linux/make_wheel.sh
else
    echo "make_wheel.sh not found, using pip3 wheel as fallback"
    echo "Deploying for pymeshlab ${PYMESHLAB_VERSION}"
    sh scripts/Linux/2_deploy.sh
    pip3 wheel . -w /opt/pymeshlab/wheels
fi

pip3 install --no-cache-dir --verbose /opt/pymeshlab/wheels/pymeshlab*.whl

# run here to test if it works
python3 -c "import pymeshlab; print(pymeshlab.version())"

echo "Pymeshlab OK\n"

