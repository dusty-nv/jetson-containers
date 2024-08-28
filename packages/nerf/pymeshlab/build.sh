#!/usr/bin/env bash
set -ex

echo "Setting up environment for pymeshlab ${PYMESHLAB_VERSION}"

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

cd /opt/pymeshlab
# Run the 1_build.sh script to build PyMeshLab
sh scripts/Linux/1_build.sh -j$(nproc)


echo "Deploying for pymeshlab ${PYMESHLAB_VERSION}"
# sh scripts/Linux/2_deploy.sh

# Run the make_wheel.sh script to create a Python wheel
sh scripts/Linux/make_wheel.sh this use linuxdeploy

echo "Building Python wheel for pymeshlab ${PYMESHLAB_VERSION}"
# pip3 wheel . -w /opt/pymeshlab/wheels
cd /

pip3 install --no-cache-dir --verbose /opt/pymeshlab/wheels/pymeshlab*.whl

# run here to test if it works
python3 -c "import pymeshlab; print(pymeshlab.version())"

echo "Pymeshlab OK\n"

