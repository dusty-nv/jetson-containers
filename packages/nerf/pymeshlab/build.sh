#!/usr/bin/env bash
set -ex

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

git clone --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab /opt/pymeshlab

cd /opt/pymeshlab

echo "Setting up environment for pymeshlab ${PYMESHLAB_VERSION}"

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

# Run the 1_build.sh script to build PyMeshLab
sh scripts/Linux/1_build.sh -j$(nproc)

echo "Building Python wheel for pymeshlab ${PYMESHLAB_VERSION}"

# Run the make_wheel.sh script to create a Python wheel
sh scripts/Linux/make_wheel.sh

cd /

pip3 install --no-cache-dir --verbose /opt/pymeshlab/wheels/*.whl