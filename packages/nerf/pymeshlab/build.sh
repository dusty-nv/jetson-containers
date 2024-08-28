#!/usr/bin/env bash
set -ex

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

RUN git clone --depth=1 --recursive https://github.com/cnr-isti-vclab/PyMeshLab /opt/pymeshlab

cd /opt/pymeshlab/src/meshlab/resources/linux
# remove all content in the linux folder
rm -rf linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt
cp -r /tmp/PYMESHLAB/extra/* /opt/pymeshlab/src/meshlab/resources/linux
chmod +x linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt

echo "Setting up environment for pymeshlab ${PYMESHLAB_VERSION}"

echo "Building pymeshlab ${PYMESHLAB_VERSION}"

cd /opt/pymeshlab
# Run the 1_build.sh script to build PyMeshLab
sh scripts/Linux/1_build.sh -j$(nproc)

echo "Building Python wheel for pymeshlab ${PYMESHLAB_VERSION}"

# Run the make_wheel.sh script to create a Python wheel
sh scripts/Linux/make_wheel.sh

cd /

pip3 install --no-cache-dir --verbose /opt/pymeshlab/wheels/pymeshlab*.whl