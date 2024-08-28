#!/usr/bin/env bash
set -ex

echo "Building meshlab ${MESHLAB_VERSION}"

git clone --branch=${MESHLAB_VERSION} --depth=1 --recursive https://github.com/cnr-isti-vclab/meshlab /opt/meshlab

cd /opt/meshlab/resources/linux
# remove all content in the linux folder
rm -rf linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt
# copy /tmp/PYMESHLAB/extra/ al content in /opt/meshlab/resources/linux
cp -r /tmp/MESHLAB/extra/* /opt/meshlab/resources/linux
chmod +x linuxdeploy linuxdeploy-plugin-appimage linuxdeploy-plugin-qt

echo "Setting up environment for meshlab ${MESHLAB_VERSION}"

echo "Building meshlab ${MESHLAB_VERSION}"
cd /opt/meshlab/
# Run the 1_build.sh script to build meshlab
sh scripts/Linux/make_it.sh -j$(nproc)

cd /