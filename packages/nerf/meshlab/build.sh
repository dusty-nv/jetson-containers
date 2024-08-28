#!/usr/bin/env bash
set -ex

echo "Building meshlab ${MESHLAB_VERSION}"

git clone --branch=${MESHLAB_VERSION} --depth=1 --recursive https://github.com/cnr-isti-vclab/meshlab /opt/meshlab

cd /opt/meshlab

echo "Setting up environment for meshlab ${MESHLAB_VERSION}"

echo "Building meshlab ${MESHLAB_VERSION}"

# Run the 1_build.sh script to build meshlab
sh scripts/Linux/make_it.sh -j$(nproc)

cd /