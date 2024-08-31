#!/usr/bin/env bash
set -ex

echo "Building meshlab ${MESHLAB_VERSION}"

echo "Setting up environment for meshlab ${MESHLAB_VERSION}"

echo "Building meshlab ${MESHLAB_VERSION}"
cd /opt/meshlab/
# Run the 1_build.sh script to build meshlab
sh scripts/Linux/make_it.sh -j$(nproc)

cd /